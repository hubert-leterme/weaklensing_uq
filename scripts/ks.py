import os
import argparse
import time
import tqdm
import torch

import wlmmuq
import wlmmuq.utils as wlutils
import wlmmuq.data.torch as wlds
import wlmmuq.models.deepinv as wldinv

from wlmmuq.data import NUM_WORKERS

import _commons
import _add_arguments

METHOD_NAME = "ks"
OUTPUT_PREFIX = None

def main(
        path_to_real_shearmap: str | None = wlmmuq.PATH_TO_REAL_SHEARMAP,
        path_to_test_dataset: str | None = wlmmuq.PATH_TO_TEST_DATASET,
        path_to_calib_dataset: str | None = wlmmuq.PATH_TO_CALIB_DATASET,
        test_dataset_name: str | None = wlmmuq.TEST_DATASET_NAME,
        real_shearmap_name: str | None = wlmmuq.REAL_SHEARMAP_NAME,
        test_on_real_data: bool = False,
        output_dir: str = wlmmuq.RESULTS_DIR,
        method_name: str = METHOD_NAME,
        path_to_std_noise: str = wlmmuq.PATH_TO_STD_NOISE,
        path_to_mask: str = wlmmuq.PATH_TO_MASK,
        std_gaussianfilter: float | None = None,
        cosmos_include_faint: bool = False, inpainting: bool = _commons.INPAINTING_DEEPMASS,
        nimgs_test: int = _commons.NIMGS_TEST,
        cqr: bool = False,
        nimgs_calib: int = _commons.NIMGS_CALIB,
        min_idx_filename_ori_calib: str | int = _commons.MIN_IDX_FILENAME_ORI_CALIB,
        imgsize: int = _commons.IMGSIZE, batch_size: int = _commons.BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        mode_cqr: str | list[str] = _commons.MODE_CQR,
        scaling_factor_chisqcqr: float | None = None,
        confidence_uq: int | float = _commons.CONFIDENCE_UQ,
        get_initial_bounds: bool = False,
        save_tensors: bool = False, nimgs_save: int = _commons.NIMGS_SAVE,
        output_prefix: str | None = OUTPUT_PREFIX,
        seed: int | None = None, verbose: bool = False, **kwargs
):
    _commons.set_seed(seed)

    output_dir = _commons.get_path_to_results(
        output_dir, method_name, test_dataset_name=test_dataset_name,
        real_shearmap_name=real_shearmap_name,
        test_on_real_data=test_on_real_data
    )   # E.g., "results/dir/test_kappaTNG/ks/",
        # or "results/dir/test_cosmos/ks/"

    now = wlutils.get_timestamp()
    device = _commons.get_device(verbose=verbose)
    if verbose:
        print(f"Number of workers: {num_workers}")

    # Load noise standard deviation and mask
    std_noise, mask = _commons.get_stdnoise_mask(
        path_to_std_noise=path_to_std_noise,
        path_to_mask=path_to_mask,
        imgsize=imgsize, cosmos_include_faint=cosmos_include_faint,
        inpainting=inpainting, verbose=verbose
    )

    # Load test set
    test_dataloader = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask, shuffle=False,
        test_on_real_data=test_on_real_data,
        path_to_real_shearmap=path_to_real_shearmap
    )

    # Load calibration set, if provided
    if cqr:
        calib_dataloader = _commons.get_dataloader_massmapping(
            path_to_calib_dataset, nimgs_calib, imgsize, batch_size,
            num_workers, std_noise, mask,
            shuffle=True, min_idx_filename_ori=min_idx_filename_ori_calib
        )
    else:
        calib_dataloader = None

    # Instantiate physics (forward model) and RMSE metric
    physics = wldinv.iterativemm.MassMapping(sigma=std_noise, mask=mask).to(device)
    rmse_fn = wldinv.iterativemm.RMSE(mask=mask).to(device)

    beg_time = time.time()

    # Instantiate the KS model
    ks = wldinv.ks.KS(std_gaussianfilter=std_gaussianfilter).to(device)

    # Run KS for each batch
    if verbose:
        if not test_on_real_data:
            print(f"Compute Kaiser-Squires on the test set ({nimgs_test} images)")
        else:
            print(f"Compute Kaiser-Squires on the COSMOS shear map")
    out_ks = run_ks_batch(
        ks, physics, test_dataloader,
        rmse_fn=rmse_fn, get_initial_bounds=get_initial_bounds,
        test_on_real_data=test_on_real_data,
        device=device, verbose=verbose,
    )
    kappa_true = out_ks["kappa_true"]
    kappa_pred = out_ks["kappa_pred"]
    var = out_ks["var"]

    rmse = out_ks["rmse"]
    l2norm = out_ks["l2norm"]
    try:
        rmse = rmse.cpu()
        l2norm = l2norm.cpu()
    except Exception:
        pass

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    out_dict = {
        "inference_time": inference_time,
        "imgsize": imgsize,
        "confidence_uq": confidence_uq,
        "rmse": rmse,
        "l2norm": l2norm,
        "test_on_real_data": test_on_real_data,
    }
    if not test_on_real_data:
        out_dict.update({
            "nimgs_test": nimgs_test,
        })
    if save_tensors:
        out_dict.update({
            "kappa_pred": kappa_pred[:nimgs_save].cpu(),
            "var": var[:nimgs_save].cpu(),
        })
        if not test_on_real_data:
            out_dict.update({
                "kappa_true": kappa_true[:nimgs_save].cpu(),
            })

    # Calibrate with CQR, if available
    if calib_dataloader is not None:
        beg_time = time.time()

        if verbose:
            print(f"Compute DeepMass on the calibration set ({nimgs_calib} images)")
        out_ks_calib = run_ks_batch(
            ks, physics, calib_dataloader,
            rmse_fn=rmse_fn, device=device, verbose=verbose,
        )
        kappa_true_calib = out_ks_calib["kappa_true"]
        kappa_pred_calib = out_ks_calib["kappa_pred"]
        var_calib = out_ks_calib["var"]

        mode_cqr, scaling_factor_chisqcqr = _commons.convert_into_list_cqr_mode(
            mode_cqr, scaling_factor_chisqcqr
        )
        for mcqr, a in zip(mode_cqr, scaling_factor_chisqcqr):
            uq_dict = _commons.apply_calibration_and_get_metrics(
                kappa_pred, var, kappa_true,
                kappa_pred_calib, var_calib, kappa_true_calib,
                confidence_uq=confidence_uq,
                imgsize=imgsize, mode=mcqr, a=a,
                mask=mask, save_tensors=save_tensors, nimgs_save=nimgs_save,
                device=device, verbose=verbose
            )
            uq_key = _commons.get_uq_keys(mode_cqr=mcqr, scaling_factor_chisqcqr=a)
            out_dict.update({
                uq_key: uq_dict
            })

        calibration_time = _commons.get_inference_time(
            beg_time, which="calibration", verbose=verbose
        )
        out_dict.update({
            "calibration_time": calibration_time,
            "nimgs_calib": nimgs_calib,
        })

    _commons.save_results(
        out_dict, output_dir, now,
        prefix=output_prefix, verbose=verbose
    )


def run_ks_batch(
        ks: wldinv.ks.KS,
        physics: wldinv.iterativemm.MassMapping,
        dataloader: wlds.HDF5DatasetMassMapping,
        rmse_fn: wldinv.iterativemm.RMSE | None = None,
        get_initial_bounds: bool = False,
        test_on_real_data: bool = False,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_rmse = []
    listof_l2norm = []

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    for kappa_true, gamma_noisy in pbar:

        if not test_on_real_data:
            kappa_true = kappa_true.to(device)
        else: # No groung truth; kappa_true is set to torch.nan or None
            kappa_true = None

        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_pred = ks(gamma_noisy, physics)
            if rmse_fn is not None and not test_on_real_data:
                rmse = rmse_fn(kappa_pred, kappa_true)
                l2norm = rmse_fn(kappa_true, 0)
            else:
                rmse = None
                l2norm = None

        if not test_on_real_data:
            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_rmse.append(rmse) # Shape = (batch_size,)
        listof_l2norm.append(l2norm) # Shape = (batch_size,)

    if not test_on_real_data:
        kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)

    if get_initial_bounds:
        with torch.no_grad():
            var = ks.get_var(physics) # Shape = (imgsize, imgsize)
        var = var.unsqueeze(0).unsqueeze(0).repeat(
            kappa_pred.shape[0], 1, 1, 1
        ) # Shape = (nimgs, 1, imgsize, imgsize)
    else:
        var = torch.zeros(kappa_pred.shape, device=device) # Shape = (nimgs, 1, imgsize, imgsize)

    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs,)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs,)
    except Exception:
        rmse = None
        l2norm = None

    out = {
        "kappa_true": kappa_true,
        "kappa_pred": kappa_pred,
        "var": var,
        "rmse": rmse,
        "l2norm": l2norm,
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--std-gaussianfilter", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Standard deviation of the Gaussian filter used in KS. "
            "If not provided, no Gaussian filter is used. "
            "Default = None"
        )
    )
    _add_arguments.test_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _add_arguments.cqr(parser, prompt_init_bounds=True, zero_init_bounds=True)
    _add_arguments.output(parser, OUTPUT_PREFIX)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
