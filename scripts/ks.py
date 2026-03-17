import os
import argparse
import time
import tqdm
import typing
import torch

import wlmmuq

from wlmmuq.datasets import NUM_WORKERS

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
        test_on_real_data: bool = False, run_both: bool = False,
        output_dir: str = wlmmuq.RESULTS_DIR,
        method_name: str = METHOD_NAME,
        fwhm: float | None = _commons.FWHM_KS,
        path_to_std_noise: str = wlmmuq.PATH_TO_STD_NOISE,
        path_to_mask: str = wlmmuq.PATH_TO_MASK,
        bin_data_from_cosmos: bool = False,
        cosmos_include_faint: bool = False,
        max_z: float | None = _commons.MAX_Z, resolution: float = _commons.RESOLUTION,
        inpainting: bool = _commons.INPAINTING_KS,
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

    # When run_both is True we create two output dirs (simulated and real)
    if run_both:
        output_dir_sim = _commons.get_path_to_results(
            output_dir, method_name, test_dataset_name=test_dataset_name,
            real_shearmap_name=real_shearmap_name,
            test_on_real_data=False
        )
        output_dir_real = _commons.get_path_to_results(
            output_dir, method_name, test_dataset_name=test_dataset_name,
            real_shearmap_name=real_shearmap_name,
            test_on_real_data=True
        )
    else:
        output_dir = _commons.get_path_to_results(
            output_dir, method_name, test_dataset_name=test_dataset_name,
            real_shearmap_name=real_shearmap_name,
            test_on_real_data=test_on_real_data
        )

    now = wlmmuq.utils.get_timestamp()
    device = _commons.get_device(verbose=verbose)

    # Load noise standard deviation and mask
    # TODO: add argument `zbins`
    raise NotImplementedError
    std_noise, mask, gamma_real = _commons.get_stdnoise_mask_shearmap(
        path_to_std_noise=path_to_std_noise,
        path_to_mask=path_to_mask,
        path_to_real_shearmap=path_to_real_shearmap,
        bin_data_from_cosmos=bin_data_from_cosmos,
        get_noisy_shear_map=test_on_real_data or run_both,
        imgsize=imgsize, cosmos_include_faint=cosmos_include_faint,
        max_z=max_z, resolution=resolution,
        inpainting=inpainting, verbose=verbose
    ) # TODO: Add arguments `east_right` and `zbins`

    # Load test set(s)
    if run_both:
        test_dataloader_sim = _commons.get_dataloader_massmapping(
            path_to_test_dataset, nimgs_test, imgsize, batch_size,
            num_workers, std_noise, mask, shuffle=False,
            test_on_real_data=False
        )
        test_dataloader_real = _commons.get_dataloader_massmapping(
            path_to_test_dataset, nimgs_test, imgsize, batch_size,
            num_workers, std_noise, mask, shuffle=False,
            test_on_real_data=True, gamma_real=gamma_real
        )
    else:
        test_dataloader = _commons.get_dataloader_massmapping(
            path_to_test_dataset, nimgs_test, imgsize, batch_size,
            num_workers, std_noise, mask, shuffle=False,
            test_on_real_data=test_on_real_data, gamma_real=gamma_real
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
    physics = wlmmuq.physics.MassMapping(sigma=std_noise, mask=mask).to(device)
    rmse_fn = wlmmuq.metric.RMSE(mask=mask).to(device)

    # Instantiate the KS model
    std_gaussianfilter = wlmmuq.utils.get_std_gaussian(fwhm, resolution)
    ks = wldinv.ks.KS(std_gaussianfilter=std_gaussianfilter).to(device)

    # Prepare runs: either single or both (simulated and real)
    runs = []
    if run_both:
        runs.append(("sim", test_dataloader_sim, False, output_dir_sim))
        runs.append(("real", test_dataloader_real, True, output_dir_real))
    else:
        runs.append(("single", test_dataloader, test_on_real_data, output_dir))

    # Run inference for each requested dataset and collect results
    run_outputs: list[tuple[str, dict, str]] = []
    for run_name, tdataloader, td_real_flag, out_dir_run in runs:
        if verbose:
            if td_real_flag:
                print(f"Compute Kaiser-Squires on the COSMOS shear map")
            else:
                print(f"Compute Kaiser-Squires on the test set ({nimgs_test} images)")

        out_run = run_ks_batch(
            ks, physics, tdataloader,
            rmse_fn=rmse_fn, get_initial_bounds=get_initial_bounds,
            test_on_real_data=td_real_flag,
            device=device, verbose=verbose,
        )
        run_outputs.append((run_name, out_run, out_dir_run))

    # Run calibration once (if requested) and then apply to each run
    if (calib_dataloader is not None) and cqr:
        if verbose:
            print(f"Compute KS on the calibration set ({nimgs_calib} images)")
        out_ks_calib = run_ks_batch(
            ks, physics, calib_dataloader,
            rmse_fn=rmse_fn, device=device, verbose=verbose,
        )
        kappa_true_calib = out_ks_calib["kappa_true"]
        kappa_pred_calib = out_ks_calib["kappa_pred"]
        var_calib = out_ks_calib["var"]
    else:
        kappa_true_calib = None
        kappa_pred_calib = None
        var_calib = None

    # Compose per-run outputs, apply calibration if available, and save
    for run_name, out_ks_run, out_dir_run in run_outputs:
        kappa_true = out_ks_run["kappa_true"]
        kappa_pred = out_ks_run["kappa_pred"]
        var = out_ks_run["var"]

        rmse = out_ks_run["rmse"]
        l2norm = out_ks_run["l2norm"]
        try:
            rmse = rmse.cpu()
            l2norm = l2norm.cpu()
        except Exception:
            pass

        inference_time = _commons.get_inference_time(time.time(), verbose=verbose)

        trd = (run_name == "real") if run_both else test_on_real_data
        out_dict = {
            "inference_time": inference_time,
            "imgsize": imgsize,
            "confidence_uq": confidence_uq,
            "rmse": rmse,
            "l2norm": l2norm,
            "test_on_real_data": trd,
        }
        if not trd:
            out_dict.update({
                "nimgs_test": nimgs_test,
            })
        if save_tensors:
            out_dict.update({
                "kappa_pred": kappa_pred[:nimgs_save].cpu(),
                "var": var[:nimgs_save].cpu(),
            })
            if not trd:
                out_dict.update({
                    "kappa_true": kappa_true[:nimgs_save].cpu(),
                })

        # Apply calibration only when calibration outputs are present
        if (kappa_pred_calib is not None) and (var_calib is not None) and (kappa_true_calib is not None):
            mode_cqr_list, scaling_factor_list = _commons.convert_into_list_cqr_mode(
                mode_cqr, typing.cast(typing.Any, scaling_factor_chisqcqr)
            )
            for mcqr, a in zip(mode_cqr_list, scaling_factor_list):
                uq_dict = _commons.apply_calibration_and_get_metrics(
                    kappa_pred, var, kappa_true,
                    kappa_pred_calib, var_calib, kappa_true_calib,
                    confidence_uq=confidence_uq,
                    imgsize=imgsize, mode=mcqr, a=a,
                    mask=mask, save_tensors=save_tensors, nimgs_save=nimgs_save,
                    device=device, verbose=verbose
                )
                uq_key = _commons.get_uq_keys(mode_cqr=mcqr, scaling_factor_chisqcqr=a)
                out_dict.update({uq_key: uq_dict})

        _commons.save_results(out_dict, out_dir_run, now, prefix=output_prefix, verbose=verbose)


def run_ks_batch(
        ks: wlmmuq.models.KS,
        physics: wlmmuq.physics.MassMapping,
        dataloader,
        rmse_fn: wlmmuq.metric.RMSE | None = None,
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
        else: # No ground truth
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
            listof_kappa_true.append(kappa_true)
        listof_kappa_pred.append(kappa_pred)
        listof_rmse.append(rmse)
        listof_l2norm.append(l2norm)

    if not test_on_real_data:
        kappa_true = torch.cat(listof_kappa_true, dim=0)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0)

    if get_initial_bounds:
        with torch.no_grad():
            var = ks.get_var(physics) # Shape = (imgsize, imgsize)
        var = var.unsqueeze(0).unsqueeze(0).repeat(
            kappa_pred.shape[0], 1, 1, 1
        )
    else:
        var = torch.zeros(kappa_pred.shape, device=device)

    try:
        rmse = torch.cat(listof_rmse, dim=0)
        l2norm = torch.cat(listof_l2norm, dim=0)
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
        "--fwhm", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Full width at half maximum (FWHM) for Gaussian smoothing. "
            f"Default = {_commons.FWHM_KS:.1f}"
        )
    )
    _add_arguments.std_noise_mask(parser)
    _add_arguments.test_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _add_arguments.cqr(parser)
    _add_arguments.output(parser, OUTPUT_PREFIX)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
