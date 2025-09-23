import argparse
import time
import tqdm
import torch

import wlmmuq.utils as wlutils
import wlmmuq.models.deepinv.iterativemm as wlpnp

from wlmmuq.data import NUM_WORKERS

import _commons
import _add_arguments

OUTPUT_DIR = ""
OUTPUT_FILENAME = "results_deepmass"

def main(
        path_to_test_dataset: str=_commons.PATH_TO_TEST_DATASET,
        path_to_calib_dataset: str=_commons.PATH_TO_CALIB_DATASET,
        checkpoint_dir: str=_commons.CHECKPOINT_DIR,
        checkpoint_subdir: str=None, checkpoint_subdir_uq: str=None,
        path_to_std_noise: str=_commons.PATH_TO_STD_NOISE,
        path_to_mask: str=_commons.PATH_TO_MASK,
        path_to_ps: str=_commons.PATH_TO_PS,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        model_specs: str | None=None,
        load_model_uq: bool=False,
        arch_uq: str=None, timestamp_uq: str=None, epoch_uq: int=None,
        model_specs_uq: str | None=None,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_DEEPMASS,
        nimgs_test: int=_commons.NIMGS_TEST,
        cqr: bool=False,
        nimgs_calib: int=_commons.NIMGS_CALIB,
        min_idx_filename_ori_calib: str=_commons.MIN_IDX_FILENAME_ORI_CALIB,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        niter_wiener: int=_commons.NITER_WIENER,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
        mode_cqr: str=_commons.MODE_CQR,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        hyperparam_precalib: list[float] | None=None,
        find_optimal_hyperparam_precalib: bool=False,
        save_tensors: bool=False, nimgs_save: int=_commons.NIMGS_SAVE,
        output_dir: str=OUTPUT_DIR, output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    checkpoint_dir, checkpoint_dir_uq = _commons.get_checkpoint_dirs(
        checkpoint_dir,
        checkpoint_subdir=checkpoint_subdir,
        checkpoint_subdir_uq=checkpoint_subdir_uq
    )

    path_to_output = _commons.get_path_to_output(
        output_dir, output_filename, checkpoint_dir=checkpoint_dir
    ) # E.g., "checkpoint/dir/results_deepmass"

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
    test_dataset = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask, shuffle=False
    )

    # Load calibration set, if provided
    if cqr:
        calib_dataset = _commons.get_dataloader_massmapping(
            path_to_calib_dataset, nimgs_calib, imgsize, batch_size,
            num_workers, std_noise, mask,
            shuffle=True, min_idx_filename_ori=min_idx_filename_ori_calib
        )
    else:
        calib_dataset = None

    # Load trained models
    deepmass, deepmass_uq = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        model_specs=model_specs,
        load_model_uq=load_model_uq, checkpoint_dir_uq=checkpoint_dir_uq,
        arch_uq=arch_uq, timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        model_specs_uq=model_specs_uq,
        imgsize=imgsize,
        std_noise=std_noise, mask=mask, path_to_ps=path_to_ps,
        eps_sup_step_size_wiener=eps_sup_step_size,
        niter_wiener=niter_wiener,
        device=device, verbose=verbose, **kwargs
    )

    # Instantiate RMSE metric
    rmse_fn = wlpnp.RMSE(mask=mask).to(device)

    hyperparam_precalib = \
        _commons.convert_into_hyperparam_list(
            hyperparam_precalib,
            find_optimal_hyperparam_precalib=find_optimal_hyperparam_precalib
        )

    beg_time = time.time()

    # Run DeepMass for each batch
    test_dataloader = iter(test_dataset)
    if verbose:
        print(f"Compute DeepMass on the test set ({nimgs_test} images)")
    out_deepmass = run_deepmass_batch(
        deepmass, deepmass_uq, test_dataloader,
        rmse_fn=rmse_fn, device=device, verbose=verbose,
    )
    kappa_true = out_deepmass["kappa_true"]
    kappa_pred = out_deepmass["kappa_pred"]
    var = out_deepmass["var"]
    rmse = out_deepmass["rmse"]
    l2norm = out_deepmass["l2norm"]

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    out_dict = {
        "inference_time": inference_time,
        "arch": arch,
        "nimgs_test": nimgs_test,
        "imgsize": imgsize,
        "confidence_uq": confidence_uq,
        "rmse": rmse.cpu(),
        "l2norm": l2norm.cpu(),
    }
    if save_tensors:
        out_dict.update({
            "kappa_true": kappa_true[:nimgs_save].cpu(),
            "kappa_pred": kappa_pred[:nimgs_save].cpu(),
            "var": var[:nimgs_save].cpu(),
        })

    # Calibrate with CQR, if available
    if calib_dataset is not None:
        beg_time = time.time()

        calib_dataloader = iter(calib_dataset)
        if verbose:
            print(f"Compute DeepMass on the calibration set ({nimgs_calib} images)")
        out_deepmass_calib = run_deepmass_batch(
            deepmass, deepmass_uq, calib_dataloader,
            rmse_fn=rmse_fn, device=device, verbose=verbose,
        )
        kappa_true_calib = out_deepmass_calib["kappa_true"]
        kappa_pred_calib = out_deepmass_calib["kappa_pred"]
        var_calib = out_deepmass_calib["var"]

        for rho in hyperparam_precalib:
            uq_dict = _commons.apply_calibration_and_get_metrics(
                kappa_pred, var, kappa_true,
                kappa_pred_calib, var_calib, kappa_true_calib,
                confidence_uq=confidence_uq,
                imgsize=imgsize, mode=mode_cqr,
                hyperparam_precalib=rho,
                find_optimal_hyperparam_precalib=find_optimal_hyperparam_precalib,
                mask=mask, save_tensors=save_tensors, nimgs_save=nimgs_save,
                device=device, verbose=verbose
            )
            uq_key = _commons.get_uq_keys(rho=rho)
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
        out_dict, path_to_output, now, verbose=verbose
    )


def run_deepmass_batch(
        deepmass: wlpnp.BaseOptim, deepmass_uq: wlpnp.BaseOptim,
        dataloader,
        rmse_fn: wlpnp.RMSE | None=None,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_var = []
    listof_rmse = []
    listof_l2norm = []

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    for kappa_true, gamma_noisy in pbar:
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_pred = deepmass(gamma_noisy)
            if deepmass_uq is not None:
                var = deepmass_uq(gamma_noisy)
            else:
                var = torch.zeros(kappa_true.shape, device=device)
            if rmse_fn is not None:
                rmse = rmse_fn(kappa_pred, kappa_true)
                l2norm = rmse_fn(kappa_true, 0)
            else:
                rmse = None
                l2norm = None

            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_rmse.append(rmse) # Shape = (batch_size,)
            listof_l2norm.append(l2norm) # Shape = (batch_size,)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs,)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs,)
    except TypeError:
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
    _add_arguments.model(parser)
    _add_arguments.model_uq(parser)
    _add_arguments.checkpoint(parser)
    _add_arguments.test_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _add_arguments.cqr(parser)
    _add_arguments.wiener(parser)
    _add_arguments.output(parser, OUTPUT_FILENAME)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
