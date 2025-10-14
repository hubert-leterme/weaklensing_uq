import os
import argparse
import time
import tqdm
import torch

import wlmmuq
import wlmmuq.models.torch as wlnn
import wlmmuq.models.deepinv.iterativemm as wlpnp
import wlmmuq.models.deepinv.pnpmcalens as wlmcalens
import wlmmuq.models.deepinv.callbacks as wlcallbacks
import wlmmuq.utils as wlutils

from wlmmuq.data import NUM_WORKERS

import _commons
import _add_arguments

OUTPUT_DIR = os.path.join(wlmmuq.CHECKPOINT_DIR, "mcalens")
OUTPUT_FILENAME = "results_mcalens"

def main(
        path_to_test_dataset: str=wlmmuq.PATH_TO_TEST_DATASET,
        path_to_calib_dataset: str=wlmmuq.PATH_TO_CALIB_DATASET,
        path_to_std_noise: str=wlmmuq.PATH_TO_STD_NOISE,
        path_to_mask: str=wlmmuq.PATH_TO_MASK,
        path_to_ps: str=wlmmuq.PATH_TO_PS,
        step_size: float | list[float]=None,
        multfact_step_size: float | list[float]=None,
        niter: int=_commons.NITER_MCALENS,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_PNPMASS,
        nimgs_test: int=_commons.NIMGS_TEST,
        cqr: bool=False,
        nimgs_calib: int=_commons.NIMGS_CALIB,
        min_idx_filename_ori_calib: str=_commons.MIN_IDX_FILENAME_ORI_CALIB,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        starlet_detection_threshold: float=wlmcalens.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
        niter_per_step_g: int=wlmcalens.NITER_PER_STEP_G,
        niter_per_step_ng: int=wlmcalens.NITER_PER_STEP_NG,
        mode_cqr: str=_commons.MODE_CQR,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        get_initial_bounds: bool=False,
        n_noise_reals_per_img: int=_commons.N_NOISE_REALS_UQ,
        hyperparam_precalib: list[float] | None=None,
        find_optimal_hyperparam_precalib: bool=False,
        save_tensors: bool=False, nimgs_save: int=_commons.NIMGS_SAVE,
        output_dir: str=OUTPUT_DIR, output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    path_to_output = _commons.get_path_to_output(
        output_dir, output_filename
    ) # E.g., "checkpoint/dir/mcalens/results_mcalens"

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

    # Load starlet denoiser
    starlet, callback_starlet_denoiser = \
            _commons.instantiate_starlet_denoiser(
        imgsize=imgsize,
        starlet_detection_threshold=starlet_detection_threshold,
        device=device, verbose=verbose, **kwargs
    )

    # Get step size
    if not isinstance(multfact_step_size, list):
        multfact_step_size = [multfact_step_size]
    if not isinstance(step_size, list):
        step_size = len(multfact_step_size) * [step_size]

    # Instantiate physics (forward model) and RMSE metric
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)
    rmse_fn = wlpnp.RMSE(mask=mask).to(device)

    hyperparam_precalib = \
        _commons.convert_into_hyperparam_list(
            hyperparam_precalib,
            find_optimal_hyperparam_precalib=find_optimal_hyperparam_precalib
        )

    for tau, alph in zip(step_size, multfact_step_size):
        beg_time = time.time()

        # Instantiate the PnP model
        mcalens, _, _, tau, _ = _commons.get_pnpmass(
            starlet, denoiser_uq=None, imgsize=imgsize,
            std_noise=std_noise, rmse_fn=rmse_fn, physics=physics,
            step_size=tau, multfact_step_size=alph,
            eps_sup_step_size=eps_sup_step_size,
            niter=niter, mode="mcalens",
            which_gaussian_extractor="wiener",
            update_ng_first=True,
            path_to_ps=path_to_ps,
            starlet_detection_threshold=starlet_detection_threshold,
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            device=device, verbose=verbose
        )

        # Set callback list
        callback_list = []
        if callback_starlet_denoiser is not None:
            callback_list.append(callback_starlet_denoiser)
        callbacks = wlcallbacks.CallbackList(callback_list)

        # Run PnPMass for each batch
        test_dataloader = iter(test_dataset)
        if verbose:
            print(f"Compute PnPMass on the test set ({nimgs_test} images)")
        out_mcalens = run_mcalens_batch(
            mcalens, physics, test_dataloader, tau, niter,
            rmse_fn=rmse_fn,
            callbacks=callbacks,
            get_initial_bounds=get_initial_bounds,
            n_noise_reals_per_img=n_noise_reals_per_img,
            device=device, verbose=verbose,
        )
        kappa_true = out_mcalens["kappa_true"]
        kappa_pred = out_mcalens["kappa_pred"]
        var = out_mcalens["var"]
        rmse = out_mcalens["rmse"]
        l2norm = out_mcalens["l2norm"]

        inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

        out_dict = {
            "inference_time": inference_time,
            "step_size": tau,
            "niter": niter,
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
                print(f"Compute PnPMass on the calibration set ({nimgs_calib} images)")
            out_pnpmass_calib = run_mcalens_batch(
                mcalens, physics, calib_dataloader, tau, niter,
                rmse_fn=rmse_fn,
                callbacks=callbacks,
                get_initial_bounds=get_initial_bounds,
                device=device, verbose=verbose,
            )
            kappa_true_calib = out_pnpmass_calib["kappa_true"]
            kappa_pred_calib = out_pnpmass_calib["kappa_pred"]
            var_calib = out_pnpmass_calib["var"]

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
            out_dict, path_to_output, now, step_size=tau,
            verbose=verbose
        )


def run_mcalens_batch(
        mcalens: wlpnp.BaseOptim,
        physics: wlpnp.MassMapping,
        dataloader, step_size, niter,
        rmse_fn: wlpnp.RMSE | None=None,
        callbacks: wlcallbacks.BaseCallback | None=None,
        get_initial_bounds: bool=False,
        n_noise_reals_per_img: int=_commons.N_NOISE_REALS_UQ,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_var = []
    listof_rmse = []
    listof_l2norm = []

    if callbacks is None:
        callbacks = wlcallbacks.BaseCallback()

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    pbar.set_description(f"Step size = {step_size:.2e}, Nb iterations = {niter}")
    for i, (kappa_true, gamma_noisy) in enumerate(pbar):
        callbacks.on_batch_begin(i)
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_pred, metrics = mcalens(
                gamma_noisy, physics, x_gt=kappa_true, compute_metrics=True
            )
            if get_initial_bounds:
                var = _commons.variance_estimation_through_noise_propagation(
                    mcalens, physics,
                    output_shape=kappa_pred.shape,
                    n_noise_reals=n_noise_reals_per_img,
                    device=device, verbose=verbose
                )
            else:
                var = torch.zeros(kappa_pred.shape, device=device)

            if rmse_fn is not None:
                l2norm = rmse_fn(kappa_true, 0)
            else:
                l2norm = None

        listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_rmse.append(metrics["rmse"]) # Shape = (batch_size, niter)
        listof_l2norm.append(l2norm) # Shape = (batch_size, niter)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs, niter)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs, niter)
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

    _add_arguments.step_size_niter(parser, default_niter=_commons.NITER_MCALENS)
    _add_arguments.gaussian_extractor(parser, wiener=False, mcalens=True)
    _add_arguments.test_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _add_arguments.cqr(parser, prompt_init_bounds=True, montecarlo=True, zero_init_bounds=False)
    _add_arguments.output(parser, OUTPUT_FILENAME)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
