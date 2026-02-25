import os
import argparse
import time
import tqdm
import torch

import wlmmuq
import wlmmuq.models.deepinv.iterativemm as wlpnp
import wlmmuq.models.deepinv.pnpmcalens as wlmcalens
import wlmmuq.models.deepinv.callbacks as wlcallbacks
import wlmmuq.utils as wlutils

from wlmmuq.data import NUM_WORKERS

import _commons
import _add_arguments

METHOD_NAME = "mcalens"
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
        path_to_ps: str = wlmmuq.PATH_TO_PS,
        step_size: float | list[float] | None = None,
        multfact_step_size: float | list[float] | None = None,
        niter: int = _commons.NITER_MCALENS,
        cosmos_include_faint: bool = False, inpainting: bool = _commons.INPAINTING_PNPMASS,
        nimgs_test: int = _commons.NIMGS_TEST,
        cqr: bool = False,
        nimgs_calib: int = _commons.NIMGS_CALIB,
        min_idx_filename_ori_calib: str | int = _commons.MIN_IDX_FILENAME_ORI_CALIB,
        imgsize: int = _commons.IMGSIZE, batch_size: int = _commons.BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        starlet_detection_threshold: float = wlmcalens.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size: float = _commons.EPS_SUP_STEP_SIZE,
        niter_per_step_g: int = wlmcalens.NITER_PER_STEP_G,
        niter_per_step_ng: int = wlmcalens.NITER_PER_STEP_NG,
        mode_cqr: str | list[str] = _commons.MODE_CQR,
        scaling_factor_chisqcqr: float | None = None,
        confidence_uq: int | float = _commons.CONFIDENCE_UQ,
        get_initial_bounds: bool = False,
        n_noise_reals_per_img: int = _commons.N_NOISE_REALS_UQ,
        hyperparam_precalib: list[float] | None = None,
        find_optimal_hyperparam_precalib: bool = False,
        save_tensors: bool = False, nimgs_save: int = _commons.NIMGS_SAVE,
        output_prefix: str | None = OUTPUT_PREFIX,
        seed: int | None = None, verbose: bool = False, **kwargs
):
    _commons.set_seed(seed)

    output_dir = _commons.get_path_to_results(
        output_dir, method_name, test_dataset_name=test_dataset_name,
        real_shearmap_name=real_shearmap_name,
        test_on_real_data=test_on_real_data
    )   # E.g., "results/dir/test_kappaTNG/mcalens/",
        # or "results/dir/test_cosmos/mcalens/"

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

    # Load starlet denoiser
    starlet, callback_starlet_denoiser = \
            _commons.instantiate_starlet_denoiser(
        imgsize=imgsize,
        detection_threshold=starlet_detection_threshold,
        device=device, verbose=verbose, **kwargs
    )

    # Instantiate physics (forward model) and RMSE metric
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)
    rmse_fn = wlpnp.RMSE(mask=mask).to(device)

    hyperparam_precalib = \
        _commons.convert_into_hyperparam_list(
            hyperparam_precalib,
            find_optimal_hyperparam_precalib=find_optimal_hyperparam_precalib
        )

    # Get step size
    multfact_step_size, step_size = _commons.convert_into_lists(
        multfact_step_size, step_size
    )

    for tau, alph in zip(step_size, multfact_step_size):
        beg_time = time.time()

        # Instantiate the PnP model
        mcalens, _, tau = _commons.get_pnpmass(
            starlet, denoiser_uq=None,
            std_noise=std_noise, rmse_fn=rmse_fn, physics=physics,
            step_size=tau, multfact_step_size=alph,
            eps_sup_step_size=eps_sup_step_size,
            niter=niter, mode="pnpmcalens",
            path_to_ps=path_to_ps,
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            device=device, verbose=verbose
        )

        # Set callback list
        callback_list = []
        if callback_starlet_denoiser is not None:
            callback_list.append(callback_starlet_denoiser)
        callbacks = wlcallbacks.CallbackList(callback_list)

        # Run PnPMass for each batch
        if verbose:
            if not test_on_real_data:
                print(f"Compute MCALens on the test set ({nimgs_test} images)")
            else:
                print(f"Compute MCALens on the COSMOS shear map")
        out_mcalens = run_mcalens_batch(
            mcalens, physics, test_dataloader, tau, niter,
            rmse_fn=rmse_fn,
            callbacks=callbacks,
            get_initial_bounds=get_initial_bounds,
            n_noise_reals_per_img=n_noise_reals_per_img,
            test_on_real_data=test_on_real_data,
            device=device, verbose=verbose,
        )
        kappa_true = out_mcalens["kappa_true"]
        kappa_pred = out_mcalens["kappa_pred"]
        var = out_mcalens["var"]

        rmse = out_mcalens["rmse"]
        l2norm = out_mcalens["l2norm"]
        try:
            rmse = rmse.cpu()
            l2norm = l2norm.cpu()
        except Exception:
            pass

        inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

        out_dict = {
            "inference_time": inference_time,
            "step_size": tau,
            "niter": niter,
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

            mode_cqr, scaling_factor_chisqcqr = _commons.convert_into_list_cqr_mode(
                mode_cqr, scaling_factor_chisqcqr
            )
            for mcqr, a in zip(mode_cqr, scaling_factor_chisqcqr):
                for rho in hyperparam_precalib:
                    uq_dict = _commons.apply_calibration_and_get_metrics(
                        kappa_pred, var, kappa_true,
                        kappa_pred_calib, var_calib, kappa_true_calib,
                        confidence_uq=confidence_uq,
                        imgsize=imgsize, mode=mcqr, a=a,
                        hyperparam_precalib=rho,
                        find_optimal_hyperparam_precalib=find_optimal_hyperparam_precalib,
                        mask=mask, save_tensors=save_tensors, nimgs_save=nimgs_save,
                        device=device, verbose=verbose
                    )
                    uq_key = _commons.get_uq_keys(
                        mode_cqr=mcqr, scaling_factor_chisqcqr=a, rho=rho
                    )
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
            out_dict, output_dir, now, step_size=tau,
            prefix=output_prefix, verbose=verbose
        )


def run_mcalens_batch(
        mcalens: wlpnp.BaseOptim,
        physics: wlpnp.MassMapping,
        dataloader, step_size, niter,
        rmse_fn: wlpnp.RMSE | None = None,
        callbacks: wlcallbacks.BaseCallback | None = None,
        get_initial_bounds: bool = False,
        n_noise_reals_per_img: int = _commons.N_NOISE_REALS_UQ,
        test_on_real_data: bool = False,
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

        if not test_on_real_data:
            kappa_true = kappa_true.to(device)
            compute_metrics = True
        else: # No groung truth; kappa_true is set to torch.nan or None
            kappa_true = None
            compute_metrics = False

        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            out_mcalens = mcalens(
                gamma_noisy, physics, x_gt=kappa_true,
                compute_metrics=compute_metrics
            )
            if not test_on_real_data:
                kappa_pred, metrics = out_mcalens
                rmse = metrics["rmse"]
            else:
                kappa_pred = out_mcalens
                rmse = None
            if get_initial_bounds:
                var = _commons.variance_estimation_through_noise_propagation(
                    mcalens, physics,
                    output_shape=kappa_pred.shape,
                    n_noise_reals=n_noise_reals_per_img,
                    device=device, verbose=verbose
                )
            else:
                var = torch.zeros(kappa_pred.shape, device=device)

            if rmse_fn is not None and not test_on_real_data:
                l2norm = rmse_fn(kappa_true, 0)
            else:
                l2norm = None

        listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_rmse.append(rmse) # Shape = (batch_size, niter)
        listof_l2norm.append(l2norm) # Shape = (batch_size, niter)

    if not test_on_real_data:
        kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs, niter)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs, niter)
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

    _add_arguments.step_size_niter(parser, default_niter=_commons.NITER_MCALENS)
    _add_arguments.gaussian_extractor(parser, wiener=False, mcalens=True)
    _add_arguments.test_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _add_arguments.cqr(parser, prompt_init_bounds=True, montecarlo=True, zero_init_bounds=False)
    _add_arguments.output(parser, OUTPUT_PREFIX)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
