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

OUTPUT_DIR = "results_pnpmass"
OUTPUT_FILENAME = "results_pnpmass"

def main(
        path_to_test_dataset: str = wlmmuq.PATH_TO_TEST_DATASET,
        path_to_calib_dataset: str = wlmmuq.PATH_TO_CALIB_DATASET,
        checkpoint_dir: str = wlmmuq.MODEL_DIR,
        checkpoint_subdir: str | None = None, checkpoint_subdir_uq: str | None = None,
        path_to_std_noise: str = wlmmuq.PATH_TO_STD_NOISE,
        path_to_mask: str = wlmmuq.PATH_TO_MASK,
        path_to_ps: str = wlmmuq.PATH_TO_PS,
        arch: str | None = None, timestamp: str | None = None, epoch: int = _commons.EPOCH,
        model_specs: str | None = None,
        load_model_uq: bool = False,
        arch_uq: str | None = None, timestamp_uq: str | None = None, epoch_uq: int | None = None,
        model_specs_uq: str | None = None,
        step_size: float | None | list[float | None] = None,
        multfact_step_size: float | None | list[float | None] = None,
        niter: int = _commons.NITER_PNPMASS,
        cosmos_include_faint: bool = False, inpainting: bool = _commons.INPAINTING_PNPMASS,
        nimgs_test: int = _commons.NIMGS_TEST,
        cqr: bool = False,
        nimgs_calib: int = _commons.NIMGS_CALIB,
        min_idx_filename_ori_calib: str | int = _commons.MIN_IDX_FILENAME_ORI_CALIB,
        imgsize: int = _commons.IMGSIZE, batch_size: int = _commons.BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        mode: str = _commons.MODE_PNPMASS,
        which_gaussian_extractor: str = _commons.WHICH_GAUSSIAN_EXTRACTOR,
        update_ng_first: bool = False,
        niter_wiener: int = _commons.NITER_WIENER,
        starlet_detection_threshold: float = _commons.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size: float = _commons.EPS_SUP_STEP_SIZE,
        niter_per_step_g: int = _commons.NITER_PER_STEP_G,
        niter_per_step_ng: int = _commons.NITER_PER_STEP_NG,
        starlet_debiasing: bool = False,
        step_size_starlet_debiasing: float | None = None,
        multfact_step_size_starlet_debiasing: float | None = None,
        niter_starlet_debiasing: int = _commons.NITER_STARLET_DEBIASING,
        mode_cqr: str | list[str] = _commons.MODE_CQR,
        scaling_factor_chisqcqr: float | None | list[float | None] = None,
        confidence_uq: int | float = _commons.CONFIDENCE_UQ,
        hyperparam_precalib: list[float] | None = None,
        find_optimal_hyperparam_precalib: bool = False,
        save_tensors: bool = False, nimgs_save: int = _commons.NIMGS_SAVE,
        output_dir: str = OUTPUT_DIR, output_filename: str = OUTPUT_FILENAME,
        seed: int | None = None, verbose: bool = False, **kwargs
):
    _commons.set_seed(seed)

    checkpoint_dir, checkpoint_dir_uq = _commons.get_checkpoint_dirs(
        checkpoint_dir,
        checkpoint_subdir=checkpoint_subdir,
        checkpoint_subdir_uq=checkpoint_subdir_uq
    )

    path_to_output = _commons.get_path_to_output(
        output_dir, output_filename, checkpoint_dir=checkpoint_dir
    ) # E.g., "checkpoint/dir/results_pnpmass/results_pnpmass"

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

    # Load trained denoisers
    denoiser, denoiser_uq = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        model_specs=model_specs,
        load_model_uq=load_model_uq, checkpoint_dir_uq=checkpoint_dir_uq,
        arch_uq=arch_uq, timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        model_specs_uq=model_specs_uq,
        imgsize=imgsize, device=device, verbose=verbose, **kwargs
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
        pnpmass, pnpmass_uq, tau = \
                    _commons.get_pnpmass(
            denoiser, denoiser_uq,
            std_noise=std_noise, rmse_fn=rmse_fn, physics=physics,
            step_size=tau, multfact_step_size=alph,
            eps_sup_step_size=eps_sup_step_size,
            niter=niter, mode=mode,
            update_ng_first=update_ng_first,
            path_to_ps=path_to_ps,
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            device=device, verbose=verbose
        )

        # Get Gaussian extractor
        # Note: the step sizes for the Gaussian extractor are computed automatically
        if mode == "residual":
            gaussian_extractor, callback_gaussian_extractor = \
                    _commons.get_gaussian_extractor(
                which=which_gaussian_extractor,
                path_to_ps=path_to_ps,
                white_noise=False,
                imgsize=imgsize, std_noise=std_noise, physics=physics,
                step_size=None, step_size_ng=None,
                eps_sup_step_size=eps_sup_step_size,
                niter=niter_wiener,
                starlet_detection_threshold=starlet_detection_threshold,
                mcalens_update_ng_first=update_ng_first,
                device=device, verbose=False
            )
        else:
            gaussian_extractor = None
            callback_gaussian_extractor = None

        # Instantiate starlet denoiser, in case of debiasing
        if starlet_debiasing:
            starlet, callback_starlet_denoiser = \
                    _commons.instantiate_starlet_denoiser(
                imgsize=imgsize,
                starlet_detection_threshold=starlet_detection_threshold,
                device=device, verbose=verbose
            )
            init_starlet_debiaser = wlpnp.ManualInit()
            starlet_debiaser, _, step_size_starlet_debiasing = \
                        _commons.get_pnpmass(
                starlet, denoiser_uq=None,
                std_noise=std_noise, rmse_fn=rmse_fn, physics=physics,
                step_size=step_size_starlet_debiasing,
                multfact_step_size=multfact_step_size_starlet_debiasing,
                eps_sup_step_size=eps_sup_step_size,
                niter=niter_starlet_debiasing,
                custom_init=init_starlet_debiaser,
                mode="regular",
                device=device, verbose=verbose
            )
        else:
            starlet = None
            callback_starlet_denoiser = None
            starlet_debiaser = None

        # Set callback list
        callback_list = []
        if callback_gaussian_extractor is not None:
            callback_list.append(callback_gaussian_extractor)
        if callback_starlet_denoiser is not None:
            callback_list.append(callback_starlet_denoiser)
        callbacks = wlcallbacks.CallbackList(callback_list)

        # Run PnPMass for each batch
        test_dataloader = iter(test_dataset)
        if verbose:
            print(f"Compute PnPMass on the test set ({nimgs_test} images)")
        out_pnpmass = run_pnpmass_batch(
            pnpmass, pnpmass_uq, physics, test_dataloader, tau, niter,
            rmse_fn=rmse_fn,
            gaussian_extractor=gaussian_extractor,
            starlet_debiaser=starlet_debiaser,
            starlet=starlet,
            callbacks=callbacks,
            device=device, verbose=verbose,
        )
        kappa_true = out_pnpmass["kappa_true"]
        kappa_pred = out_pnpmass["kappa_pred"]
        var = out_pnpmass["var"]
        rmse = out_pnpmass["rmse"]
        l2norm = out_pnpmass["l2norm"]

        inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

        out_dict = {
            "inference_time": inference_time,
            "step_size": tau,
            "arch": arch,
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
            out_pnpmass_calib = run_pnpmass_batch(
                pnpmass, pnpmass_uq, physics, calib_dataloader, tau, niter,
                rmse_fn=rmse_fn,
                gaussian_extractor=gaussian_extractor,
                starlet_debiaser=starlet_debiaser,
                starlet=starlet,
                callbacks=callbacks,
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
            out_dict, path_to_output, now, step_size=tau,
            verbose=verbose
        )


def run_pnpmass_batch(
        pnpmass: wlpnp.BaseOptim, pnpmass_uq: wlpnp.BaseOptim | None,
        physics: wlpnp.MassMapping,
        dataloader, step_size, niter,
        rmse_fn: wlpnp.RMSE | None = None,
        gaussian_extractor: wlpnp.BaseOptim | None = None,
        starlet_debiaser: wlpnp.BaseOptim | None = None,
        starlet: wlmcalens.Starlet2d | None = None,
        callbacks: wlcallbacks.BaseCallback | None = None,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_var = []
    listof_rmse = []
    listof_rmse_starlet_debiaser = []
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
            if gaussian_extractor is not None:
                kappa_g = gaussian_extractor(
                    gamma_noisy, physics, x_gt=None, compute_metrics=False
                )
                gamma_noisy = gamma_noisy - physics.A(kappa_g)
                kappa_true = kappa_true - kappa_g

            kappa_pred, metrics = pnpmass(
                gamma_noisy, physics, x_gt=kappa_true, compute_metrics=True
            )
            if starlet_debiaser is not None:
                starlet_debiaser.custom_init.X_init = (kappa_pred,)
                starlet.x_prev = kappa_pred
                kappa_pred, metrics_starlet_debiaser = starlet_debiaser(
                    gamma_noisy, physics, x_gt=kappa_true, compute_metrics=True
                )

            if pnpmass_uq is not None:
                pnpmass_uq.custom_init.X_init = (kappa_pred,)
                var = pnpmass_uq(
                    gamma_noisy, physics, compute_metrics=False
                )
            else:
                var = torch.zeros(kappa_pred.shape, device=device)

            if gaussian_extractor is not None:
                kappa_pred = kappa_pred + kappa_g
                kappa_true = kappa_true + kappa_g

            if rmse_fn is not None:
                l2norm = rmse_fn(kappa_true, 0)
            else:
                l2norm = None

        listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_rmse.append(metrics["rmse"]) # Shape = (batch_size, niter)
        if starlet_debiaser is not None:
            listof_rmse_starlet_debiaser.append(metrics_starlet_debiaser["rmse"]) # Shape = (batch_size, niter_debiaser)
        listof_l2norm.append(l2norm) # Shape = (batch_size, niter)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs, niter)
        if starlet_debiaser is not None:
            rmse_starlet_debiaser = torch.cat(listof_rmse_starlet_debiaser, dim=0) # Shape = (nimgs, niter_debiaser)
        else:
            rmse_starlet_debiaser = None
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs, niter)
    except TypeError:
        rmse = None
        rmse_starlet_debiaser = None
        l2norm = None

    out = {
        "kappa_true": kappa_true,
        "kappa_pred": kappa_pred,
        "var": var,
        "rmse": rmse,
        "rmse_starlet_debiaser": rmse_starlet_debiaser,
        "l2norm": l2norm,
    }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    _add_arguments.model(parser)
    _add_arguments.model_uq(parser)
    _add_arguments.checkpoint(parser)
    _add_arguments.step_size_niter(parser, default_niter=_commons.NITER_PNPMASS)
    parser.add_argument(
        "--mode", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Mode for PnPMass. Possible values are: "
            "'regular', 'residual', or 'pnpmcalens'. "
            "Default = 'regular'"
        )
    )
    parser.add_argument(
        "--which-gaussian-extractor", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Type of Gaussian extractor. Possible values are 'wiener' or 'mcalens'. "
            "Only used if `--mode` is set to 'residual'. "
            f"Default = '{_commons.WHICH_GAUSSIAN_EXTRACTOR}'"
        )
    )
    _add_arguments.gaussian_extractor(parser, wiener=True, mcalens=True, verbose=True)
    _add_arguments.starlet_debiasing(parser)
    _add_arguments.test_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _add_arguments.cqr(parser)
    _add_arguments.output(parser, OUTPUT_FILENAME)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
