import argparse
import time

import wlmmuq.models.deepinv.iterativemm as wlpnp
from wlmmuq.models.deepinv.callbacks import CallbackList
import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS

import _commons

OUTPUT_DIR = "results_pnpmass"
OUTPUT_FILENAME = "results_pnpmass"

def main(
        path_to_test_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
        path_to_cqr: str=None,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        starlet: bool=False,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        load_model_uq: bool=False,
        arch_uq: str=None, timestamp_uq: str=None, epoch_uq: int=None,
        step_size: float | list[float]=None,
        multfact_step_size: float | list[float]=None,
        niter: int=_commons.NITER_PNPMASS,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_PNPMASS,
        nimgs_test: int=_commons.NIMGS_TEST,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        mode: str=_commons.MODE_PNPMASS,
        which_gaussian_extractor: str=_commons.WHICH_GAUSSIAN_EXTRACTOR,
        update_ng_first: bool=False,
        multfact_step_size_gaussian: float=None,
        niter_wiener: int=_commons.NITER_WIENER, noise_whitening_wiener: bool=False,
        starlet_detection_threshold: float=_commons.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
        niter_per_step_g: int=_commons.NITER_PER_STEP_G,
        niter_per_step_ng: int=_commons.NITER_PER_STEP_NG,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        save_tensors: bool=False, nimgs_save: int=_commons.NIMGS_SAVE,
        output_dir: str=OUTPUT_DIR, output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    path_to_output = _commons.get_path_to_output(
        output_dir, output_filename, checkpoint_dir=checkpoint_dir
    ) # E.g., "checkpoint/dir/results_pnpmass/results_pnpmass"

    now = wlutils.get_timestamp()
    device = _commons.get_device(verbose=verbose)
    if verbose:
        print(f"Number of workers: {num_workers}")

    beg_time = time.time()

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

    # Load denoisers (trained models or starlet denoiser for standard MCALens)
    if not starlet:
        denoiser, denoiser_uq = _commons.load_trained_models(
            checkpoint_dir, arch, timestamp, epoch=epoch,
            load_model_uq=load_model_uq, checkpoint_dir_uq=checkpoint_dir_uq,
            arch_uq=arch_uq, timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
            imgsize=imgsize, device=device, verbose=verbose, **kwargs
        )
        callback_starlet_denoiser = None
    else:
        denoiser, denoiser_uq, callback_starlet_denoiser = \
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

    # Load CQR, if available
    nimgs_calib, cqr = _commons.load_cqr(
        path_to_cqr, confidence_uq, imgsize, parent_dir=checkpoint_dir,
        device=device, verbose=verbose
    )

    # Instantiate physics (forward model)
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)

    for tau, alpha in zip(step_size, multfact_step_size):
        # Initialize iterator
        test_dataloader = iter(test_dataset)

        # Instantiate the PnP model
        pnpmass, tau, tau_filename, callback_gaussian_extractor = \
                _commons.get_pnpmass(
            denoiser, denoiser_uq, imgsize=imgsize,
            std_noise=std_noise, mask=mask, physics=physics,
            step_size=tau, multfact_step_size=alpha,
            eps_sup_step_size=eps_sup_step_size,
            niter=niter, mode=mode,
            which_gaussian_extractor=which_gaussian_extractor,
            update_ng_first=update_ng_first,
            path_to_ps=path_to_ps,
            noise_whitening_wiener=noise_whitening_wiener,
            multfact_step_size_gaussian=multfact_step_size_gaussian,
            niter_wiener=niter_wiener,
            starlet_detection_threshold=starlet_detection_threshold,
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            device=device, verbose=verbose
        )

        # Set callback list
        callback_list = []
        if callback_gaussian_extractor is not None:
            callback_list.append(callback_gaussian_extractor)
        if callback_starlet_denoiser is not None:
            callback_list.append(callback_starlet_denoiser)
        callbacks = CallbackList(callback_list)

        # Run PnPMass for each batch
        out_wiener_pnpmass = _commons.run_pnpmass_batch(
            pnpmass, physics, test_dataloader, tau, niter,
            confidence_uq=confidence_uq, callbacks=callbacks,
            device=device, verbose=verbose,
        )
        kappa_true = out_wiener_pnpmass["kappa_true"]
        kappa_pnpmass = out_wiener_pnpmass["kappa_pnpmass"]
        kappa_pnpmass_g = out_wiener_pnpmass["kappa_pnpmass_g"]
        kappa_pnpmass_ng = out_wiener_pnpmass["kappa_pnpmass_ng"]
        var_pnpmass = out_wiener_pnpmass["var_pnpmass"]
        res_pnpmass = out_wiener_pnpmass["res_pnpmass"]
        rmse_iter = out_wiener_pnpmass["rmse_iter"]

        inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

        # Calibrate with CQR, if available
        res_pnpmass_cqr, cqr_time = _commons.get_calibrated_residuals(
            cqr, res_pnpmass, verbose=verbose
        )

        # Compute miscoverage rate and size of prediction intervals
        mask = mask.to(device)
        err_pnpmass, predinterv_pnpmass, err_pnpmass_cqr, \
                predinterv_pnpmass_cqr, metrics_time = _commons.get_metrics(
            kappa_pnpmass, res_pnpmass, kappa_true, res_cqr=res_pnpmass_cqr,
            mask=mask, verbose=verbose
        )

        out_dict = {
            "inference_time": inference_time,
            "metrics_time": metrics_time,
            "step_size": tau,
            "arch": arch,
            "niter": niter,
            "nimgs_test": nimgs_test,
            "imgsize": imgsize,
            "confidence_uq": confidence_uq,
            "rmse_iter": rmse_iter.cpu(),
            "err_pnpmass": err_pnpmass.cpu(),
            "predinterv_pnpmass": predinterv_pnpmass.cpu(),
        }
        if save_tensors:
            out_dict.update({
                "kappa_true": kappa_true[:nimgs_save].cpu(),
                "kappa_pnpmass": kappa_pnpmass[:nimgs_save].cpu(),
                "var_pnpmass": var_pnpmass[:nimgs_save].cpu(),
                "res_pnpmass": res_pnpmass[:nimgs_save].cpu(),
            })
            if kappa_pnpmass_g is not None:
                out_dict.update({
                    "kappa_pnpmass_g": kappa_pnpmass_g[:nimgs_save].cpu(),
                    "kappa_pnpmass_ng": kappa_pnpmass_ng[:nimgs_save].cpu(),
                })
        if cqr is not None:
            out_dict.update({
                "cqr_time": cqr_time,
                "nimgs_calib": nimgs_calib,
                "err_pnpmass_cqr": err_pnpmass_cqr.cpu(),
                "predinterv_pnpmass_cqr": predinterv_pnpmass_cqr.cpu(),
            })
            if save_tensors:
                out_dict.update({
                    "res_pnpmass_cqr": res_pnpmass_cqr[:nimgs_save].cpu(),
                })
        _commons.save_results(
            out_dict, path_to_output, now, step_size=tau_filename,
            load_model_uq=load_model_uq, confidence_uq=confidence_uq,
            verbose=verbose
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_test_dataset", type=str,
        help="Path to the test set (HDF5 file)"
    )
    parser.add_argument(
        "checkpoint_dir", type=str,
        help="Checkpoint directory (containing the './pe' and './var' subdirectories)"
    )
    _commons.add_arguments_cqr(parser)
    _commons.add_arguments_model(parser)
    _commons.add_arguments_model_uq(parser)
    _commons.add_arguments_checkpoint(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Step size for the PnPMass algorithm. Several values can be provided. "
            "If not provided or set to 0, the step size will be computed as "
            f"Default = (1 - {_commons.EPS_SUP_STEP_SIZE:.1e}) * upper_bound, "
            "where upper_bound is estimated from the noise standard deviation "
            "and the mask, using the power iteration method."
        )
    )
    parser.add_argument(
        "-alph", "--multfact-step-size", type=float, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Multiplicative factor for the step size. Several values can be provided."
        )
    )
    parser.add_argument(
        "-i", "--niter", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of iterations for PnPMass. "
            f"Default = {_commons.NITER_PNPMASS}"
        )
    )
    _commons.add_arguments_test_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_pnpmode(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
