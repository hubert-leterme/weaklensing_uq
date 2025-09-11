import argparse
import time

import wlmmuq.models.deepinv.iterativemm as wlpnp
from wlmmuq.models.deepinv.callbacks import CallbackList
import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS

import _commons

OUTPUT_DIR = "cqr_pnpmass"
OUTPUT_FILENAME = "cqr_pnpmass"

def main(
        path_to_calib_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        starlet: bool=False,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        load_model_uq: bool=False,
        arch_uq: str=None, timestamp_uq: str=None, epoch_uq: int=None,
        step_size: float=None, niter: int=_commons.NITER_PNPMASS,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_PNPMASS,
        nimgs_calib: int=_commons.NIMGS_CALIB, min_idx_filename_ori: str=None,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        mode: str=_commons.MODE_PNPMASS,
        which_gaussian_extractor: str=_commons.WHICH_GAUSSIAN_EXTRACTOR,
        multfact_step_size_gaussian: float=None,
        niter_wiener: int=_commons.NITER_WIENER, noise_whitening_wiener: bool=False,
        starlet_detection_threshold: float=_commons.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
        niter_per_step_g: int=_commons.NITER_PER_STEP_G,
        niter_per_step_ng: int=_commons.NITER_PER_STEP_NG,
        mode_cqr: str=_commons.MODE_CQR,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        multfact_confidence_uq: float=None,
        addconst_confidence_uq: float=None,
        output_dir: str=OUTPUT_DIR, output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    path_to_output = _commons.get_path_to_output(
        output_dir, output_filename, checkpoint_dir=checkpoint_dir
    ) # E.g., "checkpoint/dir/cqr_pnpmass/cqr_pnpmass"

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

    # Load calibration set
    calib_dataset = _commons.get_dataloader_massmapping(
        path_to_calib_dataset, nimgs_calib, imgsize, batch_size,
        num_workers, std_noise, mask,
        shuffle=True, min_idx_filename_ori=min_idx_filename_ori
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

    # Instantiate physics (forward model) and RMSE metric
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)
    rmse_fn = wlpnp.RMSE(mask=mask).to(device)

    # Instantiate the PnP model
    pnpmass, pnpmass_uq, gaussian_extractor, \
            step_size, callback_gaussian_extractor = \
                _commons.get_pnpmass(
        denoiser, denoiser_uq, imgsize=imgsize,
        std_noise=std_noise, rmse_fn=rmse_fn, physics=physics,
        step_size=step_size, eps_sup_step_size=eps_sup_step_size,
        niter=niter, mode=mode,
        which_gaussian_extractor=which_gaussian_extractor,
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
    calib_dataloader = iter(calib_dataset)
    out_pnpmass = _commons.run_pnpmass_batch(
        pnpmass, pnpmass_uq, physics, calib_dataloader, step_size, niter,
        rmse_fn=rmse_fn, gaussian_extractor=gaussian_extractor,
        callbacks=callbacks,
        device=device, verbose=verbose,
    )
    kappa_true = out_pnpmass["kappa_true"]
    kappa_pred = out_pnpmass["kappa_pred"]
    var = out_pnpmass["var"]

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    # Instantiate CQR model and compute the calibration parameters
    multfact_confidence_uq, addconst_confidence_uq = \
        _commons.convert_into_param_lists(
            multfact_confidence_uq, addconst_confidence_uq
        )

    for rho, const in zip(multfact_confidence_uq, addconst_confidence_uq):
        beg_time = time.time()
        cqr = _commons.get_cqr(
            kappa_pred, var, kappa_true,
            confidence_uq=confidence_uq,
            imgsize=imgsize, mode=mode_cqr,
            multfact_confidence_uq=rho,
            addconst_confidence_uq=const,
            device=device, verbose=verbose
        )
        calibration_time = _commons.get_inference_time(
            beg_time, which="calibration", verbose=False
        )
        out_dict = {
            "state_dict": cqr.state_dict(),
            "inference_time": inference_time,
            "calibration_time": calibration_time,
            "step_size": step_size,
            "arch": arch,
            "niter": niter,
            "nimgs_calib": nimgs_calib,
            "imgsize": imgsize,
            "confidence_uq": confidence_uq,
            "multfact_confidence_uq": rho,
        }
        _commons.save_results(
            out_dict, path_to_output, now, step_size=step_size,
            multfact_confidence_uq=rho,
            addconst_confidence_uq=const,
            verbose=verbose
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_calib_dataset", type=str,
        help="Path to the calibration set (HDF5 file)"
    )
    parser.add_argument(
        "checkpoint_dir", type=str,
        help="Checkpoint directory (containing the './pe' and './var' subdirectories)"
    )
    _commons.add_arguments_uq(parser)
    _commons.add_arguments_model(parser)
    _commons.add_arguments_model_uq(parser)
    _commons.add_arguments_checkpoint(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Step size for the PnPMass algorithm. "
            f"Default = (1 - {_commons.EPS_SUP_STEP_SIZE:.1e}) * upper_bound, "
            "where upper_bound is estimated from the noise standard deviation "
            "and the mask, using the power iteration method"
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
    _commons.add_arguments_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_pnpmode(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
