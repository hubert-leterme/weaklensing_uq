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
        mode: str=_commons.MODE_PNPMASS, switch_mode_for_uq: bool=False,
        which_gaussian_extractor: str=_commons.WHICH_GAUSSIAN_EXTRACTOR,
        niter_wiener: int=_commons.NITER_WIENER, noise_whitening_wiener: bool=False,
        starlet_detection_threshold: float=_commons.STARLET_DETECTION_THRESHOLD,
        multfact_sup_step_size: float=_commons.MULTFACT_SUP_STEP_SIZE,
        niter_per_step_g: int=_commons.NITER_PER_STEP_G,
        niter_per_step_ng: int=_commons.NITER_PER_STEP_NG,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
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

    # Instantiate physics (forward model)
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)

    # Instantiate the PnP model
    pnpmass, pnpmass_uq, step_size, step_size_filename, callback_gaussian_extractor = \
            _commons.get_pnpmass(
        denoiser, denoiser_uq, imgsize=imgsize,
        std_noise=std_noise, mask=mask, physics=physics,
        step_size=step_size, niter=niter,
        multfact_sup_step_size=multfact_sup_step_size, mode=mode,
        which_gaussian_extractor=which_gaussian_extractor,
        switch_mode_for_uq=switch_mode_for_uq,
        path_to_ps=path_to_ps,
        noise_whitening_wiener=noise_whitening_wiener,
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
        pnpmass, pnpmass_uq,
        physics, calib_dataloader, step_size, niter,
        confidence_uq=confidence_uq, callbacks=callbacks,
        device=device, verbose=verbose,
    )
    kappa_true = out_pnpmass["kappa_true"]
    kappa_pnpmass = out_pnpmass["kappa_pnpmass"]
    res_pnpmass = out_pnpmass["res_pnpmass"]

    # Instantiate CQR model and compute the calibration parameters
    cqr = _commons.get_cqr(
        kappa_pnpmass, res_pnpmass, kappa_true, imgsize, confidence_uq,
        device=device, verbose=verbose
    )

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    out_dict = {
        "state_dict": cqr.state_dict(),
        "inference_time": inference_time,
        "step_size": step_size,
        "arch": arch,
        "niter": niter,
        "nimgs_calib": nimgs_calib,
        "imgsize": imgsize,
        "confidence_uq": confidence_uq,
    }
    _commons.save_results(
        out_dict, path_to_output, now, step_size=step_size_filename,
        load_model_uq=load_model_uq, confidence_uq=confidence_uq,
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
    _commons.add_arguments_model(parser)
    _commons.add_arguments_model_uq(parser)
    _commons.add_arguments_checkpoint(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Step size for the PnPMass algorithm. "
            f"Default = {_commons.MULTFACT_SUP_STEP_SIZE:.2f} * upper_bound, "
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
