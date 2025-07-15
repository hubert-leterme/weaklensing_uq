import argparse
import time

import wlmmuq.models.cqr as wlcqr
import wlmmuq.utils as wlutils

from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

def main(
        path_to_calib_dataset: str, model_dir: str,
        output_filename: str,
        arch: str=None, load_model_uq: bool=False,
        step_size: float=None, niter: int=_commons.NITER_PNPMASS,
        cosmos_include_faint: bool=False,
        nimgs_calib: int=_commons.NIMGS_CALIB, min_idx_filename_ori: str=None,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        wiener_init: bool=False, path_to_ps: str=None,
        niter_wiener: int=NITER_WIENER,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    now = wlutils.get_timestamp()
    device = _commons.get_device(verbose=verbose)
    if verbose:
        print(f"Number of workers: {num_workers}")

    beg_time = time.time()

    # Load noise standard deviation and mask
    std_noise, mask = _commons.get_stdnoise_mask(
        imgsize, cosmos_include_faint=cosmos_include_faint,
        convert_to_torch_tensor=True, inpainting=False,
        verbose=verbose
    )

    # Load calibration set
    calib_dataset = _commons.get_dataloader_massmapping(
        path_to_calib_dataset, nimgs_calib, imgsize, batch_size,
        num_workers, std_noise, mask,
        shuffle=True, min_idx_filename_ori=min_idx_filename_ori
    )

    # Load trained denoiser
    denoiser, denoiser_uq = _commons.load_trained_model(
        arch, imgsize, checkpoint_dir=model_dir,
        load_model_uq=load_model_uq, verbose=verbose, **kwargs
    )

    # Get step size
    step_size = _commons.get_pnpmass_step_size(
        std_noise, mask, step_size=step_size
    )

    # Get iterative Wiener filtering (may be used for initialization)
    wiener = _commons.get_wiener(
        path_to_ps, std_noise, mask, niter=niter_wiener, verbose=verbose
    )

    # Initialize iterator
    calib_dataloader = iter(calib_dataset)

    # Instantiate the PnP model
    if wiener_init:
        if wiener is None:
            raise ValueError("The path to the power spectrum must be provided.")
        init_estimate = wiener
    else:
        init_estimate = None
    pnpmass, physics = _commons.get_pnpmass(
        std_noise, mask, denoiser, denoiser_uq, niter, step_size=step_size,
        init_estimate=init_estimate
    )
    pnpmass = pnpmass.to(device)
    if wiener is not None:
        wiener = wiener.to(device)
    physics = physics.to(device)

    # Run PnPMass for each batch
    _, kappa_true, _, kappa_pnpmass, _, \
            res_pnpmass, _, _ = _commons.run_wiener_pnpmass_batch(
        wiener, pnpmass, physics, calib_dataloader, step_size, niter,
        confidence_uq=confidence_uq, mask=mask,
        device=device, verbose=verbose,
    )

    # Instantiate CQR model and compute the calibration parameters
    if verbose:
        print("Instantiate CQR model and compute the calibration parameters")
    alpha = wlutils.get_alpha_from_confidence(confidence_uq)
    cqr = wlcqr.AddCQR(alpha, map_size=imgsize).to(device)
    cqr.calibrate(kappa_pnpmass, res_pnpmass, kappa_true)

    inference_time = time.time() - beg_time

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
        out_dict, model_dir, "pnpmass",
        f"{output_filename}_{confidence_uq}-sigma_{now}.pt",
        verbose=verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_calib_dataset", type=str,
        help="Path to the calibration set (HDF5 file)"
    )
    _commons.add_arguments_model_dir(parser)
    _commons.add_arguments_model(parser)
    _commons.add_arguments_checkpoint(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Step size for the PnPMass algorithm. Several values can be provided. "
            f"Default = {_commons.MULTFACT_STEP_SIZE:.2f} * upper_bound, "
            "where upper_bound is computed from the noise standard deviation "
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
    parser.add_argument(
        "--nimgs-calib", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of calibration images. "
            f"Default = {_commons.NIMGS_CALIB}"
        )
    )
    _commons.add_arguments_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_wienerinit(parser)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
