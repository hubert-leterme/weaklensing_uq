import argparse
import time

import wlmmuq.models.deepinv.iterativemm as wlpnp
import wlmmuq.models.cqr as wlcqr
import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_DIR = "cqr_pnpmass"
OUTPUT_FILENAME = "cqr_pnpmass"

def main(
        path_to_calib_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        load_model_uq: bool=False,
        arch_uq: str=None, timestamp_uq: str=None, epoch_uq: int=_commons.EPOCH,
        step_size: float=None, niter: int=_commons.NITER_PNPMASS,
        cosmos_include_faint: bool=False,
        nimgs_calib: int=_commons.NIMGS_CALIB, min_idx_filename_ori: str=None,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        nongaussian: bool=False, switch_mode_for_uq: bool=False,
        niter_wiener: int=NITER_WIENER, noise_whitening_wiener: bool=False,
        multfact_step_size: float=_commons.MULTFACT_STEP_SIZE,
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
    denoiser, denoiser_uq = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        load_model_uq=load_model_uq, checkpoint_dir_uq=checkpoint_dir_uq,
        arch_uq=arch_uq, timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        imgsize=imgsize, verbose=verbose, **kwargs
    )

    # Initialize iterator
    calib_dataloader = iter(calib_dataset)

    # Instantiate physics (forward model)
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)

    # Instantiate the Wiener model
    wiener = _commons.get_wiener(
        path_to_ps=path_to_ps,
        white_noise=False, noise_whitening=noise_whitening_wiener,
        std_noise=std_noise, physics=physics,
        multfact_step_size=multfact_step_size, niter=niter_wiener,
        device=device, verbose=verbose
    )

    # Instantiate the PnP model
    pnpmass, pnpmass_uq, step_size = _commons.get_pnpmass(
        denoiser, denoiser_uq,
        std_noise=std_noise, mask=mask, physics=physics,
        step_size=step_size, niter=niter,
        nongaussian=nongaussian, switch_mode_for_uq=switch_mode_for_uq,
        wiener=wiener, device=device
    )

    # Run PnPMass for each batch
    kappa_true, _, kappa_pnpmass, _, res_pnpmass, _ = \
            _commons.run_wiener_pnpmass_batch(
        wiener, pnpmass, pnpmass_uq,
        physics, calib_dataloader, step_size, niter,
        confidence_uq=confidence_uq,
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
    _commons.save_output_pnpmass(
        out_dict, path_to_output, step_size, now,
        load_model_uq=load_model_uq, confidence_uq=confidence_uq,
        verbose=verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_calib_dataset", type=str,
        help="Path to the test set (HDF5 file)"
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
            f"Default = {_commons.MULTFACT_STEP_SIZE:.2f} * upper_bound, "
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
    parser.add_argument(
        "--nimgs-calib", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of calibration images. "
            f"Default = {_commons.NIMGS_CALIB}"
        )
    )
    _commons.add_arguments_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_nongaussian(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
