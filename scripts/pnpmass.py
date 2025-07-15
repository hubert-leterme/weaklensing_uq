import argparse
import time
import warnings
import torch

import wlmmuq.models.deepinv.iterativemm as wlpnp
import wlmmuq.models.cqr as wlcqr
import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_DIR = "results_pnpmass"
OUTPUT_FILENAME = "results_pnpmass"

def main(
        path_to_test_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
        path_to_cqr: str=None,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        load_model_uq: bool=False,
        arch_uq: str=None, timestamp_uq: str=None, epoch_uq: int=_commons.EPOCH,
        step_size: float | list[float]=None, niter: int=_commons.NITER_PNPMASS,
        cosmos_include_faint: bool=False,
        nimgs_test: int=_commons.NIMGS_TEST,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        nongaussian: bool=False, switch_mode_for_uq: bool=False,
        niter_wiener: int=NITER_WIENER, noise_whitening_wiener: bool=False,
        multfact_step_size: float=_commons.MULTFACT_STEP_SIZE,
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
        convert_to_torch_tensor=True, inpainting=False,
        verbose=verbose
    )

    # Load test set
    test_dataset = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask, shuffle=False
    )

    # Load trained denoiser
    denoiser, denoiser_uq = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        load_model_uq=load_model_uq, checkpoint_dir_uq=checkpoint_dir_uq,
        arch_uq=arch_uq, timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        imgsize=imgsize, verbose=verbose, **kwargs
    )

    # Get step size
    if not isinstance(step_size, list):
        step_size = [step_size]

    # Load CQR, if available
    if path_to_cqr is not None:
        if verbose:
            print("Load calibration function")
        alpha = wlutils.get_alpha_from_confidence(confidence_uq)
        cqr = wlcqr.AddCQR(alpha, map_size=imgsize)
        checkpoint_cqr = torch.load(path_to_cqr)
        assert confidence_uq == checkpoint_cqr["confidence_uq"]
        nimgs_calib = checkpoint_cqr["nimgs_calib"]
        cqr.load_state_dict(checkpoint_cqr["state_dict"])
        cqr.eval().to(device)
    else:
        nimgs_calib = None
        cqr = None

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

    for tau in step_size:
        # Initialize iterator
        test_dataloader = iter(test_dataset)

        # Instantiate the PnP model
        pnpmass, pnpmass_uq, tau = _commons.get_pnpmass(
            denoiser, denoiser_uq,
            std_noise=std_noise, mask=mask, physics=physics,
            step_size=tau, niter=niter,
            nongaussian=nongaussian, switch_mode_for_uq=switch_mode_for_uq,
            wiener=wiener, device=device
        )

        # Run PnPMass for each batch
        kappa_true, kappa_wiener, kappa_pnpmass, var_pnpmass, res_pnpmass, rmse_iter = \
                _commons.run_wiener_pnpmass_batch(
            wiener, pnpmass, pnpmass_uq,
            physics, test_dataloader, tau, niter,
            confidence_uq=confidence_uq,
            device=device, verbose=verbose,
        )

        inference_time = time.time() - beg_time
        if verbose:
            print(f"Total inference time: {inference_time:.2f} seconds")

        # Calibrate with CQR, if available
        if cqr is not None:
            beg_time = time.time()
            if verbose:
                print("Calibrate residuals with CQR")
            if tau != checkpoint_cqr["step_size"]:
                warnings.warn(
                    f"Step size {tau:.2e} does not match the step size "
                    f"{checkpoint_cqr['step_size']:.2e} used for CQR. "
                    "Calibration may be inaccurate."
                )
            res_pnpmass_cqr = cqr(res_pnpmass)
            cqr_time = time.time() - beg_time
            if verbose:
                print(f"Calibration time: {cqr_time:.2f} seconds")
        else:
            res_pnpmass_cqr = None
            cqr_time = None

        # Compute miscoverage rate and size of prediction intervals
        beg_time = time.time()
        mask = mask.to(device)
        err_pnpmass, predinterv_pnpmass, _, _ = wlutils.get_metrics(
            kappa_pnpmass, res_pnpmass, kappa_true, mask=mask
        )
        if res_pnpmass_cqr is not None:
            err_pnpmass_cqr, predinterv_pnpmass_cqr, _, _ = wlutils.get_metrics(
                kappa_pnpmass, res_pnpmass_cqr, kappa_true, mask=mask
            )
        else:
            err_pnpmass_cqr = None
            predinterv_pnpmass_cqr = None
        metrics_time = time.time() - beg_time
        if verbose:
            print(f"Metrics computation time: {metrics_time:.2f} seconds")

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
            if wiener is not None:
                out_dict.update({
                    "kappa_wiener": kappa_wiener.cpu(),
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
        _commons.save_output_pnpmass(
            out_dict, path_to_output, tau, now,
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
    parser.add_argument(
        "-cqr", "--path-to-cqr", type=str, default=None,
        help=(
            "Path to the CQR checkpoint (optional). "
            "If provided, the residuals will be calibrated with CQR"
        )
    )
    _commons.add_arguments_model(parser)
    _commons.add_arguments_model_uq(parser)
    _commons.add_arguments_checkpoint(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Step size for the PnPMass algorithm. Several values can be provided. "
            "If not provided or set to 0, the step size will be computed as "
            f"{_commons.MULTFACT_STEP_SIZE:.2f} * upper_bound, "
            "where upper_bound is estimated from the noise standard deviation "
            "and the mask, using the power iteration method."
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
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of test images. "
            f"Default = {_commons.NIMGS_TEST}"
        )
    )
    _commons.add_arguments_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_nongaussian(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
