import argparse
import time
import warnings
import torch

import wlmmuq.models.cqr as wlcqr
import wlmmuq.utils as wlutils

from wlmmuq.data import NUM_WORKERS

import _commons

def main(
        path_to_test_dataset: str, checkpoint_dir: str, path_to_output: str,
        path_to_cqr: str=None,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        load_model_uq: bool=False, timestamp_uq: str=None, epoch_uq: int=None,
        step_size: float | list[float]=None, niter: int=_commons.NITER_PNPMASS,
        cosmos_include_faint: bool=False,
        nimgs_test: int=_commons.NIMGS_TEST,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
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

    # Load test set
    test_dataset = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask
    )

    # Load trained denoiser
    denoiser, denoiser_uq = _commons.load_trained_model(
        checkpoint_dir, arch, imgsize, timestamp, epoch,
        load_model_uq=load_model_uq,
        timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        verbose=verbose, **kwargs
    )

    # Instantiate data fidelity, prior, metrics, and physics
    data_fidelity, prior, prior_uq, rmse, physics = _commons.get_pnpmass_modules(
        std_noise, mask, denoiser, denoiser_uq
    )
    physics = physics.to(device)

    # Get step size
    step_size = _commons.get_pnpmass_step_size(
        std_noise, mask, step_size=step_size
    )
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

    for tau in step_size:
        # Initialize iterator
        test_dataloader = iter(test_dataset)

        # Instantiate the PnP model
        pnpmass = _commons.get_pnpmass(
            data_fidelity, prior, prior_uq, rmse, niter, step_size=tau
        ).to(device)

        # Run PnPMass for each batch
        kappa_true, kappa_pnpmass, var_pnpmass, res_pnpmass, rmse_iter = \
                _commons.run_pnpmass_batch(
            pnpmass, physics, test_dataloader, tau, niter,
            confidence_uq=confidence_uq,
            device=device, verbose=verbose,
        )

        # Calibrate with CQR, if available
        if cqr is not None:
            if verbose:
                print("Calibrate residuals with CQR")
            if tau != checkpoint_cqr["step_size"]:
                warnings.warn(
                    f"Step size {tau:.2e} does not match the step size "
                    f"{checkpoint_cqr['step_size']:.2e} used for CQR. "
                    "Calibration may be inaccurate."
                )
            res_pnpmass_cqr = cqr(res_pnpmass)
        else:
            res_pnpmass_cqr = None

        # Compute miscoverage rate and size of prediction intervals
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

        inference_time = time.time() - beg_time

        out_dict = {
            "inference_time": inference_time,
            "step_size": tau,
            "arch": arch,
            "niter": niter,
            "nimgs_test": nimgs_test,
            "imgsize": imgsize,
            "confidence_uq": confidence_uq,
            "kappa_true": kappa_true.cpu(),
            "kappa_pnpmass": kappa_pnpmass.cpu(),
            "var_pnpmass": var_pnpmass.cpu(),
            "res_pnpmass": res_pnpmass.cpu(),
            "rmse_iter": rmse_iter.cpu(),
            "err_pnpmass": err_pnpmass.cpu(),
            "predinterv_pnpmass": predinterv_pnpmass.cpu(),
        }
        if cqr is not None:
            out_dict.update({
                "nimgs_calib": nimgs_calib,
                "res_pnpmass_cqr": res_pnpmass_cqr.cpu(),
                "err_pnpmass_cqr": err_pnpmass_cqr.cpu(),
                "predinterv_pnpmass_cqr": predinterv_pnpmass_cqr.cpu(),
            })
        path_to_output_completed = (
            f"{path_to_output}_step-size_{tau:.3f}_{confidence_uq}-sigma_{now}.pt"
        )
        torch.save(out_dict, path_to_output_completed)


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
        "path_to_output", type=str,
        help="Path to the output file (without extension)"
    )
    parser.add_argument(
        "-cqr", "--path-to-cqr", type=str, default=None,
        help=(
            "Path to the CQR checkpoint (optional). "
            "If provided, the residuals will be calibrated with CQR"
        )
    )
    _commons.add_arguments_model(parser)
    _commons.add_arguments_checkpoint(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float, nargs='+',
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
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of test images. "
            f"Default = {_commons.NIMGS_TEST}"
        )
    )
    _commons.add_arguments_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
