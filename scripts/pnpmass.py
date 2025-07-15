import argparse
import time
import warnings
import torch

import wlmmuq.models.cqr as wlcqr
import wlmmuq.utils as wlutils

from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

NIMGS_SAVE = 16

import _commons

def main(
        path_to_test_dataset: str, model_dir: str,
        output_filename: str, path_to_cqr: str=None,
        arch: str=None, load_model_uq: bool=False,
        step_size: float | list[float]=None, niter: int=_commons.NITER_PNPMASS,
        cosmos_include_faint: bool=False,
        nimgs_test: int=_commons.NIMGS_TEST,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        wiener_init: bool=False, path_to_ps: str=None,
        niter_wiener: int=NITER_WIENER,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        save_tensors: bool=False, nimgs_save: int=NIMGS_SAVE,
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
        num_workers, std_noise, mask, shuffle=False
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
    if not isinstance(step_size, list):
        step_size = [step_size]

    # Get iterative Wiener filtering (may be used for initialization)
    wiener = _commons.get_wiener(
        path_to_ps, std_noise, mask, niter=niter_wiener, verbose=verbose
    )

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
        if wiener_init:
            if wiener is None:
                raise ValueError("The path to the power spectrum must be provided.")
            init_estimate = wiener
        else:
            init_estimate = None
        pnpmass, physics = _commons.get_pnpmass(
            std_noise, mask, denoiser, denoiser_uq, niter, step_size=tau,
            init_estimate=init_estimate
        )
        pnpmass = pnpmass.to(device)
        if wiener is not None:
            wiener = wiener.to(device)
        physics = physics.to(device)

        # Run PnPMass for each batch
        gamma_noisy, kappa_true, kappa_wiener, \
                kappa_pnpmass, var_pnpmass, res_pnpmass, rmse_iter = \
                    _commons.run_wiener_pnpmass_batch(
            wiener, pnpmass, physics, test_dataloader, tau, niter,
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
                "gamma_noisy": gamma_noisy[:nimgs_save].cpu(),
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
        _commons.save_results(
            out_dict, model_dir, "pnpmass",
            f"{output_filename}_step-size_{tau:.3f}_{confidence_uq}-sigma_{now}.pt",
            verbose=verbose
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_test_dataset", type=str,
        help="Path to the test set (HDF5 file)"
    )
    _commons.add_arguments_model_dir(parser)
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
    _commons.add_arguments_wienerinit(parser)
    parser.add_argument(
        "--save-tensors", action='store_true',
        help=(
            "If set, the tensors of the true convergence, "
            "the PnPMass estimate, the variance, and the residuals "
            "will be saved in the output file. WARNING: this will increase "
            "the size of the output file significantly!"
        )
    )
    parser.add_argument(
        "--nimgs-save", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images to save. "
            f"Default = {NIMGS_SAVE}"
        )
    )
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
