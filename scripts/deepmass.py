import argparse
import time

import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_DIR = ""
OUTPUT_FILENAME = "results_deepmass"

def main(
        path_to_test_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
        path_to_cqr: str=None,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        load_model_uq: bool=False,
        arch_uq: str=None, timestamp_uq: str=None, epoch_uq: int=None,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_DEEPMASS,
        nimgs_test: int=_commons.NIMGS_TEST,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        niter_wiener: int=NITER_WIENER,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        save_tensors: bool=False, nimgs_save: int=_commons.NIMGS_SAVE,
        output_dir: str=OUTPUT_DIR, output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    path_to_output = _commons.get_path_to_output(
        output_dir, output_filename, checkpoint_dir=checkpoint_dir
    ) # E.g., "checkpoint/dir/results_deepmass"

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

    # Load arguments for Wiener initialization
    args_wienerinit = _commons.get_args_wienerinit(
        std_noise, mask, path_to_ps=path_to_ps,
        eps_sup_step_size=eps_sup_step_size, niter=niter_wiener,
        device=device, verbose=verbose
    )
    kwargs.update(args_wienerinit=args_wienerinit)

    # Load trained models
    deepmass, deepmass_uq = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        load_model_uq=load_model_uq, checkpoint_dir_uq=checkpoint_dir_uq,
        arch_uq=arch_uq, timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        imgsize=imgsize, device=device, verbose=verbose, **kwargs
    )

    # Load CQR, if available
    nimgs_calib, cqr = _commons.load_cqr(
        path_to_cqr, confidence_uq, imgsize, parent_dir=checkpoint_dir,
        device=device, verbose=verbose
    )

    # Run DeepMass for each batch
    test_dataloader = iter(test_dataset)
    mask = mask.to(device)
    out_deepmass = _commons.run_deepmass_batch(
        deepmass, deepmass_uq,
        test_dataloader, confidence_uq=confidence_uq,
        mask=mask, device=device, verbose=verbose,
    )
    kappa_true = out_deepmass["kappa_true"]
    kappa_deepmass = out_deepmass["kappa_deepmass"]
    var_deepmass = out_deepmass["var_deepmass"]
    res_deepmass = out_deepmass["res_deepmass"]
    rmse = out_deepmass["rmse"]

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    # Calibrate with CQR, if available
    res_deepmass_cqr, cqr_time = _commons.get_calibrated_residuals(
        cqr, res_deepmass, verbose=verbose
    )

    # Compute miscoverage rate and size of prediction intervals
    err_deepmass, predinterv_deepmass, err_deepmass_cqr, \
            predinterv_deepmass_cqr, metrics_time = _commons.get_metrics(
        kappa_deepmass, res_deepmass, kappa_true, res_cqr=res_deepmass_cqr,
        mask=mask, verbose=verbose
    )

    out_dict = {
        "inference_time": inference_time,
        "metrics_time": metrics_time,
        "arch": arch,
        "nimgs_test": nimgs_test,
        "imgsize": imgsize,
        "confidence_uq": confidence_uq,
        "rmse": rmse.cpu(),
        "err_deepmass": err_deepmass.cpu(),
        "predinterv_deepmass": predinterv_deepmass.cpu(),
    }
    if save_tensors:
        out_dict.update({
            "kappa_true": kappa_true[:nimgs_save].cpu(),
            "kappa_deepmass": kappa_deepmass[:nimgs_save].cpu(),
            "var_deepmass": var_deepmass[:nimgs_save].cpu(),
            "res_deepmass": res_deepmass[:nimgs_save].cpu(),
        })
    if cqr is not None:
        out_dict.update({
            "cqr_time": cqr_time,
            "nimgs_calib": nimgs_calib,
            "err_deepmass_cqr": err_deepmass_cqr.cpu(),
            "predinterv_deepmass_cqr": predinterv_deepmass_cqr.cpu(),
        })
        if save_tensors:
            out_dict.update({
                "res_deepmass_cqr": res_deepmass_cqr[:nimgs_save].cpu(),
            })
    _commons.save_results(
        out_dict, path_to_output, now, verbose=verbose
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
    _commons.add_arguments_test_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_wiener(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
