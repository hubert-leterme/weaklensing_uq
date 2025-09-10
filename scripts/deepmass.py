import argparse
import time

import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_DIR = ""
OUTPUT_FILENAME = "results_deepmass"
from pnpmass_calibration import OUTPUT_DIR as CQR_DIR

def main(
        path_to_test_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
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
        mode_cqr: str=_commons.MODE_CQR,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        multfact_confidence_uq: float=None,
        addconst_confidence_uq: float=None,
        cqr_dir: str=CQR_DIR,
        cqr_filename: str=None, timestamp_cqr: str=None,
        save_tensors: bool=False, nimgs_save: int=_commons.NIMGS_SAVE,
        output_dir: str=OUTPUT_DIR, output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    if cqr_filename is not None:
        path_to_cqr = _commons.get_path_to_output(
            cqr_dir, cqr_filename, checkpoint_dir=checkpoint_dir
        ) # E.g., "checkpoint/dir/cqr_deepmass"
    else:
        path_to_cqr = None
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

    # Run DeepMass for each batch
    test_dataloader = iter(test_dataset)
    mask = mask.to(device)
    out_deepmass = _commons.run_deepmass_batch(
        deepmass, deepmass_uq, test_dataloader,
        mask=mask, device=device, verbose=verbose,
    )
    kappa_true = out_deepmass["kappa_true"]
    kappa_deepmass = out_deepmass["kappa_deepmass"]
    var_deepmass = out_deepmass["var_deepmass"]
    rmse = out_deepmass["rmse"]
    nrmse = out_deepmass["nrmse"]

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    # Calibrate with CQR, if available
    multfact_confidence_uq, addconst_confidence_uq = \
        _commons.convert_into_param_lists(
            multfact_confidence_uq, addconst_confidence_uq
        )

    for rho, const in zip(multfact_confidence_uq, addconst_confidence_uq):
        out_dict = _commons.apply_calibration_and_get_metrics(
            kappa_deepmass, var_deepmass, kappa_true,
            path_to_cqr, timestamp_cqr,
            confidence_uq=confidence_uq,
            imgsize=imgsize, mode=mode_cqr,
            multfact_confidence_uq=rho,
            addconst_confidence_uq=const,
            mask=mask, save_tensors=save_tensors, nimgs_save=nimgs_save,
            device=device, verbose=verbose
        )
        out_dict.update({
            "inference_time": inference_time,
            "arch": arch,
            "nimgs_test": nimgs_test,
            "imgsize": imgsize,
            "confidence_uq": confidence_uq,
            "rmse": rmse.cpu(),
            "nrmse": nrmse.cpu(),
        })
        if save_tensors:
            out_dict.update({
                "kappa_true": kappa_true[:nimgs_save].cpu(),
                "kappa_pred": kappa_deepmass[:nimgs_save].cpu(),
                "var": var_deepmass[:nimgs_save].cpu(),
            })
        _commons.save_results(
            out_dict, path_to_output, now,
            multfact_confidence_uq=rho,
            addconst_confidence_uq=const,
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
    _commons.add_arguments_uq(parser)
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
