import argparse
import time

import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_DIR = ""
OUTPUT_FILENAME = "cqr_deepmass"

def main(
        path_to_calib_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        arch: str=None, timestamp: str=None, epoch: int=_commons.EPOCH,
        load_model_uq: bool=False,
        arch_uq: str=None, timestamp_uq: str=None, epoch_uq: int=None,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_DEEPMASS,
        nimgs_calib: int=_commons.NIMGS_CALIB, min_idx_filename_ori: str=None,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        niter_wiener: int=NITER_WIENER,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
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
    ) # E.g., "checkpoint/dir/cqr_deepmass"

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

    # Load trained models
    deepmass, deepmass_uq = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        load_model_uq=load_model_uq, checkpoint_dir_uq=checkpoint_dir_uq,
        arch_uq=arch_uq, timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        imgsize=imgsize,
        std_noise=std_noise, mask=mask, path_to_ps=path_to_ps,
        eps_sup_step_size_wiener=eps_sup_step_size,
        niter_wiener=niter_wiener,
        device=device, verbose=verbose, **kwargs
    )

    # Run DeepMass for each batch
    calib_dataloader = iter(calib_dataset)
    mask = mask.to(device)
    out_deepmass = _commons.run_deepmass_batch(
        deepmass, deepmass_uq,
        calib_dataloader,
        mask=mask, device=device, verbose=verbose,
    )
    kappa_true = out_deepmass["kappa_true"]
    kappa_deepmass = out_deepmass["kappa_deepmass"]
    var_deepmass = out_deepmass["var_deepmass"]

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    # Instantiate CQR model and compute the calibration parameters
    multfact_confidence_uq, addconst_confidence_uq = \
        _commons.convert_into_param_lists(
            multfact_confidence_uq, addconst_confidence_uq
        )

    for rho, const in zip(multfact_confidence_uq, addconst_confidence_uq):
        beg_time = time.time()

        cqr = _commons.get_cqr(
            kappa_deepmass, var_deepmass, kappa_true,
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
            "arch": arch,
            "nimgs_calib": nimgs_calib,
            "imgsize": imgsize,
            "confidence_uq": confidence_uq,
            "multfact_confidence_uq": rho,
        }
        _commons.save_results(
            out_dict, path_to_output, now,
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
    _commons.add_arguments_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_wiener(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
