import argparse
import time

import wlmmuq.models.deepinv.iterativemm as wlpnp
import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_FILENAME = "cqr_wiener"

def main(
        path_to_calib_dataset: str, output_dir: str,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        niter_wiener: int=NITER_WIENER,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_PNPMASS,
        nimgs_calib: int=_commons.NIMGS_CALIB, min_idx_filename_ori: str=None,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False
):
    _commons.set_seed(seed)

    path_to_output = _commons.get_path_to_output(
        output_dir, output_filename
    ) # E.g., "output/dir/results_wiener"

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

    # Instantiate physics (forward model)
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)

    # Instantiate the Wiener model
    wiener = _commons.get_wiener(
        path_to_ps=path_to_ps,
        white_noise=False,
        std_noise=std_noise, physics=physics,
        eps_sup_step_size=eps_sup_step_size, niter=niter_wiener,
        device=device, verbose=verbose
    )

    # Run iterative Wiener for each batch
    calib_dataloader = iter(calib_dataset)
    mask = mask.to(device)
    out_wiener = _commons.run_wiener_batch(
        wiener, physics, calib_dataloader, confidence_uq=confidence_uq,
        mask=mask, device=device, verbose=verbose,
    )
    kappa_true = out_wiener["kappa_true"]
    kappa_wiener = out_wiener["kappa_wiener"]
    res_wiener = out_wiener["res_wiener"]

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    # Instantiate CQR model and compute the calibration parameters
    cqr = _commons.get_cqr(
        kappa_wiener, res_wiener, kappa_true, imgsize, confidence_uq,
        device=device, verbose=verbose
    )

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    out_dict = {
        "state_dict": cqr.state_dict(),
        "inference_time": inference_time,
        "nimgs_calib": nimgs_calib,
        "imgsize": imgsize,
        "confidence_uq": confidence_uq,
    }
    _commons.save_results(
        out_dict, path_to_output, now, verbose=verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_calib_dataset", type=str,
        help="Path to the calibration set (HDF5 file)"
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Output directory (where the results will be saved)"
    )
    _commons.add_arguments_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_wiener(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
