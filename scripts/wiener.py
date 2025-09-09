import argparse
import time

import wlmmuq.models.deepinv.iterativemm as wlpnp
import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_FILENAME = "results_wiener"

def main(
        path_to_test_dataset: str, output_dir: str,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        niter_wiener: int=NITER_WIENER,
        cosmos_include_faint: bool=False, inpainting: bool=_commons.INPAINTING_WIENER,
        nimgs_test: int=_commons.NIMGS_TEST,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        eps_sup_step_size: float=_commons.EPS_SUP_STEP_SIZE,
        mode_cqr: str=_commons.MODE_CQR,
        confidence_uq: int | float=_commons.CONFIDENCE_UQ,
        multfact_confidence_uq: float=None,
        addconst_confidence_uq: float=None,
        cqr_filename: str=None, timestamp_cqr: str=None,
        save_tensors: bool=False, nimgs_save: int=_commons.NIMGS_SAVE,
        output_filename: str=OUTPUT_FILENAME,
        seed: int=None, verbose: bool=False
):
    _commons.set_seed(seed)

    if cqr_filename is not None:
        path_to_cqr = _commons.get_path_to_output(
            output_dir, cqr_filename
        ) # E.g., "checkpoint/dir/cqr_wiener"
    else:
        path_to_cqr = None
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

    # Load test set
    test_dataset = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask, shuffle=False
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
    test_dataloader = iter(test_dataset)
    mask = mask.to(device)
    out_wiener = _commons.run_wiener_batch(
        wiener, physics, test_dataloader,
        mask=mask, device=device, verbose=verbose,
    )
    kappa_true = out_wiener["kappa_true"]
    kappa_wiener = out_wiener["kappa_wiener"]
    var_wiener = out_wiener["var_wiener"]
    rmse = out_wiener["rmse"]

    inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

    # Calibrate with CQR, if available
    multfact_confidence_uq, addconst_confidence_uq = \
        _commons.convert_into_param_lists(
            multfact_confidence_uq, addconst_confidence_uq
        )

    for rho, const in zip(multfact_confidence_uq, addconst_confidence_uq):
        out_dict = _commons.apply_calibration_and_get_metrics(
            kappa_wiener, var_wiener, kappa_true,
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
            "niter": niter_wiener,
            "nimgs_test": nimgs_test,
            "imgsize": imgsize,
            "confidence_uq": confidence_uq,
            "rmse": rmse.cpu(),
        })
        if save_tensors:
            out_dict.update({
                "kappa_true": kappa_true[:nimgs_save].cpu(),
                "kappa_pred": kappa_wiener[:nimgs_save].cpu(),
                "var": var_wiener[:nimgs_save].cpu(),
            })
        _commons.save_results(
            out_dict, path_to_output, now,
            multfact_confidence_uq=rho,
            addconst_confidence_uq=addconst_confidence_uq,
            verbose=verbose
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_test_dataset", type=str,
        help="Path to the test set (HDF5 file)"
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Output directory (where the results will be saved)"
    )
    _commons.add_arguments_uq(parser)
    _commons.add_arguments_cqr(parser)
    _commons.add_arguments_test_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_wiener(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
