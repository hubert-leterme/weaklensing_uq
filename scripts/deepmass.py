import argparse
import time
import tqdm
import torch
import wlmmuq.utils as wlutils

from wlmmuq.data import NUM_WORKERS

import _commons
from _commons import NIMGS_TEST, IMGSIZE, BATCH_SIZE, MULTFACT_STEP_SIZE

def main(
        path_to_test_dataset: str, path_to_checkpoint: str, path_to_ps: str, path_to_output: str,
        arch: str=None, step_size: float=None,
        multfact_step_size: float=MULTFACT_STEP_SIZE,
        cosmos_include_faint: bool=False,
        nimgs_test: int=NIMGS_TEST,
        imgsize: int=IMGSIZE, batch_size: int=BATCH_SIZE, num_workers: int=NUM_WORKERS,
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
        convert_to_torch_tensor=True, inpainting=True,
        verbose=verbose
    ) # inpainting = True, as in training

    # Load test set
    test_dataloader = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask
    )

    # Load arguments for Wiener initialization
    powerspectrum = torch.load(path_to_ps)
    step_size = multfact_step_size * wlutils.get_sup_step_size(
        std_noise=std_noise, mask=mask
    )
    args_wienerinit = dict(
        step_size=step_size, powerspectrum=powerspectrum,
        std_noise=std_noise, mask=mask
    )
    kwargs.update(args_wienerinit=args_wienerinit)

    # Initialize iterator
    test_dataloader = iter(test_dataloader)

    # Load trained model
    deepmass = _commons.load_trained_model(
        path_to_checkpoint, arch, imgsize, verbose=verbose, **kwargs
    ).to(device)

    # Run DeepMass for each batch
    listof_rmse = []
    mask = mask.to(device)
    pbar = tqdm.tqdm(test_dataloader, disable=not verbose)
    for kappa_true, gamma_noisy in pbar:
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_deepmass = deepmass(gamma_noisy)
            rmse = wlutils.rmse(kappa_deepmass, kappa_true, mask=mask)
            listof_rmse.append(rmse) # Shape = (batch_size,)
    rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs, niter)

    inference_time = time.time() - beg_time

    out_dict = {
        "rmse": rmse.cpu(),
        "inference_time": inference_time,
        "arch": arch,
        "nimgs_test": nimgs_test,
        "imgsize": imgsize,
        "step_size_wienerinit": step_size,
        "powerspectrum_wienerinit": powerspectrum
    }
    path_to_output_completed = f"{path_to_output}_{now}.pt"
    torch.save(out_dict, path_to_output_completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_test_dataset", type=str,
        help="Path to the test set (HDF5 file)"
    )
    parser.add_argument(
        "path_to_checkpoint", type=str,
        help="Path to the checkpoint containing the model's state dict (.pth.tar)"
    )
    parser.add_argument(
        "path_to_ps", type=str,
        help="Path to the power spectrum (for Wiener initialization)"
    )
    parser.add_argument(
        "path_to_output", type=str,
        help="Path to the output file (without extension)"
    )
    _commons.add_arguments_model(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Step size for Wiener initialization."
            f"Default = {MULTFACT_STEP_SIZE:.2f} * upper_bound, "
            "where upper_bound is computed from the noise standard deviation "
            "and the mask, using the power iteration method"
        )
    )
    parser.add_argument(
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of test images. "
            f"Default = {NIMGS_TEST}"
        )
    )
    _commons.add_arguments_dataset(parser, batch_size=BATCH_SIZE)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
