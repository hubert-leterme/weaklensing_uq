import argparse
import time
import tqdm
import torch

import wlmmuq.utils as wlutils

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons

OUTPUT_DIR = ""
OUTPUT_FILENAME = "results_deepmass"

def main(
        path_to_test_dataset: str, checkpoint_dir: str, checkpoint_dir_uq: str=None,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps: str=PATH_TO_PS,
        arch: str=None, timestamp: str=None, epoch: int=None,
        multfact_step_size: float=_commons.MULTFACT_STEP_SIZE,
        cosmos_include_faint: bool=False,
        nimgs_test: int=_commons.NIMGS_TEST,
        imgsize: int=_commons.IMGSIZE, batch_size: int=_commons.BATCH_SIZE,
        num_workers: int=NUM_WORKERS,
        niter_wiener: int=NITER_WIENER, noise_whitening_wiener: bool=False,
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
        convert_to_torch_tensor=True, inpainting=True,
        verbose=verbose
    ) # inpainting = True, as in training

    # Load test set
    test_dataset = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask
    )

    # Load arguments for Wiener initialization
    args_wienerinit = _commons.get_args_wienerinit(
        std_noise, mask, path_to_ps=path_to_ps,
        noise_whitening=noise_whitening_wiener,
        multfact_step_size=multfact_step_size, niter=niter_wiener,
        device=device, verbose=verbose
    )
    kwargs.update(args_wienerinit=args_wienerinit)

    # Initialize iterator
    test_dataloader = iter(test_dataset)

    # Load trained model
    deepmass, _ = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp,
        epoch=epoch, imgsize=imgsize,
        verbose=verbose, **kwargs
    )
    deepmass = deepmass.to(device)

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
        "args_wienerinit": args_wienerinit,
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
        "checkpoint_dir", type=str,
        help="Checkpoint directory (containing the './pe' and './var' subdirectories)"
    )
    parser.add_argument(
        "path_to_output", type=str,
        help="Path to the output file (without extension)"
    )
    _commons.add_arguments_model(parser)
    # _commons.add_arguments_model_uq(parser) # TODO: uncomment after update
    _commons.add_arguments_checkpoint(parser)
    parser.add_argument(
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of test images. "
            f"Default = {_commons.NIMGS_TEST}"
        )
    )
    _commons.add_arguments_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _commons.add_arguments_wiener(parser)
    _commons.add_arguments_output(parser, OUTPUT_FILENAME)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
