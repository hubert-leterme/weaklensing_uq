import argparse
import torch

from wlmmuq.data import NUM_WORKERS

import _commons
from _commons import IMGSIZE, BATCH_SIZE

NIMGS = 2048

def main(
        path_to_augmented_dataset, path_to_output, imgsize=IMGSIZE, nimgs=NIMGS,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        seed=None, verbose=False
):
    _commons.set_seed(seed)
    device = _commons.get_device(verbose=verbose)
    powerspectrum = _commons.get_powerspectrum_from_dataset(
        path_to_augmented_dataset, nimgs=nimgs,
        batch_size=batch_size, output_shape=imgsize,
        num_workers=num_workers, device=device, verbose=verbose
    ).cpu()
    torch.save(powerspectrum, path_to_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "path_to_output", type=str,
        help="Path to the output file (.pt)"
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images over which to compute the power spectrum. "
            f"Default = {NIMGS}"
        )
    )
    _commons.add_arguments_dataset(parser, batch_size=BATCH_SIZE)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
