import argparse

import _commons
from _commons import OPENINGANGLE, NINPIMGS

import wlmmuq.kappatng as wlktng

def main(
        path_to_augmented_dataset, idx_lp=None,
        openingangle=OPENINGANGLE, ninpimgs=NINPIMGS,
        seed=None, verbose=False, **kwargs
):
    _commons.set_seed(seed)
    _commons.create_dataset_from_kappatng(
        wlktng.create_augmented_dataset,
        path_to_augmented_dataset, idx_lp, openingangle, ninpimgs,
        verbose=verbose, **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    _commons.add_arguments_create_dataset(parser)
    parser.add_argument(
        "--angle-batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of angles to compute before pickling, to avoid memory overload. "
            "Default = 36"
        )
    )
    parser.add_argument(
        "--angle-step", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Interval between two rotation angles (deg). "
            "Default = 5"
        )
    )
    parser.add_argument(
        "--niter-per-angle", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of random crops for each rotation angle. "
            "Default = 1"
        )
    )
    _commons.add_arguments_seed_verbose(parser)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
