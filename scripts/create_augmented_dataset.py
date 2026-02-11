import argparse

import _commons
import _add_arguments

from _commons import NINPIMGS

import wlmmuq
import wlmmuq.datasets.kappatng as wlktng

from wlmmuq.datasets import NUM_WORKERS

IDX_LP = "002" # Lensing potential used for training/validation

def main(
        path_to_output=wlmmuq.PATH_TO_TRAIN_VAL_DATASET,
        idx_lp=IDX_LP,
        openingangle=wlktng.OPENINGANGLE, ninpimgs=NINPIMGS,
        seed=None, verbose=False, **kwargs
):
    _commons.set_seed(seed)
    _commons.create_dataset_from_kappatng(
        wlktng.create_augmented_dataset,
        path_to_output, idx_lp, openingangle, ninpimgs,
        verbose=verbose, **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _add_arguments.create_dataset(parser, wlmmuq.PATH_TO_TRAIN_VAL_DATASET, IDX_LP)
    parser.add_argument(
        "--angle-batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of angles to compute before saving to the HDF5 file, "
            f"to avoid memory overload. Default = {wlktng.ANGLE_BATCH_SIZE}"
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
    parser.add_argument(
        "-w", "--num-workers", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of workers for parallel processing. "
            f"Default = {NUM_WORKERS}"
        )
    )
    parser.add_argument(
        "-r", "--resume", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Resume process, if started but not complete."
        )
    )
    _add_arguments.seed_verbose(parser)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
