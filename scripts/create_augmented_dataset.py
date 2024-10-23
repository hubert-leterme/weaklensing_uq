import argparse
import random
import numpy as np

import wlmmuq.kappatng as wlktng
import wlmmuq.cosmos as wlcosmos

OPENINGANGLE = 1.5 # Opening angle
NIMGS = 100 # Number of input images

def main(
        path_to_augmented_dataset, idx_lp=None,
        openingangle=OPENINGANGLE, nimgs=NIMGS, seed=None, verbose=False,
        **kwargs
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Get redshift weights from the COSMOS catalog
    if verbose:
        print("Computing redshift weights from COSMOS...")
    cat_cosmos_bright, _ = wlcosmos.cosmos_catalog()
    cat_cosmos_bright = cat_cosmos_bright[
        cat_cosmos_bright['zphot'] >= np.min(wlktng.LIST_OF_Z)
    ]
    cat_cosmos_bright = cat_cosmos_bright[
        cat_cosmos_bright['zphot'] < np.max(wlktng.LIST_OF_Z)
    ]
    weights_redshift = wlktng.get_weights(cat_cosmos_bright['zphot'])

    # Get nb of pixels in output images and adjust opening angle accordingly
    imgsize, openingangle = wlktng.get_npixels_openingangle(openingangle)

    # Create augmented dataset and store data
    wlktng.create_augmented_dataset(
        path_to_augmented_dataset, idx_lp, nimgs, weights_redshift, imgsize,
        verbose=verbose, **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "--idx-lp", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Index of the learning potential, indicating which folder to look "
            "into for the HDF5 files containing the dataset (`LPxxx` where `xxx` "
            "ranges from `001` to `100`). Default = `001`"
        )
    )
    parser.add_argument(
        "--openingangle", type=float,
        default=argparse.SUPPRESS,
        help=f"Opening angle (deg). Default = {OPENINGANGLE}"
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            f"Number of images to reconstruct. Default = {NIMGS}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size, to avoid memory overload. "
            "Default = 50"
        )
    )
    parser.add_argument(
        "--angle-batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of angles to compute before pickling, to avoid memory overload. "
            "Default = 36"
        )
    )
    parser.add_argument(
        "--angle-step", type=int,
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
        "--seed", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Seed for the random number generators"
        )
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        default=argparse.SUPPRESS
    )

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
