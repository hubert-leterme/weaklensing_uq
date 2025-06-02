import os
import argparse
import random
import numpy as np
import torch

import wlmmuq.data.tensorflow as wlbl

MOMENT_ORDER = 1
IMGSIZE = 384
NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations in the original dataset
NIMGS_PS = 256 # To compute the power spectrum
BATCH_SIZE = 32
OUTPUT_DIR = '.'

def main(
        path_to_augmented_dataset, denoiser=False, moment_order=MOMENT_ORDER,
        path_to_pred_dataset=None, imgsize=IMGSIZE, nimgs=NIMGS_TRAIN,
        batch_size=BATCH_SIZE, keep_unsorted=None,
        output_dir=OUTPUT_DIR, seed=None, verbose=False, **kwargs
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Initialize batch generators
    if denoiser:
        batch_loader = wlbl.HDF5DatasetDenoiser
    else:
        batch_loader = wlbl.HDF5DatasetDeepMass

    # Check whether the dataset should be sorted by their original filenames
    if keep_unsorted is not None:
        kwargs.update(sort_by_filename_ori=False)

    if verbose:
        print("Initialize batch generators for training and validation")
    train_gen = batch_loader(
        order=moment_order, hdf5_filepath=path_to_augmented_dataset,
        pred_filepath=path_to_pred_dataset,
        nimgs=nimgs, batch_size=batch_size,
        output_shape=imgsize,
        newaxis=True, **kwargs
    )
    if verbose:
        print("Get one batch")
    kappa_inp, target = train_gen.load_batch()
    train_gen.close()
    if verbose:
        print("Save arrays")
    np.save(os.path.join(output_dir, 'kappa_inp.npy'), kappa_inp)
    np.save(os.path.join(output_dir, 'target.npy'), target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "--denoiser", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Generate noisy convergence maps instead of KS- or Wiener-estimations."
        )
    )
    parser.add_argument(
        "--scale", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Noise standard deviation, if option `--denoiser` is used."
        )
    )
    parser.add_argument(
        "--input-method", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Weak lensing method used as input ('ks' or 'wiener'). "
            "Default = None"
        )
    )
    parser.add_argument(
        "--moment-order", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Order of the moment network. "
            f"Default = {MOMENT_ORDER}"
        )
    )
    parser.add_argument(
        "--path-to-pred-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the prediction dataset (HDF5 file), computed with "
            "a previously-trained network. This is useful to train a moment "
            "network of order 2. Default = None"
        )
    )
    parser.add_argument(
        "--imgsize", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of pixels (width) in input images. "
            f"Default = {IMGSIZE}"
        )
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the dataset. "
            f"Default = {NIMGS_TRAIN}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size for training and validation. "
            f"Default = {BATCH_SIZE}"
        )
    )
    parser.add_argument(
        "--keep-unsorted", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not sort by filename in the original dataset. Useful to avoid IndexError "
            "when the dataset is incomplete."
        )
    )
    parser.add_argument(
        "-o", "--output-dir", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Directory in which to save the NumPy arrays. "
            f"Default = '{OUTPUT_DIR}'"
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
