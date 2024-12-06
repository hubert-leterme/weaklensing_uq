import argparse
import warnings
import random
import h5py
import numpy as np
from tensorflow import keras, data

import wlmmuq.batchloader as wlbl

NIMGS = 72000
IMGSIZE = 304
NIMGS_ITER = 1024
BATCH_SIZE = 32
OFFSET = 0.5
IDX_DATASET = 'kappa_pred'

def main(
        path_to_trained_model, path_to_augmented_dataset, path_to_output_dataset,
        denoiser=False, nimgs=NIMGS, imgsize=IMGSIZE, nimgs_iter=NIMGS_ITER,
        batch_size=BATCH_SIZE, offset=OFFSET, idx_dataset=IDX_DATASET,
        seed=None, verbose=False, **kwargs
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if verbose:
        print("Initialize batch generator")

    if not denoiser:
        batch_loader = wlbl.HDF5BatchLoader
    else:
        batch_loader = wlbl.BaseHDF5BatchLoaderDenoiser

    # *** CAUTION ***
    # Keyword arguments `sort_by_filename_ori` and `shuffle` must be set to
    # False in order input convergence maps `kappa_inp` to be stored in the
    # same order as the targets `kappa_true`.
    data_gen = batch_loader(
        path_to_augmented_dataset,
        nimgs=nimgs, batch_size=batch_size,
        sort_by_filename_ori=False, shuffle=False,
        output_shape=imgsize, list_of_outputs=['kappa_inp'],
        offset=offset, newaxis=True, **kwargs
    )

    # Load trained model
    cnn_model = keras.models.load_model(path_to_trained_model)
    if verbose:
        cnn_model.summary()

    with h5py.File(path_to_output_dataset, 'w') as file:
        try:
            del file[idx_dataset]
        except KeyError:
            pass
        else:
            warnings.warn(
                f"Found existing dataset for {idx_dataset}; "
                "it will be overwritten."
            )
        file.create_dataset(
            idx_dataset, shape=(nimgs, imgsize, imgsize),
            dtype='float32'
        )
        end_idx = 0
        while end_idx < nimgs:
            beg_idx = end_idx
            end_idx = min(beg_idx + nimgs_iter, nimgs)
            ds = data_gen.to_tf_dataset(
                min_idx=beg_idx, max_idx=end_idx, raise_stop_iteration=True
            ).prefetch(data.AUTOTUNE)
            print(f"Processing images {beg_idx} to {end_idx}")
            kappa_pred = cnn_model.predict(
                ds, steps=(end_idx - beg_idx) // batch_size
            )
            kappa_pred -= offset # Remove offset before saving
            file[idx_dataset][beg_idx:end_idx] = kappa_pred[..., 0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_trained_model", type=str,
        help="Path to the trained model (keras file)"
    )
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "path_to_output_dataset", type=str,
        help="Path to the output dataset to be created (HDF5 file)"
    )
    parser.add_argument(
        "--denoiser", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Reconstruct the original convergence map from an input corrupted "
            "by a white Gaussian noise."
        )
    )
    parser.add_argument(
        "--scale", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Noise standard deviation for the denoiser. "
            "Must be provided if the `--denoiser` flag is used."
        )
    )
    parser.add_argument(
        "--input-method", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Weak lensing method used as input for DeepMass "
            "('ks', 'wiener' or 'wiener_pgd'). "
            "Must be provided unless the `--denoiser` flag is used."
        )
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the training set. "
            f"Default = {NIMGS}"
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
        "--nimgs-iter", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images per iteration, to be saved into the HDF5 file. "
            f"Default = {NIMGS_ITER}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            f"Batch size. Default = {BATCH_SIZE}"
        )
    )
    parser.add_argument(
        "--offset", type=float,
        default=argparse.SUPPRESS,
        help=(
            f"Default convergence value for a perfectly uniform universe. Default = {OFFSET:.2f}"
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
