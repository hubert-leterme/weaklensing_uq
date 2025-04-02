import argparse
import random
import numpy as np
import wlmmuq.utils as wlutils
import wlmmuq.data.tensorflow as wlbl

IMGSIZE = 304
NIMGS = 2048
BATCH_SIZE = 256

def main(
        path_to_dataset, path_to_output, imgsize=IMGSIZE, nimgs=NIMGS,
        batch_size=BATCH_SIZE,
        seed=None, verbose=False
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data_loader = wlbl.HDF5Dataset(
        path_to_dataset, nimgs=nimgs, batch_size=batch_size,
        output_shape=imgsize, list_of_outputs=['kappa_true'],
        shuffle=True
    )
    data_gen = data_loader.to_tf_dataloader(raise_stop_iteration=True)
    data_gen = iter(data_gen)

    list_of_powerspectrum_1d = []
    nsteps = nimgs // batch_size
    for i in range(nsteps):
        if verbose:
            print(f"Processing batch nb {i}")
        kappa_ps = next(data_gen)
        list_of_powerspectrum_1d.append(
            wlutils.get_1d_powerspectrum(kappa_ps)
        )

    powerspectrum_1d = np.stack(list_of_powerspectrum_1d, axis=0)
    powerspectrum_1d = np.mean(powerspectrum_1d, axis=0)

    np.save(path_to_output, powerspectrum_1d)

    data_loader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_dataset", type=str,
        help=(
            "Path to the dataset from which to compute "
            "the power spectrum (HDF5 file)"
        )
    )
    parser.add_argument(
        "path_to_output", type=str,
        help=(
            "Path to the output file (.npy)"
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
            "Number of images to compute. "
            f"Default = {NIMGS}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size. "
            f"Default = {BATCH_SIZE}"
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
