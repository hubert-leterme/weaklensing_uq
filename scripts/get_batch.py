import os
import argparse
import random
import numpy as np

import wlmmuq.kappatng as wlktng
import wlmmuq.cosmos as wlcosmos
import wlmmuq.utils as wlutils

INPUT_WLMETHOD = "wiener"
FWHM = 2.4 # As in Starck et al. (2021) (Gaussian smoothing for KS)
IMGSIZE = 304
NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations in the original dataset
NIMGS_PS = 256 # To compute the power spectrum
BATCH_SIZE = 32
OFFSET = 0.5 # As in DeepMass
OUTPUT_DIR = '.'

def main(
        path_to_augmented_dataset, input_wlmethod=INPUT_WLMETHOD,
        fwhm=FWHM, path_to_powerspectrum=None, imgsize=IMGSIZE,
        nimgs_train=NIMGS_TRAIN, batch_size=BATCH_SIZE, offset=OFFSET,
        output_dir=OUTPUT_DIR, seed=None, verbose=False, **kwargs
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Compute a map of number of galaxies per pixels and a binary mask
    if verbose:
        print("Compute a map of number of galaxies per pixels and a binary mask")
    cat_cosmos_bright, _ = wlcosmos.cosmos_catalog()
    cat_cosmos_bright = wlktng.filter_by_redshifts(cat_cosmos_bright)
    data_dict = wlktng.get_data_from_cosmos_ktng(cat_cosmos_bright, imgsize)
    openingangle = data_dict["openingangle"]
    shapedisp = data_dict["shapedisp"]
    ngal = data_dict["ngal"]
    mask = data_dict["mask"]

    # Compute noise covariance matrix
    if verbose:
        print("Compute noise covariance matrix")
    std_noise = wlutils.get_std_noise(ngal, shapedisp, std_noise_mask=0)

    # Initialize batch generators for training and validation
    if input_wlmethod == 'ks':
        if fwhm is not None:
            resolution = openingangle / imgsize * 60. # arcmin/pixel
            std_gaussianfilter_arcmin = fwhm / (2 * np.sqrt(2 * np.log(2)))
            std_gaussianfilter = std_gaussianfilter_arcmin / resolution # pixels
            kwargs.update(std_gaussianfilter=std_gaussianfilter)

    elif input_wlmethod == 'wiener':
        if verbose:
            print("Estimate the power spectrum for Wiener filtering")

        if path_to_powerspectrum is None:
            # Load a set of convergence maps among the training set
            train_gen_ps = wlutils.HDF5BatchLoader(
                path_to_augmented_dataset, nimgs=NIMGS_PS, batch_size=NIMGS_PS,
                std_noise=std_noise, mask=mask, output_shape=imgsize,
                list_of_outputs=['kappa_true']
            )
            kappa_ps = train_gen_ps.load_batch()
            train_gen_ps.close()

            # Compute the 1D power spectrum
            powerspectrum_1d = wlutils.get_1d_powerspectrum(kappa_ps)
            del kappa_ps

        else:
            powerspectrum_1d = np.load(path_to_powerspectrum)

        kwargs.update(powerspectrum_1d=powerspectrum_1d)

    else:
        raise ValueError

    if verbose:
        print("Initialize batch generators for training and validation")
    train_gen = wlutils.HDF5BatchLoader(
        path_to_augmented_dataset, nimgs=nimgs_train, batch_size=batch_size,
        std_noise=std_noise, mask=mask, output_shape=imgsize,
        list_of_outputs=['kappa_inp', 'kappa_true'], offset=offset,
        newaxis=True, input_method=input_wlmethod, **kwargs
    )
    if verbose:
        print("Get one batch")
    kappa_inp, kappa_true = train_gen.load_batch()
    train_gen.close()
    if verbose:
        print("Save arrays")
    np.save(os.path.join(output_dir, 'kappa_inp.npy'), kappa_inp)
    np.save(os.path.join(output_dir, 'kappa_true.npy'), kappa_true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "--input-wlmethod", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Weak lensing method used as input ('wiener' or 'ks'). "
            f"Default = '{INPUT_WLMETHOD}'"
        )
    )
    parser.add_argument(
        "--fwhm", type=int,
        default=argparse.SUPPRESS,
        help=(
            "If the selected method is Kaiser-Squires ('ks'), FWHM of "
            f"the smoothing filter, in arcmin. Default = {FWHM}"
        )
    )
    parser.add_argument(
        "-ps", "--path-to-powerspectrum", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the .npy file containing the 1D power spectrum. "
            "If not provided, and if argument --input-wlmethod is set to "
            "'wiener', then the power spectrum will be inferred from the "
            "dataset. Default = None"
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
        "--nimgs-train", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the training set. "
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
        "--offset", type=float,
        default=argparse.SUPPRESS,
        help=(
            f"Default convergence value for a perfectly uniform universe. Default = {OFFSET:.2f}"
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
