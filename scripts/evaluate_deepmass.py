import os
import pickle
import argparse
import random

import numpy as np
from tensorflow import data, keras

import wlmmuq.kappatng as wlktng
import wlmmuq.cosmos as wlcosmos
import wlmmuq.utils as wlutils

INPUT_WLMETHOD = "wiener"
FWHM = 2.4 # As in Starck et al. (2021) (Gaussian smoothing for KS)
IMGSIZE = 304
NIMGS_TEST = 1024
BATCH_SIZE = 32
OFFSET = 0.5 # As in DeepMass

def main(
        path_to_test_set, path_to_model, path_to_output,
        input_wlmethod=INPUT_WLMETHOD,
        fwhm=FWHM, path_to_powerspectrum=None, imgsize=IMGSIZE,
        nimgs_test=NIMGS_TEST, batch_size=BATCH_SIZE,
        seed=None, verbose=False, **kwargs
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
        powerspectrum_1d = np.load(path_to_powerspectrum)
        kwargs.update(powerspectrum_1d=powerspectrum_1d)

    else:
        raise ValueError

    if verbose:
        print("Initialize batch generators for training and validation")
    test_gen = wlutils.HDF5BatchLoader(
        path_to_test_set, nimgs=nimgs_test, batch_size=batch_size,
        std_noise=std_noise, mask=mask, beg_idx=nimgs_test, shuffle=False,
        output_shape=imgsize, list_of_outputs=['kappa_inp', 'kappa_true'],
        offset=OFFSET, newaxis=True,
        input_method=input_wlmethod, **kwargs
    )

    # Load model
    cnn_model = keras.models.load_model(path_to_model)
    if verbose:
        cnn_model.summary()

    # Prefetch datasets for efficiency
    test_set_prefetched = test_gen.to_tf_dataset().prefetch(data.AUTOTUNE)

    # Fit model
    out_dict = cnn_model.evaluate(
        test_set_prefetched, return_dict=True
    )
    test_gen.close()

    # Pickle data
    with open(path_to_output, 'wb') as f:
        pickle.dump(out_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_test_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "path_to_model", type=str,
        help="Path to the trained model"
    )
    parser.add_argument(
        "path_to_output", type=str,
        help="Path to the output dictionary containing evaluation metrics"
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
            f"dataset. Default = None"
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
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the training set. "
            f"Default = {NIMGS_TEST}"
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
