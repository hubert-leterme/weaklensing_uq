import random
import argparse
import numpy as np
import torch
import astropy.table as aptable

from wlmmuq import cosmos as wlcosmos
from wlmmuq import kappatng as wlktng
from wlmmuq import utils as wlutils

from wlmmuq.kappatng import OPENINGANGLE

NINPIMGS = 100 # Number of input images

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_stdnoise_mask(
        imgsize, cosmos_include_faint=False, convert_to_torch_tensor=False,
        inpainting=False, seed=None, verbose=False
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if verbose:
        print("Load COSMOS galaxy shape catalog")
    cat_cosmos_bright, cat_cosmos_faint = wlcosmos.cosmos_catalog()
    cat_cosmos_bright = wlcosmos.filter_by_redshifts(cat_cosmos_bright, wlktng.MAX_Z)
    if cosmos_include_faint:
        cat_cosmos = aptable.vstack(
            [cat_cosmos_bright, cat_cosmos_faint], join_type='outer'
        )
    else:
        cat_cosmos = cat_cosmos_bright
    data_dict = wlktng.get_data_from_cosmos_ktng(cat_cosmos, imgsize)
    shapedisp = data_dict["shapedisp"]
    ngal = data_dict["ngal"]
    mask = data_dict["mask"]
    std_noise = wlutils.get_std_noise(ngal, shapedisp, std_noise_mask=0)
    if inpainting:
        std_noise[~mask] = np.max(std_noise) # Set the noise standard deviation for masked data

    if convert_to_torch_tensor:
        mask = torch.tensor(mask, dtype=bool)
        std_noise = torch.tensor(std_noise, dtype=torch.float32)

    return std_noise, mask


def create_dataset_from_kappatng(
        func:callable, path_to_saved_dataset:str, idx_lp: int | str,
        openingangle: float, ninpimgs: int, verbose: bool=False, **kwargs
):
    """
    Create a dataset from the KappaTNG simulation.
    The dataset is saved in the specified path.
    The dataset is created by calling the function `func` with the specified
    parameters.

    Parameters
    ----------
    func : callable
        Function to create the dataset: `wlmmuq.kappatng.create_cropped_dataset`
        or `wlmmuq.kappatng.create_augmented_dataset`.
    path_to_saved_dataset : str
        Path to save the dataset.
    idx_lp : int | str
        Index of the learning potential. It indicates which folder to look
        into for the HDF5 files containing the dataset (`LPxxx` where `xxx`
        ranges from `001` to `100`).
    openingangle : floatfrom wlmmuq.kappatng import OPENINGANGLE
        Additional arguments to pass to the function `func`.
    """
    # Get redshift weights from the COSMOS catalog
    if verbose:
        print("Computing redshift weights from COSMOS...")
    cat_cosmos_bright, _ = wlcosmos.cosmos_catalog()
    cat_cosmos_bright = wlcosmos.filter_by_redshifts(cat_cosmos_bright, wlktng.MAX_Z)
    weights_redshift = wlktng.get_weights(cat_cosmos_bright['zphot'])

    # Get nb of pixels in output images and adjust opening angle accordingly
    imgsize, openingangle = wlktng.get_npixels_openingangle(openingangle)

    # Create augmented dataset and store data
    func(
        path_to_saved_dataset, idx_lp, ninpimgs, weights_redshift, imgsize,
        verbose=verbose, **kwargs
    )


def add_arguments_create_dataset(parser):

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
        "--ninpimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            f"Number of input images. Default = {NINPIMGS}"
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
