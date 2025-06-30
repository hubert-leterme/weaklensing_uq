import random
import argparse
import numpy as np
import torch
import astropy.table as aptable

from wlmmuq import cosmos as wlcosmos
from wlmmuq import kappatng as wlktng
from wlmmuq import utils as wlutils
from wlmmuq.data import torch as wlbl
from wlmmuq import models as wlnn

from wlmmuq.kappatng import OPENINGANGLE
from wlmmuq.data import NUM_WORKERS

NINPIMGS = 100 # Number of input images before cropping
NIMGS_TEST = 512 # Images extracted from the 57 first original files
IMGSIZE = 384
BATCH_SIZE = 32
KEYS_MODEL = ['model_size', 'args_wienerinit']
MULTFACT_STEP_SIZE = 0.99 # Fraction of the upper limit for the step size

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_device(verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
    return device


def get_stdnoise_mask(
        imgsize, cosmos_include_faint=False, convert_to_torch_tensor=False,
        inpainting=False, verbose=False
):
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


def get_powerspectrum_from_dataset(
        hdf5_filepath, nimgs, device=None, verbose=False, **kwargs
):
    if verbose:
        print(f"Compute the power spectrum of {nimgs} images")
    dataloader = wlbl.HDF5DatasetKappa(
        hdf5_filepath, nimgs=nimgs, shuffle=True, **kwargs
    ).to_dataloader()
    dataloader = iter(dataloader)

    list_of_powerspectrum = []
    while True:
        try:
            kappa_ps = next(dataloader)
        except StopIteration:
            break
        if device is not None:
            kappa_ps = kappa_ps.to(device)
        list_of_powerspectrum.append(
            wlutils.get_powerspectrum(kappa_ps)
        )
    powerspectrum = torch.stack(list_of_powerspectrum, dim=0)
    powerspectrum = torch.mean(powerspectrum, dim=0)

    return powerspectrum


def load_trained_model(path_to_checkpoint, arch, imgsize, verbose=False, **kwargs):

    if arch is None:
        raise ValueError(
            "Model architecture must be provided with -a or --arch"
        )
    kwargs_model = {k: kwargs.pop(k) for k in KEYS_MODEL if k in kwargs}
    cnn_class, _ = wlnn.MODEL_CLASSES[arch]
    model = cnn_class(
        map_size=imgsize, **kwargs_model
    )
    checkpoint = torch.load(path_to_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if verbose:
        model.summary()

    return model


def get_dataloader_massmapping(
        path_to_dataset, nimgs, imgsize, batch_size, num_workers, std_noise, mask
):
    test_dataloader = wlbl.HDF5DatasetMassMapping(
        hdf5_filepath=path_to_dataset, nimgs=nimgs, batch_size=batch_size,
        std_noise=std_noise, mask=mask, shuffle=False,
        output_shape=imgsize, newaxis=True, num_workers=num_workers
    ).to_dataloader()

    return test_dataloader


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

def add_arguments_model(parser):

    parser.add_argument(
        "-a", "--arch", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the model. Possible values are: "
            f"{' | '.join(wlnn.MODEL_CLASSES.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "-s", "--model-size", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Size of the model (DRUNet only). Possible values are: "
            f"{' | '.join(wlnn.torch.MODEL_SIZE_DRUNET.keys())}. Default = None"
        )
    )

def add_arguments_dataset(parser, batch_size):

    parser.add_argument(
        "--imgsize", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of pixels (width) in input images. "
            f"Default = {IMGSIZE}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size. "
            f"Default = {batch_size}"
        )
    )
    parser.add_argument(
        "-w", "--num-workers", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of workers for parallel processing. Only work for PyTorch datasets. "
            f"Default = {NUM_WORKERS}"
        )
    )

def add_arguments_seed_verbose(parser):

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
