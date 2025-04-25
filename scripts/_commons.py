import random
import numpy as np
import torch
import astropy.table as aptable

from wlmmuq import cosmos as wlcosmos
from wlmmuq import kappatng as wlktng
from wlmmuq import utils as wlutils

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
    cat_cosmos_bright = wlktng.filter_by_redshifts(cat_cosmos_bright)
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
