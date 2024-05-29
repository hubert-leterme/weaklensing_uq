import os
import random
import numpy as np
import h5py

from . import CONFIG_DATA

KTNG_DIR = os.path.expanduser(CONFIG_DATA['ktng_dir'])

LIST_OF_Z = np.loadtxt(os.path.join(KTNG_DIR, 'zs.dat'))

WIDTH_ORI = 1024 # size of the simulated convergence maps (nb pixels)
WIDTH = 360 # size of the target convergence maps (nb pixels)
SIZE_ORI = 5. # opening angle of the simulated convergence maps (deg)
SIZE = SIZE_ORI * WIDTH / WIDTH_ORI # opening angle of the target convergence maps (deg)
RESOLUTION = SIZE_ORI / WIDTH_ORI * 60. # resolution in arcmin/pixel

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))

def get_openingangle(width=WIDTH):
    return width * RESOLUTION / 60.

def get_npixels(size=SIZE, make_even=True):
    if not make_even:
        mult = 1
    else:
        mult = 2
    width = mult * int(size / (mult * RESOLUTION) * 60.)
    size = get_openingangle(width)
    return width, size

def _split_map(kappa, width_ori, width, n_samples_per_side):

    step = (width_ori - width) // (n_samples_per_side - 1)
    out = []
    beg_i = 0
    for i in range(n_samples_per_side):
        beg_j = 0
        for j in range(n_samples_per_side):
            subkappa = kappa[beg_i:beg_i+width, beg_j:beg_j+width]
            subkappa = subkappa - np.mean(subkappa) # Center-normalize the convergence map
            out.append(subkappa)
            beg_j += step
        beg_i += step

    return out


def kappa_tng(
        weights, ninpimgs, start_idx=0, width=WIDTH, nsamples_per_side=3, shuffle=False
):
    """
    Parameters
    ----------
    weights (list of float)
    ninpimgs (int)
        Number of input images to load, before cropping and data augmentation.
    start_idx (int, default=0)
        Index of the first image to load.
    width (int, default=360)
        Size of the target convergence maps (nb pixels)
    nsamples_per_side (int, default=3)
        Used for cropping input images
    shuffle (bool, default=False)
    
    """
    list_of_idx_dataset = np.arange(start_idx, start_idx+ninpimgs) + 1
    list_of_idx_dataset = vectorized_zfill(list_of_idx_dataset)

    list_of_kappa = []
    list_of_idx_redshift = None
    for idx_dataset in list_of_idx_dataset:
        fname = os.path.join(KTNG_DIR, f"LP001_run{idx_dataset}_maps.hdf5")
        with h5py.File(fname, 'r') as file:
            kappa = np.zeros((WIDTH_ORI, WIDTH_ORI))
            if list_of_idx_redshift is None:
                list_of_idx_redshift = sorted(file.keys())[1:]
                nredshifts = len(list_of_idx_redshift)
                if len(weights) != nredshifts:
                    raise ValueError(
                        f"Positional argument `weights` must have {nredshifts} elements"
                    )
            for idx_redshift, weight in zip(list_of_idx_redshift, weights):
                kappa += weight * file[f'{idx_redshift}/kappa'][:]
        list_of_kappa += _split_map(kappa, WIDTH_ORI, width, nsamples_per_side)

    list_of_idx = list(range(len(list_of_kappa)))
    if shuffle:
        random.shuffle(list_of_idx)
    list_of_kappa = [list_of_kappa[i] for i in list_of_idx]
    kappa = np.stack(list_of_kappa)

    return kappa


def get_weights(redshifts):
    """
    Arguments
    ---------
    redshifts (np.array)
        Shape = (ngals)
    
    """
    if np.min(redshifts) < LIST_OF_Z[0] or np.max(redshifts) >= LIST_OF_Z[-1]:
        raise ValueError("Out-of-bound values for argument `redshifts`")

    idxs_sup = np.digitize(redshifts, LIST_OF_Z) # shape = (ngals,)
    idxs_inf = idxs_sup - 1 # shape = (ngals,)

    diff_redshifts = LIST_OF_Z[idxs_sup] - LIST_OF_Z[idxs_inf] # shape = (ngals,)
    weights_sup = 1 - (LIST_OF_Z[idxs_sup] - redshifts) / diff_redshifts # shape = (ngals,)
    weights_inf = 1 - (redshifts - LIST_OF_Z[idxs_inf]) / diff_redshifts # shape = (ngals,)
    # Note that (weights_inf + weights_sup) are equal to one everywhere

    idxs = np.concatenate([idxs_inf, idxs_sup])
    weights = np.concatenate([weights_inf, weights_sup])

    out = np.bincount(idxs, weights=weights, minlength=len(LIST_OF_Z)) # shape = nz
    out /= np.sum(out) # normalize

    return out
