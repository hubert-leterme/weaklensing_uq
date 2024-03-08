import os
import yaml
import random
import numpy as np

with open('../config.yml', 'r') as file:
    CONFIG_DATA = yaml.safe_load(file)

KTNG_DIR = os.path.expanduser(CONFIG_DATA['ktng_dir'])

#REDSHIFTS = [0.506, 1.034, 1.532] # red shifts for the simulated convergence maps
FILENAMES = ['kappa13', 'kappa23', 'kappa30'] # corresponding filenames

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
        index_redshift, ninpimgs, start_idx=0, width=WIDTH, nsamples_per_side=3, shuffle=True
):
    """
    Parameters
    ----------
    index_redshift (int)
    ninpimgs (int)
        Number of input images to load, before cropping and data augmentation.
    start_idx (int, default=0)
        Index of the first image to load.
    width (int, default=360)
        Size of the target convergence maps (nb pixels)
    nsamples_per_side (int, default=3)
        Used for cropping input images
    shuffle (bool, default=True)
    
    """
    bin_file = f"{FILENAMES[index_redshift]}.dat"

    list_of_idx_dataset = np.arange(start_idx, start_idx+ninpimgs) + 1
    list_of_idx_dataset = vectorized_zfill(list_of_idx_dataset)

    list_of_kappa = []
    for idx in list_of_idx_dataset:
        fname = os.path.join(KTNG_DIR, f"run{idx}", bin_file)
        with open(fname, 'rb') as f:
            _ = np.fromfile(f, dtype="int32", count=1)
            kappa = np.fromfile(f, dtype="float", count=WIDTH_ORI*WIDTH_ORI)
            _ = np.fromfile(f, dtype="int32", count=1)
        kappa = kappa.reshape((WIDTH_ORI, WIDTH_ORI))
        list_of_kappa += _split_map(kappa, WIDTH_ORI, width, nsamples_per_side)

    list_of_idx = list(range(len(list_of_kappa)))
    if shuffle:
        random.shuffle(list_of_idx)
    list_of_kappa = [list_of_kappa[i] for i in list_of_idx]
    kappa = np.stack(list_of_kappa)

    return kappa
