import os
import random
import numpy as np
import h5py

from . import CONFIG_DATA
from . import utils as wlutils

KTNG_DIR = os.path.expanduser(CONFIG_DATA['ktng_dir'])

LIST_OF_Z = np.loadtxt(os.path.join(KTNG_DIR, 'zs.dat'))
FILENAMES_OLD = ['kappa13', 'kappa23', 'kappa30'] # when using the old sample dataset
LIST_OF_Z_OLD = [0.506, 1.034, 1.532] # corresponding redshifts

WIDTH_ORI = 1024 # size of the simulated convergence maps (nb pixels)
WIDTH = 360 # size of the target convergence maps (nb pixels)
SIZE_ORI = 5. # opening angle of the simulated convergence maps (deg)
SIZE = SIZE_ORI * WIDTH / WIDTH_ORI # opening angle of the target convergence maps (deg)
RESOLUTION = SIZE_ORI / WIDTH_ORI * 60. # resolution in arcmin/pixel

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))

class BaseKappaTNG:

    def __init__(
            self, crop_maps=True, size=SIZE, n_samples_per_side=3,
            shuffle=False, ktng_dir=KTNG_DIR, verbose=False, **kwargs
    ):
        self.crop_maps = crop_maps

        width, size = get_npixels_openingangle(size, **kwargs)

        self.width = width
        self.size = size
        self.n_samples_per_side = n_samples_per_side
        self.shuffle = shuffle
        self.ktng_dir = ktng_dir
        self.list_of_idx = None
        self.verbose = verbose


    def get_kappa(self, ninpimgs, start_idx=0):
        """
        Parameters
        ----------
        ninpimgs (int)
            Number of input images to load, before cropping and data augmentation
        start_idx (int, default=0)
            Index of the first image to load
        
        """
        list_of_idx_run = np.arange(start_idx, start_idx + ninpimgs) + 1
        list_of_idx_run = vectorized_zfill(list_of_idx_run)

        list_of_kappa = []
        for idx_run in list_of_idx_run:
            self.print(f"Processing dataset {idx_run}...")
            kappa = self._get_kappa_from_file(idx_run)
            if self.crop_maps:
                list_of_kappa += self._split_map(kappa)
            else:
                list_of_kappa.append(kappa)

        list_of_idx = list(range(len(list_of_kappa)))
        if self.shuffle:
            random.shuffle(list_of_idx)
        list_of_kappa = [list_of_kappa[i] for i in list_of_idx]
        kappa = np.stack(list_of_kappa)
        self.list_of_idx = list_of_idx

        return kappa


    def _get_kappa_from_file(self, idx_run):
        raise NotImplementedError


    def _split_map(self, kappa):

        return wlutils.patchify(
            kappa, self.width, self.n_samples_per_side, inpsize=WIDTH_ORI,
            centermean=True
        )


    def print(self, msg):
        if self.verbose:
            print(msg)


class KappaTNG(BaseKappaTNG):
    """
    Class for loading convergence maps from the kappaTNG dataset:
    https://github.com/0satoken/kappaTNG

    Attributes
    ----------
    weights (list of float, default=None)
        Either one of `weights` and `idx_redshift` must be provided
    idx_redshift (int, default=None)
        Either one of `weights` and `idx_redshift` must be provided
    size (float, default=SIZE)
        Opening angle (deg)
    make_even (bool, default=True)
        Wether to force even-sized convergence maps
    n_samples_per_side (int, default=3)
        Used for cropping input images
    shuffle (bool, default=False)
    ktng_dir (str, default=KTNG_DIR)

    """
    def __init__(
            self, idx_lp, *args, weights=None, idx_redshift=None, **kwargs
    ):
        self.idx_lp = str(idx_lp).zfill(3)
        self.weights = weights
        if idx_redshift is not None:
            self.idx_redshift = f'z{str(idx_redshift + 1).zfill(2)}'
        else:
            self.idx_redshift = None
        super().__init__(*args, **kwargs)


    def _get_kappa_from_file(self, idx_run):

        def _get_kappa_oneredshift(file, idx_redshift):
            return file[os.path.join(idx_redshift, 'kappa')][:]

        fname = os.path.join(
            self.ktng_dir,
            f"LP{self.idx_lp}",
            f"LP{self.idx_lp}_run{idx_run}_maps.hdf5"
        )
        with h5py.File(fname, 'r') as file:
            list_of_idx_redshift = sorted(file.keys())[1:]
            nredshifts = len(list_of_idx_redshift)
            if self.weights is not None:
                if len(self.weights) != nredshifts:
                    raise AttributeError(
                        f"Attribute `weights` must have {nredshifts} elements"
                    )
                kappa = sum([
                    weight * _get_kappa_oneredshift(file, idx_redshift) \
                        for idx_redshift, weight in zip(list_of_idx_redshift, self.weights)
                ])
            elif self.idx_redshift is not None:
                kappa = _get_kappa_oneredshift(file, self.idx_redshift)
            else:
                raise AttributeError(
                    "Either the attribute `weights` or `idx_redshift` must be provided"
                )

        return kappa


class KappaTNGFromSamples(BaseKappaTNG):
    """
    Uses the old sample dataset provided by the authors. Only one redshift at a time.

    Attributes
    ----------
    idx_redshift (int)
    size (float, default=SIZE)
        Opening angle (deg)
    make_even (bool, default=True)
        Wether to force even-sized convergence maps
    n_samples_per_side (int, default=3)
        Used for cropping input images
    shuffle (bool, default=False)    
    ktng_dir (str, default=KTNG_DIR)
    
    """
    def __init__(self, idx_redshift, *args, **kwargs):
        self.bin_file = f"{FILENAMES_OLD[idx_redshift]}.dat"
        super().__init__(*args, **kwargs)


    def _get_kappa_from_file(self, idx_run):

        fname = os.path.join(self.ktng_dir, f"run{idx_run}", self.bin_file)
        with open(fname, 'rb') as f:
            _ = np.fromfile(f, dtype="int32", count=1)
            kappa = np.fromfile(f, dtype="float", count=WIDTH_ORI*WIDTH_ORI)
            _ = np.fromfile(f, dtype="int32", count=1)
        kappa = kappa.reshape((WIDTH_ORI, WIDTH_ORI))

        return kappa


def get_openingangle(imgsize):
    return imgsize * RESOLUTION / 60.


def get_npixels_openingangle(size, make_even=True):

    if not make_even:
        mult = 1
    else:
        mult = 2
    width = mult * int(size / (mult * RESOLUTION) * 60.)

    # Adjust opening angle to match the (integer) number of pixels
    size = get_openingangle(width)

    return width, size


def get_weights(redshifts):
    """
    Arguments
    ---------
    redshifts (np.array)
        List of redshifts, for each measured galaxy. 1D array of shape (ngals,)
    
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
