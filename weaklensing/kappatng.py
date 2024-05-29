import os
import random
import numpy as np
import h5py

from . import CONFIG_DATA

KTNG_DIR = os.path.expanduser(CONFIG_DATA['ktng_dir'])

LIST_OF_Z = np.loadtxt(os.path.join(KTNG_DIR, 'zs.dat'))
FILENAMES = ['kappa13', 'kappa23', 'kappa30'] # When using the old sample dataset

WIDTH_ORI = 1024 # size of the simulated convergence maps (nb pixels)
WIDTH = 360 # size of the target convergence maps (nb pixels)
SIZE_ORI = 5. # opening angle of the simulated convergence maps (deg)
SIZE = SIZE_ORI * WIDTH / WIDTH_ORI # opening angle of the target convergence maps (deg)
RESOLUTION = SIZE_ORI / WIDTH_ORI * 60. # resolution in arcmin/pixel

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))

class BaseKappaTNG:

    def __init__(
            self, size=SIZE, make_even=True,
            n_samples_per_side=3, shuffle=False, ktng_dir=KTNG_DIR
    ):
        # Get number of pixels
        if not make_even:
            mult = 1
        else:
            mult = 2
        width = mult * int(size / (mult * RESOLUTION) * 60.)

        # Adjust opening angle to match the (integer) number of pixels
        size = width * RESOLUTION / 60.

        self.width = width
        self.size = size
        self.n_samples_per_side = n_samples_per_side
        self.shuffle = shuffle
        self.ktng_dir = ktng_dir


    def get_kappa(self, ninpimgs, start_idx=0):
        """
        Parameters
        ----------
        ninpimgs (int)
            Number of input images to load, before cropping and data augmentation
        start_idx (int, default=0)
            Index of the first image to load
        
        """
        list_of_idx_dataset = np.arange(start_idx, start_idx + ninpimgs) + 1
        list_of_idx_dataset = vectorized_zfill(list_of_idx_dataset)

        list_of_kappa = []
        for idx_dataset in list_of_idx_dataset:
            kappa = self._get_kappa_from_file(idx_dataset)
            list_of_kappa += self._split_map(kappa)

        list_of_idx = list(range(len(list_of_kappa)))
        if self.shuffle:
            random.shuffle(list_of_idx)
        list_of_kappa = [list_of_kappa[i] for i in list_of_idx]
        kappa = np.stack(list_of_kappa)

        return kappa


    def _get_kappa_from_file(self, idx_dataset):
        raise NotImplementedError


    def _split_map(self, kappa):

        step = (WIDTH_ORI - self.width) // (self.n_samples_per_side - 1)
        out = []
        beg_i = 0
        for _ in range(self.n_samples_per_side):
            beg_j = 0
            for _ in range(self.n_samples_per_side):
                subkappa = kappa[beg_i:beg_i + self.width, beg_j:beg_j + self.width]
                subkappa = subkappa - np.mean(subkappa) # Center-normalize the convergence map
                out.append(subkappa)
                beg_j += step
            beg_i += step

        return out


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
    def __init__(self, *args, weights=None, idx_redshift=None, **kwargs):
        self.weights = weights
        self.idx_redshift = idx_redshift
        super().__init__(*args, **kwargs)


    def _get_kappa_from_file(self, idx_dataset):

        def _get_kappa_oneredshift(file, idx_redshift):
            return file[f'{idx_redshift}/kappa'][:]

        fname = os.path.join(self.ktng_dir, f"LP001_run{idx_dataset}_maps.hdf5")
        with h5py.File(fname, 'r') as file:
            list_of_idx_redshift = sorted(file.keys())[1:]
            nredshifts = len(list_of_idx_redshift)
            if self.weights is not None:
                if len(self.weights) != nredshifts:
                    raise AttributeError(
                        f"Attribute `weights` must have {nredshifts} elements"
                    )
                kappa = np.zeros((WIDTH_ORI, WIDTH_ORI))
                for idx_redshift, weight in zip(list_of_idx_redshift, self.weights):
                    kappa += weight * _get_kappa_oneredshift(file, idx_redshift)
            elif self.idx_redshift is not None:
                kappa = _get_kappa_oneredshift(file, idx_redshift)
            else:
                raise AttributeError(
                    "Either attributes `weights` or `idx_redshift` must be provided"
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
        self.bin_file = f"{FILENAMES[idx_redshift]}.dat"
        super().__init__(*args, **kwargs)


    def _get_kappa_from_file(self, idx_dataset):

        fname = os.path.join(KTNG_DIR, f"run{idx_dataset}", self.bin_file)
        with open(fname, 'rb') as f:
            _ = np.fromfile(f, dtype="int32", count=1)
            kappa = np.fromfile(f, dtype="float", count=WIDTH_ORI*WIDTH_ORI)
            _ = np.fromfile(f, dtype="int32", count=1)
        kappa = kappa.reshape((WIDTH_ORI, WIDTH_ORI))

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
