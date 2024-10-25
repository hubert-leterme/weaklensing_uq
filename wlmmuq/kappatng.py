import os
import random
import numpy as np
import h5py

from . import CONFIG_DATA
from . import utils as wlutils
from . import cosmos as wlcosmos
from . import dataaugm

KTNG_DIR = os.path.expanduser(CONFIG_DATA['ktng_dir'])

LIST_OF_Z = np.loadtxt(os.path.join(KTNG_DIR, 'zs.dat'))
FILENAMES_OLD = ['kappa13', 'kappa23', 'kappa30'] # when using the old sample dataset
LIST_OF_Z_OLD = [0.506, 1.034, 1.532] # corresponding redshifts

WIDTH_ORI = 1024 # size of the simulated convergence maps (nb pixels)
WIDTH = 360 # size of the target convergence maps (nb pixels)
SIZE_ORI = 5. # opening angle of the simulated convergence maps (deg)
OPENINGANGLE = SIZE_ORI * WIDTH / WIDTH_ORI # opening angle of the target convergence maps (deg)
RESOLUTION = SIZE_ORI / WIDTH_ORI * 60. # resolution in arcmin/pixel

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))

class BaseKappaTNG:

    def __init__(
            self, crop_maps=True, openingangle=OPENINGANGLE, n_samples_per_side=3,
            shuffle=False, ktng_dir=KTNG_DIR, verbose=False, **kwargs
    ):
        self.crop_maps = crop_maps

        width, openingangle = get_npixels_openingangle(openingangle, **kwargs)

        self.width = width
        self.openingangle = openingangle
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
    idx_lp (int or str, default=None)
        Index of the learning potential, indicating which folder to look into
        for the HDF5 files containing the dataset ("LPxxx" where "xxx" ranges
        from "001" to "100"). By default, "LP001" will be considered.
    weights (list of float, default=None)
        Either one of `weights` and `idx_redshift` must be provided
    idx_redshift (int, default=None)
        Either one of `weights` and `idx_redshift` must be provided
    openingangle (float, default=OPENINGANGLE)
        Opening angle (deg)
    make_even (bool, default=True)
        Wether to force even-sized convergence maps
    n_samples_per_side (int, default=3)
        Used for cropping input images
    shuffle (bool, default=False)
    ktng_dir (str, default=KTNG_DIR)

    """
    def __init__(
            self, *args, idx_lp=None, weights=None, idx_redshift=None, **kwargs
    ):
        if idx_lp is not None:
            self.idx_lp = str(idx_lp).zfill(3)
        else:
            self.idx_lp = "001"
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
    openingangle (float, default=OPENINGANGLE)
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


def get_npixels_openingangle(openingangle, make_even=True):

    if not make_even:
        mult = 1
    else:
        mult = 2
    width = mult * int(openingangle / (mult * RESOLUTION) * 60.)

    # Adjust opening angle to match the (integer) number of pixels
    openingangle = get_openingangle(width)

    return width, openingangle


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


def filter_by_redshifts(cat_cosmos_bright):
    cat_cosmos_bright = cat_cosmos_bright[
        cat_cosmos_bright['zphot'] >= np.min(LIST_OF_Z)
    ]
    cat_cosmos_bright = cat_cosmos_bright[
        cat_cosmos_bright['zphot'] < np.max(LIST_OF_Z)
    ]
    return cat_cosmos_bright


def get_data_from_cosmos_ktng(cat_cosmos, imgsize):

    openingangle = get_openingangle(imgsize)
    data_cosmos = wlcosmos.get_data_from_cosmos(
        cat_cosmos, openingangle
    )
    extent = data_cosmos['extent']
    shapedisp = data_cosmos["shapedisp"]
    ngal = wlutils.ngal_per_pixel(
        cat_cosmos['Ra'], cat_cosmos['Dec'],
        imgsize, extent
    )
    mask = ngal > 0

    out = {
        'openingangle': openingangle,
        'shapedisp': shapedisp,
        'ngal': ngal,
        'mask': mask
    }
    return out


def create_cropped_dataset(
        hdf5_filepath, idx_lp, ninpimgs, weights_redshift, imgsize, batch_size=None,
        **kwargs
):
    """
    Create a dataset of cropped convergence maps from kappaTNG, with combined redshifts.
    """
    # Create HDF5 file structure
    with h5py.File(hdf5_filepath, 'w') as f:
        f.create_dataset(
            "kappa", shape=(0, imgsize, imgsize), maxshape=(None, imgsize, imgsize),
            dtype='float32'
        ) # Convergence maps
        # f.create_dataset(
        #     "filename_ori", shape=(0,), maxshape=(None,),
        #     dtype=np.dtype('S17')
        # ) # Original data realizations (list of filenames)
        # f.create_dataset(
        #     "top_left_coord", shape=(0, 2), maxshape=(None, 2),
        #     dtype='int'
        # ) # Top-left coordinates

    openingangle = get_openingangle(imgsize)
    ktng = KappaTNG(
        idx_lp=idx_lp, weights=weights_redshift, openingangle=openingangle, **kwargs
    )

    if batch_size is None:
        batch_size = ninpimgs

    end_idx = 0
    while end_idx < ninpimgs:
        beg_idx = end_idx
        end_idx = min(beg_idx + batch_size, ninpimgs)

        # Load $\kappa$-TNG dataset and combine redshifts
        kappa = ktng.get_kappa(end_idx - beg_idx, start_idx=beg_idx)
        imgsize0 = kappa.shape[-1]
        assert kappa.shape[-2] == imgsize0

        # Update the HDF5 file
        with h5py.File(hdf5_filepath, 'r+') as f:
            new_size = f['kappa'].shape[0] + (end_idx - beg_idx)
            f['kappa'].resize((new_size, imgsize, imgsize))
            f['kappa'][-(end_idx - beg_idx):] = kappa


def create_augmented_dataset(
        hdf5_filepath, idx_lp, nimgs, weights_redshift, imgsize, batch_size=50,
        angle_batch_size=36, angle_step=5, niter_per_angle=1, verbose=False
):  
    """
    Create an augmented dataset from kappaTNG by rotating and randomly cropping images.
    """
    # Create HDF5 file structure
    with h5py.File(hdf5_filepath, 'w') as f:
        f.create_dataset(
            "kappa", shape=(0, imgsize, imgsize), maxshape=(None, imgsize, imgsize),
            dtype='float32'
        ) # Convergence maps
        f.create_dataset(
            "filename_ori", shape=(0,), maxshape=(None,),
            dtype=np.dtype('S17')
        ) # Original data realizations (list of filenames)
        f.create_dataset(
            "angle", shape=(0,), maxshape=(None,),
            dtype='float32'
        ) # Rotation angles
        f.create_dataset(
            "top_left_coord", shape=(0, 2), maxshape=(None, 2),
            dtype='int'
        ) # Top-left coordinates

    ktng = KappaTNG(idx_lp=idx_lp, weights=weights_redshift, crop_maps=False)

    end_idx = 0
    while end_idx < nimgs:
        beg_idx = end_idx
        end_idx = min(beg_idx + batch_size, nimgs)
        if verbose:
            print(f"Processing images {beg_idx} to {end_idx}...")

        # Load $\kappa$-TNG dataset and combine redshifts
        kappa = ktng.get_kappa(end_idx - beg_idx, start_idx=beg_idx)
        imgsize0 = kappa.shape[-1]
        assert kappa.shape[-2] == imgsize0

        end_angle = 0
        while end_angle < 360:
            beg_angle = end_angle
            end_angle = min(beg_angle + angle_batch_size * angle_step, 360)

            list_of_kappa_rot = []
            list_of_idx_rows = []
            list_of_idx_cols = []
            for angle in range(beg_angle, end_angle, angle_step):
                if verbose:
                    print(f"\tAngle = {angle}...")
                kappa_rot, rows, cols = dataaugm.rotate_and_crop(
                    kappa, angle, imgsize, niter=niter_per_angle
                )
                list_of_kappa_rot.append(kappa_rot)
                list_of_idx_rows.append(rows)
                list_of_idx_cols.append(cols)

            # Shape = (angle_batch_size * nimgs, imgsize, imgsize)
            kappa_rot = np.concatenate(list_of_kappa_rot, axis=0)
            rows = np.concatenate(list_of_idx_rows, axis=0)
            cols = np.concatenate(list_of_idx_cols, axis=0)
            angle_batch_size_adjusted = -(beg_angle - end_angle) // angle_step
            nimgs_batch = angle_batch_size_adjusted * niter_per_angle * (end_idx - beg_idx)

            # Update the HDF5 file
            with h5py.File(hdf5_filepath, 'r+') as f:
                new_size = f['kappa'].shape[0] + nimgs_batch

                f['kappa'].resize((new_size, imgsize, imgsize))
                f['kappa'][-nimgs_batch:] = kappa_rot

                f['filename_ori'].resize((new_size,))
                f['filename_ori'][-nimgs_batch:] = angle_batch_size_adjusted * niter_per_angle * [
                    f"LP001_run{idx}_maps.hdf5" for idx in vectorized_zfill(
                        np.arange(beg_idx, end_idx)
                    )
                ]

                f['angle'].resize((new_size,))
                f['angle'][-nimgs_batch:] = np.repeat(
                    np.arange(beg_angle, end_angle, angle_step),
                    niter_per_angle * (end_idx - beg_idx)
                )

                f['top_left_coord'].resize((new_size, 2))
                f['top_left_coord'][-nimgs_batch:, 0] = rows
                f['top_left_coord'][-nimgs_batch:, 1] = cols
