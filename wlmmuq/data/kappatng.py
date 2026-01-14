import os
import random
import numpy as np
import h5py
import tqdm
import typing
import torch
import astropy.table as aptable

from . import cosmos, dataaugm, base_dataset
from .. import utils
from .. import KTNG_DIR

try:
    LIST_OF_Z = np.loadtxt(os.path.join(KTNG_DIR, 'zs.dat'))
except (TypeError, FileNotFoundError):
    LIST_OF_Z = None
MAX_Z = 2.6
FILENAMES_OLD = ['kappa13', 'kappa23', 'kappa30'] # when using the old sample dataset
LIST_OF_Z_OLD = [0.506, 1.034, 1.532] # corresponding redshifts

WIDTH_ORI = 1024 # size of the simulated convergence maps (nb pixels)
SIZE_ORI = 5. # opening angle of the simulated convergence maps (deg)
N_SAMPLES_PER_SIDE = 3
RESOLUTION = SIZE_ORI / WIDTH_ORI * 60. # resolution in arcmin/pixel
OPENINGANGLE = 1.875 # opening angle of the target convergence maps (deg)

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))

class BaseKappaTNG:

    def __init__(
            self, crop_maps=True, openingangle=OPENINGANGLE, n_samples_per_side=N_SAMPLES_PER_SIDE,
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


    def get_kappa(self, ninpimgs, start_idx=0, centermean=True):
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
            kappa = self._get_kappa_from_file(idx_run, centermean=centermean)
            list_of_kappa.append(kappa)

        list_of_idx = list(range(len(list_of_kappa)))
        if self.shuffle:
            random.shuffle(list_of_idx)
        list_of_kappa = [list_of_kappa[i] for i in list_of_idx]
        kappa = np.stack(list_of_kappa)
        self.list_of_idx = list_of_idx

        if self.crop_maps:
            kappa = self._split_map(kappa, centermean=centermean)

        return kappa


    def _get_kappa_from_file(self, idx_run, centermean=True):
        raise NotImplementedError


    def _split_map(self, kappa, centermean=True):

        nimgs, _, nx, ny = kappa.shape

        step_x = (nx - self.width) // (self.n_samples_per_side - 1)
        step_y = (ny - self.width) // (self.n_samples_per_side - 1)

        rows = np.arange(0, nx, step=step_x)[:self.n_samples_per_side]
        cols = np.arange(0, ny, step=step_y)[:self.n_samples_per_side]

        rows = np.repeat(rows, self.n_samples_per_side)
        cols = np.tile(cols, self.n_samples_per_side)

        rows = np.tile(rows, nimgs)
        cols = np.tile(cols, nimgs)

        return dataaugm.get_patches(
            kappa, rows, cols, self.width,
            ncrops_per_imgs=self.n_samples_per_side**2, centermean=centermean
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
            self, *args, idx_lp: int | None = None,
            weights: np.ndarray | None = None,
            zbins: list[float] | None = None,
            idx_redshift: int | None = None, **kwargs
    ):
        if idx_lp is not None:
            self.idx_lp = str(idx_lp).zfill(3)
        else:
            self.idx_lp = "001"
        self.weights = weights
        self.zbins = zbins
        if idx_redshift is not None:
            self.idx_redshift = f'z{str(idx_redshift + 1).zfill(2)}'
        else:
            self.idx_redshift = None
        super().__init__(*args, **kwargs)


    def _get_kappa_from_file(
            self, idx_run: str, centermean: bool = True
    ) -> np.ndarray:

        def _get_kappa_oneredshift(
                file: h5py.File, idx_redshift: str
        ) -> np.ndarray:
            path = os.path.join(idx_redshift, 'kappa')
            obj = file[path]
            assert isinstance(obj, h5py.Dataset)
            kappa = obj[:]
            if centermean:
                kappa = kappa - np.mean(kappa)
            return kappa

        fname = os.path.join(
            self.ktng_dir,
            f"LP{self.idx_lp}",
            f"LP{self.idx_lp}_run{idx_run}_maps.hdf5"
        )
        with h5py.File(fname, 'r', swmr=True) as file:

            if self.weights is not None:
                list_of_idx_redshift = sorted(file.keys())[1:]
                nredshifts = len(list_of_idx_redshift)
                if len(self.weights) != nredshifts:
                    raise AttributeError(
                        f"Attribute `weights` must have {nredshifts} elements"
                    )
                
                list_of_weights = _get_list_per_zbin(
                    self.weights, self.zbins
                )
                list_of_idx_redshift_per_zbin = _get_list_per_zbin(
                    list_of_idx_redshift, self.zbins
                )
                list_of_kappa = [
                    np.stack([
                        _get_kappa_oneredshift(file, i) for i in l
                    ], axis=-1) for l in list_of_idx_redshift_per_zbin # Shape = (nx, ny, nz)
                ]
                list_of_kappa = [
                    np.sum(w * kappa, axis=-1) \
                        for w, kappa in zip(list_of_weights, list_of_kappa) # Shape = (nx, ny)
                ]
                kappa = np.stack(list_of_kappa) # Shape = (nbins, nx, ny)

            elif self.idx_redshift is not None:
                kappa = _get_kappa_oneredshift(
                    file, self.idx_redshift
                )[np.newaxis, ...] # Shape = (1, nx, ny)

            else:
                raise AttributeError(
                    "Either the attribute `weights` or `idx_redshift` must be provided"
                )

        return kappa
    

@typing.overload
def _get_list_per_zbin[T](
        inp: list[T],
        zbins: list[float] | None = None
) -> list[list[T]]: ...

@typing.overload
def _get_list_per_zbin(
        inp: np.ndarray,
        zbins: list[float] | None = None
) -> list[np.ndarray]: ...
    
def _get_list_per_zbin(inp, zbins=None):

    assert LIST_OF_Z is not None
    out = [[]]
    j = 0
    for i, z in enumerate(LIST_OF_Z):
        try:
            assert zbins is not None
            new_zbin = z >= zbins[j]
        except (AssertionError, IndexError):
            new_zbin = False
        if new_zbin:
            out.append([])
            j += 1
        assert isinstance(out[j], list)
        out[j].append(inp[i])

    if isinstance(inp, np.ndarray):
        out = [np.array(l) for l in out]

    return out


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


    def _get_kappa_from_file(self, idx_run, centermean=True):

        fname = os.path.join(self.ktng_dir, f"run{idx_run}", self.bin_file)
        with open(fname, 'rb') as f:
            _ = np.fromfile(f, dtype="int32", count=1)
            kappa = np.fromfile(f, dtype="float", count=WIDTH_ORI*WIDTH_ORI)
            _ = np.fromfile(f, dtype="int32", count=1)
        kappa = kappa.reshape((1, WIDTH_ORI, WIDTH_ORI))
        if centermean:
            kappa = kappa - np.mean(kappa)

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
    idxs_sup = np.digitize(redshifts, LIST_OF_Z) # shape = (ngals,)
    idxs_inf = idxs_sup - 1 # shape = (ngals,)

    redshifts_sup = np.array([
        LIST_OF_Z[idx] if idx < len(LIST_OF_Z) else np.nan for idx in idxs_sup
    ])
    redshifts_inf = np.array([
        LIST_OF_Z[idx] if idx >= 0 else np.nan for idx in idxs_inf
    ])

    diff_redshifts = redshifts_sup - redshifts_inf # shape = (ngals,)
    weights_sup = 1 - (redshifts_sup - redshifts) / diff_redshifts # shape = (ngals,)
    weights_inf = 1 - (redshifts - redshifts_inf) / diff_redshifts # shape = (ngals,)
    # Note that `weights_inf + weights_sup` are equal to one everywhere,
    # except when `redshifts` is below `LIST_OF_Z[0]` or above `LIST_OF_Z[-1]`, in which
    # case the value of `weights_inf` and `weights_sup` is nan.

    idxs = np.concatenate([idxs_inf, idxs_sup])
    weights = np.concatenate([weights_inf, weights_sup])

    # Galaxies with redshift below LIST_OF_Z[0] contribute entirely to the first bin
    # Galaxies with redshift above LIST_OF_Z[-1] contribute entirely to the last bin
    weights = weights[(idxs >= 0) & (idxs < len(LIST_OF_Z))]
    idxs = idxs[(idxs >= 0) & (idxs < len(LIST_OF_Z))]
    weights[np.isnan(weights)] = 1.

    out = np.bincount(idxs, weights=weights, minlength=len(LIST_OF_Z)) # shape = nz
    out /= np.sum(out) # normalize

    return out


def get_data_from_cosmos_ktng(
        cat_cosmos: aptable.Table, imgsize: int,
        zbins: list[float] | None = None
):
    openingangle = get_openingangle(imgsize)
    data_cosmos = cosmos.get_data_from_cosmos(
        cat_cosmos, openingangle
    )
    ra_cosmos_median = data_cosmos['ra_cosmos_median']
    dec_cosmos_median = data_cosmos['dec_cosmos_median']
    extent = data_cosmos['extent']
    shapedisp = data_cosmos["shapedisp"]
    if zbins is None:
        ngal = utils.ngal_per_pixel(
            cat_cosmos['Ra'], cat_cosmos['Dec'],
            imgsize, extent
        ) # Shape = (imgsize, imgsize)
    else:
        zbins = sorted(zbins + [0., MAX_Z])
        ngal_per_zbin = []
        for z_inf, z_sup in zip(zbins[:-1], zbins[1:]):
            cat_cosmos_sliced = cat_cosmos[
                (cat_cosmos["zphot"] >= z_inf) & (cat_cosmos["zphot"] < z_sup)
            ]
            ngal_per_zbin.append(utils.ngal_per_pixel(
                cat_cosmos_sliced['Ra'], cat_cosmos_sliced['Dec'],
                imgsize, extent
            ))
        
        ngal = np.stack(ngal_per_zbin) # Shape = (nbins, imgsize, imgsize)

    mask = ngal > 0

    ngal = torch.tensor(ngal, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=bool)

    out = {
        'ra_cosmos_median': ra_cosmos_median,
        'dec_cosmos_median': dec_cosmos_median,
        'extent': extent,
        'openingangle': openingangle,
        'shapedisp': shapedisp,
        'ngal': ngal,
        'mask': mask
    }
    return out


def create_cropped_dataset(
        hdf5_filepath, idx_lp, ninpimgs, weights_redshift, imgsize,
        zbins=None, batch_size=None, verbose=False, **kwargs
):
    """
    Create a dataset of cropped convergence maps from kappaTNG, with combined redshifts.
    """
    nbins = base_dataset.get_nbins(zbins)

    # Create HDF5 file structure
    with h5py.File(hdf5_filepath, 'w') as f:

        # Metadata
        f.attrs["idx_lp"] = idx_lp
        f.attrs["weights_redshift"] = weights_redshift
        if zbins is not None:
            f.attrs["zbins"] = zbins

        f.create_dataset(
            "kappa", shape=(0, nbins, imgsize, imgsize), maxshape=(None, nbins, imgsize, imgsize),
            dtype='float32'
        ) # Convergence maps
        # f.create_dataset(
        #     "filename_ori", shape=(0,), maxshape=(None,),
        #     dtype=np.dtype('S17') # TODO: use regular strings instead
        # ) # Original data realizations (list of filenames)
        # f.create_dataset(
        #     "top_left_coord", shape=(0, 2), maxshape=(None, 2),
        #     dtype='int'
        # ) # Top-left coordinates

    openingangle = get_openingangle(imgsize)
    ktng = KappaTNG(
        idx_lp=idx_lp, weights=weights_redshift, openingangle=openingangle,
        zbins=zbins, **kwargs
    )

    if batch_size is None:
        batch_size = ninpimgs

    pbar = tqdm.tqdm(
        range(-(-ninpimgs // batch_size)),
        disable=not verbose,
    )
    for i in pbar:
        beg_idx = i * batch_size
        end_idx = min(beg_idx + batch_size, ninpimgs)
        pbar.set_description(f"Images {beg_idx + 1}-{end_idx}/{ninpimgs}")

        # Load $\kappa$-TNG dataset and combine redshifts
        kappa = ktng.get_kappa(end_idx - beg_idx, start_idx=beg_idx)
        nimgs = kappa.shape[0]

        # Update the HDF5 file
        with h5py.File(hdf5_filepath, 'r+') as f:
            new_size = f['kappa'].shape[0] + nimgs
            f['kappa'].resize((new_size, nbins, imgsize, imgsize))
            f['kappa'][-nimgs:] = kappa


def create_augmented_dataset(
    hdf5_filepath, idx_lp, nimgs, weights_redshift, imgsize,
    zbins=None, batch_size=50,
    angle_batch_size=1, angle_step=5, niter_per_angle=1,
    resume=False, verbose=False
):
    """
    Create or resume an augmented dataset from kappaTNG
    by rotating and randomly cropping images.
    """

    nbins = base_dataset.get_nbins(zbins)

    # --------------------------------------------------
    # Create or open HDF5 file
    # --------------------------------------------------
    if not resume:
        if os.path.exists(hdf5_filepath):
            raise FileExistsError
        with h5py.File(hdf5_filepath, 'w') as f:

            # Metadata
            f.attrs["idx_lp"] = idx_lp
            f.attrs["weights_redshift"] = weights_redshift
            f.attrs["angle_step"] = angle_step
            f.attrs["niter_per_angle"] = niter_per_angle
            if zbins is not None:
                f.attrs["zbins"] = zbins

            f.create_dataset(
                "kappa", shape=(0, nbins, imgsize, imgsize),
                maxshape=(None, nbins, imgsize, imgsize),
                dtype='float32'
            )
            f.create_dataset(
                "filename_ori", shape=(0,),
                maxshape=(None,), dtype=np.dtype('S17')
            )
            f.create_dataset(
                "angle", shape=(0,),
                maxshape=(None,), dtype='float32'
            )
            f.create_dataset(
                "top_left_coord", shape=(0, 2),
                maxshape=(None, 2), dtype='int'
            )

            # Progress metadata
            prog = f.create_group("progress")
            prog.attrs["last_img_idx"] = 0
            prog.attrs["last_angle_block"] = 0
            prog.attrs["angle_batch_size"] = angle_batch_size

    # --------------------------------------------------
    # Read resume state
    # --------------------------------------------------
    with h5py.File(hdf5_filepath, 'r') as f:
        last_img_idx = f["progress"].attrs["last_img_idx"]
        last_angle_block = f["progress"].attrs["last_angle_block"]

    ktng = KappaTNG(
        idx_lp=idx_lp, weights=weights_redshift,
        crop_maps=False, zbins=zbins
    )

    end_idx = last_img_idx
    while end_idx < nimgs:
        beg_idx = end_idx
        end_idx = min(beg_idx + batch_size, nimgs)

        kappa = ktng.get_kappa(end_idx - beg_idx, start_idx=beg_idx)

        pbar = tqdm.tqdm(
            range(int(np.ceil(360 / (angle_batch_size * angle_step)))),
            disable=not verbose,
        )

        for i in pbar:
            # Resume logic
            if beg_idx == last_img_idx and i < last_angle_block:
                continue

            beg_angle = i * angle_batch_size * angle_step
            end_angle = min(beg_angle + angle_batch_size * angle_step, 360)
            angles = np.arange(beg_angle, end_angle, angle_step)
            nimgs_batch = len(angles) * niter_per_angle * (end_idx - beg_idx)

            list_of_kappa_rot = []
            list_of_idx_rows = []
            list_of_idx_cols = []
            for angle in angles:
                pbar.set_description(f"Images {beg_idx + 1}-{end_idx}/{nimgs}, Angle = {angle:.1f}")
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

            # --------------------------------------------------
            # Write data + update progress atomically
            # --------------------------------------------------
            with h5py.File(hdf5_filepath, 'r+') as f:
                new_size = f['kappa'].shape[0] + nimgs_batch

                f['kappa'].resize((new_size, nbins, imgsize, imgsize))
                f['kappa'][-nimgs_batch:] = kappa_rot

                f['filename_ori'].resize((new_size,))
                f['filename_ori'][-nimgs_batch:] = len(angles) * niter_per_angle * [
                    f"LP001_run{idx}_maps.hdf5" for idx in vectorized_zfill(
                        np.arange(beg_idx, end_idx) + 1
                    )
                ] # TODO: improve this by loading the original filenames from the KTNG dataset
                # In particular, "LP001" should be adaptive.

                f['angle'].resize((new_size,))
                f['angle'][-nimgs_batch:] = np.repeat(
                    angles, niter_per_angle * (end_idx - beg_idx)
                )

                f['top_left_coord'].resize((new_size, 2))
                f['top_left_coord'][-nimgs_batch:, 0] = rows
                f['top_left_coord'][-nimgs_batch:, 1] = cols

                # Update progress
                f["progress"].attrs["last_angle_block"] = i + 1

        # Reset angle block after full batch
        with h5py.File(hdf5_filepath, 'r+') as f:
            f["progress"].attrs["last_img_idx"] = end_idx
            f["progress"].attrs["last_angle_block"] = 0
