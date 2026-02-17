__level__ = 2

import os
import random
import warnings
from dataclasses import dataclass
import numpy as np
import h5py
import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
import astropy.table as aptable

from . import cosmos, dataaugm, base_dataset
from .. import lenspack, utils
from ..config import KTNG_DIR

MAX_Z = 2.6
try:
    Z = np.loadtxt(os.path.join(KTNG_DIR, 'zs.dat'))
except (TypeError, FileNotFoundError):
    Z = None
FILENAMES_OLD = ['kappa13', 'kappa23', 'kappa30'] # when using the old sample dataset
LIST_OF_Z_OLD = [0.506, 1.034, 1.532] # corresponding redshifts

WIDTH_ORI = 1024 # size of the simulated convergence maps (nb pixels)
SIZE_ORI = 5. # opening angle of the simulated convergence maps (deg)
N_SAMPLES_PER_SIDE = 3
RESOLUTION = SIZE_ORI / WIDTH_ORI * 60. # resolution in arcmin/pixel
OPENINGANGLE = 1.875 # opening angle of the target convergence maps (deg)

ANGLE_BATCH_SIZE = 1

# Cosmological parameters
C = 2.998e5
CURVATURE = 0.
H0 = 67.74
OMEGA_M = 0.3089
OMEGA_LAMBDA = 1 - (OMEGA_M + CURVATURE)

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
    weights_redshifts (list of float, default=None)
        Either one of `weights_redshifts` and `zidx` must be provided
    zidx (int, default=None)
        Either one of `weights_redshifts` and `zidx` must be provided
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
            z: np.ndarray | None = Z,
            z_sup: float = MAX_Z, c: float = C, h0: float = H0,
            omega_m: float = OMEGA_M, omega_lambda: float = OMEGA_LAMBDA, 
            weights_redshifts: np.ndarray | None = None,
            zbins: list[float] | None = None,
            zidx: int | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if idx_lp is not None:
            self.idx_lp = str(idx_lp).zfill(3)
        else:
            self.idx_lp = "001"
        self.zbins = zbins
        first_idx_run = "001"
        fname_first_idx_run = self._get_fname(first_idx_run)
        if weights_redshifts is not None:
            assert z is not None
            cdist = utils.get_cdist(
                z=z, z_sup=z_sup, c=c, h0=h0,
                omega_m=omega_m, omega_lambda=omega_lambda
            )
            msg = (
                "Argument `{}` must have {} elements, "
                "but {} were provided."
            )
            self.z = z
            self.cdist = cdist
            self.weights_redshifts = weights_redshifts

            self.list_of_z = utils.get_list_per_zbin(
                z, z, zbins
            )
            self.list_of_cdist = utils.get_list_per_zbin(
                cdist, z, zbins
            )
            self.list_of_weights_redshifts = utils.get_list_per_zbin(
                weights_redshifts, z, zbins
            )
            list_of_weights_mseloss = [
                np.sum(cdist0 * w0) for cdist0, w0 in zip(
                    self.list_of_cdist, self.list_of_weights_redshifts
                )
            ]
            self.weights_mseloss = np.array(list_of_weights_mseloss)

            try:
                with h5py.File(fname_first_idx_run, 'r', swmr=True) as file:
                    idx_redshifts = np.array(sorted(file.keys()))[1:]

            except FileNotFoundError:
                warnings.warn(f"File '{fname_first_idx_run}' not found")
                self.idx_redshifts = None
                self.list_of_idx_redshifts = None

            else:
                nredshifts = len(idx_redshifts)

                assert len(z) == nredshifts, \
                    msg.format("z", nredshifts, len(z))
                assert len(cdist) == nredshifts, \
                    msg.format("cdist", nredshifts, len(cdist))
                assert len(weights_redshifts) == nredshifts, \
                    msg.format("weights_redshifts", nredshifts, len(weights_redshifts))
                
                self.idx_redshifts = idx_redshifts
                self.list_of_idx_redshifts = utils.get_list_per_zbin(
                    idx_redshifts, z, zbins
                )

        else:
            self.z = None
            self.cdist = None
            self.weights_redshifts = None
            self.idx_redshifts = None
            
            self.list_of_z = None
            self.list_of_cdist = None
            self.list_of_weights_redshifts = None
            self.list_of_idx_redshifts = None

            self.weights_mseloss = None

        if zidx is not None:
            self.zidx = f'z{str(zidx + 1).zfill(2)}'
        else:
            self.zidx = None


    def _get_kappa_from_file(
            self, idx_run: str, centermean: bool = True
    ) -> np.ndarray:

        def _get_kappa_oneredshift(
                file: h5py.File, zidx: str
        ) -> np.ndarray:
            path = os.path.join(zidx, 'kappa')
            obj = file[path]
            assert isinstance(obj, h5py.Dataset)
            kappa = obj[:]
            if centermean:
                kappa = kappa - np.mean(kappa)
            return kappa

        fname = self._get_fname(idx_run)
        with h5py.File(fname, 'r', swmr=True) as file:

            if self.list_of_weights_redshifts is not None:
                assert self.list_of_idx_redshifts is not None

                list_of_kappa = [
                    np.stack([
                        _get_kappa_oneredshift(file, i) for i in l
                    ], axis=-1) for l in self.list_of_idx_redshifts # Shape = (nx, ny, nz)
                ]
                list_of_kappa = [
                    np.sum(w * kappa, axis=-1) \
                        for w, kappa in zip(self.list_of_weights_redshifts, list_of_kappa) # Shape = (nx, ny)
                ]
                kappa = np.stack(list_of_kappa) # Shape = (nbins, nx, ny)

            elif self.zidx is not None:
                kappa = _get_kappa_oneredshift(
                    file, self.zidx
                )[np.newaxis, ...] # Shape = (1, nx, ny)

            else:
                raise AttributeError(
                    "Either the attribute `weights_redshifts` or `zidx` must be provided"
                )

        return kappa


    def _get_fname(self, idx_run: str):
        return os.path.join(
            self.ktng_dir,
            f"LP{self.idx_lp}",
            f"LP{self.idx_lp}_run{idx_run}_maps.hdf5"
        )

class KappaTNGFromSamples(BaseKappaTNG):
    """
    Uses the old sample dataset provided by the authors. Only one redshift at a time.

    Attributes
    ----------
    zidx (int)
    openingangle (float, default=OPENINGANGLE)
        Opening angle (deg)
    make_even (bool, default=True)
        Wether to force even-sized convergence maps
    n_samples_per_side (int, default=3)
        Used for cropping input images
    shuffle (bool, default=False)    
    ktng_dir (str, default=KTNG_DIR)
    
    """
    def __init__(self, zidx, *args, **kwargs):
        self.bin_file = f"{FILENAMES_OLD[zidx]}.dat"
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


def get_weights_redshifts(
        redshifts: np.ndarray,
        z: np.ndarray | None = Z
) -> np.ndarray:
    """
    Arguments
    ---------
    redshifts: np.ndarray, shape = (ngals,)
        List of redshifts, for each measured galaxy
    z: np.ndarray, shape = (nplanes,), optional
        List of redshift planes. Default values are the one provided by
        Osato et al., MNRAS, vol. 502, no. 4, pp. 5593–5602, 2021.

    Returns
    -------
    out: np.ndarray, shape = (nplanes,)

    """
    assert z is not None
    idxs_sup = np.digitize(redshifts, z) # shape = (ngals,)
    idxs_inf = idxs_sup - 1 # shape = (ngals,)

    redshifts_sup = np.array([
        z[idx] if idx < len(z) else np.nan for idx in idxs_sup
    ])
    redshifts_inf = np.array([
        z[idx] if idx >= 0 else np.nan for idx in idxs_inf
    ])

    diff_redshifts = redshifts_sup - redshifts_inf # shape = (ngals,)
    weights_sup = 1 - (redshifts_sup - redshifts) / diff_redshifts # shape = (ngals,)
    weights_inf = 1 - (redshifts - redshifts_inf) / diff_redshifts # shape = (ngals,)
    # Note that `weights_inf + weights_sup` are equal to one everywhere,
    # except when `redshifts` is below `z[0]` or above `z[-1]`, in which
    # case the value of `weights_inf` and `weights_sup` is nan.

    idxs = np.concatenate([idxs_inf, idxs_sup])
    weights_redshifts = np.concatenate([weights_inf, weights_sup])

    # Galaxies with redshift below z[0] contribute entirely to the first bin
    # Galaxies with redshift above z[-1] contribute entirely to the last bin
    weights_redshifts = weights_redshifts[(idxs >= 0) & (idxs < len(z))]
    idxs = idxs[(idxs >= 0) & (idxs < len(z))]
    weights_redshifts[np.isnan(weights_redshifts)] = 1.

    out = np.bincount(idxs, weights=weights_redshifts, minlength=len(z)) # shape = nz
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
    ra_cosmos_median = data_cosmos["ra_cosmos_median"]
    dec_cosmos_median = data_cosmos["dec_cosmos_median"]
    extent = data_cosmos["extent"]
    shapedisp = data_cosmos["shapedisp"]

    boundaries_zbins = [0., MAX_Z]
    if zbins is not None:
        boundaries_zbins = sorted(zbins + boundaries_zbins)

    ngal = []
    for z_inf, z_sup in zip(boundaries_zbins[:-1], boundaries_zbins[1:]):
        cat_cosmos_sliced = cat_cosmos[
            (cat_cosmos["zphot"] >= z_inf) & (cat_cosmos["zphot"] < z_sup)
        ]
        ra_sliced = np.array(cat_cosmos_sliced["Ra"]) # Shape = (ngal_zbin,)
        dec_sliced = np.array(cat_cosmos_sliced["Dec"]) # Shape = (ngal_zbin,)

        ngal.append(lenspack.bin2d(
            ra_sliced, dec_sliced, npix=imgsize, extent=extent,
        ))

    ngal = [
        torch.tensor(n, dtype=torch.float32) for n in ngal
    ]
    ngal = torch.stack(ngal) # Shape = (nbins, nx, ny)

    out = {
        'ra_cosmos_median': ra_cosmos_median,
        'dec_cosmos_median': dec_cosmos_median,
        'extent': extent,
        'openingangle': openingangle,
        'shapedisp': shapedisp,
        'ngal': ngal,
    }
    return out


def create_cropped_dataset(
        hdf5_filepath, idx_lp, ninpimgs, weights_redshifts, imgsize,
        zbins=None, batch_size=None,
        update_metadata_only=False, verbose=False, **kwargs
):
    """
    Create a dataset of cropped convergence maps from kappaTNG, with combined redshifts.
    """
    openingangle = get_openingangle(imgsize)
    ktng = KappaTNG(
        idx_lp=idx_lp, weights_redshifts=weights_redshifts,
        openingangle=openingangle, zbins=zbins, **kwargs
    )
    nbins = base_dataset.get_nbins(zbins)

    # Create HDF5 file structure
    iomode = _get_iomode(update_metadata_only, hdf5_filepath)
    with h5py.File(hdf5_filepath, iomode) as f:

        # Metadata
        _update_metadata(f, ktng, zbins=zbins)
        if update_metadata_only:
            if verbose:
                print("Metadata updated")
            return

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
    hdf5_filepath, idx_lp, nimgs, weights_redshifts, imgsize,
    zbins=None, batch_size=50,
    angle_batch_size=ANGLE_BATCH_SIZE, angle_step=5, niter_per_angle=1,
    num_workers=0, update_metadata_only=False,
    resume=False, verbose=False
):
    """
    Create or resume an augmented dataset from kappaTNG
    by rotating and randomly cropping images.
    """
    openingangle = get_openingangle(imgsize)
    ktng = KappaTNG(
        idx_lp=idx_lp, weights_redshifts=weights_redshifts,
        openingangle=openingangle, crop_maps=False, zbins=zbins
    )
    nbins = base_dataset.get_nbins(zbins)

    # --------------------------------------------------
    # Create or open HDF5 file
    # --------------------------------------------------
    iomode = _get_iomode(
        update_metadata_only, hdf5_filepath, resume=resume
    )
    with h5py.File(hdf5_filepath, iomode) as f:

        # Metadata
        _update_metadata(
            f, ktng, zbins=zbins,
            angle_step=angle_step, niter_per_angle=niter_per_angle
        )
        if update_metadata_only:
            if verbose:
                print("Metadata updated")
            return

        if not resume:
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

        # Read resume state
        last_img_idx = f["progress"].attrs["last_img_idx"]
        last_angle_block = f["progress"].attrs["last_angle_block"]

    if num_workers <= 0:
        num_workers = 1
    if verbose:
        print(f"Processing images with {num_workers} workers")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        end_idx = last_img_idx
        while end_idx < nimgs:
            beg_idx = end_idx
            end_idx = min(beg_idx + batch_size, nimgs)

            kappa = ktng.get_kappa(end_idx - beg_idx, start_idx=beg_idx)

            pbar = tqdm.tqdm(
                range(int(np.ceil(360 / (angle_batch_size * angle_step)))),
                disable=not verbose,
            )
            pbar.set_description(
                f"Images {beg_idx + 1}-{end_idx}/{nimgs}"
            )
            for i in pbar:
                # Resume logic
                if beg_idx == last_img_idx and i < last_angle_block:
                    continue

                beg_angle = i * angle_batch_size * angle_step
                end_angle = min(beg_angle + angle_batch_size * angle_step, 360)
                pbar.set_postfix(
                    {"angles": f"{beg_angle:.0f}-{(end_angle-1):.0f}"}
                )

                angles = np.arange(beg_angle, end_angle, angle_step)
                nimgs_batch = len(angles) * niter_per_angle * (end_idx - beg_idx)

                process_angle = lambda angle: dataaugm.rotate_and_crop(
                    kappa, angle, imgsize, niter=niter_per_angle
                )
                results = list(executor.map(process_angle, angles))

                list_of_kappa_rot = [r[0] for r in results]
                list_of_idx_rows = [r[1] for r in results]
                list_of_idx_cols = [r[2] for r in results]

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

            pbar.set_postfix(
                {"angles": f"{0}-{(end_angle-1):.0f}"}
            )

            # Reset angle block after full batch
            with h5py.File(hdf5_filepath, 'r+') as f:
                f["progress"].attrs["last_img_idx"] = end_idx
                f["progress"].attrs["last_angle_block"] = 0


def _get_iomode(update_metadata_only, hdf5_filepath, resume=False):

    file_exists = os.path.exists(hdf5_filepath)
    if update_metadata_only or resume:
        if not file_exists:
            raise FileNotFoundError(hdf5_filepath)
        iomode = "r+"
    else:
        if file_exists:
            raise FileExistsError(hdf5_filepath)
        iomode = "w"

    return iomode


def _update_metadata(
        f: h5py.File, ktng: KappaTNG,
        zbins=None, angle_step=None, niter_per_angle=None
):
    f.attrs["idx_lp"] = ktng.idx_lp
    if ktng.z is not None:
        f.attrs["z"] = ktng.z
    if ktng.cdist is not None:
        f.attrs["cdist"] = ktng.cdist
    if ktng.weights_redshifts is not None:
        f.attrs["weights_redshifts"] = ktng.weights_redshifts
    if ktng.idx_redshifts is not None:
        f.attrs["idx_redshifts"] = ktng.idx_redshifts.astype(
            h5py.string_dtype(encoding="utf-8")
        )

    if zbins is not None:
        f.attrs["zbins"] = zbins
    if angle_step is not None:
        f.attrs["angle_step"] = angle_step
    if niter_per_angle is not None:
        f.attrs["niter_per_angle"] = niter_per_angle
