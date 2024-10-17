import os
import sys
import numpy as np
from scipy import ndimage, signal, stats
import matplotlib.pyplot as plt
import h5py

from lenspack.image.inversion import ks93, ks93inv
from lenspack.utils import bin2d

from . import CONFIG_DATA
pycs_dir = CONFIG_DATA['pycs_dir']
if pycs_dir is not None:
    pycs_dir = os.path.expanduser(pycs_dir)
    sys.path.append(pycs_dir)

import pycs.astro.wl.mass_mapping as csmm

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))
vectorized_ks93 = np.vectorize(ks93, signature='(n,m),(n,m)->(n,m),(n,m)')
vectorized_ks93inv = np.vectorize(ks93inv, signature='(n,m),(n,m)->(n,m),(n,m)')

STD_KSGAUSSIANFILTER = 2.

def test_array_shape(list_of_arr):

    shape = list_of_arr[0].shape
    for arr in list_of_arr[1:]:
        if arr is not None:
            assert arr.shape == shape[-len(arr.shape):]

    return shape


def get_alpha_from_confidence(confidence):
    """
    Parameters
    ----------
    confidence (float)
        Level of confidence (n-sigma)

    """
    return 2 - 2 * stats.norm.cdf(confidence)


def get_min_nimgs_calib(alpha):
    """
    Get minimal size for the calibration set (otherwise the adjusted quantile is above 1)
    
    """
    return np.ceil((1 - alpha) / alpha).astype(int)


def get_resolution(width, size):
    """
    Get resolution in arcmin/pixel.

    Parameters
    ----------
    width (int)
        Size of the convergence maps (nb pixels)
    size (float)
        Opening angle of the convergence maps (deg)
    
    """
    return size / width * 60.


def ngal_per_pixel(ra, dec, width, extent):
    """
    Parameters
    ----------
    ra, dec (numpy.ndarray)
    width (int)
        Size of the target convergence maps (nb pixels).
    extent (4-tuple)
        Extent of the target convergence maps (deg).
    """
    return bin2d(ra, dec, npix=width, extent=extent)


def get_shear_from_convergence(kappa, complexconjugate=False):
    """
    Parameters
    ----------
    kappa (numpy.ndarray, shape=(nimgs, width, width))
        The convergence maps.
    complexconjugate (bool, default=False)   
        Whether to use convention from jax_lensing (due to the inversion of the x-axis?)
    
    """
    bmode = np.zeros_like(kappa) # no B-mode (convergence maps are real-valued)
    gamma1, gamma2 = vectorized_ks93inv(kappa, bmode)
    if complexconjugate:
        gamma2 = -gamma2 # use convention from jax_lensing (due to the inversion of the x-axis?)

    return gamma1, gamma2


def get_std_noise(ngal, shapedisp, std_noise_mask):

    out = np.nan_to_num(
        shapedisp / np.sqrt(ngal), posinf=std_noise_mask
    ) # standard deviation of the noise

    return out


def get_masked_and_noisy_shear(
        gamma1, gamma2, std_noise=None, mask=None,
        ngal=None, shapedisp=None,
        std_noise_mask=None, multfact_std_noise=30.,
        inpainting=False
):
    """
    Parameters
    ----------
    gamma1, gamma2 (numpy.ndarray, shape = (nimgs, nx, ny))
    std_noise (numpy.ndarray, shape = (nx, ny), default=None)
        Array of noise standard deviation.
        If none is given, then arguments `ngal` and `shapedisp` must be provided.
    mask (numpy.ndarray, shape = (nx, ny))
        Array of masked data. If none is given, then argument `ngal` must be provided.
    ngal (numpy.ndarray, shape = (nx, ny))
        Number of measured galaxies per pixel
    shapedisp (float)
        Shape dispersion of galaxies
    std_noise_mask (float, default=None)
        For masked data, we set in practice a variance which makes the SNR very small,
        such that the signal becomes dominated by the noise. This argument explicitly
        provides the value of the standard deviation for masked data.
    multfact_std_noise (float, default=30.)
        Only used if `stdnoise_mask` is not provided. Then, the standard deviation for
        masked data is set to `multfact_stdnoise` times the squared norm of the shear
        map, divided by the number of pixels.
    inpainting (bool, default=False)
        If True, apply noise to the masked regions.

    Returns
    -------
    gamma1_noisy, gamma2_noisy (numpy.ndarray)
        Noisy shear maps, affected by argument `inpainting`.
    std (numpy.ndarray)
        Noise standard deviation, unaffected by argument `inpainting`.
    
    """
    nimgs, width1, width2 = test_array_shape([gamma1, gamma2, std_noise, mask, ngal])

    # Set masked values to 0
    if mask is None:
        mask = ngal > 0
    gamma1_masked = mask * gamma1
    gamma2_masked = mask * gamma2

    # Add noise
    if std_noise is None:
        if std_noise_mask is None:
            sqnorm_gamma = (
                np.linalg.norm(gamma1)**2 + np.linalg.norm(gamma2)**2
            ) / (nimgs * width1 * width2) # normalized squared norm
            std_noise_mask = multfact_std_noise * np.sqrt(sqnorm_gamma / 2)
        std_noise = get_std_noise(ngal, shapedisp, std_noise_mask)
    noise1 = std_noise * np.random.randn(nimgs, width1, width2)
    noise2 = std_noise * np.random.randn(nimgs, width1, width2)

    if not inpainting:
        noise1[:, ~mask] = 0.
        noise2[:, ~mask] = 0.
    gamma1_noisy = gamma1_masked + noise1
    gamma2_noisy = gamma2_masked + noise2

    return gamma1_noisy, gamma2_noisy, std_noise


def get_std_ks(
        std_noise, width1, width2=None, std_gaussianfilter=STD_KSGAUSSIANFILTER, crop_width=32
):
    if width2 is None:
        width2 = width1

    dirac_real = np.zeros((width1, width2))
    dirac_real[-1, -1] = 1.

    dirac_imag = np.zeros((width1, width2))

    ksmatr_real, ksmatr_imag = ks93(dirac_real, dirac_imag)
    ksmatr_gaussian_real = ndimage.gaussian_filter(ksmatr_real, std_gaussianfilter, mode="wrap")
    ksmatr_gaussian_imag = ndimage.gaussian_filter(ksmatr_imag, std_gaussianfilter, mode="wrap")
    ksmatr_gaussian_sqmodule = ksmatr_gaussian_real**2 + ksmatr_gaussian_imag**2

    ksmatr_gaussian_sqmodule = np.fft.fftshift(ksmatr_gaussian_sqmodule) # for convolution

    # Crop convolution kernel for efficiency (fast-decaying coefficients)
    start1 = (width1 - crop_width) // 2
    start2 = (width2 - crop_width) // 2
    ksmatr_gaussian_sqmodule = ksmatr_gaussian_sqmodule[
        start1:start1+crop_width, start2:start2+crop_width
    ]

    out = np.sqrt(
        signal.convolve2d(std_noise**2, ksmatr_gaussian_sqmodule, mode="same", boundary="wrap")
    )
    return out


def ksfilter(
        gamma1_noisy, gamma2_noisy, get_bounds=True, std_noise=None, confidence=None,
        smooth=True, std_gaussianfilter=STD_KSGAUSSIANFILTER, complexconjugate=False
):
    """
    Parameters
    ----------
    gamma1_noisy, gamma2_noisy (numpy.ndarray)
    get_bounds (bool, default=True)
    std_noise (numpy.ndarray, default=None)
    confidence (float, default=None)
        Level of confidence (n-sigma)
    std_gaussianfilter (float)
        Standard deviation of the smoothing filter
    complexconjugate (bool, default=éTrue)   
        Whether to use convention from jax_lensing (due to the inversion of the x-axis?)
    
    """
    arrs = [gamma1_noisy, gamma2_noisy]
    if std_noise is not None:
        arrs.append(std_noise)
    _, width1, width2 = test_array_shape(arrs)

    if complexconjugate:
        gamma2_noisy = -gamma2_noisy
    kappa_ks, _ = vectorized_ks93(gamma1_noisy, gamma2_noisy)
    if smooth:
        kappa_ks = ndimage.gaussian_filter(
            kappa_ks, std_gaussianfilter, mode="wrap", axes=(1, 2)
        ) # KS reconstruction
    if get_bounds:
        std_ks = get_std_ks(
            std_noise, width1, width2, std_gaussianfilter=std_gaussianfilter
        ) # standard deviation of the KS reconstruction
        ppf_ks = confidence * std_ks
        kappa_ks_lo = kappa_ks - ppf_ks
        kappa_ks_hi = kappa_ks + ppf_ks
        out = kappa_ks, kappa_ks_lo, kappa_ks_hi, std_ks
    else:
        out = kappa_ks

    return out


def _split_test_calib(arr, nimgs_calib, calib_first=True):
    if calib_first:
        arr_calib = arr[:nimgs_calib].copy()
        arr_test = arr[nimgs_calib:].copy()
    else:
        arr_calib = arr[-nimgs_calib:].copy()
        arr_test = arr[:-nimgs_calib].copy()
    return arr_calib, arr_test


def split_test_calib(list_of_arr, nimgs_calib, **kwargs):

    list_of_arr_calib, list_of_arr_test = [], []
    for arr in list_of_arr:
        arr_calib, arr_test = _split_test_calib(arr, nimgs_calib, **kwargs)
        list_of_arr_calib.append(arr_calib)
        list_of_arr_test.append(arr_test)

    return list_of_arr_calib, list_of_arr_test


def _get_stats(func, *args, mask=None):

    # TODO: replace by a decorator
    width1, width2 = test_array_shape(args)[-2:]
    if mask is not None:
        assert mask.shape[-2:] == (width1, width2)
    vals = func(*args) # shape = (nimgs, [npatches], nx, ny)
    if mask is not None:
        vals *= mask # shape = (nimgs, [npatches], nx, ny)
        npixels = np.sum(mask, axis=(-2, -1)) # float or shape = (npatches,)
    else:
        npixels = width1 * width2

    return np.sum(vals, axis=(-2, -1)) / npixels # shape = (nimgs, [npatches])


def miscoverage_rate(kappa_lo, kappa_hi, kappa, mask=None):
    """
    Empirical miscoverage rate of the prediction intervals.

    Parameters
    ----------
    kappa_lo, kappa_hi (numpy.ndarray)
        Arrays of shape (nimgs, nx, ny), lower- and upper-bounds of the
        prediction intervals.
    kappa (numpy.ndarray)
        Array of shape (nimgs, nx, ny), ground-truth convergence map.
    mask (numpy.ndarray, default=None)
        Array of shape (nx, ny) or (nimgs, nx, ny), boundaries of the shape catalog.

    Returns
    -------
    out (numpy.ndarray)
        Array of shape (nimgs,)
    
    """
    def func(kappa_lo, kappa_hi, kappa):
        return (kappa < kappa_lo) | (kappa > kappa_hi)
    return _get_stats(func, kappa_lo, kappa_hi, kappa, mask=mask)


def mean_predinterv(kappa_lo, kappa_hi, mask=None):
    def func(kappa_lo, kappa_hi):
        return kappa_hi - kappa_lo
    return _get_stats(func, kappa_lo, kappa_hi, mask=mask)


def normalized_mse(kappa_pred, kappa, mask=None):
    def func(kappa_pred, kappa):
        return (kappa_pred - kappa)**2
    return _get_stats(func, kappa_pred, kappa, mask=mask)


def rmse(kappa_pred, kappa, mask=None):
    return np.sqrt(normalized_mse(kappa_pred, kappa, mask=mask))


def mean_val(kappa_pred, mask=None):
    func = lambda kappa_pred: kappa_pred # identity
    return _get_stats(func, kappa_pred, mask=mask)


def skyshow(
        img, boundaries=None, c='w', cbarshrink=None, title=None,
        printcolorbar=True, printxylabels=True,
        printxticks=True, printyticks=True, offset=0., **kwargs
):
    out = plt.imshow(img - offset, origin='lower', **kwargs)
    plt.xlim(plt.gca().get_xlim()[::-1]) # Flip x-axis
    if printxylabels:
        plt.xlabel("Right ascension")
        plt.ylabel("Declination")
    if not printxticks:
        plt.xticks([])
    if not printyticks:
        plt.yticks([])
    kwargs_cbar = {}
    if cbarshrink is not None:
        kwargs_cbar.update(shrink=cbarshrink)
    if printcolorbar:
        plt.colorbar(**kwargs_cbar)
    if boundaries is not None:
        plt.plot(*boundaries, c=c, lw=1)
    if title is not None:
        plt.title(title)

    return out


def get_emp_variance(func, nreal_noise, *args, **kwargs):
    """
    Get the empirical variance of an estimator by propagating noise multiple times.

    Parameters
    ----------
    func (callable)
        Estimator from which variance is estimated. The function must have a boolean
        parameter `PropagateNoise`.
    nreal_noise (int)
        Number of noise realizations. The variance of the empirical variance depends on
        the true variance as well as the fourth central moment of the estimator, and decreases
        in O(1/nreal_noise). For more information, see
        https://math.stackexchange.com/questions/72975/variance-of-sample-variance
    *args, **kwargs
        Arguments to be passed to `func`.

    """
    est = []
    for _ in range(nreal_noise):
        est.append(func(*args, PropagateNoise=True, **kwargs))
    est = np.stack(est)
    return np.var(est, axis=0)


def illpredicted_perpixel(kappa_ori, kappa_lo, kappa_hi):
    illpredicted = (
        kappa_ori < kappa_lo
    ) | (
        kappa_ori > kappa_hi
    )
    return np.mean(illpredicted, axis=0)


def mean_predinterv_perpixel(kappa_lo, kappa_hi):
    predinterv = kappa_hi - kappa_lo
    return np.mean(predinterv, axis=0)


def patchify(
        inparray, patch_size, npatches_per_side, inpsize=None,
        centermean=False, stack=False, **kwargs
):
    if inpsize is None:
        nx, ny = inparray.shape[-2:]
    else:
        nx = inpsize
        ny = inpsize
    step_x = (nx - patch_size) // (npatches_per_side - 1)
    step_y = (ny - patch_size) // (npatches_per_side - 1)
    out = []
    beg_i = 0
    for _ in range(npatches_per_side):
        beg_j = 0
        for _ in range(npatches_per_side):
            subarray = inparray[..., beg_i:beg_i + patch_size, beg_j:beg_j + patch_size]
            if centermean:
                subarray = subarray - np.mean(subarray, axis=(-2, -1))
            out.append(subarray)
            beg_j += step_y
        beg_i += step_x

    if stack:
        out = np.stack(out, **kwargs)

    return out


def get_beg_end_idx(inpsize, outsize):

    beg_idx = (inpsize - outsize) // 2
    end_idx = beg_idx + outsize

    return beg_idx, end_idx


def crop_arr(arr, beg_idx_x, end_idx_x, *args):

    if len(args) == 2:
        beg_idx_y, end_idx_y = args
    else:
        beg_idx_y = beg_idx_x
        end_idx_y = end_idx_x

    return arr[..., beg_idx_x:end_idx_x, beg_idx_y:end_idx_y]


class HDF5BatchLoader:

    def __init__(
        self, hdf5_filepath, nimgs, batch_size, std_noise, mask,
        offset=0., beg_idx=0, shuffle=True, output_shape=None,
        sort_by_filename_ori=True, newaxis=False,
        compute_wiener=0, powerspectrum_1d=None,
        list_of_outputs=None, verbose=False, **kwargs
    ):
        """
        Initialize the batch loader for HDF5 data.

        Parameters
        ----------
        hdf5_filepath : str
            Path to the HDF5 dataset containing the simulated convergence maps.
        nimgs : int
            Number of images in the dataset. Indices from `beg_idx` to
            `beg_idx + nimgs` are considered.
        batch_size : int
            Number of images per batch.
        std_noise : numpy.ndarray
            Array of noise standard deviation.
        mask : numpy.ndarray
            Array of masked data.
        offset: float, optional
            Mean value of the convergence field (mass-sheet degeneracy). Default is 0.
        beg_idx : int, optional
            First image index to consider (e.g., for split training-test sets). Default is 0.
            CAUTION: To ensure independence between the training and test sets,
            `sort_by_filename_ori` must be set to `True`. Moreover, the split position
            must be chosen so that `filename_ori` values are different in the training and
            test sets.
        shuffle : bool, optional
            Whether to shuffle the indices. Default is True.
        output_shape : tuple, optional
            Shape to crop the output images. Default is None.
        sort_by_filename_ori: bool, optional
            If True, sort `kappa` elements by ascending order of `filename_ori`.
            Default is True.
        newaxis: bool, optional
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1),
            for training purpose.
        compute_wiener: int, optional
            If greater than 0, compute the Wiener solution using one of the following methods:
                - 1: standard Wiener filtering (one iteration)
                - 2: iterative Wiener filtering algorithm.
            Default is False.
        powerspectrum_1d: np.ndarray, optional
            1D power spectrup for iterative Wiener filtering. Its length must be half
            the image size. Default is None.
        list_of_outputs: list of str, optional
            List of outputs to returns. Can be one of 'kappa', 'gamma1', 'gamma2',
            'gamma1_noisy', 'gamma2_noisy', 'kappa_wiener', 'filename_ori'.
            If None, returns a dictionary of outputs. Default is None.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        **kwargs
            Keyword arguments for
            `pycs.astro.wl.mass_mapping.massmap2d.wiener` or
            `pycs.astro.wl.mass_mapping.massmap2d.prox_wiener_filtering`.
        """
        self.hdf5_filepath = hdf5_filepath
        self.nimgs = nimgs
        self.batch_size = batch_size
        self.std_noise = std_noise
        self.mask = mask
        self.offset = offset
        self.beg_idx = beg_idx
        self.shuffle = shuffle
        self.output_shape = output_shape
        self.sort_by_filename_ori = sort_by_filename_ori
        self.newaxis = newaxis
        self.compute_wiener = compute_wiener
        self.powerspectrum_1d = powerspectrum_1d
        self.kwargs_wiener = kwargs
        self.list_of_outputs = list_of_outputs
        self.verbose = verbose

        self.idx = None  # Will hold the shuffled indices
        self.file = None  # HDF5 file object
        self.dataset = None
        self.current_idx = 0  # To track the batch number
        self.sheardata = None # For Wiener filtering

        self._initialize_dataset()
        self._initialize_wiener()


    def _initialize_dataset(self):
        """Load the HDF5 file and initialize the dataset."""
        self.file = h5py.File(self.hdf5_filepath, 'r')  # Keep file open
        self.dataset = self.file['kappa']
        self.filename_ori = self.file['filename_ori']  # Load the `filename_ori` dataset
        nimgs_tot, nx, ny = self.dataset.shape

        # Check if requested number of images exceeds total available
        if self.beg_idx + self.nimgs > nimgs_tot:
            self.file.close()  # Close the file in case of error
            raise ValueError("The requested size exceeds the size of the dataset.")

        # Initialize list of indices
        if self.sort_by_filename_ori:
            idx = np.argsort(self.filename_ori)  # Sort indices of `filename_ori`
        else:
            idx = np.arange(nimgs_tot)
        self.idx = idx[self.beg_idx:self.beg_idx + self.nimgs]
        if self.shuffle:
            np.random.shuffle(self.idx)

        # Get crop indices, if required
        if self.output_shape is not None:
            try:
                nx_out, ny_out = self.output_shape
            except TypeError:
                nx_out = self.output_shape
                ny_out = self.output_shape
            self._beg_idx_x, self._end_idx_x = get_beg_end_idx(nx, nx_out)
            self._beg_idx_y, self._end_idx_y = get_beg_end_idx(ny, ny_out)
            self.nx = nx_out
            self.ny = ny_out
        else:
            self.nx = nx
            self.ny = ny
        assert self.std_noise.shape[-2:] == (self.nx, self.ny)
        assert self.mask.shape[-2:] == (self.nx, self.ny)


    def _initialize_wiener(self):
        """Initialize the parameters for iterative Wiener filtering."""
        if self.compute_wiener > 0:
            # Register data into a `csmm.shear_data` object
            self.sheardata = csmm.shear_data()
            self.sheardata.mask = self.mask.astype(int)
            self.sheardata.Ncov = 2 * self.std_noise**2 # Factor 2 required

            # Create a mass mapping structure and initialize it
            massmap = csmm.massmap2d(name='mass')
            massmap.init_massmap(self.nx, self.ny)
            if self.compute_wiener == 1:
                self.wiener = massmap.wiener
            else:
                self.wiener = massmap.prox_wiener_filtering


    def __call__(self, get_all_images=False):
        end_idx = 0
        while True:
            # Load the next batch of data
            beg_idx = end_idx
            if not get_all_images:
                end_idx = min(beg_idx + self.batch_size, self.nimgs)
            else:
                end_idx = self.nimgs
            batch_idx = self.idx[beg_idx:end_idx]

            # Sort batch_idx to ensure increasing order for HDF5 access
            sort_idx = np.argsort(batch_idx)
            sorted_batch_idx = batch_idx[sort_idx]

            # Load batch with sorted indices
            kappa = self.dataset[sorted_batch_idx]
            filename_ori = self.filename_ori[sorted_batch_idx]

            # Re-order the batch
            reversed_sort_idx = np.argsort(sort_idx)
            kappa = kappa[reversed_sort_idx]
            filename_ori = filename_ori[reversed_sort_idx]

            # Crop the batch if output_shape is specified
            if self.output_shape is not None:
                kappa = crop_arr(
                    kappa,
                    self._beg_idx_x, self._end_idx_x,
                    self._beg_idx_y, self._end_idx_y
                )

            # Generate noisy shear maps
            gamma1, gamma2 = get_shear_from_convergence(kappa)
            gamma1_noisy, gamma2_noisy, _ = get_masked_and_noisy_shear(
                gamma1, gamma2, std_noise=self.std_noise, mask=self.mask
            )
            if self.verbose:
                print(f"Images {beg_idx} to {end_idx} loaded.")

            if self.newaxis:
                out_dict = dict(
                    kappa=kappa[..., np.newaxis] + self.offset,
                    gamma1=gamma1[..., np.newaxis],
                    gamma2=gamma2[..., np.newaxis],
                    gamma1_noisy=gamma1_noisy[..., np.newaxis],
                    gamma2_noisy=gamma2_noisy[..., np.newaxis]
                )
            else:
                out_dict = dict(
                    kappa=kappa + self.offset,
                    gamma1=gamma1,
                    gamma2=gamma2,
                    gamma1_noisy=gamma1_noisy,
                    gamma2_noisy=gamma2_noisy
                )
            out_dict.update(filename_ori=filename_ori)


            # Compute Wiener solution if required
            if self.compute_wiener > 0:
                self.sheardata.g1 = gamma1_noisy
                self.sheardata.g2 = gamma2_noisy
                kappa_wiener, _ = self.wiener(
                    self.sheardata, self.powerspectrum_1d, **self.kwargs_wiener
                )
                if self.newaxis:
                    out_dict.update(
                        kappa_wiener=kappa_wiener[..., np.newaxis] + self.offset
                    )
                else:
                    out_dict.update(
                        kappa_wiener=kappa_wiener + self.offset
                    )
                if self.verbose:
                    print("\tWiener solution computed.")

            # Prepare output
            if self.list_of_outputs is not None:
                out = tuple(
                    [out_dict[val] for val in self.list_of_outputs]
                )
                if len(out) == 1:
                    out = out[0]
            else:
                out = out_dict

            # Handle generator looping (to avoid StopIteration error)
            # Reset generator and reshuffle indices if needed
            if end_idx == self.nimgs:
                end_idx = 0
                if self.shuffle:
                    np.random.shuffle(self.idx)

            yield out


    def close(self):
        """Close the HDF5 file when done."""
        if self.file is not None:
            self.file.close()


    def __del__(self):
        """Destructor to ensure the HDF5 file is closed when the object is deleted."""
        self.close()
