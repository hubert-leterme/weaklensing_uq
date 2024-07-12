import numpy as np
from scipy import ndimage, signal, stats
import matplotlib.pyplot as plt

from lenspack.image.inversion import ks93, ks93inv
from lenspack.utils import bin2d

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))
vectorized_ks93 = np.vectorize(ks93, signature='(n,m),(n,m)->(n,m),(n,m)')
vectorized_ks93inv = np.vectorize(ks93inv, signature='(n,m),(n,m)->(n,m),(n,m)')

STD_KSGAUSSIANFILTER = 2.

def test_array_shape(list_of_arr):

    shape = list_of_arr[0].shape
    for arr in list_of_arr[1:]:
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
    ra, dec (array-like)
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
    kappa (array-like, shape=(nimgs, width, width))
        The convergence maps.
    complexconjugate (bool, default=False)   
        Whether to use convention from jax_lensing (due to the inversion of the x-axis?)
    
    """
    bmode = np.zeros_like(kappa) # no B-mode (convergence maps are real-valued)
    gamma1, gamma2 = vectorized_ks93inv(kappa, bmode)
    if complexconjugate:
        gamma2 = -gamma2 # use convention from jax_lensing (due to the inversion of the x-axis?)

    return gamma1, gamma2


def get_masked_and_noisy_shear(
        gamma1, gamma2, ngal, shapedisp, stdnoise_mask=None, multfact_stdnoise=30.,
        inpainting=False
):
    """
    Parameters
    ----------
    gamma1, gamma2 (array-like)
    ngal (array-like)
        Number of measured galaxies per pixel
    shapedisp (float)
        Shape dispersion of galaxies
    stdnoise_mask (float, default=None)
        For masked data, we set in practice a variance which makes the SNR very small,
        such that the signal becomes dominated by the noise. This argument explicitly
        provides the value of the standard deviation for masked data.
    multfact_stdnoise (float, default=30.)
        Only used if `stdnoise_mask` is not provided. Then, the standard deviation for
        masked data is set to `multfact_stdnoise` times the squared norm of the shear
        map, divided by the number of pixels.
    inpainting (bool, default=False)
        If True, apply noise to the masked regions.

    Returns
    -------
    gamma1_noisy, gamma2_noisy (array-like)
        Noisy shear maps, affected by argument `inpainting`.
    std (array-like)
        Noise standard deviation, unaffected by argument `inpainting`.
    
    """
    nimgs, width1, width2 = test_array_shape([gamma1, gamma2, ngal])

    # Set masked values to 0
    mask = (ngal == 0)
    gamma1_masked = (1 - mask) * gamma1
    gamma2_masked = (1 - mask) * gamma2

    # Add noise
    sqnorm_gamma = (
        np.linalg.norm(gamma1)**2 + np.linalg.norm(gamma2)**2
    ) / (nimgs * width1 * width2) # normalized squared norm
    if stdnoise_mask is None:
        stdnoise_mask = multfact_stdnoise * np.sqrt(sqnorm_gamma / 2)

    std = np.nan_to_num(
        shapedisp / np.sqrt(ngal), posinf=stdnoise_mask
    ) # standard deviation of the noise
    noise1 = std * np.random.randn(nimgs, width1, width2)
    noise2 = std * np.random.randn(nimgs, width1, width2)

    if not inpainting:
        noise1[:, mask] = 0.
        noise2[:, mask] = 0.
    gamma1_noisy = gamma1_masked + noise1
    gamma2_noisy = gamma2_masked + noise2

    return gamma1_noisy, gamma2_noisy, std


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
    gamma1_noisy, gamma2_noisy (array-like)
    get_bounds (bool, default=True)
    std_noise (array-like, default=None)
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


def loss(kappa_lo, kappa_hi, kappa, mask=None):
    """
    Empirical miscoverage rate of the prediction intervals.

    Parameters
    ----------
    kappa_lo, kappa_hi (array-like)
        Arrays of shape (nimgs, nx, ny), lower- and upper-bounds of the
        prediction intervals.
    kappa (array-like)
        Array of shape (nimgs, nx, ny), ground-truth convergence map.
    mask (array-like, default=None)
        Array of shape (nx, ny) or (nimgs, nx, ny), boundaries of the shape catalog.

    Returns
    -------
    out (array-like)
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


def mean_val(kappa_pred, mask=None):
    func = lambda kappa_pred: kappa_pred # identity
    return _get_stats(func, kappa_pred, mask=mask)


def skyshow(
        img, boundaries=None, c='w', cbarshrink=None, title=None,
        printcolorbar=True, printxylabels=True,
        printxticks=True, printyticks=True, **kwargs
):
    out = plt.imshow(img, origin='lower', **kwargs)
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
    step_y = (nx - patch_size) // (npatches_per_side - 1)
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
