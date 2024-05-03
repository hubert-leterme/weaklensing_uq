import warnings

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

    nimgs, width1, width2 = list_of_arr[0].shape
    for arr in list_of_arr[1:]:
        if len(arr.shape) == 2:
            assert arr.shape == (width1, width2)
        elif len(arr.shape) == 3:
            assert arr.shape == (nimgs, width1, width2)
        else:
            raise ValueError("Wrong number of dimensions")

    return nimgs, width1, width2


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
        arr_calib = arr[:nimgs_calib]
        arr_test = arr[nimgs_calib:]
    else:
        arr_calib = arr[-nimgs_calib:]
        arr_test = arr[:-nimgs_calib]
    return arr_calib, arr_test


def split_test_calib(list_of_arr, nimgs_calib, **kwargs):

    list_of_arr_calib, list_of_arr_test = [], []
    for arr in list_of_arr:
        arr_calib, arr_test = _split_test_calib(arr, nimgs_calib, **kwargs)
        list_of_arr_calib.append(arr_calib)
        list_of_arr_test.append(arr_test)

    return list_of_arr_calib, list_of_arr_test


# Use **kwargs for argument `axis`, for instance
def loss(kappa_lo, kappa_hi, kappa, mask=None):
    """
    Compute accuracy of the prediction intervals
    
    """
    _, width1, width2 = test_array_shape([kappa, kappa_lo, kappa_hi])
    if mask is not None:
        assert mask.shape == (width1, width2)
    ill_predicted = (kappa < kappa_lo) | (kappa > kappa_hi)
    if mask is not None:
        ill_predicted *= mask
        npixels = np.sum(mask)
    else:
        npixels = width1 * width2

    return np.sum(ill_predicted, axis=(-2, -1)) / npixels


def _adjusted_quantiles(alpha, conformity_scores):
    """
    Get quantiles of a given array of conformity scores over a calibration
    set, with finite-sample correction.

    Parameters
    ----------
    alpha (float)
        Target error level
    conformity_scores (array-like)
        Array of shape (nimgs_calib, nx, ny), where nimgs_calib denotes
        the number of images in the calibration set
    
    Returns
    -------
    quantile_vals (array-like)
        Array of shape (nx, ny): the adjusted quantiles
    adjusted_quantile (float)
        Adjusted quantile index (between 0 and 1)
    
    """
    nimgs_calib = conformity_scores.shape[0]
    assert nimgs_calib >= get_min_nimgs_calib(alpha)
    adjusted_quantile = (1 - alpha) * (1 + 1/nimgs_calib)
    quantile_vals = np.percentile(conformity_scores, adjusted_quantile*100, axis=0)
    return quantile_vals, adjusted_quantile


def conformalize(
        kappa_lo_test, kappa_hi_test,
        kappa_lo_calib, kappa_hi_calib,
        kappa_calib, alpha
):
    """
    Perform conformal calibration as in Y. Romano, E. Patterson, and E. Candes,
    “Conformalized Quantile Regression,” in NeurIPS, 2023.

    Parameters
    ----------
    kappa_lo_test, kappa_hi_test (array-like)
        Predicted lower and upper bounds on the test set
    kappa_lo_calib, kappa_hi_calib (array-like)
        Predicted lower and upper bounds on the calibration set
    kappa_calib (array-like)
        Ground-truth convergence maps (calibration set)
    alpha (float)
        Target error level
    
    """
    conformity_scores = np.maximum(
        kappa_lo_calib - kappa_calib,
        kappa_calib - kappa_hi_calib
    )
    quantile_vals, adjusted_quantile = _adjusted_quantiles(alpha, conformity_scores)

    # Adjust the bounds on the test set
    kappa_lo_cqr_test = kappa_lo_test - quantile_vals
    kappa_hi_cqr_test = kappa_hi_test + quantile_vals

    return kappa_lo_cqr_test, kappa_hi_cqr_test, quantile_vals, adjusted_quantile


def conformalize_mult(
        kappa_lo_test, kappa_hi_test,
        kappa_lo_calib, kappa_hi_calib,
        kappa_calib, alpha, eps=1e-9
):
    r"""
    Perform conformal calibration in a multiplicative way:
    $C(x) = [\hat{f}(x) - \lambda \hat{r}(x),\, \hat{f}(x) + \lambda \hat{r}(x)]$,
    where $\hat{f}$ denotes a point estimator, $\hat{r}$ a residual expressing
    uncertainty, and $\lambda$ the calibration parameter. This is inspired by
    A. N. Angelopoulos et al., “Image-to-Image Regression with Distribution-Free
    Uncertainty Quantification and Applications in Imaging,” in Proceedings of
    the 39th International Conference on Machine Learning, PMLR, Jun. 2022, pp. 717–730.
    However, the conformal calibration method in itself follows the theory developed
    by Y. Romano, E. Patterson, and E. Candes, “Conformalized Quantile Regression,”
    in NeurIPS, 2023.

    Parameters
    ----------
    kappa_lo_test, kappa_hi_test (array-like)
        Predicted lower and upper bounds on the test set
    kappa_lo_calib, kappa_hi_calib (array-like)
        Predicted lower and upper bounds on the calibration set
    kappa_calib (array-like)
        Ground-truth convergence maps (calibration set)
    alpha (float)
        Target error level
    eps (float, default=1e-9)
        Small value to avoid division by 0
    
    """
    kappa_pred_test = (kappa_lo_test + kappa_hi_test) / 2
    kappa_pred_calib = (kappa_lo_calib + kappa_hi_calib) / 2
    res_test = (kappa_hi_test - kappa_lo_test) / 2
    res_calib = (kappa_hi_calib - kappa_lo_calib) / 2

    res_calib[res_calib <= eps] = eps
    res_test[res_test <= eps] = eps

    conformity_scores = np.abs(kappa_calib - kappa_pred_calib) / res_calib
    quantile_vals, adjusted_quantile = _adjusted_quantiles(alpha, conformity_scores)

    # Adjust the bounds on the test set
    kappa_lo_cqr_test = kappa_pred_test - quantile_vals * res_test
    kappa_hi_cqr_test = kappa_pred_test + quantile_vals * res_test

    return kappa_lo_cqr_test, kappa_hi_cqr_test, quantile_vals, adjusted_quantile


def conformalize_exp(
        kappa_lo_test, kappa_hi_test,
        kappa_lo_calib, kappa_hi_calib,
        kappa_calib, alpha, eps=1e-9,
        rho=None, scalefact=1., cosmos_mask=None
):
    r"""
    Perform conformal calibration using the following calibration function:
    $$
        g_\lambda: r \mapsto \rho(r) (\mathrm e^\lambda - 1) + r,
    $$
    for some user-specified function $\rho$.
    In this context, the conformity scores are equal to:
    $$
        \conformityScoreRV_i = \log\left(
            1 + \frac{
                \left|
                    \hat f(\mathsf X_i) - \mathsf Y_i
                \right| - \hat r(\mathsf X_i)
            }{
                \rho\left(
                    \hat r(\mathsf X_i)
                \right)
            }
        \right).
    $$

    Parameters
    ----------
    kappa_lo_test, kappa_hi_test (array-like)
        Predicted lower and upper bounds on the test set
    kappa_lo_calib, kappa_hi_calib (array-like)
        Predicted lower and upper bounds on the calibration set
    kappa_calib (array-like)
        Ground-truth convergence maps (calibration set)
    alpha (float)
        Target error level
    eps (float, default=1e-9)
        Small value to avoid division by 0
    rho (callable, default=None)
        Function $\rho$ used in the calibration function. If none is given, then
        the identity function is used.
    scalefact(float, default=1.)
        Scale factor > 0 used when applying the calibration function
    cosmos_mask (array-like, default=None)
        When proper calibration is impossible (due to the calibration function),
        a warning is triggered. However, the warning will be ignored if this happens
        outside the survey boundaries
    
    """
    kappa_pred_test = (kappa_lo_test + kappa_hi_test) / 2
    kappa_pred_calib = (kappa_lo_calib + kappa_hi_calib) / 2

    res_test = (kappa_hi_test - kappa_lo_test) / 2
    res_calib = (kappa_hi_calib - kappa_lo_calib) / 2

    weights_test = scalefact * rho(res_test / scalefact)
    weights_calib = scalefact * rho(res_calib / scalefact)
    weights_test[weights_test <= eps] = eps
    weights_calib[weights_calib <= eps] = eps

    conformity_scores = np.log(
        1 + (
            np.abs(kappa_pred_calib - kappa_calib) - res_calib
        ) / weights_calib
    )
    conformity_scores = np.nan_to_num(conformity_scores, nan=-np.inf)
    quantile_vals, adjusted_quantile = _adjusted_quantiles(alpha, conformity_scores)
    isnan = np.isnan(quantile_vals)
    if cosmos_mask is not None:
        isnan[cosmos_mask] = False
    sum_isnan = np.sum(isnan)
    if sum_isnan > 0:
        warnings.warn(
            f"Some pixels are impossible to calibrate ({sum_isnan / isnan.size:.0%}); the "
            "predictions will be overconservative. Choose another calibration function."
        )

    # Adjust the bounds on the test set
    res_cqr_test = res_test + weights_test * (np.exp(quantile_vals) - 1)

    kappa_lo_cqr_test = kappa_pred_test - res_cqr_test
    kappa_hi_cqr_test = kappa_pred_test + res_cqr_test

    return kappa_lo_cqr_test, kappa_hi_cqr_test, quantile_vals, adjusted_quantile


def get_bounds_proba_cqr(alpha, nimgs_calib):
    lower_bound_proba = alpha - 1 / (nimgs_calib + 1)
    upper_bound_proba = alpha
    return lower_bound_proba, upper_bound_proba


def skyshow(
        img, boundaries=None, c='w', cbarshrink=None, title=None,
        printcolorbar=True, printxylabels=True, **kwargs
):
    out = plt.imshow(img, origin='lower', **kwargs)
    plt.xlim(plt.gca().get_xlim()[::-1]) # Flip x-axis
    if printxylabels:
        plt.xlabel("Right ascension")
        plt.ylabel("Declination")
    kwargs_cbar = {}
    if cbarshrink is not None:
        kwargs_cbar.update(shrink=cbarshrink)
    if printcolorbar:
        plt.colorbar(**kwargs_cbar)
    if boundaries is not None:
        plt.plot(*boundaries,  c=c, lw=1)
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


def mean_predinterv(kappa_lo, kappa_hi, cosmos_mask=None):
    predinterv = kappa_hi - kappa_lo
    if cosmos_mask is not None:
        predinterv = predinterv[:, cosmos_mask]
    return np.mean(predinterv)


def mean_predinterv_perpixel(kappa_lo, kappa_hi):
    predinterv = kappa_hi - kappa_lo
    return np.mean(predinterv, axis=0)
