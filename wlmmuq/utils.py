__level__ = 1

# TODO: Clean this module; split it into several ones

import os
from datetime import datetime
import typing
import math
import numpy as np
import tqdm
from scipy import ndimage, signal, stats, sparse, linalg
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepinv as dinv
import astropy.io.fits as apfits

# from lenspack.image.inversion import ks93, ks93inv
# from lenspack.utils import bin2d

from . import lenspack
from .config import KEY_REPLACEMENT_DICT, PATH_TO_XCLUS, PATH_TO_ZBINS

ITS_POWER_ITERATION = 100 # The default value implemented in scipy (20) is too small

# Global variables for plotting X-ray clusters
XCLUS_ZMIN = 0.3
XCLUS_ZMAX = 0.99
XCLUS_M500MIN = 3
XCLUS_MTEXT = False # overplot x-clusters M500
XCLUS_ZTEXT = True  # overplot x-clusters redshift

vectorized_zfill = np.vectorize(lambda x: str(x).zfill(3))
#vectorized_ks93 = np.vectorize(ks93, signature='(n,m),(n,m)->(n,m),(n,m)')
#vectorized_ks93inv = np.vectorize(ks93inv, signature='(n,m),(n,m)->(n,m),(n,m)')

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


def get_min_nimgs_calib(alpha: float):
    """
    Get minimal size for the calibration set (otherwise the adjusted quantile is above 1)
    
    """
    return np.ceil((1 - alpha) / alpha).astype(int)


def get_resolution(width, openingangle):
    """
    Get resolution in arcmin/pixel.

    Parameters
    ----------
    width (int)
        Size of the convergence maps (nb pixels)
    openingangle (float)
        Opening angle of the convergence maps (deg)
    
    """
    return openingangle / width * 60.


def _get_shear_fromto_convergence(
        func: typing.Callable, inp1: np.ndarray | torch.Tensor,
        inp2: np.ndarray | torch.Tensor | None = None,
        complexconjugate=False, return_complex=False
):
    if inp2 is None:
        inp1 = convert_to_complex(inp1)
        inp2 = inp1.imag
        inp1 = inp1.real
    if complexconjugate:
        # Use convention from jax_lensing (due to the inversion of the x-axis?)
        inp2 = -inp2
    out1, out2 = func(inp1, inp2)
    if complexconjugate:
        # Use convention from jax_lensing (due to the inversion of the x-axis?)
        out2 = -out2

    if return_complex:
        out = out1 + 1j * out2
    else:
        out = (out1, out2)

    return out


def get_shear_from_convergence(
        kappa1: np.ndarray | torch.Tensor,
        kappa2: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        complexconjugate=False, return_complex=False
):
    """
    Parameters
    ----------
    kappa1, kappa2 (numpy.ndarray, shape=(nimgs, width, width), default=None for kappa2)
        Real and imaginary parts of the input shear maps. If kappa2 is None, then kappa1
        is assumed to be complex-valued.
    mask (numpy.ndarray, shape=(width, width), default=None)
        If not None, then the shear values outside the mask are set to 0 after having
        applied the inverse Kaiser-Squires filter.
    complexconjugate (bool, default=False)   
        Whether to use convention from jax_lensing (due to the inversion of the x-axis?).
    return_complex (bool, default=False)
        If True, then a complex-valued numpy array will be returned. If False, then
        two real-valued numpy arrays will be returned.

    """
    gamma = _get_shear_fromto_convergence(
        lenspack.ks93inv, kappa1, kappa2,
        complexconjugate=complexconjugate, return_complex=return_complex
    )
    if mask is not None:
        check_mask(mask)
        if return_complex:
            gamma[..., ~mask] = 0
        else:
            gamma1, gamma2 = gamma
            gamma1[..., ~mask] = 0
            gamma2[..., ~mask] = 0
    return gamma


def get_convergence_from_shear(
        gamma1: np.ndarray | torch.Tensor,
        gamma2: np.ndarray | torch.Tensor | None = None,
        mask: np.ndarray | torch.Tensor | None = None,
        complexconjugate=False, return_complex=False
):
    """
    Parameters
    ----------
    gamma1, gamma2 (numpy.ndarray, default=None for gamma2)
        Real and imaginary parts of the input shear maps. If gamma2 is None, then gamma1
        is assumed to be complex-valued.
    mask (numpy.ndarray, shape=(width, width), default=None)
        If not None, then the shear values outside the mask are set to 0 before applying
        the Kaiser-Squires filter.
    complexconjugate (bool, default=éTrue)   
        Whether to use convention from jax_lensing (due to the inversion of the x-axis?)
    return_complex (bool, default=False)
        If True, then a complex-valued numpy array will be returned. If False, then
        two real-valued numpy arrays will be returned.
    
    """
    if mask is not None:
        check_mask(mask)
        gamma1[..., ~mask] = 0
        if gamma2 is not None:
            gamma2[..., ~mask] = 0
    kappa = _get_shear_fromto_convergence(
        lenspack.ks93, gamma1, gamma2,
        complexconjugate=complexconjugate, return_complex=return_complex
    )
    return kappa


def convert_to_complex(arr: np.ndarray | torch.Tensor):

    if isinstance(arr, np.ndarray):
        if not np.iscomplexobj(arr):
            if arr.dtype == np.float32:
                arr = arr.astype(np.complex64)
            elif arr.dtype == np.float64:
                arr = arr.astype(np.complex128)
            else:
                raise TypeError(f"Unsupported NumPy dtype: {arr.dtype}")

    elif isinstance(arr, torch.Tensor):
        if not arr.is_complex():
            if arr.dtype == torch.float32:
                arr = arr.to(torch.complex64)
            elif arr.dtype == torch.float64:
                arr = arr.to(torch.complex128)
            else:
                raise TypeError(f"Unsupported PyTorch dtype: {arr.dtype}")

    else:
        raise TypeError(f"Unsupported input type: {type(arr)}")

    return arr


def get_masked_and_noisy_shear(
        gamma: np.ndarray | torch.Tensor,
        std_noise: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor | None = None,
        inpainting: bool = False,
        device=None
):
    """
    Parameters
    ----------
    gamma (numpy.ndarray | torch.tensor, shape = (nimgs, nx, ny), dtype=complex)
    std_noise (numpy.ndarray, shape = (nx, ny))
        Array of noise standard deviation.
    mask (numpy.ndarray, shape = (nx, ny), default=None)
        Array of masked data.
    inpainting (bool, default=False)
        If True, apply noise to the masked regions.
    device

    Returns
    -------
    gamma_noisy (numpy.ndarray | torch.Tensor)
        Noisy shear maps, affected by argument `inpainting`.
    
    """
    if torch.is_tensor(gamma):
        randn = torch.randn
    else:
        randn = np.random.randn

    if mask is None:
        if torch.is_tensor(std_noise):
            mask = torch.ones_like(std_noise, dtype=torch.bool)
        else:
            mask = np.ones_like(std_noise, dtype=bool)
    assert mask is not None

    if device is not None:
        assert torch.is_tensor(mask)
        mask = mask.to(device)

    shape = test_array_shape([gamma, std_noise, mask])

    # Set masked values to 0
    if mask is not None:
        check_mask(mask)
        gamma_masked = mask * gamma

    # TODO: use physics = phys.MassMapping(...)
    def _get_noisy_shear(gamma_masked, std_noise, mask, shape):
        noise = randn(*shape) + 1j * randn(*shape)
        if device is not None:
            assert torch.is_tensor(noise)
            noise = noise.to(device)
        noise *= std_noise
        if not inpainting and mask is not None:
            noise[..., ~mask] = 0.
        return gamma_masked + noise

    gamma_noisy = _get_noisy_shear(gamma_masked, std_noise, mask, shape)

    return gamma_noisy


def get_std_ks(
        std_noise, width1, width2=None, std_gaussianfilter=None, crop_width=32
):
    if width2 is None:
        width2 = width1

    dirac_real = np.zeros((width1, width2))
    dirac_real[-1, -1] = 1.

    dirac_imag = np.zeros((width1, width2))

    ksmatr_real, ksmatr_imag = lenspack.ks93(dirac_real, dirac_imag)
    if std_gaussianfilter is not None:
        ksmatr_real = ndimage.gaussian_filter(
            ksmatr_real, std_gaussianfilter, mode="wrap"
        )
        ksmatr_imag = ndimage.gaussian_filter(
            ksmatr_imag, std_gaussianfilter, mode="wrap"
        )
    ksmatr_sqmodule = ksmatr_real**2 + ksmatr_imag**2
    ksmatr_sqmodule = np.fft.fftshift(ksmatr_sqmodule) # for convolution

    # Crop convolution kernel for efficiency (fast-decaying coefficients)
    start1 = (width1 - crop_width) // 2
    start2 = (width2 - crop_width) // 2
    ksmatr_sqmodule = ksmatr_sqmodule[
        start1:start1+crop_width, start2:start2+crop_width
    ]

    out = np.sqrt(
        signal.convolve2d(std_noise**2, ksmatr_sqmodule, mode="same", boundary="wrap")
    )
    return out


def ksfilter(
        gamma1_noisy, gamma2_noisy, get_bounds=True, std_noise=None, confidence=None,
        std_gaussianfilter=None, complexconjugate=False
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
    complexconjugate (bool, default=True)   
        Whether to use convention from jax_lensing (due to the inversion of the x-axis?)
    
    """
    arrs = [gamma1_noisy, gamma2_noisy]
    if std_noise is not None:
        arrs.append(std_noise)
    _, width1, width2 = test_array_shape(arrs)

    kappa_ks = get_convergence_from_shear(
        gamma1_noisy, gamma2_noisy, complexconjugate=complexconjugate
    )
    if std_gaussianfilter is not None:
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


def _split_test_calib(
        arr: np.ndarray | torch.Tensor | float, nimgs_calib, calib_first=True
) -> tuple[np.ndarray | torch.Tensor]:

    if not isinstance(arr, float):
        if torch.is_tensor(arr):
            arr = arr.clone()
        else:
            arr = arr.copy()
        if calib_first:
            arr_calib = arr[:nimgs_calib]
            arr_test = arr[nimgs_calib:]
        else:
            arr_calib = arr[-nimgs_calib:]
            arr_test = arr[:-nimgs_calib]
    else:
        arr_calib = arr
        arr_test = arr

    return arr_calib, arr_test


def split_test_calib(list_of_arr, nimgs_calib, **kwargs):

    list_of_arr_calib, list_of_arr_test = [], []
    for arr in list_of_arr:
        arr_calib, arr_test = _split_test_calib(arr, nimgs_calib, **kwargs)
        list_of_arr_calib.append(arr_calib)
        list_of_arr_test.append(arr_test)

    return list_of_arr_calib, list_of_arr_test


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


def get_beg_end_idx(inpsize, outsize):

    assert inpsize >= outsize
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


def get_powerspectrum(kappa: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Estimate the 2D powerspectrum over a set of isotropic images.

    Parameters
    ----------
    kappa: numpy.ndarray, shape = (nimgs, imgsize, imgsize)
        Set of square images. CAUTION: `imgsize` must be even.

    """
    imgsize, imgsize0 = kappa.shape[-2:]
    assert imgsize0 == imgsize
    powerspectrum = absolute(fft2(kappa) / imgsize)**2
    powerspectrum = powerspectrum.mean(axis=0)

    return powerspectrum


def get_1d_powerspectrum(kappa: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Estimate the 1D powerspectrum over a set of isotropic images.

    Parameters
    ----------
    kappa: numpy.ndarray, shape = (nimgs, imgsize, imgsize)
        Set of square images. CAUTION: `imgsize` must be even.

    """
    imgsize, imgsize0 = kappa.shape[-2:]
    assert imgsize0 == imgsize
    powerspectrum = get_powerspectrum(kappa)
    powerspectrum = powerspectrum[:imgsize//2, :imgsize//2] # Only positive frequencies, by symmetry
    powerspectrum_1d = (powerspectrum[0, :] + powerspectrum[:, 0]) / 2 # Assumed isotropic

    return powerspectrum_1d


def get_openingangle(imgsize, resolution):
    return imgsize * resolution / 60.


def check_mask(mask: np.ndarray | torch.Tensor):
    if torch.is_tensor(mask):
        assertion = mask.dtype == torch.bool
    else:
        assertion = mask.dtype == bool
    if not assertion:
        raise ValueError("mask must be a boolean array")


def meancenter(
        arr: np.ndarray | torch.Tensor, axis: int | tuple=(-3, -2, -1),
        mask: np.ndarray | torch.Tensor | None = None
) -> np.ndarray | torch.Tensor:

    if torch.is_tensor(arr):
        unsqueeze = lambda x: x.unsqueeze(-1)
    else:
        unsqueeze = lambda x: np.expand_dims(x, axis=-1)

    arr0 = arr
    if mask is not None:
        arr0 = arr0 * mask
    mean = arr0.mean(axis=axis)
    if isinstance(axis, float):
        axis = (axis,)
    for _ in axis:
        mean = unsqueeze(mean)

    return arr - mean


def plot_means_errs(
        list_of_means, list_of_stds, list_of_methods, xticklabels=None,
        rotation_xticklabels=45, sec_xticklabels=None,
        xlabel=None, ylabel=None, drawtarget=True, alpha=None, drawbounds=True, offset=0.15,
        y_lower=None, y_upper=None, logscale=False, ymin=None, ymax=None,
        figsize=(6, 3), savefig=False, filepath=None, filename=None, extension=None,
        args_legend_main=None, args_legend_other=None
):
    """
    Plot means with error bars representing standard deviations
    
    """
    nseries = len(list_of_means)
    assert len(list_of_stds) == nseries
    if xticklabels is not None:
        nvals = len(xticklabels)
    else:
        nvals = 1

    _, ax = plt.subplots(figsize=figsize)
    handles_main = []
    for i, (means, stds, label) in enumerate(zip(list_of_means, list_of_stds, list_of_methods)):
        x_values = np.arange(nvals) + 1 + (i - (nseries - 1) / 2) * offset  # Adjusted x-coordinates
        means = np.array(means)
        stds = np.array(stds)
        mask = means != None # Array of booleans. Do not use `means is not None` as it returns False
        handle = plt.errorbar(
            x_values[mask], means[mask], yerr=stds[mask], fmt='.', capsize=3, label=label
        )
        handles_main.append(handle)
    if xticklabels is not None:
        plt.xticks(np.arange(nvals) + 1, xticklabels, rotation=rotation_xticklabels)
    else:
        plt.xticks([])
    ax.set_xlim(0.5, nvals + 0.5)

    if sec_xticklabels is not None:
        sec_nvals = len(sec_xticklabels)
        sec_xticks = nvals / sec_nvals * np.arange(sec_nvals) + (nvals / sec_nvals + 1) / 2
        sec = ax.secondary_xaxis(location=0)
        sec.set_xticks(sec_xticks, labels=sec_xticklabels)
        sec.tick_params('x', length=0)

        # lines between the classes:
        sec_xticks = nvals / sec_nvals * np.arange(sec_nvals + 1) + 0.5
        sec2 = ax.secondary_xaxis(location=0)
        sec2.set_xticks(sec_xticks, labels=[])
        sec2.tick_params('x', length=50, width=1)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    handles_other = []
    if drawtarget:
        handle = plt.axhline(
            y=alpha, color='red', linestyle='--',
            linewidth=0.8, label=r'$\alpha$ (target)'
        )
        handles_other.append(handle)
    if drawbounds:
        handle = plt.axhspan(
            y_lower, y_upper,
            color='yellow', alpha=0.3, linewidth=0., label="Theoretical bounds"
        )
        handles_other.append(handle)
    if args_legend_main is None:
        args_legend_main = {}
    if args_legend_other is None:
        args_legend_other = {}
    legend_main = plt.legend(handles=handles_main, **args_legend_main)
    plt.gca().add_artist(legend_main)
    if handles_other != []:
        plt.legend(handles=handles_other, **args_legend_other)
    if logscale:
        plt.yscale('log')
    kwargs = {}
    if ymin is not None:
        kwargs.update(bottom=ymin)
    if ymax is not None:
        kwargs.update(top=ymax)
    if kwargs != {}:
        plt.ylim(**kwargs)
    if savefig:
        plt.savefig(os.path.join(filepath, f"{filename}.{extension}"), bbox_inches='tight')

    plt.show()


def skyshow(
        img, boundaries=None, c='w', cbarshrink=None, title=None,
        printcolorbar=True, printxylabels=True,
        printxticks=True, printyticks=True,
        imgsize: int | tuple[int] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        extent_after_crop: tuple[float, float, float, float] | None = None,
        xclus: bool = False, path_to_xclus: str = PATH_TO_XCLUS,
        zmin: float = XCLUS_ZMIN, zmax: float = XCLUS_ZMAX,
        m500min: float = XCLUS_M500MIN,
        ztext: bool = XCLUS_ZTEXT, mtext: bool = XCLUS_MTEXT,
        **kwargs
):
    if imgsize is not None:
        if isinstance(imgsize, int):
            imgsize = (imgsize, imgsize)
        imgsize_ori = img.shape
        beg_i = (imgsize_ori[0] - imgsize[0]) // 2
        beg_j = (imgsize_ori[1] - imgsize[1]) // 2
        end_i = beg_i + imgsize[0]
        end_j = beg_j + imgsize[1]
        img = crop_arr(img, beg_i, end_i, beg_j, end_j)
        if extent_after_crop is not None:
            extent = extent_after_crop
        elif extent is not None:
            x_min, x_max, y_min, y_max = extent
            dx = (x_max - x_min) / imgsize_ori[1]
            dy = (y_max - y_min) / imgsize_ori[0]
            extent = (
                x_min + beg_j * dx,
                x_min + end_j * dx,
                y_min + beg_i * dy,
                y_min + end_i * dy,
            )

    out = plt.imshow(img, origin='lower', extent=extent, **kwargs)

    plt.xlim(plt.gca().get_xlim()[::-1]) # Flip x-axis (sky observations: east left)
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

    # Xray clusters
    # This section is a copy-paste from the `cosmostat` repository
    # https://github.com/CosmoStat/cosmostat.git
    if xclus:
        xclusters = np.loadtxt(path_to_xclus)
        highz = (xclusters[:, 6] >= zmin) & (xclusters[:, 6] <= zmax)
        for cluster in xclusters[highz]:
            ra_cl, dec_cl, z_cl = cluster[1], cluster[2], cluster[6]
            m500 = cluster[7]
            if m500 > m500min:
                plt.scatter(ra_cl, dec_cl, c="w", s=6)
                if ztext:
                    plt.text(
                        ra_cl + 0.03,
                        dec_cl + 0.02,
                        "{:.2f}".format(z_cl),
                        fontsize=8,
                        c="w",
                    )
                if mtext:
                    plt.text(
                        ra_cl + 0.03,
                        dec_cl - 0.02,
                        "{:.2f}".format(m500),
                        fontsize=8,
                        c="w",
                    )

    return out


class KappamapVisualizer:

    def __init__(
            self, kappa_inp=None, kappa_true=None, kappa_pred=None, var=None, res_pred=None,
            extent=None, boundaries=None, mask=None, imgsize=None,
            vmin=None, vmax=None, vmax_sqdiff=None, vmax_bounds=None,
            plot_colorbar=False,
    ):
        self.kappa_inp = kappa_inp
        self.kappa_true = kappa_true
        self.kappa_pred = kappa_pred
        self.var = var
        self.res_pred = res_pred
        self.extent = extent
        self.boundaries = boundaries
        self.mask = mask
        self.imgsize = imgsize
        self.vmin = vmin
        self.vmax = vmax
        self.vmax_sqdiff = vmax_sqdiff
        self.vmax_bounds = vmax_bounds
        self.plot_colorbar = plot_colorbar


    def bounds(self, res_pred=None):
        if res_pred is None:
            res_pred = self.res_pred
        lowerbound = self.kappa_pred - res_pred - self.kappa_true
        upperbound = self.kappa_pred + res_pred - self.kappa_true
        if self.mask is not None:
            lowerbound *= self.mask
            upperbound *= self.mask
        return lowerbound, upperbound


    def skyshow_kappamap(self, kappa, **kwargs):

        if torch.is_tensor(kappa):
            kappa = kappa.cpu().numpy()

        out = skyshow(
            kappa, vmin=self.vmin, vmax=self.vmax, extent=self.extent,
            boundaries=self.boundaries, printxylabels=False,
            printxticks=False, printyticks=False, printcolorbar=False,
            imgsize=self.imgsize, **kwargs
        )
        if self.plot_colorbar:
            plt.colorbar()

        return out

    def skyshow_inp(self, **kwargs):
        return self.skyshow_kappamap(self.kappa_inp, **kwargs)

    def skyshow_truth(self, **kwargs):
        return self.skyshow_kappamap(self.kappa_true, **kwargs)

    def skyshow_pointestimate(self, **kwargs):
        return self.skyshow_kappamap(self.kappa_pred, **kwargs)


    def skyshow_variance(self, showstd=False, **kwargs):

        img = self.var
        vmax = self.vmax_sqdiff
        if showstd:
            img = img**0.5
            vmax = vmax**0.5
        if torch.is_tensor(img):
            img = img.cpu().numpy()

        skyshow(
            img, vmin=0., vmax=vmax, extent=self.extent,
            boundaries=self.boundaries, printxylabels=False,
            printxticks=False, printyticks=False, printcolorbar=True,
            imgsize=self.imgsize, **kwargs
        )


    def skyshow_bound(self, which, res_pred=None, **kwargs):

        lower_bound, upper_bound = self.bounds(res_pred=res_pred)
        if which == "lower":
            bound = lower_bound
        elif which == "upper":
            bound = upper_bound
        else:
            raise ValueError("Argument `which` must be either 'lower' or 'upper'")
        if torch.is_tensor(bound):
            bound = bound.cpu().numpy()

        out = skyshow(
            bound,
            cmap="coolwarm", vmin=-self.vmax_bounds, vmax=self.vmax_bounds,
            extent=self.extent, boundaries=self.boundaries,
            printcolorbar=False, printxylabels=False, printxticks=False, printyticks=False,
            imgsize=self.imgsize, **kwargs
        )
        if self.plot_colorbar:
            cbar = plt.colorbar()
            cbar.set_ticks(np.linspace(-.2, .2, 5))

        return out


    def visualize(self, **kwargs):
        raise NotImplementedError


class KappamapVisualizerCompact(KappamapVisualizer):

    def visualize(self, showstd: bool = False, **kwargs):
        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        self.skyshow_pointestimate(title="Point estimate", **kwargs)
        plt.subplot(222)
        self.skyshow_variance(title="Std estimate", showstd=showstd, **kwargs)
        plt.subplot(223)
        self.skyshow_bound("lower", title=f"Lower bound", **kwargs)
        plt.subplot(224)
        self.skyshow_bound("upper", title=f"Upper bound", **kwargs)
        plt.show()


class KappamapVisualizerSavefig(KappamapVisualizer):

    def __init__(
            self, *args, savefig=False, save_dir=None, extension=None,
            showinp=True, showtruth=True, showpred=True, showbounds=True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.savefig = savefig
        self.save_dir = save_dir
        self.extension = extension
        self.showinp = showinp
        self.showtruth = showtruth
        self.showpred = showpred
        self.showbounds = showbounds


    def visualize(
            self, title: str | None = None, filename: str | None = None,
            showvar: bool = False, showstd: bool = False, **kwargs
    ):
        if self.showinp:
            plt.figure(figsize=(5, 3))
            self.skyshow_inp(title=title, **kwargs)
            if self.savefig:
                plt.savefig(
                    os.path.join(self.save_dir, f"{filename}_inp.{self.extension}"),
                    bbox_inches='tight'
                )
            plt.show()

        if self.showtruth:
            plt.figure(figsize=(5, 3))
            self.skyshow_truth(title=title, **kwargs)
            if self.savefig:
                plt.savefig(
                    os.path.join(self.save_dir, f"{filename}_true.{self.extension}"),
                    bbox_inches='tight'
                )
            plt.show()

        if self.showpred:
            plt.figure(figsize=(5, 3))
            self.skyshow_pointestimate(title=title, **kwargs)
            if self.savefig:
                plt.savefig(
                    os.path.join(self.save_dir, f"{filename}_pred.{self.extension}"),
                    bbox_inches='tight'
                )
            plt.show()

        if showvar or showstd:
            plt.figure(figsize=(5, 3))
            self.skyshow_variance(title=title, showstd=showstd, **kwargs)
            if self.savefig:
                if showstd:
                    suffix = "std"
                else:
                    suffix = "var"
                plt.savefig(
                    os.path.join(self.save_dir, f"{filename}_{suffix}.{self.extension}"),
                    bbox_inches='tight'
                )
            plt.show()

        if self.showbounds:
            plt.figure(figsize=(5, 3))
            self.skyshow_bound("lower", title=title, **kwargs)
            if self.savefig:
                plt.savefig(
                    os.path.join(self.save_dir, f"{filename}_low.{self.extension}"),
                    bbox_inches='tight'
                )
            plt.show()

            plt.figure(figsize=(5, 3))
            self.skyshow_bound("upper", title=title, **kwargs)
            if self.savefig:
                plt.savefig(
                    os.path.join(self.save_dir, f"{filename}_high.{self.extension}"),
                    bbox_inches='tight'
                )

        plt.show()


def get_sup_step_size(
        param_mahalanobis: float | torch.Tensor, its=ITS_POWER_ITERATION,
        dims: tuple[int, ...] | None = None,
        physics=None, device: str | torch.device = "cpu"
):
    """
    Get the upper bound for the step size in PGD algorithms where the data
    fidelity term is the MSE using the Mahalanobis norm.
    This function uses the power iteration method.

    Parameters
    ----------
    param_mahalanobis: float or torch.Tensor
        SPD matrix for the Mahalanobis norm (std_noise**2 for the negative log-likelihood,
        std_noise for the noise-whitening data fidelity)
      if torch.is_tensor(param_mahalanobis):      The noise model is not used for this function. If none is given,
        then the identity is used.
    device: str, optional
        Device to which `physics` is stored. Default is "cpu"
    """
    # TODO: retrieve `param_mahalanobis` from `physics`
    if torch.is_tensor(param_mahalanobis):
        param_mahalanobis = param_mahalanobis.to(device)
        dims = param_mahalanobis.shape
        nelts = param_mahalanobis.numel()
    else:
        assert dims is not None
        nelts = math.prod(dims)

    if physics is None:
        physics = dinv.physics.LinearPhysics().to(device) # Identity

    def matvec(kappa_flattened):
        kappa = kappa_flattened.reshape(*dims)
        kappa = torch.tensor(
            kappa, dtype=torch.float32, device=device
        )
        gamma = physics.A(kappa)
        gamma /= param_mahalanobis
        out = physics.A_adjoint(gamma)
        return out.cpu().numpy().astype(np.float64).flatten()

    linearop = sparse.linalg.LinearOperator(
        shape=(nelts, nelts), matvec=matvec, rmatvec=matvec
    )
    spectrnorm = linalg.interpolative.estimate_spectral_norm(linearop, its=its)

    return 2 / spectrnorm


def get_g_param(std_noise, noise_whitening):

    if not noise_whitening:
        g_param = std_noise**2 # Negative log-likelihood as data fidelity
    else:
        g_param = std_noise # Noise-whitening data fidelity

    return g_param


def infer_model(
        model, dataloader, idx_list=None, idx_dict=None,
        device='cpu', verbose=False, **kwargs
):
    model.eval().to(device)
    outputs = []
    with torch.no_grad():
        for inp in tqdm.tqdm(dataloader, disable=not verbose):

            if torch.is_tensor(inp):
                inp = (inp,)
            inp = tuple(x.to(device) for x in inp)
            if idx_list is None:
                idx_list = range(len(inp))
            args = tuple(inp[idx] for idx in idx_list)
            if idx_dict is not None:
                kwargs.update({
                    key: inp[idx] for key, idx in idx_dict.items()
                })
            outputs.append(model(*args, **kwargs))

        out = cat_arrays(outputs, dim=0)

    return out


def cat_arrays(inp_list, **kwargs):

    first_elt = inp_list[0]
    if torch.is_tensor(first_elt):
        out = torch.cat(inp_list, **kwargs)
    elif isinstance(first_elt, list):
        out = [
            cat_arrays([
                l[i] for l in inp_list
            ], **kwargs) for i in range(len(first_elt))
        ]
    elif isinstance(first_elt, tuple):
        out = tuple(
            cat_arrays([
                t[i] for t in inp_list
            ], **kwargs) for i in range(len(first_elt))
        )
    elif isinstance(first_elt, dict):
        out = {
            key: cat_arrays([
                d[key] for d in inp_list
            ], **kwargs) for key in first_elt.keys()
        }
    else:
        raise TypeError(
            f"Unsupported input type: {type(first_elt)}. "
            "Expected torch.Tensor, list, tuple or dict."
        )
    return out


def get_timestamp():
    return datetime.now().strftime(r"%Y%m%d_%H%M%S")


def get_weights_redshifts(
        vals: np.ndarray, zplanes: np.ndarray,
        max_z: float | None = None,
        weights: np.ndarray | None = None
) -> np.ndarray:
    """
    Arguments
    ---------
    vals: np.ndarray, shape = (ngals,)
        List of redshifts, for each measured galaxy
    zplanes: np.ndarray, shape = (nplanes,)
        List of redshift planes
    max_z: float, optional
    weights: np.ndarray, shape = (ngals,), optional
        An array of weights, of the same shape as vals. Each value in vals
        only contributes its associated weight towards the bin count (instead of 1).

    Returns
    -------
    out: np.ndarray, shape = (nplanes,)
        The corresponding source distribution on zplanes; sums to one
    """
    if max_z is None:
        max_z = float(np.max(vals))
    bins = (zplanes[:-1] + zplanes[1:]) / 2
    bins = np.concatenate([[0.], bins, [max_z]])
    dbins = bins[1:] - bins[:-1]
    hist, _ = np.histogram(vals, bins=bins, weights=weights, density=True)

    return dbins * hist


#=================================================================================
# Functions on torch tensors or numpy arrays
#=================================================================================

def absolute(arr: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(arr):
        out = torch.abs(arr)
    else:
        out = np.abs(arr)
    return out


def quantile(
        inp: np.ndarray | torch.Tensor, q, axis=None, **kwargs
) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(inp):
        out = torch.quantile(inp, q, dim=axis, **kwargs)
    else:
        out = np.quantile(inp, q, axis=axis, **kwargs)
    return out


def _min_max(
        which: str,
        inp: np.ndarray | torch.Tensor,
        other: float | np.ndarray | torch.Tensor,
        *args, **kwargs
) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(inp):
        fn = getattr(torch, which)
        if not torch.is_tensor(other):
            other = other * torch.ones_like(inp)
    else:
        fn = getattr(np, which)
    return fn(inp, other, *args, **kwargs)

def maximum(
        inp: np.ndarray | torch.Tensor,
        other: float | np.ndarray | torch.Tensor,
        *args, **kwargs
) -> np.ndarray | torch.Tensor:
    return _min_max(
        'maximum', inp, other, *args, **kwargs
    )

def minimum(
        inp: np.ndarray | torch.Tensor,
        other: np.ndarray | torch.Tensor,
        *args, **kwargs
) -> np.ndarray | torch.Tensor:
    return _min_max(
        'minimum', inp, other, *args, **kwargs
    )


def count_elts(arr: np.ndarray | torch.Tensor) -> int:
    if torch.is_tensor(arr):
        out = arr.numel()
    else:
        out = arr.size
    return out


def pad(
        arr: np.ndarray | torch.Tensor, pad_width: tuple[int, int, int, int],
        **kwargs
) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(arr):
        out = F.pad(arr, pad_width, **kwargs)
    else:
        raise NotImplementedError
    return out


def fft2(
        arr: np.ndarray | torch.Tensor, **kwargs
) -> np.ndarray | torch.Tensor:
    if torch.is_tensor(arr):
        out = torch.fft.fft2(arr, **kwargs)
    else:
        out = np.fft.fft2(arr, **kwargs)
    return out


def load_checkpoint_state_dict(
        filename: str,
        key_replacement_dict: dict[str, str] | None = KEY_REPLACEMENT_DICT,
        verbose: bool = False
) -> dict:
    checkpoint = torch.load(filename, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    if key_replacement_dict is not None:
        for old_key, new_key in key_replacement_dict.items():
            if old_key in state_dict:
                if verbose:
                    print(f"Replacing key '{old_key}' with '{new_key}'")
                state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


def get_list_per_zbin(
        inp: np.ndarray, z: np.ndarray, zbins: list[float] | None = None
) -> list[np.ndarray]:
    if zbins is not None:
        idx_per_bin = np.searchsorted(z, zbins)
    else:
        idx_per_bin = np.array([])
    return np.split(inp, idx_per_bin)


def get_zbins(
        path_to_zbins: str | None = PATH_TO_ZBINS,
        idx_zbins: list[int] | None = None
) -> list[float]:
    if path_to_zbins is None:
        raise ValueError("Argument `path_to_zbins` must be provided.")
    hdul = apfits.open(path_to_zbins)
    zbins = hdul[1].data["BIN_STOP"]
    assert isinstance(zbins, np.ndarray)
    zbins = zbins[:-1] # Exclude the upper limit
    if idx_zbins is not None:
        zbins = zbins[idx_zbins]
    return zbins.tolist()


def get_mask_onezbin(mask: torch.Tensor):
    return torch.sum(mask, dim=0).bool()


def get_tensor_components(x):
    # Shape of x: (batch_size, 2, nchannels, nx, ny)
    return x[:, 0], x[:, 1]


def stack_tensor_components(x_g, x_ng):
    # Shape of x_g and x_ng: (batch_size, nchannels, nx, ny)
    return torch.stack((x_g, x_ng), dim=1)


def add_tensor_components(x):
    # Shape of x: (batch_size, 2, nchannels, nx, ny)
    return torch.sum(x, dim=1)


def get_cdist(
        z: np.ndarray, z_sup: float | None,
        c: float, h0: float,
        omega_m: float, omega_lambda: float
) -> np.ndarray:

    if z_sup is None:
        z_sup = np.inf
    z_bounds = np.concatenate((
        [0.0], (z[:-1] + z[1:]) / 2, [z_sup]
    )) # Shape = (nz + 1,)
    dz = z_bounds[1:] - z_bounds[:-1] # Shape = (nz,)
    h = get_hubble_param(
        z, h0=h0, omega_m=omega_m, omega_lambda=omega_lambda
    ) # Shape = (nz,)
    nz = len(z)
    triang = np.tril(np.ones((nz, nz))) # Shape = (nz, nz)
    out = c * np.sum(triang * dz / h, axis=1) # Shape = (nelts,)

    return out


def get_hubble_param(
        z: np.ndarray, h0: float,
        omega_m: float, omega_lambda: float
):
    return h0 * np.sqrt(
        omega_m * (1 + z)**3 + omega_lambda
    )


class ComponentWrapper:
    """
    Wrapper class to hold Gaussian and non-Gaussian components.
    This is used instead of a tuple to avoid being considered
    as an iterable by the optimizers.
    """
    def __init__(self, val_g, val_ng):
        self.g = val_g
        self.ng = val_ng

    def get_components(self):
        return self.g, self.ng

    def __str__(self):
        return f"{self.get_components()}"

    def __repr__(self):
        val_g, val_ng = self.get_components()
        return f"ComponentWrapper({val_g}, {val_ng})"


class ModuleWrapper(nn.Module):
    """
    Wrapper class to hold Gaussian and non-Gaussian components of type
    `optim.BaseOptim` (e.g., data fidelity or prior).
    This is used instead of `torch.nn.ModuleDict` to avoid being considered
    as an iterable by the optimizers.
    """
    def __init__(
            self,
            module_g: nn.Module | None,
            module_ng: nn.Module | None
    ):
        super().__init__()
        self.g = module_g
        self.ng = module_ng

    def get_components(self):
        return self.g, self.ng


def merge_dict(d_g, d_ng):
    d = {}
    for k in d_g.keys() | d_ng.keys():
        d[k] = ComponentWrapper(d_g.get(k), d_ng.get(k))
    return d


def unmerge_dict(d):
    d_g = {}
    d_ng = {}
    for k, v in d.items():
        if isinstance(v, ComponentWrapper):
            d_g[k], d_ng[k] = v.get_components()
        else:
            d_g[k] = d_ng[k] = v
    return d_g, d_ng


def get_std_gaussian(fwhm, resolution):
    """
    Compute standard deviation of a Gaussian distribution in pixel unit,
    from the full width at half maximum (FWHM).
    
    :param fwhm: FWHM (arcmin)
    :param resolution: Resolution (arcmin per pixel)
    """
    std_gaussian_arcmin = fwhm / (2 * np.sqrt(2 * np.log(2)))
    std_gaussian = std_gaussian_arcmin / resolution
    return std_gaussian
