# -*- coding: utf-8 -*-

# Functions adapted from the lenspack library:
# https://github.com/CosmoStat/lenspack

__level__ = 0

import numpy as np
import torch
from typing import overload, Sequence

# ===========================================================================
# Functions ks93 and ks93inv from module `inversion.py`.
# These functions can now take arrays of shape (..., nx, ny) as inputs.
# Either torch or numpy arrays can be used as input.
# ===========================================================================

def ks93(g1, g2):
    """Direct inversion of weak-lensing shear to convergence.

    This function is an implementation of the Kaiser & Squires (1993) mass
    mapping algorithm. Due to the mass sheet degeneracy, the convergence is
    recovered only up to an overall additive constant. It is chosen here to
    produce output maps of mean zero. The inversion is performed in Fourier
    space for speed.

    Parameters
    ----------
    g1, g2 : array_like
        2D input arrays corresponding to the first and second (i.e., real and
        imaginary) components of shear, binned spatially to a regular grid.

    Returns
    -------
    kE, kB : tuple of numpy arrays
        E-mode and B-mode maps of convergence.

    Raises
    ------
    AssertionError
        For input arrays of different sizes.

    See Also
    --------
    bin2d
        For binning a galaxy shear catalog.

    Examples
    --------
    >>> # (g1, g2) should in practice be measurements from a real galaxy survey
    >>> g1, g2 = 0.1 * np.random.randn(2, 32, 32) + 0.1 * np.ones((2, 32, 32))
    >>> kE, kB = ks93(g1, g2)
    >>> kE.shape
    (32, 32)
    >>> kE.mean()
    1.0842021724855044e-18

    """
    if torch.is_tensor(g1):
        lib = torch
        device = g1.device
    else:
        lib = np
        device = None

    # Check consistency of input maps
    assert g1.shape == g2.shape

    # Compute Fourier space grids
    (nx, ny) = g1.shape[-2:]
    k1, k2 = lib.meshgrid(lib.fft.fftfreq(ny), lib.fft.fftfreq(nx))
    if lib == torch:
        k1 = k1.to(device)
        k2 = k2.to(device)

    # Compute Fourier transforms of g1 and g2
    g1hat = lib.fft.fft2(g1)
    g2hat = lib.fft.fft2(g2)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[0, 0] = 1  # avoid division by 0
    kEhat = (p1 * g1hat + p2 * g2hat) / k2
    kBhat = -(p2 * g1hat - p1 * g2hat) / k2

    # Transform back to pixel space
    kE = lib.fft.ifft2(kEhat).real
    kB = lib.fft.ifft2(kBhat).real

    return kE, kB


def ks93inv(kE, kB):
    """Direct inversion of weak-lensing convergence to shear.

    This function provides the inverse of the Kaiser & Squires (1993) mass
    mapping algorithm, namely the shear is recovered from input E-mode and
    B-mode convergence maps.

    Parameters
    ----------
    kE, kB : array_like
        2D input arrays corresponding to the E-mode and B-mode (i.e., real and
        imaginary) components of convergence.

    Returns
    -------
    g1, g2 : tuple of numpy arrays
        Maps of the two components of shear.

    Raises
    ------
    AssertionError
        For input arrays of different sizes.

    See Also
    --------
    ks93
        For the forward operation (shear to convergence).

    """
    if torch.is_tensor(kE):
        lib = torch
        device = kE.device
    else:
        lib = np
        device = None

    # Check consistency of input maps
    assert kE.shape == kB.shape

    # Compute Fourier space grids
    (nx, ny) = kE.shape[-2:]
    k1, k2 = lib.meshgrid(lib.fft.fftfreq(ny), lib.fft.fftfreq(nx))
    if lib == torch:
        k1 = k1.to(device)
        k2 = k2.to(device)

    # Compute Fourier transforms of kE and kB
    kEhat = lib.fft.fft2(kE)
    kBhat = lib.fft.fft2(kB)

    # Apply Fourier space inversion operator
    p1 = k1 * k1 - k2 * k2
    p2 = 2 * k1 * k2
    k2 = k1 * k1 + k2 * k2
    k2[0, 0] = 1  # avoid division by 0
    g1hat = (p1 * kEhat - p2 * kBhat) / k2
    g2hat = (p2 * kEhat + p1 * kBhat) / k2

    # Transform back to pixel space
    g1 = lib.fft.ifft2(g1hat).real
    g2 = lib.fft.ifft2(g2hat).real

    return g1, g2


# ===========================================================================
# Function bin2d from module `utils.py`
# There is now an option to get the sum or the weighted average.
# Originally, only the weighted average was possible
# ===========================================================================

@overload
def bin2d(
        x: np.ndarray, y: np.ndarray, npix: int = 10,
        v: None = None,
        w: np.ndarray | None = None,
        extent: Sequence[float] | np.ndarray | None = None,
        sum_instead_of_average: bool = False, verbose: bool = False
) -> np.ndarray: ...


@overload
def bin2d(
        x: np.ndarray, y: np.ndarray, npix: int = 10,
        v: np.ndarray = ...,  # single array to bin
        w: np.ndarray | None = None,
        extent: Sequence[float] | np.ndarray | None = None,
        sum_instead_of_average: bool = False, verbose: bool = False
) -> np.ndarray: ...


@overload
def bin2d(
        x: np.ndarray, y: np.ndarray, npix: int = 10,
        v: tuple[np.ndarray, ...] = ...,  # one or more arrays to bin
        w: np.ndarray | None = None,
        extent: Sequence[float] | np.ndarray | None = None,
        sum_instead_of_average: bool = False, verbose: bool = False
) -> tuple[np.ndarray, ...]: ...


def bin2d(
        x: np.ndarray, y: np.ndarray, npix: int = 10,
        v: tuple[np.ndarray, ...] | np.ndarray | None = None,
        w: np.ndarray | None = None,
        extent: Sequence[float] | np.ndarray | None = None,
        sum_instead_of_average: bool = False, verbose: bool = False
) -> tuple[np.ndarray, ...] | np.ndarray:
    """Bin samples of a spatially varying quantity according to position.

    The sum or (weighted) average is taken of values falling into the same bin. This
    function is relatively general, but it is mainly used within this package
    to produce maps of the two components of shear from a galaxy catalog.

    Parameters
    ----------
    x, y : array_like
        1D position arrays.
    npix : int or list or tuple as (nx, ny), optional
        Number of bins in the `x` and `y` directions. If an int N is given,
        use (N, N). Binning defaults to (10, 10) if not provided.
    v : array_like, optional
        Values at positions (`x`, `y`). This can be given as many arrays
        (v1, v2, ...) of len(`x`) to bin simultaneously. If None, the bin
        count in each pixel is returned.
    w : array_like, optional
        Weights for `v` during averaging. If provided, the same weights are
        applied to each input `v`.
    extent : array_like, optional
        Boundaries of the resulting grid, given as (xmin, xmax, ymin, ymax).
        If None, bin edges are set as the min/max coordinate values of the
        input position arrays.
    sum_instead_of_average: boolean, optional
        If True, and if `v` is provided, then compute the sum of values within
        each bin, instead of the weighted average as originally intended by
        this function.
    verbose : boolean, optional
        If True, print details of the binning.

    Returns
    -------
    ndarray or tuple of ndarray
        2D numpy arrays of values `v` binned into pixels. The number of
        outputs matches the number of input `v` arrays.

    Examples
    --------
    >>> # 100 values at random positions within the ranges -0.5 < x, y < 0.5
    >>> # and binned within -1 < x, y < 1 to a (5, 5) grid.
    >>> x = np.random.random(100) - 0.5
    >>> y = np.random.random(100) - 0.5
    >>> v = np.random.randn(100) * 5
    >>> bin2d(x, y, v=v, npix=5, extent=(-1, 1, -1, 1))
    array([[ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  4.43560619, -2.33308373,  0.48447844,  0.        ],
           [ 0.        ,  1.94903524, -0.29253335,  1.3694618 ,  0.        ],
           [ 0.        , -1.0202718 ,  0.37112266, -1.43062585,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])

    """
    # Regroup extent if necessary
    if extent is not None:
        assert len(extent) == 4
        rg = [extent[:2], extent[2:]]
    else:
        rg = None

    if v is None:
        # Return the simple bin count map
        bincount, xbins, ybins = np.histogram2d(x, y, bins=npix, range=rg)
        result = bincount.T
    else:
        # Prepare values to bin
        v = np.atleast_1d(v)
        if len(v.shape) == 1:
            v = v.reshape(1, len(v))

        # Prepare weights
        if w is not None:
            w = np.atleast_1d(w)
            has_weights = True
        else:
            w = np.ones_like(x)
            has_weights = False

        # Compute weighted bin count map
        wmap, xbins, ybins = np.histogram2d(x, y, bins=npix, range=rg,
                                            weights=w)
        # Handle division by zero (i.e., empty pixels)
        wmap[wmap == 0] = np.inf

        # Compute weighted sum
        result = tuple((np.histogram2d(x, y, bins=npix, range=rg,
                        weights=(vv * w))[0]) for vv in v)

        if not sum_instead_of_average:
            # Compute mean values per pixel
            result = tuple(res / wmap for res in result)

        # Transpose
        result = tuple(res.T for res in result)

        # Clean up
        if len(result) == 1:
            result = result[0]

    if verbose:
        if v is not None:
            print("Binning {} array{} with{} weights.".format(len(v),
                  ['', 's'][(len(v) > 1)], ['out', ''][has_weights]))
        else:
            print("Returning bin count map.")
        print("npix : {}".format(npix))
        print("extent : {}".format([xbins[0], xbins[-1], ybins[0], ybins[-1]]))
        print("(dx, dy) : ({}, {})".format(xbins[1] - xbins[0],
                                           ybins[1] - ybins[0]))

    return result
