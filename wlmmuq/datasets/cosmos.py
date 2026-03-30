"""
Data available at https://archive.stsci.edu/hlsp/candels/cosmos-catalogs

"""
__level__ = 1

from dataclasses import dataclass
import numpy as np
import matplotlib.path as mpath

import astropy.table as aptable

import torch
from ..lenspack import bin2d
from .. import utils

from ..config import COSMOS_DIR

COSMOS_VERTICES = [(149.508, 2.880),
                   (149.767, 2.836),
                   (149.780, 2.887),
                   (150.040, 2.842),
                   (150.051, 2.893),
                   (150.363, 2.840),
                   (150.376, 2.890),
                   (150.746, 2.826),
                   (150.737, 2.774),
                   (150.790, 2.765),
                   (150.734, 2.449),
                   (150.787, 2.441),
                   (150.730, 2.125),
                   (150.785, 2.118),
                   (150.758, 2.013),
                   (150.768, 2.010),
                   (150.747, 1.910),
                   (150.799, 1.897),
                   (150.740, 1.580),
                   (150.481, 1.625),
                   (150.466, 1.572),
                   (150.211, 1.619),
                   (150.196, 1.567),
                   (149.887, 1.621),
                   (149.872, 1.571),
                   (149.617, 1.615),
                   (149.602, 1.566),
                   (149.493, 1.584),
                   (149.504, 1.637),
                   (149.450, 1.646),
                   (149.488, 1.855),
                   (149.433, 1.862),
                   (149.491, 2.178),
                   (149.436, 2.186),
                   (149.484, 2.445),
                   (149.431, 2.455),
                   (149.508, 2.880)]

RA, DEC = np.array(COSMOS_VERTICES).T


@dataclass
class CosmosCatalogs:
    cat_bright: aptable.Table
    cat_faint: aptable.Table
    zdist_faint: aptable.Table


@dataclass
class DataFromCosmos:
    shapedisp: float
    openingangle: float
    ra_cosmos_median: float
    dec_cosmos_median: float
    extent: tuple[float, float, float, float]

    std_noise: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    gamma: torch.Tensor | None = None


def cosmos_catalog():

    # Load catalogs
    cat_bright = aptable.Table.read(f'{COSMOS_DIR}/cosmos_bright_cat_min.asc', format='ascii')
    cat_faint = aptable.Table.read(f'{COSMOS_DIR}/cosmos_faint_cat.asc', format='ascii')

    # Discard galaxies with redshift measurement problem (zphot < 0.6 and i+ > 24)
    # For more details, see B. Remy et al., “Probabilistic mass-mapping with neural score 
    # estimation,” A&A, vol. 672, p. A51, Apr. 2023.
    cat_bright = cat_bright[cat_bright['z_problem'] == 0]

    # Load source redshift distribution for the faint catalog
    # We use the weighted distribution ("w1")
    zdist_faint = aptable.Table.read(f'{COSMOS_DIR}/cosmos_zdist_faint_w1.asc', format='ascii')

    return CosmosCatalogs(cat_bright=cat_bright, cat_faint=cat_faint, zdist_faint=zdist_faint)


def get_extent(ra_cosmos_median, dec_cosmos_median, openingangle):
    extent = (
        ra_cosmos_median - openingangle/2, ra_cosmos_median + openingangle/2,
        dec_cosmos_median - openingangle/2, dec_cosmos_median + openingangle/2
    )
    return extent


def get_data_from_cosmos(
        cat_cosmos: aptable.Table, imgsize: int, resolution: float,
        metadata_only: bool = False, get_noisy_shear_map: bool = False,
        east_right: bool = False,
        zbins: list[float] | None = None, max_z: float | None = None
) -> DataFromCosmos:

    shapedisp1 = np.std(np.array(cat_cosmos['e1iso_rot4_gr_snCal']))
    shapedisp2 = np.std(np.array(cat_cosmos['e2iso_rot4_gr_snCal']))
    shapedisp = float((shapedisp1 + shapedisp2) / 2)

    openingangle = utils.get_openingangle(imgsize, resolution)
    ra_cosmos_median = float(np.median(np.array(cat_cosmos['Ra']))) # right ascension (longitude)
    dec_cosmos_median = float(np.median(np.array(cat_cosmos['Dec']))) # declination (latitude)
    extent = get_extent(ra_cosmos_median, dec_cosmos_median, openingangle)

    out = DataFromCosmos(
        shapedisp=shapedisp,
        openingangle=openingangle,
        ra_cosmos_median=ra_cosmos_median,
        dec_cosmos_median=dec_cosmos_median,
        extent=extent
    )

    if not metadata_only:

        if max_z is None:
            max_z = np.inf
        boundaries_zbins = [0., max_z]
        if zbins is not None:
            boundaries_zbins = sorted(zbins + boundaries_zbins)

        list_of_std_noise: list[torch.Tensor] = []
        list_of_mask: list[torch.Tensor] = []
        list_of_gamma: list[torch.Tensor] = []
        for z_inf, z_sup in zip(boundaries_zbins[:-1], boundaries_zbins[1:]):
            cat_cosmos_sliced = cat_cosmos[
                (cat_cosmos["zphot"] >= z_inf) & (cat_cosmos["zphot"] < z_sup)
            ]
            e1_sliced = np.array(cat_cosmos_sliced['e1iso_rot4_gr_snCal'])
            e2_sliced = np.array(cat_cosmos_sliced['e2iso_rot4_gr_snCal'])
            ra_sliced = np.array(cat_cosmos_sliced['Ra'])
            dec_sliced = np.array(cat_cosmos_sliced['Dec'])
            nhweight_int_sliced = np.array(
                cat_cosmos_sliced['nhweight_int']
            ) # Similar to Hoekstra et al. 1998 (see jax-lensing)
            l2norm_nhweight_int = bin2d(
                ra_sliced, dec_sliced,
                v=nhweight_int_sliced**2,
                npix=imgsize, extent=extent,
                sum_instead_of_average=True
            )**0.5
            sum_nhweight_int = bin2d(
                ra_sliced, dec_sliced,
                v=nhweight_int_sliced,
                npix=imgsize, extent=extent,
                sum_instead_of_average=True
            )
            std_noise = np.nan_to_num(
                shapedisp * l2norm_nhweight_int / sum_nhweight_int,
                posinf=0.
            )
            mask = sum_nhweight_int > 0

            list_of_std_noise.append(torch.tensor(std_noise, dtype=torch.float32))
            list_of_mask.append(torch.tensor(mask, dtype=torch.bool))

            if get_noisy_shear_map:
                if east_right:
                    e2_sliced = -e2_sliced  # Use the complex conjugate of ellipticities
                                            # (East left in the COSMOS catalog)
                e1map, e2map = bin2d(
                    ra_sliced, dec_sliced, 
                    v=(e1_sliced, e2_sliced), w=nhweight_int_sliced,
                    npix=imgsize, extent=extent
                )
                list_of_gamma.append(torch.tensor(e1map + 1j * e2map, dtype=torch.complex64))

        out.std_noise = torch.stack(list_of_std_noise)
        out.mask = torch.stack(list_of_mask)
        if get_noisy_shear_map:
            out.gamma = torch.stack(list_of_gamma)
    
    return out


def cosmos_boundaries(extent, width, boundaries=None):
    """
    Create binary mask to exclude the regions outside the COSMOS boundaries.

    Parameters
    ----------
    extent (4-tuple)
        Extent of the target convergence maps (deg).
    width (int)
        Size of the target convergence maps (nb pixels).
    boundaries (list of 2-tuples)
    
    """
    if boundaries is None:
        boundaries = COSMOS_VERTICES

    # Map the COSMOS_VERTICES to pixel coordinates
    pixel_vertices = [(
        int((vertex[0] - extent[0]) / (extent[1] - extent[0]) * width),
        int((vertex[1] - extent[2]) / (extent[3] - extent[2]) * width)
    ) for vertex in boundaries]

    # Create a path from the pixel coordinates
    cosmos_path = mpath.Path(pixel_vertices)

    # Create a grid of coo_smoothedrdinates
    x, y = np.meshgrid(np.arange(width), np.arange(width))

    # Flatten the grid coordinates
    x_flat, y_flat = x.flatten(), y.flatten()

    # Stack the flattened coordinates to create an array of (x, y) pairs
    points = np.column_stack((x_flat, y_flat))

    # Check if each point is inside the defined path
    cosmos_mask = cosmos_path.contains_points(points)

    # Reshape the mask to the original grid shape
    cosmos_mask = cosmos_mask.reshape((width, width))

    # Longitude and latitude of COSMOS boundaries
    ra, dec = np.array(boundaries).T

    return cosmos_mask, ra, dec


def filter_by_redshifts(
        cat_cosmos: aptable.Table, max_z: float
) -> aptable.Table:
    cat_cosmos = cat_cosmos[
        cat_cosmos['zphot'] < max_z
    ]
    return cat_cosmos
