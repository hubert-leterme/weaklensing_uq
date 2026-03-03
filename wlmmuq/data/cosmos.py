"""
Data available at https://archive.stsci.edu/hlsp/candels/cosmos-catalogs

"""
import numpy as np
import matplotlib.path as mpath

import astropy.table as aptable

import torch
from ..lenspack import bin2d

from .. import utils
from .. import COSMOS_DIR

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


def cosmos_catalog():

    # Load data
    cat_bright = aptable.Table.read(f'{COSMOS_DIR}/cosmos_bright_cat_min.asc', format='ascii')
    cat_faint = aptable.Table.read(f'{COSMOS_DIR}/cosmos_faint_cat.asc', format='ascii')

    # Discard galaxies with redshift measurement problem
    cat_bright = cat_bright[cat_bright['z_problem'] == 0]

    return cat_bright, cat_faint


def get_extent(ra_cosmos_median, dec_cosmos_median, openingangle):
    extent = [
        ra_cosmos_median - openingangle/2, ra_cosmos_median + openingangle/2,
        dec_cosmos_median - openingangle/2, dec_cosmos_median + openingangle/2
    ]
    return extent


def get_data_from_cosmos(
        cat_cosmos, imgsize, resolution, get_noisy_shear_map=False, east_right=False
):
    e1 = cat_cosmos['e1iso_rot4_gr_snCal']
    e2 = cat_cosmos['e2iso_rot4_gr_snCal']
    ra = cat_cosmos['Ra']
    dec = cat_cosmos['Dec']
    nhweight_int = cat_cosmos['nhweight_int']
    
    shapedisp1 = np.std(e1)
    shapedisp2 = np.std(e2)
    shapedisp = (shapedisp1 + shapedisp2) / 2

    openingangle = utils.get_openingangle(imgsize, resolution)
    ra_cosmos_median = np.median(ra) # right ascension (longitude)
    dec_cosmos_median = np.median(dec) # declination (latitude)
    extent = get_extent(ra_cosmos_median, dec_cosmos_median, openingangle)
    ngal = bin2d(
        ra, dec,
        npix=imgsize, extent=extent
    )
    assert isinstance(ngal, np.ndarray)
    mask = ngal > 0

    ngal = torch.tensor(ngal, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)

    out = {
        'ra_cosmos_median': ra_cosmos_median,
        'dec_cosmos_median': dec_cosmos_median,
        'extent': extent,
        'openingangle': openingangle,
        'shapedisp': shapedisp,
        'ngal': ngal,
        'mask': mask
    }

    if get_noisy_shear_map:
        if east_right:
            e2 = -e2 # Use the complex conjugate of ellipticities (East left in the COSMOS catalog)
        e1map, e2map = bin2d(
            ra, dec, 
            v=(e1, e2), w=nhweight_int,
            npix=imgsize, extent=extent
        )
        gamma = torch.tensor(e1map + 1j * e2map, dtype=torch.complex64)

        out.update({
            'gamma': gamma
        })
    
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


def filter_by_redshifts(cat_cosmos, max_z):
    cat_cosmos = cat_cosmos[
        cat_cosmos['zphot'] < max_z
    ]
    return cat_cosmos
