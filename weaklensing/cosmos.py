"""
Data available at https://archive.stsci.edu/hlsp/candels/cosmos-catalogs

"""
import os
import yaml
import numpy as np
import matplotlib.path as mpath

import astropy as ap

with open('../config.yml', 'r') as file:
    CONFIG_DATA = yaml.safe_load(file)

COSMOS_DIR = os.path.expanduser(CONFIG_DATA['cosmos_dir'])

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
    cat_bright = ap.table.Table.read(f'{COSMOS_DIR}/cosmos_bright_cat_min.asc', format='ascii')
    cat_faint = ap.table.Table.read(f'{COSMOS_DIR}/cosmos_faint_cat.asc', format='ascii')

    # Discard galaxies with redshift measurement problem
    cat_bright = cat_bright[cat_bright['z_problem'] == 0]

    # Merge catalog
    cat_full = ap.table.Table()
    cat_full['Ra'] = np.concatenate([cat_bright['Ra'], cat_faint['Ra']])
    cat_full['Dec'] = np.concatenate([cat_bright['Dec'], cat_faint['Dec']])
    cat_full['e1iso_rot4_gr_snCal'] = np.concatenate(
        [cat_bright['e1iso_rot4_gr_snCal'], cat_faint['e1iso_rot4_gr_snCal']]
    )
    cat_full['e2iso_rot4_gr_snCal'] = np.concatenate(
        [cat_bright['e2iso_rot4_gr_snCal'], cat_faint['e2iso_rot4_gr_snCal']]
    )
    cat_full['nhweight_int'] = np.concatenate(
        [cat_bright['nhweight_int'], cat_faint['nhweight_int']]
    )

    return cat_full


def get_data_from_cosmos(cat_cosmos, size):
    """
    Parameters
    ----------
    cat_cosmos (astropy.Table)
    size (float)
        Opening angle of the target convergence maps (deg).

    """
    out = {}
    ra = cat_cosmos['Ra'] # right ascension (longitude)
    dec = cat_cosmos['Dec'] # declination (latitude)
    ra_cosmos_median = np.median(cat_cosmos['Ra']) # right ascension (longitude)
    dec_cosmos_median = np.median(cat_cosmos['Dec']) # declination (latitude)
    extent = [
        ra_cosmos_median - size/2, ra_cosmos_median + size/2,
        dec_cosmos_median - size/2, dec_cosmos_median + size/2
    ]

    shapedisp1 = np.std(cat_cosmos['e1iso_rot4_gr_snCal'])
    shapedisp2 = np.std(cat_cosmos['e2iso_rot4_gr_snCal'])

    out.update(
        ra=ra, dec=dec,
        ra_cosmos_median=ra_cosmos_median, dec_cosmos_median=dec_cosmos_median,
        extent=extent, shapedisp=(shapedisp1, shapedisp2)
    )

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
