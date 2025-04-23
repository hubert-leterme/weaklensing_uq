import numpy as np
from scipy import ndimage
from skimage.util.shape import view_as_windows

from . import utils

class Rotation:

    def __init__(self, angle, imgsize0):

        angle = angle % 90 # Spin-1/4 property
        angle_rad = np.deg2rad(angle)
        self.angle_rad = angle_rad
        self.imgsize0 = imgsize0
        self.rotated_imgsize_0 = int(np.round(imgsize0 * np.cos(angle_rad), 0))
        self.rotated_imgsize_1 = int(np.round(imgsize0 * np.sin(angle_rad), 0))
        self.rotated_imgsize = self.rotated_imgsize_0 + self.rotated_imgsize_1


    def _vmin_j(self, rows):

        condition = rows < self.rotated_imgsize_1

        out = np.zeros(rows.shape, dtype=int)
        out[condition] = self.rotated_imgsize_0 * (1 - rows[condition] / self.rotated_imgsize_1)
        out[~condition] = self.rotated_imgsize_1 / self.rotated_imgsize_0 * (
            rows[~condition] - self.rotated_imgsize_1
        )

        out = np.ceil(out).astype(int)

        return out


    def _vmax_j(self, rows):
        out = self.rotated_imgsize - self._vmin_j(
            self.rotated_imgsize - rows
        )
        return out


    def bounds_i(self, cropsize):

        rows = np.arange(self.rotated_imgsize)
        vmin_j = self._vmin_j(rows)
        vmax_j = self._vmax_j(rows)
        admissible_rows = np.where(vmax_j - vmin_j >= cropsize)[0]
        vmin_i = admissible_rows[0]
        vmax_i = admissible_rows[-1] + 1

        return vmin_i, vmax_i


    def bounds_j(self, cropsize, rows):

        vupperleft_j = self._vmin_j(rows)
        vupperright_j = self._vmax_j(rows)
        vlowerleft_j = self._vmin_j(rows + cropsize - 1)
        vlowerright_j = self._vmax_j(rows + cropsize - 1)

        vmin_j = utils.maximum(vupperleft_j, vlowerleft_j) # Shape = (nimgs,)
        vmax_j = utils.minimum(vupperright_j, vlowerright_j) # Shape = (nimgs,)

        return vmin_j, vmax_j


def get_patches(imgs0, rows, cols, imgsize):
    """
    Crop kappa_rot according to rows and cols

    """
    nimgs, _, _ = imgs0.shape
    assert rows.shape == (nimgs,)
    assert cols.shape == (nimgs,)

    # Get 2D sliding windows for each element
    # Shape = (nimgs, 1024-imgsize, 1024-imgsize, 1, imgsize, imgsize)
    imgs0_window = view_as_windows(imgs0, (1, imgsize, imgsize))

    # Use fancy/advanced indexing to select the required ones
    # Shape = (nimgs, imgsize, imgsize)
    out = imgs0_window[np.arange(nimgs), rows, cols, 0]

    return out


def rotate_and_crop(kappa, angle, imgsize, niter=1):

    nimgs, imgsize0, _ = kappa.shape

    # Rotate kappa
    kappa_rot = ndimage.rotate(kappa, angle, axes=(-2, -1)) # Shape = (nimgs, 1024+, 1024+)

    # Get min and max row index for random cropping
    rot = Rotation(angle, imgsize0)
    vmin_i, vmax_i = rot.bounds_i(imgsize) # Scalars

    list_of_kappa_rot_patches = []
    list_of_rows = []
    list_of_cols = [] 
    for _ in range(niter):
        # Get random row indices, for each input image
        rows = np.random.randint(vmin_i, vmax_i - imgsize + 1, size=nimgs)

        # Get min and max column indices, for each input image
        vmin_j, vmax_j = rot.bounds_j(imgsize, rows) # Shape = (nimgs,)

        # Get random column indices, for each input image
        cols = np.random.randint(vmin_j, vmax_j - imgsize + 1)

        # Crop kappa_rot according to rows and cols
        # Shape = (nimgs, imgsize, imgsize)
        kappa_rot_patches = get_patches(kappa_rot, rows, cols, imgsize)

        list_of_kappa_rot_patches.append(kappa_rot_patches)
        list_of_rows.append(rows)
        list_of_cols.append(cols)

    kappa_rot_patches = np.concatenate(list_of_kappa_rot_patches, axis=0)
    rows = np.concatenate(list_of_rows, axis=0)
    cols = np.concatenate(list_of_cols, axis=0)

    return kappa_rot_patches, rows, cols
