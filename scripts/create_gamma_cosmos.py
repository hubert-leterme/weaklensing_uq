"""Create binned ellipticity (gamma) maps from COSMOS catalogs.

This script follows the conventions in the project's COSMOS notebook:
- uses `e1iso_rot4_gr_snCal` / `e2iso_rot4_gr_snCal` as ellipticity (East left, North up)
- compute the complex conjugate of the above ellipticities, in order to follow the
    standard latitude / longitude convention used in `lenspack.bin2d`, `np.meshgrid`, and
    probably kappaTNG simulations:
        - first axis: increasing declination (South → North);
        - second axis: increasing RA (West → East).
- uses `nhweight_int` as per-galaxy weight when available
- applies the recommended quality cut `z_problem == 0` by default

The output is written as a `.py` file.
"""
import argparse
import torch

import wlmmuq
import wlmmuq.data.cosmos as wlcosmos
import wlmmuq.data.kappatng as wlktng

import _commons
import _add_arguments

def main(
        path_to_output=wlmmuq.PATH_TO_REAL_SHEARMAP,
        imgsize: int = _commons.IMGSIZE,
        max_z: float = wlktng.MAX_Z,
        resolution: float = wlktng.RESOLUTION,
        verbose: bool = False
):
    cat_cosmos, _ = wlcosmos.cosmos_catalog()
    cat_cosmos = wlcosmos.filter_by_redshifts(cat_cosmos, max_z)
    data_dict = wlcosmos.get_data_from_cosmos(
        cat_cosmos, imgsize, resolution,
        get_noisy_shear_map=True, east_right=True
    )
    gamma = data_dict["gamma"]
    torch.save(gamma, path_to_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _add_arguments.imgsize(parser)
    parser.add_argument(
        "-o", "--path-to-output", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the output file (.pt) "
            f"Default = {wlmmuq.PATH_TO_REAL_SHEARMAP}"
        )
    )
    _add_arguments.seed_verbose(parser)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
