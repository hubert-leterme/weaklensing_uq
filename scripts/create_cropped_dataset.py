import argparse

import _commons
import _add_arguments

from _commons import NINPIMGS

import wlmmuq
import wlmmuq.datasets.kappatng as wlktng

IDX_LP = "001" # Lensing potential used for testing/calibration

def main(
        path_to_output=wlmmuq.PATH_TO_TEST_DATASET,
        idx_lp=IDX_LP,
        openingangle=wlktng.OPENINGANGLE, ninpimgs=NINPIMGS,
        seed=None, verbose=False, **kwargs
):
    _commons.set_seed(seed)
    _commons.create_dataset_from_kappatng(
        wlktng.create_cropped_dataset,
        path_to_output, idx_lp, openingangle, ninpimgs,
        verbose=verbose, **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _add_arguments.create_dataset(parser, wlmmuq.PATH_TO_TEST_DATASET, IDX_LP)
    _add_arguments.seed_verbose(parser)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
