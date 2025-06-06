import argparse

import _commons
from _commons import OPENINGANGLE, NINPIMGS

import wlmmuq.kappatng as wlktng

def main(
        path_to_cropped_dataset, idx_lp=None,
        openingangle=OPENINGANGLE, ninpimgs=NINPIMGS,
        seed=None, verbose=False, **kwargs
):
    _commons.set_seed(seed)
    _commons.create_dataset_from_kappatng(
        wlktng.create_cropped_dataset,
        path_to_cropped_dataset, idx_lp, openingangle, ninpimgs,
        verbose=verbose, **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_cropped_dataset", type=str,
        help="Path to the cropped dataset (HDF5 file)"
    )
    _commons.add_arguments_create_dataset(parser)
    _commons.add_arguments_seed_verbose(parser)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
