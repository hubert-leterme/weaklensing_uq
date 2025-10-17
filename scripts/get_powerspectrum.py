import argparse
import torch

import wlmmuq
import wlmmuq.utils as wlutils
import wlmmuq.data.torch as wlbl

from wlmmuq.data import NUM_WORKERS

import _commons
import _add_arguments

from _commons import IMGSIZE, BATCH_SIZE

NIMGS = 2048

def main(
        path_to_train_dataset=wlmmuq.PATH_TO_TRAIN_VAL_DATASET,
        path_to_output=wlmmuq.PATH_TO_PS,
        imgsize=IMGSIZE, nimgs=NIMGS,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        seed=None, verbose=False
):
    _commons.set_seed(seed)
    device = _commons.get_device(verbose=verbose)
    powerspectrum = get_powerspectrum_from_dataset(
        path_to_train_dataset, nimgs=nimgs,
        batch_size=batch_size, output_shape=imgsize,
        num_workers=num_workers, device=device, verbose=verbose
    ).cpu()
    torch.save(powerspectrum, path_to_output)


def get_powerspectrum_from_dataset(
        hdf5_filepath, nimgs, device=None, verbose=False, **kwargs
):
    if verbose:
        print(f"Compute the power spectrum of {nimgs} images")
    dataloader = wlbl.HDF5DatasetKappa(
        hdf5_filepath, nimgs=nimgs, shuffle=True, **kwargs
    ).to_dataloader()
    dataloader = iter(dataloader)

    list_of_powerspectrum = []
    while True:
        try:
            kappa_ps = next(dataloader)
        except StopIteration:
            break
        if device is not None:
            kappa_ps = kappa_ps.to(device)
        list_of_powerspectrum.append(
            wlutils.get_powerspectrum(kappa_ps)
        )
    powerspectrum = torch.stack(list_of_powerspectrum, dim=0)
    powerspectrum = torch.mean(powerspectrum, dim=0)

    return powerspectrum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-train-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the training set (HDF5 file) "
            f"Default = {wlmmuq.PATH_TO_TRAIN_VAL_DATASET}"
        )
    )
    parser.add_argument(
        "-o", "--path-to-output", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the output file (.pt) "
            f"Default = {wlmmuq.PATH_TO_PS}"
        )
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images over which to compute the power spectrum. "
            f"Default = {NIMGS}"
        )
    )
    _add_arguments.dataset(parser, batch_size=BATCH_SIZE)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
