import argparse
import os
from tensorflow import keras

import _commons

import wlmmuq.data.tensorflow as wlbl
import wlmmuq.iterativemm.iterativemm as wlpnp

from wlmmuq import OFFSET

# TODO: Update (DeepInverse)

DATA_FIDELITY = 'noisewhitening'
STEP_SIZE = 1.4e-1
NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations in the original dataset (discarded)
NIMGS_VAL = 1440 # Remaining 2 realizations
IMGSIZE = 304
BATCH_SIZE = 32
NITER = 8

METHOD_DICT = {
    'bayesian_noprecond': wlpnp.BayesianPGDMassMappingNoPrecond,
    'bayesian_precond': wlpnp.BayesianPGDMassMappingPrecond,
    'l2': wlpnp.L2PGDMassMapping,
    'noisewhitening': wlpnp.NoisewhiteningPGDMassMapping
}

def main(
        path_to_trained_denoiser, path_to_augmented_dataset, path_to_saved_stats,
        cosmos_include_faint=False, data_fidelity=DATA_FIDELITY, step_size=STEP_SIZE,
        nimgs_train=NIMGS_TRAIN, nimgs_val=NIMGS_VAL, imgsize=IMGSIZE,
        batch_size=BATCH_SIZE, niter=NITER,
        offset=OFFSET, seed=None, verbose=False
):
    _commons.set_seed(seed)
    std_noise, mask = _commons.get_stdnoise_mask(
        imgsize, cosmos_include_faint=cosmos_include_faint,
        convert_to_torch_tensor=True, inpainting=True,
        seed=seed, verbose=verbose
    )

    # Initialize batch generator for the validation set
    val_gen = wlbl.HDF5DatasetMassMapping(
        path_to_augmented_dataset, nimgs=nimgs_val, beg_idx=nimgs_train,
        batch_size=batch_size, sort_by_filename_ori=True, shuffle=False,
        std_noise=std_noise, mask=mask, inpainting=True,
        output_shape=imgsize
    )

    # Load the trained denoiser
    if verbose:
        print("Load the trained denoiser")
    model = keras.models.load_model(os.path.join(path_to_trained_denoiser))
    backward = wlpnp.KerasDenoiser(model, offset=offset)

    # Instantiate the PnP class and RMSE callback
    pnp = METHOD_DICT[data_fidelity](
        step_size=step_size, backward=backward, niter=niter,
        std_noise=std_noise, mask=mask
    )
    rmse = wlpnp.RMSEMultibatch(
        niter=niter, mask=mask, meancentering=True,
        path_to_saved_stats=path_to_saved_stats
    )

    # Iterate through batches
    end_idx = 0
    while end_idx < nimgs_val:
        beg_idx = end_idx
        (gamma, kappa_true), end_idx = val_gen.load_batch(
            beg_idx=beg_idx, max_idx=nimgs_val, return_end_idx=True
        )
        rmse.kappa_true = kappa_true # Update RMSE callback (ground truth)

        # Run PnP for this batch
        if verbose:
            print(f"Run PnP for images {beg_idx} to {end_idx}")
        _ = pnp(gamma, callbacks=[rmse])

    # Average RMSE over batches
    rmse.average_over_batches() # FIXME:

    # Save array of RMSE per iteration
    rmse.save() # FIXME:
    if verbose:
        print(f"Array of RMSE saved in '{path_to_saved_stats}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_trained_denoiser", type=str,
        help="Path to the trained denoiser (keras file)."
    )
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)."
    )
    parser.add_argument(
        "path_to_saved_stats", type=str,
        help="Path to the .npy file where the arrays of RMSE are saved."
    )
    parser.add_argument(
        "-df", "--data-fidelity", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Data fidelity function, for the forward step. "
            f"Default = {DATA_FIDELITY}"
        )
    )
    parser.add_argument(
        "-s", "--step-size", type=float,
        default=argparse.SUPPRESS,
        help=(
            "The step size of the PnP algorithm. "
            f"Default = {STEP_SIZE:.1e}"
        )
    )
    parser.add_argument(
        "--nimgs-train", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the training set. Will be discarded. "
            f"Default = {NIMGS_TRAIN}"
        )
    )
    parser.add_argument(
        "--nimgs-val", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the validation set. "
            f"Default = {NIMGS_VAL}"
        )
    )
    parser.add_argument(
        "--imgsize", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of pixels (width) in input images. "
            f"Default = {IMGSIZE}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            f"Batch size. Default = {BATCH_SIZE}"
        )
    )
    parser.add_argument(
        "-n", "--niter", type=int,
        default=argparse.SUPPRESS,
        help=(
            f"Number of iterations. Default = {NITER}"
        )
    )
    parser.add_argument(
        "--offset", type=float,
        default=argparse.SUPPRESS,
        help=(
            f"Default convergence value for a perfectly uniform universe. Default = {OFFSET:.2f}"
        )
    )
    parser.add_argument(
        "--seed", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Seed for the random number generators"
        )
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        default=argparse.SUPPRESS
    )

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
