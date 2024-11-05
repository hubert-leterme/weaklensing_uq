import os
import argparse
import random
import time
import cProfile
import threading

import numpy as np
from tensorflow import data, keras

from deepmass import map_functions as mf
from deepmass import lens_data as ld
from deepmass import wiener
from deepmass import cnn_keras as cnn

import wlmmuq.kappatng as wlktng
import wlmmuq.cosmos as wlcosmos
import wlmmuq.utils as wlutils

INPUT_WLMETHOD = "wiener"
FWHM = 2.4 # As in Starck et al. (2021) (Gaussian smoothing for KS)
IMGSIZE = 304
NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations in the original dataset
NIMGS_VAL = 1440 # Remaining 2 realizations
NIMGS_PS = 256 # To compute the power spectrum
NEPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OFFSET = 0.5 # As in DeepMass

# Monkey-patch Adam (to avoid `ValueError: Argument(s) not recognized: {'lr': 1e-05}`)
_init_ = keras.optimizers.Adam.__init__

def new_init(self, *args, **kwargs):
    if 'lr' in kwargs:
        kwargs['learning_rate'] = kwargs.pop('lr')
    _init_(self, *args, **kwargs)

keras.optimizers.Adam.__init__ = new_init


def main(
        path_to_augmented_dataset, input_wlmethod=INPUT_WLMETHOD,
        fwhm=FWHM, path_to_powerspectrum=None, imgsize=IMGSIZE,
        nimgs_train=NIMGS_TRAIN, nimgs_val=NIMGS_VAL,
        nepochs=NEPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        lr_scheduler=True, offset=OFFSET, checkpoint_dir=None, save_freq=None,
        backup_dir=None, path_to_csv_log=None, path_to_tensorboard_log=None,
        seed=None, verbose=False, **kwargs
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Compute a map of number of galaxies per pixels and a binary mask
    if verbose:
        print("Compute a map of number of galaxies per pixels and a binary mask")
    cat_cosmos_bright, _ = wlcosmos.cosmos_catalog()
    cat_cosmos_bright = wlktng.filter_by_redshifts(cat_cosmos_bright)
    data_dict = wlktng.get_data_from_cosmos_ktng(cat_cosmos_bright, imgsize)
    openingangle = data_dict["openingangle"]
    shapedisp = data_dict["shapedisp"]
    ngal = data_dict["ngal"]
    mask = data_dict["mask"]

    # Compute noise covariance matrix
    if verbose:
        print("Compute noise covariance matrix")
    std_noise = wlutils.get_std_noise(ngal, shapedisp, std_noise_mask=0)

    # Initialize batch generators for training and validation
    if input_wlmethod == 'ks':
        if fwhm is not None:
            resolution = openingangle / imgsize * 60. # arcmin/pixel
            std_gaussianfilter_arcmin = fwhm / (2 * np.sqrt(2 * np.log(2)))
            std_gaussianfilter = std_gaussianfilter_arcmin / resolution # pixels
            kwargs.update(std_gaussianfilter=std_gaussianfilter)

    elif input_wlmethod == 'wiener':
        if verbose:
            print("Estimate the power spectrum for Wiener filtering")

        if path_to_powerspectrum is None:
            # Load a set of convergence maps among the training set
            train_gen_ps = wlutils.HDF5BatchLoader(
                path_to_augmented_dataset, nimgs=NIMGS_PS, batch_size=NIMGS_PS,
                std_noise=std_noise, mask=mask, output_shape=imgsize,
                list_of_outputs=['kappa_true']
            )
            kappa_ps = train_gen_ps.load_batch()
            train_gen_ps.close()

            # Compute the 1D power spectrum
            powerspectrum_1d = wlutils.get_1d_powerspectrum(kappa_ps)
            del kappa_ps

        else:
            powerspectrum_1d = np.load(path_to_powerspectrum)

        kwargs.update(powerspectrum_1d=powerspectrum_1d)

    else:
        raise ValueError

    if verbose:
        print("Initialize batch generators for training and validation")
    train_gen = wlutils.HDF5BatchLoader(
        path_to_augmented_dataset, nimgs=nimgs_train, batch_size=batch_size,
        std_noise=std_noise, mask=mask, output_shape=imgsize,
        list_of_outputs=['kappa_inp', 'kappa_true'], offset=offset, newaxis=True,
        input_method=input_wlmethod, **kwargs
    )
    val_gen = wlutils.HDF5BatchLoader(
        path_to_augmented_dataset, nimgs=nimgs_val, batch_size=batch_size,
        std_noise=std_noise, mask=mask, beg_idx=nimgs_train, shuffle=False,
        output_shape=imgsize, list_of_outputs=['kappa_inp', 'kappa_true'],
        offset=offset, newaxis=True,
        input_method=input_wlmethod, **kwargs
    )

    # Initialize model
    cnn_instance = cnn.UnetlikeBaseline(map_size=imgsize, learning_rate=learning_rate)
    cnn_model = cnn_instance.model()

    # Define the checkpoint callback
    callbacks = []
    if checkpoint_dir is not None:
        filepath = os.path.join(checkpoint_dir, "{epoch:02d}.keras")
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False,
            save_best_only=False,
            save_freq=save_freq
        )
        callbacks.append(checkpoint_callback)
    if backup_dir is not None:
        backup_callback = keras.callbacks.BackupAndRestore(
            backup_dir=backup_dir,
            save_freq="epoch"
        )
        callbacks.append(backup_callback)
    if path_to_csv_log is not None:
        csvlogger_callback = keras.callbacks.CSVLogger(
            path_to_csv_log, append=True
        )
        callbacks.append(csvlogger_callback)
    if path_to_tensorboard_log is not None:
        tblogger_callback = keras.callbacks.TensorBoard(
            log_dir=path_to_tensorboard_log
        )
        callbacks.append(tblogger_callback)
    if lr_scheduler:
        def schedule(epoch, lr):
            drop_rate = 0.1
            epochs_drop = nepochs // 4
            if epoch % epochs_drop == 0 and epoch > 0:
                return lr * drop_rate
            else:
                return lr

        lrscheduler_callback = keras.callbacks.LearningRateScheduler(
            schedule, verbose=verbose
        )
        callbacks.append(lrscheduler_callback)

    # Prefetch datasets for efficiency
    train_set_prefetched = train_gen.to_tf_dataset().prefetch(data.AUTOTUNE)
    val_set_prefetched = val_gen.to_tf_dataset().prefetch(data.AUTOTUNE)

    # Fit model
    cnn_model.fit(
        train_set_prefetched, epochs=nepochs,
        steps_per_epoch=nimgs_train // batch_size,
        validation_data=val_set_prefetched,
        validation_steps=nimgs_val // batch_size,
        callbacks=callbacks
    )
    train_gen.close()
    val_gen.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "--input-wlmethod", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Weak lensing method used as input ('wiener' or 'ks'). "
            f"Default = '{INPUT_WLMETHOD}'"
        )
    )
    parser.add_argument(
        "--fwhm", type=int,
        default=argparse.SUPPRESS,
        help=(
            "If the selected method is Kaiser-Squires ('ks'), FWHM of "
            f"the smoothing filter, in arcmin. Default = {FWHM}"
        )
    )
    parser.add_argument(
        "-ps", "--path-to-powerspectrum", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the .npy file containing the 1D power spectrum. "
            "If not provided, and if argument --input-wlmethod is set to "
            "'wiener', then the power spectrum will be inferred from the "
            "dataset. Default = None"
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
        "--nimgs-train", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images in the training set. "
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
        "--nepochs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of training epochs. "
            f"Default = {NEPOCHS}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size for training and validation. "
            f"Default = {BATCH_SIZE}"
        )
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float,
        default=argparse.SUPPRESS,
        help=(
            f"Learning rate. Default = {LEARNING_RATE}"
        )
    )
    parser.add_argument(
        "--lr-scheduler", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Drop the learning rate by a factor 10 three times during training"
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
        "--checkpoint-dir", type=str,
        default=argparse.SUPPRESS,
        help="Path to checkpoint directory (saving model after each epoch). Default = None"
    )
    parser.add_argument(
        "--save-freq", type=int,
        default=argparse.SUPPRESS,
        help=(
            "TensorFlow documentation: the callback saves the model at end of this many batches. "
            "Default = None (saved after each epoch)"
        )
    )
    parser.add_argument(
        "--backup-dir", type=str,
        default=argparse.SUPPRESS,
        help=(
            "TensorFlow documentation: path of directory where to store the data needed to "
            "restore the model. The directory cannot be reused elsewhere to store other files, "
            "e.g. by the `BackupAndRestore` callback of another training run, or by another "
            "callback (e.g. `ModelCheckpoint`) of the same training run. Default = None"
        )
    )
    parser.add_argument(
        "--path-to-csv-log", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the CSV file where epoch results are stored. Default = None"
        )
    )
    parser.add_argument(
        "--path-to-tensorboard-log", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the TensorBoard log file. Default = None"
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

    def print_stats():
        while True:
            time.sleep(15)
            profiler.dump_stats('profile_results.prof')

    profiler = cProfile.Profile()
    profiler.enable()
    stats_thread = threading.Thread(target=print_stats, daemon=True)
    stats_thread.start()

    main(**kwargs)

    profiler.disable()
