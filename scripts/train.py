import os
import argparse
import random
import time
import cProfile
import threading

import numpy as np
from tensorflow import data, keras

import wlmmuq.batchloader as wlbl
import wlmmuq.cnn_deepmass as wlcnn
import wlmmuq.iterativemm as wlpnp
import wlmmuq.utils as wlutils

MOMENT_ORDER = 1
FWHM = 2.4 # As in Starck et al. (2021) (Gaussian smoothing for KS)
IMGSIZE = 304
NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations in the original dataset
NIMGS_VAL = 1440 # Remaining 2 realizations
NIMGS_PS = 256 # To compute the power spectrum
NEPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LOSS = 'mse'
L2_LAMBDA = 1e-4
OFFSET = 0.5 # As in DeepMass

# Monkey-patch Adam (to avoid `ValueError: Argument(s) not recognized: {'lr': 1e-05}`)
_init_ = keras.optimizers.Adam.__init__

def new_init(self, *args, **kwargs):
    if 'lr' in kwargs:
        kwargs['learning_rate'] = kwargs.pop('lr')
    _init_(self, *args, **kwargs)

keras.optimizers.Adam.__init__ = new_init


def main(
        path_to_augmented_dataset, denoiser=False,
        moment_order=MOMENT_ORDER, path_to_pred_dataset=None, imgsize=IMGSIZE,
        nimgs_train=NIMGS_TRAIN, nimgs_val=NIMGS_VAL, mean_centering=False,
        no_bias=False,
        nepochs=NEPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
        lr_scheduler=False, loss=LOSS, l2_lambda=L2_LAMBDA,
        offset=OFFSET, checkpoint_dir=None, save_freq=None, backup_dir=None,
        path_to_csv_log=None, path_to_tensorboard_log=None, seed=None,
        verbose=False, **kwargs
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize batch generators for training and validation
    if denoiser:
        batch_loader = wlbl.HDF5BatchLoaderDenoiser
    else:
        batch_loader = wlbl.HDF5BatchLoaderDeepMass

    if verbose:
        print("Initialize batch generators for training and validation")
    train_gen = batch_loader(
        order=moment_order, hdf5_filepath=path_to_augmented_dataset,
        pred_filepath=path_to_pred_dataset,
        nimgs=nimgs_train, batch_size=batch_size,
        output_shape=imgsize,
        offset=offset, newaxis=True, **kwargs
    )
    val_gen = batch_loader(
        order=moment_order, hdf5_filepath=path_to_augmented_dataset,
        pred_filepath=path_to_pred_dataset,
        nimgs=nimgs_val, batch_size=batch_size,
        beg_idx=nimgs_train, shuffle=False,
        output_shape=imgsize, offset=offset, newaxis=True, **kwargs
    )

    # Initialize model
    cnn_instance = wlcnn.UNet(
        map_size=imgsize, learning_rate=learning_rate, loss=loss,
        l2_lambda=l2_lambda,
        mean_centering=mean_centering, use_bias=not no_bias
    )
    cnn_model = cnn_instance.model()

    # Define the checkpoint callback
    callbacks = []
    if checkpoint_dir is not None:
        if moment_order == 1:
            output_type = "pe" # Point estimate
        elif moment_order == 2:
            output_type = "var" # Variance
        else:
            raise ValueError
        filepath = os.path.join(
            checkpoint_dir, output_type,
            f"{os.path.basename(checkpoint_dir)}_{output_type}_e" + "{epoch:02d}.keras"
        )
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=False,
            save_best_only=False,
            save_freq=save_freq
        )
        callbacks.append(checkpoint_callback)
    if backup_dir is not None:
        backup_callback = keras.callbacks.BackupAndRestore(
            backup_dir=os.path.join(backup_dir, output_type), save_freq="epoch"
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
        "--denoiser", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Reconstruct the original convergence map from an input corrupted "
            "by a white Gaussian noise."
        )
    )
    parser.add_argument(
        "--scale", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Noise standard deviation, if flag `--denoiser` is used."
        )
    )
    parser.add_argument(
        "--scale-range", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Train over a range of noise standard deviations, uniformly drawn "
            "between `0` and `SCALE` for each input image."
        )
    )
    parser.add_argument(
        "--input-method", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Weak lensing method used as input ('ks', 'wiener' or 'wiener_pgd'). "
            "If option `--denoiser` is activated, then the network will be trained "
            "on the residuals between the ground truth convergence maps and the "
            "reconstructed image using the provided method. Default = None"
        )
    )
    parser.add_argument(
        "--moment-order", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Order of the moment network. "
            f"Default = {MOMENT_ORDER}"
        )
    )
    parser.add_argument(
        "--path-to-pred-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the prediction dataset (HDF5 file), computed with "
            "a previously-trained network. This is useful to train a moment "
            "network of order 2. Default = None"
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
        "--mean-centering", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Apply a mean-centering operator at the output of the network."
        )
    )
    parser.add_argument(
        "--no-bias", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not use bias in convolution or batch "
            "normalization layers."
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
        "--loss", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Training loss function, e.g., 'mse', 'mae', 'l2reg_mse' or 'l2reg_mae'. "
            f"Default = {LOSS}"
        )
    )
    parser.add_argument(
        "--l2-lambda", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Regularization parameter for 'l2reg_mse' or 'l2reg_mae'. "
            f"Default = {L2_LAMBDA}"
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
