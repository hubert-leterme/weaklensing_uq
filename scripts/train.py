import os
import argparse
import time
import cProfile
import threading

import torch
import tensorflow as tf
import deepinv as dinv

import _commons

import wlmmuq.data as wlds
import wlmmuq.models as wlcnn

from wlmmuq import OFFSET
from wlmmuq.data import SCALE, NUM_WORKERS
from wlmmuq.models import L2_LAMBDA

IMGSIZE = 304
NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations in the original dataset
NIMGS_VAL = 1440 # Remaining 2 realizations
NREAL_PER_IMG = 1
NEPOCHS = 20
BATCH_SIZE = 32
LOSS = 'mse'
LEARNING_RATE = 1e-4
DROP_RATE = 0.1 # Drop rate for the learning rate scheduler
NDECAYS = 4 # Number of decays for the learning rate scheduler

MODEL_CLASSES = {
    "tensorflow.UNet": (wlcnn.tensorflow.UNet, False),
    "tensorflow.UNetScoreMatching": (wlcnn.tensorflow.UNetScoreMatching, True),
    "torch.DRUNet": (wlcnn.torch.DRUNet, True),
    "torch.SUNet": (wlcnn.torch.SUNet, False)
} # (model_class, scale_as_input)

def main(
        path_to_augmented_dataset, path_to_pretrained_model=None,
        cosmos_include_faint=False,
        backend=None, arch=None, denoiser=False, use_stdnoise_mask=False,
        order2=False, path_to_pred_dataset=None,
        path_to_order1_model=None, imgsize=IMGSIZE,
        nimgs_train=NIMGS_TRAIN, nimgs_val=NIMGS_VAL, nreal_per_img=NREAL_PER_IMG,
        nepochs=NEPOCHS, batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, lr_scheduler=False, drop_rate=DROP_RATE,
        ndecays=NDECAYS, loss=LOSS, l2_lambda=L2_LAMBDA,
        offset=OFFSET, checkpoint_dir=None, save_freq=None, backup_dir=None,
        path_to_csv_log=None, path_to_tensorboard_log=None, seed=None,
        verbose=False, **kwargs
):
    _commons.set_seed(seed)

    keys_model = [
        'meancentering', 'sigmoid_activation', 'small_model', 'pretrained'
    ]
    kwargs_model = {k: kwargs.pop(k) for k in keys_model if k in kwargs}
    try:
        no_bias = kwargs.pop("no_bias")
    except KeyError:
        pass
    else:
        kwargs_model.update(use_bias=not no_bias)

    # Initialize batch generators for training and validation
    if use_stdnoise_mask:
        std_noise, mask = _commons.get_stdnoise_mask(
            imgsize, cosmos_include_faint=cosmos_include_faint,
            convert_to_torch_tensor=True, inpainting=True,
            seed=seed, verbose=verbose
        )
        kwargs.update(std_noise=std_noise, mask=mask)

    if arch is not None:
        backend = arch.split(".")[0]
        cnn_class, scale_as_input = MODEL_CLASSES[arch]
        if scale_as_input:
            kwargs.update(scale_as_input=scale_as_input)
    else:
        cnn_class = None
        scale_as_input = False

    kwargs_model_order1 = None
    if order2:
        if path_to_pred_dataset is not None:
            kwargs.update(
                order=2, pred_filepath=path_to_pred_dataset
            )
        elif backend == 'tensorflow':
            raise NotImplementedError
        else:
            # No mean centering and only positive values in order-2 moment networks
            kwargs_model_order1 = kwargs_model.copy()
            kwargs_model.update(
                meancentering=False, onlypositive=True
            )

    if backend == 'tensorflow': # Use Keras (TensorFlow backend)
        data_module = wlds.tensorflow
        model_module = wlcnn.tensorflow
    elif backend == 'torch': # Use DeepInverse (PyTorch backend)
        data_module = wlds.torch
        model_module = wlcnn.torch
    else:
        raise ValueError

    if denoiser:
        dataset_class = data_module.HDF5DatasetDenoiser
    else:
        dataset_class = data_module.HDF5DatasetDeepMass

    if verbose:
        print("Initialize batch generators for training and validation")
    train_dataset = dataset_class(
        hdf5_filepath=path_to_augmented_dataset,
        nimgs=nimgs_train, batch_size=batch_size,
        output_shape=imgsize,
        offset=offset, newaxis=True,
        nreal_per_img=nreal_per_img, **kwargs
    )
    train_dataloader = train_dataset.to_dataloader()
    val_dataset = dataset_class(
        hdf5_filepath=path_to_augmented_dataset,
        nimgs=nimgs_val, batch_size=batch_size,
        beg_idx=nimgs_train, shuffle=False,
        output_shape=imgsize, offset=offset, newaxis=True, **kwargs
    )
    val_dataloader = val_dataset.to_dataloader()

    # Initialize model
    if path_to_pretrained_model is None:
        model = cnn_class(
            map_size=imgsize, offset=offset, **kwargs_model
        )
    else:
        model = model_module.load_model(
            path_to_pretrained_model, **kwargs_model
        )

    if verbose:
        model_module.print_model(model)

    if checkpoint_dir is not None:
        if not order2:
            output_type = "pe" # Point estimate
        else:
            output_type = "var" # Variance
        filename = f"{os.path.basename(checkpoint_dir)}_{output_type}_e" + \
            "{epoch:02d}.keras"
        checkpoint_dir = os.path.join(checkpoint_dir, output_type)
        filepath = os.path.join(checkpoint_dir, filename)

    if backend == 'tensorflow':

        # Compile model
        wlcnn.tensorflow.compile_kerasmodel(
            model, loss=loss, l2_lambda=l2_lambda, offset=offset,
            learning_rate=learning_rate
        )

        # Define the checkpoint callback
        callbacks = []
        if checkpoint_dir is not None:
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_weights_only=False,
                save_best_only=False,
                save_freq=save_freq
            )
            callbacks.append(checkpoint_callback)
        if backup_dir is not None:
            backup_callback = tf.keras.callbacks.BackupAndRestore(
                backup_dir=os.path.join(backup_dir, output_type), save_freq="epoch"
            )
            callbacks.append(backup_callback)
        if path_to_csv_log is not None:
            csvlogger_callback = tf.keras.callbacks.CSVLogger(
                path_to_csv_log, append=True
            )
            callbacks.append(csvlogger_callback)
        if path_to_tensorboard_log is not None:
            tblogger_callback = tf.keras.callbacks.TensorBoard(
                log_dir=path_to_tensorboard_log
            )
            callbacks.append(tblogger_callback)
        if lr_scheduler:
            def schedule(epoch, lr):
                epochs_drop = nepochs // ndecays
                if epoch % epochs_drop == 0 and epoch > 0:
                    return lr * drop_rate
                else:
                    return lr

            lrscheduler_callback = tf.keras.callbacks.LearningRateScheduler(
                schedule, verbose=verbose
            )
            callbacks.append(lrscheduler_callback)

        # Fit model
        model.fit(
            train_dataloader, epochs=nepochs,
            steps_per_epoch=nreal_per_img * nimgs_train // batch_size,
            validation_data=val_dataloader,
            validation_steps=nimgs_val // batch_size,
            callbacks=callbacks
        )

    elif backend == 'torch':

        # Set loss function
        metric = wlcnn.torch.METRIC_DICT[loss]
        if order2 and path_to_pred_dataset is None:
            order1_model = cnn_class(
                map_size=imgsize, offset=offset, **kwargs_model_order1
            )
            checkpoint_order1_model = torch.load(path_to_order1_model)
            order1_model.load_state_dict(checkpoint_order1_model['state_dict'])

            loss_fun = wlcnn.torch.Order2SupLoss(
                order1_model=order1_model, metric=metric
            )
        else:
            loss_fun = dinv.loss.SupLoss(metric=metric)

        # Set optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-8
        )
        if lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=nepochs // ndecays, gamma=drop_rate
            )
        else:
            scheduler = None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            print(f"Device: {device}")
        model.to(device)
        loss_fun.to(device)
        trainer = wlcnn.torch.Trainer(
            model,
            device=device,
            save_path=checkpoint_dir,
            verbose=verbose,
            scale_as_input=scale_as_input,
            physics=None,
            online_measurements=False,
            epochs=nepochs,
            scheduler=scheduler,
            losses=loss_fun,
            optimizer=optimizer,
            show_progress_bar=True,
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
        )
        trainer.train()

    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    parser.add_argument(
        "-m", "--path-to-pretrained-model", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the pretrained model. If none is given, then the model is "
            "initialized and trained from scratch. If provided, then arguments "
            "`--meancentering` and `--no-bias` are ineffective; "
            "moreover, `--imgsize` must be compatible with the provided model. "
            "Default = None"
        )
    )
    parser.add_argument(
        "--backend", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Deep learning framework used to train the model ('tensorflow' or 'torch'). "
            "Only useful if `--path-to-pretrained-model` is provided. Default = None"
        )
    )
    parser.add_argument(
        "-a", "--arch", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the model. Possible values are: "
            f"{' | '.join(MODEL_CLASSES.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "-s", "--small-model", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Whether to use a small model. Only available for PyTorch models."
        )
    )
    parser.add_argument(
        "-p", "--pretrained", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Whether to use a pretrained network. Only available for PyTorch models. "
            "Not available if `--small-model` is used."
        )
    )
    parser.add_argument(
        "-d", "--denoiser", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Reconstruct the original convergence map from an input corrupted "
            "by a white Gaussian noise."
        )
    )
    parser.add_argument(
        "--use-std-noise", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Whether to apply a heteroscedastic noise to the input images "
            "(denoiser only)."
        )
    )
    parser.add_argument(
        "--scale", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Multiplicative factor for the noise level if the flag `--use-std-noise` "
            "if used, or noise standard deviation otherwise. If `scale_min` is provided, "
            "upper bound of the uniform distribution over which the actual scale is drawn. "
            f"Only useful if the flag `--denoiser` is used. Default = {SCALE}"
        )
    )
    parser.add_argument(
        "--scale-min", type=float,
        default=argparse.SUPPRESS,
        help=(
            "If provided, then the scale for the noise standard deviation will be drawn "
            "uniformly between `scale_min` and `scale` for each input image."
        )
    )
    parser.add_argument(
        "--input-method", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Weak lensing method used as input ('ks', 'wiener' or 'wiener_pgd'). "
            "Only used if option `--denoiser` is not activated. Default = None"
        )
    )
    parser.add_argument(
        "--order2", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Train order-2 moment network. If activated, then either `--path-to-pred-dataset` "
            "or `--path-to-order1-model` must be provided."
        )
    )
    parser.add_argument(
        "-pred", "--path-to-pred-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the prediction dataset (HDF5 file), computed with "
            "a previously-trained network. This is useful to train a moment "
            "network of order 2. Default = None"
        )
    )
    parser.add_argument(
        "-o1", "--path-to-order1-model", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the trained order-1 moment network. "
            "This is useful to train a moment network of order 2. "
            "Only works for PyTorch models. Default = None"
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
        "--nreal-per-img", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of noise realizations per image. "
            f"Default = {NREAL_PER_IMG}"
        )
    )
    parser.add_argument(
        "--meancentering", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Apply a meancentering operator at the output of the network."
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
        "--sigmoid-activation", action='store_true',
        default=argparse.SUPPRESS,
        help="Use sigmoid activation function in the output layer."
    )
    parser.add_argument(
        "-e", "--nepochs", type=int,
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
        "-w", "--num-workers", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of workers for parallel processing. Only work for PyTorch datasets. "
            f"Default = {NUM_WORKERS}"
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
