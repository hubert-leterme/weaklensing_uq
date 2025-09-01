import argparse

import torch
import deepinv as dinv

import wlmmuq.data.torch as wlds
import wlmmuq.models as wlnn

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.data import SCALE, NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER

import _commons
from _commons import IMGSIZE, BATCH_SIZE, KEYS_MODEL, EPS_SUP_STEP_SIZE

NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations in the original dataset
NIMGS_VAL = 1440 # Remaining 2 realizations
NIMGS_PS = 2048
BATCH_SIZE_PS = 256
NREAL_PER_IMG = 1
LOSS = 'mse'
LEARNING_RATE = 1e-4
DROP_RATE = 0.1 # Drop rate for the learning rate scheduler
NDECAYS = 4 # Number of decays for the learning rate scheduler
CHECKPOINT_DIR = "."

def main(
        path_to_augmented_dataset,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        path_to_ps=PATH_TO_PS,
        cosmos_include_faint=False,
        inpainting_deepmass=_commons.INPAINTING_DEEPMASS,
        backend=None, arch=None, denoiser=False,
        wiener_init=False, nongaussian=False,
        which_gaussian_extractor=_commons.WHICH_GAUSSIAN_EXTRACTOR,
        niter_wiener=NITER_WIENER,
        noise_whitening_wiener=False,
        starlet_detection_threshold=_commons.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size_wiener=EPS_SUP_STEP_SIZE,
        order2=False, timestamp_order1=None, epoch_order1=None,
        imgsize=IMGSIZE,
        nimgs_train=NIMGS_TRAIN, nimgs_val=NIMGS_VAL, nreal_per_img=NREAL_PER_IMG,
        nepochs=_commons.EPOCH, batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE, lr_scheduler=False, drop_rate=DROP_RATE,
        ndecays=NDECAYS, loss=LOSS, checkpoint_dir=CHECKPOINT_DIR,
        num_workers=NUM_WORKERS,
        resume=False, timestamp_resume=None, epoch_resume=None,
        cprofiler=False, cprofiler_max_nbatches=None, cprofiler_wait=None,
        cprofiler_cuda_synchronize=False,
        seed=None, verbose=False, **kwargs
):
    _commons.set_seed(seed)
    device = _commons.get_device(verbose=verbose)
    if verbose:
        print(f"Number of workers: {num_workers}")

    callback_list = []

    kwargs_model = {k: kwargs.pop(k) for k in KEYS_MODEL if k in kwargs}
    try:
        no_bias = kwargs.pop("no_bias")
    except KeyError:
        pass
    else:
        kwargs_model.update(bias=not no_bias)

    if denoiser:
        dataset_class = wlds.HDF5DatasetDenoiser
        noise_model = dinv.physics.GaussianNoise(sigma=0) # sigma to be updated
        physics = dinv.physics.LinearPhysics(noise_model=noise_model)

    else:
        # Get noise srtandard deviation and mask
        std_noise, mask = _commons.get_stdnoise_mask(
            path_to_std_noise=path_to_std_noise,
            path_to_mask=path_to_mask,
            imgsize=imgsize, cosmos_include_faint=cosmos_include_faint,
            inpainting=inpainting_deepmass, verbose=verbose
        )
        # Update arguments for data loading
        kwargs.update(std_noise=std_noise, mask=mask)

        dataset_class = wlds.HDF5DatasetMassMapping
        physics = wlnn.deepinv.iterativemm.MassMapping(
            sigma=std_noise, mask=mask
        )

        if wiener_init:
            # Load arguments for Wiener initialization
            # Only for DeepMass (denoiser = False)
            args_wienerinit = _commons.get_args_wienerinit(
                std_noise, mask, path_to_ps=path_to_ps,
                noise_whitening=noise_whitening_wiener,
                eps_sup_step_size=eps_sup_step_size_wiener,
                niter=niter_wiener, device=device, verbose=verbose
            )
            kwargs_model.update(args_wienerinit=args_wienerinit)

    if arch is not None:
        backend = arch.split(".")[0]
        cnn_class, scale_as_input = wlnn.MODEL_CLASSES[arch]
        if scale_as_input:
            kwargs.update(scale_as_input=scale_as_input)
    else:
        cnn_class = None
        scale_as_input = False

    if backend == 'tensorflow':
        raise ValueError("Deprecated TensorFlow backend. Use PyTorch instead.")
    elif backend != 'torch':
        raise ValueError("Unsupported backend.")

    if verbose:
        print("Initialize batch generators for training and validation")
    train_dataset = dataset_class(
        hdf5_filepath=path_to_augmented_dataset,
        nimgs=nimgs_train, batch_size=batch_size,
        output_shape=imgsize,
        newaxis=True, nreal_per_img=nreal_per_img,
        num_workers=num_workers, **kwargs
    )
    train_dataloader = train_dataset.to_dataloader()
    val_dataset = dataset_class(
        hdf5_filepath=path_to_augmented_dataset,
        nimgs=nimgs_val, batch_size=batch_size,
        beg_idx=nimgs_train, shuffle=False,
        output_shape=imgsize, newaxis=True,
        num_workers=num_workers, **kwargs
    )
    val_dataloader = val_dataset.to_dataloader()

    # Initialize model
    model = cnn_class(
        map_size=imgsize, **kwargs_model
    ).to(device)

    if verbose:
        model.summary()

    # Set loss function
    metric = wlnn.torch.METRIC_DICT[loss]
    if order2:
        order1_model = _commons.load_trained_model(
            checkpoint_dir, arch, timestamp_order1, epoch_order1,
            imgsize=imgsize, order2=False,
            device=device, verbose=verbose, **kwargs_model
        )
        loss_fun = wlnn.torch.Order2SupLoss(
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
        if resume:
            # Useful if the state dict of the scheduler is not saved in the checkpoint
            scheduler.last_epoch = epoch_resume
    else:
        scheduler = None

    loss_fun.to(device)
    kwargs_trainer = {}

    if nongaussian:
        assert denoiser # Only for training a denoiser
        gaussian_extractor, callback_gaussian_extractor = \
                _commons.get_gaussian_extractor(
            which=which_gaussian_extractor,
            path_to_ps=path_to_ps,
            white_noise=True, noise_whitening_wiener=noise_whitening_wiener,
            imgsize=imgsize, physics=physics,
            niter=1, # Convergence in one iteration (white noise)
            starlet_detection_threshold=starlet_detection_threshold,
            mcalens_update_ng_first=True, # Otherwise, MCALens will produce the same output as Wiener
            device=device, verbose=verbose
        ) # Not all arguments are needed here (`white_noise=True`)
        kwargs_trainer.update(preproc=gaussian_extractor)
        if callback_gaussian_extractor is not None:
            callback_list.append(callback_gaussian_extractor)
        callback_list.append(
            wlnn.deepinv.pnpmcalens.ParamsAlgoUpdater(
                optim=gaussian_extractor,
                noise_whitening_wiener=noise_whitening_wiener
            )
        )

    if resume:
        path_to_checkpoint_pretrained = _commons.get_path_to_checkpoint(
            checkpoint_dir, timestamp_resume, epoch_resume, order2=order2
        )
        if verbose:
            print(f"Resuming training from {path_to_checkpoint_pretrained}")
        kwargs_trainer.update(ckpt_pretrained=path_to_checkpoint_pretrained)
    trainer = wlnn.deepinv.trainer.Trainer(
        model,
        device=device,
        save_path=checkpoint_dir,
        verbose=verbose,
        scale_as_input=scale_as_input,
        physics=physics,
        online_measurements=False,
        epochs=nepochs,
        scheduler=scheduler,
        losses=loss_fun,
        optimizer=optimizer,
        show_progress_bar=True,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        **kwargs_trainer
    )

    # Profiling callback
    if cprofiler:
        callback_list.append(
            wlnn.deepinv.callbacks.CProfilerCallback(
                trainer, max_nbatches=cprofiler_max_nbatches, wait=cprofiler_wait,
                cuda_synchronize=cprofiler_cuda_synchronize, verbose=verbose
            )
        )

    # Train model
    callbacks = wlnn.deepinv.callbacks.CallbackList(callback_list)
    trainer.train(callbacks=callbacks)

    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_augmented_dataset", type=str,
        help="Path to the augmented dataset (HDF5 file)"
    )
    _commons.add_arguments_model(parser)
    parser.add_argument(
        "-d", "--denoiser", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Reconstruct the original convergence map from an input corrupted "
            "by a white Gaussian noise."
        )
    )
    parser.add_argument(
        "--wiener-init", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Use Wiener initialization."
        )
    )
    parser.add_argument(
        "--nongaussian", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Split the Gaussian and non-Gaussian parts of the convergence maps."
        )
    )
    parser.add_argument(
        "--which-gaussian-extractor", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Type of Gaussian extractor. Possible values are 'wiener' or 'mcalens'. "
            "Only used if `--nongaussian` is activated. "
            f"Default = '{_commons.WHICH_GAUSSIAN_EXTRACTOR}'"
        )
    )
    parser.add_argument(
        "-thresh", "--starlet-detection-threshold", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Detection threshold for computing the support of active "
            "starlet coefficients. "
            "Works with `--nongaussian --which-gaussian-extractor mcalens`. "
            f"Default = {int(_commons.STARLET_DETECTION_THRESHOLD)}-sigma"
        )
    )
    _commons.add_arguments_wiener(parser)
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
        "-uq", "--order2", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Train order-2 moment network. If activated, then "
            "`--timestamp-order1` and `--epoch-order1` must be provided."
        )
    )
    parser.add_argument(
        "-t1", "--timestamp-order1", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Timestamp of the trained order-1 moment network. "
            "Only used if `--order2` is activated. "
            "Default = None"
        )
    )
    parser.add_argument(
        "-e1", "--epoch-order1", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Epoch of the checkpoint for the trained order-1 network. "
            "Only used if `--order2` is activated. "
            "Default = None"
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
    _commons.add_arguments_dataset(parser, batch_size=BATCH_SIZE)
    parser.add_argument(
        "--nreal-per-img", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of noise realizations per image. "
            f"Default = {NREAL_PER_IMG}"
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
        "-e", "--nepochs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of training epochs. "
            f"Default = {_commons.EPOCH}"
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
            "Training loss function, e.g., 'mse' or 'mae'. "
            f"Default = {LOSS}"
        )
    )
    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=argparse.SUPPRESS,
        help="Path to checkpoint directory (saving model after each epoch). Default = None"
    )
    parser.add_argument(
        "-r", "--resume", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Resume training from a previous checkpoint. "
            "If activated, then `--timestamp-resume` and `--epoch-resume` must be provided."
        )
    )
    parser.add_argument(
        "-tr", "--timestamp-resume", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Timestamp of the checkpoint to resume training from. "
            "Only used if `--resume` is activated. "
            "Default = None"
        )
    )
    parser.add_argument(
        "-er", "--epoch-resume", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Epoch of the checkpoint to resume training from. "
            "Only used if `--resume` is activated. "
            "Default = None"
        )
    )
    parser.add_argument(
        "--cprofiler", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Profile training using cProfile."
        )
    )
    parser.add_argument(
        "--cprofiler-max-nbatches", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Maximum number of batches to profile. "
            "If None, then profiling is done until the end of the training. "
            "Default = None"
        )
    )
    parser.add_argument(
        "--cprofiler-wait", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of batches to wait before starting profiling. "
            "Default = None"
        )
    )
    parser.add_argument(
        "--cprofiler-cuda-synchronize", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Synchronize CUDA when profiling, after the forward pass, the "
            "loss evaluation, the backward pass, and the optimizer step. "
            "See: https://discuss.pytorch.org/t/to-device-slowing-the-code/80973/3. "
            "WARNING: This will slow down the training."
        )
    )
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
