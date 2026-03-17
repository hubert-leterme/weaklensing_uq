import os
import argparse

import torch
import deepinv as dinv

import wlmmuq
import wlmmuq.datasets.torch as wlds

from wlmmuq.datasets import SCALE, NUM_WORKERS

import _commons
import _add_arguments

METRIC_DICT = {
    'mse': wlmmuq.metric.MSE,
    'mae': wlmmuq.metric.MAE,
}

NREAL_PER_IMG = 1
LOSS = 'mse'
LEARNING_RATE = 1e-4
DROP_RATE = 0.1 # Drop rate for the learning rate scheduler
NDECAYS = 4 # Number of decays for the learning rate scheduler

def main(
        path_to_train_val_dataset: str = wlmmuq.PATH_TO_TRAIN_VAL_DATASET,
        path_to_std_noise: str = wlmmuq.PATH_TO_STD_NOISE,
        path_to_mask: str = wlmmuq.PATH_TO_MASK,
        path_to_ps=wlmmuq.PATH_TO_PS,
        bin_data_from_cosmos=False,
        cosmos_include_faint=False,
        max_z: float | None = _commons.MAX_Z, resolution: float = _commons.RESOLUTION,
        inpainting_deepmass=_commons.INPAINTING_DEEPMASS,
        arch=None, denoiser=False,
        nongaussian=False,
        which_gaussian_extractor=_commons.WHICH_GAUSSIAN_EXTRACTOR_PNPMASS,
        niter_wiener=_commons.NITER_WIENER,
        starlet_detection_threshold=_commons.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size_wiener=_commons.EPS_SUP_STEP_SIZE,
        model_specs: str | None = None,
        order2=False, additional_outlayer=None,
        arch_order1=None,
        timestamp_order1=None, epoch_order1=None,
        model_specs_order1: str | None = None,
        imgsize=_commons.IMGSIZE,
        nimgs_train=_commons.NIMGS_TRAIN, nimgs_val=_commons.NIMGS_VAL,
        nreal_per_img=NREAL_PER_IMG,
        nepochs=_commons.EPOCH, batch_size=_commons.BATCH_SIZE,
        learning_rate=LEARNING_RATE, lr_scheduler=False, drop_rate=DROP_RATE,
        ndecays=NDECAYS, loss=LOSS,
        model_dir: str = wlmmuq.MODEL_DIR,
        train_val_dataset_name: str | None = wlmmuq.TRAIN_VAL_DATASET_NAME,
        model_name: str | None = None,
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

    checkpoint_dir, _ = _commons.get_checkpoint_dirs(
        model_dir, train_val_dataset_name=train_val_dataset_name,
        model_name=model_name
    )
    callback_list = []

    if denoiser:
        std_noise = mask = None
        dataset_class = wlds.HDF5DatasetDenoiser
        noise_model = dinv.physics.GaussianNoise(sigma=0) # sigma to be updated
        physics = dinv.physics.LinearPhysics(noise_model=noise_model)

    else:
        # Get noise srtandard deviation and mask
        # TODO: add argument `zbins`
        raise NotImplementedError
        std_noise, mask, _ = _commons.get_stdnoise_mask_shearmap(
            path_to_std_noise=path_to_std_noise,
            path_to_mask=path_to_mask,
            bin_data_from_cosmos=bin_data_from_cosmos,
            imgsize=imgsize, cosmos_include_faint=cosmos_include_faint,
            max_z=max_z, resolution=resolution,
            inpainting=inpainting_deepmass, verbose=verbose
        ) # TODO: Add arguments `east_right` and `zbins`
        # Update arguments for data loading
        kwargs.update(std_noise=std_noise, mask=mask)

        dataset_class = wlds.HDF5DatasetMassMapping
        physics = wlmmuq.physics.MassMapping(
            sigma=std_noise, mask=mask
        )

    # Initialize data loaders
    if verbose:
        print("Initialize batch generators for training and validation")
    model_class, scale_as_input = _commons.get_model_class(arch)
    kwargs.update(scale_as_input=scale_as_input)
    train_dataset = dataset_class(
        hdf5_filepath=path_to_train_val_dataset,
        nimgs=nimgs_train, batch_size=batch_size,
        output_shape=imgsize,
        nreal_per_img=nreal_per_img,
        num_workers=num_workers, **kwargs
    )
    train_dataloader = train_dataset.to_dataloader()
    val_dataset = dataset_class(
        hdf5_filepath=path_to_train_val_dataset,
        nimgs=nimgs_val, batch_size=batch_size,
        beg_idx=nimgs_train, shuffle=False,
        output_shape=imgsize,
        num_workers=num_workers, **kwargs
    )
    val_dataloader = val_dataset.to_dataloader()

    # Initialize model
    if verbose:
        print("Initialize model")
    kwargs_model = {k: kwargs.pop(k) for k in _commons.KEYS_MODEL if k in kwargs}
    _commons.update_kwargs_model(
        kwargs_model,
        std_noise=std_noise, mask=mask, path_to_ps=path_to_ps,
        eps_sup_step_size_wiener=eps_sup_step_size_wiener,
        niter_wiener=niter_wiener, nbins=train_dataset.nbins,
        device=device, verbose=verbose
    )
    model = _commons.instantiate_model(
        model_class, imgsize=imgsize, order2=order2,
        additional_outlayer=additional_outlayer,
        device=device, verbose=verbose, **kwargs_model
    )
    model.train()

    # Set loss function
    kwargs_metric = {k: kwargs.pop(k) for k in _commons.KEYS_METRIC if k in kwargs}
    if train_dataset.nbins > 1: # Tomographic mass mapping
        normfact_zbins = torch.Tensor(train_dataset.normfact_zbins)
        kwargs_metric.update(channelwise_normfact=normfact_zbins)
    metric = METRIC_DICT[loss](**kwargs_metric)
    if order2:
        if verbose:
            print("Load trained order-1 moment network")
        if arch_order1 is None:
            arch_order1 = arch
            kwargs_model_order1 = kwargs_model.copy()
        else:
            kwargs_model_order1 = {}
            for k in _commons.KEYS_MODEL:
                k1 = f"{k}_order1"
                if k1 in kwargs:
                    kwargs_model_order1.update({k: kwargs.pop(k1)})
        assert epoch_order1 is not None
        order1_model = _commons.load_trained_model(
            checkpoint_dir, arch_order1, timestamp_order1, epoch_order1,
            model_specs=model_specs_order1, imgsize=imgsize, order2=False,
            nbins=train_dataset.nbins,
            device=device, verbose=verbose, **kwargs_model_order1
        )
        loss_fun = wlmmuq.loss.Order2SupLoss(
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
            white_noise=True,
            imgsize=imgsize, physics=physics,
            niter=1, # Convergence in one iteration (white noise)
            starlet_detection_threshold=starlet_detection_threshold,
            device=device, verbose=verbose
        ) # Not all arguments are needed here (`white_noise=True`)
        kwargs_trainer.update(preproc_for_residual=gaussian_extractor)
        if callback_gaussian_extractor is not None:
            callback_list.append(callback_gaussian_extractor)
        callback_list.append(
            wlmmuq.training.ParamsAlgoUpdater(
                optim=gaussian_extractor
            )
        )

    if model_specs is None:
        model_specs = _commons.MODEL_SPECS[order2]
    save_path = os.path.join(checkpoint_dir, model_specs)
    if resume:
        path_to_checkpoint_pretrained = _commons.get_path_to_checkpoint(
            save_path, timestamp_resume, epoch_resume
        )
        if verbose:
            print(f"Resuming training from {path_to_checkpoint_pretrained}")
        kwargs_trainer.update(ckpt_pretrained=path_to_checkpoint_pretrained)
    trainer = wlmmuq.training.Trainer(
        model,
        device=device,
        save_path=save_path,
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
            wlmmuq.callbacks.CProfilerCallback(
                trainer, max_nbatches=cprofiler_max_nbatches, wait=cprofiler_wait,
                cuda_synchronize=cprofiler_cuda_synchronize, verbose=verbose
            )
        )

    # Train model
    callbacks = wlmmuq.callbacks.CallbackList(callback_list)
    trainer.train(callbacks=callbacks)

    train_dataset.close()
    val_dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _add_arguments.model(parser, uq=True, denoiser=True, deepmass=True)
    parser.add_argument(
        "-d", "--denoiser", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Reconstruct the original convergence map from an input corrupted "
            "by a white Gaussian noise."
        )
    )
    parser.add_argument(
        "-ng", "--nongaussian", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Split the Gaussian and non-Gaussian parts of the convergence maps. "
            "This option is only compatible with flag `--denoiser`."
        )
    )
    parser.add_argument(
        "--which-gaussian-extractor", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Type of Gaussian extractor. Possible values are 'wiener' or 'mcalens'. "
            "Only used if `--nongaussian` is activated. "
            f"Default = '{_commons.WHICH_GAUSSIAN_EXTRACTOR_PNPMASS}'"
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
    _add_arguments.gaussian_extractor(parser, wiener=True)
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
    _add_arguments.std_noise_mask(parser)
    parser.add_argument(
        "-uq", "--order2", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Train order-2 moment network. If activated, then "
            "`--timestamp-order1` and `--epoch-order1` must be provided."
        )
    )
    _add_arguments.model_order1(parser, deepmass=True)
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
    _add_arguments.train_val_dataset(
        parser, batch_size=_commons.BATCH_SIZE
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
    _add_arguments.model_name(parser)
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
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
