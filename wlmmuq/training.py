__level__ = 1

import os
import warnings
import time
import inspect
from tqdm import tqdm
import wandb
from pathlib import Path
import numpy as np
import torch
import deepinv as dinv

from . import utils, optim
from .callbacks import BaseCallback

#=================================================================================
# deepinv/training/trainer.py
#=================================================================================

class Trainer(dinv.Trainer):

    def __init__(
            self, *args, scale_as_input=False, pbar_logs=True,
            preproc_for_residual: dinv.optim.BaseOptim | None = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scale_as_input = scale_as_input
        self.pbar_logs = pbar_logs
        self.preproc_for_residual = preproc_for_residual

        self.current_iterators = None
        self.total_training_time = 0
        self.training_time_per_epoch = []
        self.eval_time_per_epoch = []


    def setup_train(self, train=True, **kwargs):
        super().setup_train(train, **kwargs)
        now = utils.get_timestamp()
        if self.save_path is not None:
            # Change date-time format to ease navigation from the terminal
            savedir = os.path.dirname(self.save_path)
            self.save_path = f"{savedir}/{now}"


    def get_samples(self, iterators, g):
        x, y, physics = super().get_samples(iterators, g)
        if self.scale_as_input:
            if physics is not None:
                warnings.warn("Output `physics` overriden.")
            y, scale = y
            physics = scale
            # TODO: should return an object of type `dinv.physics.Physics`
        return x, y, physics


    def plot(self, epoch, physics, x, y, x_net, train=True):
        if torch.is_complex(y):
            y = y.real
        super().plot(epoch, physics, x, y, x_net, train=train)


    def get_samples_offline(self, iterators, g):
        r"""
        Get the samples for the offline measurements.

        In this setting, samples have been generated offline and are loaded from the dataloader.
        This function returns a tuple containing necessary data for the model inference. It needs to contain
        the measurement, the ground truth, and the current physics operator, but can also contain additional data
        (you can override this function to add custom data).

        If the dataloader returns 3-tuples, this is assumed to be ``(x, y, params)`` where
        ``params`` is a dict of physics generator params. These params are then used to update
        the physics.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        Monitor running time for getting data and sending input to device.

        ************************************************************

        :param list iterators: List of dataloader iterators.
        :param int g: Current dataloader index.
        :returns: a dictionary containing at least: the ground truth, the measurement, and the current physics operator.
        """
        data = next(iterators[g])
        if (type(data) is not tuple and type(data) is not list) or len(data) != 2:
            raise ValueError(
                "If online_measurements=False, the dataloader should output a tuple (x, y)"
            )

        if len(data) == 2:
            x, y, params = *data, None
        elif len(data) == 3:
            x, y, params = data
        else:
            raise ValueError

        if type(x) is list or type(x) is tuple:
            x = [s.to(self.device, non_blocking=True) for s in x]
        else:
            x = x.to(self.device, non_blocking=True)

        y = y.to(self.device, non_blocking=True)
        physics = self.physics[g]

        if params is not None:
            params = {k: p.to(self.device, non_blocking=True) for k, p in params.items()}
            physics.update_parameters(**params)

        return x, y, physics


    def model_inference(self, y, physics, x=None, train=True, **kwargs):
        r"""
        Perform the model inference.

        It returns the network reconstruction given the samples.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        Avoid sending input to device at this stage
        Remove `with torch.no_grad()` context manager (inference was done twice)

        ************************************************************

        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Optional ground truth, used for computing convergence metrics.
        :returns: The network reconstruction.
        """
        kwargs = {}

        # check if the forward has 'update_parameters' method, and if so, update the parameters
        if "update_parameters" in inspect.signature(self.model.forward).parameters:
            kwargs["update_parameters"] = True

        # TODO: use `sigma` instead of `physics`, in case of denoiser training
        if self.plot_convergence_metrics and not train:
            x_net, self.conv_metrics = self.model(
                y, physics, x_gt=x, compute_metrics=True, **kwargs
            )
        else:
            x_net = self.model(y, physics, **kwargs)

        return x_net


    def compute_loss(
            self, physics, x, y, train=True, epoch: int | None = None,
            callbacks: BaseCallback | None = None
    ):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        - Callbacks
        - Option to avoid calling `.item()` and `.cpu()` for each batch.

        ************************************************************

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :param BaseCallback callbacks: Callbacks to be executed at each step.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        if train:
            self.optimizer.zero_grad()

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics, x=x, train=train)
        callbacks.on_forward_end(x_net)

        if train or self.display_losses_eval:
            # Compute the losses
            loss_total = 0
            for k, l in enumerate(self.losses):
                loss = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                    epoch=epoch,
                )
                loss_total += loss.mean()
                callbacks.on_loss_end(loss_total)
                if len(self.losses) > 1 and self.verbose_individual_losses:
                    if self.pbar_logs:
                        meters = (
                            self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                        )
                        meters.update(loss.detach().cpu().numpy())
                        cur_loss = meters.avg
                        logs[l.__class__.__name__] = cur_loss

            if self.pbar_logs:
                meters = self.logs_total_loss_train if train else self.logs_total_loss_eval
                meters.update(loss_total.item())
                logs["TotalLoss"] = meters.avg

        if train:
            loss_total.backward()  # Backward the total loss
            callbacks.on_backward_end()

            if self.pbar_logs:
                norm = self.check_clip_grad()  # Optional gradient clipping
                if norm is not None:
                    logs["gradient_norm"] = self.check_grad_val.avg

            # Optimizer step
            self.optimizer.step()
            callbacks.on_optimizer_step_end()

        return x_net, logs


    def compute_metrics(
        self, x, x_net, y, physics, logs, train=True, epoch: int | None = None
    ):
        r"""
        Compute the metrics.

        It computes the metrics over the batch.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        Option to avoid calling `.item()` and `.cpu()` for each batch.

        ************************************************************

        :param torch.Tensor x: Ground truth.
        :param torch.Tensor x_net: Network reconstruction.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Current physics operator.
        :param dict logs: Dictionary containing the logs for printing the training progress.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: The logs with the metrics.
        """
        if self.pbar_logs:
            # Compute the metrics over the batch
            with torch.no_grad():
                for k, l in enumerate(self.metrics):
                    metric = l(
                        x_net=x_net,
                        x=x,
                        epoch=epoch,
                    )

                    current_log = (
                        self.logs_metrics_train[k] if train else self.logs_metrics_eval[k]
                    )
                    current_log.update(metric.detach().cpu().numpy())
                    logs[l.__class__.__name__] = current_log.avg

                    if not train and self.compare_no_learning:
                        x_lin = self.no_learning_inference(y, physics)
                        metric = l(x=x, x_net=x_lin, y=y, physics=physics, model=self.model)
                        self.logs_metrics_linear[k].update(metric.detach().cpu().numpy())
                        logs[f"{l.__class__.__name__} no learning"] = (
                            self.logs_metrics_linear[k].avg
                        )
        return logs


    def load_model(self):
        r"""
        Load a pretrained model if required.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        - Load loss history and training time if available

        ************************************************************
        """
        if self.ckpt_pretrained is not None:
            checkpoint = torch.load(self.ckpt_pretrained)
            self.model.load_state_dict(checkpoint["state_dict"])
            if "optimizer" in checkpoint and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if "wandb_id" in checkpoint and self.wandb_vis:
                self.wandb_setup["id"] = checkpoint["wandb_id"]
                self.wandb_setup["resume"] = "allow"
            if "epoch" in checkpoint:
                self.epoch_start = checkpoint["epoch"]
            if "loss" in checkpoint:
                self.loss_history = checkpoint["loss"]
            if "total_training_time" in checkpoint:
                self.total_training_time = checkpoint["total_training_time"]
            if "training_time_per_epoch" in checkpoint:
                self.training_time_per_epoch = checkpoint["training_time_per_epoch"]
            if "eval_time_per_epoch" in checkpoint:
                self.eval_time_per_epoch = checkpoint["eval_time_per_epoch"]


    def save_model(self, epoch, eval_metrics=None, state=None):
        r"""
        Save the model.

        It saves the model every ``ckp_interval`` epochs.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        - Bugfix with epoch number
        - Save state dictionary for the learning rate scheduler

        ************************************************************

        :param int epoch: Current epoch.
        :param None, float eval_metrics: Evaluation metrics across epochs.
        :param dict state: custom objects to save with model
        """
        if state is None:
            state = {}

        if not self.save_path:
            return

        epoch += 1 # Ranges from 1 to self.epochs included
        if (epoch > 0 and epoch % self.ckp_interval == 0) or epoch == self.epochs:
            os.makedirs(str(self.save_path), exist_ok=True)
            state = state | {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "loss": self.loss_history,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "total_training_time": self.total_training_time,
                "training_time_per_epoch": self.training_time_per_epoch,
                "eval_time_per_epoch": self.eval_time_per_epoch
            }
            if eval_metrics is not None:
                state["eval_metrics"] = eval_metrics
            if self.wandb_vis:
                state["wandb_id"] = wandb.run.id
            torch.save(
                state,
                os.path.join(
                    Path(self.save_path), Path("ckp_{}.pth.tar".format(epoch))
                ),
            )


    def train(
        self, callbacks: BaseCallback=None
    ):
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        - Optional argument `callbacks`
        - Optional pre-processing

        ************************************************************

        :param BaseCallback callbacks: Callbacks to be executed at each step.
        :returns: The trained model.
        """
        if callbacks is None:
            callbacks = BaseCallback()

        self.setup_train()

        callbacks.on_train_begin()

        try:
            for epoch in range(self.epoch_start, self.epochs):
                beg_time_train = time.time()
                callbacks.on_epoch_begin(epoch)
                self.reset_metrics()

                ## Training
                self.current_iterators = [iter(loader) for loader in self.train_dataloader]

                batches = min(
                    [len(loader) - loader.drop_last for loader in self.train_dataloader]
                )

                if self.loop_physics_generator and self.physics_generator is not None:
                    for physics_generator in self.physics_generator:
                        physics_generator.reset_rng()

                self.model.train()
                for i in (
                    progress_bar := tqdm(
                        range(batches),
                        ncols=150,
                        disable=(not self.verbose or not self.show_progress_bar),
                    )
                ):
                    callbacks.on_batch_begin(i)
                    progress_bar.set_description(f"Train epoch {epoch + 1}/{self.epochs}")
                    self.step(
                        epoch, progress_bar, train=True, last_batch=(i == batches - 1),
                        callbacks=callbacks
                    )
                    callbacks.on_batch_end(i)

                self.loss_history.append(self.logs_total_loss_train.avg)

                if self.scheduler:
                    self.scheduler.step()

                self.training_time_per_epoch.append(time.time() - beg_time_train)

                ## Evaluation
                beg_time_eval = time.time()
                perform_eval = self.eval_dataloader and (
                    epoch % self.eval_interval == 0 or epoch + 1 == self.epochs
                )
                if perform_eval:
                    self.current_iterators = [
                        iter(loader) for loader in self.eval_dataloader
                    ]

                    batches = min(
                        [len(loader) - loader.drop_last for loader in self.eval_dataloader]
                    )

                    self.model.eval()
                    for i in (
                        progress_bar := tqdm(
                            range(batches),
                            ncols=150,
                            disable=(not self.verbose or not self.show_progress_bar),
                        )
                    ):
                        callbacks.on_eval_batch_begin(i)
                        progress_bar.set_description(
                            f"Eval epoch {epoch + 1}/{self.epochs}"
                        )
                        self.step(
                            epoch, progress_bar, train=False, last_batch=(i == batches - 1),
                            callbacks=callbacks
                        )
                        callbacks.on_eval_batch_end(i)

                    for l in self.logs_losses_eval:
                        self.eval_metrics_history[l.__class__.__name__] = l.avg

                    self.eval_time_per_epoch.append(time.time() - beg_time_eval)

                self.total_training_time += time.time() - beg_time_train

                # Saving the model
                self.save_model(epoch, self.eval_metrics_history if perform_eval else None)

                callbacks.on_epoch_end(epoch)

        finally:
            callbacks.on_train_end()

        if self.wandb_vis:
            wandb.save("model.h5")
            wandb.finish()

        return self.model


    def step(
            self, epoch, progress_bar, train=True, last_batch=False, callbacks: BaseCallback=None
    ):
        r"""
        Train/Eval a batch.

        It performs the forward pass, the backward pass, and the evaluation at each iteration.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        Optional argument `callbacks`

        ************************************************************

        :param int epoch: Current epoch.
        :param tqdm progress_bar: Progress bar.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param bool last_batch: If ``True``, the last batch of the epoch is being processed.
        :param BaseCallback callbacks: Callbacks to be executed at each step.
        :returns: The current physics operator, the ground truth, the measurement, and the network reconstruction.
        """
        if callbacks is None:
            callbacks = BaseCallback()

        # random permulation of the dataloaders
        G_perm = np.random.permutation(self.G)

        for g in G_perm:  # for each dataloader
            x, y, physics_cur = self.get_samples(self.current_iterators, g)
            callbacks.on_get_samples_end(physics_cur)

            # If required, compute residuals for both the input and the ground truth
            # TODO: Include `preproc_for_residual` in `get_samples`
            if self.preproc_for_residual is not None:
                x_preproc = self.preproc_for_residual(y, self.physics[g])
                x = x - x_preproc
                y = y - self.physics[g].A(x_preproc)
                # TODO: `self.physics[g]` to be replaced by `physics_cur` when fixed
                # The noise parameters are currently not stored into `self.physics[g]`,
                # but `self.preproc_for_residual` is assumed to only use the noiseless forward operator.

            # Compute loss and perform backprop
            x_net, logs = self.compute_loss(
                physics_cur, x, y, train=train, epoch=epoch, callbacks=callbacks
            )

            # detach the network output for metrics and plotting
            x_net = x_net.detach()

            # Log metrics
            logs = self.compute_metrics(
                x, x_net, y, physics_cur, logs, train=train, epoch=epoch
            )

            # Update the progress bar
            progress_bar.set_postfix(logs)

        if last_batch:
            if self.verbose and not self.show_progress_bar:
                if self.verbose_individual_losses:
                    print(
                        f"{'Train' if train else 'Eval'} epoch {epoch}:"
                        f" {', '.join([f'{k}={round(v, 3)}' for (k, v) in logs.items()])}"
                    )
                else:
                    print(
                        f"{'Train' if train else 'Eval'} epoch {epoch}: Total loss: {logs['TotalLoss']}"
                    )

            if train:
                logs["step"] = epoch

            self.log_metrics_wandb(logs, epoch, train)  # Log metrics to wandb
            self.plot(
                epoch,
                physics_cur,
                x,
                y,
                x_net,
                train=train,
            )  # plot images


class ParamsAlgoUpdater(BaseCallback):

    def __init__(
            self,
            optim: optim.BaseMCALens | optim.BaseOptim
    ):
        self.optim = optim

    def on_get_samples_end(self, physics):
        # Get white noise standard deviation
        # sigma = physics.noise_model.sigma # Float or tensor, shape = (batch_size,)
        sigma = physics # TODO: to be updated when `physics` will be fixed (uncomment above line)
        g_param_g = utils.get_g_param(sigma, noise_whitening=False)
        if isinstance(self.optim, optim.BaseMCALens):
            g_param_ng = utils.get_g_param(sigma, noise_whitening=True)

        for i, step_size in enumerate(
            self.optim.init_params_algo["stepsize"]
        ): # Possibly, one step size per iteration
            if isinstance(self.optim, optim.BaseMCALens):
                self.optim.init_params_algo["g_param"][i].g = step_size.g * g_param_g
                self.optim.init_params_algo["g_param"][i].ng = step_size.ng * g_param_ng
            else:
                self.optim.init_params_algo["g_param"][i] = step_size * g_param_g
