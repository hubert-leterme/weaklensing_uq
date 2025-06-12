import os
import warnings
import time
import cProfile
import threading
from datetime import datetime
from tqdm import tqdm
import wandb
from pathlib import Path
import torch
import deepinv as dinv

#=================================================================================
# Class inheriting from dinv.Trainer, used for training
#=================================================================================

class Trainer(dinv.Trainer):

    def __init__(
            self, *args, scale_as_input=False, pbar_logs=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scale_as_input = scale_as_input
        self.pbar_logs = pbar_logs

        self.current_iterators = None


    def setup_train(self, train=True, **kwargs):
        super().setup_train(train, **kwargs)
        now = datetime.now().strftime(r"%Y%m%d_%H%M%S")
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
        return x, y, physics


    def plot(self, epoch, physics, x, y, x_net, train=True):
        if torch.is_complex(y):
            y = y.real
        super().plot(epoch, physics, x, y, x_net, train=train)


    def compute_loss(self, physics, x, y, train=True, epoch: int = None):
        r"""
        Compute the loss and perform the backward pass.

        It evaluates the reconstruction network, computes the losses, and performs the backward pass.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        Option to avoid calling `.item()` and `.cpu()` for each batch.

        ************************************************************

        :param deepinv.physics.Physics physics: Current physics operator.
        :param torch.Tensor x: Ground truth.
        :param torch.Tensor y: Measurement.
        :param bool train: If ``True``, the model is trained, otherwise it is evaluated.
        :param int epoch: current epoch.
        :returns: (tuple) The network reconstruction x_net (for plotting and computing metrics) and
            the logs (for printing the training progress).
        """
        logs = {}

        if train:
            self.optimizer.zero_grad()

        # Evaluate reconstruction network
        x_net = self.model_inference(y=y, physics=physics, x=x, train=train)

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
                logs[f"TotalLoss"] = meters.avg

        if train:
            loss_total.backward()  # Backward the total loss

            if self.pbar_logs:
                norm = self.check_clip_grad()  # Optional gradient clipping
                if norm is not None:
                    logs["gradient_norm"] = self.check_grad_val.avg

            # Optimizer step
            self.optimizer.step()

        return x_net, logs


    def compute_metrics(
        self, x, x_net, y, physics, logs, train=True, epoch: int = None
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


    def save_model(self, epoch, eval_metrics=None, state=None):
        r"""
        Save the model.

        It saves the model every ``ckp_interval`` epochs.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        Bugfix with epoch number

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
        self, callbacks=None
    ):
        r"""
        Train the model.

        It performs the training process, including the setup, the evaluation, the forward and backward passes,
        and the visualization.

        ********** MODIFIED VERSION OF THE DEEPINV METHOD **********

        Optional argument `callbacks`

        ************************************************************

        :returns: The trained model.
        """
        if callbacks is None:
            callbacks = BaseCallback()

        self.setup_train()

        callbacks.on_train_begin()

        try:
            for epoch in range(self.epoch_start, self.epochs):
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
                        epoch, progress_bar, train=True, last_batch=(i == batches - 1)
                    )
                    callbacks.on_batch_end(i)

                self.loss_history.append(self.logs_total_loss_train.avg)

                if self.scheduler:
                    self.scheduler.step()

                ## Evaluation
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
                            epoch, progress_bar, train=False, last_batch=(i == batches - 1)
                        )
                        callbacks.on_eval_batch_end(i)

                    for l in self.logs_losses_eval:
                        self.eval_metrics_history[l.__class__.__name__] = l.avg

                # Saving the model
                self.save_model(epoch, self.eval_metrics_history if perform_eval else None)

                callbacks.on_epoch_end(epoch)

        finally:
            callbacks.on_train_end()

        if self.wandb_vis:
            wandb.save("model.h5")
            wandb.finish()

        return self.model


#=================================================================================
# Callbacks
#=================================================================================

class BaseCallback:
    def on_train_begin(self):
        pass
    def on_train_end(self):
        pass
    def on_epoch_begin(self, epoch):
        pass
    def on_epoch_end(self, epoch):
        pass
    def on_batch_begin(self, batch):
        pass
    def on_batch_end(self, batch):
        pass
    def on_eval_batch_begin(self, batch):
        pass
    def on_eval_batch_end(self, batch):
        pass


class CProfilerCallback(BaseCallback):

    def __init__(
            self, trainer, max_nbatches=None, wait=None,
            filename_stats='stats.prof', verbose=False
    ):
        super().__init__()
        self.trainer = trainer
        self.max_nbatches = max_nbatches
        self.wait = wait
        self.filename_stats = filename_stats
        self.verbose = verbose

        self.profiler = cProfile.Profile()

        self._nbatches = 0
        self._profiling_started = False
        self._profiling_ended = False

    def on_train_begin(self):
        os.makedirs(self.trainer.save_path, exist_ok=True)
        self.filename_stats = os.path.join(
            self.trainer.save_path, self.filename_stats
        )
        if self.verbose:
            print(
                f"Profiling will be saved to {self.filename_stats}"
            )
        if self.wait is None:
            self._start_profiling()

    def on_train_end(self):
        self._end_profiling()

    def on_batch_end(self, batch):
        self._nbatches += 1
        if not self._profiling_started \
                and self.wait is not None \
                and self._nbatches >= self.wait:
            self._nbatches = 0
            self._start_profiling()
        if not self._profiling_ended \
                and self.max_nbatches is not None \
                and self._nbatches >= self.max_nbatches:
            self._end_profiling()

    def _print_stats(self):
        while True:
            time.sleep(15)
            if not self._profiling_ended:
                self.profiler.dump_stats(self.filename_stats)
            else:
                break

    def _start_profiling(self):
        self.profiler.enable()
        self._profiling_started = True
        stats_thread = threading.Thread(target=self._print_stats, daemon=True)
        stats_thread.start()

    def _end_profiling(self):
        self.profiler.dump_stats(self.filename_stats)
        self.profiler.disable()
        self._profiling_ended = True


class PyTorchProfilerCallback(BaseCallback):

    def __init__(
            self, trainer, logdir='pytorch_profiler', **kwargs
    ):
        super().__init__()
        self.trainer = trainer
        self.logdir = logdir
        self.kwargs = kwargs
        self.profiler = None

    def on_train_begin(self):
        logdir = os.path.join(self.trainer.save_path, self.logdir)
        os.makedirs(self.trainer.save_path, exist_ok=True)
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
            **self.kwargs
        )
        self.profiler.__enter__()

    def on_batch_end(self, batch):
        self.profiler.step()

    def on_train_end(self):
        self.profiler.__exit__(None, None, None)


class CallbackList(BaseCallback):

    def __init__(self, callbacks=None):
        super().__init__()
        self.callbacks = callbacks if callbacks is not None else []

    def _loop_over_callbacks(self, method_name, *args, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                getattr(callback, method_name)(*args, **kwargs)

    def on_train_begin(self):
        self._loop_over_callbacks("on_train_begin")
    def on_train_end(self):
        self._loop_over_callbacks("on_train_end")
    def on_epoch_begin(self, epoch):
        self._loop_over_callbacks("on_epoch_begin", epoch)
    def on_epoch_end(self, epoch):
        self._loop_over_callbacks("on_epoch_end", epoch)
    def on_batch_begin(self, batch):
        self._loop_over_callbacks("on_batch_begin", batch)
    def on_batch_end(self, batch):
        self._loop_over_callbacks("on_batch_end", batch)
    def on_eval_batch_begin(self, batch):
        self._loop_over_callbacks("on_eval_batch_begin", batch)
    def on_eval_batch_end(self, batch):
        self._loop_over_callbacks("on_eval_batch_end", batch)
