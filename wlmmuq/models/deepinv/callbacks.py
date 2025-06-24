import os
import time
import cProfile
import threading
import torch

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
    def on_forward_end(self):
        pass
    def on_loss_end(self, loss):
        pass
    def on_backward_end(self):
        pass
    def on_optimizer_step_end(self):
        pass


class CProfilerCallback(BaseCallback):

    def __init__(
            self, trainer, max_nbatches=None, wait=None,
            filename_stats='stats.prof', cuda_synchronize=False,
            verbose=False
    ):
        super().__init__()
        self.trainer = trainer
        self.max_nbatches = max_nbatches
        self.wait = wait
        self.filename_stats = filename_stats
        self.cuda_synchronize = cuda_synchronize
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

    def on_forward_end(self):
        self._cuda_synchronize()

    def on_loss_end(self, loss):
        self._cuda_synchronize()

    def on_backward_end(self):
        self._cuda_synchronize()

    def on_optimizer_step_end(self):
        self._cuda_synchronize()

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

    def _cuda_synchronize(self):
        if self.cuda_synchronize \
                and torch.cuda.is_available() \
                and self._profiling_started \
                and not self._profiling_ended:
            torch.cuda.synchronize()


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

    def __init__(self, callbacks: BaseCallback=None):
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
    def on_forward_end(self):
        self._loop_over_callbacks("on_forward_end")
    def on_loss_end(self, loss):
        self._loop_over_callbacks("on_loss_end", loss)
    def on_backward_end(self):
        self._loop_over_callbacks("on_backward_end")
    def on_optimizer_step_end(self):
        self._loop_over_callbacks("on_optimizer_step_end")
