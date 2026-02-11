__level__ = 1

import warnings
import torch
import torch.nn as nn
import deepinv as dinv

from .. import callbacks
from ..loss import metric

#=================================================================================
# deepinv/optim/distance.py
#=================================================================================

class MahalanobisDistance(dinv.optim.Distance):
    r"""
    Implementation of :math:`\distancename` as

    .. math::
        f(x) = \frac{1}{2}\|x-y\|_{\Sigma^{-1}}^2 = \frac{1}{2} (x-y)^\top \Sigma^{-1} (x-y)

    where :math:`\Sigma` is a diagonal matrix with positive entries.

    :param torch.Tensor param_vector: tensor representing the diagonal of
    the matrix :math:`\Sigma`. Default: ``None``.
    """

    def __init__(
            self, param_vector: float | torch.Tensor | None = None,
            sigma: float | torch.Tensor | None = None
    ):
        super().__init__()
        if sigma is not None:
            if param_vector is not None:
                raise ValueError(
                    "Either `sigma` or `param_vector` should be provided, not both."
                )
            warnings.warn(
                "The `sigma` parameter is deprecated and will be removed in future versions. "
                "Please use `param_vector` instead (`sigma**2`).",
                DeprecationWarning
            )
            param_vector = sigma**2
        # The tensor is properly sent to GPU when applying `self.to(device)`
        if torch.is_tensor(param_vector):
            self.register_buffer("param_vector", param_vector)
        else:
            self.param_vector = param_vector


    def fn(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs):
        z = x - y # Shape = ([batch_size], [nchannels], nx, ny)
        dim = tuple(range(1, z.dim())) # Exclude batch dimension
        d = 0.5 * torch.sum(
            torch.abs(z)**2 / self.param_vector, dim=dim
        ) # Shape = ([batch_size],)
        return d


    def grad(self, x, y, *args, **kwargs):
        r"""
        Computes the gradient of :math:`\distancename`, that is  :math:`\nabla_{x}\distance{x}{y}`, i.e.

        .. math::

            \nabla_{x}\distance{x}{y} = \Sigma^{-1} (x-y)


        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Observation :math:`y`.
        :return: (:class:`torch.Tensor`) gradient of the distance function :math:`\nabla_{x}\distance{x}{y}`.
        """
        return (x - y) / self.param_vector # Shape = ([batch_size], [nchannels], nx, ny)


    def prox(self, x, y, *args, gamma=1.0, **kwargs):
        raise NotImplementedError


#=================================================================================
# deepinv/optim/data_fidelity.py
#=================================================================================

class Mahalanobis(dinv.optim.DataFidelity):

    def __init__(
            self, param_vector: float | torch.Tensor | None = None,
            sigma: float | torch.Tensor | None = None
    ):
        super().__init__()
        self.d = MahalanobisDistance(
            param_vector=param_vector, sigma=sigma
        )


#=================================================================================
# deepinv/optim/optimizers.py
#=================================================================================

class BaseOptim(dinv.optim.BaseOptim):

    def __init__(
            self, *args, metric_dict: metric.MetricDict | None = None, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Bugfix: the prior (list of instances of :class:`deepinv.optim.Prior`),
        # and data_fidelity are converted to a `nn.ModuleList` to be properly registered.
        # In particular, the modules can be moved to GPU with `model.to(device)`.
        # This is copy-pasted from `deepinv.unfolded.unfolded.py`.
        self.prior = nn.ModuleList(self.prior) if self.prior else None
        self.data_fidelity = (
            nn.ModuleList(self.data_fidelity) if self.data_fidelity else None
        )
        # End of bugfix

        if metric_dict is not None:
            self.metric_dict = nn.ModuleDict(metric_dict)
        else:
            self.metric_dict = None
        self.batch_size = None

        get_output_0 = self.get_output
        def _get_output(X):
            if isinstance(X, tuple):
                X, X_uq = X
                out = get_output_0(X), get_output_0(X_uq)
            else:
                out = get_output_0(X)
            return out
        self.get_output = _get_output


    def _update_metrics(
            self, metrics: metric.MetricDict,
            x: torch.Tensor, x_gt: torch.Tensor | None = None
    ):
        if self.metric_dict is not None:
            for metric_name, metric_fn in self.metric_dict.items():
                metric = metric_fn(
                    x, x_gt, metrics[metric_name], None
                ) # Shape = (batch_size,)
                metrics.cat(metric_name, metric)

        return metrics


    def init_metrics_fn(self, X_init, x_gt=None):

        x_init = self.get_output(X_init)
        self.batch_size = x_init.shape[0]
        init = metric.MetricDict(
            batch_size=self.batch_size, dtype=x_init.dtype, device=x_init.device
        )
        if self.metric_dict is not None:
            for metric_name in self.metric_dict.keys():
                init.init_metric(metric_name)
        if self.has_cost:
            init.init_metric("cost")
        init.init_metric("residual")

        return self._update_metrics(init, x_init, x_gt)


    def update_metrics_fn(self, metrics: metric.MetricDict, X_prev, X, x_gt=None):

        if metrics is not None:
            x_prev = self.get_output(X_prev)
            x = self.get_output(X)

            # Shape = (batch_size, npixels)
            diff_flattened = (x_prev - x).reshape(self.batch_size, -1)
            x_flattened = x.reshape(self.batch_size, -1)
            residual = torch.linalg.norm(diff_flattened, dim=1) / \
                torch.linalg.norm(x_flattened, dim=1)  # Shape = (batch_size,)
            metrics.cat("residual", residual)

            if self.has_cost:
                metrics.cat("cost", X["cost"])

            metrics = self._update_metrics(metrics, x, x_gt)

        return metrics


def zero_init(y: torch.Tensor, _unused_physics):
    """The optimization algorithm is initialized with zero-valued tensors"""
    x_init = torch.zeros_like(y, dtype=torch.float32, device=y.device)
    z_init = torch.zeros_like(y, dtype=torch.float32, device=y.device)
    return {"est": (x_init, z_init)}


class ManualInit:
    """
    Manual initialization with user-provided tensors.
    """
    def __init__(self):
        self._X_init = None
    
    @property
    def X_init(self) -> tuple[torch.Tensor] | None:
        return self._X_init
    
    @X_init.setter
    def X_init(self, val: tuple[torch.Tensor] | None):
        self._X_init = val

    def reset(self):
        self.X_init = None

    def __call__(self, _unused_y, _unused_physics):
        return {"est": self.X_init}


class CallbackWrapperForFGSteps(nn.Module):

    def __init__(self, fg_step, callback: callbacks.BaseCallback | None = None):
        super().__init__()
        self.fg_step = fg_step
        self.callback = callback

    def forward(self, *args, **kwargs):
        out = self.fg_step(*args, **kwargs)
        if self.callback is not None:
            self.callback.on_forward_end(out)
        return out


def optim_builder(
    iteration,
    max_iter=100,
    params_algo={"lambda": 1.0, "stepsize": 1.0, "g_param": 0.05},
    data_fidelity=None,
    prior=None,
    F_fn=None,
    g_first=False,
    bregman_potential=None,
    callback_f_step: callbacks.BaseCallback | None = None,
    callback_g_step: callbacks.BaseCallback | None = None,
    **kwargs,
):
    r"""
    Helper function for building an instance of the :class:`deepinv.optim.BaseOptim` class.

    :param str, deepinv.optim.OptimIterator iteration: either the name of the algorithm to be used,
        or directly an optim iterator.
        If an algorithm name (string), should be either ``"GD"`` (gradient descent),
        ``"PGD"`` (proximal gradient descent), ``"ADMM"`` (ADMM),
        ``"HQS"`` (half-quadratic splitting), ``"CP"`` (Chambolle-Pock) or ``"DRS"`` (Douglas Rachford).
    :param int max_iter: maximum number of iterations of the optimization algorithm. Default: 100.
    :param dict params_algo: dictionary containing all the relevant parameters for running the algorithm,
                            e.g. the stepsize, regularisation parameter, denoising standart deviation.
                            Each value of the dictionary can be either Iterable (distinct value for each iteration) or
                            a single float (same value for each iteration). See :any:`optim-params` for more details.
                            Default: ``{"stepsize": 1.0, "lambda": 1.0}``.
    :param list, deepinv.optim.DataFidelity: data-fidelity term.
                            Either a single instance (same data-fidelity for each iteration) or a list of instances of
                            :class:`deepinv.optim.DataFidelity` (distinct data-fidelity for each iteration). Default: ``None``.
    :param list, deepinv.optim.Prior prior: regularization prior.
                            Either a single instance (same prior for each iteration) or a list of instances of
                            deepinv.optim.Prior (distinct prior for each iteration). Default: ``None``.
    :param Callable F_fn: Custom user input cost function. default: ``None``.
    :param bool g_first: whether to perform the step on :math:`g` before that on :math:`f` before or not. Default: `False`
    :param deepinv.optim.Bregman bregman_potential: Bregman potential used for Bregman optimization algorithms such as Mirror Descent. Default: ``None``, uses standart Euclidean optimization.
    :param kwargs: additional arguments to be passed to the :class:`deepinv.optim.BaseOptim` class.
    :return: an instance of the :class:`deepinv.optim.BaseOptim` class.

    """
    iterator = dinv.optim.optimizers.create_iterator(
        iteration,
        prior=prior,
        F_fn=F_fn,
        g_first=g_first,
        bregman_potential=bregman_potential,
    )
    if callback_f_step is not None:
        iterator.f_step = CallbackWrapperForFGSteps(iterator.f_step, callback=callback_f_step)
    if callback_g_step is not None:
        iterator.g_step = CallbackWrapperForFGSteps(iterator.g_step, callback=callback_g_step)
    return BaseOptim(
        iterator,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        prior=prior,
        params_algo=params_algo,
        max_iter=max_iter,
        **kwargs,
    ).eval()
