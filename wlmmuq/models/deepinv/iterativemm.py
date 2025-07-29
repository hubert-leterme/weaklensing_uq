import shutil
import warnings
import torch
from torch import nn
import deepinv as dinv

from ... import utils
from . import callbacks

#########################################################################
# Monkey-patch `shutil` to avoid bugs when rendering LaTeX in matplotlib
#########################################################################

def fake_which(cmd):
    if cmd == "latex":
        return None
    return shutil.which(cmd)

shutil.which = fake_which


#########################################################################
# Custom classes for PnPMass
#########################################################################

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
            self, param_vector: float | torch.Tensor=None,
            sigma: float | torch.Tensor=None
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


class Mahalanobis(dinv.optim.data_fidelity.DataFidelity):

    def __init__(
            self, param_vector: float | torch.Tensor=None,
            sigma: float | torch.Tensor=None
    ):
        super().__init__()
        self.d = MahalanobisDistance(
            param_vector=param_vector, sigma=sigma
        )


class MassMapping(dinv.physics.LinearPhysics):

    def __init__(
            self, sigma: float | torch.Tensor=0.,
            mask: torch.Tensor=None, **kwargs
    ):
        noise_model = dinv.physics.GaussianNoise(sigma=sigma)
        super().__init__(
            A=self.get_shear_from_convergence,
            A_adjoint=self.get_convergence_from_shear,
            noise_model=noise_model, **kwargs
        )
        if mask is not None:
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = None

    def get_shear_from_convergence(self, kappa):
        return utils.get_shear_from_convergence(
            kappa, mask=self.mask, return_complex=True
        )

    def get_convergence_from_shear(self, gamma):
        return utils.get_convergence_from_shear(
            gamma, mask=self.mask, return_complex=True
        ).real # E-mode only


#########################################################################
# Custom classes for Wiener iterative filtering
#########################################################################

class ProximalWiener(nn.Module):

    def __init__(self, powerspectrum):
        super().__init__()
        self.register_buffer("powerspectrum", powerspectrum)


    def forward(
            self, inp: torch.Tensor,
            g_param: float | torch.Tensor
    ):
        # Either one scalar parameter for the whole batch, or one specific
        # parameter for each image
        out = torch.fft.fft2(inp)
        out /= (1 + g_param / self.powerspectrum)
        out = torch.fft.ifft2(out)

        return out.real


class WienerWhiteNoiseParamsAlgoUpdater(callbacks.BaseCallback):

    def __init__(self, optim: dinv.optim.BaseOptim, noise_whitening=False):
        self.optim = optim
        self.noise_whitening = noise_whitening

    def on_get_samples_end(self, physics):
        # Get white noise standard deviation
        # sigma = physics.noise_model.sigma # Float or tensor, shape = (batch_size,)
        sigma = physics # TODO: to be updated when `physics` will be fixed (uncomment above line)
        g_param = utils.get_g_param(sigma, self.noise_whitening)
        for i, step_size in enumerate(
            self.optim.init_params_algo["stepsize"]
        ): # Possibly, one step size per iteration
            self.optim.init_params_algo["g_param"][i] = step_size * g_param


#########################################################################
# Metrics
#########################################################################

class MSE(dinv.metric.MSE):

    def __init__(
            self, mask: torch.Tensor=None, meancentering: bool=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        if mask is not None:
            utils.check_mask(mask)
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = None
        self.meancentering = meancentering

    def metric(self, x_net, x, *args, **kwargs):
        if self.meancentering:
            x_net = utils.meancenter(x_net, mask=self.mask)
            x = utils.meancenter(x, mask=self.mask)
        if self.mask is not None:
            x_net = x_net[..., self.mask]
            x = x[..., self.mask]
        return super().metric(x_net, x, *args, **kwargs)


class RMSE(MSE):
    def metric(self, x_net, x, *args, **kwargs):
        return super().metric(x_net, x, *args, **kwargs) ** 0.5


#########################################################################
# Improve the BaseOptim class from deepinv.optim
#########################################################################

class MetricDict(dict):

    def __init__(
            self, batch_size, *args,
            dtype=torch.float32, device="cpu", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

    def init_metric(self, metric_name: str):
        self[metric_name] = torch.empty(
            (self.batch_size, 0),
            dtype=self.dtype, device=self.device
        )

    def cat(self, metric_name: str, metric: torch.Tensor):
        metric = metric.unsqueeze(1) # Shape = (batch_size, 1)
        self[metric_name] = torch.cat(
            [self[metric_name], metric], dim=1
        ) # Shape = (batch_size, niter + 1)


class FixedPointUQ(nn.Module):
    def __init__(self, fixed_point:dinv.optim.FixedPoint):
        super().__init__()
        self.fixed_point = fixed_point

    def forward(self, *args, compute_metrics=False, x_gt=None, **kwargs):
        X, metrics = self.fixed_point.forward(
            *args, compute_metrics=compute_metrics, x_gt=x_gt, **kwargs
        )
        X_uq = self.fixed_point.single_iteration(
            X,
            self.fixed_point.max_iter,
            *args,
            **kwargs,
        )
        return (X, X_uq), metrics


class BaseOptim(dinv.optim.BaseOptim):

    def __init__(
            self, *args, metric_dict: MetricDict=None,
            prior_uq: dinv.optim.Prior=None,
            init_estimate: dinv.optim.BaseOptim=None, **kwargs
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
        if prior_uq is not None:
            self.fixed_point = FixedPointUQ(self.fixed_point)
        self.prior_uq = prior_uq

        get_output_0 = self.get_output
        def _get_output(X):
            if isinstance(X, tuple):
                X, X_uq = X
                out = get_output_0(X), get_output_0(X_uq)
            else:
                out = get_output_0(X)
            return out
        self.get_output = _get_output

        self.init_estimate = init_estimate


    def _update_metrics(
            self, metrics: MetricDict, x: torch.Tensor, x_gt: torch.Tensor=None
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
        init = MetricDict(
            batch_size=self.batch_size, dtype=x_init.dtype, device=x_init.device
        )
        if self.metric_dict is not None:
            for metric_name in self.metric_dict.keys():
                init.init_metric(metric_name)
        if self.has_cost:
            init.init_metric("cost")
        init.init_metric("residual")

        return self._update_metrics(init, x_init, x_gt)


    def update_metrics_fn(self, metrics: MetricDict, X_prev, X, x_gt=None):

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


    def update_prior_fn(self, it):
        r"""
        For each prior function in `prior`, selects the prior value for iteration ``it``
        (if this prior depends on the iteration number).
        If `it == self.max_iter`, then the optimizer is set to UQ mode.

        :param int it: iteration number.
        :return: the prior at iteration ``it``.
        """
        if it < self.max_iter:
            # Do not use `super().update_prior_fn(it)` to avoid passing a bound method
            # without class context to the FixedPoint module
            cur_prior = self.prior[it] if len(self.prior) > 1 else self.prior[0]
        elif self.prior_uq is not None:
            cur_prior = self.prior_uq
        else:
            raise ValueError
        return cur_prior


    def forward(
            self, y, physics, x_gt=None, compute_metrics=False,
            kwargs_init_estimate=None, **kwargs
    ):
        if self.init_estimate is not None:
            if kwargs_init_estimate is None:
                kwargs_init_estimate = {}
            with torch.no_grad():
                x_init = self.init_estimate(
                    y, physics, x_gt=None,
                    compute_metrics=False, **kwargs_init_estimate
                )
                # Get residuals (input and ground truth)
                y = y - physics.A(x_init)
                x_gt = x_gt - x_init

        out = super().forward(
            y, physics, x_gt=x_gt, compute_metrics=compute_metrics, **kwargs
        )

        if self.init_estimate is not None:
            with torch.no_grad():
                if compute_metrics:
                    x, metrics = out
                else:
                    x = out
                    metrics = None
                x = x + x_init # Add initial estimate
                if compute_metrics:
                    out = x, metrics
                else:
                    out = x

        return out


def zero_init(y: torch.Tensor, _unused_physics):
    """The optimization algorithm is initialized with zero-valued tensors"""
    x_init = torch.zeros_like(y, dtype=torch.float32, device=y.device)
    z_init = torch.zeros_like(y, dtype=torch.float32, device=y.device)
    return {"est": (x_init, z_init)}


def optim_builder(
    iteration,
    max_iter=100,
    params_algo={"lambda": 1.0, "stepsize": 1.0, "g_param": 0.05},
    data_fidelity=None,
    prior=None,
    prior_uq=None,
    F_fn=None,
    g_first=False,
    bregman_potential=None,
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
    return BaseOptim(
        iterator,
        has_cost=iterator.has_cost,
        data_fidelity=data_fidelity,
        prior=prior,
        prior_uq=prior_uq,
        params_algo=params_algo,
        max_iter=max_iter,
        **kwargs,
    ).eval()
