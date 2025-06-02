import shutil
import torch
from torch import nn
import deepinv as dinv

from . import utils

# Monkey-patch `shutil` to avoid bugs when rendering LaTeX in matplotlib
def fake_which(cmd):
    if cmd == "latex":
        return None
    return shutil.which(cmd)

shutil.which = fake_which


class MahalanobisDistance(dinv.optim.Distance):
    r"""
    Implementation of :math:`\distancename` as

    .. math::
        f(x) = \frac{1}{2}\|x-y\|_{\Sigma^{-1}}^2 = \frac{1}{2} (x-y)^\top \Sigma^{-1} (x-y)

    where :math:`\Sigma` is a diagonal covariance matrix with positive entries.

    :param torch.Tensor sigma: standard deviation for each pixel (square root of the variance).
        Default: None.
    """

    def __init__(self, sigma: float | torch.Tensor=1.):
        super().__init__()
        # The tensor is properly sent to GPU when applying `self.to(device)`
        if torch.is_tensor(sigma):
            self.var = nn.Parameter(sigma**2, requires_grad=False)
        else:
            self.var = sigma**2


    def fn(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs):
        z = x - y # Shape = ([batch_size], [nchannels], nx, ny)
        dim = tuple(range(1, z.dim())) # Exclude batch dimension
        d = 0.5 * torch.sum(torch.abs(z)**2 / self.var, dim=dim) # Shape = ([batch_size],)
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
        return (x - y) / self.var # Shape = ([batch_size], [nchannels], nx, ny)


    def prox(self, x, y, *args, gamma=1.0, **kwargs):
        raise NotImplementedError


class Mahalanobis(dinv.optim.data_fidelity.DataFidelity):

    def __init__(self, sigma: float | torch.Tensor=1.):
        super().__init__()
        self.d = MahalanobisDistance(sigma=sigma)


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


class BaseOptim(dinv.optim.BaseOptim):

    def __init__(
            self, *args, custom_metrics: MetricDict=None,
            prior_uq: dinv.optim.Prior=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.psnr_metric = dinv.metric.PSNR()
        if custom_metrics is not None:
            self.custom_metrics = nn.ModuleDict(custom_metrics)
        else:
            self.custom_metrics = None
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


    def _update_metrics(
            self, metrics: MetricDict, x: torch.Tensor, x_gt: torch.Tensor=None
    ):
        if x_gt is not None:
            psnr = self.psnr_metric.metric(x, x_gt) # Shape = (batch_size,)
            metrics.cat("psnr", psnr)
        if self.custom_metrics is not None:
            for custom_metric_name, custom_metric_fn in self.custom_metrics.items():
                custom_metric = custom_metric_fn(
                    x, x_gt, metrics[custom_metric_name], None
                ) # Shape = (batch_size,)
                metrics.cat(custom_metric_name, custom_metric)

        return metrics


    def init_metrics_fn(self, X_init, x_gt=None):

        x_init = self.get_output(X_init)
        self.batch_size = x_init.shape[0]
        init = MetricDict(
            batch_size=self.batch_size, dtype=x_init.dtype, device=x_init.device
        )
        if x_gt is not None:
            init.init_metric("psnr")
        if self.custom_metrics is not None:
            for custom_metric_name in self.custom_metrics.keys():
                init.init_metric(custom_metric_name)
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
