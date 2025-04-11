import shutil
import torch
import torch.nn as nn
import deepinv as dinv

from .. import utils

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
            self.var = torch.nn.Parameter(sigma**2, requires_grad=False)
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


class PnP(dinv.optim.PnP):
    r"""
    Plug-and-play prior with offset and output mean centering:
    
    :math:`\operatorname{prox}_{\gamma \regname}(x) = \operatorname{D}_{\sigma}(x + c) - c`.


    :param Callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    """

    def __init__(
            self, denoiser, *args, offset: float=0., offset_out=True,
            meancentering=False, **kwargs
    ):
        super().__init__(denoiser, *args, **kwargs)
        self.offset = offset
        self.offset_out = offset_out
        self.meancentering = meancentering


    def prox(self, x, sigma_denoiser, *args, **kwargs):

        out = _forward_offset_meancentering(
            x, sigma_denoiser, *args, forward=super().prox,
            offset=self.offset, offset_out=self.offset_out,
            meancentering=self.meancentering, **kwargs
        )
        return out


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
            self.mask = torch.nn.Parameter(mask, requires_grad=False)
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

    def __init__(self, mask: torch.Tensor=None, **kwargs):
        super().__init__(**kwargs)
        if mask is not None:
            utils.check_mask(mask)
            self.mask = torch.nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = None

    def metric(self, x_net, x, *args, **kwargs):
        if self.mask is not None:
            x_net = x_net[..., self.mask]
            x = x[..., self.mask]
        return super().metric(x_net, x, *args, **kwargs)


class RMSE(MSE):
    def metric(self, x_net, x, *args, **kwargs):
        return super().metric(x_net, x, *args, **kwargs) ** 0.5


class OffsetMeancenteringWrapper(nn.Module):
    r"""
    Wrapper to add an offset to the input of a model and remove it from the output.
    It also allows for mean centering the output.

    :param torch.nn.Module model: Model to be wrapped.
    :param float offset: Offset to be added to the input. Default: 0.
    :param bool offset_out: If True, the offset is removed from the output. Default: True.
    :param bool meancentering: If True, the output is mean centered. Default: False.
    """

    def __init__(
            self, model: nn.Module, offset: float=0., offset_out: bool=True,
            meancentering: bool=False
    ):
        super().__init__()
        self.model = model
        self.offset = offset
        self.offset_out = offset_out
        self.meancentering = meancentering

    def forward(self, inp, *args, **kwargs):

        out = _forward_offset_meancentering(
            inp, *args, forward=self.model.forward,
            offset=self.offset, offset_out=self.offset_out,
            meancentering=self.meancentering, **kwargs
        )
        return out


class BaseOptim(dinv.optim.BaseOptim):

    def __init__(self, *args, custom_metrics: dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.psnr_metric = dinv.metric.PSNR()
        if custom_metrics is not None:
            self.custom_metrics = nn.ModuleDict(custom_metrics)
        else:
            self.custom_metrics = None
        self.batch_size = None


    def _update_metrics(
            self, metrics: dict, x: torch.Tensor, x_gt: torch.Tensor=None
    ):
        if x_gt is not None:
            psnr = self.psnr_metric.metric(x, x_gt) # Shape = (batch_size,)
            psnr = psnr.unsqueeze(1) # Shape = (batch_size, 1)
            metrics["psnr"] = torch.cat(
                [metrics["psnr"], psnr], dim=1
            ) # Shape = (batch_size, niter + 1)
        if self.custom_metrics is not None:
            for custom_metric_name, custom_metric_fn in self.custom_metrics.items():
                custom_metric = custom_metric_fn(
                    x, x_gt, metrics[custom_metric_name], None
                ) # Shape = (batch_size,)
                custom_metric = custom_metric.unsqueeze(1) # Shape = (batch_size, 1)
                metrics[custom_metric_name] = torch.cat(
                    [metrics[custom_metric_name], custom_metric], dim=1
                ) # Shape = (batch_size, niter + 1)

        return metrics


    def init_metrics_fn(self, X_init, x_gt=None):

        init = {}
        x_init = self.get_output(X_init)
        self.batch_size = x_init.shape[0]

        if x_gt is not None:
            init["psnr"] = torch.empty(
                (self.batch_size, 0),
                dtype=x_init.dtype, device=x_init.device
            )
        if self.custom_metrics is not None:
            for custom_metric_name in self.custom_metrics.keys():
                init[custom_metric_name] = torch.empty(
                    (self.batch_size, 0),
                    dtype=x_init.dtype, device=x_init.device
                )
        if self.has_cost:
            init["cost"] = torch.empty(
                (self.batch_size, 0),
                dtype=x_init.dtype, device=x_init.device
            )
        init["residual"] = torch.empty(
            (self.batch_size, 0),
            dtype=x_init.dtype, device=x_init.device
        )

        return self._update_metrics(init, x_init, x_gt)


    def update_metrics_fn(self, metrics, X_prev, X, x_gt=None):

        if metrics is not None:
            x_prev = self.get_output(X_prev)
            x = self.get_output(X)

            # Shape = (batch_size, npixels)
            diff_flattened = (x_prev - x).reshape(self.batch_size, -1)
            x_flattened = x.reshape(self.batch_size, -1)
            residual = torch.linalg.norm(diff_flattened, dim=1) / \
                torch.linalg.norm(x_flattened, dim=1)  # Shape = (batch_size,)
            residual = residual.unsqueeze(1) # Shape = (batch_size, 1)
            metrics["residual"] = torch.cat(
                [metrics["residual"], residual], dim=1
            ) # Shape = (batch_size, niter + 1)

            if self.has_cost:
                cost = X["cost"].unsqueeze(1) # Shape = (batch_size, 1)
                metrics["cost"] = torch.cat(
                    [metrics["cost"], cost], dim=1
                ) # Shape = (batch_size, niter + 1)

            metrics = self._update_metrics(metrics, x, x_gt)

        return metrics


def optim_builder(
    iteration,
    max_iter=100,
    params_algo={"lambda": 1.0, "stepsize": 1.0, "g_param": 0.05},
    data_fidelity=None,
    prior=None,
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
        params_algo=params_algo,
        max_iter=max_iter,
        **kwargs,
    ).eval()


def _forward_offset_meancentering(
        inp: torch.Tensor, *args, forward: callable=None, offset: float=0.,
        offset_out: bool=True, meancentering: bool=False, **kwargs
) -> torch.Tensor:

    # TODO: replace by a decorator?
    inp = inp + offset
    if forward is not None:
        out = forward(inp, *args, **kwargs)
    if offset_out:
        out = out - offset
    if meancentering:
        out = out - torch.mean(out, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)

    return out
