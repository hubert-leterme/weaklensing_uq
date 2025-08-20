import torch
from torch import nn
import deepinv as dinv

from . import iterativemm

PARAMS_ALGO = {"lambda": 1.0, "stepsize": 1.0, "g_param": 0.05}

NITER_PER_STEP_G = 1
NITER_PER_STEP_NG = 1

_C1 = 1.0 / 16.0
_C2 = 1.0 / 4.0
_C3 = 3.0 / 8.0
STARLET_KERNEL1D = torch.tensor([_C1, _C2, _C3, _C2, _C1])
STARLET_FIRST_DETECT_SCALE = 1 # By default, discard the finest scale (high frequencies)
STARLET_LAST_SCALE_DETECTION = 0 # 0, 1 or None. Default = 0 (discard low frequencies)
STARLET_DETECTION_THRESHOLD = 5.0
STARLET_L2NORM = True
STARLET_ONLY_POSITIVE = True
STARLET_GEN2 = True
STARLET_ENFORCE_GEN1_TABNORM = True
STARLET_RETAIN_PREVIOUS_REC = True

class MCAIteration(nn.Module):

    def __init__(
            self,
            iterator_g: dinv.optim.OptimIterator,
            iterator_ng: dinv.optim.OptimIterator,
            niter_per_step_g: int=NITER_PER_STEP_G,
            niter_per_step_ng: int=NITER_PER_STEP_NG,
            update_ng_first: bool=False
    ):
        # Switch arguments if required
        if update_ng_first:
            iterator_g, iterator_ng = iterator_ng, iterator_g
            niter_per_step_g, niter_per_step_ng = (
                niter_per_step_ng, niter_per_step_g
            )

        super().__init__()
        self.iterator_g = iterator_g
        self.iterator_ng = iterator_ng
        self.niter_per_step_g = niter_per_step_g
        self.niter_per_step_ng = niter_per_step_ng

        self.g_first = _ComponentWrapper(
            iterator_g.g_first, iterator_ng.g_first
        )
        self.F_fn = _ComponentWrapper(
            iterator_g.F_fn, iterator_ng.F_fn
        )
        self.has_cost = _ComponentWrapper(
            iterator_g.has_cost, iterator_ng.has_cost
        )
        self.update_ng_first = update_ng_first


    def forward(
            self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        # Retrieve the Gaussian and non-Gaussian components
        x_g, x_ng = get_tensor_components(X["est"][0])
        z_g, z_ng = get_tensor_components(X["est"][1])
        if X["cost"] is not None:
            F_g, F_ng = X["cost"].get_components()
        else:
            F_g = F_ng = None

        X_g = {"est": (x_g, z_g), "cost": F_g}
        X_ng = {"est": (x_ng, z_ng), "cost": F_ng}

        # Retrieve the parameters for each step
        cur_data_fidelity_g, cur_data_fidelity_ng = cur_data_fidelity.get_components()
        cur_prior_g, cur_prior_ng = cur_prior.get_components()
        cur_params_g, cur_params_ng = _unmerge_dict(cur_params)

        # Switch variables if required
        if self.update_ng_first:
            X_g, X_ng = X_ng, X_g
            cur_data_fidelity_g, cur_data_fidelity_ng = (
                cur_data_fidelity_ng, cur_data_fidelity_g
            )
            cur_prior_g, cur_prior_ng = cur_prior_ng, cur_prior_g
            cur_params_g, cur_params_ng = (
                cur_params_ng, cur_params_g
            )

        # Compute the residual and update the Gaussian component
        y_g = _get_residual(y, x_ng, physics)
        for _ in range(self.niter_per_step_g):
            X_g = self.iterator_g(
                X_g, cur_data_fidelity_g, cur_prior_g, cur_params_g, y_g,
                physics, *args, **kwargs
            )
        x_g = X_g["est"][0]
        z_g = X_g["est"][1]
        F_g = X_g["cost"]

        # Compute the residual and update the non-Gaussian component
        y_ng = _get_residual(y, x_g, physics)
        for _ in range(self.niter_per_step_ng):
            X_ng = self.iterator_ng(
                X_ng, cur_data_fidelity_ng, cur_prior_ng, cur_params_ng, y_ng,
                physics, *args, **kwargs
            )
        x_ng = X_ng["est"][0]
        z_ng = X_ng["est"][1]
        F_ng = X_ng["cost"]

        # Switch back variables if required
        if self.update_ng_first:
            x_g, x_ng = x_ng, x_g
            z_g, z_ng = z_ng, z_g
            F_g, F_ng = F_ng, F_g

        x = stack_tensor_components(x_g, x_ng)
        z = stack_tensor_components(z_g, z_ng)
        F = _ComponentWrapper(F_g, F_ng)

        return {"est": (x, z), "cost": F}


class BaseMCALens(iterativemm.BaseOptim):

    def __init__(
            self,
            iterator_g, iterator_ng,
            niter_per_step_g=NITER_PER_STEP_G,
            niter_per_step_ng=NITER_PER_STEP_NG,
            params_algo_g=PARAMS_ALGO.copy(), params_algo_ng=PARAMS_ALGO.copy(),
            data_fidelity_g=None, data_fidelity_ng=None,
            prior_g=None, prior_ng=None,
            custom_init=iterativemm.zero_init,
            set_output=lambda x: {"est": (x,)},
            update_ng_first=False, **kwargs
    ):
        iterator = MCAIteration(
            iterator_g, iterator_ng,
            niter_per_step_g=niter_per_step_g,
            niter_per_step_ng=niter_per_step_ng,
            update_ng_first=update_ng_first
        )
        params_algo = _merge_dict(params_algo_g, params_algo_ng)
        data_fidelity = _ModuleWrapper(data_fidelity_g, data_fidelity_ng)
        prior = _ModuleWrapper(prior_g, prior_ng)
        if custom_init is not None:
            custom_init = _wrap_custom_init(custom_init, custom_init)
        super().__init__(
            iterator,
            params_algo=params_algo,
            data_fidelity=data_fidelity,
            prior=prior,
            custom_init=custom_init,
            wiener_estimate=None,
            **kwargs
        )
        self.set_output = set_output


    def init_metrics_fn(self, X_init, x_gt=None):
        X_init_sum = self._add_components_for_metrics(X_init)
        return super().init_metrics_fn(X_init_sum, x_gt)


    def update_metrics_fn(
            self, metrics: iterativemm.MetricDict, X_prev, X, x_gt=None
    ):
        X_prev_sum = self._add_components_for_metrics(X_prev)
        X_sum = self._add_components_for_metrics(X)
        return super().update_metrics_fn(metrics, X_prev_sum, X_sum, x_gt)


    def _add_components_for_metrics(self, X):

        x = self.get_output(X)
        x_out = add_tensor_components(x)
        X_out = X.copy()
        X_out.update(self.set_output(x_out))

        return X_out


class Starlet2d(nn.Module):

    def __init__(
            self, in_channels, nx, ny, ns: int | None=None,
            kernel1d: torch.Tensor=STARLET_KERNEL1D,
            first_detect_scale: int=STARLET_FIRST_DETECT_SCALE,
            last_scale_detection: int | None=STARLET_LAST_SCALE_DETECTION,
            detection_threshold: float=STARLET_DETECTION_THRESHOLD,
            l2norm: bool=STARLET_L2NORM,
            only_positive: bool=STARLET_ONLY_POSITIVE,
            gen2: bool=STARLET_GEN2,
            enforce_gen1_tabnorm: bool=STARLET_ENFORCE_GEN1_TABNORM,
            retain_previous_rec: bool=STARLET_RETAIN_PREVIOUS_REC
    ):
        super().__init__()

        # Number of scales
        if ns is None or ns == 0:
            min_dim = torch.tensor(min(nx, ny))
            ns = int(torch.log(min_dim))

        self.in_channels = in_channels
        self.nx = nx
        self.ny = ny
        self.ns = ns
        self.first_detect_scale = first_detect_scale
        self.last_scale_detection = last_scale_detection
        self.detection_threshold = detection_threshold
        self.l2norm = l2norm
        self.only_positive = only_positive
        self.gen2 = gen2
        self.enforce_gen1_tabnorm = enforce_gen1_tabnorm
        self.retain_previous_rec = retain_previous_rec

        # Normalisation coefficients at each scale, to be set at first need
        self.register_buffer("tabnorm", None) # Shape = (ns,)

        # Mask of active wavelet coefficients, to be set during the first forward pass
        self.register_buffer("active_coefs", None) # Shape = (nimgs, in_channels, ns, nx, ny)

        # Previous reconstruction, to be set after each forward pass, if required
        self.register_buffer("x_prev", None) # Shape = (nimgs, in_channels, nx, ny)

        # Starlet convolution kernel
        kernel_size = kernel1d.numel()
        kernel2d = torch.outer(kernel1d, kernel1d)
        kernel2d = kernel2d.view(1, 1, kernel_size, kernel_size)
        kernel2d = (
            kernel2d.repeat(in_channels, 1, 1, 1)
        ) # Shape = (in_channels, 1, kernel_size, kernel_size)

        # List of convolution layers
        self.convlist = nn.ModuleList()
        step_hole = 1
        for _ in range(ns - 1):
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels,
                kernel_size=kernel_size, groups=in_channels,
                bias=False, padding="same", padding_mode="reflect",
                dilation=step_hole
            )
            with torch.no_grad():
                conv.weight.copy_(kernel2d)
            self.convlist.append(conv)

            step_hole *= 2


    @property
    def device(self):
        return self.convlist[0].weight.device


    def reset_buffers(self):
        self.tabnorm = None
        self.active_coefs = None
        self.x_prev = None


    def forward(
            self, x: torch.Tensor, sigma: float | torch.Tensor
    ):
        if self.x_prev is not None:
            x -= self.x_prev
        wt = self.dec(x) # Wavelet decomposition
        if self.active_coefs is None:
            self._set_active_coefs(wt, sigma) # Set for all subsequent forward passes
        wt *= self.active_coefs # Projection onto the support of active coefficients
        x_denoised = self.rec(wt) # Wavelet reconstruction
        if self.x_prev is not None:
            x_denoised += self.x_prev
        if self.retain_previous_rec:
            self.x_prev = x_denoised.clone()

        return x_denoised


    def dec(self, x):
        wt = self._star2d(x, gen2=self.gen2)
        if self.l2norm:
            if self.tabnorm is None:
                self._set_tabnorm()
            wt /= self.tabnorm
        return wt


    def rec(self, wt):
        if self.l2norm:
            if self.tabnorm is None:
                self._set_tabnorm()
            wt *= self.tabnorm
        x = self._istar2d(wt, gen2=self.gen2)
        return x


    def _set_active_coefs(self, wt, sigma):

        detection_threshold = self.detection_threshold * torch.ones(
            self.ns, 1, 1, device=self.device
        ) # shape = (ns, 1, 1)
        detection_threshold[0] += 1 # TODO: why? See cosmostat repository

        inp = torch.abs(wt) if not self.only_positive else wt
        active_coefs = (
            inp > sigma * detection_threshold
        ) # shape = (nimgs, in_channels, ns, nx, ny)
        active_coefs[..., :self.first_detect_scale, :, :] = 0
        if self.last_scale_detection is not None:
            active_coefs[..., -1, :, :] = self.last_scale_detection

        # Update buffer
        self.active_coefs = active_coefs


    def _set_tabnorm(self):
        """
        Compute the normalisation coefficients at each scale.
        """
        x = torch.zeros(
            1, 1, self.nx, self.ny, device=self.device
        ) # Shape = (1, 1, nx, ny)
        x[..., self.nx // 2, self.ny // 2] = 1.0
        if self.enforce_gen1_tabnorm:
            # Specifically set gen2 = False for computing the normalization coefficients
            # (similar to the cosmostat repository)
            gen2 = False
        else:
            gen2 = self.gen2
        wt = self._star2d(x, gen2=gen2) # Shape = (1, 1, ns, nx, ny)

        # Update buffer
        self.tabnorm = (
            torch.linalg.norm(wt, dim=(-2, -1), keepdim=True)
        ) # Shape = (1, 1, ns, 1, 1)


    def _star2d(self, x, gen2=STARLET_GEN2):

        nimgs = x.shape[0]
        wt = torch.zeros(
            nimgs, self.in_channels, self.ns, self.nx, self.ny,
            device=self.device
        ) # Shape = (nimgs, in_channels, ns, nx, ny)
        for i in range(self.ns - 1):
            y = self.convlist[i](x) # Shape = (nimgs, in_channels, nx, ny)
            if gen2:
                z = self.convlist[i](y) # Shape = (nimgs, in_channels, nx, ny)
                wt[..., i, :, :] = x - z
            else:
                wt[..., i, :, :] = x - y
            x = y # Shape = (nimgs, in_channels, nx, ny)
        wt[..., -1, :, :] = x # Residual

        return wt


    def _istar2d(self, wt, gen2=STARLET_GEN2):

        x = wt[..., -1, :, :] # Shape = (nimgs, in_channels, nx, ny)
        for i in range(self.ns - 1):
            j = -1 - i # We start with the last index
            y = self.convlist[j](x)
            if gen2:
                x = wt[..., j - 1, :, :] + y
            else:
                z = self.convlist[j](wt[..., j - 1, :, :])
                x = wt[..., j - 1, :, :] + y + z

        return x


class _ComponentWrapper:
    """
    Wrapper class to hold Gaussian and non-Gaussian components.
    This is used instead of a tuple to avoid being considered
    as an iterable by the optimizers.
    """
    def __init__(self, val_g, val_ng):
        self.g = val_g
        self.ng = val_ng

    def get_components(self):
        return self.g, self.ng

    def __str__(self):
        return f"{self.get_components()}"

    def __repr__(self):
        val_g, val_ng = self.get_components()
        return f"ComponentWrapper({val_g}, {val_ng})"


class _ModuleWrapper(nn.Module):
    """
    Wrapper class to hold Gaussian and non-Gaussian components of type
    `iterativemm.BaseOptim` (e.g., data fidelity or prior).
    This is used instead of `torch.nn.ModuleDict` to avoid being considered
    as an iterable by the optimizers.
    """
    def __init__(
            self,
            module_g: iterativemm.BaseOptim,
            module_ng: iterativemm.BaseOptim
    ):
        super().__init__()
        self.g = module_g
        self.ng = module_ng

    def get_components(self):
        return self.g, self.ng


def _get_residual(y, x_pred, physics):
    return y - physics.A(x_pred)

def _merge_dict(d_g, d_ng):
    d = {}
    for k in d_g.keys() | d_ng.keys():
        d[k] = _ComponentWrapper(d_g.get(k), d_ng.get(k))
    return d

def _unmerge_dict(d):
    d_g = {}
    d_ng = {}
    for k, v in d.items():
        if isinstance(v, _ComponentWrapper):
            d_g[k], d_ng[k] = v.get_components()
        else:
            d_g[k] = d_ng[k] = v
    return d_g, d_ng


def get_tensor_components(x):
    # Shape of x: (batch_size, 2, nchannels, nx, ny)
    return x[:, 0], x[:, 1]


def stack_tensor_components(x_g, x_ng):
    # Shape of x_g and x_ng: (batch_size, nchannels, nx, ny)
    return torch.stack((x_g, x_ng), dim=1)


def add_tensor_components(x):
    # Shape of x: (batch_size, 2, nchannels, nx, ny)
    return torch.sum(x, dim=1)


def _wrap_custom_init(custom_init_g, custom_init_ng):

    def fn(y, physics):
        x_g, z_g = custom_init_g(y, physics)["est"]
        x_ng, z_ng = custom_init_ng(y, physics)["est"]
        x = stack_tensor_components(x_g, x_ng)
        z = stack_tensor_components(z_g, z_ng)
        return {"est": (x, z)}

    return fn


def optim_builder_mcalens(
    iteration_g, iteration_ng,
    niter_per_step_g=NITER_PER_STEP_G,
    niter_per_step_ng=NITER_PER_STEP_NG,
    max_iter=100,
    params_algo_g=PARAMS_ALGO.copy(), params_algo_ng=PARAMS_ALGO.copy(),
    data_fidelity_g=None, data_fidelity_ng=None,
    prior_g=None, prior_ng=None,
    F_fn_g=None, F_fn_ng=None,
    g_first_g=False, g_first_ng=False,
    bregman_potential_g=None, bregman_potential_ng=None,
    update_ng_first=False, **kwargs,
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
    iterator_g = dinv.optim.optimizers.create_iterator(
        iteration_g,
        prior=prior_g,
        F_fn=F_fn_g,
        g_first=g_first_g,
        bregman_potential=bregman_potential_g,
    )
    iterator_ng = dinv.optim.optimizers.create_iterator(
        iteration_ng,
        prior=prior_ng,
        F_fn=F_fn_ng,
        g_first=g_first_ng,
        bregman_potential=bregman_potential_ng,
    )
    has_cost = iterator_g.has_cost or iterator_ng.has_cost
    return BaseMCALens(
        iterator_g, iterator_ng,
        niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
        has_cost=has_cost,
        data_fidelity_g=data_fidelity_g, data_fidelity_ng=data_fidelity_ng,
        prior_g=prior_g, prior_ng=prior_ng,
        params_algo_g=params_algo_g, params_algo_ng=params_algo_ng,
        max_iter=max_iter, update_ng_first=update_ng_first,
        **kwargs,
    ).eval()
