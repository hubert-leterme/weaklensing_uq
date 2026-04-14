__level__ = 0

import torch
import torch.nn as nn
import deepinv as dinv

from .. import utils

#=================================================================================
# deepinv/loss/metric/distortion.py
#=================================================================================

class MeancenterMaskMixin:

    def __init__(
            self, mask: torch.Tensor | None = None,
            meancentering: bool = True,
            per_zbin: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        if mask is not None:
            utils.check_mask(mask)
            self.mask = nn.Parameter(mask, requires_grad=False)
        else:
            self.mask = None
        self.meancentering = meancentering
        self.per_zbin = per_zbin


    def metric(
            self, x_net: torch.Tensor, x: torch.Tensor | float,
            *args, **kwargs
    ):
        if isinstance(x, float):
            x = x * torch.ones_like(x_net)
        assert isinstance(x, torch.Tensor)
        if x.shape != x_net.shape:
            raise ValueError("Shape mismatch")
        if self.meancentering:
            x_net = utils.meancenter(
                x_net, mask=self.mask,
                axis=tuple(range(1, x_net.ndim))
            )
            x = utils.meancenter(
                x, mask=self.mask,
                axis=tuple(range(1, x.ndim))
            )
        if self.mask is not None:
            x_net = x_net[..., self.mask]
            x = x[..., self.mask]

        if self.per_zbin:
            nimgs, nbins, *shape = x_net.shape
            x_net = x_net.view(-1, 1, *shape)
            x = x.view(-1, 1, *shape)
        out = super().metric(x_net, x, *args, **kwargs)
        if self.per_zbin:
            out = out.view(nimgs, nbins)
        
        return out


class SquareRootMixin:
    def metric(self, x_net, x, *args, **kwargs):
        return super().metric(x_net, x, *args, **kwargs) ** 0.5


class MSE(MeancenterMaskMixin, dinv.metric.MSE):
    pass

class MAE(MeancenterMaskMixin, dinv.metric.MAE):
    pass

class NMSE(MeancenterMaskMixin, dinv.metric.NMSE):
    pass

class RMSE(SquareRootMixin, MSE):
    """Root Mean Squared Error metric."""

class NRMSE(SquareRootMixin, NMSE):
    """Normalized Root Mean Squared Error metric."""


class BaseMetricOnLowerUpperBounds(dinv.metric.Metric):

    def metric(self, x_net, x, *args, **kwargs):
        x_lo = x_net[:, 0]
        x_hi = x_net[:, 1]
        out = self._unreduced_metric(x_lo, x_hi, x, *args, **kwargs)
        return out.mean(dim=tuple(range(1, x.ndim)), keepdim=False)

    def _unreduced_metric(self, x_lo, x_hi, x, *args, **kwargs):
        raise NotImplementedError

class MetricOnLowerUpperBounds(MeancenterMaskMixin, BaseMetricOnLowerUpperBounds):
    pass

class MiscoverageRate(MetricOnLowerUpperBounds):
    def _unreduced_metric(self, x_lo, x_hi, x, *args, **kwargs):
        return ((x < x_lo) | (x > x_hi)).to(torch.float32)

class PredInterv(MetricOnLowerUpperBounds):
    def _unreduced_metric(self, x_lo, x_hi, x, *args, **kwargs):
        return x_hi - x_lo
    

#=================================================================================
# Utility class
#=================================================================================

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
