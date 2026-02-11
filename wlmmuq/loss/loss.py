__level__ = 0

import torch
import torch.nn as nn
import deepinv as dinv

#=================================================================================
# deepinv/loss/sup.py
#=================================================================================

class Order2SupLoss(dinv.loss.SupLoss):
    r"""
    Supervised loss for order-2 moment networks

    The supervised loss is defined as

    .. math::

        \|(x - F(y))^2 - G(y)\|^2

    where :math:`F(y)` is the reconstructed signal using a previously-trained network,
    :math:`G(y)` is the output of the order-2 moment network and :math:`x` is the ground truth target.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.
    If called with arguments ``x_net, x``, this is simply a wrapper for the metric ``metric``.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    """
    def __init__(self, order1_model: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.order1_model = order1_model

    def forward(self, x_net, x, y, physics=None, **kwargs):
        with torch.no_grad():
            x_pred = self.order1_model(y, physics) # physics = sigma in the case of DRUNet
        x = (x - x_pred)**2
        return super().forward(x_net=x_net, x=x, y=y, physics=physics, **kwargs)
