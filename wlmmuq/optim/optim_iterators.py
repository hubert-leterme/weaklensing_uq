__level__ = 0

import torch.nn as nn
import deepinv as dinv

from .. import utils

NITER_PER_STEP_G = 1
NITER_PER_STEP_NG = 1

class MCAIteration(nn.Module):

    def __init__(
            self,
            iterator_g: dinv.optim.OptimIterator,
            iterator_ng: dinv.optim.OptimIterator,
            niter_per_step_g: int = NITER_PER_STEP_G,
            niter_per_step_ng: int = NITER_PER_STEP_NG,
            update_ng_first: bool = False
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

        self.g_first = utils.ComponentWrapper(
            iterator_g.g_first, iterator_ng.g_first
        )
        self.F_fn = utils.ComponentWrapper(
            iterator_g.F_fn, iterator_ng.F_fn
        )
        self.has_cost = utils.ComponentWrapper(
            iterator_g.has_cost, iterator_ng.has_cost
        )
        self.update_ng_first = update_ng_first


    def forward(
            self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs
    ):
        # Retrieve the Gaussian and non-Gaussian components
        x_g, x_ng = utils.get_tensor_components(X["est"][0])
        z_g, z_ng = utils.get_tensor_components(X["est"][1])
        if X["cost"] is not None:
            F_g, F_ng = X["cost"].get_components()
        else:
            F_g = F_ng = None

        X_g = {"est": (x_g, z_g), "cost": F_g}
        X_ng = {"est": (x_ng, z_ng), "cost": F_ng}

        # Retrieve the parameters for each step
        cur_data_fidelity_g, cur_data_fidelity_ng = cur_data_fidelity.get_components()
        cur_prior_g, cur_prior_ng = cur_prior.get_components()
        cur_params_g, cur_params_ng = utils.unmerge_dict(cur_params)

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

        x = utils.stack_tensor_components(x_g, x_ng)
        z = utils.stack_tensor_components(z_g, z_ng)
        F = utils.ComponentWrapper(F_g, F_ng)

        return {"est": (x, z), "cost": F}
    

def _get_residual(y, x_pred, physics):
    return y - physics.A(x_pred)
