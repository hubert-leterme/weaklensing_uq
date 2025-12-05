import torch
from torch import nn
import deepinv as dinv

from ..torch import UNet, SUNet
from . import iterativemm

NITER_WIENER = 12

#=================================================================================
# DeepMass with Wiener or KS initialization
#=================================================================================

class WienerInit(nn.Module):

    def __init__(
            self, step_size: float,
            powerspectrum: torch.Tensor, std_noise: torch.Tensor,
            mask:torch.Tensor | None = None, niter: int = NITER_WIENER,
            noise_whitening: bool = False
    ):
        super().__init__()
        if not noise_whitening:
            param_vector = std_noise**2 # Bayesian data fidelity
        else:
            param_vector = std_noise # Noise whitening data fidelity
        data_fidelity = iterativemm.Mahalanobis(param_vector=param_vector)
        prior = dinv.optim.PnP(iterativemm.ProximalWiener(powerspectrum))

        self.optim = iterativemm.optim_builder(
            iteration="PGD",
            params_algo={"stepsize": step_size, "g_param": step_size},
            data_fidelity=data_fidelity, prior=prior,
            early_stop=False, max_iter=niter, custom_init=iterativemm.zero_init,
        )
        self.physics = iterativemm.MassMapping(sigma=std_noise, mask=mask)


    def forward(self, gamma_noisy):
        return self.optim(gamma_noisy, self.physics)


class KSInit(nn.Module):

    def __init__(
            self, std_noise: torch.Tensor, mask:torch.Tensor | None = None
    ):
        super().__init__()
        self.physics = iterativemm.MassMapping(sigma=std_noise, mask=mask)


    def forward(self, gamma_noisy):
        return self.physics.A_adjoint(gamma_noisy)


class PreprocMixin:

    def __init__(
            self, mode_preproc: str,
            *args, args_preproc: dict=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if mode_preproc == "wiener":
            preproc_class = WienerInit
        elif mode_preproc == "ks":
            preproc_class = KSInit
        else:
            raise ValueError(
                f"Unknown preprocessing mode: {mode_preproc}. "
                "Available modes: 'wiener', 'ks'."
            )
        if args_preproc is None:
            args_preproc = {}
        self.preproc = preproc_class(**args_preproc)


    def forward(self, inp, *args, **kwargs):
        with torch.no_grad():
            inp = self.preproc(inp)
        out = super().forward(inp, *args, **kwargs)
        return out


class UNetPreproc(PreprocMixin, UNet):
    pass

class SUNetPreproc(PreprocMixin, SUNet):
    pass