__level__ = 1

import torch
import torch.nn as nn
import deepinv as dinv

from . import utils

#=================================================================================
# deepinv/physics/noise.py
#=================================================================================

class ComplexGaussianNoise(dinv.physics.GaussianNoise):
    """
    Proper complex Gaussian noise model.
    """
    # TODO: check whether __add__ and __mul__ must be redefined
    def __init__(self, sigma_real: float | torch.Tensor = 0., **kwargs):
        super().__init__(sigma=sigma_real, **kwargs)

    def forward(self, x, sigma_real=None, seed=None, **kwargs):
        out_real = super().forward(x.real, sigma=sigma_real, seed=seed, **kwargs)
        seed = seed + 1 if seed is not None else None
        out_imag = super().forward(x.imag, sigma=sigma_real, seed=seed, **kwargs)
        return out_real + 1j * out_imag


#=================================================================================
# deepinv/physics/massmappping.py ?
#=================================================================================

class MassMapping(dinv.physics.LinearPhysics):

    def __init__(
            self, sigma: float | torch.Tensor = 0.,
            mask: torch.Tensor | None = None, **kwargs
    ):
        noise_model = ComplexGaussianNoise(sigma_real=sigma)
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
