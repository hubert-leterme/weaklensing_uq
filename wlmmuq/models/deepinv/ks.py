__level__ = 1

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import deepinv as dinv


class KS(dinv.models.Reconstructor):
    """
    Simple Kaiser-Squires inversion.

    Parameters
    ----------
    std_gaussianfilter: float, optional
        Standard deviation of the Gaussian filter to apply to the output convergence map.
        Default is None.
    """
    def __init__(self, std_gaussianfilter: float | None = None):

        super().__init__()
        self.std_gaussianfilter = std_gaussianfilter
        if std_gaussianfilter is not None:
            radius = round(4 * std_gaussianfilter)
            self.kernel_size_gaussianfilter = 2 * radius + 1
        else:
            self.kernel_size_gaussianfilter = None


    def forward(self, gamma, physics):

        kappa_ks = physics.A_adjoint(gamma)
        if self.std_gaussianfilter is not None:
            kappa_ks = F.gaussian_blur(
                kappa_ks, kernel_size=self.kernel_size_gaussianfilter,
                sigma=self.std_gaussianfilter
            ).real

        return kappa_ks
    

    def get_var(self, physics, kernel_size=32):
        
        # TODO: Adapt function for 3D std_noise (redshift bins)
        raise NotImplementedError
        std_noise = (
            physics.noise_model.sigma
        ).unsqueeze(0).unsqueeze(0) # Shape = (1, 1, nx, ny)
        nx, ny = std_noise.shape
        dirac = torch.zeros(
            (nx, ny), device=std_noise.device
        ).unsqueeze(0).unsqueeze(0) # Shape = (1, 1, imgsize, imgsize)
        dirac[..., -1, -1] = 1.
        ksmatr = physics.A_adjoint(dirac)
        ksmatr_real = ksmatr.real
        ksmatr_imag = ksmatr.imag
        if self.std_gaussianfilter is not None:
            ksmatr_real = F.gaussian_blur(
                ksmatr_real, kernel_size=self.kernel_size_gaussianfilter,
                sigma=self.std_gaussianfilter
            )
            ksmatr_imag = F.gaussian_blur(
                ksmatr_imag, kernel_size=self.kernel_size_gaussianfilter,
                sigma=self.std_gaussianfilter
            )
        ksmatr_sqmodule = ksmatr_real**2 + ksmatr_imag**2

        # Crop convolution kernel for efficiency (fast-decaying coefficients)
        start1 = (nx - kernel_size) // 2
        start2 = (ny - kernel_size) // 2
        ksmatr_sqmodule = ksmatr_sqmodule[
            ...,
            start1:start1 + kernel_size,
            start2:start2 + kernel_size
        ]
        self.conv_uq.weight.data = ksmatr_sqmodule
        conv_uq = nn.Conv2d(
            1, 1, kernel_size=kernel_size, bias=False,
            padding='same', padding_mode='circular', requires_grad=False
        ).to(std_noise.device)
        conv_uq.weight.data = ksmatr_sqmodule

        return conv_uq(std_noise**2).squeeze(0).squeeze(0) # Shape = (nx, ny)
