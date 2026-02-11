__level__ = 1

import torch
from torch import nn

from .. import utils

class ProximalWiener(nn.Module):

    def __init__(self, powerspectrum, meancentering=True):
        super().__init__()
        self.register_buffer("powerspectrum", powerspectrum)
        self.meancentering = meancentering


    def forward(
            self, inp: torch.Tensor,
            g_param: float | torch.Tensor
    ):
        # Either one scalar parameter for the whole batch, or one specific
        # parameter for each image
        out = torch.fft.fft2(inp)
        out /= (1 + g_param / self.powerspectrum)
        out = torch.fft.ifft2(out)
        if self.meancentering:
            out = utils.meancenter(out, axis=tuple(range(1, out.ndim)))

        return out.real
