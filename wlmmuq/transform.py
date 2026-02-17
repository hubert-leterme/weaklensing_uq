import torch
import torch.nn as nn
import deepinv as dinv

#=================================================================================
# deepinv/transform/massmappping.py ?
#=================================================================================

class BNT(dinv.transform.Transform):
    r"""
    Parameters
    ----------
    chi: torch.Tensor, shape = (nplanes,) or (nbins,)
        Comoving distances $\chi(z_i)$ of the source planes, or harmonic mean for
        each redshift bin:
        $$
            \tilde\chi_i := \frac1{\int_{z_{i-1}}^{z_i} dz \, \frac{n_i(z)}{\chi(z)}},
        $$
        where $n_i$ denotes the (normalized) number density in the $i$-th redshift bin.
    """
    def __init__(self, chi: torch.Tensor):
        super().__init__()

        if len(chi.shape) != 1:
            raise ValueError("Parameter `chi` must be one-dimensional")
        nbins = chi.numel()
        bntmatr = torch.eye(nbins)
        bntmatr[1, 0] = -1
        idx = torch.arange(2, nbins)
        den = chi[idx] * (chi[idx - 2] - chi[idx - 1])
        bntmatr[idx, idx - 2] = chi[idx - 2] * (chi[idx - 1] - chi[idx]) / den
        bntmatr[idx, idx - 1] = chi[idx - 1] * (chi[idx] - chi[idx - 2]) / den
        bntmatr_inv = torch.linalg.inv(bntmatr)

        self.bntmatr = nn.Parameter(bntmatr, requires_grad=False)
        self.bntmatr_inv = nn.Parameter(bntmatr_inv, requires_grad=False)


    def _get_params(self, x: torch.Tensor) -> dict:
        return {}


    def _transform(
            self, x: torch.Tensor,
            which_way: int = 1,     # > 0 for BNT, < 0 for inverse BNT
            transpose: bool = False
    ):
        if which_way > 0:
            w = self.bntmatr
        elif which_way < 0:
            w = self.bntmatr_inv
        else:
            raise ValueError("Argument 'which_way' must be non-zero")
        if transpose:
            w = w.T

        return torch.einsum("ibxy,ob->ioxy", x, w)
