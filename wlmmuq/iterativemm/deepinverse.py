import torch
import deepinv as dinv

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
        self.var = sigma**2 # Float or torch.Tensor, shape = (nx, ny)


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

        x = x + self.offset
        out = super().prox(x, sigma_denoiser, *args, **kwargs)
        if self.offset_out:
            out = out - self.offset
        if self.meancentering:
            out = out - torch.mean(out, dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)

        return out
