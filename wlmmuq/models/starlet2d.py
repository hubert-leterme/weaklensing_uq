__level__ = 1

import torch
from torch import nn

from .. import utils, callbacks

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
STARLET_RETAIN_PREVIOUS_REC = True # If true, then the starlet decomposition is performed on the
                                   # residuals only (as in MCALens)

class Starlet2d(nn.Module):

    def __init__(
            self, in_channels, nx, ny, ns: int | None = None,
            kernel1d: torch.Tensor = STARLET_KERNEL1D,
            first_detect_scale: int = STARLET_FIRST_DETECT_SCALE,
            last_scale_detection: int | None=STARLET_LAST_SCALE_DETECTION,
            detection_threshold: float = STARLET_DETECTION_THRESHOLD,
            l2norm: bool = STARLET_L2NORM,
            only_positive: bool = STARLET_ONLY_POSITIVE,
            gen2: bool = STARLET_GEN2,
            enforce_gen1_tabnorm: bool = STARLET_ENFORCE_GEN1_TABNORM,
            retain_previous_rec: bool = STARLET_RETAIN_PREVIOUS_REC,
            meancentering: bool = True
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
        self.meancentering = meancentering

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
        self.tabnorm: torch.Tensor | None = None
        self.active_coefs: torch.Tensor | None = None
        self.x_prev: torch.Tensor | None = None


    def forward(
            self, x: torch.Tensor, sigma: float | torch.Tensor
    ):
        if self.x_prev is not None:
            # The starlet decomposition is done on the residual only
            # See Algorithm 1 in Starck et al. (2021) (MCALens)
            x -= self.x_prev
        wt = self.dec(x) # Wavelet decomposition

        if self.active_coefs is None:
            if torch.is_tensor(sigma):
                # The current shape of sigma is exptected to be (nimgs, 1, 1, 1).
                # We need to broadcast it to (nimgs, 1, 1, 1, 1) to
                # match the shape of the wavelet coefficients.
                sigma = sigma.unsqueeze(-3) # Shape = (nimgs, 1, 1, 1, 1)
            self._set_active_coefs(wt, sigma) # Set for all subsequent forward passes
        assert self.active_coefs is not None

        wt *= self.active_coefs # Projection onto the support of active coefficients
        x_denoised = self.rec(wt) # Wavelet reconstruction
        if self.only_positive:
            x_denoised = torch.relu(x_denoised)
        # Do not include self.x_prev in the positivity constrain
        # (already positive up to a mean-centering constant)
        if self.x_prev is not None:
            x_denoised += self.x_prev
        if self.meancentering:
            x_denoised = utils.meancenter(x_denoised)
        if self.retain_previous_rec:
            self.x_prev = x_denoised.clone()

        return x_denoised


    def dec(self, x: torch.Tensor) -> torch.Tensor:
        wt = self._star2d(x, gen2=self.gen2)
        if self.l2norm:
            if self.tabnorm is None:
                self._set_tabnorm()
            assert self.tabnorm is not None
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
    

class StarletResetter(callbacks.BaseCallback):

    def __init__(self, *starlet: Starlet2d):
        self.starlet: list[Starlet2d] = list(starlet)

    def on_batch_begin(self, batch):
        for starlet in self.starlet:
            starlet.reset_buffers()
