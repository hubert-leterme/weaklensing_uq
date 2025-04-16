import warnings
import torch
import deepinv as dinv

from .cszn_models import network_unet
from .. import utils
from .. import OFFSET

LOSS_DICT = {
    'mse': dinv.loss.SupLoss(metric=dinv.metric.MSE()),
    'mae': dinv.loss.SupLoss(metric=dinv.metric.MSE())
}
NC = [16, 32, 64, 64] # Number of channels
ACT_MODE = 'BR' # Activation mode: BatchNorm + ReLU
DOWNSAMPLE_MODE = 'avgpool'

class DRUNetMixin:

    def __init__(
            self, map_size=None, meancentering: bool=False,
            offset: float=OFFSET, in_channels=1, out_channels=1,
            small_model=False, act_mode: str=ACT_MODE,
            downsample_mode: str=DOWNSAMPLE_MODE, **kwargs
    ):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels).
            Unused.
        :param bool meancentering: whether to apply mean centering at the output of
            the network. Default = False.
        :param float offset: mean value of the convergence maps (used for mean centering).
            Default = 0.
        :param in_channels: number of input channels. Default = 1
        :param out_channels: number of output channels. Default = 1
        :param bool small_model: whether to use a small model. Default = False
        :param str act_mode: activation mode. Default = 'BR' (BatchNorm + ReLU)
        :param str downsample_mode: downsample mode. Default = 'avgpool'
        """
        if small_model:
            kwargs.update(nc=NC)
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            act_mode=act_mode, downsample_mode=downsample_mode,
            **kwargs
        )
        self.offset = offset
        self.meancentering = meancentering


    def forward(self, inp, *args, **kwargs):
        out = super().forward(inp, *args, **kwargs)
        if self.meancentering:
            out = utils.meancenter(out, offset=self.offset)
        return out


class CSZNMixin(DRUNetMixin):

    def __init__(
            self, *args, in_channels=1, out_channels=1, use_bias=True, **kwargs
    ):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels).
            Unused.
        :param offset: mean value of the convergence maps (for mean centering).
            Unused.
        :param in_channels: number of input channels. Default = 1
        :param out_channels: number of output channels. Default = 1
        :param small_model: whether to use a small model. Default = False
        :param act_mode: activation mode. Default = 'BR' (BatchNorm + ReLU)
        :param downsample_mode: downsample mode. Default = 'avgpool'
        :param use_bias: whether to use bias in the convolutional and batch
            normalization layers (not used for DeepInverse). Default = True
        """
        super().__init__(
            *args, in_nc=in_channels, out_nc=out_channels, bias=use_bias, **kwargs
        )


class UNetRes(CSZNMixin, network_unet.UNetRes):
    pass

class ResUNet(CSZNMixin, network_unet.ResUNet):
    pass

class DRUNet(DRUNetMixin, dinv.models.DRUNet):
    def __init__(self, *args, pretrained=False, **kwargs):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels).
            Unused.
        :param offset: mean value of the convergence maps (for mean centering).
            Unused.
        :param in_channels: number of input channels. Default = 1
        :param out_channels: number of output channels. Default = 1
        :param small_model: whether to use a small model. Default = False
        :param act_mode: activation mode. Default = 'BR' (BatchNorm + ReLU)
        :param downsample_mode: downsample mode. Default = 'avgpool'
        :param pretrained: whether to use a pretrained model. Default = False
        """
        if not pretrained:
            pretrained = None
        else:
            pretrained = 'download'
        super().__init__(*args, pretrained=pretrained, **kwargs)


class ScoreMatchingMixin:

    def forward(self, inp):
        inp, sigma = inp
        out = super().forward(inp) # Shape = (batch_size, 1, map_size, map_size)
        var = sigma**2 # Shape = (batch_size, 1, 1, 1)
        return inp + var * out

class UNetResScoreMatching(ScoreMatchingMixin, UNetRes):
    pass

class ResUNetScoreMatching(ScoreMatchingMixin, ResUNet):
    pass


class Trainer(dinv.Trainer):

    def __init__(self, *args, scale_as_input=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_as_input = scale_as_input

    def get_samples(self, iterators, g):
        x, y, physics = super().get_samples(iterators, g)
        if self.scale_as_input:
            if physics is not None:
                warnings.warn("Output `physics` overriden.")
            y, scale = y
            physics = scale
        return x, y, physics

    def plot(self, epoch, physics, x, y, x_net, train=True):
        if torch.is_complex(y):
            y = y.real
        super().plot(epoch, physics, x, y, x_net, train=train)


def load_model(path_to_pretrained_model, **kwargs):
    raise NotImplementedError

def print_model(model):
    print(model)
