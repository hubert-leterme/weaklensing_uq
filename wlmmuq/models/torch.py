import warnings
import deepinv as dinv

from .cszn_models import network_unet

LOSS_DICT = {
    'mse': dinv.loss.SupLoss(metric=dinv.metric.MSE()),
    'mae': dinv.loss.SupLoss(metric=dinv.metric.MSE())
}
NC = [16, 32, 64, 64] # Number of channels
ACT_MODE = 'BR' # Activation mode: BatchNorm + ReLU

class BaseModelMixin:

    def __init__(
            self, map_size=None, offset=0., in_channels=1, out_channels=1,
            nc=NC, act_mode=ACT_MODE, mean_centering=False,
            use_bias=None, sigmoid_activation=False, **kwargs
    ):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param offset: mean value of the convergence maps (for mean centering).
            Default = 0.
        :param in_channels: number of input channels. Default = 1
        :param out_channels: number of output channels. Default = 1
        :param nc: list of numbers of channels. Default = NC
        :param mean_centering: whether to apply mean centering at the output.
            Default = False
        :param use_bias: whether to use bias in the convolutional and batch
            normalization layers (not used for DeepInverse). Default = None
        :param sigmoid_activation: whether to apply a sigmoid activation function
            at the output. Default = True
        """
        self.map_size = map_size
        self.offset = offset
        if mean_centering:
            raise NotImplementedError
        if sigmoid_activation:
            raise NotImplementedError

        super().__init__(
            in_channels=in_channels, out_channels=out_channels, nc=nc,
            downsample_mode='avgpool', bias=use_bias, act_mode=act_mode, **kwargs
        )

class CSZNMixin:
    def __init__(self, *args, in_channels=1, out_channels=1, bias=None, **kwargs):
        if bias is not None:
            kwargs.update(bias=bias)
        super().__init__(*args, in_nc=in_channels, out_nc=out_channels, **kwargs)

class DeepInvMixin:
    def __init__(self, *args, bias=True, **kwargs):
        if bias is not None:
            warnings.warn("Argument 'bias' not used in DeepInverse models.")
        super().__init__(*args, pretrained=None, **kwargs)


class UNetRes(BaseModelMixin, CSZNMixin, network_unet.UNetRes):
    pass

class ResUNet(BaseModelMixin, CSZNMixin, network_unet.ResUNet):
    pass

class DRUNet(BaseModelMixin, DeepInvMixin, dinv.models.DRUNet):
    pass


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


def load_model(path_to_pretrained_model):
    raise NotImplementedError

def print_model(model):
    print(model)
