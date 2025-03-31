from .cszn_models import network_unet

class BaseModelMixin:

    def __init__(
            self, map_size=None, offset=0., in_channels=1, out_channels=1,
            mean_centering=False, use_bias=True, sigmoid_activation=False, **kwargs
    ):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param offset: mean value of the convergence maps (for mean centering).
            Default = 0.
        :param in_channels: number of input channels. Default = 1
        :param out_channels: number of output channels. Default = 1
        :param mean_centering: whether to apply mean centering at the output.
            Default = False
        :param use_bias: whether to use bias in the convolutional and batch
            normalization layers. Default = True
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
            in_nc=in_channels, out_nc=out_channels, downsample_mode='avgpool',
            bias=use_bias, **kwargs
        )

class UNetRes(BaseModelMixin, network_unet.UNetRes):
    pass

class ResUNet(BaseModelMixin, network_unet.ResUNet):
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
