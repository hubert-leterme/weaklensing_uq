import warnings
import torch
from torch import nn
import torchinfo
import deepinv as dinv

from .sunet import sunet
from .. import utils

METRIC_DICT = {
    'mse': dinv.metric.MSE(),
    'mae': dinv.metric.MAE()
}

# Default parameters for DRUNet
NC_DRUNET = [16, 32, 64, 64] # Number of channels
ACT_MODE_DRUNET = 'BR' # Activation mode: BatchNorm + ReLU
DOWNSAMPLE_MODE_DRUNET = 'avgpool'

# Default parameters for SUNet, as specified in the `training.yaml`
# file from the repository: https://github.com/utsav-akhaury/SUNet
PATCH_SIZE_SUNET = 4
WIN_SIZE_SUNET = 6 # Works with imgsize = 384
EMB_DIM_SUNET = 48
DEPTH_EN_SUNET = [2, 6, 8, 16] # https://github.com/megvii-research/NAFNet/issues/9
HEAD_NUM_SUNET = [8, 8, 8, 8]
MLP_RATIO_SUNET = 4.0
QKV_BIAS_SUNET = True
QK_SCALE_SUNET = 8
DROP_RATE_SUNET = 0.
ATTN_DROP_RATE_SUNET = 0.
DROP_PATH_RATE_SUNET = 0.1
APE_SUNET = False
PATCH_NORM_SUNET = True
USE_CHECKPOINTS_SUNET = False
FINAL_UPSAMPLE_SUNET = 'bilinear' # Avoids checkerboard effects

#=================================================================================
# Models
#=================================================================================

class ModelMixin:

    def __init__(
            self, map_size=None, in_channels=1, out_channels=1,
            meancentering: bool=True, onlypositive: bool=False, **kwargs
    ):
        kwargs = self._preprocess_kwargs(
            map_size=map_size, in_channels=in_channels, out_channels=out_channels,
            **kwargs
        )
        super().__init__(**kwargs)
        if hasattr(self, 'additional_output'):
            raise NotImplementedError("Attribute `additional_output` already exists.")
        if meancentering:
            if onlypositive:
                warnings.warn("`onlypositive` is ignored when `meancentering` is True.")
            self.additional_output = Meancentering()
        elif onlypositive:
            self.additional_output = nn.ReLU()
        else:
            warnings.warn("No meancentering or positivity constraint applied.")
            self.additional_output = None

        # For printing summary
        fake_input_data = self._get_fake_input_data(map_size, in_channels)
        self._n_inputs = len(fake_input_data)
        for i, inp in enumerate(fake_input_data):
            self.register_buffer(f"_fake_input_data_{i}", inp, persistent=False)


    def _preprocess_kwargs(self, **kwargs):
        raise NotImplementedError


    def forward(self, inp, *args, **kwargs):
        out = super().forward(inp, *args, **kwargs)
        if self.additional_output is not None:
            out = self.additional_output(out)
        return out


    def _get_fake_input_data(self, map_size, in_channels):
        raise NotImplementedError


    def summary(self):
        fake_input_data = tuple(
            self._buffers[f"_fake_input_data_{i}"] for i in range(self._n_inputs)
        )
        print(torchinfo.summary(self, input_data=fake_input_data))


class DRUNet(ModelMixin, dinv.models.DRUNet):

    def _preprocess_kwargs(
            self, map_size=None, small_model=False,
            act_mode: str=ACT_MODE_DRUNET,
            downsample_mode: str=DOWNSAMPLE_MODE_DRUNET,
            pretrained=False, **kwargs
    ):
        # map_size is discarded
        if small_model:
            kwargs.update(nc=NC_DRUNET)
        if not pretrained:
            pretrained = None
        else:
            pretrained = 'download'
        kwargs.update(
            act_mode=act_mode, downsample_mode=downsample_mode,
            pretrained=pretrained
        )

        return kwargs


    def _get_fake_input_data(self, map_size, in_channels):
        return (torch.randn(1, in_channels, map_size, map_size), torch.randn(1,))


class SUNet(ModelMixin, sunet.SUNet):

    def _preprocess_kwargs(
            self, map_size=None, in_channels=1, out_channels=1,
            patch_size=PATCH_SIZE_SUNET,
            embed_dim=EMB_DIM_SUNET,
            depths=None, num_heads=None,
            window_size=WIN_SIZE_SUNET,
            mlp_ratio=MLP_RATIO_SUNET,
            qkv_bias=QKV_BIAS_SUNET,
            qk_scale=QK_SCALE_SUNET,
            drop_rate=DROP_RATE_SUNET,
            drop_path_rate=DROP_PATH_RATE_SUNET,
            ape=APE_SUNET,
            patch_norm=PATCH_NORM_SUNET,
            use_checkpoint=USE_CHECKPOINTS_SUNET,
            final_upsample=FINAL_UPSAMPLE_SUNET, **kwargs
    ):
        if depths is None:
            kwargs.update(depth=DEPTH_EN_SUNET)
        if num_heads is None:
            kwargs.update(num_heads=HEAD_NUM_SUNET)
        kwargs.update(
            img_size=map_size,
            patch_size=patch_size,
            in_chans=in_channels, out_chans=out_channels,
            embed_dim=embed_dim, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, ape=ape,
            patch_norm=patch_norm, use_checkpoint=use_checkpoint,
            final_upsample=final_upsample
        )
        return kwargs


    def forward(self, inp, sigma=None, **kwargs):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image, of shape B, C, W, H.
        :param float sigma: noise level (not used).
        """
        # The signature of this forward method follows the specifications of DeepInverse,
        # to be able to use the `Trainer` class for training.
        return super().forward(inp, **kwargs)


    def _get_fake_input_data(self, map_size, in_channels):
        return (torch.randn(1, in_channels, map_size, map_size),)


class ScoreMatchingMixin:

    def forward(self, inp):
        inp, sigma = inp
        out = super().forward(inp) # Shape = (batch_size, 1, map_size, map_size)
        var = sigma**2 # Shape = (batch_size, 1, 1, 1)
        return inp + var * out


#=================================================================================
# Building blocks (e.g., activation modules)
#=================================================================================

class Meancentering(nn.Module):
    r"""
    Module for meancentering the input tensor.
    """
    def forward(self, x):
        return utils.meancenter(x)


#=================================================================================
# Class inheriting from dinv.Trainer, used for training
#=================================================================================

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


#=================================================================================
# Losses
#=================================================================================

class Order2SupLoss(dinv.loss.SupLoss):
    r"""
    Supervised loss for order-2 moment networks

    The supervised loss is defined as

    .. math::

        \|(x - F(y))^2 - G(y)\|^2

    where :math:`F(y)` is the reconstructed signal using a previously-trained network,
    :math:`G(y)` is the output of the order-2 moment network and :math:`x` is the ground truth target.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.
    If called with arguments ``x_net, x``, this is simply a wrapper for the metric ``metric``.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    """
    def __init__(self, order1_model: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.order1_model = order1_model

    def forward(self, x_net, x, y, physics=None, **kwargs):
        with torch.no_grad():
            x_pred = self.order1_model(y, physics) # physics = sigma in the case of DRUNet
        x = (x - x_pred)**2
        return super().forward(x_net, x, **kwargs)


#=================================================================================
# Functions
#=================================================================================

def load_model(path_to_pretrained_model, **kwargs):
    raise NotImplementedError
