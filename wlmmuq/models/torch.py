import warnings
import torch
from torch import nn
import torchinfo
import deepinv as dinv

from .sunet import sunet
from .deepinv import iterativemm
from .. import utils

METRIC_DICT = {
    'mse': dinv.metric.MSE(),
    'mae': dinv.metric.MAE()
}

NITER_WIENERINIT = 12 # For DeepMass

# Default parameters for DRUNet
MODEL_SIZE_DRUNET = {
    'tiny': [8, 16, 32, 64],
    'small': [16, 32, 64, 64],
    'medium': [32, 64, 128, 256],
    'large': [64, 128, 256, 512], # Default value
}
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


    def summary(self, **kwargs):
        fake_input_data = tuple(
            self._buffers[f"_fake_input_data_{i}"] for i in range(self._n_inputs)
        )
        print(torchinfo.summary(self, input_data=fake_input_data, **kwargs))


class BaseUNet(nn.Module):
    """
    PyTorch adaptation and modification of the 'cnn_keras.py' module from DeepMass
    https://github.com/NiallJeffrey/DeepMass

    """
    def __init__(self, in_channels=1, out_channels=1, bias=True):

        super().__init__()

        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding='same', bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(16, affine=bias)
        )
        self.pool1 = nn.AvgPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding='same', bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(32, affine=bias)
        )
        self.pool2 = nn.AvgPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding='same', bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(64, affine=bias)
        )
        self.pool3 = nn.AvgPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding='same', bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(64, affine=bias)
        )
        self.pool4 = nn.AvgPool2d(2)

        self.encdeep = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding='same', bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(64, affine=bias)
        )

        # Decoder convolutions
        self.decdeep = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding='same', bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(64, affine=bias),
        )
        self.dec5 = nn.Sequential(
            nn.BatchNorm2d(128, affine=bias),
            nn.Conv2d(128, 64, 3, padding='same', bias=bias),
            nn.ReLU()
        )
        self.dec6 = nn.Sequential(
            nn.BatchNorm2d(96, affine=bias),
            nn.Conv2d(96, 32, 3, padding='same', bias=bias),
            nn.ReLU()
        )
        self.dec7 = nn.Sequential(
            nn.BatchNorm2d(48, affine=bias),
            nn.Conv2d(48, 16, 3, padding='same', bias=bias),
            nn.ReLU()
        )

        # Final convolution layer
        self.final = nn.Conv2d(16, out_channels, 1)

        # Upsampling and concatenation layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.concatenate = Concatenate()


    def forward(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        xdeep = self.encdeep(self.pool4(x4))

        updeep = self.upsample(xdeep)
        mergedeep = self.concatenate([x4, updeep])
        xdeep2 = self.decdeep(mergedeep)

        up5 = self.upsample(xdeep2)
        merge5 = self.concatenate([x3, up5])
        x5 = self.dec5(merge5)

        up6 = self.upsample(x5)
        merge6 = self.concatenate([x2, up6])
        x6 = self.dec6(merge6)

        up7 = self.upsample(x6)
        merge7 = self.concatenate([x1, up7])
        x7 = self.dec7(merge7)

        out = self.final(x7)

        return out


class UNet(ModelMixin, BaseUNet):

    def _preprocess_kwargs(
            self, map_size=None, **kwargs
    ):
        # map_size is discarded
        return kwargs

    def forward(self, inp, sigma=None, **kwargs):
        r"""
        The noise level is not used in this model.

        :param torch.Tensor x: noisy image, of shape B, C, W, H.
        :param float sigma: noise level (not used).
        """
        # The signature of this forward method follows the specifications of DeepInverse,
        # to be able to use the `Trainer` class from DeepInverse for training.
        return super().forward(inp, **kwargs)

    def _get_fake_input_data(self, map_size, in_channels):
        return (torch.randn(1, in_channels, map_size, map_size),)


class DRUNet(ModelMixin, dinv.models.DRUNet):

    def _preprocess_kwargs(
            self, map_size=None, model_size: str=None,
            act_mode: str=ACT_MODE_DRUNET,
            downsample_mode: str=DOWNSAMPLE_MODE_DRUNET,
            pretrained=False, **kwargs
    ):
        # map_size is discarded
        if model_size is not None:
            if model_size not in MODEL_SIZE_DRUNET:
                raise ValueError(
                    f"Unknown model size: {model_size}. "
                    f"Available sizes: {list(MODEL_SIZE_DRUNET.keys())}."
                )
            kwargs.update(nc=MODEL_SIZE_DRUNET[model_size])
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


class BaseSUNetNoiseAware(sunet.SUNet):

    def __init__(self, in_chans=3, **kwargs):
        # On additional input channel for noise level
        super().__init__(in_chans=in_chans + 1, **kwargs)


    def forward(self, x, sigma):
        r"""
        Run the denoiser on image with noise level :math:`\sigma`, similar to DRUNet.

        :param torch.Tensor x: noisy image
        :param float, torch.Tensor sigma: noise level. If ``sigma`` is a float, it is used for all images in the batch.
            If ``sigma`` is a tensor, it must be of shape ``(batch_size,)``.
        """
        # This code block is copy-pasted from the original SUNet implementation
        if isinstance(sigma, torch.Tensor):
            if sigma.ndim > 0:
                noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                noise_level_map = noise_level_map.expand(-1, 1, x.size(2), x.size(3))
            else:
                noise_level_map = torch.ones(
                    (x.size(0), 1, x.size(2), x.size(3)), device=x.device
                ) * sigma[None, None, None, None]
        else:
            noise_level_map = (
                torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
                * sigma
            )
        x = torch.cat((x, noise_level_map), 1)
        # End of copy-pasted code block

        return super().forward(x)


class SUNetMixin(ModelMixin):

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


class SUNet(SUNetMixin, sunet.SUNet):

    def forward(self, inp, sigma=None, **kwargs):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image, of shape B, C, W, H.
        :param float sigma: noise level (not used).
        """
        # The signature of this forward method follows the specifications of DeepInverse,
        # to be able to use the `Trainer` class from DeepInverse for training.
        return super().forward(inp, **kwargs)


    def _get_fake_input_data(self, map_size, in_channels):
        return (torch.randn(1, in_channels, map_size, map_size),)


class SUNetNoiseAware(SUNetMixin, BaseSUNetNoiseAware):

    def _get_fake_input_data(self, map_size, in_channels):
        return (torch.randn(1, in_channels, map_size, map_size), torch.randn(1,))


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


class Concatenate(nn.Module):
    r"""
    Module for concatenating a list of tensors along the channel dimension.
    """
    def forward(self, inps):
        return torch.cat(inps, dim=1)


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
# DeepMass with Wiener initialization
#=================================================================================

class IterativeWiener(nn.Module):

    def __init__(
            self, step_size: float,
            powerspectrum: torch.Tensor, std_noise: torch.Tensor,
            mask:torch.Tensor=None, niter: int=NITER_WIENERINIT
    ):
        super().__init__()
        data_fidelity = iterativemm.Mahalanobis(sigma=std_noise)
        prior = dinv.optim.PnP(iterativemm.ProximalWiener(powerspectrum))

        self.optim = iterativemm.optim_builder(
            iteration="PGD", prior=prior,
            data_fidelity=data_fidelity,
            early_stop=False, max_iter=niter, custom_init=zero_init,
            params_algo={"stepsize": step_size, "g_param": step_size},
        )
        self.physics = iterativemm.MassMapping(sigma=std_noise, mask=mask)


    def forward(self, gamma_noisy):
        return self.optim(gamma_noisy, self.physics)


class WienerInitMixin:

    def __init__(
            self, args_wienerinit: dict, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.wiener_init = IterativeWiener(**args_wienerinit)


    def forward(self, inp, *args, **kwargs):
        inp = self.wiener_init(inp)
        out = super().forward(inp, *args, **kwargs)
        return out

    def _get_fake_input_data(self, map_size, in_channels):
        return (torch.randn(1, in_channels, map_size, map_size, dtype=torch.complex64),)


class UNetWienerInit(WienerInitMixin, UNet):
    pass

class SUNetWienerInit(WienerInitMixin, SUNet):
    pass


def zero_init(y: torch.Tensor, _unused_physics):
    """The optimization algorithm is initialized with zero-valued tensors"""
    x_init = torch.zeros_like(y, dtype=torch.float32, device=y.device)
    z_init = torch.zeros_like(y, dtype=torch.float32, device=y.device)
    return {"est": (x_init, z_init)}


#=================================================================================
# Functions
#=================================================================================

def load_model(path_to_pretrained_model, **kwargs):
    raise NotImplementedError
