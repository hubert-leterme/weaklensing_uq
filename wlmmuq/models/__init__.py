from . import torch
from . import deepinv

MODEL_CLASSES = {
    "torch.UNet": (torch.UNet, False),
    "torch.UNetWienerInit": (torch.UNetWienerInit, False),
    "torch.SUNetWienerInit": (torch.SUNetWienerInit, False),
    "torch.DRUNet": (torch.DRUNet, True),
    "torch.SUNet": (torch.SUNet, False),
    "torch.SUNetNoiseAware": (torch.SUNetNoiseAware, True),
} # (model_class, scale_as_input)
