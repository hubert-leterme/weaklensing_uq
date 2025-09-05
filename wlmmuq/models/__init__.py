from . import torch
from . import deepinv
from . import cqr, rcps

MODEL_CLASSES = {
    "torch.UNet": (torch.UNet, False),
    "torch.UNetNoiseAware": (torch.UNetNoiseAware, True),
    "torch.DRUNet": (torch.DRUNet, True),
    "torch.SUNet": (torch.SUNet, False),
    "torch.SUNetNoiseAware": (torch.SUNetNoiseAware, True),
    "torch.Learnlet": (torch.Learnlet, True),
    "torch.UNetPreproc": (torch.UNetPreproc, False),
    "torch.SUNetPreproc": (torch.SUNetPreproc, False),
} # (model_class, scale_as_input)
