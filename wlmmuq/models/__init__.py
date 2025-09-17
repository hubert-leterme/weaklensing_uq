from . import torch
from . import deepinv
from . import cqr, rcps

MODEL_CLASSES = {
    "UNet": (torch.UNet, False),
    "UNetNoiseAware": (torch.UNetNoiseAware, True),
    "DRUNet": (torch.DRUNet, True),
    "SUNet": (torch.SUNet, False),
    "SUNetNoiseAware": (torch.SUNetNoiseAware, True),
    "Learnlet": (torch.Learnlet, True),
    "UNetPreproc": (torch.UNetPreproc, False),
    "SUNetPreproc": (torch.SUNetPreproc, False),
} # (model_class, scale_as_input)
