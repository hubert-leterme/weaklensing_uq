from . import torch
from . import deepinv
from . import cqr, rcps

MODEL_CLASSES_DENOISER = {
    "UNet": (torch.UNet, False),
    "UNetNoiseAware": (torch.UNetNoiseAware, True),
    "DRUNet": (torch.DRUNet, True),
    "SUNet": (torch.SUNet, False),
    "SUNetNoiseAware": (torch.SUNetNoiseAware, True),
    "Learnlet": (torch.Learnlet, True)
} # (model_class, scale_as_input)

MODEL_CLASSES_DEEPMASS = {
    "UNetPreproc": (deepinv.preproc_models.UNetPreproc, False),
    "SUNetPreproc": (deepinv.preproc_models.SUNetPreproc, False),
} # (model_class, scale_as_input)

MODEL_CLASSES = MODEL_CLASSES_DENOISER | MODEL_CLASSES_DEEPMASS

CQR_CLASSES = {
    "addcqr": cqr.AddCQR,
    "multcqr": cqr.MultCQR,
    "chisqcqr": cqr.ChisqCQR
}
