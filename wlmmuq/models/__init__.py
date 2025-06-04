from . import torch

from .. import USE_TENSORFLOW
if USE_TENSORFLOW:
    from . import tensorflow

MODEL_CLASSES = {
    "torch.UNet": (torch.UNet, False),
    "torch.DRUNet": (torch.DRUNet, True),
    "torch.SUNet": (torch.SUNet, False),
    "torch.SUNetNoiseAware": (torch.SUNetNoiseAware, True),
} # (model_class, scale_as_input)
if USE_TENSORFLOW:
    MODEL_CLASSES.update({
        "tensorflow.UNet": (tensorflow.UNet, False),
        "tensorflow.UNetScoreMatching": (tensorflow.UNetScoreMatching, True)
    })
