__level__ = 3

# TODO: Inherit classes from DeepInverse, in particular `deepin.models.Denoisers`
# (ProximalWiener, Starlet2d) and `deepin.models.Reconstructor` (UNetPreproc, SUNetPreproc)
# TODO: Move cqr.py and rcps.py into their own module, `uq` for instance
# (check if DeepInverse provides tools for UQ)

from . import cqr, rcps

# Denoisers
from .nn import UNet, SUNet, UNetNoiseAware, SUNetNoiseAware, DRUNet, Learnlet
from .proxwiener import ProximalWiener
from .starlet2d import Starlet2d

# Reconstructors
from .preproc_models import UNetPreproc, SUNetPreproc
from .ks import KS

MODEL_CLASSES_DENOISER = {
    "UNet": (UNet, False),
    "UNetNoiseAware": (UNetNoiseAware, True),
    "DRUNet": (DRUNet, True),
    "SUNet": (SUNet, False),
    "SUNetNoiseAware": (SUNetNoiseAware, True),
    "Learnlet": (Learnlet, True)
} # (model_class, scale_as_input)

MODEL_CLASSES_DEEPMASS = {
    "UNetPreproc": (UNetPreproc, False),
    "SUNetPreproc": (SUNetPreproc, False),
} # (model_class, scale_as_input)

MODEL_CLASSES = MODEL_CLASSES_DENOISER | MODEL_CLASSES_DEEPMASS

CQR_CLASSES = {
    "addcqr": cqr.AddCQR,
    "multcqr": cqr.MultCQR,
    "chisqcqr": cqr.ChisqCQR
}
