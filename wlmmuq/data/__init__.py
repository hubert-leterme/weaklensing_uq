from . import base_dataset, torch
from .base_dataset import SCALE
from .torch import NUM_WORKERS

from .. import USE_TENSORFLOW
if USE_TENSORFLOW:
    from . import tensorflow
