__level__ = 3

from . import base_dataset, torch
from .base_dataset import SCALE
from .torch import NUM_WORKERS
from .kappatng import OPENINGANGLE

# TODO: Split this module into a module `datasets` following the DeepInverse structure,
# containing `base_dataset.py` and `torch.py`, and another module `cosmology` (or
# another suitable name) containing `cosmos.py`, `dataaugm.py`, and `kappatng.py`
