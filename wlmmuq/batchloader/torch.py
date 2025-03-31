import torch
from torch.utils import data

from . import batchloader as wlbl

class HDF5BatchLoader(wlbl.HDF5BatchLoader, data.Dataset):

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        idx_file = self.idx[idx]
        return self._load_maps(idx_file)

    def _convert_to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32)

    def _add_newaxis_arr(self, arr: torch.tensor) -> torch.tensor:
        return arr.unsqueeze(-3) # Shape = ([nimgs, ]1, H, W)

class BaseHDF5BatchLoaderDenoiser(wlbl.DenoiserMixin, HDF5BatchLoader):
    pass

class HDF5BatchLoaderDeepMass(wlbl.MomentNetworkMixin, HDF5BatchLoader):
    """Batch loader for training DeepMass."""

class HDF5BatchLoaderDenoiser(wlbl.MomentNetworkMixin, BaseHDF5BatchLoaderDenoiser):
    """Batch loader for training a Gaussian denoiser."""
