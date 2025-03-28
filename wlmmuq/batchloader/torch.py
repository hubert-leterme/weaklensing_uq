import torch
from torch.utils import data
import numpy as np

from . import batchloader as wlbl

class HDF5BatchLoader(wlbl.HDF5BatchLoader, data.Dataset):

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        idx_file = self.idx[idx]
        out_dict = self._load_maps(idx_file)
        out = self._prepare_output(out_dict)
        return out

    def _load_maps(self, idx, transform: callable = None) -> dict:
        def to_torch(arr):
            if transform is not None:
                arr = transform(arr)
            return torch.tensor(arr, dtype=torch.float32)
        return super()._load_maps(idx, transform=to_torch)

    def _add_newaxis_arr(self, arr: np.ndarray) -> np.ndarray:
        return arr[..., np.newaxis, :, :] # Shape = ([nimgs, ]1, H, W)


class HDF5BatchLoaderGammaKappa(wlbl.GammaKappaMixin, HDF5BatchLoader):
    pass

class BaseHDF5BatchLoaderDenoiser(wlbl.DenoiserMixin, HDF5BatchLoader):
    pass

class HDF5BatchLoaderDeepMass(wlbl.MomentNetworkMixin, HDF5BatchLoader):
    """Batch loader for training DeepMass."""

class HDF5BatchLoaderDenoiser(wlbl.MomentNetworkMixin, BaseHDF5BatchLoaderDenoiser):
    """Batch loader for training a Gaussian denoiser."""
