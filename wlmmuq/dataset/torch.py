import torch
from torch.utils import data

from . import base_dataset as wlbl

class HDF5Dataset(wlbl.HDF5Dataset, data.Dataset):

    def __len__(self):
        return self.nimgs

    def __getitem__(self, idx):
        idx_file = self.idx[idx]
        return self._load_maps(idx_file)

    def _convert_to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32)

    def _add_newaxis_arr(self, arr: torch.tensor) -> torch.tensor:
        return arr.unsqueeze(-3) # Shape = ([nimgs, ]1, H, W)

    def to_torch_dataloader(self, **kwargs):
        out = data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=self.shuffle, **kwargs
        )
        return out


class BaseHDF5DatasetDenoiser(wlbl.DenoiserMixin, HDF5Dataset):
    pass

class HDF5DatasetDeepMass(wlbl.MomentNetworkMixin, HDF5Dataset):
    """Batch loader for training DeepMass."""

class HDF5DatasetDenoiser(wlbl.MomentNetworkMixin, BaseHDF5DatasetDenoiser):
    """Batch loader for training a Gaussian denoiser."""
