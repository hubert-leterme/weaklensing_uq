import torch
from torch.utils import data

from . import base_dataset

class HDF5Dataset(base_dataset.HDF5Dataset, data.Dataset):

    def __len__(self):
        return self.nreal_per_img * self.nimgs

    def __getitem__(self, idx):
        idx = idx // self.nreal_per_img
        idx_file = self.idx[idx]
        return self._load_maps(idx_file)

    def _convert_to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32)

    def _add_newaxis_arr(self, arr: torch.Tensor) -> torch.Tensor:
        return arr.unsqueeze(-3) # Shape = ([nimgs,] 1, nx, ny)

    def to_torch_dataloader(self, **kwargs):
        out = data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=self.shuffle, **kwargs
        )
        return out


class HDF5DatasetGammaKappa(base_dataset.GammaKappaMixin, HDF5Dataset):
    pass

class BaseHDF5DatasetDenoiser(base_dataset.DenoiserMixin, HDF5Dataset):
    pass


class InputTargetMixin(base_dataset.InputTargetMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode='TI', **kwargs)

    def _prepare_output(self, out_dict):
        out = super()._prepare_output(out_dict)
        if self.scale_as_input:
            target, (kappa_inp, scale) = out
            out = target, TensorList([kappa_inp, scale])
        return out


class TensorList(list[torch.Tensor]):
    def to(self, device):
        return TensorList(t.to(device) for t in self)


class HDF5DatasetMassMapping(InputTargetMixin, HDF5DatasetGammaKappa):
    """Batch loader for iterative mass mapping methods."""

class HDF5DatasetDeepMass(InputTargetMixin, HDF5Dataset):
    """Batch loader for training DeepMass."""

class HDF5DatasetDenoiser(InputTargetMixin, BaseHDF5DatasetDenoiser):
    """Batch loader for training a Gaussian denoiser."""
