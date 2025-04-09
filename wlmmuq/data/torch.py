import torch
from torch.utils import data

from . import base_dataset

class TorchMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode='TI', **kwargs) # Target-Input mode

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


class HDF5Dataset(TorchMixin, base_dataset.HDF5Dataset):
    pass

class HDF5DatasetMassMapping(TorchMixin, base_dataset.HDF5DatasetMassMapping):
    pass

class HDF5DatasetDeepMass(TorchMixin, base_dataset.HDF5DatasetDeepMass):
    pass


class HDF5DatasetDenoiser(TorchMixin, base_dataset.HDF5DatasetDenoiser):

    def _postprocess(self, out_dict, idx):
        out_dict = super()._postprocess(out_dict, idx)
        if self.scale_as_input:
            out_dict["kappa_inp"] = TensorList(out_dict["kappa_inp"])
        return out_dict


class TensorList(list[torch.Tensor]):
    def to(self, device):
        return TensorList(t.to(device) for t in self)
