import numpy as np
import tensorflow as tf

from . import base_dataset

class TensorflowMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode='IT', **kwargs) # Input-Target mode


    def _add_newaxis_arr(self, arr):
        return arr[..., np.newaxis] # Shape = (batch_size, H, W, 1)


    def to_dataloader(
            self, min_idx=0, max_idx=None, raise_stop_iteration=False, **kwargs
    ):
        if max_idx is None:
            max_idx = self.nimgs

        def generator():
            beg_idx = min_idx
            while beg_idx < max_idx:
                # Load the next batch of data
                out, end_idx = self.load_batch(
                    beg_idx, max_idx=max_idx, return_end_idx=True, **kwargs
                )
                self.current_real += 1
                if self.current_real == self.nreal_per_img:
                    beg_idx = end_idx # Update beg_idx
                    self.current_real = 0 # Reset current realization

                    # Handle generator looping (to avoid StopIteration error)
                    # Reset generator and reshuffle indices if needed
                    if beg_idx == max_idx and not raise_stop_iteration:
                        beg_idx = min_idx
                        if self.shuffle:
                            np.random.shuffle(self.idx)

                yield out

        output_signature = self._get_output_signature()

        out = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )
        return out.prefetch(tf.data.AUTOTUNE) # Prefetch data loader for efficiency


    def _get_output_signature(self):

        try:
            tensor_shape = (None, *self.output_shape)
        except TypeError:
            tensor_shape = (None, self.output_shape, self.output_shape)
        if self.newaxis:
            tensor_shape += (1,)
        out = tf.TensorSpec(shape=tensor_shape, dtype=tf.float32)
        if self.noutputs > 1:
            out = self.noutputs * (out,)

        return out


class HDF5DatasetKappa(TensorflowMixin, base_dataset.HDF5DatasetKappa):
    pass

class BaseHDF5DatasetGammaKappa(TensorflowMixin, base_dataset.BaseHDF5DatasetGammaKappa):
    pass

class HDF5DatasetMassMapping(TensorflowMixin, base_dataset.HDF5DatasetMassMapping):
    pass


class HDF5DatasetDenoiser(TensorflowMixin, base_dataset.HDF5DatasetDenoiser):

    def _get_output_signature(self):

        out = super()._get_output_signature()
        if self.scale_as_input:
            # Inputs are given as (kappa_inp, scale)
            tensor_shape_scale = (None, 1, 1)
            if self.newaxis:
                tensor_shape_scale += (1,)
            tensorspec_scale = tf.TensorSpec(
                shape=tensor_shape_scale, dtype=tf.float32
            )
            out = list(out) # Convert to list to allow item assignment
            for idx, val in enumerate(self.list_of_outputs):
                if val == 'kappa_inp':
                    out[idx] = (out[idx], tensorspec_scale)
            out = tuple(out)

        return out
