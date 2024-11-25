import warnings
import numpy as np
import h5py
import tensorflow as tf
import pycs.astro.wl.mass_mapping as csmm

from . import iterativemm as wlpgd
from . import utils as wlutils

class HDF5BatchLoader:

    def __init__(
            self, hdf5_filepath, nimgs, pred_filepath=None, batch_size=None,
            std_noise=None, mask=None,
            offset=0., beg_idx=0, shuffle=True, output_shape=None,
            sort_by_filename_ori=True, newaxis=False,
            list_of_outputs=None, close_after_batch=False,
            verbose=False, **kwargs
    ):
        """
        Initialize the batch loader for HDF5 data.

        Parameters
        ----------
        hdf5_filepath : str
            Path to the HDF5 dataset containing the simulated convergence maps.
        nimgs : int
            Number of images in the dataset. Indices from `beg_idx` to
            `beg_idx + nimgs` are considered.
        pred_filepath : str, optional
            Path to the HDF5 dataset containing predictions. Only required for
            order-2 moment networks.
        batch_size : int, optional
            Number of images per batch. Default is None.
        std_noise : numpy.ndarray, optional
            Array of noise standard deviation. Default is None.
        mask : numpy.ndarray, optional
            Array of masked data. Default is None.
        offset: float, optional
            Mean value of the convergence field (mass-sheet degeneracy). Default is 0.
        beg_idx : int, optional
            First image index to consider (e.g., for split training-test sets). Default is 0.
            CAUTION: To ensure independence between the training and test sets,
            `sort_by_filename_ori` must be set to `True`. Moreover, the split position
            must be chosen so that `filename_ori` values are different in the training and
            test sets.
        shuffle : bool, optional
            Whether to shuffle the indices. Default is True.
        output_shape : int or tuple, optional
            Shape to crop the output images. Default is None.
        sort_by_filename_ori: bool, optional
            If True, sort `kappa` elements by ascending order of `filename_ori`.
            Default is True.
        newaxis: bool, optional
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1),
            for training purpose. Default is False.
        list_of_outputs: list of str, optional
            List of outputs to returns. Can be one of 'kappa_true', 'gamma1', 'gamma2',
            'gamma1_noisy', 'gamma2_noisy', 'kappa_inp'.
            If None, returns a dictionary of outputs. Default is None.
        close_after_batch: bool, optional
            Default is False.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        **kwargs
            Keyword arguments for
            `pycs.astro.wl.mass_mapping.massmap2d.prox_wiener_filtering`.
        """
        self.hdf5_filepath = hdf5_filepath
        self.pred_filepath = pred_filepath
        self.nimgs = nimgs
        self.batch_size = batch_size
        self.std_noise = std_noise
        self.mask = mask
        self.offset = offset
        self.beg_idx = beg_idx
        self.shuffle = shuffle
        self.output_shape = output_shape
        self.sort_by_filename_ori = sort_by_filename_ori
        self.newaxis = newaxis
        self.kwargs_wiener = kwargs
        self.list_of_outputs = list_of_outputs
        self.close_after_batch = close_after_batch
        self.verbose = verbose

        self.idx = None  # Will hold the shuffled indices
        self.file = None  # HDF5 file object
        self.file_pred = None # HDF5 file object
        self.ds_kappa_true = None
        self.ds_kappa_pred = None
        self.current_idx = 0  # To track the batch number

        self.sorted_batch_idx = None # Sorted indices for a given batch
        self.reversed_sort_idx = None

        if self.list_of_outputs is not None:
            self.noutputs = len(self.list_of_outputs)
        else:
            self.noutputs = None

        self._initialize_dataset()


    def _open_and_get_dataset(self):
        self.file = h5py.File(self.hdf5_filepath, 'r')  # Keep file open
        self.ds_kappa_true = self.file['kappa']

        # Load dataset of predictions
        if self.pred_filepath is not None:
            self.file_pred = h5py.File(self.pred_filepath, 'r') # Keep file open
            self.ds_kappa_pred = self.file_pred['kappa_deepmass']


    def _initialize_dataset(self):
        """Load the HDF5 file and initialize the dataset."""
        self._open_and_get_dataset()
        if self.sort_by_filename_ori:
            try:
                filename_ori = self.file['filename_ori']
            except KeyError:
                warnings.warn(
                    "The 'filename_ori' dataset is missing; input images will not be sorted."
                )
                self.sort_by_filename_ori = False
        nimgs_tot, nx, ny = self.ds_kappa_true.shape

        # Check if requested number of images exceeds total available
        if self.beg_idx + self.nimgs > nimgs_tot:
            self.file.close()  # Close the file in case of error
            raise ValueError("The requested size exceeds the size of the dataset.")

        # Initialize list of indices
        if self.sort_by_filename_ori:
            idx = np.argsort(filename_ori)  # Sort indices of `filename_ori`
        else:
            idx = np.arange(nimgs_tot)
        self.idx = idx[self.beg_idx:self.beg_idx + self.nimgs]
        if self.shuffle:
            np.random.shuffle(self.idx)

        # Get crop indices, if required
        if self.output_shape is not None:
            try:
                nx_out, ny_out = self.output_shape
            except TypeError:
                nx_out = self.output_shape
                ny_out = self.output_shape
            self._beg_idx_x, self._end_idx_x = wlutils.get_beg_end_idx(nx, nx_out)
            self._beg_idx_y, self._end_idx_y = wlutils.get_beg_end_idx(ny, ny_out)
            self.nx = nx_out
            self.ny = ny_out
        else:
            self.nx = nx
            self.ny = ny
        if self.std_noise is not None:
            assert self.std_noise.shape[-2:] == (self.nx, self.ny)
        if self.mask is not None:
            assert self.mask.shape[-2:] == (self.nx, self.ny)

        if self.close_after_batch:
            self.close()


    def _load_batch_dict(self, beg_idx, max_idx, get_all_images):

        if max_idx is None:
            max_idx = self.nimgs
        if not get_all_images:
            if self.batch_size is None:
                raise ValueError("Attribute 'batch_size' must be specified.")
            end_idx = min(beg_idx + self.batch_size, max_idx)
        else:
            end_idx = max_idx

        batch_idx = self.idx[beg_idx:end_idx]

        # Sort batch_idx to ensure increasing order for HDF5 access
        sort_idx = np.argsort(batch_idx)
        self.sorted_batch_idx = batch_idx[sort_idx]

        # Load batches with sorted indices
        # TODO: use `with self.open():`
        if self.close_after_batch:
            self._open_and_get_dataset()
        kappa_true = self.ds_kappa_true[self.sorted_batch_idx]
        if self.pred_filepath is not None:
            kappa_pred = self.ds_kappa_pred[self.sorted_batch_idx]
        if self.close_after_batch:
            self.close()

        # Re-order the batch
        self.reversed_sort_idx = np.argsort(sort_idx)
        kappa_true = kappa_true[self.reversed_sort_idx]
        if self.pred_filepath is not None:
            kappa_pred = kappa_pred[self.reversed_sort_idx]

        # Crop the batches if output_shape is specified
        # No cropping for kappa_pred as it was already computed
        # from cropped inputs
        if self.output_shape is not None:
            kappa_true = wlutils.crop_arr(
                kappa_true,
                self._beg_idx_x, self._end_idx_x,
                self._beg_idx_y, self._end_idx_y
            )

        out_dict = {
            "kappa_true": kappa_true + self.offset
        }
        if self.pred_filepath is not None:
            assert kappa_pred.shape == kappa_true.shape
            out_dict.update({
                "kappa_pred": kappa_pred + self.offset
            })

        if self.verbose:
            print(f"Images {beg_idx} to {end_idx} loaded.")

        return out_dict, end_idx


    def _prepare_output(self, out_dict):
        if self.list_of_outputs is not None:
            out = tuple(
                [out_dict[val] for val in self.list_of_outputs]
            )
            if len(out) == 1:
                out = out[0]
        else:
            out = out_dict

        return out


    def load_batch(
            self, beg_idx=0, max_idx=None, get_all_images=False, return_end_idx=False
    ):
        out_dict, end_idx = self._load_batch_dict(
            beg_idx=beg_idx, max_idx=max_idx, get_all_images=get_all_images
        )
        if self.newaxis:
            for key in out_dict:
                out_dict[key] = out_dict[key][..., np.newaxis]
        out = self._prepare_output(out_dict)
        if return_end_idx:
            out = (out, end_idx)

        return out


    def to_tf_dataset(
            self, min_idx=0, max_idx=None, raise_stop_iteration=False, **kwargs
    ):
        if max_idx is None:
            max_idx = self.nimgs

        def generator():
            end_idx = min_idx
            while end_idx < max_idx:
                # Load the next batch of data
                beg_idx = end_idx
                out, end_idx = self.load_batch(
                    beg_idx, max_idx=max_idx, return_end_idx=True, **kwargs
                )

                # Handle generator looping (to avoid StopIteration error)
                # Reset generator and reshuffle indices if needed
                if end_idx == max_idx and not raise_stop_iteration:
                    end_idx = 0
                    if self.shuffle:
                        np.random.shuffle(self.idx)

                yield out

        try:
            tensor_shape = (None, *self.output_shape)
        except TypeError:
            tensor_shape = (None, self.output_shape, self.output_shape)

        if self.newaxis:
            tensor_shape += (1,)

        output_signature = tf.TensorSpec(shape=tensor_shape, dtype=tf.float32)
        if self.noutputs > 1:
            output_signature = self.noutputs * (output_signature,)

        out = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )
        return out


    def close(self):
        """Close the HDF5 file when done."""
        if self.file is not None:
            self.file.close()
        if self.file_pred is not None:
            self.file_pred.close()


    def __del__(self):
        """Destructor to ensure the HDF5 file is closed when the object is deleted."""
        self.close()


class BaseHDF5BatchLoaderDeepMass(HDF5BatchLoader):

    def __init__(
            self, *args, input_method=None, recompute_inputs=False,
            std_gaussianfilter=None, powerspectrum_1d=None, step_size=None, niter=1,
            **kwargs
    ):
        """
        Initialize the batch loader for HDF5 data, with input prepared for DeepMass.

        Parameters
        ----------
        hdf5_filepath : str
            Path to the HDF5 dataset containing the simulated convergence maps.
        nimgs : int
            Number of images in the dataset. Indices from `beg_idx` to
            `beg_idx + nimgs` are considered.
        input_method: str
            Input mass mapping method: 'ks', 'wiener' or 'wiener_pgd'.
            If set to 'ks' or 'wiener', the implementations from
            `pycs.astro.wl.mass_mapping` will be used. If set to 'wiener_pgd, the
            implementation from `iterativemm.PGDMassMapping` will be used. Default is None.
        pred_filepath : str, optional
            Path to the HDF5 dataset containing predictions. Only required for
            order-2 moment networks.
        batch_size : int, optional
            Number of images per batch. Default is None.
        std_noise : numpy.ndarray, optional
            Array of noise standard deviation. Default is None.
        mask : numpy.ndarray, optional
            Array of masked data. Default is None.
        offset: float, optional
            Mean value of the convergence field (mass-sheet degeneracy). Default is 0.
        beg_idx : int, optional
            First image index to consider (e.g., for split training-test sets). Default is 0.
            CAUTION: To ensure independence between the training and test sets,
            `sort_by_filename_ori` must be set to `True`. Moreover, the split position
            must be chosen so that `filename_ori` values are different in the training and
            test sets.
        shuffle : bool, optional
            Whether to shuffle the indices. Default is True.
        output_shape : int or tuple, optional
            Shape to crop the output images. Default is None.
        sort_by_filename_ori: bool, optional
            If True, sort `kappa` elements by ascending order of `filename_ori`.
            Default is True.
        newaxis: bool, optional
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1),
            for training purpose. Default is False.
        recompute_inputs: bool, optional
            If True, compute inputs (KS or Wiener solution) from the groud truth convergence maps,
            even if the corresponding dataset already exists in the HDF5 file. Default is False.
        std_gaussianfilter: float, optional
            If `input_method` is set to 'ks', standard deviation of the smoothing filter.
            Default is None.
        powerspectrum_1d: np.ndarray, optional
            If `input_method` is set to 'wiener' or 'wiener_pdg', 1D power spectrum.
            Its length must be half the image size. Default is None.
        step_size: If `input_method` is set to 'wiener_pgd', step size of the gradient descent
            operator. If `input_method` is set to 'wiener', the step size if inferred
            automatically from `std_noise`. Default is None.
        niter: int, optional
            If `input_method` is set to 'wiener' or 'wiener_pgd', number of iterations.
            Default is 1.
        list_of_outputs: list of str, optional
            List of outputs to returns. Can be one of 'kappa_true', 'gamma1', 'gamma2',
            'gamma1_noisy', 'gamma2_noisy', 'kappa_inp'.
            If None, returns a dictionary of outputs. Default is None.
        close_after_batch: bool, optional
            Default is False.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        **kwargs
            Keyword arguments for
            `pycs.astro.wl.mass_mapping.massmap2d.prox_wiener_filtering`.
        """
        self.input_method = input_method
        self.recompute_inputs = recompute_inputs
        self.std_gaussianfilter = std_gaussianfilter
        self.powerspectrum_1d = powerspectrum_1d
        self.step_size = step_size
        self.niter = niter

        self.ds_kappa_inp = None
        self.sheardata = None # For Wiener filtering

        super().__init__(*args, **kwargs)

        self._initialize_wiener()


    def _initialize_wiener(self):
        """Initialize the parameters for iterative Wiener filtering."""
        if self.input_method == 'wiener' and self.recompute_inputs:
            # Register data into a `csmm.shear_data` object
            self.sheardata = csmm.shear_data()
            self.sheardata.mask = self.mask.astype(int)
            self.sheardata.Ncov = 2 * self.std_noise**2 # Factor 2 required

            # Create a mass mapping structure and initialize it
            self.massmap = csmm.massmap2d(name='mass')
            self.massmap.init_massmap(self.nx, self.ny)


    def _open_and_get_dataset(self):

        super()._open_and_get_dataset()

        if self.input_method is not None and not self.recompute_inputs:
            try:
                self.ds_kappa_inp = self.file[f'kappa_{self.input_method}']
            except KeyError:
                warnings.warn(
                    f"Dataset 'kappa_{self.input_method}' absent from the HDF5 file. "
                    f"The {self.input_method} solution will be computed for each new batch."
                )
                self.recompute_inputs = True


    def _load_batch_dict(self, beg_idx, max_idx, get_all_images):

        out_dict, end_idx = super()._load_batch_dict(beg_idx, max_idx, get_all_images)

        if self.input_method is not None and not self.recompute_inputs:
            kappa_inp = self.ds_kappa_inp[self.sorted_batch_idx]
            kappa_inp = kappa_inp[self.reversed_sort_idx]
            if self.output_shape is not None:
                kappa_inp = wlutils.crop_arr(
                    kappa_inp,
                    self._beg_idx_x, self._end_idx_x,
                    self._beg_idx_y, self._end_idx_y
                )

        else:
            # Generate noisy shear maps
            kappa_true = out_dict["kappa_true"] - self.offset
            gamma1, gamma2 = wlutils.get_shear_from_convergence(kappa_true)
            gamma1_noisy, gamma2_noisy, _ = wlutils.get_masked_and_noisy_shear(
                gamma1, gamma2, std_noise=self.std_noise, mask=self.mask
            )
            out_dict.update({
                "gamma1": gamma1,
                "gamma2": gamma2,
                "gamma1_noisy": gamma1_noisy,
                "gamma2_noisy": gamma2_noisy
            })

            # Compute KS solution if required
            if self.input_method is not None:
                if self.input_method == 'ks':
                    if self.verbose:
                        print("\tCompute Kaiser-Squires solution")
                    kappa_inp = wlutils.ksfilter(
                        gamma1_noisy, gamma2_noisy, get_bounds=False,
                        std_gaussianfilter=self.std_gaussianfilter
                    )
                # Compute Wiener solution if required
                elif self.input_method == 'wiener':
                    if self.verbose:
                        print("\tCompute Wiener solution")
                    self.sheardata.g1 = gamma1_noisy
                    self.sheardata.g2 = gamma2_noisy
                    kappa_inp, _ = self.massmap.prox_wiener_filtering(
                        self.sheardata, self.powerspectrum_1d, niter=self.niter,
                        **self.kwargs_wiener
                    )
                elif self.input_method == 'wiener_pgd':
                    nx, ny = kappa_true.shape[-2:]
                    assert nx == ny
                    imgsize = nx
                    prox_wiener = wlpgd.ProximalWiener(
                        imgsize, self.powerspectrum_1d, self.step_size
                    )
                    kappa_inp = wlpgd.PGDMassMapping(
                        std_noise=self.std_noise, step_size=self.step_size,
                        niter=self.niter, backward=prox_wiener, mask=self.mask,
                        verbose=self.verbose
                    )
                else:
                    raise ValueError

        try:
            out_dict.update({
                "kappa_inp": kappa_inp + self.offset
            })
        except NameError: # Name 'kappa_inp' is not defined
            pass

        return out_dict, end_idx


class BaseHDF5BatchLoaderDenoiser(HDF5BatchLoader):

    def __init__(
            self, *args, scale, **kwargs
    ):
        """
        Initialize the batch loader for HDF5 data, with input prepared for DeepMass.

        Parameters
        ----------
        hdf5_filepath : str
            Path to the HDF5 dataset containing the simulated convergence maps.
        nimgs : int
            Number of images in the dataset. Indices from `beg_idx` to
            `beg_idx + nimgs` are considered.
        scale : float
            Standard deviation of the white noise.
        pred_filepath : str, optional
            Path to the HDF5 dataset containing predictions. Only required for
            order-2 moment networks.
        batch_size : int, optional
            Number of images per batch. Default is None.
        std_noise : numpy.ndarray, optional
            Array of noise standard deviation. Default is None.
        mask : numpy.ndarray, optional
            Array of masked data. Default is None.
        offset: float, optional
            Mean value of the convergence field (mass-sheet degeneracy). Default is 0.
        beg_idx : int, optional
            First image index to consider (e.g., for split training-test sets). Default is 0.
            CAUTION: To ensure independence between the training and test sets,
            `sort_by_filename_ori` must be set to `True`. Moreover, the split position
            must be chosen so that `filename_ori` values are different in the training and
            test sets.
        shuffle : bool, optional
            Whether to shuffle the indices. Default is True.
        output_shape : int or tuple, optional
            Shape to crop the output images. Default is None.
        sort_by_filename_ori: bool, optional
            If True, sort `kappa` elements by ascending order of `filename_ori`.
            Default is True.
        newaxis: bool, optional
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1),
            for training purpose. Default is False.
        list_of_outputs: list of str, optional
            List of outputs to returns. Can be one of 'kappa_true', 'gamma1', 'gamma2',
            'gamma1_noisy', 'gamma2_noisy', 'kappa_inp'.
            If None, returns a dictionary of outputs. Default is None.
        close_after_batch: bool, optional
            Default is False.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        **kwargs
            Keyword arguments for
            `pycs.astro.wl.mass_mapping.massmap2d.prox_wiener_filtering`.
        """
        super().__init__(*args, **kwargs)
        self.scale = scale


    def _load_batch_dict(self, beg_idx, max_idx, get_all_images):

        out_dict, end_idx = super()._load_batch_dict(beg_idx, max_idx, get_all_images)

        kappa_true = out_dict["kappa_true"]
        if self.verbose:
            print("Generate white Gaussian noise")
        noise = np.random.normal(0, self.scale, size=kappa_true.shape)
        out_dict["kappa_inp"] = kappa_true + noise

        return out_dict, end_idx


class MomentNetworkMixin:

    def __init__(self, order, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = order # Must be equal to 1 or 2
        self.list_of_outputs = None
        self.noutputs = 2 # Network's input and target

    def _prepare_output(self, out_dict):
        kappa_inp = out_dict["kappa_inp"]
        kappa_true = out_dict["kappa_true"]

        if self.order == 1:
            target = kappa_true
        elif self.order == 2:
            kappa_pred = out_dict["kappa_pred"] # Estimates the posterior mean
            target = (kappa_true - kappa_pred)**2
        else:
            raise ValueError("Wrong value for argument `order`: must be equal to 1 or 2.")

        return kappa_inp, target


class HDF5BatchLoaderDeepMass(MomentNetworkMixin, BaseHDF5BatchLoaderDeepMass):
    """Batch loader for training DeepMass."""

class HDF5BatchLoaderDenoiser(MomentNetworkMixin, BaseHDF5BatchLoaderDenoiser):
    """Batch loader for training a Gaussian denoiser."""
