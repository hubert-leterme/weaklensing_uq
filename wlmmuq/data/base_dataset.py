import warnings
import numpy as np
import h5py
import torch

from ..iterativemm import iterativemm
try:
    import pycs.astro.wl.mass_mapping as csmm
except ImportError:
    warnings.warn("Module `pycs` not found.")

from .. import utils
from .. import OFFSET

SCALE = 1.

# TODO: Update docstrings

class BaseHDF5Dataset:

    def __init__(
            self, hdf5_filepath, nimgs, pred_filepath=None, batch_size=None,
            std_noise=None, mask=None, input_method=None,
            offset=OFFSET, beg_idx=0, shuffle=True, output_shape=None,
            sort_by_filename_ori=True, newaxis=False,
            list_of_outputs=None, close_after_batch=False,
            nreal_per_img=1, verbose=False, **kwargs
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
        input_method: str, optional
            Input mass mapping method: 'ks', 'wiener' or 'wiener_pgd'. Only if already
            registered in the HDF5 dataset. Default is None.
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
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1) (for TensorFlow),
            or (nimgs, 1, nx, ny) (for PyTorch), for training purpose. Default is False.
        list_of_outputs: list of str, optional
            List of outputs to returns. Can be one of 'kappa_true', 'gamma1', 'gamma2',
            'gamma1_noisy', 'gamma2_noisy', 'kappa_inp'.
            If None, returns a dictionary of outputs. Default is None.
        close_after_batch: bool, optional
            Default is False.
        nreal_per_img: int, optional
        num_workers: int, optional
            Number of workers for parallel processing. Only work for PyTorch datasets.
            Default is 0.
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
        self.input_method = input_method
        self.offset = offset
        self.beg_idx = beg_idx
        self.shuffle = shuffle
        self.output_shape = output_shape
        self.sort_by_filename_ori = sort_by_filename_ori
        self.newaxis = newaxis
        self.kwargs_wiener = kwargs
        self.list_of_outputs = list_of_outputs
        self.close_after_batch = close_after_batch
        self.nreal_per_img = nreal_per_img
        self.verbose = verbose

        self.idx = None  # Will hold the shuffled indices
        self.file = None  # HDF5 file object
        self.file_pred = None # HDF5 file object
        self.ds_kappa_inp = None
        self.ds_kappa_true = None
        self.ds_kappa_pred = None
        self.input_exists = False
        self.current_idx = 0  # To track the batch number
        self.current_real = 0 # Useful when self.nreal_per_img > 1
        self.nx = None
        self.ny = None

        if self.list_of_outputs is not None:
            self.noutputs = len(self.list_of_outputs)
        else:
            self.noutputs = None

        self._initialize_dataset()


    def _open_and_get_dataset(self):
        self.file = h5py.File(self.hdf5_filepath, 'r', swmr=True)  # Keep file open
        self.ds_kappa_true = self.file['kappa']

        # Load dataset of input mass mapping method
        if self.input_method is not None:
            try:
                self.ds_kappa_inp = self.file[f'kappa_{self.input_method}']
            except KeyError:
                warnings.warn(
                    f"Dataset 'kappa_{self.input_method}' absent from the HDF5 file."
                )
            else:
                self.input_exists = True

        # Load dataset of predictions (for order-2 moment networks)
        if self.pred_filepath is not None:
            self.file_pred = h5py.File(self.pred_filepath, 'r', swmr=True) # Keep file open
            self.ds_kappa_pred = self.file_pred['kappa_pred']


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
        reversed_sort_idx = np.argsort(sort_idx)
        sorted_batch_idx = batch_idx[sort_idx]

        # Load batches with sorted indices
        def reorder(arr):
            return arr[reversed_sort_idx]
        out_dict = self._load_maps(
            sorted_batch_idx, transform=reorder
        )
        if self.verbose:
            print(f"Images {beg_idx} to {end_idx} loaded.")

        return out_dict, end_idx


    def load_batch(
            self, beg_idx=0, max_idx=None, get_all_images=False, return_end_idx=False
    ):
        out, end_idx = self._load_batch_dict(
            beg_idx=beg_idx, max_idx=max_idx, get_all_images=get_all_images
        )
        if return_end_idx:
            out = (out, end_idx)

        return out


    def _load_maps(self, idx, transform: callable = None):

        # TODO: use `with self.open():`
        if self.close_after_batch:
            self._open_and_get_dataset()
        kappa_true = self.ds_kappa_true[idx]
        if self.input_exists:
            kappa_inp = self.ds_kappa_inp[idx]
        else:
            kappa_inp = None
        if self.pred_filepath is not None:
            kappa_pred = self.ds_kappa_pred[idx]
        else:
            kappa_pred = None
        if self.close_after_batch:
            self.close()

        transforms = []

        # Crop the batches if output_shape is specified
        # No cropping for kappa_pred as it was already computed
        # from cropped inputs
        if self.output_shape is not None:
            def crop(arr):
                try:
                    assert arr.shape[-2:] == self.output_shape
                except AssertionError:
                    out = utils.crop_arr(
                        arr,
                        self._beg_idx_x, self._end_idx_x,
                        self._beg_idx_y, self._end_idx_y
                    )
                else:
                    out = arr
                return out
            transforms.append(crop)

        if transform is not None:
            transforms.append(transform)
        transforms.append(self._convert_to_tensor) # Identity by default
        transforms.append(self._meancenter_offset)

        transform = _pipe(*transforms)

        # Output transformations
        kappa_true = transform(kappa_true)
        if kappa_inp is not None:
            kappa_inp = transform(kappa_inp)
        if kappa_pred is not None:
            kappa_pred = transform(kappa_pred)

        out_dict = {
            "kappa_true": kappa_true,
            "kappa_inp": kappa_inp,
            "kappa_pred": kappa_pred
        }
        out_dict = self._postprocess(out_dict, idx)

        # Add new axis for channel dimension
        if self.newaxis:
            out_dict = self._add_newaxis(out_dict)

        out = self._prepare_output(out_dict)

        return out


    def _convert_to_tensor(self, arr):
        return arr # By default, no conversion


    def _meancenter_offset(self, arr):
        return utils.meancenter(arr) + self.offset


    def _postprocess(self, out_dict, _):
        return out_dict


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


    def _add_newaxis(self, arrdict):

        if arrdict is None:
            pass
        elif isinstance(arrdict, (np.ndarray, torch.Tensor)):
            arrdict = self._add_newaxis_arr(arrdict)
        else:
            convert_back_to_tuple = False
            if isinstance(arrdict, dict):
                enumobject = arrdict.items()
            else:
                enumobject = enumerate(arrdict)
                if isinstance(arrdict, tuple):
                    # Convert to list to allow item assignment
                    arrdict = list(arrdict)
                    convert_back_to_tuple = True
            for idx, subarrdict in enumobject:
                arrdict[idx] = self._add_newaxis(subarrdict)
            if convert_back_to_tuple:
                arrdict = tuple(arrdict)

        return arrdict


    def _add_newaxis_arr(self, arr):
        """New axis for channel dimension."""
        raise NotImplementedError


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

        # Get crop indices, if required
        if self.output_shape is not None:
            try:
                nx_out, ny_out = self.output_shape
            except TypeError:
                nx_out = self.output_shape
                ny_out = self.output_shape
            self._beg_idx_x, self._end_idx_x = utils.get_beg_end_idx(nx, nx_out)
            self._beg_idx_y, self._end_idx_y = utils.get_beg_end_idx(ny, ny_out)
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


    def close(self):
        """Close the HDF5 file when done."""
        if self.file is not None:
            self.file.close()
        if self.file_pred is not None:
            self.file_pred.close()


    def __del__(self):
        """Destructor to ensure the HDF5 file is closed when the object is deleted."""
        self.close()


class BaseHDF5DatasetGammaKappa(BaseHDF5Dataset):

    def __init__(
            self, *args, inpainting=False, std_gaussianfilter=None, powerspectrum_1d=None,
            step_size=None, niter=1, return_complex=False, **kwargs
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
        batch_size : int, optional
            Number of images per batch. Default is None.
        std_noise : numpy.ndarray, optional
            Array of noise standard deviation. Default is None.
        mask : numpy.ndarray, optional
            Array of masked data. Default is None.
        inpainting: bool, optional
            If True, then apply noise in masked regions of the shear. Otherwise, set masked
            values to 0. Default is False.
        input_method: str, optional
            Input mass mapping method: 'ks', 'wiener' or 'wiener_pgd'.
            If set to 'ks' or 'wiener', the implementations from
            `pycs.astro.wl.mass_mapping` will be used. If set to 'wiener_pgd, the
            implementation from `iterativemm.PGDMassMapping` will be used. Default is None.
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
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1) (for TensorFlow),
            or (nimgs, 1, nx, ny) (for PyTorch), for training purpose. Default is False.
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
        return_complex (bool, default=False)
            If True, then complex-valued arrays will be returned. If False, then
            real and imaginary parts will be returned separately.
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
        self.inpainting = inpainting
        self.std_gaussianfilter = std_gaussianfilter
        self.powerspectrum_1d = powerspectrum_1d
        self.step_size = step_size
        self.niter = niter
        self.return_complex = return_complex

        self.sheardata = None # For Wiener filtering

        super().__init__(*args, **kwargs)

        self._initialize_wiener()


    def _initialize_wiener(self):
        """Initialize the parameters for iterative Wiener filtering."""
        if self.input_method == 'wiener' and not self.input_exists:
            # Register data into a `csmm.shear_data` object
            self.sheardata = csmm.shear_data()
            self.sheardata.mask = self.mask.astype(int)
            self.sheardata.Ncov = 2 * self.std_noise**2 # Factor 2 required

            # Create a mass mapping structure and initialize it
            self.massmap = csmm.massmap2d(name='mass')
            self.massmap.init_massmap(self.nx, self.ny)


    def _postprocess(self, out_dict, idx):
        out_dict = super()._postprocess(out_dict, idx)

        # Generate noisy shear maps
        kappa_true = out_dict["kappa_true"] - self.offset
        gamma1, gamma2 = utils.get_shear_from_convergence(
            kappa_true, return_complex=False
        )
        gamma1_noisy, gamma2_noisy, _ = utils.get_masked_and_noisy_shear(
            gamma1, gamma2, std_noise=self.std_noise,
            mask=self.mask, inpainting=self.inpainting
        )
        if not self.return_complex:
            out_dict.update({
                "gamma1": gamma1,
                "gamma2": gamma2,
                "gamma1_noisy": gamma1_noisy,
                "gamma2_noisy": gamma2_noisy
            })
        else:
            out_dict.update({
                "gamma": gamma1 + 1j * gamma2,
                "gamma_noisy": gamma1_noisy + 1j * gamma2_noisy
            })

        # Compute KS solution if required
        if self.input_method is not None and not self.input_exists:
            if self.input_method == 'ks':
                if self.verbose:
                    print("\tCompute Kaiser-Squires solution")
                kappa_inp = utils.ksfilter(
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
                prox_wiener = iterativemm.ProximalWiener(
                    imgsize, self.powerspectrum_1d, self.step_size
                )
                wiener_pdg = iterativemm.BayesianPGDMassMappingNoPrecond(
                    std_noise=self.std_noise, step_size=self.step_size,
                    niter=self.niter, backward=prox_wiener, mask=self.mask,
                    verbose=self.verbose
                )
                kappa_inp = wiener_pdg(gamma1_noisy + 1j* gamma2_noisy)
            else:
                raise ValueError

            out_dict.update({
                "kappa_inp": kappa_inp + self.offset
            })

        return out_dict


class BaseHDF5DatasetDenoiser(BaseHDF5Dataset):

    def __init__(
            self, *args, std_noise=None, scale=SCALE, scale_min=None,
            scale_as_input=False, **kwargs
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
        std_noise : numpy.ndarray, optional
            Array of noise standard deviation. If none is given, a white noise
            will be applied (identity). Default is None.
        scale : float, optional
            Multiplicative factor for std_noise, or upper bound of the uniform
            distribution over which the scale is drawn, if `scale_min` is provided.
            Default is 1.
        scale_min: float, optional
            If provided, then the noise standard deviation will be drawn uniformly
            between `scale_min` and `scale` for each input image. Default is None
            (one single noise level).
        scale_as_input: bool, optional
            If set to True, then the input is given by (kappa_inp, scale), where
            kappa_inp denotes a batch of noisy images and scale denotes an array of
            noise levels (standard deviations if std_noise is None), for each input
            image. If set to False, then only kappa_inp is provided. Default is False.
        pred_filepath : str, optional
            Path to the HDF5 dataset containing predictions. Only required for
            order-2 moment networks.
        batch_size : int, optional
            Number of images per batch. Default is None.
        input_method: str, optional
            Input mass mapping method: 'ks', 'wiener' or 'wiener_pgd'. Only if already
            registered in the HDF5 dataset. Default is None.
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
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1) (for TensorFlow),
            or (nimgs, 1, nx, ny) (for PyTorch), for training purpose. Default is False.
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
        self.std_noise = std_noise
        self.scale_max = scale
        if scale_min is not None:
            self.scale_min = scale_min
        else:
            self.scale_min = scale
        self.scale_as_input = scale_as_input


    def _postprocess(self, out_dict, idx):

        out_dict = super()._postprocess(out_dict, idx)
        kappa_true = out_dict["kappa_true"]

        if self.verbose:
            print("Generate white Gaussian noise")

        # Get the noise standard deviation
        try:
            size = (len(idx), 1, 1)
        except TypeError:
            size = (1, 1)
        scale = np.random.uniform(
            self.scale_min, self.scale_max, size=size
        ) # Shape = ([nimgs], 1, 1)
        std_noise = scale.copy()
        if self.std_noise is not None:
            std_noise *= self.std_noise # Shape = ([nimgs], nx, ny)

        # Generate noise realizations
        noise = std_noise * np.random.normal(size=kappa_true.shape)
        noise = self._convert_to_tensor(noise)
        scale = self._convert_to_tensor(scale)

        # Get noisy kappa maps
        kappa_inp = kappa_true + noise

        out_dict["kappa_true"] = kappa_true
        if not self.scale_as_input:
            out_dict["kappa_inp"] = kappa_inp
        else:
            out_dict["kappa_inp"] = (kappa_inp, scale)

        return out_dict


class InputTargetMixin:

    def __init__(self, *args, input_type='kappa_inp', order=1, mode='IT', **kwargs):
        """
        Dataset adapted for batch loading of type (input, target) or (target, input).

        Parameters
        ----------
        hdf5_filepath : str
            Path to the HDF5 dataset containing the simulated convergence maps.
        nimgs : int
            Number of images in the dataset. Indices from `beg_idx` to
            `beg_idx + nimgs` are considered.
        input_type : str, optional
            Type of input data: 'kappa_inp' (noisy convergence map, naive estimation,
            or None), or 'gamma_noisy' (shear map).
        order : int, optional
            Order of the moment network: 1 for standard posterior mean estimate,
            2 for posterior variance. Default is 1.
        mode : str, optional
            Whether to yield tuples of input (noisy) / target (clean) images
            ('IT', used by Keras / TensorFlow), or tuples of target / input images
            ('TI', used by DeepInverse / PyTorch). Default is 'IT'.
        std_noise : numpy.ndarray, optional
            Array of noise standard deviation. If none is given, a white noise
            will be applied (identity). Default is None.
        scale : float, optional
            Multiplicative factor for std_noise, or upper bound of the uniform
            distribution over which the scale is drawn, if `scale_min` is provided.
            Default is 1.
        scale_min: float, optional
            If provided, then the noise standard deviation will be drawn uniformly
            between `scale_min` and `scale` for each input image. Default is None
            (one single noise level).
        scale_as_input: bool, optional
            If set to True, then the input is given by (kappa_inp, scale), where
            kappa_inp denotes a batch of noisy images and scale denotes an array of
            noise levels (standard deviations if std_noise is None), for each input
            image. If set to False, then only kappa_inp is provided. Default is False.
        pred_filepath : str, optional
            Path to the HDF5 dataset containing predictions. Only required for
            order-2 moment networks.
        batch_size : int, optional
            Number of images per batch. Default is None.
        input_method: str, optional
            Input mass mapping method: 'ks', 'wiener' or 'wiener_pgd'. Only if already
            registered in the HDF5 dataset. Default is None.
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
            If True, the returned arrays will be of shape (nimgs, nx, ny, 1) (for TensorFlow),
            or (nimgs, 1, nx, ny) (for PyTorch), for training purpose. Default is False.
        close_after_batch: bool, optional
            Default is False.
        verbose : bool, optional
            If True, print progress messages. Default is False.
        **kwargs
            Keyword arguments for
            `pycs.astro.wl.mass_mapping.massmap2d.prox_wiener_filtering`.
        """
        if order == 1:
            target_type = "kappa_true"
        elif order == 2:
            target_type = "sqdiff_true_pred"
        else:
            raise ValueError("Wrong value for argument `order`: must be equal to 1 or 2.")

        if mode == 'IT':
            list_of_outputs = [input_type, target_type]
        elif mode == 'TI':
            list_of_outputs = [target_type, input_type]
        else:
            raise ValueError

        super().__init__(*args, list_of_outputs=list_of_outputs, **kwargs)
        self.order = order


    def _postprocess(self, out_dict, idx):
        out_dict = super()._postprocess(out_dict, idx)

        kappa_true = out_dict["kappa_true"]
        kappa_pred = out_dict["kappa_pred"] # Estimates the posterior mean
        if kappa_pred is not None:
            out_dict["sqdiff_true_pred"] = (kappa_true - kappa_pred)**2

        return out_dict


class HDF5Dataset(InputTargetMixin, BaseHDF5Dataset):
    """
    Dataset with ground truth convergence maps only.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, input_type='kappa_inp', **kwargs
        ) # kappa_inp is None, unless a mass mapping method is specified

class HDF5DatasetMassMapping(InputTargetMixin, BaseHDF5DatasetGammaKappa):
    """
    Dataset for iterative mass mapping methods. The dataset takes as input
    the noisy shear maps in a complex format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, input_type='gamma_noisy', return_complex=True, **kwargs
        )

class HDF5DatasetDeepMass(InputTargetMixin, BaseHDF5DatasetGammaKappa):
    """
    Dataset for training DeepMass. The dataset takes as input
    an initial estimation (Kaiser-Squires or Wiener) of the convergence maps.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, input_type='kappa_inp', return_complex=False, **kwargs
        )

class HDF5DatasetDenoiser(InputTargetMixin, BaseHDF5DatasetDenoiser):
    """
    Batch loader for training a Gaussian denoiser. The dataset takes as input
    the noisy convergence maps.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, input_type='kappa_inp', **kwargs
        )


def _pipeline(inp, *transforms):
    if len(transforms) == 0:
        out = inp
    else:
        out = _pipeline(inp, *transforms[:-1])
        out = transforms[-1](out)
    return out


def _pipe(*transforms):
    return lambda x: _pipeline(x, *transforms)
