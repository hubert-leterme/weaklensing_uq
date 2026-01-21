__level__ = 1

import os
import re
import warnings
import typing
import numpy as np
import h5py
from contextlib import contextmanager
import torch

from .. import utils

SCALE = 1.
PATTERN_FILENAME_ORI = r"LP001_run(\d{3})_maps" # Valid for kappaTNG, lensing potential 001
MSG_NOT_INITIALIZED = "The dataset has not been properly initialized."

# TODO: Update docstrings

class BaseHDF5Dataset:

    def __init__(
            self, hdf5_filepath, nimgs, pred_filepath=None, batch_size=None,
            std_noise=None, mask=None, beg_idx=0, shuffle=True, output_shape=None,
            meancentering=False, sort_by_filename_ori=True,
            pattern_filename_ori=PATTERN_FILENAME_ORI, min_idx_filename_ori=None,
            newaxis=False, list_of_outputs=None, close_after_batch=False,
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
        meancentering : bool, optional
            If True, meancenter the convergence maps. Default is False.
        sort_by_filename_ori: bool, optional
            If True, sort `kappa` elements by ascending order of `filename_ori`.
            Default is True.
        pattern_filename_ori: str, optional
            Regex pattern to filter `filename_ori` values. Only images
            with `filename_ori` matching the pattern will be considered.
            Default is PATTERN_FILENAME_ORI.
        min_idx_filename_ori: int, optional
            Filter images by filenames with indices equal or larger than this value.
            Default is None.
        newaxis: bool, optional, DEPRECATED
            If True, the returned arrays will be of shape (nimgs, 1, nx, ny),
            for training purpose. Default is False.
            The channel dimension is now assumed to be included by default.
        list_of_outputs: list of str, optional
            List of outputs to returns. Can be one of 'kappa_true', 'gamma1', 'gamma2',
            'gamma1_noisy', 'gamma2_noisy', 'kappa_inp', or None.
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
        self.hdf5_filepath = os.path.expanduser(hdf5_filepath)
        self.pred_filepath = pred_filepath
        self.nimgs = nimgs
        self.batch_size = batch_size
        self.std_noise = std_noise
        self.mask = mask
        self.beg_idx = beg_idx
        self.shuffle = shuffle
        self.output_shape = output_shape
        self.meancentering = meancentering
        self.sort_by_filename_ori = sort_by_filename_ori
        self.pattern_filename_ori = pattern_filename_ori
        self.min_idx_filename_ori = min_idx_filename_ori
        self.newaxis = newaxis
        self.kwargs_wiener = kwargs
        self.list_of_outputs = list_of_outputs
        self.close_after_batch = close_after_batch
        self.nreal_per_img = nreal_per_img
        self.verbose = verbose

        self.idx: np.ndarray | None = None  # Will hold the shuffled indices
        self.file: h5py.File | None = None  # HDF5 file object
        self.file_pred: h5py.File | None = None # HDF5 file object
        self.ds_kappa_inp: h5py.Dataset | None = None
        self.ds_kappa_true: h5py.Dataset | None = None
        self.ds_kappa_pred: h5py.Dataset | None = None
        self.current_idx: int = 0  # To track the batch number
        self.current_real: int = 0 # Useful when self.nreal_per_img > 1
        self.nx: int = -1
        self.ny: int = -1
        self.nbins: int = -1

        if self.list_of_outputs is not None:
            self.noutputs = len(self.list_of_outputs)
        else:
            self.noutputs = None

        self.initialized: bool = False
        self._initialize_dataset()


    @property
    def z(self) -> np.ndarray | None:
        out = self._get_attr_hdf5("z")
        if out is not None:
            assert isinstance(out, np.ndarray)
        return out
    
    @property
    def cdist(self) -> np.ndarray | None:
        out = self._get_attr_hdf5("cdist")
        if out is not None:
            assert isinstance(out, np.ndarray)
        return out

    @property
    def weights_redshifts(self) -> np.ndarray | None:
        out = self._get_attr_hdf5("weights_redshifts")
        if out is not None:
            assert isinstance(out, np.ndarray)
        return out

    @property
    def zbins(self) -> list[float] | None:
        out = self._get_attr_hdf5("zbins")
        if out is not None:
            assert isinstance(out, np.ndarray)
            assert get_nbins(out) == self.nbins
            out = out.tolist()
        return out

    def _get_attr_hdf5(self, attrname: str):
        with self.open():
            if self.file is not None:
                try:
                    out = self.file.attrs[attrname]
                except KeyError:
                    out = None
            else:
                out = None
        return out
    
    @property
    def normfact_zbins(self):
        assert self.weights_redshifts is not None
        assert self.cdist is not None
        assert self.z is not None
        assert self.zbins is not None
        weighted_cdistsq = self.weights_redshifts * self.cdist**2
        list_of_weighted_cdistsq = utils.get_list_per_zbin(
            weighted_cdistsq, self.z, zbins=self.zbins
        )
        out = np.array([np.sum(w) for w in list_of_weighted_cdistsq])
        out /= np.linalg.norm(out)
        return out


    @contextmanager
    def open(self):
        if self.file is None or not self.file.id.valid:
            self.file = h5py.File(self.hdf5_filepath, 'r', swmr=True) # Keep file open
            ds_kappa_true = self.file['kappa']
            assert isinstance(ds_kappa_true, h5py.Dataset)
            self.ds_kappa_true = ds_kappa_true

        # Load dataset of predictions (for order-2 moment networks)
        if self.pred_filepath is not None:
            if self.file_pred is None or not self.file_pred.id.valid:
                self.file_pred = h5py.File(self.pred_filepath, 'r', swmr=True) # Keep file open
                ds_kappa_pred = self.file_pred['kappa_pred']
                assert isinstance(ds_kappa_pred, h5py.Dataset)
                self.ds_kappa_pred = ds_kappa_pred

        try:
            yield self.file, self.file_pred
        finally:
            if not self.initialized or self.close_after_batch:
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

        assert self.idx is not None, MSG_NOT_INITIALIZED
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


    def _load_maps(self, idx, transform: typing.Callable | None = None):
        
        assert self.initialized, MSG_NOT_INITIALIZED
        with self.open():
            assert self.ds_kappa_true is not None
            kappa_true = self.ds_kappa_true[idx]
            if self.pred_filepath is not None:
                assert self.ds_kappa_pred is not None
                kappa_pred = self.ds_kappa_pred[idx]
            else:
                kappa_pred = None

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
        if self.meancentering:
            transforms.append(utils.meancenter)

        transform = _pipe(*transforms)

        # Output transformations
        kappa_true = transform(kappa_true)
        if kappa_pred is not None:
            kappa_pred = transform(kappa_pred)

        out_dict = {
            "kappa_true": kappa_true,
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


    def _postprocess(self, out_dict, _):
        return out_dict


    def _prepare_output(self, out_dict):
        if self.list_of_outputs is not None:
            out = tuple(
                [out_dict[val] if val is not None else None for val in self.list_of_outputs]
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
        with self.open():
            assert self.file is not None
            assert self.ds_kappa_true is not None
            try:
                filename_ori = self.file['filename_ori']
            except KeyError:
                warnings.warn(
                    "The 'filename_ori' dataset is missing; input images will not be sorted "
                    "or filtered by filename."
                )
                filename_ori = None
                self.sort_by_filename_ori = False
                self.pattern_filename_ori = None
                self.min_idx_filename_ori = None
            if len(self.ds_kappa_true.shape) == 3: # Shape = (nimgs, nx, ny) (deprecated)
                nimgs_tot, nx, ny = self.ds_kappa_true.shape
                nbins = 1
            elif len(self.ds_kappa_true.shape) == 4: # Shape = (nimgs, nbins, nx, ny)
                nimgs_tot, nbins, nx, ny = self.ds_kappa_true.shape

            # Initialize list of indices
            if self.sort_by_filename_ori:
                assert isinstance(filename_ori, h5py.Dataset)
                idx = np.argsort(filename_ori)  # Sort indices of `filename_ori`
            else:
                idx = np.arange(nimgs_tot)
            if self.pattern_filename_ori is not None:
                assert isinstance(filename_ori, h5py.Dataset)
                pattern = re.compile(self.pattern_filename_ori)
                unique_filename_ori = np.unique(filename_ori)
                def keep_filename(s):
                    if isinstance(s, bytes):
                        s = s.decode('utf-8')
                    match = pattern.match(s)
                    out = bool(match)
                    if out and self.min_idx_filename_ori is not None:
                        # Filter by file indice
                        assert match is not None
                        run_num = int(match.group(1))
                        if run_num < self.min_idx_filename_ori:
                            out = False
                    return out
                match_dict = {s: keep_filename(s) for s in unique_filename_ori}
                mask = np.array([match_dict[filename_ori[i]] for i in idx])
                idx = idx[mask]
            self.idx = idx[self.beg_idx:self.beg_idx + self.nimgs]

            # Check if requested number of images exceeds total available
            if self.beg_idx + self.nimgs > len(idx):
                raise ValueError("The requested size exceeds the size of the dataset.")

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
            self.nbins = nbins
            if self.std_noise is not None:
                assert self.std_noise.shape[-2:] == (self.nx, self.ny)
            if self.mask is not None:
                assert self.mask.shape[-2:] == (self.nx, self.ny)

            self.initialized = True


    def close(self):
        """Close the HDF5 file when done."""
        if self.file is not None:
            self.file.close()
        if self.file_pred is not None:
            self.file_pred.close()


    def __del__(self):
        """Destructor to ensure the HDF5 file is closed when the object is deleted."""
        self.close()


class HDF5DatasetKappa(BaseHDF5Dataset):
    """
    Dataset with ground truth convergence maps only.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, list_of_outputs=["kappa_true"], **kwargs
        )


class BaseHDF5DatasetGammaKappa(BaseHDF5Dataset):

    def __init__(
            self, *args, inpainting=False,
            complexconjugate=False, return_complex=False, **kwargs
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
        newaxis: bool, optional, DEPRECATED
            If True, the returned arrays will be of shape (nimgs, 1, nx, ny),
            for training purpose. Default is False.
            The channel dimension is now assumed to be included by default.
        complexconjugate (bool, default=True)   
            Whether to use convention from jax_lensing (due to the inversion of the x-axis?).
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
        super().__init__(*args, **kwargs)

        self.inpainting = inpainting
        self.complexconjugate = complexconjugate
        self.return_complex = return_complex


    def _postprocess(self, out_dict, idx):
        # TODO: use physics = iterativemm.MassMapping(...) 
        out_dict = super()._postprocess(out_dict, idx)

        # Generate noisy shear maps
        kappa_true = out_dict["kappa_true"]
        gamma = utils.get_shear_from_convergence(
            kappa_true, complexconjugate=self.complexconjugate, return_complex=True
        )
        gamma_noisy = utils.get_masked_and_noisy_shear(
            gamma, std_noise=self.std_noise,
            mask=self.mask, inpainting=self.inpainting,
        )
        if not self.return_complex:
            out_dict.update({
                "gamma1": gamma.real,
                "gamma2": gamma.imag,
                "gamma1_noisy": gamma_noisy.real,
                "gamma2_noisy": gamma_noisy.imag
            })
        else:
            out_dict.update({
                "gamma": gamma,
                "gamma_noisy": gamma_noisy,
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
        newaxis: bool, optional, DEPRECATED
            If True, the returned arrays will be of shape (nimgs, 1, nx, ny),
            for training purpose. Default is False.
            The channel dimension is now assumed to be included by default.
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
        # TODO: use physics = iterativemm.MassMapping(...)
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

    def __init__(self, *args, input_type=None, order=1, mode='TI', **kwargs):
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
            Type of input data: 'kappa_inp' (noisy convergence map),
            'gamma_noisy' (shear map), or None.
        order : int, optional
            Order of the moment network: 1 for standard posterior mean estimate,
            2 for posterior variance. Default is 1.
        mode : str, optional
            Whether to yield tuples of input (noisy) / target (clean) images
            ('IT', deprecated), or tuples of target / input images
            ('TI', used by DeepInverse / PyTorch). Default is 'TI'.
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
        newaxis: bool, optional, DEPRECATED
            If True, the returned arrays will be of shape (nimgs, 1, nx, ny),
            for training purpose. Default is False.
            The channel dimension is now assumed to be included by default.
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


class HDF5DatasetMassMapping(InputTargetMixin, BaseHDF5DatasetGammaKappa):
    """
    Dataset for iterative mass mapping methods. The dataset takes as input
    the noisy shear maps in a complex format.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, input_type='gamma_noisy', return_complex=True, **kwargs
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


def get_nbins(zbins: list[float] | np.ndarray | None) -> int:
    if zbins is None:
        nbins = 1
    else:
        nbins = len(zbins) + 1
    return nbins
