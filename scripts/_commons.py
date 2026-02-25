import os
import warnings
import time
import random
import typing

import numpy as np
from scipy.optimize import minimize
import torch
import astropy.table as aptable
import deepinv as dinv

from wlmmuq import utils as wlutils
from wlmmuq.data import cosmos as wlcosmos
from wlmmuq.data import kappatng as wlktng
from wlmmuq.data import torch as wlbl
from wlmmuq import models as wlnn
from wlmmuq.models.deepinv import iterativemm as wlpnp
from wlmmuq.models.deepinv import pnpmcalens as wlpnpmcalens
from wlmmuq.models import cqr as wlcqr

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.models.deepinv.preproc_models import NITER_WIENER
from wlmmuq.models.deepinv.pnpmcalens import \
    NITER_PER_STEP_G, NITER_PER_STEP_NG, STARLET_DETECTION_THRESHOLD

# The following global variables are valid for the kappaTNG dataset
NINPIMGS = 100 # Number of input images before cropping
NIMGS_TRAIN = 70560 # Corresponding to the 98 first realizations
NIMGS_VAL = 1440 # Remaining 2 realizations
NIMGS_TEST = 512 # Images extracted from the 57 first original files (copped dataset)
NIMGS_CALIB = 1024 # Images extracted from the 43 remaining original files (augmented dataset)
MIN_IDX_FILENAME_ORI_VAL = 98 # To avoid overlaps with the training set
MIN_IDX_FILENAME_ORI_CALIB = 58 # To avoid overlaps with the test set

EPOCH = 100 # Epoch of the trained models to load
IMGSIZE = 384
BATCH_SIZE = 32
NIMGS_SAVE = 16
KEYS_MODEL = [
    "no_bias", "model_size", "mode_preproc",
    "args_preproc", "additional_outlayer"
] # Arguments passed to the model's constructor
EPS_SUP_STEP_SIZE = 1e-9 # Avoid the upper limit itself (strict inequality)

WHICH_GAUSSIAN_EXTRACTOR = "wiener" # "wiener" or "mcalens"
MODE_PNPMASS = "regular" # "regular", "residual", or "pnpmcalens"
NITER_PNPMASS = 8
NITER_MCALENS = 32
NITER_STARLET_DEBIASING = 32
CONFIDENCE_UQ = 2 # 2-sigma confidence

INPAINTING_WIENER = False
INPAINTING_PNPMASS = False
INPAINTING_DEEPMASS = True

N_NOISE_REALS_UQ = 32

MODE_CQR = "addcqr"
BOUNDS_MULTFACT_CONFIDENCE_UQ = (0., 2.)
BOUNDS_ADDCONST_CONFIDENCE_UQ = (-0.005, 0.005)

MODEL_SPECS = [
    "pe", # Point estimate (order-1 models)
    "var" # Variance estimate (order-2 models)
]

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_path_to_results(
        output_dir: str, method_name: str,
        test_dataset_name: str | None = None,
        real_shearmap_name: str | None = None,
        test_on_real_data: bool = False,
        train_val_dataset_name: str | None = None,
        model_name: str | None = None,
):
    dirs = [output_dir, method_name]
    if not test_on_real_data:
        assert test_dataset_name is not None
        dirs.append(test_dataset_name)
    else:
        assert real_shearmap_name is not None
        dirs.append(real_shearmap_name)
    if train_val_dataset_name is not None:
        dirs.append(train_val_dataset_name)
    if model_name is not None:
        dirs.append(model_name)

    return os.path.join(*dirs)


def get_device(verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Device: {device}")
    return device


def get_stdnoise_mask(
        path_to_std_noise=PATH_TO_STD_NOISE, path_to_mask=PATH_TO_MASK,
        imgsize=IMGSIZE,
        cosmos_include_faint=False,
        inpainting=False, verbose=False
):
    if path_to_std_noise is not None:
        assert path_to_mask is not None, (
            "If `path_to_std_noise` is provided, `path_to_mask` must also be provided."
        )
        if verbose:
            print("Load noise standard deviation and mask from files")
        std_noise = torch.load(path_to_std_noise)
        mask = torch.load(path_to_mask)
    else:
        if verbose:
            print("Load COSMOS galaxy shape catalog")
        cat_cosmos_bright, cat_cosmos_faint = wlcosmos.cosmos_catalog()
        cat_cosmos_bright = wlcosmos.filter_by_redshifts(cat_cosmos_bright, wlktng.MAX_Z)
        if cosmos_include_faint:
            cat_cosmos = aptable.vstack(
                [cat_cosmos_bright, cat_cosmos_faint], join_type='outer'
            )
        else:
            cat_cosmos = cat_cosmos_bright
        data_dict = wlcosmos.get_data_from_cosmos(cat_cosmos, imgsize, wlktng.RESOLUTION)
        shapedisp = data_dict["shapedisp"]
        ngal = data_dict["ngal"]
        mask = data_dict["mask"]
        std_noise = wlutils.get_std_noise(ngal, shapedisp, std_noise_mask=0)

    if inpainting:
        # Set the noise standard deviation for masked data
        std_noise[~mask] = std_noise.max()

    return std_noise, mask


def create_dataset_from_kappatng(
        func: typing.Callable, path_to_saved_dataset:str, idx_lp: int | str,
        openingangle: float, ninpimgs: int, verbose: bool = False, **kwargs
):
    """
    Create a dataset from the KappaTNG simulation.
    The dataset is saved in the specified path.
    The dataset is created by calling the function `func` with the specified
    parameters.

    Parameters
    ----------
    func : callable
        Function to create the dataset: `wlmmuq.kappatng.create_cropped_dataset`
        or `wlmmuq.kappatng.create_augmented_dataset`.
    path_to_saved_dataset : str
        Path to save the dataset.
    idx_lp : int | str
        Index of the learning potential. It indicates which folder to look
        into for the HDF5 files containing the dataset (`LPxxx` where `xxx`
        ranges from `001` to `100`).
    openingangle : float
        Opening angle of the input images in degrees.
    ninpimgs : int
        Number of input images before cropping/augmentation.
    verbose : bool, optional
    **kwargs
        Additional arguments to pass to the function `func`.
    """
    # Get redshift weights from the COSMOS catalog
    if verbose:
        print("Computing redshift weights from COSMOS...")
    cat_cosmos_bright, _ = wlcosmos.cosmos_catalog()
    cat_cosmos_bright = wlcosmos.filter_by_redshifts(cat_cosmos_bright, wlktng.MAX_Z)
    weights_redshift = wlktng.get_weights(cat_cosmos_bright['zphot'])

    # Get nb of pixels in output images and adjust opening angle accordingly
    imgsize, openingangle = wlktng.get_npixels_openingangle(openingangle)

    # Create augmented dataset and store data
    func(
        path_to_saved_dataset, idx_lp, ninpimgs, weights_redshift, imgsize,
        verbose=verbose, **kwargs
    )


def get_checkpoint_dirs(
        model_dir,
        train_val_dataset_name=None, train_val_dataset_name_uq=None,
        model_name=None, model_name_uq=None
):
    checkpoint_dir = model_dir
    checkpoint_dir_uq = model_dir

    def _join_subdir(
            checkpoint_dir, checkpoint_dir_uq,
            arg, arg_uq, msg_missing_arg, msg_mismatch
    ):
        if arg is not None:
            if arg_uq is not None:
                warnings.warn(msg_mismatch)
            else:
                arg_uq = arg
            checkpoint_dir = os.path.join(checkpoint_dir, arg)
            checkpoint_dir_uq = os.path.join(checkpoint_dir_uq, arg_uq)

        else:
            raise ValueError(msg_missing_arg)

        return checkpoint_dir, checkpoint_dir_uq
    
    checkpoint_dir, checkpoint_dir_uq = _join_subdir(
        checkpoint_dir, checkpoint_dir_uq,
        train_val_dataset_name, train_val_dataset_name_uq,
        "Argument `train_val_dataset_name` must be provided.",
        "Mismatched datasets between order-1 and order-2 training."
    )
    checkpoint_dir, checkpoint_dir_uq = _join_subdir(
        checkpoint_dir, checkpoint_dir_uq,
        model_name, model_name_uq,
        "Argument `model_name` must be provided.",
        "Mismatched models between order-1 and order-2 training."
    )

    return checkpoint_dir, checkpoint_dir_uq


def get_path_to_checkpoint(save_path, timestamp, epoch):
    path_to_checkpoint = os.path.join(
        save_path, timestamp, f"ckp_{epoch}.pth.tar"
    )
    return path_to_checkpoint


def update_kwargs_model(
        kwargs_model,
        std_noise=None, mask=None, path_to_ps=None,
        eps_sup_step_size_wiener=EPS_SUP_STEP_SIZE,
        niter_wiener=NITER_WIENER, device="cpu", verbose=False
):
    try:
        mode_preproc = kwargs_model["mode_preproc"]
    except KeyError:
        mode_preproc = None
    if mode_preproc is not None and "args_preproc" not in kwargs_model:
        # Load arguments for Wiener or KS initialization
        # Only for DeepMass (denoiser = False)
        if mode_preproc == "wiener":
            args_preproc = _get_args_wienerinit(
                std_noise, mask, path_to_ps=path_to_ps,
                eps_sup_step_size=eps_sup_step_size_wiener,
                niter=niter_wiener, device=device, verbose=verbose
            )
        elif mode_preproc == "ks":
            args_preproc = {"std_noise": std_noise, "mask": mask}
        else:
            raise ValueError(
                f"Invalid preprocessing mode '{mode_preproc}'. "
                "Supported modes are 'wiener' and 'ks'."
            )
        kwargs_model.update(args_preproc=args_preproc)


def instantiate_model(
        arch, imgsize=IMGSIZE,
        device="cpu", verbose=False, **kwargs
):

    if arch is None:
        raise ValueError(
            "Model architecture must be provided with -a or --arch"
        )
    cnn_class, scale_as_input = wlnn.MODEL_CLASSES[arch]
    model = cnn_class(map_size=imgsize, **kwargs).to(device)
    if verbose:
        model.summary()

    return model, scale_as_input


def load_trained_model(
        checkpoint_dir, arch, timestamp,
        epoch=EPOCH, imgsize=IMGSIZE, order2=False,
        additional_outlayer=None,
        std_noise=None, mask=None, path_to_ps=PATH_TO_PS,
        eps_sup_step_size_wiener=EPS_SUP_STEP_SIZE,
        niter_wiener=NITER_WIENER, model_specs=None,
        device="cpu", verbose=False, **kwargs
):
    update_kwargs_model(
        kwargs,
        std_noise=std_noise, mask=mask, path_to_ps=path_to_ps,
        eps_sup_step_size_wiener=eps_sup_step_size_wiener,
        niter_wiener=niter_wiener,
        device=device, verbose=verbose
    )
    model, _ = instantiate_model(
        arch, imgsize=imgsize, order2=order2,
        additional_outlayer=additional_outlayer,
        device=device, verbose=verbose, **kwargs
    )
    checkpoint_dir = os.path.expanduser(checkpoint_dir)
    if timestamp is None:
        path_to_checkpoint = checkpoint_dir
    else:
        if model_specs is None:
            model_specs = MODEL_SPECS[order2]
        save_path = os.path.join(checkpoint_dir, model_specs)
        path_to_checkpoint = get_path_to_checkpoint(
            save_path, timestamp, epoch
        )
    state_dict = wlutils.load_checkpoint_state_dict(
        path_to_checkpoint, verbose=verbose
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=EPOCH,
        model_specs=None,
        load_model_uq=False, checkpoint_dir_uq=None,
        arch_uq=None, timestamp_uq=None, epoch_uq=None,
        model_specs_uq=None,
        imgsize=IMGSIZE,
        std_noise=None, mask=None, path_to_ps=PATH_TO_PS,
        eps_sup_step_size_wiener=EPS_SUP_STEP_SIZE,
        niter_wiener=NITER_WIENER,
        device="cpu", verbose=False, **kwargs
):
    kwargs_model = {k: kwargs.pop(k) for k in KEYS_MODEL if k in kwargs}
    if verbose:
        print("Load trained order-1 model")
    model = load_trained_model(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        imgsize=imgsize, order2=False,
        std_noise=std_noise, mask=mask, path_to_ps=path_to_ps,
        eps_sup_step_size_wiener=eps_sup_step_size_wiener,
        niter_wiener=niter_wiener, model_specs=model_specs,
        device=device, verbose=verbose,
        **kwargs_model
    )
    if load_model_uq:
        if arch_uq is None:
            arch_uq = arch
            kwargs_model_uq = kwargs_model.copy()
        else:
            kwargs_model_uq = {}
            for k in KEYS_MODEL:
                kuq = f"{k}_uq"
                if kuq in kwargs:
                    kwargs_model_uq.update({k: kwargs.pop(kuq)})
        if epoch_uq is None:
            epoch_uq = epoch

        if verbose:
            print("Load trained order-2 model")
        model_uq = load_trained_model(
            checkpoint_dir_uq, arch_uq, timestamp_uq,
            epoch=epoch_uq, imgsize=imgsize, order2=True,
            std_noise=std_noise, mask=mask, path_to_ps=path_to_ps,
            eps_sup_step_size_wiener=eps_sup_step_size_wiener,
            niter_wiener=niter_wiener, model_specs=model_specs_uq,
            device=device, verbose=verbose,
            **kwargs_model_uq
        )
    else:
        model_uq = None

    return model, model_uq


def instantiate_starlet_denoiser(
        imgsize=IMGSIZE,
        detection_threshold=STARLET_DETECTION_THRESHOLD,
        callback: wlpnpmcalens.StarletResetter | None = None,
        device="cpu", verbose=False
):
    denoiser = wlpnpmcalens.Starlet2d(
        in_channels=1, nx=imgsize, ny=imgsize,
        detection_threshold=detection_threshold
    ).to(device)
    if verbose:
        print(
            f"Starlet denoiser instantiated with {denoiser.ns} scales and "
            f"a {detection_threshold:.1f}-sigma detection threshold."
        )
    if callback is None:
        callback = wlpnpmcalens.StarletResetter(denoiser)
    else:
        callback.starlet.append(denoiser)

    return denoiser, callback


def get_dataloader_massmapping(
        path_to_dataset, nimgs, imgsize, batch_size, num_workers, std_noise, mask,
        test_on_real_data=False, path_to_real_shearmap=None, **kwargs
):
    if not test_on_real_data:
        test_dataloader = wlbl.HDF5DatasetMassMapping(
            hdf5_filepath=path_to_dataset, nimgs=nimgs, batch_size=batch_size,
            std_noise=std_noise, mask=mask, output_shape=imgsize,
            newaxis=True, num_workers=num_workers, **kwargs
        ).to_dataloader()
    else:
        if path_to_real_shearmap is None:
            raise ValueError("Argument `path_to_real_shearmap` must be provided.")
        test_dataloader = wlbl.RealShearMapDataset(
            path_to_real_shearmap, newaxis=True,
            also_get_complex_conjugates=True
        ).to_dataloader() # TODO: Switch to `newaxis=False` after merge with `tomographic`

    return test_dataloader


def _get_args_wienerinit(
        std_noise, mask, path_to_ps=PATH_TO_PS,
        white_noise=False,
        step_size=None, eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_WIENER, device="cpu", verbose=False
):
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)
    powerspectrum = torch.load(path_to_ps)
    step_size, _ = _get_step_size_param_mahalanobis(
        white_noise=white_noise,
        std_noise=std_noise, physics=physics,
        step_size=step_size, eps=eps_sup_step_size,
        device=device, verbose=verbose
    ) # Bayesian Wiener filtering
    args_wienerinit = dict(
        step_size=step_size, powerspectrum=powerspectrum,
        std_noise=std_noise, mask=mask, niter=niter
    )
    return args_wienerinit


def _get_datafidelity_params(
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        step_size=None, multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    step_size, param_mahalanobis = \
            _get_step_size_param_mahalanobis(
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_step_size=multfact_step_size,
        eps=eps_sup_step_size,
        device=device, verbose=verbose
    )
    if not white_noise:
        data_fidelity = wlpnp.Mahalanobis(param_vector=param_mahalanobis)
        g_param = step_size
    else:
        # Regular L2 data fidelity with unitary variance
        data_fidelity = dinv.optim.data_fidelity.L2()
        g_param = None # To be updated for each new noise realization
    params_algo = {"stepsize": step_size, "g_param": g_param}

    return data_fidelity, params_algo


def _get_datafidelity_prior_params_gaussian(
        path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        step_size=None,
        multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    data_fidelity, params_algo = _get_datafidelity_params(
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size,
        multfact_step_size=multfact_step_size,
        eps_sup_step_size=eps_sup_step_size,
        device=device, verbose=verbose
    )
    powerspectrum = torch.load(path_to_ps)
    prior = dinv.optim.PnP(wlpnp.ProximalWiener(powerspectrum))

    return data_fidelity, prior, params_algo


def _get_datafidelity_prior_params_nongaussian(
        denoiser, denoiser_uq=None,
        white_noise=False, std_noise=None, physics=None,
        step_size=None, multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    data_fidelity, params_algo = _get_datafidelity_params(
        white_noise=white_noise, noise_whitening=True,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_step_size=multfact_step_size,
        eps_sup_step_size=eps_sup_step_size,
        device=device, verbose=verbose
    ) # Noise-whitening data fidelity
    prior = dinv.optim.prior.PnP(denoiser)
    if denoiser_uq is not None:
        prior_uq = dinv.optim.prior.PnP(denoiser_uq)
    else:
        prior_uq = None

    return data_fidelity, prior, prior_uq, params_algo


def _get_step_size_param_mahalanobis(
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        step_size=None, multfact_step_size=None,
        eps=EPS_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    if not white_noise:
        assert std_noise is not None
        param_mahalanobis = wlutils.get_g_param(std_noise, noise_whitening)
        if step_size is None or step_size <= 0:
            step_size = wlutils.get_sup_step_size(
                param_mahalanobis=param_mahalanobis,
                physics=physics, device=device
            )
            if verbose:
                print(
                    f"Step size upper bound computed using power iteration = {step_size:.2e}"
                )
            step_size *= (1 - eps)
        if multfact_step_size is not None:
            step_size *= multfact_step_size
    else:
        # The standard MSE is used as data fidelity
        # The parameter `g_param` for the proximal operator must be updated accordingly
        if step_size is not None:
            warnings.warn(
                "The step size is not used for white noise. "
                "It will be set to 1."
            )
        step_size = 1
        param_mahalanobis = None

    return step_size, param_mahalanobis


def get_wiener(
        path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        step_size=None, eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_WIENER, device="cpu", verbose=False
):
    if verbose:
        print("Get optimizer for iterative Wiener filtering")
    data_fidelity, prior, params_algo = _get_datafidelity_prior_params_gaussian(
        path_to_ps=path_to_ps,
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size, eps_sup_step_size=eps_sup_step_size,
        device=device, verbose=verbose
    )
    wiener = wlpnp.optim_builder(
        iteration="PGD",
        params_algo=params_algo.copy(),
        data_fidelity=data_fidelity, prior=prior,
        early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
    ).to(device)

    return wiener


def get_gaussian_extractor(
        which=WHICH_GAUSSIAN_EXTRACTOR,
        path_to_ps=PATH_TO_PS,
        white_noise=False,
        imgsize=IMGSIZE, std_noise=None, physics=None,
        step_size=None, step_size_ng=None,
        multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_WIENER,
        starlet_detection_threshold=STARLET_DETECTION_THRESHOLD,
        device="cpu", verbose=False
):
    data_fidelity_g, prior_g, params_algo_g = _get_datafidelity_prior_params_gaussian(
        path_to_ps=path_to_ps,
        white_noise=white_noise,
        std_noise=std_noise, physics=physics,
        step_size=step_size,
        multfact_step_size=multfact_step_size,
        eps_sup_step_size=eps_sup_step_size,
        device=device, verbose=verbose
    )
    if which == "wiener":
        if verbose:
            print("Wiener used as Gaussian extractor")
        extractor = wlpnp.optim_builder(
            iteration="PGD",
            params_algo=params_algo_g.copy(),
            data_fidelity=data_fidelity_g, prior=prior_g,
            early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
        ).to(device)
        callback = None

    elif which == "mcalens":
        if verbose:
            print("MCALens used as Gaussian extractor")
        denoiser_ng, callback = instantiate_starlet_denoiser(
            imgsize=imgsize,
            detection_threshold=starlet_detection_threshold,
            device=device, verbose=verbose
        )
        data_fidelity_ng, prior_ng, _, params_algo_ng = \
                _get_datafidelity_prior_params_nongaussian(
            denoiser_ng, white_noise=white_noise,
            std_noise=std_noise, physics=physics,
            step_size=step_size_ng, eps_sup_step_size=eps_sup_step_size,
            device=device, verbose=verbose
        )
        extractor = wlpnpmcalens.optim_builder_mcalens(
            iteration_g="PGD", iteration_ng="PGD",
            params_algo_g=params_algo_g.copy(), params_algo_ng=params_algo_ng.copy(),
            data_fidelity_g=data_fidelity_g, data_fidelity_ng=data_fidelity_ng,
            prior_g=prior_g, prior_ng=prior_ng,
            early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
            update_ng_first=True,
            output_mode="discard_ng", verbose=verbose
        ).to(device)

    else:
        raise ValueError(
            f"Invalid extractor '{which}'. "
            "Supported extractors are 'wiener' and 'mcalens'."
        )

    return extractor, callback


def get_pnpmass(
        denoiser, denoiser_uq,
        std_noise=None, rmse_fn=None, physics=None,
        step_size=None, multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_PNPMASS,
        custom_init=wlpnp.zero_init,
        mode=MODE_PNPMASS,
        path_to_ps=PATH_TO_PS,
        niter_per_step_g=NITER_PER_STEP_G,
        niter_per_step_ng=NITER_PER_STEP_NG,
        device="cpu", verbose=False
):
    data_fidelity, prior, prior_uq, params_algo = \
            _get_datafidelity_prior_params_nongaussian(
        denoiser, denoiser_uq=denoiser_uq,
        white_noise=False, std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_step_size=multfact_step_size,
        eps_sup_step_size=eps_sup_step_size,
        device=device, verbose=verbose
    )
    step_size: float = params_algo["stepsize"]
    kwargs = {}
    if rmse_fn is not None:
        kwargs.update(metric_dict={"rmse": rmse_fn})

    if mode in ["regular", "residual"]:
        if verbose:
            print("Instantiate PnPMass")

        pnpmass = wlpnp.optim_builder(
            iteration="PGD", params_algo=params_algo.copy(),
            data_fidelity=data_fidelity, prior=prior,
            early_stop=False, max_iter=niter, custom_init=custom_init,
            verbose=verbose, **kwargs
        ).to(device)

    elif mode == "pnpmcalens":
        if verbose:
            print("Instantiate PnPMCALens")

        # Note: the step size for the Gaussian component is computed automatically
        data_fidelity_g, prior_g, params_algo_g = _get_datafidelity_prior_params_gaussian(
            path_to_ps=path_to_ps,
            white_noise=False,
            std_noise=std_noise, physics=physics,
            step_size=None,
            eps_sup_step_size=eps_sup_step_size,
            device=device, verbose=verbose
        )
        pnpmass = wlpnpmcalens.optim_builder_mcalens(
            iteration_g="PGD", iteration_ng="PGD",
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            params_algo_g=params_algo_g.copy(), params_algo_ng=params_algo.copy(),
            data_fidelity_g=data_fidelity_g, data_fidelity_ng=data_fidelity,
            prior_g=prior_g, prior_ng=prior,
            early_stop=False, max_iter=niter, custom_init=custom_init,
            update_ng_first=True,
            output_mode="add_components", verbose=True, **kwargs
        ).to(device)

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Supported modes are 'regular', 'residual', and 'pnpmcalens'."
        )

    if denoiser_uq is not None:
        pnpmass_uq = wlpnp.optim_builder(
            iteration="PGD", params_algo=params_algo.copy(),
            data_fidelity=data_fidelity, prior=prior_uq,
            early_stop=False, max_iter=1, custom_init=wlpnp.ManualInit(),
            verbose=verbose
        ).to(device)
    else:
        pnpmass_uq = None

    return pnpmass, pnpmass_uq, step_size


def variance_estimation_through_noise_propagation(
        method: wlpnp.BaseOptim,
        physics: wlpnp.MassMapping,
        output_shape: tuple | torch.Size,
        n_noise_reals: int = N_NOISE_REALS_UQ,
        device="cpu", verbose=False, **kwargs
):
    noise_outputs = torch.zeros(
        n_noise_reals, *output_shape, device=device
    ) # Shape = (nreals, batch_size, 1, imgsize, imgsize), dtype = float32
    zeros = torch.zeros(
        *output_shape, dtype=torch.complex64, device=device
    ) # Shape = (batch_size, 1, imgsize, imgsize), dtype = complex64
    if verbose:
        print(f"Propagate {n_noise_reals} noise realisations through the pipeline")
    for i in range(n_noise_reals):
        # Generate noise realisations
        noise = physics.noise_model(zeros) # Shape = (batch_size, 1, imgsize, imgsize), dtype = complex64
        # Propagate noise realisations through the pipeline
        # For MCALens, the support of active wavelet coefficients is assumed to be already initialized.
        noise_outputs[i] = method(noise, physics, **kwargs)

    return torch.std(noise_outputs, dim=0)**2 # Shape = (batch_size, 1, imgsize, imgsize)


def get_inference_time(beg_time, which="inference", verbose=False):
    inference_time = time.time() - beg_time
    if verbose:
        print(f"Total {which} time: {inference_time:.2f} seconds")
    return inference_time


def _convert_into_list[T](
        val: T | list[T],
        mult: int = 1
) -> list[T]:
    if not isinstance(val, list):
        val = [val for _ in range(mult)]
    else:
        assert len(val) == mult
    return val


def _get_multiplicity(*args) -> int:
    mult = None
    for val in args:
        if isinstance(val, list):
            if mult is not None:
                assert len(val) == mult
            else:
                mult = len(val)
    if mult is None:
        mult = 1
    return mult


def convert_into_lists(*args) -> tuple[list, ...]:
    mult = _get_multiplicity(*args)
    out = tuple([
        _convert_into_list(val, mult=mult) for val in args
    ])
    return out


def convert_into_list_cqr_mode(
        mode_cqr: str | list[str],
        scaling_factor_chisqcqr: float | None | list[float | None]
) -> tuple[list[str], list[float | None]]:

    mode_cqr, scaling_factor_chisqcqr = convert_into_lists(
        mode_cqr, scaling_factor_chisqcqr
    )
    for i, mcqr in enumerate(mode_cqr):
        if mcqr != "chisqcqr":
            scaling_factor_chisqcqr[i] = None

    return mode_cqr, scaling_factor_chisqcqr


def convert_into_hyperparam_list(
        hyperparam, find_optimal_hyperparam_precalib=False
):
    hyperparam = _convert_into_list(hyperparam)
    if find_optimal_hyperparam_precalib and None not in hyperparam:
        hyperparam.append(None)

    return hyperparam


def apply_calibration_and_get_metrics(
        kappa_pred: torch.Tensor,
        var: torch.Tensor,
        kappa_true: torch.Tensor | None,
        kappa_pred_calib: torch.Tensor,
        var_calib: torch.Tensor,
        kappa_true_calib: torch.Tensor,
        confidence_uq: int | float = CONFIDENCE_UQ,
        imgsize: int = IMGSIZE, mode: str = MODE_CQR,
        hyperparam_precalib: float | None = None,
        find_optimal_hyperparam_precalib: bool = False,
        mask: torch.Tensor | None = None,
        save_tensors: bool = False, nimgs_save: int = NIMGS_SAVE,
        device="cpu", verbose=False, **kwargs
):
    err_metric = wlpnp.MiscoverageRate(meancentering=False, mask=mask).to(device)
    predinterv_metric = wlpnp.PredInterv(meancentering=False, mask=mask).to(device)

    # Compute the calibration parameters on the calibration set
    if verbose:
        print("Instantiate CQR model and compute the calibration parameters")
    cqr = _instantiate_cqr(
        confidence_uq=confidence_uq, imgsize=imgsize,
        mode=mode, mask=mask, device=device, **kwargs
    )
    if hyperparam_precalib is None and find_optimal_hyperparam_precalib:
        hyperparam_precalib = _get_optimal_hyperparam_precalib(
            cqr,
            kappa_pred, var,
            kappa_pred_calib, var_calib, kappa_true_calib,
            predinterv_metric, confidence_uq=confidence_uq,
            verbose=verbose
        )

    # Compute pre- and post-calibration residuals
    if verbose:
        print("Calibrate residuals with CQR")
    res, res_cqr = _get_residuals_cqr(
        cqr, var, kappa_pred_calib, var_calib,
        kappa_true_calib, confidence_uq=confidence_uq,
        hyperparam_precalib=hyperparam_precalib
    )

    bounds = _get_bounds(kappa_pred, res)
    bounds_cqr = _get_bounds(kappa_pred, res_cqr)

    if kappa_true is not None:
        err = err_metric(bounds, kappa_true).cpu()
        err_cqr = err_metric(bounds_cqr, kappa_true).cpu()
    else:
        err = None
        err_cqr = None

    fake_kappa_true = _get_fake_kappa_true(
        kappa_pred
    ) # No need to have access to the ground truth to compute the error bar size
    predinterv = predinterv_metric(bounds, fake_kappa_true).cpu()
    predinterv_cqr = predinterv_metric(bounds_cqr, fake_kappa_true).cpu()

    out_dict = {
        "state_dict_cqr": cqr.state_dict(),
        "err": err,
        "predinterv": predinterv,
        "err_cqr": err_cqr,
        "predinterv_cqr": predinterv_cqr,
        "hyperparam_precalib": hyperparam_precalib,
    }
    if save_tensors:
        out_dict.update({
            "res": res[:nimgs_save].cpu(),
            "res_cqr": res_cqr[:nimgs_save].cpu() \
                if res_cqr is not None else None,
        })

    return out_dict


def _instantiate_cqr(
        confidence_uq: int | float = CONFIDENCE_UQ,
        imgsize=IMGSIZE,
        mode=MODE_CQR, a=None, mask=None, device="cpu", **kwargs
):
    cqr_class = wlnn.CQR_CLASSES[mode]
    if mode == "chisqcqr":
        kwargs.update(a=a, mask=mask)
    alpha = wlutils.get_alpha_from_confidence(confidence_uq)
    cqr = cqr_class(
        alpha, map_size=imgsize, **kwargs
    ).eval().to(device)

    return cqr


def _get_optimal_hyperparam_precalib(
        cqr: wlcqr.AddCQR | wlcqr.MultCQR,
        kappa_pred: torch.Tensor,
        var: torch.Tensor,
        kappa_pred_calib: torch.Tensor,
        var_calib: torch.Tensor,
        kappa_true_calib: torch.Tensor,
        predinterv_metric: wlpnp.PredInterv,
        confidence_uq: int | float = CONFIDENCE_UQ,
        verbose=False
) -> float:
    if verbose:
        print("Find optimal hyperparameters for CQR")
    if isinstance(cqr, wlcqr.AddCQR):
        bounds_hyperparam = BOUNDS_MULTFACT_CONFIDENCE_UQ
        init_hyperparam = 1.0
    else:
        bounds_hyperparam = BOUNDS_ADDCONST_CONFIDENCE_UQ
        init_hyperparam = 0.0

    def mean_predinterv(params: np.ndarray):

        _, res_cqr = _get_residuals_cqr(
            cqr, var, kappa_pred_calib, var_calib,
            kappa_true_calib, confidence_uq=confidence_uq,
            hyperparam_precalib=params[0]
        )
        bounds_cqr = _get_bounds(kappa_pred, res_cqr)
        fake_kappa_true = _get_fake_kappa_true(
            kappa_pred
        ) # No need to have access to the ground truth to compute the error bar size
        predinterv_cqr = predinterv_metric(
            bounds_cqr, fake_kappa_true
        ) # Shape = (nimgs,)

        return predinterv_cqr.mean().item()

    results_optim = minimize(
        mean_predinterv, x0=init_hyperparam,
        method="Nelder-Mead",
        bounds=(bounds_hyperparam,)
    )
    if verbose:
        print(results_optim)

    return float(results_optim.x[0])


def _get_fake_kappa_true(kappa_pred):
    return torch.empty_like(
        kappa_pred
    )


def _get_bounds(kappa_pred, res):
    kappa_lo = kappa_pred - res
    kappa_hi = kappa_pred + res
    out = torch.stack([kappa_lo, kappa_hi], dim=1) # Shape = (nimgs, 2, 1, nx, ny)
    return out


def _get_residuals_cqr(
        cqr: wlcqr.AddCQR | wlcqr.MultCQR,
        var: torch.Tensor,
        kappa_pred_calib: torch.Tensor,
        var_calib:torch.Tensor,
        kappa_true_calib: torch.Tensor,
        confidence_uq: int | float = CONFIDENCE_UQ,
        hyperparam_precalib: float | None = None
):
    cqr.reset()
    cqr.hyperparam_precalib = hyperparam_precalib
    res = _get_error_bars(
        var, confidence_uq=confidence_uq
    )
    res_calib = _get_error_bars(
        var_calib, confidence_uq=confidence_uq
    )
    cqr.calibrate(kappa_pred_calib, res_calib, kappa_true_calib)
    res_cqr = cqr(res)

    return res, res_cqr


def _get_error_bars(
        var, confidence_uq=CONFIDENCE_UQ
):  
    return confidence_uq * var**0.5


def get_uq_keys(
        mode_cqr=None, scaling_factor_chisqcqr=None,
        rho=None, const=None,
):
    uq_key = "uq"
    if mode_cqr is not None and mode_cqr != "addcqr":
        uq_key = f"{uq_key}_{mode_cqr}"
        if mode_cqr == "chisqcqr" and scaling_factor_chisqcqr is not None:
            uq_key = f"{uq_key}_a_{scaling_factor_chisqcqr:.3f}"
    if rho is not None:
        uq_key = f"{uq_key}_rho_{rho:.3f}"
    if const is not None:
        uq_key = f"{uq_key}_const_{const:.3f}"
    return uq_key


def save_results(
        out_dict, output_dir, now, prefix=None,
        verbose=False, **kwargs
):
    path_to_output = _complete_path_to_torch_saved_objects(
        output_dir, now, prefix=prefix, **kwargs
    )
    if verbose:
        print(f"Save results to {path_to_output}")

    os.makedirs(os.path.dirname(path_to_output), exist_ok=True)
    torch.save(out_dict, path_to_output)


def _complete_path_to_torch_saved_objects(
        output_dir, timestamp, prefix=None, step_size=None
):
    filename = prefix if prefix is not None else ""
    if step_size is not None:
        filename = f"{filename}_step-size_{step_size:.3f}"
    filename = f"{filename}_{timestamp}.pt"
    filename = filename.lstrip("_")

    return os.path.join(output_dir, filename)
