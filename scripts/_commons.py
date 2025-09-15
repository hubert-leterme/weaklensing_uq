import os
import warnings
import time
import random
import argparse
import tqdm
import numpy as np
from scipy.optimize import minimize
import torch
import astropy.table as aptable
import deepinv as dinv

from wlmmuq import cosmos as wlcosmos
from wlmmuq import kappatng as wlktng
from wlmmuq import utils as wlutils
from wlmmuq.data import torch as wlbl
from wlmmuq import models as wlnn
from wlmmuq.models.deepinv import iterativemm as wlpnp
from wlmmuq.models.deepinv import pnpmcalens as wlpnpmcalens
from wlmmuq.models import cqr as wlcqr
from wlmmuq.models.deepinv import callbacks as wlcallbacks

from wlmmuq import CHECKPOINT_DIR, PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS, \
    PATH_TO_TEST_DATASET, PATH_TO_CALIB_DATASET, KEY_REPLACEMENT_DICT
from wlmmuq.kappatng import OPENINGANGLE
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER
from wlmmuq.models.deepinv.pnpmcalens import \
    NITER_PER_STEP_G, NITER_PER_STEP_NG, STARLET_DETECTION_THRESHOLD

NINPIMGS = 100 # Number of input images before cropping
NIMGS_TEST = 512 # Images extracted from the 57 first original files (copped dataset)
NIMGS_CALIB = 1024 # Images extracted from the 43 remaining original files (augmented dataset)
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
CONFIDENCE_UQ = 2 # 2-sigma confidence

INPAINTING_WIENER = False
INPAINTING_PNPMASS = False
INPAINTING_DEEPMASS = True

MODE_CQR = "addcqr"
BOUNDS_MULTFACT_CONFIDENCE_UQ = (0., 2.)
BOUNDS_ADDCONST_CONFIDENCE_UQ = (-0.005, 0.005)

def set_seed(seed):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_path_to_output(output_dir, output_filename, checkpoint_dir=None):
    if checkpoint_dir is not None:
        output_dir = os.path.join(checkpoint_dir, output_dir)
    return os.path.join(output_dir, output_filename)


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
        data_dict = wlktng.get_data_from_cosmos_ktng(cat_cosmos, imgsize)
        shapedisp = data_dict["shapedisp"]
        ngal = data_dict["ngal"]
        mask = data_dict["mask"]
        std_noise = wlutils.get_std_noise(ngal, shapedisp, std_noise_mask=0)

    if inpainting:
        # Set the noise standard deviation for masked data
        std_noise[~mask] = std_noise.max()

    return std_noise, mask


def create_dataset_from_kappatng(
        func:callable, path_to_saved_dataset:str, idx_lp: int | str,
        openingangle: float, ninpimgs: int, verbose: bool=False, **kwargs
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
    openingangle : floatfrom wlmmuq.kappatng import OPENINGANGLE
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


def get_powerspectrum_from_dataset(
        hdf5_filepath, nimgs, device=None, verbose=False, **kwargs
):
    if verbose:
        print(f"Compute the power spectrum of {nimgs} images")
    dataloader = wlbl.HDF5DatasetKappa(
        hdf5_filepath, nimgs=nimgs, shuffle=True, **kwargs
    ).to_dataloader()
    dataloader = iter(dataloader)

    list_of_powerspectrum = []
    while True:
        try:
            kappa_ps = next(dataloader)
        except StopIteration:
            break
        if device is not None:
            kappa_ps = kappa_ps.to(device)
        list_of_powerspectrum.append(
            wlutils.get_powerspectrum(kappa_ps)
        )
    powerspectrum = torch.stack(list_of_powerspectrum, dim=0)
    powerspectrum = torch.mean(powerspectrum, dim=0)

    return powerspectrum


def get_output_type(order2=False, additional_outlayer=None):
    if not order2:
        output_type = "pe" # Point estimate
    else:
        output_type = "var" # Variance
    if additional_outlayer is not None:
        output_type = f"{output_type}_{additional_outlayer}"
    return output_type


def get_checkpoint_dirs(
        checkpoint_dir, checkpoint_subdir=None, checkpoint_subdir_uq=None
):
    checkpoint_dir0 = checkpoint_dir
    if checkpoint_subdir is not None:
        checkpoint_dir = os.path.join(checkpoint_dir0, checkpoint_subdir)
    else:
        raise ValueError("Argument `checkpoint_subdir` must be provided.")
    if checkpoint_subdir_uq is not None:
        checkpoint_dir_uq = os.path.join(checkpoint_dir0, checkpoint_subdir_uq)
        warnings.warn(
            f"The model used for UQ ({checkpoint_dir_uq}) is not the same as "
            f"the one used for the point estimate ({checkpoint_dir})"
        )
    else:
        checkpoint_dir_uq = checkpoint_dir

    return checkpoint_dir, checkpoint_dir_uq


def get_path_to_checkpoint(save_path, timestamp, epoch):
    path_to_checkpoint = os.path.join(
        save_path, timestamp, f"ckp_{epoch}.pth.tar"
    )
    return path_to_checkpoint


def update_kwargs_model(
        kwargs_model,
        std_noise=None, mask=None, path_to_ps=None,
        noise_whitening_wiener=None,
        eps_sup_step_size_wiener=None,
        niter_wiener=None, device="cpu", verbose=False
):
    try:
        mode_preproc = kwargs_model["mode_preproc"]
    except KeyError:
        mode_preproc = None
    if mode_preproc is not None:
        # Load arguments for Wiener or KS initialization
        # Only for DeepMass (denoiser = False)
        if mode_preproc == "wiener":
            args_preproc = get_args_wienerinit(
                std_noise, mask, path_to_ps=path_to_ps,
                noise_whitening=noise_whitening_wiener,
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
        key_replacement_dict=KEY_REPLACEMENT_DICT,
        std_noise=None, mask=None, path_to_ps=PATH_TO_PS,
        noise_whitening_wiener=False,
        eps_sup_step_size_wiener=EPS_SUP_STEP_SIZE,
        niter_wiener=NITER_WIENER,
        device="cpu", verbose=False, **kwargs
):
    update_kwargs_model(
        kwargs,
        std_noise=std_noise, mask=mask, path_to_ps=path_to_ps,
        noise_whitening_wiener=noise_whitening_wiener,
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
        output_type = get_output_type(
            order2=order2,
            additional_outlayer=additional_outlayer
        )
        save_path = os.path.join(checkpoint_dir, output_type)
        path_to_checkpoint = get_path_to_checkpoint(
            save_path, timestamp, epoch
        )
    checkpoint = torch.load(path_to_checkpoint, map_location=device)
    state_dict = checkpoint['state_dict']
    if key_replacement_dict is not None:
        for old_key, new_key in key_replacement_dict.items():
            if old_key in state_dict:
                if verbose:
                    print(f"Replacing key '{old_key}' with '{new_key}'")
                state_dict[new_key] = state_dict.pop(old_key)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=EPOCH,
        load_model_uq=False, checkpoint_dir_uq=None,
        arch_uq=None, timestamp_uq=None, epoch_uq=None,
        imgsize=IMGSIZE,
        std_noise=None, mask=None, path_to_ps=PATH_TO_PS,
        noise_whitening_wiener=False,
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
        noise_whitening_wiener=noise_whitening_wiener,
        eps_sup_step_size_wiener=eps_sup_step_size_wiener,
        niter_wiener=niter_wiener,
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
            noise_whitening_wiener=noise_whitening_wiener,
            eps_sup_step_size_wiener=eps_sup_step_size_wiener,
            niter_wiener=niter_wiener,
            device=device, verbose=verbose,
            **kwargs_model_uq
        )
    else:
        model_uq = None

    return model, model_uq


def instantiate_starlet_denoiser(
        imgsize=IMGSIZE,
        starlet_detection_threshold=STARLET_DETECTION_THRESHOLD,
        device="cpu", verbose=False
):
    denoiser = wlpnpmcalens.Starlet2d(
        in_channels=1, nx=imgsize, ny=imgsize,
        detection_threshold=starlet_detection_threshold
    ).to(device)
    if verbose:
        print(
            f"Starlet denoiser instantiated with {denoiser.ns} scales and "
            f"a {int(starlet_detection_threshold)}-sigma detection threshold."
        )
    denoiser_uq = None # Conformal prediction computed from zero-valued error bars
    callback = wlpnpmcalens.StarletResetter(denoiser)

    return denoiser, denoiser_uq, callback


def get_dataloader_massmapping(
        path_to_dataset, nimgs, imgsize, batch_size, num_workers, std_noise, mask,
        **kwargs
):
    test_dataloader = wlbl.HDF5DatasetMassMapping(
        hdf5_filepath=path_to_dataset, nimgs=nimgs, batch_size=batch_size,
        std_noise=std_noise, mask=mask, output_shape=imgsize,
        newaxis=True, num_workers=num_workers, **kwargs
    ).to_dataloader()

    return test_dataloader


def _get_datafidelity_params(
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        step_size=None, multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    step_size, param_mahalanobis = \
            get_step_size_param_mahalanobis(
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


def get_datafidelity_prior_params_gaussian(
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


def get_datafidelity_prior_params_nongaussian(
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


def get_wiener(
        path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        step_size=None, eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_WIENER, device="cpu", verbose=False
):
    if verbose:
        print("Get optimizer for iterative Wiener filtering")
    data_fidelity, prior, params_algo = get_datafidelity_prior_params_gaussian(
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
        white_noise=False, noise_whitening_wiener=False,
        imgsize=IMGSIZE, std_noise=None, physics=None,
        step_size=None, step_size_ng=None,
        multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_WIENER,
        starlet_detection_threshold=STARLET_DETECTION_THRESHOLD,
        mcalens_update_ng_first=False,
        device="cpu", verbose=False
):
    data_fidelity_g, prior_g, params_algo_g = get_datafidelity_prior_params_gaussian(
        path_to_ps=path_to_ps,
        white_noise=white_noise, noise_whitening=noise_whitening_wiener,
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
        denoiser_ng, _, callback = instantiate_starlet_denoiser(
            imgsize=imgsize,
            starlet_detection_threshold=starlet_detection_threshold,
            device=device, verbose=verbose
        )
        data_fidelity_ng, prior_ng, _, params_algo_ng = \
                get_datafidelity_prior_params_nongaussian(
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
            update_ng_first=mcalens_update_ng_first,
            output_mode="discard_ng", verbose=verbose
        ).to(device)

    else:
        raise ValueError(
            f"Invalid extractor '{which}'. "
            "Supported extractors are 'wiener' and 'mcalens'."
        )

    return extractor, callback


def get_pnpmass(
        denoiser, denoiser_uq, imgsize=IMGSIZE,
        std_noise=None, rmse_fn=None, physics=None,
        step_size=None, multfact_step_size=None,
        eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_PNPMASS, mode="regular",
        which_gaussian_extractor=WHICH_GAUSSIAN_EXTRACTOR,
        update_ng_first=False,
        path_to_ps=PATH_TO_PS,
        noise_whitening_wiener=False,
        multfact_step_size_gaussian=None,
        niter_wiener=NITER_WIENER,
        starlet_detection_threshold=STARLET_DETECTION_THRESHOLD,
        niter_per_step_g=NITER_PER_STEP_G,
        niter_per_step_ng=NITER_PER_STEP_NG,
        device="cpu", verbose=False
):
    data_fidelity, prior, prior_uq, params_algo = \
            get_datafidelity_prior_params_nongaussian(
        denoiser, denoiser_uq=denoiser_uq,
        white_noise=False, std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_step_size=multfact_step_size,
        eps_sup_step_size=eps_sup_step_size,
        device=device, verbose=verbose
    )
    step_size = params_algo["stepsize"]
    kwargs = {}
    if rmse_fn is not None:
        kwargs.update(metric_dict={"rmse": rmse_fn})

    if mode in ["regular", "residual"]:
        if mode == "residual":
            if verbose:
                print("Instantiate PnPMass on residuals (ResPnPMass)")

            # Note: the step sizes for the Gaussian extractor are computed automatically
            gaussian_extractor, callback_gaussian_extractor = \
                    get_gaussian_extractor(
                which=which_gaussian_extractor,
                path_to_ps=path_to_ps,
                white_noise=False, noise_whitening_wiener=noise_whitening_wiener,
                imgsize=imgsize, std_noise=std_noise, physics=physics,
                step_size=None, step_size_ng=None,
                multfact_step_size=multfact_step_size_gaussian,
                eps_sup_step_size=eps_sup_step_size,
                niter=niter_wiener,
                starlet_detection_threshold=starlet_detection_threshold,
                mcalens_update_ng_first=update_ng_first,
                device=device, verbose=False
            )
        else:
            if verbose:
                print("Instantiate PnPMass")
            gaussian_extractor = None
            callback_gaussian_extractor = None
        pnpmass = wlpnp.optim_builder(
            iteration="PGD", params_algo=params_algo.copy(),
            data_fidelity=data_fidelity, prior=prior,
            early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
            verbose=verbose, **kwargs
        ).to(device)

    elif mode == "pnpmcalens":
        if verbose:
            print("Instantiate PnPMCALens")

        # Note: the step size for the Gaussian component is computed automatically
        data_fidelity_g, prior_g, params_algo_g = get_datafidelity_prior_params_gaussian(
            path_to_ps=path_to_ps,
            white_noise=False, noise_whitening=noise_whitening_wiener,
            std_noise=std_noise, physics=physics,
            step_size=None,
            multfact_step_size=multfact_step_size,
            eps_sup_step_size=eps_sup_step_size,
            device=device, verbose=verbose
        )
        pnpmass = wlpnpmcalens.optim_builder_mcalens(
            iteration_g="PGD", iteration_ng="PGD",
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            params_algo_g=params_algo_g.copy(), params_algo_ng=params_algo.copy(),
            data_fidelity_g=data_fidelity_g, data_fidelity_ng=data_fidelity,
            prior_g=prior_g, prior_ng=prior,
            early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
            update_ng_first=update_ng_first,
            output_mode="add_components", verbose=True, **kwargs
        ).to(device)
        gaussian_extractor = None
        callback_gaussian_extractor = None

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Supported modes are 'regular', 'residual', and 'pnpmcalens'."
        )

    if denoiser_uq is not None:
        pnpmass_uq = wlpnp.optim_builder(
            iteration="PGD", params_algo=params_algo.copy(),
            data_fidelity=data_fidelity, prior=prior_uq,
            early_stop=False, max_iter=1, custom_init=wlpnp.zero_init,
            verbose=verbose
        ).to(device)
    else:
        pnpmass_uq = None

    out = (
        pnpmass, pnpmass_uq, gaussian_extractor,
        step_size, callback_gaussian_extractor
    )

    return out


# TODO: Merge the 3 functions below into one single function
def run_wiener_batch(
        wiener: wlpnp.BaseOptim, physics: wlpnp.MassMapping,
        dataloader,
        rmse_fn: wlpnp.RMSE | None=None,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_var = [] # Zero-valued tensors
    listof_rmse = []
    listof_l2norm = []

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    for kappa_true, gamma_noisy in pbar:
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_pred = wiener(gamma_noisy, physics)
            var = torch.zeros(kappa_true.shape, device=device)
            if rmse_fn is not None:
                rmse = rmse_fn(kappa_pred, kappa_true)
                l2norm = rmse_fn(kappa_true, 0)
            else:
                rmse = None
                l2norm = None

            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_rmse.append(rmse) # Shape = (batch_size,)
            listof_l2norm.append(l2norm) # Shape = (batch_size,)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs,)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs,)
    except TypeError:
        rmse = None
        l2norm = None

    out = {
        "kappa_true": kappa_true,
        "kappa_pred": kappa_pred,
        "var": var,
        "rmse": rmse,
        "l2norm": l2norm,
    }
    return out


def run_pnpmass_batch(
        pnpmass: wlpnp.BaseOptim, pnpmass_uq: wlpnp.BaseOptim | None,
        physics: wlpnp.MassMapping,
        dataloader, step_size, niter,
        rmse_fn: wlpnp.RMSE | None=None,
        gaussian_extractor: wlpnp.BaseOptim | None=None,
        callbacks: wlcallbacks.CallbackList | None=None,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_var = []
    listof_rmse = []
    listof_l2norm = []

    if callbacks is None:
        callbacks = wlcallbacks.BaseCallback()

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    pbar.set_description(f"Step size = {step_size:.2e}, Nb iterations = {niter}")
    for i, (kappa_true, gamma_noisy) in enumerate(pbar):
        callbacks.on_batch_begin(i)
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            if gaussian_extractor is not None:
                kappa_g = gaussian_extractor(
                    gamma_noisy, physics, x_gt=None, compute_metrics=False
                )
                gamma_noisy = gamma_noisy - physics.A(kappa_g)
                kappa_true = kappa_true - kappa_g

            kappa_pred, metrics = pnpmass(
                gamma_noisy, physics, x_gt=kappa_true, compute_metrics=True
            )
            if pnpmass_uq is not None:
                pnpmass_uq.custom_init.X_init = (kappa_pred,)
                var = pnpmass_uq(
                    gamma_noisy, physics, compute_metrics=False
                )
            else:
                var = torch.zeros(kappa_pred.shape, device=device)

            if gaussian_extractor is not None:
                kappa_pred = kappa_pred + kappa_g
                kappa_true = kappa_true + kappa_g

            if rmse_fn is not None:
                l2norm = rmse_fn(kappa_true, 0)
            else:
                l2norm = None

        listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_rmse.append(metrics["rmse"]) # Shape = (batch_size, niter)
        listof_l2norm.append(l2norm) # Shape = (batch_size, niter)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs, niter)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs, niter)
    except TypeError:
        rmse = None
        l2norm = None

    out = {
        "kappa_true": kappa_true,
        "kappa_pred": kappa_pred,
        "var": var,
        "rmse": rmse,
        "l2norm": l2norm,
    }
    return out


def run_deepmass_batch(
        deepmass: wlpnp.BaseOptim, deepmass_uq: wlpnp.BaseOptim,
        dataloader,
        rmse_fn: wlpnp.RMSE | None=None,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_var = []
    listof_rmse = []
    listof_l2norm = []

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    for kappa_true, gamma_noisy in pbar:
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_pred = deepmass(gamma_noisy)
            if deepmass_uq is not None:
                var = deepmass_uq(gamma_noisy)
            else:
                var = torch.zeros(kappa_true.shape, device=device)
            if rmse_fn is not None:
                rmse = rmse_fn(kappa_pred, kappa_true)
                l2norm = rmse_fn(kappa_true, 0)
            else:
                rmse = None
                l2norm = None

            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_rmse.append(rmse) # Shape = (batch_size,)
            listof_l2norm.append(l2norm) # Shape = (batch_size,)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs,)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs,)
    except TypeError:
        rmse = None
        l2norm = None

    out = {
        "kappa_true": kappa_true,
        "kappa_pred": kappa_pred,
        "var": var,
        "rmse": rmse,
        "l2norm": l2norm,
    }
    return out


def get_error_bars(
        var, confidence_uq=CONFIDENCE_UQ,
        multfact_confidence_uq=None,
        addconst_confidence_uq=None
):  
    if multfact_confidence_uq is None:
        multfact_confidence_uq = 1.
    if addconst_confidence_uq is None:
        addconst_confidence_uq = 0.
    out = multfact_confidence_uq * var**0.5 + addconst_confidence_uq
    out = torch.relu(out)
    out = confidence_uq * out

    return out


def get_inference_time(beg_time, which="inference", verbose=False):
    inference_time = time.time() - beg_time
    if verbose:
        print(f"Total {which} time: {inference_time:.2f} seconds")
    return inference_time


def get_args_wienerinit(
        std_noise, mask, path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        step_size=None, eps_sup_step_size=EPS_SUP_STEP_SIZE,
        niter=NITER_WIENER, device="cpu", verbose=False
):
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)
    powerspectrum = torch.load(path_to_ps)
    step_size, _ = get_step_size_param_mahalanobis(
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size, eps=eps_sup_step_size,
        device=device, verbose=verbose
    ) # Bayesian Wiener filtering
    args_wienerinit = dict(
        step_size=step_size, powerspectrum=powerspectrum,
        std_noise=std_noise, mask=mask, niter=niter,
        noise_whitening=noise_whitening
    )
    return args_wienerinit


def get_step_size_param_mahalanobis(
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        step_size=None, multfact_step_size=None,
        eps=EPS_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    if not white_noise:
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


def convert_into_param_lists(
        params1, params2, find_optimal_hyperparam=False
):
    if isinstance(params1, list) and isinstance(params2, list):
        assert len(params1) == len(params2)
    else:
        if isinstance(params1, list) and not isinstance(params2, list):
            params2 = len(params1) * [params2]
        elif not isinstance(params1, list) and isinstance(params2, list):
            params1 = len(params2) * [params1]
        else:
            params1 = [params1]
            params2 = [params2]

    if find_optimal_hyperparam and (None, None) not in zip(params1, params2):
        params1.append(None)
        params2.append(None)

    return params1, params2


def instantiate_cqr(
        confidence_uq=CONFIDENCE_UQ, imgsize=IMGSIZE,
        mode=MODE_CQR, device="cpu"
):
    if mode == "addcqr":
        cqr_class = wlcqr.AddCQR
    elif mode == "multcqr":
        cqr_class = wlcqr.MultCQR
    else:
        raise ValueError(
            f"Invalid CQR mode '{mode}'. "
            "Supported modes are 'addcqr' and 'multcqr'."
        )
    alpha = wlutils.get_alpha_from_confidence(confidence_uq)
    cqr = cqr_class(alpha, map_size=imgsize).eval().to(device)

    return cqr


def apply_calibration_and_get_metrics(
        kappa_pred, var, kappa_true,
        kappa_pred_calib, var_calib, kappa_true_calib,
        confidence_uq=CONFIDENCE_UQ,
        imgsize=IMGSIZE, mode=MODE_CQR,
        multfact_confidence_uq=None,
        addconst_confidence_uq=None,
        find_optimal_hyperparam=False,
        mask=None, save_tensors=False, nimgs_save=NIMGS_SAVE,
        device="cpu", verbose=False
):
    err_metric = wlpnp.MiscoverageRate(meancentering=False, mask=mask).to(device)
    predinterv_metric = wlpnp.PredInterv(meancentering=False, mask=mask).to(device)

    # Compute the calibration parameters on the calibration set
    if verbose:
        print("Instantiate CQR model and compute the calibration parameters")
    cqr = instantiate_cqr(
        confidence_uq=confidence_uq, imgsize=imgsize,
        mode=mode, device=device
    )
    if find_optimal_hyperparam \
            and multfact_confidence_uq is None \
            and addconst_confidence_uq is None:
        kwargs = get_optimal_hyperparams_uq(
            cqr,
            kappa_pred, var,
            kappa_pred_calib, var_calib, kappa_true_calib,
            predinterv_metric, confidence_uq=confidence_uq,
            verbose=verbose
        )
    else:
        kwargs = dict(
            multfact_confidence_uq=multfact_confidence_uq,
            addconst_confidence_uq=addconst_confidence_uq
        )

    # Compute pre- and post-calibration residuals
    if verbose:
        print("Calibrate residuals with CQR")
    res, res_cqr = _get_residuals_cqr(
        cqr, var, kappa_pred_calib, var_calib,
        kappa_true_calib, confidence_uq=confidence_uq,
        **kwargs
    )

    bounds = _get_bounds(kappa_pred, res)
    err = err_metric(bounds, kappa_true)
    predinterv = predinterv_metric(bounds, kappa_true)

    bounds_cqr = _get_bounds(kappa_pred, res_cqr)
    err_cqr = err_metric(bounds_cqr, kappa_true)
    predinterv_cqr = predinterv_metric(bounds_cqr, kappa_true)

    out_dict = {
        "state_dict_cqr": cqr.state_dict(),
        "err": err.cpu(),
        "predinterv": predinterv.cpu(),
        "err_cqr": err_cqr.cpu(),
        "predinterv_cqr": predinterv_cqr.cpu(),
    }
    if save_tensors:
        out_dict.update({
            "res": res[:nimgs_save].cpu(),
            "res_cqr": res_cqr[:nimgs_save].cpu() \
                if res_cqr is not None else None,
        })

    return out_dict


def get_optimal_hyperparams_uq(
        cqr: wlcqr.AddCQR | wlcqr.MultCQR,
        kappa_pred: torch.Tensor,
        var: torch.Tensor,
        kappa_pred_calib: torch.Tensor,
        var_calib: torch.Tensor,
        kappa_true_calib: torch.Tensor,
        predinterv_metric: wlpnp.PredInterv,
        confidence_uq: int | float=CONFIDENCE_UQ,
        verbose=False
):
    if verbose:
        print("Find optimal hyperparameters for CQR")
    if isinstance(cqr, wlcqr.AddCQR):
        active_param_key = "multfact_confidence_uq"
        other_param_key = "addconst_confidence_uq"
        bounds_hyperparam = BOUNDS_MULTFACT_CONFIDENCE_UQ
        init_hyperparam = 1.0
    else:
        active_param_key = "addconst_confidence_uq"
        other_param_key = "multfact_confidence_uq"
        bounds_hyperparam = BOUNDS_ADDCONST_CONFIDENCE_UQ
        init_hyperparam = 0.0

    def mean_predinterv(params: np.ndarray):

        kwargs = {active_param_key: params[0]}
        _, res_cqr = _get_residuals_cqr(
            cqr, var, kappa_pred_calib, var_calib,
            kappa_true_calib, confidence_uq=confidence_uq,
            **kwargs
        )
        bounds_cqr = _get_bounds(kappa_pred, res_cqr)
        fake_kappa_true = torch.empty_like(
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
    out_dict = {
        active_param_key: results_optim.x[0],
        other_param_key: None
    }
    return out_dict


def get_uq_keys(rho=None, const=None):
    uq_key = "uq"
    if rho is not None:
        uq_key = f"{uq_key}_rho_{rho:.3f}"
    if const is not None:
        uq_key = f"{uq_key}_const_{const:.3f}"
    return uq_key


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
        confidence_uq: int | float=CONFIDENCE_UQ,
        **kwargs
):
    res = get_error_bars(
        var, confidence_uq=confidence_uq, **kwargs
    )
    res_calib = get_error_bars(
        var_calib, confidence_uq=confidence_uq, **kwargs
    )
    cqr.calibrate(kappa_pred_calib, res_calib, kappa_true_calib)
    res_cqr = cqr(res)

    return res, res_cqr


def save_results(
        out_dict, path_to_output, now,
        verbose=False, **kwargs
):
    path_to_output = _complete_path_to_torch_saved_objects(
        path_to_output, now, **kwargs
    )
    if verbose:
        print(f"Save results to {path_to_output}")

    torch.save(out_dict, path_to_output)


def _complete_path_to_torch_saved_objects(
        path, timestamp, step_size=None
):
    if step_size is not None:
        path = f"{path}_step-size_{step_size:.3f}"
    path = f"{path}_{timestamp}.pt"

    return path


def add_arguments_create_dataset(parser):

    parser.add_argument(
        "--idx-lp", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Index of the learning potential, indicating which folder to look "
            "into for the HDF5 files containing the dataset (`LPxxx` where `xxx` "
            "ranges from `001` to `100`). Default = `001`"
        )
    )
    parser.add_argument(
        "--openingangle", type=float,
        default=argparse.SUPPRESS,
        help=f"Opening angle (deg). Default = {OPENINGANGLE}"
    )
    parser.add_argument(
        "--ninpimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            f"Number of input images. Default = {NINPIMGS}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size, to avoid memory overload. "
            "Default = 50"
        )
    )


def add_arguments_model(parser, uq=False):

    parser.add_argument(
        "-a", "--arch", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the model. Possible values are: "
            f"{' | '.join(wlnn.MODEL_CLASSES.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "-s", "--model-size", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Size of the model (DRUNet only). Possible values are: "
            f"{' | '.join(wlnn.torch.MODEL_SIZE_DRUNET.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "--no-bias", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not use bias in convolution or batch "
            "normalization layers."
        )
    )
    parser.add_argument(
        "-m", "--mode-preproc", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Preprocessing mode for DeepMass: 'wiener' or 'ks'. "
            "Default = None"
        )
    )
    if uq:
        parser.add_argument(
            "--additional-outlayer", type=str,
            default=argparse.SUPPRESS,
            help=(
                "Type of additional output layer. "
                "Only used for training order-2 models. "
                "Possible values are: 'meancentering' | 'leakyrelu'. "
                "In any case, ReLU is applied at the output in evaluation mode. "
                "Default = None"
            )
        )


def add_arguments_model_order1(parser):

    parser.add_argument(
        "-a1", "--arch-order1", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the order-1 model. Possible values are: "
            f"{' | '.join(wlnn.MODEL_CLASSES.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "-s1", "--model-size-order1", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Size of the order-1 model (DRUNet only). Possible values are: "
            f"{' | '.join(wlnn.torch.MODEL_SIZE_DRUNET.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "--no-bias-order1", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not use bias in convolution or batch "
            "normalization layers (order-1 model)."
        )
    )
    parser.add_argument(
        "-m1", "--mode-preproc-order1", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Preprocessing mode for DeepMass (order-1 model): 'wiener' or 'ks'. "
            "Default = None"
        )
    )


def add_arguments_model_uq(parser):

    parser.add_argument(
        "-auq", "--arch-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Architecture of the order-2 model, if different from `--arch`. "
            "Possible values are: "
            f"{' | '.join(wlnn.MODEL_CLASSES.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "-suq", "--model-size-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Size of the order-2 model (DRUNet only). Possible values are: "
            f"{' | '.join(wlnn.torch.MODEL_SIZE_DRUNET.keys())}. Default = None"
        )
    )
    parser.add_argument(
        "--no-bias-uq", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Do not use bias in convolution or batch "
            "normalization layers (order-2 models)."
        )
    )
    parser.add_argument(
        "-muq", "--mode-preproc-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Preprocessing mode for DeepMass (order-2 model): 'wiener' or 'ks'. "
            "Default = None"
        )
    )
    parser.add_argument(
        "--additional-outlayer-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Type of additional output layer (order-2 model). "
            "Possible values are: 'meancentering' | 'leakyrelu'. "
            "In any case, ReLU is applied at the output in evaluation mode. "
            "Default = None"
        )
    )


def add_arguments_checkpoint(parser):

    parser.add_argument(
        "--checkpoint-dir", type=str,
        default=argparse.SUPPRESS,
        help=(
            f"Checkpoint parent directory. Default = {CHECKPOINT_DIR}"
        )
    )
    parser.add_argument(
        "-c", "--checkpoint-subdir", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Subdirectory containing the save checkpoints. Default is None."
        )
    )
    parser.add_argument(
        "-t", "--timestamp", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Timestamp of the model to load. "
            "Default = None"
        )
    )
    parser.add_argument(
        "-e", "--epoch", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Epoch of the model to load. "
            f"Default = {EPOCH}"
        )
    )
    parser.add_argument(
        "-uq", "--load-model-uq", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Load the order-2 moment network, for UQ."
        )
    )
    parser.add_argument(
        "-c0", "--checkpoint-subdir-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Checkpoint subdirectory for the order-2 moment network, "
            "if different from `--checkpoint-subdir`."
        )
    )
    parser.add_argument(
        "-t0", "--timestamp_uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Timestamp of the order-1 model to load. "
            "Default = None"
        )
    )
    parser.add_argument(
        "-e0", "--epoch_uq", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Epoch of the model to load. "
            f"Default is the same value as `--epoch` ({EPOCH} if not provided)."
        )
    )


def add_arguments_dataset(parser, batch_size):

    parser.add_argument(
        "--imgsize", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of pixels (width) in input images. "
            f"Default = {IMGSIZE}"
        )
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size. "
            f"Default = {batch_size}"
        )
    )
    parser.add_argument(
        "-w", "--num-workers", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of workers for parallel processing. Only work for PyTorch datasets. "
            f"Default = {NUM_WORKERS}"
        )
    )


def add_arguments_test_calib_dataset(parser, batch_size):

    parser.add_argument(
        "--path-to-test-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the test set (HDF5 file). "
            f"Default = {PATH_TO_TEST_DATASET}"
        )
    )
    parser.add_argument(
        "--path-to-calib-dataset", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the calibration set (HDF5 file). "
            f"Default = {PATH_TO_CALIB_DATASET}"
        )
    )
    parser.add_argument(
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of test images. "
            f"Default = {NIMGS_TEST}"
        )
    )
    parser.add_argument(
        "--nimgs-calib", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of calibration images. "
            f"Default = {NIMGS_CALIB}"
        )
    )
    parser.add_argument(
        "-f", "--min-idx-filename-ori-calib",
        type=int, default=argparse.SUPPRESS,
        help=(
            "Filter images by filenames with indices equal or larger than this value. "
            f"Default = {MIN_IDX_FILENAME_ORI_CALIB}."
        )
    )
    add_arguments_dataset(parser, batch_size)


def add_arguments_cqr(parser, zero_init_bounds=False):

    parser.add_argument(
        "--cqr", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Calibrate with CQR."
        )
    )
    parser.add_argument(
        "--mode-cqr", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Mode for CQR. Possible values are: 'addcqr' | 'multcqr'. "
            f"Default = {MODE_CQR}"
        )
    )
    parser.add_argument(
        "--confidence-uq", type=float,
        default=argparse.SUPPRESS,
        help=f"Level of confidence for UQ. Default = {CONFIDENCE_UQ:.1f}-sigma"
    )
    if not zero_init_bounds:
        parser.add_argument(
            "-rho", "--multfact-confidence-uq", type=float, nargs='+',
            default=argparse.SUPPRESS,
            help=(
                "Multiplicative factor for the level of confidence for UQ. "
                "Several values can be provided. "
                "Default = None"
            )
        )
        parser.add_argument(
            "-const", "--addconst-confidence-uq", type=float, nargs='+',
            default=argparse.SUPPRESS,
            help=(
                "Additive constant for the level of confidence for UQ. "
                "Several values can be provided. "
                "Default = None"
            )
        )
        parser.add_argument(
            "--find-optimal-hyperparam-cqr", action='store_true',
            default=argparse.SUPPRESS
        )


def add_arguments_wiener(parser):

    parser.add_argument(
        "-ps", "--path-to-ps", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the power spectrum file used for Wiener initialization. "
            f"Default = '{PATH_TO_PS}'"
        )
    )
    parser.add_argument(
        "--niter-wiener", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of iterations for Wiener initialization. "
            f"Default = {NITER_WIENER}"
        )
    )
    parser.add_argument(
        "-nw", "--noise-whitening-wiener", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Iterative Wiener filtering with noise-whitening data fidelity."
        )
    )


def add_arguments_pnpmode(parser):

    parser.add_argument(
        "--mode", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Mode for PnPMass. Possible values are: "
            "'regular', 'residual', 'pnpmcalens'. "
            "Default = 'regular'"
        )
    )
    parser.add_argument(
        "--which-gaussian-extractor", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Type of Gaussian extractor. Possible values are 'wiener' or 'mcalens'. "
            "Only used if `--mode` is set to 'residual'. "
            f"Default = '{WHICH_GAUSSIAN_EXTRACTOR}'"
        )
    )
    parser.add_argument(
        "--update-ng-first", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Update the non-Gaussian component before the Gaussian component ."
            "Works with `--mode residual --which-gaussian-extractor mcalens` "
            "or `--mode pnpmcalens`."
        )
    )
    parser.add_argument(
        "--starlet", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Use a starlet denoiser instead of a trained model. "
            "Only used if `--mode` is set to 'pnpmcalens'. "
            "This option should be activated for standard MCALens."
        )
    )
    parser.add_argument(
        "-thresh", "--starlet-detection-threshold", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Detection threshold for computing the support of active "
            "starlet coefficients. "
            "Works with `--mode residual --which-gaussian-extractor mcalens` "
            "or `--mode pnpmcalens --starlet`. "
            f"Default = {int(STARLET_DETECTION_THRESHOLD)}-sigma"
        )
    )
    parser.add_argument(
        "-ig", "--niter-per-step-g", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of iterations for one Gaussian step in PnPMCALens. "
            f"Default = {NITER_PER_STEP_G}"
        )
    )
    parser.add_argument(
        "-ing", "--niter-per-step-ng", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of iterations for one non-Gaussian step in PnPMCALens. "
            f"Default = {NITER_PER_STEP_NG}"
        )
    )
    parser.add_argument(
        "--multfact-step-size-gaussian", type=float,
        default=argparse.SUPPRESS,
        help=(
            "Multiplicative factor for the step size in Gaussian extraction. "
            "Only used if `--mode` is set to 'residual' or 'pnpmcalens'. "
        )
    )
    add_arguments_wiener(parser)


def add_arguments_output(parser, output_filename):

    parser.add_argument(
        "-o", "--output-filename", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Output filename (without extension). "
            f"Default = '{output_filename}'"
        )
    )
    parser.add_argument(
        "--save-tensors", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "If set, the tensors of the true convergence, "
            "the kappa map estimate, the variance, and the residuals "
            "will be saved in the output file. WARNING: this will increase "
            "the size of the output file significantly!"
        )
    )
    parser.add_argument(
        "--nimgs-save", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images to save. "
            f"Default = {NIMGS_SAVE}"
        )
    )


def add_arguments_seed_verbose(parser):

    parser.add_argument(
        "--seed", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Seed for the random number generators"
        )
    )
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        default=argparse.SUPPRESS
    ) 
