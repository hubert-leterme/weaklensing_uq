import os
import warnings
import time
import random
import argparse
import tqdm
import numpy as np
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

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS, KEY_REPLACEMENT_DICT
from wlmmuq.kappatng import OPENINGANGLE
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER, MULTFACT_SUP_STEP_SIZE
from wlmmuq.models.deepinv.pnpmcalens import \
    NITER_PER_STEP_G, NITER_PER_STEP_NG, STARLET_DETECTION_THRESHOLD

NINPIMGS = 100 # Number of input images before cropping
NIMGS_TEST = 512 # Images extracted from the 57 first original files (copped dataset)
NIMGS_CALIB = 1024 # Images extracted from the 43 remaining original files (augmented dataset)
EPOCH = 100 # Epoch of the trained models to load
IMGSIZE = 384
BATCH_SIZE = 32
NIMGS_SAVE = 16
KEYS_MODEL = ['model_size', 'args_wienerinit']

WHICH_GAUSSIAN_EXTRACTOR = "wiener" # "wiener" or "mcalens"
MODE_PNPMASS = "regular" # "regular", "residual", or "pnpmcalens"
NITER_PNPMASS = 8
CONFIDENCE_UQ = 2 # 2-sigma confidence

INPAINTING_WIENER = False
INPAINTING_PNPMASS = False
INPAINTING_DEEPMASS = True

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


def get_output_type(order2=False):
    if not order2:
        output_type = "pe" # Point estimate
    else:
        output_type = "var" # Variance
    return output_type


def _load_trained_model(
        checkpoint_dir, arch, timestamp,
        epoch=EPOCH, imgsize=IMGSIZE, order2=False,
        key_replacement_dict=KEY_REPLACEMENT_DICT,
        device="cpu", verbose=False, **kwargs
):
    cnn_class, _ = wlnn.MODEL_CLASSES[arch]
    if timestamp is None:
        path_to_checkpoint = checkpoint_dir
    else:
        output_type = get_output_type(order2)
        path_to_checkpoint = os.path.join(
            checkpoint_dir, output_type, timestamp, f"ckp_{epoch}.pth.tar"
        )
    if not order2:
        kwargs.update(meancentering=True, onlypositive=False)
    else:
        kwargs.update(meancentering=False, onlypositive=True)

    model = cnn_class(map_size=imgsize, **kwargs)
    checkpoint = torch.load(path_to_checkpoint)
    state_dict = checkpoint['state_dict']
    if key_replacement_dict is not None:
        for old_key, new_key in key_replacement_dict.items():
            if old_key in state_dict:
                if verbose:
                    print(f"Replacing key '{old_key}' with '{new_key}'")
                state_dict[new_key] = state_dict.pop(old_key)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval().to(device)
    if verbose:
        model.summary()

    return model


def load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=EPOCH,
        load_model_uq=False, checkpoint_dir_uq=None,
        arch_uq=None, timestamp_uq=None, epoch_uq=None,
        imgsize=IMGSIZE, device="cpu", verbose=False, **kwargs
):
    if arch is None:
        raise ValueError(
            "Model architecture must be provided with -a or --arch"
        )
    kwargs_model = {k: kwargs.pop(k) for k in KEYS_MODEL if k in kwargs}
    model = _load_trained_model(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        imgsize=imgsize, order2=False, device=device,
        verbose=verbose, **kwargs_model
    )
    if load_model_uq:
        if checkpoint_dir_uq is None:
            checkpoint_dir_uq = checkpoint_dir
        else:
            warnings.warn(
                f"The model used for UQ ({checkpoint_dir_uq}) is not the same as "
                f"the one used for the point estimate ({checkpoint_dir})"
            )
        if arch_uq is None:
            arch_uq = arch
            kwargs_model_uq = kwargs_model.copy()
            verbose_uq = False
        else:
            kwargs_model_uq = {}
            for k in KEYS_MODEL:
                kuq = f"{k}_uq"
                if kuq in kwargs:
                    kwargs_model_uq.update({k: kwargs.pop(kuq)})
            verbose_uq = verbose
        if epoch_uq is None:
            epoch_uq = epoch

        model_uq = _load_trained_model(
            checkpoint_dir_uq, arch_uq, timestamp_uq,
            epoch=epoch_uq, imgsize=imgsize, order2=True,
            device=device, verbose=verbose_uq,
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
        step_size=None, multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    step_size, param_mahalanobis = \
            get_step_size_param_mahalanobis(
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_sup_step_size=multfact_sup_step_size,
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
        step_size=None, multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    data_fidelity, params_algo = _get_datafidelity_params(
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_sup_step_size=multfact_sup_step_size,
        device=device, verbose=verbose
    )
    powerspectrum = torch.load(path_to_ps)
    prior = dinv.optim.PnP(wlpnp.ProximalWiener(powerspectrum))

    return data_fidelity, prior, params_algo


def get_datafidelity_prior_params_nongaussian(
        denoiser, denoiser_uq=None,
        white_noise=False, std_noise=None, physics=None,
        step_size=None, multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    data_fidelity, params_algo = _get_datafidelity_params(
        white_noise=white_noise, noise_whitening=True,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_sup_step_size=multfact_sup_step_size,
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
        step_size=None, multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE,
        niter=NITER_WIENER, device="cpu", verbose=False
):
    if verbose:
        print("Get optimizer for iterative Wiener filtering")
    data_fidelity, prior, params_algo = get_datafidelity_prior_params_gaussian(
        path_to_ps=path_to_ps,
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_sup_step_size=multfact_sup_step_size,
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
        multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE,
        niter=NITER_WIENER,
        starlet_detection_threshold=STARLET_DETECTION_THRESHOLD,
        mcalens_update_ng_first=False,
        device="cpu", verbose=False
):
    data_fidelity_g, prior_g, params_algo_g = get_datafidelity_prior_params_gaussian(
        path_to_ps=path_to_ps,
        white_noise=white_noise, noise_whitening=noise_whitening_wiener,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_sup_step_size=multfact_sup_step_size,
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
            step_size=step_size_ng, multfact_sup_step_size=multfact_sup_step_size,
            device=device, verbose=verbose
        )
        extractor = wlpnpmcalens.optim_builder_mcalens(
            iteration_g="PGD", iteration_ng="PGD",
            params_algo_g=params_algo_g.copy(), params_algo_ng=params_algo_ng.copy(),
            data_fidelity_g=data_fidelity_g, data_fidelity_ng=data_fidelity_ng,
            prior_g=prior_g, prior_ng=prior_ng,
            early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
            update_ng_first=mcalens_update_ng_first,
            discard_ng=True, verbose=verbose
        ).to(device)

    else:
        raise ValueError(
            f"Invalid extractor '{which}'. "
            "Supported extractors are 'wiener' and 'mcalens'."
        )

    return extractor, callback


def get_pnpmass(
        denoiser, denoiser_uq, imgsize=IMGSIZE,
        std_noise=None, mask=None, physics=None,
        step_size=None, niter=NITER_PNPMASS,
        multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE, mode="regular",
        which_gaussian_extractor=WHICH_GAUSSIAN_EXTRACTOR,
        update_ng_first=False,
        switch_mode_for_uq=False,
        path_to_ps=PATH_TO_PS,
        noise_whitening_wiener=False,
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
        step_size=step_size, multfact_sup_step_size=multfact_sup_step_size,
        device=device, verbose=verbose
    )
    if step_size is None or step_size <= 0:
        step_size_filename = "auto"
        step_size = params_algo["stepsize"]
    else:
        step_size_filename = f"{step_size:.3f}"
    metric_dict={"rmse": wlpnp.RMSE(mask=mask)} # RMSE computed within the mask

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
                multfact_sup_step_size=multfact_sup_step_size,
                niter=niter_wiener,
                starlet_detection_threshold=starlet_detection_threshold,
                mcalens_update_ng_first=False,
                device="cpu", verbose=False
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
            metric_dict=metric_dict, verbose=verbose,
            gaussian_extractor=gaussian_extractor
        ).to(device)

    elif mode == "pnpmcalens":
        if verbose:
            print("Instantiate PnPMCALens")

        # Note: the step size for the Gaussian component is computed automatically
        data_fidelity_g, prior_g, params_algo_g = get_datafidelity_prior_params_gaussian(
            path_to_ps=path_to_ps,
            white_noise=False, noise_whitening=noise_whitening_wiener,
            std_noise=std_noise, physics=physics,
            step_size=None, multfact_sup_step_size=multfact_sup_step_size,
            device=device, verbose=verbose
        )
        pnpmass = wlpnpmcalens.optim_builder_mcalens(
            iteration_g="PGD", iteration_ng="PGD",
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            params_algo_g=params_algo_g.copy(), params_algo_ng=params_algo.copy(),
            data_fidelity_g=data_fidelity_g, data_fidelity_ng=data_fidelity,
            prior_g=prior_g, prior_ng=prior,
            early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
            metric_dict=metric_dict, update_ng_first=update_ng_first, verbose=True
        ).to(device)
        callback_gaussian_extractor = None

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. "
            "Supported modes are 'regular', 'residual', and 'pnpmcalens'."
        )

    if prior_uq is not None:
        if not switch_mode_for_uq:
            gaussian_extractor_uq = gaussian_extractor
        else:
            gaussian_extractor_uq = None
        pnpmass_uq = wlpnp.optim_builder(
            iteration="PGD", params_algo=params_algo.copy(),
            data_fidelity=data_fidelity, prior=prior_uq,
            early_stop=False, max_iter=1, custom_init=wlpnp.ManualInit(),
            metric_dict=metric_dict, verbose=verbose,
            gaussian_extractor=gaussian_extractor_uq
        ).to(device)
    else:
        pnpmass_uq = None

    return pnpmass, pnpmass_uq, step_size, \
        step_size_filename, callback_gaussian_extractor


# TODO: Merge the 3 functions below into one single function
def run_wiener_batch(
        wiener: wlpnp.BaseOptim, physics: wlpnp.MassMapping,
        dataloader, confidence_uq=CONFIDENCE_UQ,
        mask=None, device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_wiener = []
    listof_var_wiener = [] # Zero-valued tensors
    listof_rmse = []

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    for kappa_true, gamma_noisy in pbar:
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_wiener = wiener(gamma_noisy, physics)
            var_wiener = torch.zeros(kappa_true.shape, device=device)
            rmse = wlutils.rmse(kappa_wiener, kappa_true, mask=mask)

            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_kappa_wiener.append(kappa_wiener) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_var_wiener.append(var_wiener) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_rmse.append(rmse) # Shape = (batch_size,)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_wiener = torch.cat(listof_kappa_wiener, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var_wiener = torch.cat(listof_var_wiener, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs,)

    res_wiener = confidence_uq * var_wiener**0.5

    out = {
        "kappa_true": kappa_true,
        "kappa_wiener": kappa_wiener,
        "var_wiener": var_wiener,
        "res_wiener": res_wiener,
        "rmse": rmse
    }
    return out


def run_pnpmass_batch(
        pnpmass: wlpnp.BaseOptim, pnpmass_uq: wlpnp.BaseOptim,
        physics: wlpnp.MassMapping,
        dataloader, step_size, niter, confidence_uq=CONFIDENCE_UQ,
        callbacks: wlcallbacks.CallbackList | None = None,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pnpmass = []
    listof_var_pnpmass = []
    listof_rmse_iter = []

    if callbacks is None:
        callbacks = wlcallbacks.BaseCallback()

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    pbar.set_description(f"Step size = {step_size:.2e}, Nb iterations = {niter}")
    for i, (kappa_true, gamma_noisy) in enumerate(pbar):
        callbacks.on_batch_begin(i)
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_pnpmass, metrics = pnpmass(
                gamma_noisy, physics, x_gt=kappa_true, compute_metrics=True
            )
            if isinstance(pnpmass, wlpnpmcalens.BaseMCALens):
                kappa_pnpmass_g, kappa_pnpmass_ng = \
                    wlpnpmcalens.get_tensor_components(kappa_pnpmass)
                kappa_pnpmass = wlpnpmcalens.add_tensor_components(kappa_pnpmass)
            else:
                kappa_pnpmass_g = kappa_pnpmass_ng = None
            if pnpmass_uq is not None:
                # Initialize the UQ iteration with the predicted kappa
                pnpmass_uq.custom_init.X_init = (kappa_pnpmass,)
                var_pnpmass = pnpmass_uq(
                    gamma_noisy, physics, compute_metrics=False
                )
            else:
                var_pnpmass = torch.zeros(kappa_true.shape, device=device)

            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_kappa_pnpmass.append(kappa_pnpmass) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_var_pnpmass.append(var_pnpmass) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_rmse_iter.append(metrics["rmse"]) # Shape = (batch_size, niter)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pnpmass = torch.cat(listof_kappa_pnpmass, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var_pnpmass = torch.cat(listof_var_pnpmass, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    rmse_iter = torch.cat(listof_rmse_iter, dim=0) # Shape = (nimgs, niter)

    res_pnpmass = confidence_uq * var_pnpmass**0.5

    out = {
        "kappa_true": kappa_true,
        "kappa_pnpmass": kappa_pnpmass,
        "kappa_pnpmass_g": kappa_pnpmass_g,
        "kappa_pnpmass_ng": kappa_pnpmass_ng,
        "var_pnpmass": var_pnpmass,
        "res_pnpmass": res_pnpmass,
        "rmse_iter": rmse_iter
    }
    return out


def run_deepmass_batch(
        deepmass: wlpnp.BaseOptim, deepmass_uq: wlpnp.BaseOptim,
        dataloader, confidence_uq=CONFIDENCE_UQ,
        mask=None, device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_deepmass = []
    listof_var_deepmass = []
    listof_rmse = []

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    for kappa_true, gamma_noisy in pbar:
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            kappa_deepmass = deepmass(gamma_noisy)
            if deepmass_uq is not None:
                var_deepmass = deepmass_uq(gamma_noisy)
            else:
                var_deepmass = torch.zeros(kappa_true.shape, device=device)
            rmse = wlutils.rmse(kappa_deepmass, kappa_true, mask=mask)

            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_kappa_deepmass.append(kappa_deepmass) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_var_deepmass.append(var_deepmass) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_rmse.append(rmse) # Shape = (batch_size,)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_deepmass = torch.cat(listof_kappa_deepmass, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var_deepmass = torch.cat(listof_var_deepmass, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs,)

    res_deepmass = confidence_uq * var_deepmass**0.5

    out = {
        "kappa_true": kappa_true,
        "kappa_deepmass": kappa_deepmass,
        "var_deepmass": var_deepmass,
        "res_deepmass": res_deepmass,
        "rmse": rmse
    }
    return out


def get_inference_time(beg_time, verbose=False):
    inference_time = time.time() - beg_time
    if verbose:
        print(f"Total inference time: {inference_time:.2f} seconds")
    return inference_time


def get_args_wienerinit(
        std_noise, mask, path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        step_size=None, multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE,
        niter=NITER_WIENER, device="cpu", verbose=False
):
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)
    powerspectrum = torch.load(path_to_ps)
    step_size, _ = get_step_size_param_mahalanobis(
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        step_size=step_size, multfact_sup_step_size=multfact_sup_step_size,
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
        step_size=None, multfact_sup_step_size=MULTFACT_SUP_STEP_SIZE,
        device="cpu", verbose=False
):
    if not white_noise:
        param_mahalanobis = wlutils.get_g_param(std_noise, noise_whitening)
        if step_size is None or step_size <= 0:
            step_size = wlutils.get_sup_step_size(
                param_mahalanobis=param_mahalanobis,
                physics=physics, device=device
            )
            step_size *= multfact_sup_step_size
            if verbose:
                print(f"Step size computed using power iteration = {step_size:.2e}")
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


def load_cqr(
        path_to_cqr, confidence_uq, imgsize,
        device="cpu", verbose=False
):
    if path_to_cqr is not None:
        if verbose:
            print("Load calibration function")
        alpha = wlutils.get_alpha_from_confidence(confidence_uq)
        cqr = wlcqr.AddCQR(alpha, map_size=imgsize)
        checkpoint_cqr = torch.load(path_to_cqr)
        assert confidence_uq == checkpoint_cqr["confidence_uq"]
        nimgs_calib = checkpoint_cqr["nimgs_calib"]
        cqr.load_state_dict(checkpoint_cqr["state_dict"])
        cqr.eval().to(device)
    else:
        nimgs_calib = None
        cqr = None

    return nimgs_calib, cqr


def get_cqr(
        kappa_pred, res, kappa_true,
        imgsize, confidence_uq, device="cpu", verbose=False
):
    if verbose:
        print("Instantiate CQR model and compute the calibration parameters")
    alpha = wlutils.get_alpha_from_confidence(confidence_uq)
    cqr = wlcqr.AddCQR(alpha, map_size=imgsize).to(device)
    cqr.calibrate(kappa_pred, res, kappa_true)

    return cqr


def get_calibrated_residuals(cqr, res, verbose=False):

    if cqr is not None:
        beg_time = time.time()
        if verbose:
            print("Calibrate residuals with CQR")
        res_cqr = cqr(res)
        cqr_time = time.time() - beg_time
        if verbose:
            print(f"Calibration time: {cqr_time:.2f} seconds")
    else:
        res_cqr = None
        cqr_time = None

    return res_cqr, cqr_time


def get_metrics(
        kappa_pred, res, kappa_true, res_cqr=None,
        mask=None, verbose=False
):
    beg_time = time.time()
    err, predinterv, _, _ = wlutils.get_metrics(
        kappa_pred, res, kappa_true, mask=mask
    )
    if res_cqr is not None:
        err_cqr, predinterv_cqr, _, _ = wlutils.get_metrics(
            kappa_pred, res_cqr, kappa_true, mask=mask
        )
    else:
        err_cqr = None
        predinterv_cqr = None
    metrics_time = time.time() - beg_time
    if verbose:
        print(f"Metrics computation time: {metrics_time:.2f} seconds")

    return err, predinterv, err_cqr, predinterv_cqr, metrics_time


def save_results(
        out_dict, path_to_output, now,
        step_size=None, load_model_uq=False,
        confidence_uq=None, verbose=False
):
    if step_size is not None:
        path_to_output = f"{path_to_output}_step-size_{step_size}"
    if load_model_uq:
        path_to_output = (
            f"{path_to_output}_{confidence_uq}-sigma"
        )
    path_to_output = f"{path_to_output}_{now}.pt"
    if verbose:
        print(f"Save results to {path_to_output}")

    torch.save(out_dict, path_to_output)


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


def add_arguments_cqr(parser):

    parser.add_argument(
        "-cqr", "--path-to-cqr", type=str, default=None,
        help=(
            "Path to the CQR checkpoint (optional). "
            "If provided, the residuals will be calibrated with CQR"
        )
    )


def add_arguments_model(parser):

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


def add_arguments_checkpoint(parser):

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
        "--checkpoint-dir-uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Checkpoint directory for the order-2 moment network, "
            "if different from `checkpoint_dir`."
        )
    )
    parser.add_argument(
        "-t0", "--timestamp_uq", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Timestamp of the model to load. "
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
        "-f", "--min-idx-filename-ori",
        type=int, default=argparse.SUPPRESS,
        help=(
            "Filter images by filenames with indices equal or larger than this value. "
            "Default is None."
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


def add_arguments_test_dataset(parser, batch_size):

    parser.add_argument(
        "--nimgs-test", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of test images. "
            f"Default = {NIMGS_TEST}"
        )
    )
    add_arguments_dataset(parser, batch_size)


def add_arguments_calib_dataset(parser, batch_size):

    parser.add_argument(
        "--nimgs-calib", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of calibration images. "
            f"Default = {NIMGS_CALIB}"
        )
    )
    add_arguments_dataset(parser, batch_size)


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
            "Only used if `--mode` is set to 'pnpmcalens'."
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
        "--switch-mode-for-uq", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "If this argument is set and `--mode` is set to 'residual', "
            "then UQ will not be computed on the residuals. This is useful when "
            "the model used for UQ is different from the one used for the "
            "point estimate, and is not trained on the residuals."
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
