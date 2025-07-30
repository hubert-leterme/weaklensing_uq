import os
import warnings
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

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK, PATH_TO_PS
from wlmmuq.kappatng import OPENINGANGLE
from wlmmuq.data import NUM_WORKERS
from wlmmuq.models.torch import NITER_WIENER, MULTFACT_STEP_SIZE

NINPIMGS = 100 # Number of input images before cropping
NIMGS_TEST = 512 # Images extracted from the 57 first original files (copped dataset)
NIMGS_CALIB = 1024 # Images extracted from the 43 remaining original files (augmented dataset)
EPOCH = 100 # Epoch of the trained models to load
IMGSIZE = 384
BATCH_SIZE = 32
KEYS_MODEL = ['model_size', 'args_wienerinit']

NITER_PNPMASS = 8
CONFIDENCE_UQ = 2 # 2-sigma confidence

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
        cosmos_include_faint=False, convert_to_torch_tensor=False,
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

        if convert_to_torch_tensor:
            mask = torch.tensor(mask, dtype=bool)
            std_noise = torch.tensor(std_noise, dtype=torch.float32)

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
        verbose=False, **kwargs
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
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if verbose:
        model.summary()

    return model


def load_trained_models(
        checkpoint_dir, arch, timestamp, epoch=EPOCH,
        load_model_uq=False, checkpoint_dir_uq=None,
        arch_uq=None, timestamp_uq=None, epoch_uq=None,
        imgsize=IMGSIZE, verbose=False, **kwargs
):
    if arch is None:
        raise ValueError(
            "Model architecture must be provided with -a or --arch"
        )
    kwargs_model = {k: kwargs.pop(k) for k in KEYS_MODEL if k in kwargs}
    model = _load_trained_model(
        checkpoint_dir, arch, timestamp, epoch=epoch,
        imgsize=imgsize, order2=False,
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
            verbose=verbose_uq, **kwargs_model_uq
        )
    else:
        model_uq = None

    return model, model_uq


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


def get_pnpmass_modules(std_noise, mask, denoiser, denoiser_uq=None):

    # Instantiate data fidelity, prior and metrics
    data_fidelity = wlpnp.Mahalanobis(
        param_vector=std_noise
    ) # Noise-whitening data fidelity
    prior = dinv.optim.prior.PnP(denoiser)
    if denoiser_uq is not None:
        prior_uq = dinv.optim.prior.PnP(denoiser_uq)
    else:
        prior_uq = None
    rmse = wlpnp.RMSE(mask=mask) # RMSE computed within the mask

    return data_fidelity, prior, prior_uq, rmse


def get_wiener(
        path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        multfact_step_size=MULTFACT_STEP_SIZE, niter=NITER_WIENER,
        device="cpu", verbose=False
):
    if verbose:
        print("Get optimizer for iterative Wiener filtering")
    powerspectrum, step_size, param_mahalanobis = \
            get_powerspectrum_step_size_wienerinit(
        path_to_ps=path_to_ps,
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        multfact_step_size=multfact_step_size,
        device=device, verbose=verbose
    )
    if not white_noise:
        data_fidelity = wlpnp.Mahalanobis(param_vector=param_mahalanobis)
        g_param = step_size
    else:
        # Regular L2 data fidelity with unitary variance
        data_fidelity = dinv.optim.data_fidelity.L2()
        g_param = None # To be updated for each new noise realization
    prior = dinv.optim.PnP(wlpnp.ProximalWiener(powerspectrum))
    out = wlpnp.optim_builder(
        iteration="PGD", prior=prior,
        data_fidelity=data_fidelity,
        early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
        params_algo={"stepsize": step_size, "g_param": g_param},
    ).to(device)

    return out


def get_pnpmass(
        denoiser, denoiser_uq,
        std_noise=None, mask=None, physics=None,
        step_size=None, niter=NITER_PNPMASS,
        multfact_step_size=MULTFACT_STEP_SIZE, nongaussian=False,
        switch_mode_for_uq=False, wiener=None, device="cpu"
):
    data_fidelity, prior, prior_uq, rmse = get_pnpmass_modules(
        std_noise, mask, denoiser, denoiser_uq
    )
    if step_size is None or step_size <= 0:
        upperbound_step_size = wlutils.get_sup_step_size(
            param_mahalanobis=std_noise, # Noise-whitening data fidelity term
            physics=physics, device=device
        )
        step_size = multfact_step_size * upperbound_step_size

    wiener_estimate = _get_wiener_estimate(nongaussian, wiener)
    pnpmass = wlpnp.optim_builder(
        iteration="PGD", prior=prior,
        data_fidelity=data_fidelity,
        early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
        metric_dict={"rmse": rmse}, verbose=True,
        params_algo={"stepsize": step_size, "g_param": step_size},
        wiener_estimate=wiener_estimate,
    ).to(device)

    if prior_uq is not None:
        if not switch_mode_for_uq:
            nongaussian_uq = nongaussian
        else:
            nongaussian_uq = not nongaussian
        wiener_estimate_uq = _get_wiener_estimate(nongaussian_uq, wiener)
        pnpmass_uq = wlpnp.optim_builder(
            iteration="PGD", prior=prior_uq,
            data_fidelity=data_fidelity,
            early_stop=False, max_iter=1, custom_init=wlpnp.ManualInit(),
            verbose=True,
            params_algo={"stepsize": step_size, "g_param": step_size},
            wiener_estimate=wiener_estimate_uq,
        ).to(device)
    else:
        pnpmass_uq = None

    return pnpmass, pnpmass_uq, step_size


def _get_wiener_estimate(nongaussian, wiener):
    if nongaussian:
        if wiener is None:
            raise ValueError("Missing model for iterative Wiener filtering.")
        wiener_estimate = wiener
    else:
        wiener_estimate = None
    return wiener_estimate


def run_wiener_pnpmass_batch(
        wiener: wlpnp.BaseOptim, pnpmass: wlpnp.BaseOptim,
        pnpmass_uq: wlpnp.BaseOptim, physics: wlpnp.MassMapping,
        dataloader, step_size, niter, confidence_uq=CONFIDENCE_UQ,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_wiener = []
    listof_kappa_pnpmass = []
    listof_var_pnpmass = []
    listof_rmse_iter = []

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    pbar.set_description(f"Step size = {step_size:.2e}, Nb iterations = {niter}")
    for kappa_true, gamma_noisy in pbar:
        kappa_true = kappa_true.to(device)
        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            if wiener is not None:
                kappa_wiener = wiener(gamma_noisy, physics, compute_metrics=False)
            else:
                kappa_wiener = None

            kappa_pnpmass, metrics = pnpmass(
                gamma_noisy, physics, x_gt=kappa_true, compute_metrics=True
            )
            if pnpmass_uq is not None:
                # Initialize the UQ iteration with the predicted kappa
                pnpmass_uq.custom_init.X_init = (kappa_pnpmass,)
                var_pnpmass = pnpmass_uq(
                    gamma_noisy, physics, compute_metrics=False
                )
            else:
                var_pnpmass = torch.zeros(kappa_true.shape, device=device)

            listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
            if wiener is not None:
                listof_kappa_wiener.append(kappa_wiener) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_kappa_pnpmass.append(kappa_pnpmass) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_var_pnpmass.append(var_pnpmass) # Shape = (batch_size, 1, imgsize, imgsize)
            listof_rmse_iter.append(metrics["rmse"]) # Shape = (batch_size, niter)

    kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    if wiener is not None:
        kappa_wiener = torch.cat(listof_kappa_wiener, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    else:
        kappa_wiener = None
    kappa_pnpmass = torch.cat(listof_kappa_pnpmass, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var_pnpmass = torch.cat(listof_var_pnpmass, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    rmse_iter = torch.cat(listof_rmse_iter, dim=0) # Shape = (nimgs, niter)

    res_pnpmass = confidence_uq * var_pnpmass**0.5

    return kappa_true, kappa_wiener, kappa_pnpmass, var_pnpmass, res_pnpmass, rmse_iter


def get_args_wienerinit(
        std_noise, mask, path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        multfact_step_size=MULTFACT_STEP_SIZE, niter=NITER_WIENER,
        device="cpu", verbose=False
):
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)
    powerspectrum, step_size, _ = \
            get_powerspectrum_step_size_wienerinit(
        path_to_ps=path_to_ps,
        white_noise=white_noise, noise_whitening=noise_whitening,
        std_noise=std_noise, physics=physics,
        multfact_step_size=multfact_step_size,
        device=device, verbose=verbose
    ) # Bayesian Wiener filtering
    args_wienerinit = dict(
        step_size=step_size, powerspectrum=powerspectrum,
        std_noise=std_noise, mask=mask, niter=niter,
        noise_whitening=noise_whitening
    )
    return args_wienerinit


def get_powerspectrum_step_size_wienerinit(
        path_to_ps=PATH_TO_PS,
        white_noise=False, noise_whitening=False,
        std_noise=None, physics=None,
        multfact_step_size=MULTFACT_STEP_SIZE,
        device="cpu", verbose=False
):
    if verbose:
        print("Get Wiener initialization parameters")
    powerspectrum = torch.load(path_to_ps)
    if not white_noise:
        param_mahalanobis = wlutils.get_g_param(std_noise, noise_whitening)
        step_size = wlutils.get_sup_step_size(
            param_mahalanobis=param_mahalanobis,
            physics=physics, device=device
        )
        step_size *= multfact_step_size
    else:
        # The standard MSE is used as data fidelity
        # The parameter `g_param` for the proximal operator must be updated accordingly
        step_size = 1
        param_mahalanobis = None

    return powerspectrum, step_size, param_mahalanobis


def save_output_pnpmass(
        out_dict, path_to_output, step_size, now,
        load_model_uq=False, confidence_uq=None,
        verbose=False
):
    path_to_output_completed = (
        f"{path_to_output}_step-size_{step_size:.3f}"
    )
    if load_model_uq:
        path_to_output_completed = (
            f"{path_to_output_completed}_{confidence_uq}-sigma"
        )
    path_to_output_completed = f"{path_to_output_completed}_{now}.pt"
    if verbose:
        print(f"Save results to {path_to_output_completed}")

    torch.save(out_dict, path_to_output_completed)


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


def add_arguments_wiener(parser):

    parser.add_argument(
        "-ps", "--path-to-ps", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the power spectrum file used for Wiener initialization. "
            f"Default = {PATH_TO_PS}"
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


def add_arguments_nongaussian(parser):

    parser.add_argument(
        "-ng", "--nongaussian", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Split the Gaussian and non-Gaussian parts of the convergence maps."
        )
    )
    parser.add_argument(
        "--switch-mode-for-uq", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "If both this argument and `--nongaussian` are set, then "
            "UQ will not be computed on the residuals. This is useful when "
            "the model used for UQ is different from the one used for the "
            "point estimate, and is not trained on the residuals."
        )
    )
    add_arguments_wiener(parser)


def add_arguments_output(parser, output_filename):

    parser.add_argument(
        "-o", "--output-filename", type=str,
        help=(
            "Output filename (without extension). "
            f"Default = {output_filename}"
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
