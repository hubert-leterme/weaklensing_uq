import argparse
import time
import tqdm
import torch
import deepinv as dinv

import wlmmuq.models.deepinv.iterativemm as wlpnp
import wlmmuq.utils as wlutils

from wlmmuq.data import NUM_WORKERS

import _commons
from _commons import NIMGS_TEST, IMGSIZE, BATCH_SIZE, MULTFACT_STEP_SIZE

NITER = 8

def main(
        path_to_test_dataset: str, path_to_checkpoint: str, path_to_output: str,
        arch: str=None, step_size: float | list[float]=None,
        multfact_step_size: float=MULTFACT_STEP_SIZE, niter: int=NITER,
        cosmos_include_faint: bool=False,
        nimgs_test: int=NIMGS_TEST,
        imgsize: int=IMGSIZE, batch_size: int=BATCH_SIZE, num_workers: int=NUM_WORKERS,
        seed: int=None, verbose: bool=False, **kwargs
):
    _commons.set_seed(seed)

    now = wlutils.get_timestamp()
    device = _commons.get_device(verbose=verbose)
    if verbose:
        print(f"Number of workers: {num_workers}")

    beg_time = time.time()

    # Load noise standard deviation and mask
    std_noise, mask = _commons.get_stdnoise_mask(
        imgsize, cosmos_include_faint=cosmos_include_faint,
        convert_to_torch_tensor=True, inpainting=False,
        verbose=verbose
    )

    # Load test set
    test_dataloader = _commons.get_dataloader_massmapping(
        path_to_test_dataset, nimgs_test, imgsize, batch_size,
        num_workers, std_noise, mask
    )

    # Load trained denoiser
    denoiser = _commons.load_trained_model(
        path_to_checkpoint, arch, imgsize, verbose=verbose, **kwargs
    )

    # Instantiate data fidelity, prior and metrics
    data_fidelity = wlpnp.Mahalanobis(
        sigma=torch.sqrt(std_noise)
    ) # torch.sqrt is on purpose ("noise-whitening" data fidelity)
    prior = dinv.optim.prior.PnP(denoiser)
    rmse = wlpnp.RMSE(mask=mask) # RMSE computed within the mask

    # Instantiate physics (forward model)
    physics = wlpnp.MassMapping(sigma=std_noise, mask=mask).to(device)

    # Get step size
    if step_size is None:
        upperbound_step_size = wlutils.get_sup_step_size(
            std_noise**0.5, # Sqrt of noise stdev because we do not consider the negative log-likelihood
            mask=mask
        )
        step_size = multfact_step_size * upperbound_step_size
    if not isinstance(step_size, list):
        step_size = [step_size]

    for tau in step_size:
        # Instantiate the PnP model
        pnpmass = wlpnp.optim_builder(
            iteration="PGD", prior=prior,
            data_fidelity=data_fidelity,
            early_stop=False, max_iter=niter, custom_init=wlpnp.zero_init,
            metric_dict={"rmse": rmse}, verbose=True,
            params_algo={"stepsize": tau, "g_param": tau},
        ).to(device) # The noise stdev for the denoiser is equal to the step size

        # Run PnPMass for each batch
        listof_rmse_iter = []
        pbar = tqdm.tqdm(test_dataloader, disable=not verbose)
        pbar.set_description(f"Step size = {tau:.2e}, Nb iterations = {niter}")
        for kappa_true, gamma_noisy in pbar:
            kappa_true = kappa_true.to(device)
            gamma_noisy = gamma_noisy.to(device)
            with torch.no_grad():
                kappa_pnpmass, metrics = pnpmass(
                    gamma_noisy, physics, x_gt=kappa_true, compute_metrics=True
                )
                listof_rmse_iter.append(metrics["rmse"]) # Shape = (batch_size, niter)
        rmse_iter = torch.cat(listof_rmse_iter, dim=0) # Shape = (nimgs, niter)

        inference_time = time.time() - beg_time

        out_dict = {
            "rmse_iter": rmse_iter.cpu(),
            "inference_time": inference_time,
            "step_size": tau,
            "arch": arch,
            "niter": niter,
            "nimgs_test": nimgs_test,
            "imgsize": imgsize
        }
        path_to_output_completed = f"{path_to_output}_step-size_{tau:.3f}_{now}.pt"
        torch.save(out_dict, path_to_output_completed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_test_dataset", type=str,
        help="Path to the test set (HDF5 file)"
    )
    parser.add_argument(
        "path_to_checkpoint", type=str,
        help="Path to the checkpoint containing the model's state dict (.pth.tar)"
    )
    parser.add_argument(
        "path_to_output", type=str,
        help="Path to the output file (without extension)"
    )
    _commons.add_arguments_model(parser)
    parser.add_argument(
        "-tau", "--step-size", type=float, nargs='+',
        default=argparse.SUPPRESS,
        help=(
            "Step size for the PnPMass algorithm. Several values can be provided. "
            f"Default = {MULTFACT_STEP_SIZE:.2f} * upper_bound, "
            "where upper_bound is computed from the noise standard deviation "
            "and the mask, using the power iteration method"
        )
    )
    parser.add_argument(
        "-i", "--niter", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of iterations for PnPMass. "
            f"Default = {NITER}"
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
    _commons.add_arguments_dataset(parser, batch_size=BATCH_SIZE)
    _commons.add_arguments_seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
