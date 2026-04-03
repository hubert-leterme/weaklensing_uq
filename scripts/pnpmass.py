import argparse
import time
import typing
import tqdm
import torch

import wlmmuq

from wlmmuq.datasets import NUM_WORKERS

import _commons
import _add_arguments

METHOD_NAME = "pnpmass"
OUTPUT_PREFIX = None

def main(
        path_to_real_shearmap: str | None = wlmmuq.PATH_TO_REAL_SHEARMAP,
        path_to_test_dataset: str | None = wlmmuq.PATH_TO_TEST_DATASET,
        path_to_calib_dataset: str | None = wlmmuq.PATH_TO_CALIB_DATASET,
        train_val_dataset_name: str | None = wlmmuq.TRAIN_VAL_DATASET_NAME,
        test_dataset_name: str | None = wlmmuq.TEST_DATASET_NAME,
        real_shearmap_name: str | None = wlmmuq.REAL_SHEARMAP_NAME,
        test_on_real_data: bool = False, run_both: bool = False,
        model_dir: str | None = wlmmuq.MODEL_DIR,
        model_name: str | None = None, model_name_uq: str | None = None,
        output_dir: str = wlmmuq.RESULTS_DIR,
        method_name: str = METHOD_NAME,
        path_to_std_noise: str = wlmmuq.PATH_TO_STD_NOISE,
        path_to_mask: str = wlmmuq.PATH_TO_MASK,
        path_to_ps: str = wlmmuq.PATH_TO_PS,
        arch: str | None = None, timestamp: str | None = None, epoch: int = _commons.EPOCH,
        model_specs: str | None = None,
        load_model_uq: bool = False,
        arch_uq: str | None = None, timestamp_uq: str | None = None, epoch_uq: int | None = None,
        model_specs_uq: str | None = None,
        step_size: float | None | list[float | None] = None,
        multfact_step_size: float | None | list[float | None] = None,
        niter: int = _commons.NITER_PNPMASS,
        bin_data_from_cosmos: bool = False,
        cosmos_include_faint: bool = False,
        max_z: float | None = _commons.MAX_Z,
        use_zbins: bool = False,
        path_to_zbins: str | None = wlmmuq.PATH_TO_ZBINS,
        idx_zbins: list[int] = _commons.IDX_ZBINS,
        resolution: float = _commons.RESOLUTION,
        inpainting: bool = _commons.INPAINTING_PNPMASS,
        nimgs_test: int = _commons.NIMGS_TEST,
        cqr: bool = False,
        nimgs_calib: int = _commons.NIMGS_CALIB,
        min_idx_filename_ori_calib: str | int = _commons.MIN_IDX_FILENAME_ORI_CALIB,
        imgsize: int = _commons.IMGSIZE, batch_size: int = _commons.BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        mode: str = _commons.MODE_PNPMASS,
        which_gaussian_extractor: str = _commons.WHICH_GAUSSIAN_EXTRACTOR_PNPMASS,
        niter_wiener: int = _commons.NITER_WIENER,
        starlet_detection_threshold: float = _commons.STARLET_DETECTION_THRESHOLD,
        eps_sup_step_size: float = _commons.EPS_SUP_STEP_SIZE,
        niter_per_step_g: int = _commons.NITER_PER_STEP_G,
        niter_per_step_ng: int = _commons.NITER_PER_STEP_NG,
        mode_cqr: str | list[str] = _commons.MODE_CQR,
        scaling_factor_chisqcqr: float | None | list[float | None] = None,
        confidence_uq: int | float = _commons.CONFIDENCE_UQ,
        hyperparam_precalib: list[float] | None = None,
        find_optimal_hyperparam_precalib: bool = False,
        save_tensors: bool = False, nimgs_save: int = _commons.NIMGS_SAVE,
        output_prefix: str | None = OUTPUT_PREFIX,
        seed: int | None = None, verbose: bool = False, **kwargs
):
    _commons.set_seed(seed)

    assert model_name is not None
    checkpoint_dir, checkpoint_dir_uq = _commons.get_checkpoint_dirs(
        model_dir,
        train_val_dataset_name=train_val_dataset_name,
        model_name=model_name,
        model_name_uq=model_name_uq
    )

    # When run_both is True we create two output dirs (simulated and real)
    if run_both:
        output_dir_sim = _commons.get_path_to_results(
            output_dir, method_name, test_dataset_name=test_dataset_name,
            real_shearmap_name=real_shearmap_name,
            test_on_real_data=False,
            train_val_dataset_name=train_val_dataset_name,
            model_name=model_name
        )
        output_dir_real = _commons.get_path_to_results(
            output_dir, method_name, test_dataset_name=test_dataset_name,
            real_shearmap_name=real_shearmap_name,
            test_on_real_data=True,
            train_val_dataset_name=train_val_dataset_name,
            model_name=model_name
        )
    else:
        output_dir = _commons.get_path_to_results(
            output_dir, method_name, test_dataset_name=test_dataset_name,
            real_shearmap_name=real_shearmap_name,
            test_on_real_data=test_on_real_data,
            train_val_dataset_name=train_val_dataset_name,
            model_name=model_name
        )   # E.g., "results/dir/test_kappaTNG/pnpmass/kappaTNG/model_name/",
            # or "results/dir/test_cosmos/pnpmass/kappaTNG/model_name/"

    now = wlmmuq.utils.get_timestamp()
    device = _commons.get_device(verbose=verbose)
    if verbose:
        print(f"Number of workers: {num_workers}")

    # Load noise standard deviation and mask
    if use_zbins:
        assert path_to_zbins is not None
        zbins = wlmmuq.utils.get_zbins(path_to_zbins, idx_zbins=idx_zbins)
    else:
        zbins = None

    std_noise, mask, gamma_real = _commons.get_stdnoise_mask_shearmap(
        path_to_std_noise=path_to_std_noise,
        path_to_mask=path_to_mask,
        path_to_real_shearmap=path_to_real_shearmap,
        bin_data_from_cosmos=bin_data_from_cosmos,
        get_noisy_shear_map=test_on_real_data or run_both,
        imgsize=imgsize, cosmos_include_faint=cosmos_include_faint,
        max_z=max_z, resolution=resolution,
        east_right=True, zbins=zbins,
        inpainting=inpainting, verbose=verbose
    )

    # Load test set(s)
    if run_both:
        test_dataloader_sim, nbins = _commons.get_dataloader_massmapping(
            path_to_test_dataset, nimgs_test, imgsize, batch_size,
            num_workers, std_noise, mask, shuffle=False,
            test_on_real_data=False
        )
        test_dataloader_real, nbins_real = _commons.get_dataloader_massmapping(
            path_to_test_dataset, nimgs_test, imgsize, batch_size,
            num_workers, std_noise, mask, shuffle=False,
            test_on_real_data=True, gamma_real=gamma_real
        )
        _commons.check_nbins(nbins, nbins_real)
    else:
        test_dataloader, nbins = _commons.get_dataloader_massmapping(
            path_to_test_dataset, nimgs_test, imgsize, batch_size,
            num_workers, std_noise, mask, shuffle=False,
            test_on_real_data=test_on_real_data, gamma_real=gamma_real
        )

    # Load calibration set, if provided
    if cqr:
        calib_dataloader, nbins_calib = _commons.get_dataloader_massmapping(
            path_to_calib_dataset, nimgs_calib, imgsize, batch_size,
            num_workers, std_noise, mask,
            shuffle=True, min_idx_filename_ori=min_idx_filename_ori_calib
        )
        _commons.check_nbins(nbins, nbins_calib)
    else:
        calib_dataloader = None

    # Load trained denoisers
    denoiser, denoiser_uq = _commons.load_trained_models(
        checkpoint_dir, arch, timestamp,
        epoch=epoch, imgsize=imgsize,
        model_specs=model_specs, nbins=nbins,
        load_model_uq=load_model_uq,
        checkpoint_dir_uq=checkpoint_dir_uq, arch_uq=arch_uq,
        timestamp_uq=timestamp_uq, epoch_uq=epoch_uq,
        model_specs_uq=model_specs_uq,
        device=device, verbose=verbose, **kwargs
    )

    # Instantiate physics (forward model) and RMSE metric
    # When computing the RMSE, masked pixels (i.e., without any measured galaxy
    # in any redshift bin) are discarded
    mask_physics = None if inpainting else mask
    mask_onezbin = wlmmuq.utils.get_mask_onezbin(mask) # Shape = (nx, ny)
    physics = wlmmuq.physics.MassMapping(sigma=std_noise, mask=mask_physics).to(device)
    rmse_fn = wlmmuq.metric.RMSE(mask=mask_onezbin).to(device)

    hyperparam_precalib = \
        _commons.convert_into_hyperparam_list(
            hyperparam_precalib,
            find_optimal_hyperparam_precalib=find_optimal_hyperparam_precalib
        )

    # Convert arguments into lists
    multfact_step_size, step_size = _commons.convert_into_lists(
        multfact_step_size, step_size
    )

    for tau, alph in zip(step_size, multfact_step_size):
        beg_time = time.time()

        # Instantiate the PnP model
        pnpmass, pnpmass_uq, tau = \
                    _commons.get_pnpmass(
            denoiser, denoiser_uq,
            std_noise=std_noise, rmse_fn=rmse_fn, physics=physics,
            step_size=tau, multfact_step_size=alph,
            eps_sup_step_size=eps_sup_step_size,
            niter=niter, mode=mode,
            path_to_ps=path_to_ps,
            niter_per_step_g=niter_per_step_g, niter_per_step_ng=niter_per_step_ng,
            device=device, verbose=verbose
        )

        # Get Gaussian extractor
        # Note: the step sizes for the Gaussian extractor are computed automatically
        if mode == "residual":
            gaussian_extractor, callback_gaussian_extractor = \
                    _commons.get_gaussian_extractor(
                which=which_gaussian_extractor,
                path_to_ps=path_to_ps,
                white_noise=False,
                imgsize=imgsize, std_noise=std_noise, physics=physics,
                step_size=None, step_size_ng=None,
                eps_sup_step_size=eps_sup_step_size,
                niter=niter_wiener,
                starlet_detection_threshold=starlet_detection_threshold,
                device=device, verbose=False
            )
        else:
            gaussian_extractor = None
            callback_gaussian_extractor = None

        # Set callback list
        callback_list = []
        if callback_gaussian_extractor is not None:
            callback_list.append(callback_gaussian_extractor)
        callbacks = wlmmuq.callbacks.CallbackList(callback_list)

        # Prepare runs: either single or both (simulated and real)
        runs = []
        if run_both:
            runs.append(("sim", test_dataloader_sim, False, output_dir_sim))
            runs.append(("real", test_dataloader_real, True, output_dir_real))
        else:
            runs.append(("single", test_dataloader, test_on_real_data, output_dir))

        # Run inference for each requested dataset and collect results
        run_outputs: list[tuple[str, dict, str]] = []  # (name, out_pnpmass, out_dir)
        for run_name, tdataloader, td_real_flag, out_dir_run in runs:
            if verbose:
                if td_real_flag:
                    print(f"Compute PnPMass on the COSMOS shear map")
                else:
                    print(f"Compute PnPMass on the test set ({nimgs_test} images)")

            out_pnpmass_run = run_pnpmass_batch(
                pnpmass, pnpmass_uq, physics, tdataloader, tau, niter,
                rmse_fn=rmse_fn,
                gaussian_extractor=gaussian_extractor,
                test_on_real_data=td_real_flag,
                callbacks=callbacks,
                device=device, verbose=verbose
            )
            run_outputs.append((run_name, out_pnpmass_run, out_dir_run))

        # Run calibration once (if requested) and then apply to each inference result
        if (calib_dataloader is not None) and cqr:
            if verbose:
                print(f"Compute PnPMass on the calibration set ({nimgs_calib} images)")
            out_pnpmass_calib = run_pnpmass_batch(
                pnpmass, pnpmass_uq, physics, calib_dataloader, tau, niter,
                rmse_fn=rmse_fn,
                gaussian_extractor=gaussian_extractor,
                test_on_real_data=False,
                callbacks=callbacks,
                device=device, verbose=verbose,
            )
            kappa_true_calib = out_pnpmass_calib["kappa_true"]
            kappa_pred_calib = out_pnpmass_calib["kappa_pred"]
            var_calib = out_pnpmass_calib["var"]
        else:
            kappa_true_calib = None
            kappa_pred_calib = None
            var_calib = None

        # For each run, prepare out_dict and apply calibration if available, then save
        for run_name, out_pnpmass_run, out_dir_run in run_outputs:
            kappa_true = out_pnpmass_run["kappa_true"]
            kappa_pred = out_pnpmass_run["kappa_pred"]
            var = out_pnpmass_run["var"]
            rmse = out_pnpmass_run["rmse"]
            l2norm = out_pnpmass_run["l2norm"]

            try:
                rmse = rmse.cpu()
                l2norm = l2norm.cpu()
            except Exception:
                pass

            inference_time = _commons.get_inference_time(beg_time, verbose=verbose)

            trd = (run_name == "real") if run_both else test_on_real_data
            out_dict = {
                "inference_time": inference_time,
                "step_size": tau,
                "arch": arch,
                "niter": niter,
                "imgsize": imgsize,
                "confidence_uq": confidence_uq,
                "rmse": rmse,
                "l2norm": l2norm,
                "test_on_real_data": trd,
            }
            if not trd:
                out_dict.update({
                    "nimgs_test": nimgs_test,
                })

            if save_tensors:
                out_dict.update({
                    "kappa_pred": kappa_pred[:nimgs_save].cpu(),
                    "var": var[:nimgs_save].cpu(),
                })
                if not trd:
                    out_dict.update({
                        "kappa_true": kappa_true[:nimgs_save].cpu(),
                    })

            # Apply calibration if available
            if (kappa_pred_calib is not None) and (var_calib is not None):
                # mypy/static check help: these must be set when calib outputs exist
                assert kappa_true_calib is not None
                mode_cqr, scaling_factor_chisqcqr = _commons.convert_into_list_cqr_mode(
                    mode_cqr, scaling_factor_chisqcqr
                )
                for mcqr, a in zip(mode_cqr, scaling_factor_chisqcqr):
                    for rho in hyperparam_precalib:
                        # TODO: calibrate only once, instead of doing it for each run
                        uq_results = _commons.apply_calibration_and_get_metrics(
                            kappa_pred, var, kappa_true,
                            kappa_pred_calib, var_calib, kappa_true_calib,
                            confidence_uq=confidence_uq,
                            imgsize=imgsize, mode=mcqr, a=a,
                            hyperparam_precalib=rho,
                            find_optimal_hyperparam_precalib=find_optimal_hyperparam_precalib,
                            mask=mask_onezbin, save_tensors=save_tensors,
                            nimgs_save=nimgs_save,
                            device=device, verbose=verbose
                        )
                        uq_key = _commons.get_uq_keys(
                            mode_cqr=mcqr, scaling_factor_chisqcqr=a, rho=rho
                        )
                        out_dict.update({
                            uq_key: uq_results
                        })

            calibration_time = _commons.get_inference_time(
                beg_time, which="calibration", verbose=verbose
            )
            out_dict.update({
                "calibration_time": calibration_time,
                "nimgs_calib": nimgs_calib,
            })

            _commons.save_results(
                out_dict, out_dir_run, now, step_size=tau,
                prefix=output_prefix, verbose=verbose
            )


def run_pnpmass_batch(
        pnpmass: wlmmuq.optim.BaseOptim, pnpmass_uq: wlmmuq.optim.BaseOptim | None,
        physics: wlmmuq.physics.MassMapping,
        dataloader, step_size, niter,
        rmse_fn: wlmmuq.metric.RMSE | None = None,
        gaussian_extractor: wlmmuq.optim.BaseOptim | None = None,
        test_on_real_data: bool = False,
        callbacks: wlmmuq.callbacks.BaseCallback | None = None,
        device="cpu", verbose=False
):
    listof_kappa_true = []
    listof_kappa_pred = []
    listof_var = []
    listof_rmse = []
    listof_l2norm = []

    if callbacks is None:
        callbacks = wlmmuq.callbacks.BaseCallback()

    pbar = tqdm.tqdm(dataloader, disable=not verbose)
    pbar.set_description(f"Step size = {step_size:.2e}, Nb iterations = {niter}")
    for i, (kappa_true, gamma_noisy) in enumerate(pbar):
        callbacks.on_batch_begin(i)

        if not test_on_real_data:
            kappa_true = kappa_true.to(device)
            compute_metrics = True
        else: # No groung truth; kappa_true is set to torch.nan or None
            kappa_true = None
            compute_metrics = False

        gamma_noisy = gamma_noisy.to(device)
        with torch.no_grad():
            if gaussian_extractor is not None:
                kappa_g = gaussian_extractor(
                    gamma_noisy, physics, x_gt=None, compute_metrics=False
                )
                gamma_noisy = gamma_noisy - physics.A(kappa_g)
                if not test_on_real_data:
                    kappa_true = kappa_true - kappa_g

            out_pnpmass = pnpmass(
                gamma_noisy, physics, x_gt=kappa_true,
                compute_metrics=compute_metrics
            )
            if not test_on_real_data:
                kappa_pred, metrics = out_pnpmass
                rmse = metrics["rmse"]
            else:
                kappa_pred = out_pnpmass
                rmse = None

            if pnpmass_uq is not None:
                assert pnpmass_uq.custom_init is not None
                pnpmass_uq.custom_init.X_init = (kappa_pred,)
                var = pnpmass_uq(
                    gamma_noisy, physics, compute_metrics=False
                )
            else:
                var = torch.zeros(kappa_pred.shape, device=device)

            if gaussian_extractor is not None:
                kappa_pred = kappa_pred + kappa_g
                if not test_on_real_data:
                    assert kappa_true is not None
                    kappa_true = kappa_true + kappa_g

            if rmse_fn is not None and not test_on_real_data:
                l2norm = rmse_fn(kappa_true, 0)
            else:
                l2norm = None

        listof_kappa_true.append(kappa_true) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_kappa_pred.append(kappa_pred) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_var.append(var) # Shape = (batch_size, 1, imgsize, imgsize)
        listof_rmse.append(rmse) # Shape = (batch_size, niter)
        listof_l2norm.append(l2norm) # Shape = (batch_size, niter)

    if not test_on_real_data:
        kappa_true = torch.cat(listof_kappa_true, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    kappa_pred = torch.cat(listof_kappa_pred, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)
    var = torch.cat(listof_var, dim=0) # Shape = (nimgs, 1, imgsize, imgsize)

    try:
        rmse = torch.cat(listof_rmse, dim=0) # Shape = (nimgs, niter)
        l2norm = torch.cat(listof_l2norm, dim=0) # Shape = (nimgs, niter)
    except Exception:
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


def _merge_dicts[K, T](listof_dict: list[dict[K, T]]) -> dict[K, list[T]]:

    out: dict[K, list[T]] = {}
    for d in listof_dict:
        for k, v in d.items():
            out.setdefault(k, []).append(v)

    return out


def _merge_dicts_debiasing[T](
        listof_dict: list[dict[int, dict[int, T]]]
) -> dict[int, dict[int, list[T]]]:
    
    stage1: dict[int, list[dict[int, T]]] = _merge_dicts(listof_dict)
    out: dict[int, dict[int, list[T]]] = {}
    for thresh, v_list in stage1.items():
        out[thresh] = _merge_dicts(v_list)

    return out


def _apply_fn_inside_dict_debiasing[U, V](
    fn: typing.Callable[[U], V],
    dictof_dicts: dict[int, dict[int, U]],
    return_none_on_error: bool = False,
) -> dict[int, dict[int, typing.Optional[V]]]:
    """Apply `fn` to the inner dict values for debiasing dictionaries.

    If ``return_none_on_error`` is True, exceptions raised by ``fn(v)`` are
    caught and ``None`` is returned for that entry instead of propagating
    the error.
    """

    if return_none_on_error:
        def _safe_fn(v: U) -> V | None:
            try:
                return fn(v)
            except Exception:
                return None

        apply_fn = _safe_fn
    else:
        apply_fn = fn

    return {
        thresh: {
            tau: apply_fn(v) for tau, v in d.items()
        } for thresh, d in dictof_dicts.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    _add_arguments.model(parser)
    _add_arguments.model_uq(parser)
    _add_arguments.checkpoint(parser)
    _add_arguments.step_size_niter(parser, default_niter=_commons.NITER_PNPMASS)
    parser.add_argument(
        "--mode", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Mode for PnPMass. Possible values are: "
            "'regular', 'residual', or 'pnpmcalens'. "
            "Default = 'regular'"
        )
    )
    parser.add_argument(
        "--which-gaussian-extractor", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Type of Gaussian extractor. Possible values are 'wiener' or 'mcalens'. "
            "Only used if `--mode` is set to 'residual'. "
            f"Default = '{_commons.WHICH_GAUSSIAN_EXTRACTOR_PNPMASS}'"
        )
    )
    _add_arguments.gaussian_extractor(parser, wiener=True, mcalens=True, verbose=True)
    _add_arguments.std_noise_mask(parser)
    _add_arguments.test_calib_dataset(parser, batch_size=_commons.BATCH_SIZE)
    _add_arguments.cqr(parser)
    _add_arguments.output(parser, OUTPUT_PREFIX)
    _add_arguments.seed_verbose(parser)
    args = parser.parse_args()
    kwargs = vars(args).copy()

    main(**kwargs)
