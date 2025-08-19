import os
import argparse
import time
import threading
import warnings
import tqdm
import cProfile
import numpy as np

import wlmmuq.utils as wlutils
import wlmmuq.data.torch as wldata

try:
    import pycs.astro.wl.mass_mapping as csmm
except ImportError:
    warnings.warn("Module `pycs` not found.")

from wlmmuq import PATH_TO_STD_NOISE, PATH_TO_MASK
from wlmmuq.data import NUM_WORKERS

import _commons
from _commons import IMGSIZE

CONFIDENCE = 2 # number of sigmas
NIMGS = 225
NIMGS_PS = 256 # images used to compute the power spectrum

METHOD_LIST = ["wiener", "mcalens"]

def main(
        method, pickledir, picklename, path_to_test_dataset,
        path_to_std_noise: str=PATH_TO_STD_NOISE,
        path_to_mask: str=PATH_TO_MASK,
        cosmos_include_faint=False, imgsize=IMGSIZE,
        nimgs=NIMGS, nimgs_ps=NIMGS_PS, path_to_powerspectrum=None,
        batch_size=None, uq=False, nsamples=None,
        batch_size_noise=None, cprofiler=False,
        seed=None, verbose=False, **kwargs
):
    _commons.set_seed(seed)

    beg = time.time()
    pickledir = os.path.expanduser(pickledir)
    os.makedirs(pickledir, exist_ok=True)
    assert method in METHOD_LIST

    keys_massmapping = ['niter', 'Nsigma', 'Inpaint']
    kwargs_massmapping = {k: kwargs.pop(k) for k in keys_massmapping if k in kwargs}

    std_noise, mask = _commons.get_stdnoise_mask(
        path_to_std_noise=path_to_std_noise,
        path_to_mask=path_to_mask,
        imgsize=imgsize, cosmos_include_faint=cosmos_include_faint,
        convert_to_torch_tensor=True, verbose=verbose
    )

    # Initialize dataset
    dataset = wldata.HDF5DatasetMassMapping(
        path_to_test_dataset, nimgs=nimgs,
        batch_size=batch_size,
        std_noise=std_noise, mask=mask, inpainting=False, shuffle=False,
        output_shape=imgsize, **kwargs
    )
    dataloader = dataset.to_dataloader()

    # Initialize `csmm.shear_data` object
    sheardata = csmm.shear_data()
    sheardata.mask = mask.numpy().astype(int)
    sheardata.Ncov = 2 * std_noise.numpy()**2 # factor 2 required
                                              # (variance of the complex-valued noise)

    # Initialize `csmm.massmap2d` object
    massmap = csmm.massmap2d()
    massmap.init_massmap(imgsize, imgsize)
    if verbose:
        massmap.Verbose = True

    # Compute the 1D power spectrum from simulated convergence maps
    if method in ("wiener", "mcalens"):
        if path_to_powerspectrum is None:
            dataset_ps = wldata.HDF5DatasetKappa(
                path_to_test_dataset, nimgs=nimgs_ps, sort_by_filename_ori=True,
                shuffle=False, beg_idx=nimgs, output_shape=imgsize, **kwargs
            )
            kappa_ps = dataset_ps.load_batch(get_all_images=True)
            kappa_ps = kappa_ps.numpy()
            powerspectrum_1d = wlutils.get_1d_powerspectrum(kappa_ps)
            del kappa_ps
        else:
            path_to_powerspectrum = os.path.expanduser(path_to_powerspectrum)
            powerspectrum_1d = np.load(path_to_powerspectrum)
        kwargs_massmapping.update(PowSpecSignal=powerspectrum_1d)

    # Select mass mapping method
    if method == "wiener":
        func = massmap.prox_wiener_filtering
    elif method == "mcalens":
        func = massmap.sparse_wiener_filtering
    else:
        raise ValueError("Unknown method.")

    # Uncertainty quantification: Monte-Carlo approach
    if uq:
        if batch_size_noise is None:
            batch_size_noise = nsamples
        kwargs_massmapping.update(PropagateNoise=True)

    # Start profiling
    if cprofiler:
        profiler = cProfile.Profile()
        filename_stats = os.path.join(pickledir, f"{picklename}.prof")
        def _print_stats():
            while True:
                time.sleep(15)
                profiler.dump_stats(filename_stats)
        profiler.enable()
        stats_thread = threading.Thread(target=_print_stats, daemon=True)
        stats_thread.start()

    recs = []
    exec_times = []
    elapsed_times = []

    # Loop over batches of images
    max_idx = 0
    dataloader = iter(dataloader)
    if batch_size is None:
        batch_size = nimgs
    pbar = tqdm.tqdm(
        range(-(-nimgs // batch_size)),
        disable=not verbose,
    )
    for i in pbar:
        min_idx = i * batch_size
        max_idx = min(min_idx + batch_size, nimgs)
        pbar.set_description(f"Images {min_idx + 1}-{max_idx}/{nimgs}")
        beg_loop = time.time()
        _, gamma_noisy = next(dataloader)

        # Register data into the `csmm.shear_data` object
        gamma_noisy = gamma_noisy.numpy()
        gamma_noisy = -np.conjugate(gamma_noisy) # TODO: I don't know why this is needed
        sheardata.g1 = gamma_noisy.real
        sheardata.g2 = gamma_noisy.imag

        recs_batch = []
        if uq:
            nremainingsamples = nsamples

        # Loop over batches of noise realizations (useful if `uq` is True)
        while True:
            if uq:
                Nrea = min(batch_size_noise, nremainingsamples)
                kwargs_massmapping.update(Nrea=Nrea)
                nremainingsamples -= Nrea
            rec = func(
                sheardata, **kwargs_massmapping
            )[0]
            recs_batch.append(rec)
            if (not uq) or (nremainingsamples == 0):
                break

        if uq:
            # Concatenate over Nrea
            rec = np.concatenate(recs_batch, axis=-3) # shape = (nimgs, Nrea, nx, ny)
            # Compute output standard deviation, for each input image
            rec = tuple(np.std(rec, axis=-3)) # shape = (nimgs, nx, ny)
        else:
            # The list recs_batch contains only one array
            rec = recs_batch[0] # shape = (nimgs, nx, ny)

        recs.append(rec)

        end_loop = time.time()
        exec_time = end_loop - beg_loop
        elapsed_time = end_loop - beg
        exec_times.append(exec_time)
        elapsed_times.append(elapsed_time)
        if verbose:
            hours = int(exec_time // 3600)
            minutes = int((exec_time % 3600) // 60)
            seconds = int(exec_time % 60)
            print(f"Execution time: {hours} h, {minutes} min, {seconds} sec")

            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            print(f"Elapsed time: {hours} h, {minutes} min, {seconds} sec")

    rec = np.concatenate(recs, axis=0) # shape = (nimgs, nx, ny)
    exec_times = np.array(exec_times) # shape = -(-nimgs // batch_size,)
    elapsed_times = np.array(elapsed_times) # shape = -(-nimgs // batch_size,)

    save_data = {
        "rec": rec,
        "exec_times": exec_times,
        "elapsed_times": elapsed_times
    }
    if batch_size is not None:
        save_data["batch_size"] = batch_size
    if nimgs_ps is not None:
        save_data["nimgs_ps"] = nimgs_ps
    if powerspectrum_1d is not None:
        save_data["powerspectrum_1d"] = powerspectrum_1d

    # Pickle data
    np.savez(os.path.join(pickledir, f"{picklename}.npz"), **save_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method", type=str,
        help=f"Mass mapping method: {' | '.join(METHOD_LIST)}"
    )
    parser.add_argument(
        "pickledir", type=str,
        help=(
            "Directory where to save the pickled data. "
            "If the directory does not exist, it will be created."
        )
    )
    parser.add_argument(
        "picklename", type=str,
        help=(
            "File name (without extension)."
        )
    )
    parser.add_argument(
        "path_to_test_dataset", type=str,
        help="Path to the test dataset (HDF5 file)"
    )
    parser.add_argument(
        "--cosmos-include-faint", action='store_true',
        default=argparse.SUPPRESS,
        help="Include the faint galaxies from the COSMOS S10 shear catalog"
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=f"Number of input images. Default = {NIMGS}"
    )
    parser.add_argument(
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size, to avoid memory overload. "
            "Default = None (all input images are computed in a single batch)"
        )
    )
    parser.add_argument(
        "--nimgs-ps", type=int,
        default=argparse.SUPPRESS,
        help=f"Number of additional images to compute the power spectrum. Default = {NIMGS_PS}"
    )
    parser.add_argument(
        "-ps", "--path-to-powerspectrum", type=str,
        default=argparse.SUPPRESS,
        help=(
            "Path to the .npy file containing the 1D power spectrum. "
            "If not provided, then the power spectrum will be inferred from the "
            "dataset. Default = None"
        )
    )
    parser.add_argument(
        "--niter", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of iterations, in case of iterative mass mapping method. "
            "Default = massmap2d.DEF_niter"
        )
    )
    parser.add_argument(
        "--Nsigma", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Default detection level in wavelet space, for MCALens. "
            "Default = massmap2d.DEF_Sigma"
        )
    )
    parser.add_argument(
        "--Inpaint", action='store_true',
        default=argparse.SUPPRESS,
        help="Inpaint the missing data."
    )
    parser.add_argument(
        "-w", "--num-workers", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of workers for parallel processing. "
            f"Default = {NUM_WORKERS}"
        )
    )
    parser.add_argument(
        "--uq", action='store_true',
        default=argparse.SUPPRESS,
        help="Propagate noise, for uncertainty quantification."
    )
    parser.add_argument(
        "--nsamples", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of noise realizations. Depending on the mass mapping method, each input image "
            "may or may not get its own set of noise realizations. Must be provided if option --uq is activated."
        )
    )
    parser.add_argument(
        "-bn", "--batch-size-noise", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of noise realizations. Depending on the mass mapping method, each input image "
            "may or may not get its own set of noise realizations. Default = None (all noise realizations "
            "computed in a single batch)"
        )
    )
    parser.add_argument(
        "--cprofiler", action='store_true',
        default=argparse.SUPPRESS,
        help=(
            "Profile training using cProfile."
        )
    )
    _commons.add_arguments_seed_verbose(parser)

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
