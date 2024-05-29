import os
import sys
import warnings
import pickle
import argparse
import time
import numpy as np

current_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(current_dir)

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import weaklensing as wl
import weaklensing.utils as wlutils
import weaklensing.kappatng as wlktng
import weaklensing.cosmos as wlcosmos

pycs_dir = os.path.expanduser(wl.CONFIG_DATA['pycs_dir'])
sys.path.append(pycs_dir) # tested with commit nb XXX

import pycs.astro.wl.mass_mapping as csmm

SIZE = 1.5 # opening angle (deg)
NINPIMGS = 25 # number of images to load from the kappaTNG dataset
NINPIMGS_PS = 20 # separate set of images to compute the power spectrum
CONFIDENCE = 2 # number of sigmas
NIMGS_CALIB = 100 # size of the calibration set

METHOD_LIST = ["mle", "wiener", "mcalens"]

def main(
        method, picklename, size=SIZE,
        ninpimgs=NINPIMGS, ninpimgs_ps=NINPIMGS_PS,
        nimgs=None, niter=None, Nsigma=None, batch_size=None, uq=False, nsamples=None,
        batch_size_noise=None, verbose=False, **kwargs
):
    beg = time.time()
    assert method in METHOD_LIST

    # Get number of pixels and adjusted opening angle
    width, size = wlktng.get_npixels(size)

    # Load data from the COSMOS catalog
    cat_cosmos = wlcosmos.cosmos_catalog(
        include_faint=False
    )
    data_cosmos = wlcosmos.get_data_from_cosmos(cat_cosmos, size)
    extent = data_cosmos["extent"]

    # Remove galaxies that are not in the redshift range of the kappaTNG dataset
    cat_cosmos = cat_cosmos[cat_cosmos['zphot'] >= np.min(wlktng.LIST_OF_Z)]
    cat_cosmos = cat_cosmos[cat_cosmos['zphot'] < np.max(wlktng.LIST_OF_Z)]

    # Get standard deviation of galaxy ellipticities
    shapedisp1, shapedisp2 = data_cosmos["shapedisp"]
    shapedisp = (shapedisp1 + shapedisp2) / 2

    # Get map of number of galaxies per pixel
    ngal = wlutils.ngal_per_pixel(cat_cosmos["Ra"], cat_cosmos["Dec"], width, extent)

    # Get a list of weights, for each redshift in the $\kappa$-TNG dataset
    weights_redshift = wlktng.get_weights(cat_cosmos['zphot'])

    # Load convergence maps from the kappaTNG dataset
    kappa = wlktng.kappa_tng(weights_redshift, ninpimgs, width=width)
    if nimgs is None:
        nimgs = kappa.shape[0]
    else:
        assert nimgs <= kappa.shape[0]

    # Create noisy shear maps
    gamma1, gamma2 = wlutils.get_shear_from_convergence(kappa)
    gamma1_noisy, gamma2_noisy, std_noise = wlutils.get_masked_and_noisy_shear(
        gamma1, gamma2, ngal, shapedisp, stdnoise_mask=0.
    ) # do not add noise to masked data

    # Initialize `csmm.shear_data` object
    sheardata = csmm.shear_data()
    sheardata.mask = (ngal > 0).astype(int)
    sheardata.Ncov = 2 * std_noise**2 # factor 2 required

    # Initialize `csmm.massmap2d` object
    massmap = csmm.massmap2d()
    massmap.init_massmap(width, width)
    if niter is not None:
        massmap.DEF_niter = niter
    if Nsigma is not None:
        massmap.DEF_Nsigma = Nsigma
    if verbose:
        massmap.Verbose = True

    # Compute the 1D power spectrum from simulated convergence maps
    if method in ("wiener", "mcalens"):
        kappa_ps = wlktng.kappa_tng(
            weights_redshift, ninpimgs_ps, start_idx=ninpimgs, width=width
        )
        powerspectrum = np.mean(
            np.abs(np.fft.fft2(kappa_ps) / width)**2, axis=0
        ) # expected value of the squared Fourier modulus
        powerspectrum = powerspectrum[:width//2, :width//2] # only positive frequencies, by symmetry
        powerspectrum_1d = (
            powerspectrum[0, :] + powerspectrum[:, 0]
        ) / 2 # assumed isotropic
        del kappa_ps

    # Select mass mapping method
    if method == "wiener":
        func = massmap.prox_wiener_filtering
        kwargs.update(PowSpecSignal=powerspectrum_1d)
        idx_rec = (0,)
    elif method == "mle":
        func = massmap.prox_mse
        kwargs.update(sigma=wlutils.STD_KSGAUSSIANFILTER)
        idx_rec = (0,)
    elif method == "mcalens":
        func = massmap.sparse_wiener_filtering
        kwargs.update(PowSpecSignal=powerspectrum_1d, Bmode=False)
        idx_rec = (0, 2)
    else:
        raise ValueError("Unknown method.")

    # Batch size to avoid memory overload
    if batch_size is None:
        batch_size = nimgs

    # Uncertainty quantification: Monte-Carlo approach
    if uq:
        if batch_size_noise is None:
            batch_size_noise = nsamples
        kwargs.update(PropagateNoise=True)

    recs = tuple([] for _ in idx_rec)
    max_idx = 0

    # Loop over batches of images
    while max_idx < nimgs:
        beg_loop = time.time()
        min_idx = max_idx
        max_idx = min(min_idx + batch_size, nimgs)

        if verbose:
            print(f"Images {min_idx} to {max_idx - 1}...")

        # Register data into the `csmm.shear_data` object
        sheardata.g1 = gamma1_noisy[min_idx:max_idx]
        sheardata.g2 = gamma2_noisy[min_idx:max_idx]

        recs_batch = tuple([] for _ in idx_rec)
        if uq:
            nremainingsamples = nsamples

        # Loop over batches of noise realizations (useful if `uq` is True)
        while True:
            if uq:
                Nrea = min(batch_size_noise, nremainingsamples)
                kwargs.update(Nrea=Nrea)
                nremainingsamples -= Nrea
                if verbose:
                    print(f"Propagating {Nrea} noise realizations...")
            rec = func(
                sheardata, Inpaint=False, **kwargs
            )
            for i, j in enumerate(idx_rec):
                recs_batch[i].append(rec[j])
            if (not uq) or (nremainingsamples == 0):
                break

        if uq:
            # Concatenate over Nrea
            rec = tuple(
                np.concatenate(r, axis=-3) for r in recs_batch
            ) # tuple of arrays, shape = ([nimgs], Nrea, nx, ny])
            # Compute output standard deviation, for each input image
            rec = tuple(
                np.std(r, axis=-3) for r in rec
            ) # tuple of arrays, shape = ([nimgs], nx, ny)
        else:
            # Each list in recs_batch contains only one array
            rec = tuple(
                r[0] for r in recs_batch
            ) # tuple of arrays, shape = (nimgs, nx, ny)

        for i, _ in enumerate(idx_rec):
            recs[i].append(rec[i])

        end_loop = time.time()
        if verbose:
            exec_time = end_loop - beg_loop
            hours = int(exec_time // 3600)
            minutes = int((exec_time % 3600) // 60)
            seconds = int(exec_time % 60)
            print(f"Execution time: {hours} h, {minutes} min, {seconds} sec")

            elapsed_time = end_loop - beg
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            print(f"Elapsed time: {hours} h, {minutes} min, {seconds} sec")


    # Concatenate over nimgs, except when `uq` is True and uncertainty does not depend on
    # the input image. In this case, each list in `recs` contains only one array of shape (nx, ny).
    # This happens when the chosen mass mapping method is the Wiener filtering or the MLE reconstruction.
    # On the other hand, for MCALens, each input image comes with its own uncertainty array.
    if len(recs[0][0].shape) == 2:
        # Each list in recs should contain only one array
        if len(recs[0]) > 1:
            warnings.warn((
                "The uncertainty matrix has been computed multiple times; all but "
                "one will be discarded."
            ), UserWarning)
        rec = tuple(r[0] for r in recs) # shape = (nx, ny)
    else:
        rec = tuple(
            np.concatenate(r, axis=0) for r in recs
        ) # shape = (nimgs, nx, ny)

    # Pickle data
    pickle_dir = os.path.expanduser(wl.CONFIG_DATA['pickle_dir'])
    fn = os.path.join(pickle_dir, picklename)
    with open(fn, 'wb') as f:
        pickle.dump((kappa, *rec), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "method", type=str,
        help=f"Mass mapping method: {' | '.join(METHOD_LIST)}"
    )
    parser.add_argument(
        "picklename", type=str,
        help="File name for the saved object"
    )
    parser.add_argument(
        "-s", "--size", type=float,
        default=argparse.SUPPRESS,
        help="Opening angle (deg)"
    )
    parser.add_argument(
        "--ninpimgs", type=int,
        default=argparse.SUPPRESS,
        help=f"Number of input images to load from the kappaTNG dataset. Default = {NINPIMGS}"
    )
    parser.add_argument(
        "--ninpimgs-ps", type=int,
        default=argparse.SUPPRESS,
        help=f"Number of additional input images to compute the power spectrum. Default = {NINPIMGS_PS}"
    )
    parser.add_argument(
        "--nimgs", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Number of images to reconstruct. Default = None (all images are considered)"
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
        "-b", "--batch-size", type=int,
        default=argparse.SUPPRESS,
        help=(
            "Batch size, to avoid memory overload. "
            "Default = None (all input images are computed in a single batch)"
        )
    )
    parser.add_argument(
        "--uq", action='store_true',
        default=argparse.SUPPRESS,
        help="Whether to propagate noise, for uncertainty quantification. Default = False"
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
        "-v", "--verbose", action='store_true',
        default=argparse.SUPPRESS
    )

    args = parser.parse_args()
    kwargs = vars(args).copy()
    main(**kwargs)
