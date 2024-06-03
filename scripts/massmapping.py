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
        method, picklename, size=SIZE, idx_redshift=None,
        ninpimgs=NINPIMGS, ninpimgs_ps=NINPIMGS_PS,
        nimgs=None, niter=None, Nsigma=None, batch_size=None, uq=False, nsamples=None,
        batch_size_noise=None, verbose=False, **kwargs
):
    beg = time.time()
    assert method in METHOD_LIST

    # Instantiate KappaTNG object
    ktng = wlktng.KappaTNG(size=size, idx_redshift=idx_redshift)
    size = ktng.size # adjusted opening angle

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

    # Get a list of weights, for each redshift in the $\kappa$-TNG dataset
    if ktng.idx_redshift is None:
        weights_redshift = wlktng.get_weights(cat_cosmos['zphot'])
        ktng.weights = weights_redshift

    # Load convergence maps from the kappaTNG dataset
    kappa = ktng.get_kappa(ninpimgs)
    if nimgs is None:
        nimgs = kappa.shape[0]
    else:
        assert nimgs <= kappa.shape[0]

    # Get map of number of galaxies per pixel
    ngal = wlutils.ngal_per_pixel(
        cat_cosmos["Ra"], cat_cosmos["Dec"], ktng.width, extent
    )

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
    massmap.init_massmap(ktng.width, ktng.width)
    if niter is not None:
        massmap.DEF_niter = niter
    if Nsigma is not None:
        massmap.DEF_Nsigma = Nsigma
    if verbose:
        massmap.Verbose = True

    # Compute the 1D power spectrum from simulated convergence maps
    if method in ("wiener", "mcalens"):
        kappa_ps = ktng.get_kappa(ninpimgs_ps, start_idx=ninpimgs)
        powerspectrum = np.mean(
            np.abs(np.fft.fft2(kappa_ps) / ktng.width)**2, axis=0
        ) # expected value of the squared Fourier modulus
        # Only positive frequencies, by symmetry
        powerspectrum = powerspectrum[:ktng.width//2, :ktng.width//2]
        powerspectrum_1d = (
            powerspectrum[0, :] + powerspectrum[:, 0]
        ) / 2 # assumed isotropic
        del kappa_ps

    # Select mass mapping method
    if method == "wiener":
        func = massmap.prox_wiener_filtering
        kwargs.update(PowSpecSignal=powerspectrum_1d)
    elif method == "mle":
        func = massmap.prox_mse
        kwargs.update(sigma=wlutils.STD_KSGAUSSIANFILTER)
    elif method == "mcalens":
        func = massmap.sparse_wiener_filtering
        kwargs.update(PowSpecSignal=powerspectrum_1d, Bmode=False)
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

    recs = []
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

        recs_batch = []
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
            )[0]
            if method == 'mcalens':
                # Mean centering
                rec -= np.mean(rec, axis=(-2, -1), keepdims=True)
            recs_batch.append(rec)
            if (not uq) or (nremainingsamples == 0):
                break

        if uq:
            # Concatenate over Nrea
            rec = np.concatenate(recs_batch, axis=-3) # shape = (nimgs, Nrea, nx, ny])
            # Compute output standard deviation, for each input image
            rec = tuple(np.std(rec, axis=-3)) # shape = (nimgs, nx, ny)
        else:
            # The list recs_batch contains only one array
            rec = recs_batch[0] # shape = (nimgs, nx, ny)

        recs.append(rec)

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


    # Concatenate over nimgs
    rec = np.concatenate(recs, axis=0) # shape = (nimgs, nx, ny)

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
        "--idx-redshift", type=int,
        default=argparse.SUPPRESS,
        help="Default = None"
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
