import os
import numpy as np
import matplotlib.pyplot as plt

import pycs.astro.wl.mass_mapping as csmm

from . import utils as wlutils

############################################################################
# PGD mass mapping
############################################################################

class BasePGDMassMapping:
    """
    Base class for iterative proximal gradient descent (or forward-backward)
    algorithm, applied to the mass mapping problem.

    """
    def __init__(
            self, step_size, backward, niter, std_noise=None, mask=None,
            verbose=False
    ):
        """
        Parameters
        ----------
        step_size: float
            The step size of the PGD algorithm.
        backward: callable
            The proximal operator, or a trained denoiser for plug-and-play PGD.
            The denoiser should be trained on images corrupted by a white
            Gaussian noise with zero mean and variance equal to step_size.
        niter: int
            Number of iterations.
        std_noise: np.ndarray, shape = (nx, ny), default = None
            Array representing the noise standard deviation for each pixel
            (diagonal elements of the noise covariance matrix).
        mask: np.ndarray, shape = (nx, ny), default = None
            Mask to apply in case of missing data.
        verbose: bool, default=False
        
        """
        self.step_size = step_size
        self.backward = backward
        self.niter = niter
        self.std_noise = std_noise
        self.mask = mask
        self.verbose = verbose


    def forward(self, kappa, gamma, i=None, callbacks=None):
        # Gradient-descent step
        kappa = kappa + self.step_size * self.neg_grad(
            kappa, gamma, i=i, callbacks=callbacks
        )

        return kappa


    def neg_grad(self, kappa, gamma, i=None, callbacks=None):
        raise NotImplementedError


    def conv2shear_masked(self, kappa):
        gamma = wlutils.get_shear_from_convergence(
            kappa, return_complex=True
        )
        if self.mask is not None:
            gamma[..., ~self.mask] = 0
        return gamma


    def shear2conv_masked(self, gamma):
        if self.mask is not None:
            gamma[..., ~self.mask] = 0
        kappa = wlutils.get_convergence_from_shear(
            gamma, return_complex=True
        )
        return kappa


    def __call__(self, gamma, kappa0=None, callbacks=None):

        if callbacks is None:
            callbacks = []
        if kappa0 is not None:
            assert kappa0.shape == gamma.shape
            kappa = kappa0.copy() # Shape = ([nimgs], nx, ny)
        else:
            kappa = np.zeros(gamma.shape) # Shape = ([nimgs], nx, ny)
        for callback in callbacks:
            callback.on_predict_begin(kappa)
        for i in range(self.niter):
            if self.verbose:
                print(f'Iteration {i+1}')

            # Forward step
            kappa = self.forward(kappa, gamma, i=i, callbacks=callbacks)
            for callback in callbacks:
                callback.on_forward_end(i, kappa)

            # Backward step
            kappa = self.backward(kappa)
            for callback in callbacks:
                callback.on_backward_end(i, kappa)

        for callback in callbacks:
            callback.on_predict_end(kappa)

        return kappa


class BayesianPGDMassMappingNoPrecond(BasePGDMassMapping):
    r"""
    FB algorithm with Bayesian data-fidelity term:
    $$
    f(\kappa) := \frac12 \|\gamma - A\kappa\|_{\Sigma^{-1}}^2,
    $$
    without pre-conditioning.
    In the PnP version, the denoiser is trained on images corrupted by white noise
    with variance equal to self.step_size.
    The step size self.step_size should be smaller than 2 sigma_min**2, where
    sigma_min denotes the minimum standard deviation given by self.std_noise.
    
    """
    def neg_grad(self, kappa, gamma, i=None, callbacks=None):
        resgamma = gamma - self.conv2shear_masked(kappa)
        for callback in callbacks:
            callback.on_debug_event(i=i, eventname='residual', intarray=resgamma.real)
        resgamma /= self.std_noise**2
        for callback in callbacks:
            callback.on_debug_event(
                i=i, eventname=r'$\Sigma^{-1}$-scaling', intarray=resgamma.real
            )
        out = self.shear2conv_masked(resgamma).real
        for callback in callbacks:
            callback.on_debug_event(i=i, eventname='KS filtering', intarray=out)
        return out


class BayesianPGDMassMappingPrecond(BayesianPGDMassMappingNoPrecond):
    r"""
    FB algorithm with Bayesian data-fidelity term:
    $$
    f(\kappa) := \frac12 \|\gamma - A\kappa\|_{\Sigma^{-1}}^2,
    $$
    with pre-conditioning.
    In the PnP version, the denoiser is trained on images corrupted by heteroscedastic
    noise with variance equal to self.step_size * self.std_noise**2.
    The step size self.step_size should be smaller than 2 sigma_min**2 / sigma_max**2,
    where sigma_min and sigma_max denote the minimum and maximum standard deviations
    given by self.std_noise, respectively.
    
    """
    def neg_grad(self, kappa, gamma, i=None, callbacks=None):
        out = super().neg_grad(kappa, gamma, i=i, callbacks=callbacks)
        out *= self.std_noise**2 # Pre-conditioning
        for callback in callbacks:
            callback.on_debug_event(
                i=i, eventname=r'$\Sigma$-scaling', intarray=out
            )
        return out


class L2PGDMassMapping(BasePGDMassMapping):
    r"""
    FB algorithm with L2 data-fidelity term:
    $$
    f(\kappa) := \frac12 \|\gamma - A\kappa\|_2^2.
    $$
    In the PnP version, the denoiser is trained on images corrupted by heteroscedastic
    noise with variance equal to self.step_size * self.self.std_noise**2.
    The step size self.step_size should be smaller than 2.
    
    """
    def neg_grad(self, kappa, gamma, i=None, callbacks=None):
        resgamma = gamma - self.conv2shear_masked(kappa)
        for callback in callbacks:
            callback.on_debug_event(i=i, eventname='residual', intarray=resgamma.real)
        out = self.shear2conv_masked(resgamma).real
        for callback in callbacks:
            callback.on_debug_event(i=i, eventname='KS filtering', intarray=out)
        return out


class NoisewhiteningPGDMassMapping(BasePGDMassMapping):
    r"""
    FB algorithm with the following noise-whitening data-fidelity term:
    $$
    f(\kappa) := \frac12 \|\gamma - A\kappa\|_{\Sigma^{-1/2}}^2.
    $$
    In the PnP version, the denoiser is trained on images corrupted by white noise
    with variance equal to self.step_size**2.
    The step size self.step_size should be smaller than 2 sigma_min, where
    sigma_min denotes the minimum standard deviation given by self.std_noise.

    """
    def neg_grad(self, kappa, gamma, i=None, callbacks=None):
        resgamma = gamma - self.conv2shear_masked(kappa)
        for callback in callbacks:
            callback.on_debug_event(i=i, eventname='residual', intarray=resgamma.real)
        resgamma /= self.std_noise
        for callback in callbacks:
            callback.on_debug_event(
                i=i, eventname=r'$\Sigma^{-1/2}$-scaling', intarray=resgamma.real
            )
        out = self.shear2conv_masked(resgamma).real
        for callback in callbacks:
            callback.on_debug_event(i=i, eventname='KS filtering', intarray=out)
        return out


############################################################################
# Callbacks
############################################################################

class Callback:

    def on_forward_end(self, i, kappa):
        pass

    def on_backward_end(self, i, kappa):
        pass

    def on_predict_begin(self, kappa):
        pass

    def on_predict_end(self, kappa):
        pass

    def on_debug_event(self, i, eventname, intarray):
        pass


class SaveIntermediateMaps(Callback):

    def __init__(self, savedir, saveevery=1):
        """
        Parameters
        ----------
        savedir: str, default = None
            Directory where to save the intermediate arrays as .npy files.
        
        """
        self.savedir = savedir
        self.saveevery = saveevery

    def _save(self, i, kappa, savedir):
        if (i+1) % self.saveevery == 0:
            np.save(
                os.path.join(
                    self.savedir, savedir, f'kappa_{i+1}.npy'
                ), kappa
            )

    def on_forward_end(self, i, kappa):
        self._save(i, kappa, 'forward')

    def on_backward_end(self, i, kappa):
        self._save(i, kappa, 'backward')


class ShowIntermediateMaps(Callback):

    def __init__(
            self, idx, showevery=1, figsize=(8, 3), debug=False,
            step_size=1., **kwargs
    ):
        self.idx = idx
        self.showevery = showevery
        self.figsize = figsize
        self.debug = debug
        self.step_size = step_size
        self.kwargs = kwargs

    def _show(self, kappa, **kwargs):
        kwargs.update(**self.kwargs)
        wlutils.skyshow(
            kappa[self.idx],
            printxylabels=False, printxticks=False, printyticks=False,
            printcolorbar=True, **kwargs
        )

    def on_forward_end(self, i, kappa):
        if (i+1) % self.showevery == 0:
            plt.figure(figsize=self.figsize)
            plt.subplot(121)
            self._show(kappa, title=f'Iteration {i+1} (forward)')

    def on_backward_end(self, i, kappa):
        if (i+1) % self.showevery == 0:
            plt.subplot(122)
            self._show(kappa, title=f'Iteration {i+1} (backward)')
            plt.show()

    def on_debug_event(self, i, eventname, intarray):
        if self.debug and i == 0:
            plt.figure(figsize=(4, 3))
            self._show(self.step_size * intarray, title=eventname)
            plt.show()


class RMSE(Callback):

    def __init__(self, kappa_true, mask=None, path_to_saved_stats=None):
        """
        Parameters
        ----------
        kappa_true: np.ndarray, shape = (nimgs, nx, ny)
            Ground truth against which to compute RMSE.
        mask: np.ndarray, shape = (nx, ny), default = None
            If specified, compute RMSE over the mask.
        path_to_saved_stats: str, default = None
            Path to the .npy file where the arrays of RMSE are saved.
        
        """
        self.kappa_true = kappa_true
        self.mask = mask
        self.path_to_saved_stats = path_to_saved_stats

        self.rmse_backward = None
        self._reset()


    def _reset(self):
        self.rmse_backward = []

    def _rmse(self, kappa, stat_list):
        out = wlutils.rmse(
            kappa, self.kappa_true, mask=self.mask
        )
        stat_list.append(out)

    def on_backward_end(self, _, kappa):
        self._rmse(kappa, self.rmse_backward)

    def on_predict_begin(self, kappa):
        self._reset()
        self._rmse(kappa, self.rmse_backward)

    def on_predict_end(self, _):
        self.rmse_backward = np.stack(self.rmse_backward)
        if self.path_to_saved_stats is not None:
            np.save(self.path_to_saved_stats, self.rmse_backward)


class UQ(Callback):

    def __init__(
            self, pgd_massmapping: BasePGDMassMapping, backward_uq,
            gamma: np.ndarray
    ):
        """
        Parameters
        ----------
        pgd_massmapping: BasePGDMassMapping instance
        backward_uq: callable, default = None
            Performs a specific forward-backward step after the last
            PGD iteration for uncertainty quantification.
        gamma: np.ndarray, shape = (nimgs, nx, ny)
            Input noisy shear map
        
        """
        self.pgd_massmapping = pgd_massmapping
        self.backward_uq = backward_uq
        self.gamma = gamma

        self.kappa_uq = None

    def on_predict_end(self, kappa):
        print("Uncertainty quantification...")
        kappa_uq = kappa.copy()
        kappa_uq = self.pgd_massmapping.forward(kappa_uq, self.gamma)
        self.kappa_uq = self.backward_uq(kappa)


############################################################################
# Backward operators
############################################################################

class ProximalWiener:
    r"""
    Class for instantiating a backward operator for iterative Wiener filtering.
    $$
    \text{prox}_{\tau g}(\kappa) := \mathbf F^\ast \left(
        \mathbf I + \tau \Sigma_{\kappa}^{-1}
    \right)^{-1} \mathbf F \kappa
    $$

    """
    def __init__(self, imgsize, powerspectrum_1d, step_size):

        assert 2 * len(powerspectrum_1d) == imgsize
        powerspectrum = csmm.get_ima_spectrum_map(powerspectrum_1d, imgsize, imgsize)
        powerspectrum = np.fft.fftshift(powerspectrum)
        self.fourierfilter = 1 / (1 + step_size / powerspectrum)


    def __call__(self, inp):

        out = np.fft.fft2(inp)
        out = self.fourierfilter * out
        out = np.fft.ifft2(out)

        return out.real


class BaseKerasDenoiser:

    def __init__(self, models, offset=0., offset_out=True, **kwargs):

        self.models = models
        self.offset = offset
        self.offset_out = offset_out
        self.kwargs = kwargs


    def __call__(self, inp):

        list_of_outputs = []
        inp = inp[..., np.newaxis] + self.offset
        for model in self.models:
            out = model.predict(inp, **self.kwargs)
            out = out[..., 0]
            if self.offset_out:
                out -= self.offset
            list_of_outputs.append(out)

        return list_of_outputs


class KerasDenoiser(BaseKerasDenoiser):

    def __init__(self, model, **kwargs):
        super().__init__([model], **kwargs)


    def __call__(self, inp):

        out = super().__call__(inp)[0]

        # Projection onto the subspace orthogonal to the kernel of the
        # Kaiser-Squires operator
        out -= np.mean(out, axis=(-2, -1))[..., np.newaxis, np.newaxis]

        return out


class KerasDenoiserVar(BaseKerasDenoiser):

    def __init__(self, model, **kwargs):
        super().__init__([model], offset_out=False, **kwargs)

    def __call__(self, inp):
        return super().__call__(inp)[0]
