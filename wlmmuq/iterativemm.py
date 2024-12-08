import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import pycs.astro.wl.mass_mapping as csmm

from . import utils as wlutils

class PGDMassMapping:
    """
    Class for iterative proximal gradient descent (or forward-backward)
    algorithm, applied to the mass mapping problem.

    """
    def __init__(
            self, std_noise, step_size, backward, niter, mask=None, verbose=False
    ):
        """
        Parameters
        ----------
        std_noise: np.ndarray, shape = (nx, ny)
            Array representing the noise standard deviation for each pixel
            (diagonal elements of the noise covariance matrix).
        step_size: float
            The step size of the PGD algorithm. It should be smaller
            than 2*(std_min**2), where std_min denotes the smallest value of
            std_noise outside the mask. This is due to the convergence
            result established by Combettes and Wajs (2005).
        backward: callable
            The proximal operator, or a trained denoiser for plug-and-play PGD.
            The denoiser should be trained on images corrupted by a white
            Gaussian noise with zero mean and variance equal to step_size.
        niter: int
            Number of iterations.
        mask: np.ndarray, shape = (nx, ny), default = None
            Mask to apply in case of missing data. In practice, the noise
            covariance matrix is set to infinity in the masked regions.
        verbose: bool, default=False
        
        """
        self.std_noise = std_noise
        self.step_size = step_size
        self.backward = backward
        self.niter = niter
        if mask is not None:
            self.mask = mask
        else:
            self.mask = std_noise != 0
        self.verbose = verbose


    def __call__(self, gamma, kappa0=None, callbacks=None):

        if callbacks is None:
            callbacks = []
        var_noise = self.std_noise**2
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

            #########################################################################
            # Forward step
            #########################################################################
            resgamma = gamma - wlutils.get_shear_from_convergence(
                kappa, return_complex=True
            )
            resgamma /= var_noise

            # In masked pixels, the noise variance is assumed infinite, and therefore the
            # intermediate array is set to 0.
            resgamma[..., ~self.mask] = 0.

            # Compute the negative-gradient, projected onto the subspace of real-valued
            # arrays orthogonal to the kernel of the Kaiser-Squires operator
            neg_grad = wlutils.get_convergence_from_shear(
                resgamma, return_complex=True
            ).real
            neg_grad -= np.mean(neg_grad, axis=(-2, -1))[..., np.newaxis, np.newaxis]

            # Gradient-descent step
            # The convergence is real-valued, therefore the gradient is also real-valued
            kappa += self.step_size * neg_grad

            for callback in callbacks:
                callback.on_forward_end(i, kappa)

            #########################################################################
            # Backward step
            #########################################################################

            # Backward operator (e.g., deep denoiser for PnP)
            kappa = self.backward(kappa)

            # Projection onto the subspace orthogonal to the kernel of the
            # Kaiser-Squires operator
            kappa -= np.mean(kappa, axis=(-2, -1))[..., np.newaxis, np.newaxis]

            for callback in callbacks:
                callback.on_backward_end(i, kappa)

        for callback in callbacks:
            callback.on_predict_end(kappa)

        return kappa


class Callback:

    def on_forward_end(self, i, kappa):
        pass

    def on_backward_end(self, i, kappa):
        pass

    def on_predict_begin(self, kappa):
        pass

    def on_predict_end(self, kappa):
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

    def __init__(self, idx, showevery=1, figsize=(8, 3), **kwargs):
        self.idx = idx
        self.showevery = showevery
        self.figsize = figsize
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

        self.rmse_forward = None
        self.rmse_backward = None
        self._reset()


    def _reset(self):
        self.rmse_forward = []
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


class KerasDenoiser:

    def __init__(self, model, offset=0., **kwargs):
        self.model = model
        self.offset = offset
        self.kwargs = kwargs

    def __call__(self, inp):
        inp = inp[..., np.newaxis] + self.offset 
        out = self.model.predict(inp, **self.kwargs)
        out = out[..., 0] - self.offset
        return out


class PinballLoss(keras.losses.Loss):

    def __init__(
            self, quantile=0.5, reduction=tf.keras.losses.Reduction.AUTO,
            name="pinball_loss"
    ):
        """
        Initialize the Pinball Loss.
        :param quantile: The desired quantile (e.g., 0.5 for median).
        :param name: Optional name for the loss instance.
        """
        super().__init__(reduction=reduction, name=name)
        self.quantile = quantile


    def call(self, y_true, y_pred):
        """
        Compute the Pinball Loss.
        :param y_true: Ground truth values.
        :param y_pred: Predicted values.
        :return: Computed loss.
        """
        error = y_true - y_pred
        loss = tf.maximum(self.quantile * error, (self.quantile - 1) * error)
        return tf.reduce_mean(loss)

    def get_config(self):
        """
        Serialize the loss configuration for saving and loading.
        """
        config = super().get_config()
        config.update({
            "quantile": self.quantile,
        })
        return config
