import os
import numpy as np
import matplotlib.pyplot as plt

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

            # Gradient-descent step
            # The convergence is real-valued, therefore the gradient is also real-valued
            kappa += self.step_size * wlutils.get_convergence_from_shear(
                resgamma, return_complex=True
            ).real
            for callback in callbacks:
                callback.on_forward_end(i, kappa)

            #########################################################################
            # Backward step
            #########################################################################
            kappa = self.backward(kappa)
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
            self._show(kappa, title='Forward')

    def on_backward_end(self, i, kappa):
        if (i+1) % self.showevery == 0:
            plt.subplot(122)
            self._show(kappa, title='Backward')
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


    def on_forward_end(self, _, kappa):
        self._rmse(kappa, self.rmse_forward)


    def on_backward_end(self, _, kappa):
        self._rmse(kappa, self.rmse_backward)


    def on_predict_begin(self, _):
        self._reset()


    def on_predict_end(self, _):
        self.rmse_forward = np.stack(self.rmse_forward)
        self.rmse_backward = np.stack(self.rmse_backward)
        if self.path_to_saved_stats is not None:
            rmse = np.stack([self.rmse_forward, self.rmse_backward])
            np.save(self.path_to_saved_stats, rmse)
