import os
import numpy as np

from . import utils as wlutils

class PGDMassMapping:
    """
    Class for iterative proximal gradient descent (or forward-backward)
    algorithm, applied to the mass mapping problem.

    """
    def __init__(
            self, var_noise, step_size, backward, niter, mask=None,
            path_to_saved_arrays=None
    ):
        """
        Parameters
        ----------
        var_noise: np.ndarray, shape = (nx, ny)
            Array representing the noise variance for each pixel
            (diagonal elements of the noise covariance matrix).
        step_size: float
            The step size of the PGD algorithm. It should be smaller
            than 2*var_min, where var_min denotes the smallest value of
            var_noise outside the mask. This is due to the convergence
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
        path_to_saved_arrays: str, default = None
            If specified, intermediate arrays before and after the backward
            pass will be saved as a .npy file.
        
        """
        self.var_noise = var_noise
        self.step_size = step_size
        self.backward = backward
        self.niter = niter
        if mask is not None:
            self.mask = mask
        else:
            self.mask = var_noise != 0
        self.path_to_saved_arrays = path_to_saved_arrays


    def __call__(self, gamma):

        deltagamma = np.zeros(gamma.shape, dtype=complex) # Shape = ([nimgs], nx, ny)
        kappa = np.zeros(gamma.shape) # Shape = ([nimgs], nx, ny)

        for i in range(self.niter):
            print(f'Iteration {i+1}')

            #########################################################################
            # Forward step
            #########################################################################
            deltagamma = wlutils.get_shear_from_convergence(
                kappa, return_complex=True
            )
            deltagamma = (gamma - deltagamma) / self.var_noise

            # In masked pixels, the noise variance is assumed infinite, and therefore the
            # intermediate array is set to 0.
            deltagamma[..., ~self.mask] = 0.

            # Gradient-descent step
            # The convergence is real-valued, therefore the gradient is also real-valued
            kappa += self.step_size * wlutils.get_convergence_from_shear(
                deltagamma, return_complex=True
            ).real

            if self.path_to_saved_arrays is not None:
                np.save(
                    os.path.join(
                        self.path_to_saved_arrays, f'forward/kappa_{i+1}.npy'
                    ), kappa
                )

            #########################################################################
            # Backward step (proximal opertator replaced by a deep denoiser)
            #########################################################################
            kappa = self.backward(kappa)
            if self.path_to_saved_arrays is not None:
                np.save(
                    os.path.join(
                        self.path_to_saved_arrays, f'backward/kappa_{i+1}.npy'
                    ), kappa
                )

        return kappa
