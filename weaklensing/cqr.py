import warnings

import numpy as np
from scipy import stats, optimize

from . import utils

class BaseCQR:
    """
    Base class for conformalized quantile regression.

    Attributes
    ----------
    alpha (float)
        Target error level

    """
    def __init__(self, alpha):
        self.alpha = alpha


    def _calibration_fun(self, lamb, res):
        raise NotImplementedError


    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        raise NotImplementedError


    def _adjusted_quantiles(self, conformity_scores):
        """
        Get quantiles of a given array of conformity scores over a calibration
        set, with finite-sample correction.

        Parameters
        ----------
        conformity_scores (array-like)
            Array of shape (nimgs_calib, nx, ny), where nimgs_calib denotes
            the number of images in the calibration set
        
        Returns
        -------
        quantile_vals (array-like)
            Array of shape (ns, ny): the adjusted quantiles
        adjusted_quantile (float)
            Adjusted quantile index (between 0 and 1)
        
        """
        nimgs_calib = conformity_scores.shape[0]
        assert nimgs_calib >= utils.get_min_nimgs_calib(self.alpha)
        adjusted_quantile = (1 - self.alpha) * (1 + 1/nimgs_calib)
        quantile_vals = np.percentile(conformity_scores, adjusted_quantile*100, axis=0)
        return quantile_vals, adjusted_quantile


    def conformalize(self, res_test, pred_calib, res_calib, kappa_calib):
        """
        Perform conformal calibration.

        Parameters
        ----------
        res_test (array-like)
            Estimated residuals to be calibrated (test set), shape = (nimgs_test, nx, ny).
        pred_calib, res_calib (array-like)
            Estimated convergence maps and residuals (calibration set),
            shape = (nimgs_calib, nx, ny).
        kappa_calib (array-like)
            Ground-truth convergence maps (calibration set),
            shape = (nimgs_calib, nx, ny).
        
        """
        conformity_scores = self._conformity_scores(pred_calib, res_calib, kappa_calib)
        quantile_vals, adjusted_quantile = self._adjusted_quantiles(conformity_scores)
        res_cqr_test = self._calibration_fun(quantile_vals, res_test)

        return res_cqr_test, quantile_vals, adjusted_quantile


class AddCQR(BaseCQR):
    r"""
    Additive CQR, originally proposed by Y. Romano, E. Patterson, and E. Candes,
    “Conformalized Quantile Regression,” in NeurIPS, 2023.
    The calibration functions are defined by
    $$
        g_\lambda: r \mapsto \max(r + \lambda,\, 0).
    $$

    Attributes
    ----------
    alpha (float)
        Target error level
    
    """
    def _calibration_fun(self, lamb, res):
        return np.maximum(res + lamb, 0)

    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        return np.abs(pred_calib - kappa_calib) - res_calib


class MultCQR(BaseCQR):
    r"""
    Multiplicative CQR. The calibration functions are defined by
    $$
        g_\lambda: r \mapsto \lambda r,
    $$
    as used, in the context of RCPS, by
    A. N. Angelopoulos et al., “Image-to-Image Regression with Distribution-Free
    Uncertainty Quantification and Applications in Imaging,” in Proceedings of
    the 39th International Conference on Machine Learning, PMLR, Jun. 2022, pp. 717–730.

    Attributes
    ----------
    alpha (float)
        Target error level
    eps (float, default=1e-9)
        Small value to avoid division by 0 (in case of zero residual)

    """
    def __init__(self, alpha, eps=1e-9):
        super().__init__(alpha)
        self.eps = eps

    def _calibration_fun(self, lamb, res):
        res[res <= self.eps] = self.eps
        return lamb * res

    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        res_calib[res_calib <= self.eps] = self.eps
        return np.abs(kappa_calib - pred_calib) / res_calib


class GenCQR(BaseCQR):
    r"""
    Base class for CQR with used-defined calibration functions, in the form
    $$
        g_\lambda: r \mapsto r + \rho(r) (\lambda - 1),
    $$
    for some user-specified function $\rho$, to be implemented as a method `_rho`.
    In this context, the conformity scores are equal to:
    $$
        \max\left(
            0,\, \lambda_i = 1 + \frac{
                \left|
                    \hat f(x_i) - y_i
                \right| - \hat r(x_i)
            }{
                \rho\left(
                    \hat r(x_i)
                \right)
            }
        \right).
    $$

    Attributes
    ----------
    alpha (float)
        Target error level
    eps (float, default=1e-9)
        Small value to avoid division by 0 (in case of zero residual)
    mask (array-like, default=None)
        When proper calibration is impossible (due to the calibration function),
        a warning is triggered. However, the warning will be ignored if this happens
        outside the survey boundaries, delimited by this attribute. The shape is (nx, ny).

    """
    def __init__(
            self, alpha, eps=1e-9, mask=None
    ):
        super().__init__(alpha)
        self.eps = eps
        self.mask = mask

    def _rho(self, res):
        raise NotImplementedError

    def _rho_nonzero(self, res):
        out = self._rho(res)
        out[out <= self.eps] = self.eps
        return out

    def _calibration_fun(self, lamb, res):
        return res + self._rho_nonzero(res) * (lamb - 1)

    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        weights_calib = self._rho_nonzero(res_calib)
        out = 1 + (
            np.abs(pred_calib - kappa_calib) - res_calib
        ) / weights_calib
        out[out < 0] = 0 # The calibration parameters must be positive
        return out

    def conformalize(self, res_test, pred_calib, res_calib, kappa_calib):
        res_cqr_test, quantile_vals, adjusted_quantile = super().conformalize(
            res_test, pred_calib, res_calib, kappa_calib
        )
        iszero = quantile_vals == 0
        if self.mask is not None:
            iszero[self.mask] = False
        sum_iszero = np.sum(iszero)
        if sum_iszero > 0:
            warnings.warn(
                f"Some pixels are impossible to calibrate ({sum_iszero / iszero.size:.0%}); the "
                "predictions will be overconservative. Choose another calibration function."
            )
        return res_cqr_test, quantile_vals, adjusted_quantile


class ChisqCQR(GenCQR):
    r"""
    CQR with chi-squared-based calibration functions, in the form
    $$
        g_\lambda: r \mapsto r + b F_{\chi^2_k}(r / a) (\lambda - 1),
    $$
    where $F_{\chi^2(k)}$ denotes the cumulative distribution function of a
    chi-squared distribution with $k$ degrees of freedom, and $a$ and $b$ denote
    positive real numbers. The former is user-defined, whereas the latter is set to
    the highest value such that $g_\lambda$ remains non-descending for all
    $\lambda \geq 0$.

    Attributes
    ----------
    alpha (float)
        Target error level
    eps (float, default=1e-9)
        Small value to avoid division by 0 (in case of zero residual)
    a (float, default=1.)
        Scaling factor
    df (int, default=3)
        Number of degrees of freedom
    mask (array-like, default=None)
        When proper calibration is impossible (due to the calibration function),
        a warning is triggered. However, the warning will be ignored if this happens
        outside the survey boundaries, delimited by this attribute. The shape is (nx, ny).

    """
    def __init__(
            self, alpha, eps=1e-9, a=1., df=3, mask=None
    ):
        super().__init__(alpha, eps=eps, mask=mask)
        self.a = a
        self.df = df

    @property
    def b(self):
        neg_chi2_pdf = lambda x: -stats.chi2.pdf(x, self.df)
        opt = optimize.minimize_scalar(neg_chi2_pdf)
        max_pdf = -opt.fun
        max_b = self.a / max_pdf
        return max_b

    def _rho(self, res):
        return self.b * stats.chi2.cdf(res / self.a, self.df)
