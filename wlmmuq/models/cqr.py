import warnings

from scipy import stats, optimize
import torch
from torch import nn

from .. import utils

class BaseCQR(nn.Module):
    """
    Base class for conformalized quantile regression.

    Attributes
    ----------
    alpha (float)
        Target error level

    """
    def __init__(self, alpha, map_size, in_channels=1):

        super().__init__()
        self.alpha = alpha
        self.map_size = map_size
        self.in_channels = in_channels

        self.adjusted_quantile = None
        self.calib_param = nn.Parameter(
            torch.zeros(in_channels, map_size, map_size),
            requires_grad=False
        )
        self.nimgs_calib = 0


    def reset(self):
        self.calib_param.copy_(
            torch.zeros_like(self.calib_param)
        )
        self.nimgs_calib = 0


    def _calibration_fun(self, res):
        raise NotImplementedError


    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        raise NotImplementedError


    def calibrate(
            self, pred_calib: torch.Tensor, res_calib: torch.Tensor | float,
            kappa_calib: torch.Tensor
    ):
        """
        Get calibration parameters and store them as non-trainable parameters
        of the model.

        Parameters
        ----------
        pred_calib, res_calib: torch.Tensor, shape = (nimgs, nx, ny)
            Estimated convergence maps and residuals (calibration set).
        kappa_calib: torch.Tensor, shape = (nimgs, nx, ny)
            Ground-truth convergence maps (calibration set).

        Returns
        -------
        quantile_vals: torch.Tensor, shape = (nx, ny)
            The per-pixel quantile values, used for estimating the calibration parameter.
        adjusted_quantiles: torch.Tensor, shape = (1,)
            Adjusted quantile index (between 0 and 1)

        """
        assert pred_calib.shape == res_calib.shape == kappa_calib.shape
        nimgs, in_channels, nx, ny = pred_calib.shape
        assert nimgs >= utils.get_min_nimgs_calib(self.alpha)
        assert in_channels == self.in_channels
        assert nx == ny == self.map_size

        with torch.no_grad():

            conformity_scores = self._conformity_scores(pred_calib, res_calib, kappa_calib)
            adjusted_quantile = (1 - self.alpha) * (1 + 1/nimgs) # For finite-sample correction
            quantile_vals = utils.quantile(conformity_scores, adjusted_quantile, axis=0)
            calib_param = (
                self.nimgs_calib * self.calib_param + nimgs * quantile_vals
            ) / (self.nimgs_calib + nimgs)

            self.calib_param.copy_(calib_param)
            self.nimgs_calib += nimgs

        return quantile_vals, adjusted_quantile


    def forward(self, res: torch.Tensor | float):
        """
        Perform conformal calibration.

        Parameters
        ----------
        res (torch.Tensor or float)
            Estimated residuals to be calibrated (test set), shape = (nimgs_test, nx, ny).

        Returns
        -------
        res_cqr (torch.Tensor)
            Calibrated residuals, shape = (nimgs_test, nx, ny).

        """
        with torch.no_grad():
            res_cqr = self._calibration_fun(res)
            res_cqr = res_cqr.reshape(
                -1, self.in_channels, self.map_size, self.map_size
            )
        return res_cqr


    def get_bounds_proba(self, nimgs_calib):
        lower_bound_proba = self.alpha - 1 / (nimgs_calib + 1)
        upper_bound_proba = self.alpha
        return lower_bound_proba, upper_bound_proba


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
    def _calibration_fun(self, res):
        return utils.maximum(res + self.calib_param, 0)

    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        return utils.absolute(pred_calib - kappa_calib) - res_calib


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
    def __init__(self, alpha, map_size, in_channels=1, eps=1e-9):
        super().__init__(alpha, map_size, in_channels=in_channels)
        self.eps = eps

    def _calibration_fun(self, res):
        if isinstance(res, float):
            res = self.eps if res <= self.eps else res
        else:
            res[res <= self.eps] = self.eps
        return self.calib_param * res

    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        if isinstance(res_calib, float):
            res_calib = self.eps if res_calib <= self.eps else res_calib
        else:
            res_calib[res_calib <= self.eps] = self.eps
        return utils.absolute(kappa_calib - pred_calib) / res_calib


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
    mask (torch.Tensor, default=None)
        When proper calibration is impossible (due to the calibration function),
        a warning is triggered. However, the warning will be ignored if this happens
        outside the survey boundaries, delimited by this attribute. The shape is (nx, ny).

    """
    def __init__(
            self, alpha, map_size, in_channels=1,
            eps=1e-9, mask: torch.Tensor=None
    ):
        super().__init__(alpha, map_size, in_channels=in_channels)
        self.eps = eps
        if mask is not None:
            utils.check_mask(mask)
        self.register_buffer('mask', mask)

    def _rho(self, res):
        raise NotImplementedError

    def _rho_nonzero(self, res):
        out = self._rho(res)
        if isinstance(out, float):
            out = self.eps if out <= self.eps else out
        else:
            out[out <= self.eps] = self.eps
        return out

    def _calibration_fun(self, res):
        return res + self._rho_nonzero(res) * (self.calib_param - 1)

    def _conformity_scores(self, pred_calib, res_calib, kappa_calib):
        weights_calib = self._rho_nonzero(res_calib)
        out = 1 + (
            utils.absolute(pred_calib - kappa_calib) - res_calib
        ) / weights_calib
        out[out < 0] = 0 # The calibration parameters must be positive
        return out

    def forward(self, res: torch.Tensor | float):
        iszero = self.calib_param == 0
        if self.mask is not None:
            iszero[self.mask] = False
        sum_iszero = iszero.sum()
        numel = utils.count_elts(iszero)
        if sum_iszero > 0:
            warnings.warn(
                f"Some pixels are impossible to calibrate ({sum_iszero / numel:.0%}); the "
                "predictions will be overconservative. Choose another calibration function."
            )
        return super().forward(res)


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
    mask (torch.Tensor, default=None)
        When proper calibration is impossible (due to the calibration function),
        a warning is triggered. However, the warning will be ignored if this happens
        outside the survey boundaries, delimited by this attribute. The shape is (nx, ny).

    """
    def __init__(
            self, alpha, map_size, in_channels=1,
            eps=1e-9, a=1., df=3, mask: torch.Tensor=None
    ):
        super().__init__(alpha, map_size, in_channels=in_channels, eps=eps, mask=mask)
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
