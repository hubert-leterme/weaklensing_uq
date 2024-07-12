"""
Code adapted from https://github.com/aangelopoulos/im2im-uq.git

"""
import numpy as np
from scipy import stats, optimize

from . import utils


class BaseRCPS:
    r"""
    Base class for calibration based on risk-controlling prediction sets.
    From A. N. Angelopoulos et al., “Image-to-Image Regression with Distribution-Free Uncertainty
    Quantification and Applications in Imaging,” in ICML, 2022.

    Attributes
    ----------
    alpha, delta (float)
        Target risk and error levels, such that $\mathbb{P}\{R(\lambda) > \alpha\} \leq \delta$,
        where $\lambda$ denotes the calibration parameter.
    nimgs_calib (int)
        Number of calibration examples.
    eps (float, default=1e-9)
        Small value to avoid division by 0 (in case of zero residual).
    maxiters (int, default=1000)
        Maximum number of iterations to estimate Hoeffding's upper confidence bound.

    """
    def __init__(
            self, alpha, delta, nimgs_calib, eps=1e-9, maxiters=1000
    ):
        self.alpha = alpha
        self.delta = delta
        self.nimgs_calib = nimgs_calib
        self.eps = eps
        self.maxiters = maxiters

        self.lower_lamb = self.LOWER_LAMB
        self.upper_lamb = self.UPPER_LAMB

    # TO BE OVERRIDDEN ###########################################
    def _calibration_fun(self, lamb, res):
        raise NotImplementedError

    LOWER_LAMB = NotImplemented
    UPPER_LAMB = NotImplemented
    ##############################################################


    ### UCB of mean via Hoeffding-Bentkus hybridization
    def _h1(self, y, mu):
        return y * np.log(y / mu) + (1 - y) * np.log((1 - y) / (1 - mu))


    ### Log tail inequalities of mean
    def _hoeffding_plus(self, mu, miscovrate, nimgs_calib=None):

        if nimgs_calib is None:
            nimgs_calib = self.nimgs_calib

        return -nimgs_calib * self._h1(np.minimum(mu, miscovrate), mu)


    def _bentkus_plus(self, mu, miscovrate, nimgs_calib=None):

        if nimgs_calib is None:
            nimgs_calib = self.nimgs_calib
        out = np.log(
            max(
                stats.binom.cdf(
                    np.floor(nimgs_calib * miscovrate), nimgs_calib, mu
                ), self.eps
            )) + 1

        return out


    def _hoeffding_upperbound(self, miscovrate, delta=None, **kwargs):

        if delta is None:
            delta = self.delta

        def _tailprob(mu):
            hoeffding_mu = self._hoeffding_plus(mu, miscovrate, **kwargs)
            bentkus_mu = self._bentkus_plus(mu, miscovrate, **kwargs)
            return min(hoeffding_mu, bentkus_mu) - np.log(delta)

        miscovrate = max(miscovrate, self.eps) # line added to avoid divisions by zero
        miscovrate = min(miscovrate, 1 - self.eps)

        if _tailprob(1 - self.eps) > 0:
            raise ValueError

        try:
            out = optimize.brentq(_tailprob, miscovrate, 1 - self.eps, maxiter=self.maxiters)
        except ValueError:
            out = 1 - self.eps

        return out


    def get_min_alpha(self, **kwargs):
        """
        Get the minimum admissible value of alpha, for a given delta.

        Parameters
        ----------
        delta (float, default=None)
            Desired error level. If none is given, set to self.delta.
        
        """
        return self._hoeffding_upperbound(0., **kwargs)


    def get_min_delta(self):
        """
        Get the minimum value of delta such that the corresponding minimum
        admissible value of alpha remains below self.alpha.
        
        """
        def _diff_alpha(delta):
            # Positive if delta is too small
            return self.get_min_alpha(delta=delta) - self.alpha

        lower_delta = self.eps
        upper_delta = 1.
        if _diff_alpha(upper_delta) > 0:
            raise ValueError("No admissible value for delta")
        if _diff_alpha(lower_delta) <= 0:
            out = lower_delta
        else:
            # _diff_alpha(lower_delta) and _diff_alpha(upper_delta) have opposite signs
            out = optimize.brentq(
                _diff_alpha, lower_delta, upper_delta, maxiter=self.maxiters
            )

        return out


    def get_min_miscovrate(self):
        return self.eps


    def get_max_miscovrate(self):
        raise NotImplementedError


    def get_min_nimgs_calib(self):
        """
        Get the minimum number of calibration images such that self.alpha and self.delta
        are mutually compatible.
        
        """
        def _diff_alpha(nimgs_calib):
            # Positive if nimgs_calib is too small
            return self.get_min_alpha(nimgs_calib=nimgs_calib) - self.alpha

        lower_nimgs = 1
        upper_nimgs = 1e9
        if _diff_alpha(upper_nimgs) > 0:
            raise ValueError("No admissible value for nimgs_calib")
        if _diff_alpha(lower_nimgs) <= 0:
            out = lower_nimgs
        else:
            # _diff_alpha(lower_nimgs) and _diff_alpha(upper_nimgs) have opposite signs
            out = optimize.brentq(
                _diff_alpha, lower_nimgs, upper_nimgs, maxiter=self.maxiters
            )
            out = int(np.ceil(out))

        return out


    def calibrate(self, res_test, pred_calib, res_calib, kappa_calib, **kwargs):
        """
        Perform RCPS-based calibration.

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

        Returns
        -------
        res_rcps_test (array-like)
            Calibrated residual (test set), shape = (nimgs_test, nx, ny).
        lamb (float)
            Optimal calibration parameter.
        
        """
        nimgs_calib, _, _ = utils.test_array_shape([
            pred_calib, res_calib, kappa_calib
        ])
        assert nimgs_calib == self.nimgs_calib

        # Check whether there is an admissible solution
        min_alpha = self.get_min_alpha()
        if self.alpha < min_alpha:
            min_nimgs = self.get_min_nimgs_calib()
            msg = (
                f"The solution cannot be calibrated.\n"
                f"Set alpha to at least {min_alpha:.1e}, or "
                f"increase the size of the calibration to at least {min_nimgs}"
            )
            try:
                min_delta = self.get_min_delta()
            except ValueError:
                msg += "."
            else:
                msg += (
                    f", or set delta to at least {min_delta:.1e}."
                )
            raise ValueError(msg)

        def _diff_alpha(lamb):
            res_rcps_calib = self._calibration_fun(lamb, res_calib)
            losses = utils.miscoverage_rate(
                pred_calib - res_rcps_calib, pred_calib + res_rcps_calib,
                kappa_calib, **kwargs
            ) # shape = (nimgs_calib,)
            miscovrate = losses.mean()
            miscovrate_hoeff = self._hoeffding_upperbound(miscovrate)
            return miscovrate_hoeff - self.alpha

        if _diff_alpha(self.upper_lamb) > 0:
            raise ValueError("Calibration impossible: underconfident.")
        if _diff_alpha(self.lower_lamb) < 0:
            raise ValueError("Calibration impossible: overconfident.")
        lamb = optimize.brentq(
            _diff_alpha, self.upper_lamb, self.lower_lamb, maxiter=self.maxiters
        )

        res_rcps_test = self._calibration_fun(lamb, res_test)

        return res_rcps_test, lamb


class AddRCPS(BaseRCPS):
    r"""
    Additive RCPS. The calibration functions are defined by
    $$
        g_\lambda: r \mapsto \max(r + \lambda,\, 0),
    $$
    as used, in the context of CQR, by
    Y. Romano, E. Patterson, and E. Candes, “Conformalized Quantile Regression,”
    in NeurIPS, 2023.

    """
    def _calibration_fun(self, lamb, res):
        return np.maximum(res + lamb, 0)

    LOWER_LAMB = -1.
    UPPER_LAMB = 1.


class MultRCPS(BaseRCPS):
    r"""
    Multiplicative RCPS, originally proposed by A. N. Angelopoulos et al.,
    “Image-to-Image Regression with Distribution-Free Uncertainty Quantification
    and Applications in Imaging,” in ICML, 2022. The calibration functions are defined by
    $$
        g_\lambda: r \mapsto \lambda r.
    $$

    """
    def _calibration_fun(self, lamb, res):
        return lamb * res

    LOWER_LAMB = 0.
    UPPER_LAMB = 10.
