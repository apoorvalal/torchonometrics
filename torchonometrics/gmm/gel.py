import numpy as np
from scipy.optimize import minimize
from typing import Callable, Optional
import logging
from scipy.linalg import inv, pinv
from scipy.stats import chi2


# Tilt functions for GEL
def rho_exponential(v: np.ndarray) -> np.ndarray:
    """Exponential tilting (ET): rho(v) = 1 - exp(v)"""
    return 1 - np.exp(v)


def rho_cue(v: np.ndarray) -> np.ndarray:
    """Continuously Updated Estimator (CUE): rho(v) = -0.5*v^2 - v"""
    return -0.5 * v**2 - v


def rho_el(v: np.ndarray) -> np.ndarray:
    """Empirical Likelihood (EL): rho(v) = log(1-v)
    Note: requires v < 1 for all observations
    """
    # Add small epsilon to avoid log(0)
    v_safe = np.clip(v, -np.inf, 1 - 1e-10)
    return np.log(1 - v_safe)


class GELEstimator:
    """
    Class for Generalized empirical likelihood estimation for vector-valued problems.
    """

    def __init__(
        self,
        m: Callable[[np.ndarray, np.ndarray], np.ndarray],
        rho: Callable[[np.ndarray], np.ndarray] = rho_exponential,
        min_method: str = "L-BFGS-B",
        verbose: bool = False,
        log: bool = False,
    ):
        self.m = m
        self.rho = rho
        self.rho_prime = self._get_rho_derivative(rho)
        self.rho_double_prime = self._get_rho_second_derivative(rho)
        self._min_method = min_method
        self._verbose = verbose
        self.est: Optional[np.ndarray] = None
        self.lam_hat: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
        self.se: Optional[np.ndarray] = None
        self.J_stat: Optional[float] = None
        self.J_pvalue: Optional[float] = None

        if log:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def fit(
        self,
        D: np.ndarray,
        startval: np.ndarray,
        startval2: Optional[np.ndarray] = None,
    ) -> None:
        """Fit GEL estimator with proper asymptotic standard errors"""
        if startval2 is None:
            startval2 = np.zeros(self.m(D, startval).shape[1])  # Start lambda at zero

        self.D_ = D
        self.n_ = D.shape[0]

        # Outer maximization
        result = minimize(
            lambda theta: self._outer_maximisation(theta, D, startval2),
            startval,
            method=self._min_method,
            options={"disp": self._verbose},
        )

        self.est = result.x

        # Get optimal lambda for final theta
        lam_result = minimize(
            self._inner_minimisation,
            startval2,
            args=(self.est, D),
            method=self._min_method,
            options={"disp": False},
        )
        self.lam_hat = lam_result.x

        # Compute proper asymptotic standard errors
        self._compute_asymptotic_covariance()

        # Compute J-test statistic
        self._compute_j_test()

    def summary(self, alpha: float = 0.05) -> dict:
        """Summary table with test statistics"""
        if self.est is None or self.se is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        from scipy.stats import norm

        t_stats = self.est / self.se
        p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

        critical_val = norm.ppf(1 - alpha / 2)
        ci_lower = self.est - critical_val * self.se
        ci_upper = self.est + critical_val * self.se

        summary_dict = {
            "coefficients": self.est,
            "std_errors": self.se,
            "t_statistics": t_stats,
            "p_values": p_values,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "J_statistic": self.J_stat,
            "J_pvalue": self.J_pvalue,
            "n_obs": self.n_,
        }

        return summary_dict

    def _outer_maximisation(
        self, theta: np.ndarray, D: np.ndarray, startval2: np.ndarray
    ) -> float:
        result = minimize(
            self._inner_minimisation,
            startval2,
            args=(theta, D),
            method=self._min_method,
            options={"disp": False},  # Suppress inner loop output
        )
        return -result.fun

    def _inner_minimisation(
        self, lam: np.ndarray, theta: np.ndarray, D: np.ndarray
    ) -> float:
        moments = self.m(D, theta)  # Moment conditions (n x k)
        tilts = np.dot(moments, lam)  # (n,)
        obj_value = -np.sum(self.rho(tilts))
        logging.info(f"Inner minimisation: lam={lam}, Objective value: {obj_value}")
        return obj_value

    def _get_rho_derivative(self, rho_func):
        """Return derivative of rho function"""
        if rho_func == rho_exponential:
            return lambda v: -np.exp(v)
        elif rho_func == rho_cue:
            return lambda v: -v - 1
        elif rho_func == rho_el:
            return lambda v: -1 / (1 - v)
        else:
            raise ValueError("Unknown rho function")

    def _get_rho_second_derivative(self, rho_func):
        """Return second derivative of rho function"""
        if rho_func == rho_exponential:
            return lambda v: -np.exp(v)
        elif rho_func == rho_cue:
            return lambda v: -np.ones_like(v)
        elif rho_func == rho_el:
            return lambda v: -1 / (1 - v) ** 2
        else:
            raise ValueError("Unknown rho function")

    def _compute_asymptotic_covariance(self):
        """Compute asymptotic covariance matrix using GEL theory"""
        moments = self.m(self.D_, self.est)  # n x q
        n, q = moments.shape
        p = len(self.est)  # number of parameters

        # Compute tilts and weights
        tilts = np.dot(moments, self.lam_hat)  # n x 1
        rho_prime_vals = self.rho_prime(tilts)  # n x 1
        rho_double_prime_vals = self.rho_double_prime(tilts)  # n x 1

        # Gradient of moment conditions w.r.t. theta
        # Use numerical differentiation if analytical not available
        eps = 1e-8
        G = np.zeros((q, p))
        for j in range(p):
            theta_plus = self.est.copy()
            theta_minus = self.est.copy()
            theta_plus[j] += eps
            theta_minus[j] -= eps

            moments_plus = self.m(self.D_, theta_plus)
            moments_minus = self.m(self.D_, theta_minus)

            G[:, j] = (moments_plus - moments_minus).mean(axis=0) / (2 * eps)

        # Compute blocks of the Hessian
        try:
            # H_λλ: second derivative w.r.t. λ
            weighted_moments = moments * rho_double_prime_vals[:, np.newaxis]
            H_lam_lam = weighted_moments.T @ moments / n

            # H_θλ: cross derivative
            H_theta_lam = G

            # Inverse of H_λλ (regularized if needed)
            try:
                H_lam_lam_inv = inv(H_lam_lam)
            except np.linalg.LinAlgError:
                H_lam_lam_inv = pinv(H_lam_lam)

            # Asymptotic variance: (H_θλ H_λλ^{-1} H_λθ)^{-1}
            V_theta = inv(H_theta_lam @ H_lam_lam_inv @ H_theta_lam.T)

            self.Sigma = V_theta / n
            self.se = np.sqrt(np.diag(self.Sigma))

        except (np.linalg.LinAlgError, ValueError) as e:
            logging.warning(f"Could not compute asymptotic covariance: {e}")
            # Fallback to simple covariance
            self.Sigma = np.cov(moments.T) / n
            self.se = np.sqrt(np.diag(self.Sigma))

    def _compute_j_test(self):
        """Compute J-test for overidentifying restrictions"""
        moments = self.m(self.D_, self.est)
        n, q = moments.shape
        p = len(self.est)

        if q <= p:
            # Just identified or under-identified
            self.J_stat = None
            self.J_pvalue = None
            return

        # J-statistic: n * objective function value at optimum
        moment_avg = moments.mean(axis=0)
        tilts = np.dot(moments, self.lam_hat)

        # GEL J-statistic
        self.J_stat = 2 * n * np.sum(self.rho(tilts)) / n

        # Under null, J ~ chi2(q-p)
        df = q - p
        self.J_pvalue = 1 - chi2.cdf(self.J_stat, df)
