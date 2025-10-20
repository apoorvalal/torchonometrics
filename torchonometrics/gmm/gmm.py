from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import scipy
import torch
import torchmin


class GMMEstimator(ABC):
    """Abstract base class for GMM estimators."""

    def __new__(
        cls,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
        backend: str = "scipy",
    ):
        backend = backend.lower()
        estimator = _BACKENDS.get(backend)
        if estimator is None:
            raise ValueError(
                f"Backend {backend} is not supported. "
                f"Supported backends are: {list(_BACKENDS.keys())}"
            )
        return super(GMMEstimator, cls).__new__(estimator)

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
        backend: str = "scipy",
    ):
        self.moment_cond = moment_cond
        self.weighting_matrix = weighting_matrix

    @abstractmethod
    def gmm_objective(self, beta: np.ndarray) -> float:
        pass

    @abstractmethod
    def optimal_weighting_matrix(self, moments: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        verbose: bool = False,
        fit_method: Optional[str] = None,
        iid: bool = True,
    ) -> None:
        pass

    @abstractmethod
    def jacobian_moment_cond(self) -> np.ndarray:
        pass

    def summary(self, prec: int = 4, alpha: float = 0.05) -> pd.DataFrame:
        if not hasattr(self, "theta_") and not hasattr(self, "std_errors_"):
            raise ValueError(
                "Estimator not fitted yet. Make sure you call `fit()` before `summary()`."
            )
        return pd.DataFrame(
            {
                "coef": np.round(self.theta_, prec),
                "std err": np.round(self.std_errors_, prec),
                "t": np.round(self.theta_ / self.std_errors_, prec),
                "p-value": np.round(
                    2
                    * (
                        1 - scipy.stats.norm.cdf(np.abs(self.theta_ / self.std_errors_))
                    ),
                    prec,
                ),
                f"[{alpha / 2}": np.round(
                    self.theta_
                    - scipy.stats.norm.ppf(1 - alpha / 2) * self.std_errors_,
                    prec,
                ),
                f"{1 - alpha / 2}]": np.round(
                    self.theta_
                    + scipy.stats.norm.ppf(1 - alpha / 2) * self.std_errors_,
                    prec,
                ),
            }
        )


class GMMEstimatorScipy(GMMEstimator):
    """Class to create GMM estimator using scipy"""

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, np.ndarray] = "optimal",
        backend: str = "scipy",
    ):
        super().__init__(moment_cond, weighting_matrix, backend)
        self.z_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.x_: Optional[np.ndarray] = None
        self.n_: Optional[int] = None
        self.k_: Optional[int] = None
        self.W_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.Gamma_: Optional[np.ndarray] = None
        self.vtheta_: Optional[np.ndarray] = None
        self.std_errors_: Optional[np.ndarray] = None
        self.Omega_: Optional[np.ndarray] = None

    def gmm_objective(self, beta: np.ndarray) -> float:
        moments = self.moment_cond(self.z_, self.y_, self.x_, beta)
        if self.weighting_matrix == "optimal":
            if not hasattr(self, 'W_') or self.W_ is None:
                # Use identity matrix for first stage, then update
                self.W_ = np.eye(moments.shape[1])
        elif isinstance(self.weighting_matrix, np.ndarray):
            self.W_ = self.weighting_matrix
        else:
            self.W_ = np.eye(moments.shape[1])
        mavg = moments.mean(axis=0)
        return float(mavg.T @ self.W_ @ mavg)

    def optimal_weighting_matrix(self, moments: np.ndarray) -> np.ndarray:
        # Compute sample covariance matrix of moments: Omega = E[g_i g_i']
        moment_cov = np.cov(moments.T, ddof=1)
        # Handle numerical issues with small eigenvalues
        eigenvals, eigenvecs = np.linalg.eigh(moment_cov)
        eigenvals = np.maximum(eigenvals, 1e-12)  # Regularize small eigenvalues
        moment_cov_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        return np.linalg.inv(moment_cov_reg)

    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        verbose: bool = False,
        fit_method: Optional[str] = None,
        iid: bool = True,
        two_step: bool = True,
    ) -> None:
        if fit_method is None:
            fit_method = "L-BFGS-B"
        self.z_, self.y_, self.x_ = z, y, x
        self.n_, self.k_ = x.shape
        
        # First stage: use identity weighting matrix
        self.W_ = np.eye(self.z_.shape[1])  # Number of instruments
        result = scipy.optimize.minimize(
            self.gmm_objective,
            x0=np.random.rand(self.k_),
            method=fit_method,
            options={"disp": verbose},
        )
        theta_first = result.x
        
        # Two-step GMM if optimal weighting requested
        if self.weighting_matrix == "optimal" and two_step:
            # Compute optimal weighting matrix using first-stage residuals
            moments_first = self.moment_cond(self.z_, self.y_, self.x_, theta_first)
            self.W_ = self.optimal_weighting_matrix(moments_first)
            
            # Second stage optimization
            result = scipy.optimize.minimize(
                self.gmm_objective,
                x0=theta_first,
                method=fit_method,
                options={"disp": verbose},
            )
        
        self.theta_ = result.x
        
        # Compute standard errors
        try:
            moments_final = self.moment_cond(self.z_, self.y_, self.x_, self.theta_)
            self.Gamma_ = self.jacobian_moment_cond()
            
            # Compute robust covariance matrix
            if iid:
                # IID case: Omega = sigma^2 * I (for IV regression)
                self.Omega_ = np.cov(moments_final.T, ddof=1)
            else:
                # HAC-robust covariance
                self.Omega_ = self._compute_hac_covariance(moments_final)
            
            # Sandwich formula: (G'WG)^{-1} G'W Omega W G (G'WG)^{-1}
            GWG_inv = np.linalg.inv(self.Gamma_.T @ self.W_ @ self.Gamma_)
            if iid and np.allclose(self.W_, np.linalg.inv(self.Omega_), atol=1e-6):
                # Efficient case: W = Omega^{-1}
                self.vtheta_ = GWG_inv
            else:
                # General sandwich formula
                middle = self.Gamma_.T @ self.W_ @ self.Omega_ @ self.W_ @ self.Gamma_
                self.vtheta_ = GWG_inv @ middle @ GWG_inv
            
            self.std_errors_ = np.sqrt(np.diag(self.vtheta_) / self.n_)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute standard errors: {e}")
            self.std_errors_ = None

    def jacobian_moment_cond(self) -> np.ndarray:
        # For IV moment condition g(z,y,x,beta) = z * (y - x*beta)
        # Jacobian w.r.t. beta is -z'x / n
        self.jac_est_ = -self.z_.T @ self.x_ / self.n_
        return self.jac_est_
    
    def _compute_hac_covariance(self, moments: np.ndarray, max_lags: int = None) -> np.ndarray:
        """Compute HAC-robust covariance matrix using Newey-West estimator"""
        n, k = moments.shape
        if max_lags is None:
            max_lags = int(np.floor(4 * (n / 100) ** (2/9)))  # Rule of thumb
        
        # Center moments
        moments_centered = moments - moments.mean(axis=0)
        
        # Compute covariance matrix
        Omega = np.zeros((k, k))
        
        # Lag 0 (variance)
        Omega += moments_centered.T @ moments_centered / n
        
        # Higher order lags with Bartlett kernel
        for lag in range(1, max_lags + 1):
            weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
            gamma_lag = np.zeros((k, k))
            
            for t in range(lag, n):
                gamma_lag += np.outer(moments_centered[t], moments_centered[t - lag])
            
            gamma_lag /= n
            Omega += weight * (gamma_lag + gamma_lag.T)
        
        return Omega

    @staticmethod
    def iv_moment(
        z: np.ndarray, y: np.ndarray, x: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        return z * (y - x @ beta)[:, np.newaxis]


class GMMEstimatorTorch(GMMEstimator):
    """Class to create GMM estimator using torch"""

    def __init__(
        self,
        moment_cond: Callable,
        weighting_matrix: Union[str, torch.Tensor] = "optimal",
        backend: str = "torch",
    ):
        super().__init__(moment_cond, weighting_matrix, backend)
        self.z_: Optional[torch.Tensor] = None
        self.y_: Optional[torch.Tensor] = None
        self.x_: Optional[torch.Tensor] = None
        self.n_: Optional[int] = None
        self.k_: Optional[int] = None
        self.W_: Optional[torch.Tensor] = None
        self.theta_: Optional[torch.Tensor] = None
        self.Gamma_: Optional[np.ndarray] = None
        self.vtheta_: Optional[np.ndarray] = None
        self.std_errors_: Optional[np.ndarray] = None
        self.Omega_: Optional[np.ndarray] = None

    def gmm_objective(self, beta: torch.Tensor) -> torch.Tensor:
        moments = self.moment_cond(self.z_, self.y_, self.x_, beta)
        if self.weighting_matrix == "optimal":
            if not hasattr(self, 'W_') or self.W_ is None:
                # Use identity matrix for first stage
                self.W_ = torch.eye(moments.shape[1], dtype=moments.dtype, device=moments.device)
        elif isinstance(self.weighting_matrix, torch.Tensor):
            self.W_ = self.weighting_matrix.to(moments.device)
        else:
            self.W_ = torch.eye(moments.shape[1], dtype=moments.dtype, device=moments.device)
        
        mavg = moments.mean(dim=0)
        return torch.matmul(mavg.unsqueeze(0), torch.matmul(self.W_, mavg.unsqueeze(-1))).squeeze()

    def optimal_weighting_matrix(self, moments: torch.Tensor) -> torch.Tensor:
        # Convert to numpy for covariance computation, then back to torch
        moments_np = moments.detach().cpu().numpy()
        moment_cov = np.cov(moments_np.T, ddof=1)
        
        # Handle numerical issues
        eigenvals, eigenvecs = np.linalg.eigh(moment_cov)
        eigenvals = np.maximum(eigenvals, 1e-12)
        moment_cov_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Convert back to torch tensor on same device as input
        W_inv = torch.tensor(np.linalg.inv(moment_cov_reg), 
                           dtype=moments.dtype, device=moments.device)
        return W_inv

    def fit(
        self,
        z: np.ndarray,
        y: np.ndarray,
        x: np.ndarray,
        verbose: bool = False,
        fit_method: Optional[str] = None,
        iid: bool = True,
        two_step: bool = True,
        device: Optional[str] = None,
    ) -> None:
        if fit_method is None:
            fit_method = "l-bfgs"
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Convert to tensors
        self.z_ = torch.tensor(z, dtype=torch.float64, device=device)
        self.y_ = torch.tensor(y, dtype=torch.float64, device=device)
        self.x_ = torch.tensor(x, dtype=torch.float64, device=device)
        self.n_, self.k_ = x.shape
        
        # First stage: identity weighting
        self.W_ = torch.eye(self.z_.shape[1], dtype=torch.float64, device=device)
        beta_init = torch.tensor(
            np.random.rand(self.k_), dtype=torch.float64, device=device, requires_grad=True
        )
        
        result = torchmin.minimize(
            self.gmm_objective, beta_init, method=fit_method, tol=1e-5, disp=verbose
        )
        theta_first = result.x
        
        # Two-step GMM if optimal weighting requested
        if self.weighting_matrix == "optimal" and two_step:
            # Compute optimal weighting matrix using first-stage residuals
            moments_first = self.moment_cond(self.z_, self.y_, self.x_, theta_first)
            self.W_ = self.optimal_weighting_matrix(moments_first)
            
            # Second stage optimization
            result = torchmin.minimize(
                self.gmm_objective, theta_first, method=fit_method, tol=1e-5, disp=verbose
            )
        
        self.theta_ = result.x.detach().cpu().numpy()
        
        # Compute standard errors
        try:
            moments_final = self.moment_cond(self.z_, self.y_, self.x_, 
                                           torch.tensor(self.theta_, device=device))
            self.Gamma_ = self.jacobian_moment_cond()
            
            # Convert W to numpy for standard error computation
            W_np = self.W_.detach().cpu().numpy()
            
            # Compute robust covariance matrix
            moments_np = moments_final.detach().cpu().numpy()
            if iid:
                self.Omega_ = np.cov(moments_np.T, ddof=1)
            else:
                self.Omega_ = self._compute_hac_covariance(moments_np)
            
            # Sandwich formula
            GWG_inv = np.linalg.inv(self.Gamma_.T @ W_np @ self.Gamma_)
            if iid and np.allclose(W_np, np.linalg.inv(self.Omega_), atol=1e-6):
                self.vtheta_ = GWG_inv
            else:
                middle = self.Gamma_.T @ W_np @ self.Omega_ @ W_np @ self.Gamma_
                self.vtheta_ = GWG_inv @ middle @ GWG_inv
            
            self.std_errors_ = np.sqrt(np.diag(self.vtheta_) / self.n_)
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute standard errors: {e}")
            self.std_errors_ = None

    def jacobian_moment_cond(self) -> np.ndarray:
        # For IV moment condition g(z,y,x,beta) = z * (y - x*beta)
        # Jacobian w.r.t. beta is -z'x / n
        z_np = self.z_.detach().cpu().numpy()
        x_np = self.x_.detach().cpu().numpy()
        self.jac_est_ = -z_np.T @ x_np / self.n_
        return self.jac_est_
    
    def _compute_hac_covariance(self, moments: np.ndarray, max_lags: int = None) -> np.ndarray:
        """Compute HAC-robust covariance matrix using Newey-West estimator"""
        n, k = moments.shape
        if max_lags is None:
            max_lags = int(np.floor(4 * (n / 100) ** (2/9)))  # Rule of thumb
        
        # Center moments
        moments_centered = moments - moments.mean(axis=0)
        
        # Compute covariance matrix
        Omega = np.zeros((k, k))
        
        # Lag 0 (variance)
        Omega += moments_centered.T @ moments_centered / n
        
        # Higher order lags with Bartlett kernel
        for lag in range(1, max_lags + 1):
            weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
            gamma_lag = np.zeros((k, k))
            
            for t in range(lag, n):
                gamma_lag += np.outer(moments_centered[t], moments_centered[t - lag])
            
            gamma_lag /= n
            Omega += weight * (gamma_lag + gamma_lag.T)
        
        return Omega

    @staticmethod
    def iv_moment(
        z: torch.Tensor, y: torch.Tensor, x: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        return z * (y - x @ beta).unsqueeze(-1)


_BACKENDS = {
    "scipy": GMMEstimatorScipy,
    "torch": GMMEstimatorTorch,
}
