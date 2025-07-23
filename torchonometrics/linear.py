from typing import Dict, Optional, Union, List

import numpy as np
import torch

from .base import BaseEstimator
from .demean import demean_torch, prepare_fixed_effects


def _calculate_vcov_details(
    coef: torch.Tensor, X: torch.Tensor, y: torch.Tensor, se_type: str, n: int, k: int
):
    """Helper function to compute standard errors."""
    ε = y - X @ coef
    if se_type == "HC1":
        M = torch.einsum("ij,i,ik->jk", X, ε**2, X)
        XtX_inv = torch.linalg.inv(X.T @ X)
        Σ = XtX_inv @ M @ XtX_inv
        return torch.sqrt((n / (n - k)) * torch.diag(Σ))
    elif se_type == "classical":
        XtX_inv = torch.linalg.inv(X.T @ X)
        return torch.sqrt(torch.diag(XtX_inv) * torch.var(ε, correction=k))
    return None


class LinearRegression(BaseEstimator):
    """
    Linear regression model using PyTorch for efficient solving.

    This class provides a simple interface for fitting a linear regression
    model with support for fixed effects and various standard error types.

    Parameters
    ----------
    solver : str, default="torch"
        Solver to use. Options are "torch" (PyTorch's lstsq) or "numpy".

    Examples
    --------
    >>> import torch
    >>> from torchonometrics import LinearRegression
    >>>
    >>> # Basic regression
    >>> X = torch.randn(100, 5)
    >>> y = X @ torch.randn(5) + 0.1 * torch.randn(100)
    >>> model = LinearRegression()
    >>> model.fit(X, y)
    >>>
    >>> # With fixed effects
    >>> firm_ids = torch.randint(0, 10, (100,))
    >>> model.fit(X, y, fe=[firm_ids])
    """

    def __init__(self, solver="torch"):
        """Initialize the LinearRegression model.

        Args:
            solver (str, optional): Solver. Defaults to "torch", can also be "numpy".
        """
        super().__init__()
        self.solver: str = solver

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        se: str = None,
        fe: Optional[Union[List, torch.Tensor]] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> "LinearRegression":
        """
        Fit the linear model.

        Args:
            X: The design matrix of shape (n_samples, n_features).
            y: The target vector of shape (n_samples,).
            se: Whether to compute standard errors. "HC1" for robust standard errors, "classical" for classical SEs.
            fe: Fixed effects variables. Can be a list of arrays or a 2D array.
            weights: Sample weights of shape (n_samples,).

        Returns:
            The fitted estimator.
        """

        # Store original data for potential SE calculation
        X_orig, y_orig = X, y

        # Handle fixed effects demeaning
        if fe is not None:
            # Prepare fixed effects
            if isinstance(fe, list):
                flist = prepare_fixed_effects(fe)
            else:
                flist = torch.as_tensor(fe, dtype=torch.int64)
                if flist.ndim == 1:
                    flist = flist[:, None]

            # Demean both X and y
            X_demeaned, X_converged = demean_torch(X, flist, weights)
            y_demeaned, y_converged = demean_torch(y[:, None], flist, weights)
            y_demeaned = y_demeaned.flatten()

            if not (X_converged and y_converged):
                print("Warning: Demeaning did not converge")

            # Use demeaned data for regression
            X, y = X_demeaned, y_demeaned

        if self.solver == "torch":
            sol = torch.linalg.lstsq(X, y)
            self.params = {"coef": sol.solution}
        elif self.solver == "numpy":  # for completeness
            X_np, y_np = X.detach().cpu().numpy(), y.detach().cpu().numpy()
            sol = np.linalg.lstsq(X_np, y_np, rcond=None)
            self.params = {"coef": torch.from_numpy(sol[0]).to(X.device)}

        if se:
            self._vcov(
                y=y_orig if fe is not None else y,
                X=X_orig if fe is not None else X,
                se_type=se,
            )
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using the fitted model.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Predicted values of shape (n_samples,).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        return torch.matmul(X, self.params["coef"])

    def _vcov(
        self,
        y: torch.Tensor,
        X: torch.Tensor,
        se_type: str = "HC1",
    ) -> None:
        """Compute variance-covariance matrix and standard errors."""
        n, k = X.shape
        if self.params and "coef" in self.params:
            coef = self.params["coef"]
            se_values = _calculate_vcov_details(coef, X, y, se_type, n, k)
            if se_values is not None:
                self.params["se"] = se_values
        else:
            print("Coefficients not available for SE calculation.")
