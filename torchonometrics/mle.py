from abc import abstractmethod
from typing import Dict, Optional

import torch
import numpy as np
from scipy import stats

from .base import BaseEstimator


class MaximumLikelihoodEstimator(BaseEstimator):
    """
    Base class for Maximum Likelihood Estimators using PyTorch optimizers.
    """

    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        maxiter: int = 5000,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.optimizer_class = optimizer if optimizer is not None else torch.optim.LBFGS
        self.maxiter = maxiter
        self.tol = tol
        self.params: Dict[str, torch.Tensor] = {}  # Initialize params
        self.history: Dict[str, list] = {"loss": []}  # To store loss history
        self._fitted_X: Optional[torch.Tensor] = None
        self._fitted_y: Optional[torch.Tensor] = None

    @abstractmethod
    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        Computes the negative log-likelihood for the model.
        Must be implemented by subclasses.
        Args:
            params: Model parameters.
            X: Design matrix.
            y: Target vector.
        Returns:
            Negative log-likelihood value.
        """
        raise NotImplementedError

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        init_params: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> "MaximumLikelihoodEstimator":
        """
        Fit the model using the specified PyTorch optimizer.

        Args:
            X: Design matrix of shape (n_samples, n_features).
               It's assumed that X includes an intercept column if one is desired.
            y: Target vector of shape (n_samples,).
            init_params: Optional initial parameters. If None, defaults to zeros
                         or small random numbers if a PRNGKey can be obtained.

        Returns:
            The fitted estimator.
        """
        n_features = X.shape[1]
        if init_params is None:
            # Initialize with small random values for better convergence
            torch.manual_seed(0)  # For reproducibility
            init_params_val = (
                torch.randn(n_features, device=X.device, dtype=X.dtype) * 0.01
            )
        else:
            init_params_val = init_params.to(X.device)

        # Set up parameters for optimization
        current_params = init_params_val.clone().requires_grad_(True)

        # Initialize optimizer
        if self.optimizer_class == torch.optim.LBFGS:
            optimizer = self.optimizer_class([current_params], max_iter=20)
        else:
            optimizer = self.optimizer_class([current_params])

        self.history["loss"] = []  # Reset loss history

        # Optimization loop
        for i in range(self.maxiter):

            def closure():
                optimizer.zero_grad()
                loss = self._negative_log_likelihood(current_params, X, y)
                loss.backward()
                return loss

            if self.optimizer_class == torch.optim.LBFGS:
                loss_val = optimizer.step(closure)
            else:
                loss_val = closure()
                optimizer.step()

            self.history["loss"].append(loss_val.item())

            # Check convergence
            if i > 10 and self.tol > 0:
                loss_change = abs(
                    self.history["loss"][-2] - self.history["loss"][-1]
                ) / (abs(self.history["loss"][-2]) + 1e-8)
                if loss_change < self.tol:
                    if verbose:
                        print(f"Convergence tolerance {self.tol} met at iteration {i}.")
                    break

        # Store final parameters and compute standard errors
        self.params = {"coef": current_params.detach()}
        self.iterations_run = i + 1  # Store how many iterations actually ran
        
        # Store fitted data for computing standard errors
        self._fitted_X = X.detach()
        self._fitted_y = y.detach()
        
        # Compute standard errors using Fisher information
        self._compute_standard_errors()
        
        return self

    def _compute_standard_errors(self) -> None:
        """Compute standard errors using the Fisher information matrix."""
        if self._fitted_X is None or self._fitted_y is None:
            return
            
        try:
            # Compute Fisher information matrix (Hessian of negative log-likelihood)
            fisher_info = self._compute_fisher_information(self.params["coef"], 
                                                         self._fitted_X, 
                                                         self._fitted_y)
            
            # Standard errors are sqrt of diagonal of inverse Fisher information
            fisher_inv = torch.linalg.inv(fisher_info)
            self.params["se"] = torch.sqrt(torch.diag(fisher_inv))
            self.params["vcov"] = fisher_inv
            
        except Exception as e:
            # If Fisher information computation fails, set standard errors to None
            print(f"Warning: Could not compute standard errors: {e}")
            self.params["se"] = None
            self.params["vcov"] = None
    
    @abstractmethod
    def _compute_fisher_information(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Fisher information matrix. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def summary(self, alpha: float = 0.05) -> None:
        """Print a summary of the model results with statistical inference."""
        if not self.params or "coef" not in self.params:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 50)
        print(f"Optimizer: {self.optimizer_class.__name__}")
        if hasattr(self, "iterations_run"):
            print(f"Optimization: {self.iterations_run}/{self.maxiter} iterations")
        if self.history["loss"]:
            print(f"Final Log-Likelihood: {-self.history['loss'][-1]:.4f}")
        
        n_obs = self._fitted_X.shape[0] if self._fitted_X is not None else "Unknown"
        print(f"No. Observations: {n_obs}")
        print("\n" + "=" * 50)
        
        # Coefficient table
        coef = self.params["coef"].detach().cpu().numpy()
        
        if self.params.get("se") is not None:
            se = self.params["se"].detach().cpu().numpy()
            t_stats = coef / se
            p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
            
            # Confidence intervals
            critical_val = stats.norm.ppf(1 - alpha / 2)
            ci_lower = coef - critical_val * se
            ci_upper = coef + critical_val * se
            
            print(f"{'Variable':<12} {'Coef.':<10} {'Std.Err.':<10} {'t':<8} {'P>|t|':<8} {'[{:.1f}%'.format((1-alpha)*100):<8} {'Conf. Interval]':<8}")
            print("-" * 70)
            
            for i in range(len(coef)):
                var_name = f"x{i}" if i > 0 else "const" if i == 0 else f"x{i}"
                print(f"{var_name:<12} {coef[i]:<10.4f} {se[i]:<10.4f} {t_stats[i]:<8.3f} {p_values[i]:<8.3f} {ci_lower[i]:<8.3f} {ci_upper[i]:<8.3f}")
        else:
            print(f"{'Variable':<12} {'Coef.':<10}")
            print("-" * 22)
            for i in range(len(coef)):
                var_name = f"x{i}" if i > 0 else "const" if i == 0 else f"x{i}"
                print(f"{var_name:<12} {coef[i]:<10.4f}")
            print("\nNote: Standard errors could not be computed.")
            
        print("=" * 50)


class LogisticRegression(MaximumLikelihoodEstimator):
    """
    Logistic Regression model with proper Fisher information-based standard errors.
    """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        Computes the negative log-likelihood for logistic regression.
        NLL = -Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
        where p_i = σ(X_i @ β) = 1 / (1 + exp(-X_i @ β))
        Using numerically stable log_sigmoid:
        log(p_i) = log_sigmoid(X_i @ β)
        log(1-p_i) = log_sigmoid(-(X_i @ β))
        """
        logits = X @ params
        # Use PyTorch's numerically stable functions
        nll = -torch.sum(
            y * torch.nn.functional.logsigmoid(logits)
            + (1 - y) * torch.nn.functional.logsigmoid(-logits)
        )
        return nll
    
    def _compute_fisher_information(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Fisher information matrix for logistic regression.
        
        For logistic regression, the Fisher information matrix is:
        I(β) = X'WX where W = diag(p_i(1-p_i)) and p_i = σ(X_i'β)
        """
        logits = X @ params
        probs = torch.sigmoid(logits)  # p_i = P(y_i = 1 | x_i)
        weights = probs * (1 - probs)  # Variance of Bernoulli: p(1-p)
        
        # Fisher information: X'WX where W = diag(weights)
        weighted_X = X * weights.unsqueeze(1)  # Broadcasting weights across features
        fisher_info = weighted_X.T @ X
        
        return fisher_info

    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for each class.
        Args:
            X: Design matrix of shape (n_samples, n_features).
        Returns:
            Array of probabilities of shape (n_samples,).
        """
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        logits = X @ self.params["coef"]
        return torch.sigmoid(logits)

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict class labels.
        Args:
            X: Design matrix of shape (n_samples, n_features).
            threshold: Probability threshold for class assignment.
        Returns:
            Array of predicted class labels (0 or 1).
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).to(torch.int32)


class PoissonRegression(MaximumLikelihoodEstimator):
    """
    Poisson Regression model with proper Fisher information-based standard errors.
    """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> float:
        """
        Computes the negative log-likelihood for Poisson regression.
        The log(y_i!) term is constant w.r.t params, so ignored for optimization.
        NLL = Σ [exp(X_i @ β) - y_i * (X_i @ β)]
        """
        linear_predictor = X @ params
        lambda_i = torch.exp(linear_predictor)  # Predicted rates

        # Sum over samples
        nll = torch.sum(lambda_i - y * linear_predictor)
        return nll
    
    def _compute_fisher_information(self, params: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Fisher information matrix for Poisson regression.
        
        For Poisson regression, the Fisher information matrix is:
        I(β) = X'ΛX where Λ = diag(λ_i) and λ_i = exp(X_i'β)
        """
        linear_predictor = X @ params
        lambda_i = torch.exp(linear_predictor)  # E[y_i] = Var[y_i] = λ_i
        
        # Fisher information: X'ΛX where Λ = diag(λ_i)
        weighted_X = X * lambda_i.unsqueeze(1)  # Broadcasting λ_i across features
        fisher_info = weighted_X.T @ X
        
        return fisher_info

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict expected counts (lambda_i).
        Args:
            X: Design matrix of shape (n_samples, n_features).
        Returns:
            Array of predicted counts.
        """
        if not self.params or "coef" not in self.params:
            raise ValueError("Model has not been fitted yet.")

        linear_predictor = X @ self.params["coef"]
        return torch.exp(linear_predictor)
