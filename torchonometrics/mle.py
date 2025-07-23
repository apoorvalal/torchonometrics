from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

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
            init_params_val = torch.randn(n_features, device=X.device, dtype=X.dtype) * 0.01
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

        self.params = {"coef": current_params.detach()}
        self.iterations_run = i + 1  # Store how many iterations actually ran

        return self

    def summary(self) -> None:
        """Print a summary of the model results."""
        if not self.params or "coef" not in self.params:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 30)
        print(f"Optimizer: {self.optimizer_class.__name__}")
        if hasattr(self, "iterations_run"):
            print(
                f"Optimization ran for {self.iterations_run}/{self.maxiter} iterations."
            )
        if self.history["loss"]:
            print(f"Final Loss: {self.history['loss'][-1]:.4e}")

        print(f"Coefficients: {self.params['coef']}")
        print("=" * 30)


class LogisticRegression(MaximumLikelihoodEstimator):
    """
    Logistic Regression model.
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
        nll = -torch.sum(y * torch.nn.functional.logsigmoid(logits) + (1 - y) * torch.nn.functional.logsigmoid(-logits))
        return nll

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
    Poisson Regression model.
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