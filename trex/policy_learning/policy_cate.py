from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from scipy.optimize import minimize

from ..base import BaseEstimator


def _to_tensor(
    value: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device)
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    tensor = torch.tensor(np.asarray(value).copy(), device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _to_numpy(value: Any, dtype: np.dtype = np.float64) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


@dataclass
class _DistributionTerms:
    objective: np.ndarray
    attention_weight: np.ndarray
    treatment_probability: np.ndarray


class policyCATElearner(BaseEstimator):
    """Policy-aligned linear CATE learner from arXiv:2512.13400."""

    def __init__(
        self,
        cost: float = 0.0,
        sigma: float = 1.0,
        distribution: str = "logistic",
        fit_intercept: bool = True,
        l2_penalty: float = 0.0,
        optimizer: Any = None,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        maxiter: int = 200,
        tol: float = 1e-8,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(device=device)
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive")
        distribution = distribution.lower()
        if distribution not in {"logistic", "normal", "uniform"}:
            raise ValueError("distribution must be one of: logistic, normal, uniform")

        self.cost = float(cost)
        self.sigma = float(sigma)
        self.distribution = distribution
        self.fit_intercept = fit_intercept
        self.l2_penalty = float(l2_penalty)
        self.optimizer = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.dtype = dtype
        self.history: dict[str, list[float]] = {"loss": [], "objective": []}
        self._fitted_feature_dim: Optional[int] = None

    def _prepare_X_numpy(self, X: Any) -> np.ndarray:
        X_np = _to_numpy(X)
        if X_np.ndim == 1:
            X_np = X_np[:, None]
        if self.fit_intercept:
            X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])
        return X_np

    def _prepare_X_tensor(self, X: Any) -> torch.Tensor:
        X_t = _to_tensor(X, self.device, self.dtype)
        if X_t.ndim == 1:
            X_t = X_t[:, None]
        if self.fit_intercept:
            ones = torch.ones((X_t.shape[0], 1), device=self.device, dtype=self.dtype)
            X_t = torch.cat([ones, X_t], dim=1)
        return X_t

    def _pseudo_outcome(self, y: Any, treatment: Any, propensity: Any) -> np.ndarray:
        y_np = _to_numpy(y).reshape(-1)
        w_np = _to_numpy(treatment).reshape(-1)
        e_np = _to_numpy(propensity).reshape(-1)
        if e_np.size == 1:
            e_np = np.full_like(y_np, float(e_np.item()))
        e_np = np.clip(e_np, 1e-6, 1 - 1e-6)
        return y_np * (w_np / e_np - (1 - w_np) / (1 - e_np))

    def _distribution_terms(self, tau: np.ndarray, y_star: np.ndarray) -> _DistributionTerms:
        z = (tau - self.cost) / self.sigma
        if self.distribution == "logistic":
            p = 1.0 / (1.0 + np.exp(-z))
            p_clip = np.clip(p, 1e-8, 1 - 1e-8)
            entropy = -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip))
            objective = p * (y_star - self.cost) + self.sigma * entropy
            attention_weight = (p * (1 - p)) / self.sigma
            return _DistributionTerms(objective, attention_weight, p)

        if self.distribution == "normal":
            pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            cdf = 0.5 * (1 + np.erf(z / np.sqrt(2.0)))
            objective = cdf * (y_star - self.cost) + self.sigma * pdf
            attention_weight = pdf / self.sigma
            return _DistributionTerms(objective, attention_weight, cdf)

        objective = -(y_star - tau) ** 2
        attention_weight = np.ones_like(tau)
        treatment_probability = np.clip((tau - self.cost) / self.sigma, 0.0, 1.0)
        return _DistributionTerms(objective, attention_weight, treatment_probability)

    def _gradient_multiplier(self, tau: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        if self.distribution == "uniform":
            return 2.0 * (y_star - tau)
        z = (tau - self.cost) / self.sigma
        if self.distribution == "logistic":
            density = (1.0 / (1.0 + np.exp(-z))) * (1.0 - 1.0 / (1.0 + np.exp(-z))) / self.sigma
        else:
            density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) / self.sigma
        return density * (y_star - tau)

    def fit(
        self,
        X: Any,
        y: Any,
        treatment: Any,
        propensity: Any = 0.5,
        sample_weight: Optional[Any] = None,
        init_coef: Optional[Any] = None,
    ) -> "policyCATElearner":
        X_np = self._prepare_X_numpy(X)
        y_star = self._pseudo_outcome(y, treatment, propensity)
        n_obs, n_features = X_np.shape
        self._fitted_feature_dim = n_features

        if sample_weight is None:
            weight_np = np.ones(n_obs)
        else:
            weight_np = _to_numpy(sample_weight).reshape(-1)
        weight_np = weight_np / max(weight_np.mean(), 1e-12)

        if init_coef is None:
            beta0, *_ = np.linalg.lstsq(X_np, y_star, rcond=None)
        else:
            beta0 = _to_numpy(init_coef).reshape(-1)

        def objective_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
            tau = X_np @ beta
            terms = self._distribution_terms(tau, y_star)
            avg_objective = np.mean(weight_np * terms.objective)
            multiplier = weight_np * self._gradient_multiplier(tau, y_star)
            grad = X_np.T @ multiplier / n_obs
            if self.fit_intercept and beta.size > 1:
                penalty_beta = beta.copy()
                penalty_beta[0] = 0.0
            else:
                penalty_beta = beta
            penalty = 0.5 * self.l2_penalty * np.sum(penalty_beta**2)
            grad_penalty = self.l2_penalty * penalty_beta
            loss = -avg_objective + penalty
            grad_loss = -grad + grad_penalty
            return loss, grad_loss

        def fun(beta: np.ndarray) -> float:
            loss, grad = objective_and_grad(beta)
            self._last_grad = grad
            return loss

        def jac(beta: np.ndarray) -> np.ndarray:
            _, grad = objective_and_grad(beta)
            return grad

        result = minimize(
            fun=fun,
            x0=beta0,
            jac=jac,
            method="L-BFGS-B",
            options={"maxiter": self.maxiter, "ftol": self.tol, **self.optimizer_kwargs},
        )

        beta = result.x
        tau = X_np @ beta
        terms = self._distribution_terms(tau, y_star)
        self.history["loss"] = [float(result.fun)]
        self.history["objective"] = [float(np.mean(weight_np * terms.objective))]
        self.params = {
            "coef": torch.tensor(beta, device=self.device, dtype=self.dtype),
            "cost": torch.tensor(self.cost, device=self.device, dtype=self.dtype),
            "sigma": torch.tensor(self.sigma, device=self.device, dtype=self.dtype),
        }
        self.optim_result_ = result
        return self

    def predict(self, X: Any) -> torch.Tensor:
        if self.params is None:
            raise ValueError("Model has not been fitted yet")
        X_t = self._prepare_X_tensor(X)
        return X_t @ self.params["coef"]

    def cate(self, X: Any) -> torch.Tensor:
        return self.predict(X)

    def decision_function(self, X: Any) -> torch.Tensor:
        return (self.predict(X) - self.cost) / self.sigma

    def predict_proba(self, X: Any) -> torch.Tensor:
        score = self.decision_function(X)
        if self.distribution == "logistic":
            return torch.sigmoid(score)
        if self.distribution == "normal":
            sqrt_two = torch.sqrt(torch.tensor(2.0, device=self.device, dtype=self.dtype))
            return 0.5 * (1 + torch.erf(score / sqrt_two))
        return torch.clamp(score, 0.0, 1.0)

    def predict_policy(self, X: Any, cost: Optional[float] = None) -> torch.Tensor:
        threshold = self.cost if cost is None else float(cost)
        return (self.predict(X) >= threshold).to(torch.int64)

    def attention_weights(self, X: Any) -> torch.Tensor:
        tau = self.predict(X).detach().cpu().numpy()
        y_placeholder = np.zeros_like(tau)
        weights = self._distribution_terms(tau, y_placeholder).attention_weight
        return torch.tensor(weights, device=self.device, dtype=self.dtype)
