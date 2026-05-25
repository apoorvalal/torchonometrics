"""
Score-matching estimators.

This module ports the core idea of the R package `asm`: learn an antitonic
projection of the error score from pilot residuals, then fit a linear model
with the induced convex loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal, Optional, Union

import numpy as np
import torch

from .base import BaseEstimator


ArrayLike = Union[np.ndarray, torch.Tensor]
Pilot = Literal["lad", "ols"]


def _as_numpy_2d(x: ArrayLike, *, name: str) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_numpy_1d(y: ArrayLike, *, name: str) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    arr = np.asarray(y, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _lad_irls(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 80,
    tol: float = 1e-8,
    eps: float = 1e-6,
) -> np.ndarray:
    """Approximate least-absolute-deviation regression by IRLS."""
    beta = _ols(X, y)
    for _ in range(max_iter):
        resid = y - X @ beta
        weights = 1.0 / np.maximum(np.abs(resid), eps)
        Xw = X * np.sqrt(weights)[:, None]
        yw = y * np.sqrt(weights)
        beta_next = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        denom = 1.0 + np.linalg.norm(beta)
        if np.linalg.norm(beta_next - beta) / denom < tol:
            beta = beta_next
            break
        beta = beta_next
    return beta


def _pava_decreasing(values: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Pool-adjacent-violators projection onto nonincreasing sequences."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if weights is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != y.shape:
            raise ValueError("weights must have the same shape as values.")
        if np.any(w <= 0):
            raise ValueError("weights must be positive.")

    levels: list[float] = []
    level_weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for idx, (value, weight) in enumerate(zip(y, w)):
        levels.append(float(value))
        level_weights.append(float(weight))
        starts.append(idx)
        ends.append(idx + 1)
        while len(levels) >= 2 and levels[-2] < levels[-1]:
            total_weight = level_weights[-2] + level_weights[-1]
            pooled = (
                levels[-2] * level_weights[-2] + levels[-1] * level_weights[-1]
            ) / total_weight
            levels[-2] = pooled
            level_weights[-2] = total_weight
            ends[-2] = ends[-1]
            levels.pop()
            level_weights.pop()
            starts.pop()
            ends.pop()

    out = np.empty_like(y)
    for level, start, end in zip(levels, starts, ends):
        out[start:end] = level
    return out


def _bw_nrd0(x: np.ndarray) -> float:
    """R's default normal-reference bandwidth, with stable fallbacks."""
    n = x.size
    if n < 2:
        raise ValueError("At least two observations are required.")
    sd = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    scale = min(sd, (q75 - q25) / 1.34)
    if not np.isfinite(scale) or scale <= 0:
        scale = sd
    if not np.isfinite(scale) or scale <= 0:
        mad = np.median(np.abs(x - np.median(x))) / 0.6745
        scale = mad if np.isfinite(mad) and mad > 0 else 1.0
    return 0.9 * scale * n ** (-0.2)


def _gaussian_kde_on_grid(
    samples: np.ndarray,
    *,
    grid_size: int,
    bandwidth: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    bw = _bw_nrd0(samples) if bandwidth is None else float(bandwidth)
    if bw <= 0 or not np.isfinite(bw):
        raise ValueError("bandwidth must be positive and finite.")

    support1 = np.array([samples.min(), samples.max()]) + 3.0 * bw * np.array([-1.0, 1.0])
    iqr = np.percentile(samples, 75) - np.percentile(samples, 25)
    support2 = np.median(samples) + iqr * np.array([-20.0, 20.0])
    lower = max(support1[0], support2[0])
    upper = min(support1[1], support2[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower, upper = support1

    grid = np.linspace(lower, upper, grid_size)
    density = np.zeros_like(grid)
    chunk = 256
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * bw * samples.size)
    for start in range(0, grid_size, chunk):
        stop = min(start + chunk, grid_size)
        u = (grid[start:stop, None] - samples[None, :]) / bw
        density[start:stop] = norm * np.exp(-0.5 * u * u).sum(axis=1)
    density = np.maximum(density, np.finfo(float).tiny)
    return grid, density


@dataclass
class AntitonicScore:
    """Piecewise-linear antitonic score estimate."""

    domain: np.ndarray
    values: np.ndarray
    slopes: np.ndarray
    symmetric: bool = True

    def __post_init__(self) -> None:
        if self.domain.ndim != 1 or self.values.ndim != 1:
            raise ValueError("domain and values must be one-dimensional.")
        if self.domain.size != self.values.size:
            raise ValueError("domain and values must have the same length.")
        if self.domain.size < 2:
            raise ValueError("At least two score support points are required.")
        if np.any(np.diff(self.domain) <= 0):
            raise ValueError("domain must be strictly increasing.")

    def _raw(self, z: np.ndarray) -> np.ndarray:
        return np.interp(
            z,
            self.domain,
            self.values,
            left=self.values[0],
            right=self.values[-1],
        )

    def _raw_derivative(self, z: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.domain[1:], z, side="right")
        idx = np.clip(idx, 0, self.slopes.size - 1)
        return self.slopes[idx]

    def __call__(self, z: ArrayLike) -> np.ndarray:
        arr = _as_numpy_1d(z, name="z")
        if self.symmetric:
            return 0.5 * (self._raw(arr) - self._raw(-arr))
        return self._raw(arr)

    def derivative(self, z: ArrayLike) -> np.ndarray:
        arr = _as_numpy_1d(z, name="z")
        if self.symmetric:
            return 0.5 * (self._raw_derivative(arr) + self._raw_derivative(-arr))
        return self._raw_derivative(arr)


def estimate_antitonic_score(
    residuals: ArrayLike,
    *,
    symmetric: bool = True,
    k: int = 3000,
    kernel_grid_size: int = 2**15,
    bandwidth: Optional[float] = None,
) -> AntitonicScore:
    """
    Estimate the antitonic projection of an error score from residuals.

    The estimator follows `asm`'s practical path: smooth residuals by Gaussian
    KDE, evaluate the density quantile function, project its discrete derivative
    onto the nonincreasing cone by PAVA, and interpolate the projected score.
    """
    eps = np.sort(_as_numpy_1d(residuals, name="residuals"))
    n = eps.size
    if n < 3:
        raise ValueError("At least three residuals are required.")

    grid, density = _gaussian_kde_on_grid(
        eps, grid_size=kernel_grid_size, bandwidth=bandwidth
    )
    masses = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(masses)])
    total_mass = cdf[-1]
    if total_mass <= 0 or not np.isfinite(total_mass):
        raise ValueError("KDE produced invalid total mass.")
    cdf = cdf / total_mass

    k = max(int(k), 2 * n, 10)
    u = np.linspace(0.0, 1.0, k + 1)
    quantiles = np.interp(u, cdf, grid)
    density_quantiles = np.interp(quantiles, grid, density)
    init_scores = np.diff(density_quantiles) * k
    domain = 0.5 * (quantiles[1:] + quantiles[:-1])
    projected = _pava_decreasing(init_scores)

    skip = 2 * (k // n) if n > 2 else 0
    if skip > 0 and 2 * skip < projected.size - 2:
        domain = domain[skip:-skip]
        projected = projected[skip:-skip]

    jumps = np.diff(projected) != 0
    keep = np.r_[jumps, True] | np.r_[True, jumps]
    domain = domain[keep]
    projected = projected[keep]
    if domain.size < 2:
        domain = np.array([eps.min() - 1.0, eps.max() + 1.0])
        projected = np.array([1.0, -1.0])

    slopes = np.diff(projected) / np.diff(domain)
    slopes = np.minimum(slopes, 0.0)
    return AntitonicScore(domain=domain, values=projected, slopes=slopes, symmetric=symmetric)


class AntitonicScoreMatchingRegression(BaseEstimator):
    """
    Linear regression via antitonic score matching.

    The estimator learns a data-driven convex loss from pilot residuals, then
    fits the regression coefficients by solving the associated convex
    M-estimation score equations.
    """

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        symmetric: bool = True,
        pilot: Pilot = "lad",
        alt_iter: int = 2,
        k: int = 3000,
        kernel_grid_size: int = 2**15,
        bandwidth: Optional[float] = None,
        max_iter: int = 80,
        tol: float = 1e-8,
        ridge: float = 1e-8,
        device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__(device=device)
        if pilot not in {"lad", "ols"}:
            raise ValueError("pilot must be 'lad' or 'ols'.")
        if alt_iter < 1:
            raise ValueError("alt_iter must be at least one.")
        self.fit_intercept = fit_intercept
        self.symmetric = symmetric
        self.pilot = pilot
        self.alt_iter = alt_iter
        self.k = k
        self.kernel_grid_size = kernel_grid_size
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge

    def fit(self, X: ArrayLike, y: ArrayLike) -> "AntitonicScoreMatchingRegression":
        X_np = _as_numpy_2d(X, name="X")
        y_np = _as_numpy_1d(y, name="y")
        if X_np.shape[0] != y_np.size:
            raise ValueError("X and y have incompatible row counts.")
        if y_np.size < 3:
            raise ValueError("At least three observations are required.")

        design = _add_intercept(X_np) if self.fit_intercept else X_np.copy()
        beta = _lad_irls(design, y_np) if self.pilot == "lad" else _ols(design, y_np)

        score = None
        converged = False
        n, p = design.shape
        for _ in range(self.alt_iter):
            residuals = y_np - design @ beta
            score = estimate_antitonic_score(
                residuals,
                symmetric=self.symmetric,
                k=self.k,
                kernel_grid_size=self.kernel_grid_size,
                bandwidth=self.bandwidth,
            )
            beta, converged = self._solve_for_beta(design, y_np, beta, score)

        residuals = y_np - design @ beta
        if score is None:
            raise RuntimeError("Score estimation failed.")
        psi = score(residuals)
        info = float(np.mean(psi * psi))
        x_second = design.T @ design / n
        info_matrix = info * x_second
        cov_asymptotic = np.linalg.pinv(info_matrix)
        std_errors = np.sqrt(np.clip(np.diag(cov_asymptotic) / n, 0.0, np.inf))

        coef = beta[1:] if self.fit_intercept else beta
        intercept = float(beta[0]) if self.fit_intercept else 0.0
        coef_t = torch.as_tensor(coef, dtype=torch.float32, device=self.device)
        self.params = {
            "coef": coef_t,
            "intercept": torch.tensor(intercept, dtype=torch.float32, device=self.device),
            "se": torch.as_tensor(std_errors, dtype=torch.float32, device=self.device),
        }
        self.beta_ = beta
        self.coef_ = coef
        self.intercept_ = intercept
        self.std_errors_ = std_errors
        self.covariance_ = cov_asymptotic / n
        self.info_ = info
        self.info_matrix_ = info_matrix
        self.residuals_ = residuals
        self.fitted_values_ = design @ beta
        self.score_ = score
        self.converged_ = converged
        self.n_features_in_ = X_np.shape[1]
        return self

    def _solve_for_beta(
        self,
        X: np.ndarray,
        y: np.ndarray,
        beta_init: np.ndarray,
        score: AntitonicScore,
    ) -> tuple[np.ndarray, bool]:
        beta = beta_init.copy()
        n, p = X.shape
        converged = False

        for _ in range(self.max_iter):
            residuals = y - X @ beta
            psi = score(residuals)
            grad = -(X.T @ psi) / n
            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < self.tol * np.sqrt(p):
                converged = True
                break

            psi_deriv = np.minimum(score.derivative(residuals), 0.0)
            weights = np.maximum(-psi_deriv, 0.0)
            hessian = (X.T * weights) @ X / n
            hessian = hessian + self.ridge * np.eye(p)
            try:
                step = np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ grad

            alpha = 1.0
            while alpha > 1e-4:
                candidate = beta - alpha * step
                candidate_resid = y - X @ candidate
                candidate_grad = -(X.T @ score(candidate_resid)) / n
                if np.linalg.norm(candidate_grad) <= grad_norm:
                    beta = candidate
                    break
                alpha *= 0.5
            else:
                beta = beta - 1e-4 * step

        return beta, converged

    def predict(self, X: ArrayLike) -> torch.Tensor:
        if self.params is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_np = _as_numpy_2d(X, name="X")
        if X_np.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of columns.")
        values = X_np @ self.coef_ + self.intercept_
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def confidence_interval(self, level: float = 0.95) -> np.ndarray:
        """Return Wald confidence intervals for intercept and slopes."""
        if not hasattr(self, "std_errors_"):
            raise RuntimeError("Model has not been fitted yet.")
        if not 0 < level < 1:
            raise ValueError("level must lie between zero and one.")
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        center = self.beta_ if self.fit_intercept else self.coef_
        return np.column_stack([center - z * self.std_errors_, center + z * self.std_errors_])
