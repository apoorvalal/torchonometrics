"""
torchonometrics: GPU-accelerated econometrics in PyTorch.

A PyTorch-based library for high-performance econometric analysis with first-class
support for fixed effects, causal inference, and maximum likelihood estimation.

API entry points:

- `torchonometrics.linear` for linear models and fixed effects.
- `torchonometrics.mle` for maximum-likelihood estimators.
- `torchonometrics.gmm` for GMM and GEL estimators.
- `torchonometrics.choice` for discrete choice models.
- `torchonometrics.choice.dynamic` for dynamic discrete choice models.
"""

__version__ = "0.1.0"

from .base import BaseEstimator
from .linear import LinearRegression
from .mle import LogisticRegression, PoissonRegression, MaximumLikelihoodEstimator
from .demean import demean_torch, prepare_fixed_effects

__all__ = [
    "BaseEstimator", 
    "LinearRegression",
    "MaximumLikelihoodEstimator",
    "LogisticRegression",
    "PoissonRegression",
    "demean_torch",
    "prepare_fixed_effects",
]
