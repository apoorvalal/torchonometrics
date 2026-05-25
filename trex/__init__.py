"""
trex: GPU-accelerated econometrics in PyTorch.

A PyTorch-based library for high-performance econometric analysis with first-class
support for fixed effects, causal inference, and maximum likelihood estimation.

API entry points:

- `trex.linear` for linear models and fixed effects.
- `trex.mle` for maximum-likelihood estimators.
- `trex.gmm` for GMM and GEL estimators.
- `trex.choice` for discrete choice models.
- `trex.choice.dynamic` for dynamic discrete choice models.
"""

__version__ = "0.1.0"

from .base import BaseEstimator
from .linear import LinearRegression
from .mle import LogisticRegression, PoissonRegression, MaximumLikelihoodEstimator
from .demean import demean_torch, prepare_fixed_effects
from .grouped_fe import KNNGroupedFixedEffects, build_panel_embeddings
from .latent_factor import LatentFactorGLM
from .panel import (
    NuclearNormMatrixCompletion,
    SyntheticDID,
    did_estimate,
    matrix_completion_estimate,
    panel_estimates,
    sc_estimate,
    synthdid_estimate,
)
from .simdgp import (
    CompletionEndpointLLMInContextGenerator,
    SafetensorsLLMInContextGenerator,
    SafetensorsQLORAGenerator,
    TabularDiffusion,
    TabularTransformer,
    TabularWGAN,
    distribution_metrics,
    sliced_wasserstein_distance,
)

__all__ = [
    "BaseEstimator", 
    "LinearRegression",
    "MaximumLikelihoodEstimator",
    "LogisticRegression",
    "PoissonRegression",
    "KNNGroupedFixedEffects",
    "build_panel_embeddings",
    "LatentFactorGLM",
    "NuclearNormMatrixCompletion",
    "SyntheticDID",
    "did_estimate",
    "sc_estimate",
    "matrix_completion_estimate",
    "panel_estimates",
    "synthdid_estimate",
    "demean_torch",
    "prepare_fixed_effects",
    "TabularTransformer",
    "TabularWGAN",
    "TabularDiffusion",
    "CompletionEndpointLLMInContextGenerator",
    "SafetensorsLLMInContextGenerator",
    "SafetensorsQLORAGenerator",
    "distribution_metrics",
    "sliced_wasserstein_distance",
]
