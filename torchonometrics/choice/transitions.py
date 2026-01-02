"""
State space and transition estimation utilities for dynamic choice models.

This module provides tools for estimating state transition probabilities P(x'|x,a)
from panel data, as well as function approximation methods for high-dimensional
state spaces.

References:
    Rawat & Rust (2025): "Structural Econometrics and Reinforcement Learning"
    Rust (1987): "Optimal replacement of GMC bus engines"
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


def estimate_transition_matrix(
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    n_states: int,
    n_choices: int,
    method: Literal["frequency", "kernel", "parametric"] = "frequency",
    bandwidth: Optional[float] = None,
) -> torch.Tensor:
    """
    Estimate state transition probabilities P(x'|x,a) from panel data.

    Three approaches supported:
    1. "frequency": Nonparametric frequency estimator (implemented)
    2. "kernel": Kernel smoothing for continuous states (future)
    3. "parametric": Ordered probit/logit for ordinal transitions (future)

    Args:
        states: (n_obs,) observed states x_t
        actions: (n_obs,) observed actions a_t
        next_states: (n_obs,) observed next states x_{t+1}
        n_states: Number of discrete states
        n_choices: Number of actions
        method: Estimation approach
        bandwidth: For kernel methods (not yet implemented)

    Returns:
        P: (n_states, n_choices, n_states) estimated transition matrix
           where P[x, a, x'] ≈ Pr(x_{t+1}=x' | x_t=x, a_t=a)

    Example:
        >>> # Estimate transitions from bus maintenance data
        >>> P_hat = estimate_transition_matrix(
        ...     states=data.states,
        ...     actions=data.actions,
        ...     next_states=data.next_states,
        ...     n_states=90,
        ...     n_choices=2,
        ...     method="frequency"
        ... )
        >>> # Verify probabilities sum to 1
        >>> assert torch.allclose(P_hat.sum(dim=2), torch.ones(90, 2))

    References:
        Rawat & Rust (2025), Section 3: RL methods work with estimated transitions
    """
    if method == "frequency":
        # Nonparametric frequency estimator
        P = torch.zeros(n_states, n_choices, n_states)
        for x in range(n_states):
            for a in range(n_choices):
                mask = (states == x) & (actions == a)
                if mask.sum() > 0:
                    next_state_counts = torch.bincount(
                        next_states[mask], minlength=n_states
                    )
                    P[x, a, :] = next_state_counts / next_state_counts.sum()
                else:
                    # If (x, a) never observed, use uniform distribution
                    P[x, a, :] = 1.0 / n_states
        return P
    elif method == "kernel":
        # Kernel smoothing for continuous/high-dimensional states
        raise NotImplementedError("Kernel estimation coming in Phase 3")
    elif method == "parametric":
        # Parametric specification (e.g., ordered probit for mileage)
        raise NotImplementedError("Parametric transitions coming in Phase 3")
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: frequency, kernel, parametric"
        )


def discretize_state(
    continuous_state: torch.Tensor,
    n_bins: int,
    method: Literal["quantile", "uniform"] = "quantile",
    state_min: Optional[float] = None,
    state_max: Optional[float] = None,
) -> torch.Tensor:
    """
    Discretize continuous state variable into bins.

    Args:
        continuous_state: (n_obs,) continuous state values
        n_bins: Number of discrete bins
        method: "quantile" (equal counts) or "uniform" (equal width)
        state_min: Override minimum for uniform binning
        state_max: Override maximum for uniform binning

    Returns:
        discrete_state: (n_obs,) discretized state indices in {0,...,n_bins-1}

    Example:
        >>> # Discretize mileage into 90 bins
        >>> mileage_discrete = discretize_state(
        ...     continuous_state=mileage,
        ...     n_bins=90,
        ...     method="uniform",
        ...     state_min=0,
        ...     state_max=450000
        ... )

    References:
        Rust (1987): Uses uniform discretization for mileage state space
    """
    if method == "quantile":
        # Equal counts per bin using quantiles
        quantiles = torch.linspace(0, 1, n_bins + 1)
        bin_edges = torch.quantile(continuous_state, quantiles)
        bin_edges[-1] = continuous_state.max() + 1  # Ensure last value included
        discrete_state = torch.bucketize(continuous_state, bin_edges[1:-1])
        return discrete_state
    elif method == "uniform":
        # Equal width bins
        if state_min is None:
            state_min = continuous_state.min().item()
        if state_max is None:
            state_max = continuous_state.max().item()

        bin_width = (state_max - state_min) / n_bins
        discrete_state = torch.floor((continuous_state - state_min) / bin_width).long()
        # Clamp to valid range
        discrete_state = torch.clamp(discrete_state, 0, n_bins - 1)
        return discrete_state
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: quantile, uniform")


class DeepValueFunction(nn.Module):
    """
    Neural network value function approximation for high-dimensional states.

    Enables dynamic choice estimation when |X| > 10,000 makes tabular
    methods intractable. Uses deep Q-network (DQN) style architecture.

    References:
        Rawat & Rust (2025), pp. 16-17: "Deep RL can indeed overcome the curse
        of dimensionality that limits traditional Dynamic Programming"
    """

    def __init__(
        self,
        state_dim: int,
        n_choices: int,
        hidden_dims: list[int] = [128, 64],
    ):
        """
        Initialize deep value function network.

        Args:
            state_dim: Dimension of continuous/high-dim state space
            n_choices: Number of discrete actions
            hidden_dims: Architecture of hidden layers
        """
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                ]
            )
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_choices))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute value function for each choice at given states.

        Args:
            state: (batch, state_dim) continuous states

        Returns:
            values: (batch, n_choices) choice-specific values v̄(s,a)
        """
        return self.network(state)
