"""
Conditional Choice Probability (CCP) estimation methods.

This module provides estimators for P(a|x) from observed data,
which are used as the first stage in CCP-based dynamic model estimation.
"""

import torch
from typing import Optional, Literal


def estimate_ccps(
    states: torch.Tensor,
    actions: torch.Tensor,
    n_states: int,
    n_choices: int,
    method: Literal["frequency"] = "frequency",
    bandwidth: Optional[float] = None,
) -> torch.Tensor:
    """
    Estimate Conditional Choice Probabilities P(a|x) from data.

    Args:
        states: (n_obs,) observed states
        actions: (n_obs,) observed actions
        n_states: Number of discrete states
        n_choices: Number of discrete actions
        method: Estimation method ("frequency")
        bandwidth: Smoothing parameter (not used for frequency)

    Returns:
        ccp_hat: (n_states, n_choices) estimated CCPs
                 ccp_hat[x, a] = P(a|x)
    """
    if method == "frequency":
        ccp_hat = torch.zeros(n_states, n_choices)
        state_counts = torch.bincount(states, minlength=n_states)

        # Avoid division by zero for unobserved states
        # If a state is never observed, we can't estimate CCPs from data.
        # Uniform prior is a safe fallback or handle as NaN.
        # Here we use uniform for unobserved states.

        for x in range(n_states):
            if state_counts[x] > 0:
                mask = states == x
                action_counts = torch.bincount(actions[mask], minlength=n_choices)
                ccp_hat[x, :] = action_counts / action_counts.sum()
            else:
                ccp_hat[x, :] = 1.0 / n_choices

        return ccp_hat
    else:
        raise NotImplementedError(f"Method {method} not implemented")
