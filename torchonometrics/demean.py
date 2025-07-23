"""
PyTorch-based demeaning for fixed effects regression.
"""
from typing import Optional, Union

import torch
import numpy as np


# @torch.compile  # Disabled for compatibility
def _demean_torch_impl(
    x: torch.Tensor,
    flist: torch.Tensor,
    weights: torch.Tensor,
    n_groups: int,
    tol: float,
    maxiter: int,
) -> tuple[torch.Tensor, bool]:
    """Compiled implementation of demeaning via alternating projections."""
    n_factors = flist.shape[1]

    def _apply_factor(x_curr, j):
        """Process a single factor."""
        factor_ids = flist[:, j]
        wx = x_curr * weights[:, None]

        # Compute group weights and weighted sums using scatter_add
        group_weights = torch.zeros(n_groups, device=x_curr.device, dtype=weights.dtype)
        group_weights.scatter_add_(0, factor_ids, weights)

        # For each column in wx, compute group sums
        group_sums = torch.zeros(n_groups, x_curr.shape[1], device=x_curr.device, dtype=x_curr.dtype)
        for col_idx in range(x_curr.shape[1]):
            group_sums[:, col_idx].scatter_add_(0, factor_ids, wx[:, col_idx])

        # Compute and subtract means
        means = group_sums / torch.clamp(group_weights[:, None], min=1e-12)
        return x_curr - means[factor_ids]

    def _demean_step(x_curr):
        """Single demeaning step for all factors."""
        result = x_curr
        for j in range(n_factors):
            result = _apply_factor(result, j)
        return result

    # Run the iteration loop
    x_curr = x.clone()
    converged = False

    for i in range(maxiter):
        x_new = _demean_step(x_curr)
        max_diff = torch.max(torch.abs(x_new - x_curr))

        if max_diff < tol:
            converged = True
            break

        x_curr = x_new

    return x_curr, converged


def demean_torch(
    x: Union[np.ndarray, torch.Tensor],
    flist: Union[np.ndarray, torch.Tensor],
    weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    tol: float = 1e-08,
    maxiter: int = 100_000,
) -> tuple[torch.Tensor, bool]:
    """
    Demean array using PyTorch implementation of alternating projections.

    Parameters
    ----------
    x : array-like
        Input array of shape (n_samples, n_features) to demean.
    flist : array-like
        Fixed effects array of shape (n_samples, n_factors) with integer factor IDs.
    weights : array-like, optional
        Weights array of shape (n_samples,). If None, uses uniform weights.
    tol : float, optional
        Tolerance for convergence. Default is 1e-08.
    maxiter : int, optional
        Maximum number of iterations. Default is 100_000.

    Returns
    -------
    tuple[torch.Tensor, bool]
        Tuple of (demeaned_array, converged).
    """
    # Convert inputs to PyTorch tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)
    else:
        x = x.to(torch.float64)

    if not isinstance(flist, torch.Tensor):
        flist = torch.tensor(flist, dtype=torch.int64)
    else:
        flist = flist.to(torch.int64)

    # Handle weights
    if weights is None:
        weights = torch.ones(x.shape[0], dtype=torch.float64, device=x.device)
    else:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float64, device=x.device)
        else:
            weights = weights.to(torch.float64).to(x.device)

    # Ensure x is 2D
    if x.ndim == 1:
        x = x[:, None]

    # Ensure flist is 2D
    if flist.ndim == 1:
        flist = flist[:, None]

    # Move flist to same device as x
    flist = flist.to(x.device)

    # Compute number of groups across all factors
    n_groups = int(torch.max(flist).item() + 1)

    # Call the compiled implementation
    result, converged = _demean_torch_impl(
        x, flist, weights, n_groups, tol, maxiter
    )

    return result, converged


def prepare_fixed_effects(fe_vars: list) -> torch.Tensor:
    """
    Prepare fixed effects variables for demeaning.

    Parameters
    ----------
    fe_vars : list
        List of arrays containing fixed effects variables.

    Returns
    -------
    torch.Tensor
        Array of shape (n_samples, n_factors) with integer factor IDs.
    """
    if not fe_vars:
        return None

    # Convert each FE variable to consecutive integers starting from 0
    fe_arrays = []
    offset = 0

    for fe_var in fe_vars:
        if not isinstance(fe_var, torch.Tensor):
            fe_array = torch.tensor(fe_var)
        else:
            fe_array = fe_var

        # Get unique values and create mapping
        unique_vals = torch.unique(fe_array)
        n_unique = len(unique_vals)

        # Create consecutive integer mapping
        fe_mapped = torch.searchsorted(unique_vals, fe_array) + offset
        fe_arrays.append(fe_mapped)
        offset += n_unique

    # Stack all FE variables
    return torch.stack(fe_arrays, dim=1)
