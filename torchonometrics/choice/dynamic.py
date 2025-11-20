"""
Dynamic discrete choice models with forward-looking agents.

This module implements structural models where agents solve sequential decision
problems under uncertainty, including the Rust (1987) nested fixed point method
and Hotz-Miller (1993) CCP inversion approach.

References:
    Rust (1987): "Optimal replacement of GMC bus engines"
    Hotz & Miller (1993): "Conditional choice probabilities and estimation"
    Aguirregabiria & Mira (2002): "Nested pseudo-likelihood"
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Literal, Union

import torch

from .base import ChoiceModel


@dataclass
class DynamicChoiceData:
    """
    Container for dynamic choice panel data.

    Attributes:
        states: (n_obs,) observed states x_t
        actions: (n_obs,) observed actions a_t
        next_states: (n_obs,) observed next states x_{t+1}
        individual_ids: (n_obs,) panel identifier
        time_periods: (n_obs,) time index
        covariates: Optional (n_obs, n_features) additional state variables
    """

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    individual_ids: torch.Tensor
    time_periods: torch.Tensor
    covariates: Optional[torch.Tensor] = None

    def validate(self) -> None:
        """
        Check data integrity and stationarity.

        Raises:
            ValueError: If data shapes are inconsistent or contain invalid values
        """
        n_obs = len(self.states)
        if not all(
            len(x) == n_obs
            for x in [
                self.actions,
                self.next_states,
                self.individual_ids,
                self.time_periods,
            ]
        ):
            raise ValueError("All tensors must have the same length")

        if self.covariates is not None and len(self.covariates) != n_obs:
            raise ValueError(f"Covariates length {len(self.covariates)} != {n_obs}")

        # Check for negative values in states/actions
        if torch.any(self.states < 0) or torch.any(self.actions < 0):
            raise ValueError("States and actions must be non-negative integers")

    def to_dict(self) -> dict:
        """Convert to dict for compatibility with existing code."""
        result = {
            "states": self.states,
            "actions": self.actions,
            "next_states": self.next_states,
            "individual_ids": self.individual_ids,
            "time_periods": self.time_periods,
        }
        if self.covariates is not None:
            result["covariates"] = self.covariates
        return result


class DynamicChoiceModel(ChoiceModel):
    """
    Base class for dynamic discrete choice models.

    Implements common infrastructure for:
    - State space management
    - Transition probability specification
    - Value function iteration
    - Choice probability computation

    Subclasses implement specific estimation methods (NFP, CCP, NPL).
    """

    def __init__(
        self,
        n_states: int,
        n_choices: int,
        discount_factor: float,
        transition_type: Literal["parametric", "nonparametric"] = "parametric",
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        maxiter: int = 1000,
        tol: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize dynamic choice model.

        Args:
            n_states: Number of discrete states in X
            n_choices: Number of discrete actions in J
            discount_factor: β ∈ (0,1), typically 0.95-0.9999
            transition_type: How to model P(x'|x,a)
            optimizer: PyTorch optimizer class
            maxiter: Maximum iterations for value iteration
            tol: Convergence tolerance
            device: Device for computations (auto-detects if None)
        """
        super().__init__(optimizer=optimizer, maxiter=maxiter, tol=tol, device=device)
        self.n_states = n_states
        self.n_choices = n_choices
        self.discount_factor = discount_factor
        self.transition_type = transition_type

        # Will be set by set_transition_probabilities()
        self.transition_matrix: Optional[torch.Tensor] = None
        self.transition_params: Optional[dict] = None

        # Will be set by set_flow_utility()
        self.utility_fn: Optional[Callable] = None
        self.utility_params: Optional[dict] = None

    def to(self, device: Union[torch.device, str]) -> "DynamicChoiceModel":
        """Move model and transition matrix to specified device."""
        super().to(device)
        if self.transition_matrix is not None:
            self.transition_matrix = self.transition_matrix.to(self.device)
        return self

    def set_transition_probabilities(
        self,
        transition_matrix: torch.Tensor,
        transition_params: Optional[dict] = None,
    ) -> None:
        """
        Specify state transition mechanism P(x_{t+1} | x_t, a_t).

        Args:
            transition_matrix: (n_states, n_choices, n_states) tensor
                              P[x, a, x'] = Pr(x_{t+1}=x' | x_t=x, a_t=a)
            transition_params: Parameters φ if transitions are parametric

        Raises:
            ValueError: If transition matrix shape is incorrect
        """
        expected_shape = (self.n_states, self.n_choices, self.n_states)
        if transition_matrix.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, got {transition_matrix.shape}"
            )

        # Verify probabilities sum to 1
        row_sums = transition_matrix.sum(dim=2)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            raise ValueError("Transition probabilities must sum to 1 for each (x,a)")

        self.transition_matrix = transition_matrix.to(self.device)
        self.transition_params = transition_params

    def set_flow_utility(
        self,
        utility_fn: Callable,
        utility_params: Optional[dict] = None,
    ) -> None:
        """
        Specify per-period utility ū(x, a; θ).

        Args:
            utility_fn: Function computing u(x, a; θ)
            utility_params: Initial values for θ
        """
        self.utility_fn = utility_fn
        self.utility_params = utility_params

    def _bellman_operator(
        self,
        v_bar: torch.Tensor,
        flow_utility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Bellman operator T to value functions.

        Computes:
            T v̄_j(x) = ū_j(x) + β Σ_{x'} P(x'|x,a=j) log[Σ_k exp(v̄_k(x'))]

        Args:
            v_bar: (n_states, n_choices) current value functions
            flow_utility: (n_states, n_choices) per-period utilities ū(x,a)

        Returns:
            T_v_bar: (n_states, n_choices) updated value functions
        """
        if self.transition_matrix is None:
            raise ValueError("Must call set_transition_probabilities() first")

        # Compute log-sum-exp for continuation value
        # emax(x') = log[Σ_k exp(v̄_k(x'))]
        emax = torch.logsumexp(v_bar, dim=1)  # (n_states,)

        # Expected continuation value for each (x, a)
        # E[emax(x') | x, a] = Σ_{x'} P(x'|x,a) emax(x')
        # Shape: (n_states, n_choices, n_states) @ (n_states,) -> (n_states, n_choices)
        continuation_value = torch.einsum("xay,y->xa", self.transition_matrix, emax)

        # Bellman update
        T_v_bar = flow_utility + self.discount_factor * continuation_value

        return T_v_bar

    def solve_value_functions(
        self,
        flow_utility: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
    ) -> torch.Tensor:
        """
        Solve value function fixed point via successive approximations.

        Iterates: v̄^{k+1} = T(v̄^k) until convergence, where T is
        the Bellman operator.

        Args:
            flow_utility: (n_states, n_choices) per-period utilities ū(x,a)
            tol: Convergence tolerance for sup-norm
            max_iter: Maximum iterations

        Returns:
            v_bar: (n_states, n_choices) converged value functions

        Raises:
            RuntimeError: If iteration does not converge

        Example:
            >>> model = DynamicChoiceModel(n_states=90, n_choices=2,
            ...                            discount_factor=0.95)
            >>> flow_u = torch.randn(90, 2)
            >>> v_bar = model.solve_value_functions(flow_u)
            >>> print(v_bar.shape)
            torch.Size([90, 2])

        References:
            Rust (1987), Equation (3.4): Contraction mapping theorem
        """
        v_bar = torch.zeros(self.n_states, self.n_choices, device=self.device)

        for iteration in range(max_iter):
            v_bar_new = self._bellman_operator(v_bar, flow_utility)
            error = torch.max(torch.abs(v_bar_new - v_bar)).item()

            if error < tol:
                return v_bar_new

            v_bar = v_bar_new

        raise RuntimeError(
            f"Value iteration did not converge after {max_iter} iterations. "
            f"Final error: {error:.6e}"
        )

    def _compute_choice_probs(
        self,
        v_bar: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute conditional choice probabilities from value functions.

        Under Type I extreme value errors:
            P(a|x) = exp(v̄_a(x)) / Σ_j exp(v̄_j(x))

        Args:
            v_bar: (n_states, n_choices) value functions
            states: (n_obs,) state indices

        Returns:
            probs: (n_obs, n_choices) choice probabilities
        """
        # Extract value functions for observed states
        v_obs = v_bar[states]  # (n_obs, n_choices)

        # Softmax to get probabilities
        probs = torch.softmax(v_obs, dim=1)

        return probs

    @abstractmethod
    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood. Must be implemented by subclasses.

        Args:
            params: Model parameters (θ, φ)
            data: Dictionary with 'states', 'actions', 'next_states'

        Returns:
            Negative log-likelihood value
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_fisher_information(self) -> torch.Tensor:
        """
        Compute Fisher information matrix for standard errors.

        Must be implemented by subclasses.

        Returns:
            Fisher information matrix
        """
        raise NotImplementedError

    def predict_proba(self, states: torch.Tensor) -> torch.Tensor:
        """
        Predict choice probabilities for given states.

        Args:
            states: (n_obs,) state indices

        Returns:
            Choice probabilities (n_obs, n_choices)

        Raises:
            ValueError: If model has not been fitted yet
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")

        # This will be implemented by subclasses with specific parameter handling
        raise NotImplementedError("Subclass must implement predict_proba")

    def simulate(
        self,
        initial_states: torch.Tensor,
        n_periods: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate forward paths from initial states.

        Args:
            initial_states: (n_agents,) initial state indices
            n_periods: Number of periods to simulate

        Returns:
            states_path: (n_agents, n_periods+1) simulated states
            actions_path: (n_agents, n_periods) simulated actions

        Raises:
            ValueError: If model has not been fitted yet
        """
        if self.params is None:
            raise ValueError("Model must be fitted before simulation")

        raise NotImplementedError("Subclass must implement simulate")

    def counterfactual(
        self,
        states: torch.Tensor,
        policy_change: dict,
    ) -> dict:
        """
        Perform counterfactual policy analysis.

        Args:
            states: (n_obs,) state indices for evaluation
            policy_change: Dictionary specifying parameter changes

        Returns:
            Dictionary with counterfactual results

        Raises:
            ValueError: If model has not been fitted yet
        """
        if self.params is None:
            raise ValueError("Model must be fitted before counterfactual analysis")

        raise NotImplementedError("Subclass must implement counterfactual")


# ============================================================================
# Utility Specifications
# ============================================================================


class LinearFlowUtility:
    """
    Linear per-period utility: ū_j(x; θ) = x'θ_j

    Common specification with state variables entering linearly.
    """

    def __init__(
        self,
        n_features: int,
        n_choices: int,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize linear utility.

        Args:
            n_features: Dimension of state vector
            n_choices: Number of discrete choices
            device: Device for parameters (auto-detects if None)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.theta = torch.randn(n_features, n_choices, device=self.device) * 0.01

    def compute(
        self,
        states: torch.Tensor,
        choice: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute ū_j(x) for all observations.

        Args:
            states: (n_obs, n_features) state vectors
            choice: If specified, return utility only for this choice.
                   Otherwise return utilities for all choices.

        Returns:
            utilities: (n_obs,) if choice specified, else (n_obs, n_choices)
        """
        if choice is not None:
            return states @ self.theta[:, choice]
        else:
            return states @ self.theta


class ReplacementUtility:
    """
    Rust (1987) bus engine replacement utility.

    ū_0(x; θ) = -θ_1 * x           # Maintain (x = mileage)
    ū_1(x; θ) = -θ_2 - θ_1 * 0     # Replace (pay RC + reset to 0)

    References:
        Rust (1987), Section 3.3: Specification of utility function
    """

    def __init__(
        self,
        theta_maintenance: float = 0.001,
        theta_replacement_cost: float = 10.0,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize replacement utility parameters.

        Args:
            theta_maintenance: θ_1, cost per unit mileage
            theta_replacement_cost: θ_2, replacement cost (RC)
            device: Device for parameters (auto-detects if None)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.theta_maintenance = torch.tensor(theta_maintenance, device=self.device)
        self.theta_replacement_cost = torch.tensor(theta_replacement_cost, device=self.device)

    def compute(
        self,
        state: torch.Tensor,
        choice: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Rust replacement utility.

        Args:
            state: (n_obs,) mileage values
            choice: If 0, return maintain utility. If 1, return replace utility.
                   If None, return both as (n_obs, 2) tensor.

        Returns:
            utilities: (n_obs,) or (n_obs, 2) depending on choice argument
        """
        maintain_utility = -self.theta_maintenance * state
        replace_utility = -self.theta_replacement_cost + torch.zeros_like(state)

        if choice == 0:
            return maintain_utility
        elif choice == 1:
            return replace_utility
        else:
            # Return both
            return torch.stack([maintain_utility, replace_utility], dim=1)

    def get_params(self) -> dict:
        """Return current parameter values."""
        return {
            "theta_maintenance": self.theta_maintenance.item(),
            "theta_replacement_cost": self.theta_replacement_cost.item(),
        }

    def set_params(self, theta_maintenance: float, theta_replacement_cost: float):
        """Update parameter values."""
        self.theta_maintenance = torch.tensor(theta_maintenance, device=self.device)
        self.theta_replacement_cost = torch.tensor(theta_replacement_cost, device=self.device)
