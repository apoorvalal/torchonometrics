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
import torch.nn as nn

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
        # Initialize these first to prevent AttributeError during super().__init__ calls
        # that might try to access self.utility_fn via self.to()
        self.utility_fn: Optional[nn.Module] = None
        self.utility_params: Optional[dict] = None
        
        super().__init__(optimizer=optimizer, maxiter=maxiter, tol=tol, device=device)
        self.n_states = n_states
        self.n_choices = n_choices
        self.discount_factor = discount_factor
        self.transition_type = transition_type

        # Will be set by set_transition_probabilities()
        self.transition_matrix: Optional[torch.Tensor] = None
        self.transition_params: Optional[dict] = None

    def to(self, device: Union[torch.device, str]) -> "DynamicChoiceModel":
        """Move model and transition matrix to specified device."""
        super().to(device)
        if self.transition_matrix is not None:
            self.transition_matrix = self.transition_matrix.to(self.device)
        if self.utility_fn is not None and isinstance(self.utility_fn, nn.Module):
            self.utility_fn.to(self.device)
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
        utility_fn: nn.Module,
        utility_params: Optional[dict] = None,
    ) -> None:
        """
        Specify per-period utility ū(x, a; θ).

        Args:
            utility_fn: Function computing u(x, a; θ)
            utility_params: Initial values for θ
        """
        self.utility_fn = utility_fn.to(self.device)
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
        continuation_value = torch.einsum(
            "xay,y->xa", 
            self.transition_matrix.to(dtype=emax.dtype), 
            emax
        )

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
        v_bar = torch.zeros(
            self.n_states, 
            self.n_choices, 
            device=self.device, 
            dtype=flow_utility.dtype
        )

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


class RustNFP(DynamicChoiceModel):
    """
    Rust (1987) full maximum likelihood via nested fixed point (NFP).

    At each parameter guess (θ, φ):
    1. Solve for value functions v̄_j(x; θ, φ) via contraction mapping
    2. Compute choice probabilities P(a|x; θ, φ)
    3. Evaluate likelihood L(θ, φ; data)
    4. Update parameters

    References:
        Rust, J. (1987). "Optimal replacement of GMC bus engines:
        An empirical model of Harold Zurcher." Econometrica, 55(5), 999-1033.
    """

    def __init__(
        self,
        n_states: int,
        n_choices: int,
        discount_factor: float,
        transition_type: Literal["parametric", "nonparametric"] = "nonparametric",
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        maxiter: int = 1000,
        tol: float = 1e-6,
        device: Optional[Union[torch.device, str]] = None,
        estimate_transitions: bool = False,
    ):
        """
        Initialize RustNFP model.

        Args:
            n_states: Number of discrete states in X
            n_choices: Number of discrete actions in J
            discount_factor: β ∈ (0,1), typically 0.95-0.9999
            transition_type: How to model P(x'|x,a). Currently only "nonparametric" is supported.
            optimizer: PyTorch optimizer class
            maxiter: Maximum iterations for optimization
            tol: Convergence tolerance for optimization
            device: Device for computations (auto-detects if None)
            estimate_transitions: If True, transition parameters φ are estimated.
                                  Otherwise, they are assumed known (set via set_transition_probabilities).
        """
        super().__init__(
            n_states=n_states,
            n_choices=n_choices,
            discount_factor=discount_factor,
            transition_type=transition_type,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            device=device,
        )
        self.estimate_transitions = estimate_transitions
        self.params_dict = {}

    def _unpack_params(self, params: torch.Tensor) -> tuple[dict, Optional[torch.Tensor]]:
        """
        Unpack concatenated parameter tensor into utility (theta) and transition (phi) parameters.
        This method will need to be customized based on the actual utility and transition functions.
        For now, it assumes utility params are directly in `params`.
        """
        # Placeholder: This will need to be refined based on the actual utility_fn's parameters
        # and if transition_params are also optimized.
        # For Rust's bus model, theta has 2 elements: maintenance cost and replacement cost.
        if isinstance(self.utility_fn, ReplacementUtility):
            # Assuming params is [theta_maintenance, theta_replacement_cost]
            theta_dict = {
                "theta_maintenance": params[0],
                "theta_replacement_cost": params[1]
            }
            phi = None  # Not estimating transitions in this simple case
        elif isinstance(self.utility_fn, LinearFlowUtility):
            theta_dict = {"theta": params.reshape(self.utility_fn.theta.shape)}
            phi = None # Placeholder for transition params if they were to be estimated
        else:
            raise NotImplementedError("Parameter unpacking not implemented for this utility function.")

        return theta_dict, phi

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        """
        Compute -log L(θ, φ | data) via nested fixed point.

        Args:
            params: Concatenated [theta; phi] if both are estimated.
                    Otherwise, just utility parameters.
            data: Dict with keys 'states', 'actions', 'next_states'

        Returns:
            Negative log-likelihood
        """
        theta_dict, phi = self._unpack_params(params)

        if isinstance(self.utility_fn, ReplacementUtility):
            flow_utility = self.utility_fn.forward(
                state=torch.arange(self.n_states, device=self.device),
                theta_maintenance=theta_dict["theta_maintenance"],
                theta_replacement_cost=theta_dict["theta_replacement_cost"],
            ) # (n_states, n_choices)
        elif isinstance(self.utility_fn, LinearFlowUtility):
            if 'all_states_features' not in data:
                raise ValueError("LinearFlowUtility requires 'all_states_features' in data for NFP likelihood calculation.")
            self.utility_fn.theta.data = theta_dict["theta"].to(self.utility_fn.device)
            flow_utility = self.utility_fn.forward(states=data['all_states_features'])
        else:
            raise NotImplementedError("Flow utility computation not implemented for this utility function.")


        # Inner loop: solve for value functions
        v_bar = self.solve_value_functions(flow_utility=flow_utility, tol=self.tol, max_iter=10000) # Use a higher max_iter for inner loop

        # Compute choice probabilities
        choice_probs = self._compute_choice_probs(v_bar, data['states'])

        # Likelihood of observed actions
        # Ensure we don't take log of zero, add a small epsilon
        log_likelihood = torch.sum(torch.log(choice_probs[
            range(len(data['actions'])), data['actions']
        ] + 1e-10))

        # Add transition likelihood if φ unknown (not implemented yet for RustNFP in Phase 2)
        if self.estimate_transitions:
            raise NotImplementedError("Transition parameter estimation not yet implemented for RustNFP.")

        return -log_likelihood

    def _compute_fisher_information(self) -> torch.Tensor:
        """
        Compute Fisher information matrix for standard errors.
        For NFP, this typically involves the Hessian of the likelihood function.
        """
        raise NotImplementedError("Fisher information for RustNFP not yet implemented.")

    def predict_proba(self, states: torch.Tensor) -> torch.Tensor:
        """
        Predict choice probabilities for given states after model fitting.

        Args:
            states: (n_obs,) state indices for prediction

        Returns:
            Choice probabilities (n_obs, n_choices)
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")

        theta_dict, _ = self._unpack_params(self.params)

        if isinstance(self.utility_fn, ReplacementUtility):
            flow_utility = self.utility_fn.forward(
                state=torch.arange(self.n_states, device=self.device),
                theta_maintenance=theta_dict["theta_maintenance"],
                theta_replacement_cost=theta_dict["theta_replacement_cost"],
            )
        elif isinstance(self.utility_fn, LinearFlowUtility):
            if not hasattr(self, 'all_states_features'):
                raise ValueError("LinearFlowUtility predict_proba requires 'all_states_features' to be set during fit.")
            self.utility_fn.theta.data = theta_dict["theta"].to(self.utility_fn.device)
            flow_utility = self.utility_fn.forward(states=self.all_states_features)
        else:
            raise NotImplementedError("Predict_proba not implemented for this utility function.")

        v_bar = self.solve_value_functions(flow_utility=flow_utility, tol=self.tol)
        return self._compute_choice_probs(v_bar, states)

    def simulate(
        self,
        initial_states: torch.Tensor,
        n_periods: int,
        rng: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate forward paths from initial states after model fitting.

        Args:
            initial_states: (n_agents,) initial state indices
            n_periods: Number of periods to simulate
            rng: Optional[torch.Generator] for reproducibility

        Returns:
            states_path: (n_agents, n_periods+1) simulated states
            actions_path: (n_agents, n_periods) simulated actions
        """
        if self.params is None or self.transition_matrix is None:
            raise ValueError("Model must be fitted and transition matrix set before simulation")

        theta_dict, _ = self._unpack_params(self.params)

        if isinstance(self.utility_fn, ReplacementUtility):
            flow_utility = self.utility_fn.forward(
                state=torch.arange(self.n_states, device=self.device),
                theta_maintenance=theta_dict["theta_maintenance"],
                theta_replacement_cost=theta_dict["theta_replacement_cost"],
            )
        elif isinstance(self.utility_fn, LinearFlowUtility):
            if not hasattr(self, 'all_states_features'):
                raise ValueError("LinearFlowUtility simulate requires 'all_states_features' to be set during fit.")
            self.utility_fn.theta.data = theta_dict["theta"].to(self.utility_fn.device)
            flow_utility = self.utility_fn.forward(states=self.all_states_features)
        else:
            raise NotImplementedError("Simulate not implemented for this utility function.")

        v_bar = self.solve_value_functions(flow_utility=flow_utility, tol=self.tol)
        choice_probs_all_states = self._compute_choice_probs(v_bar, torch.arange(self.n_states, device=self.device))

        n_agents = len(initial_states)
        states_path = torch.zeros(n_agents, n_periods + 1, dtype=torch.long, device=self.device)
        actions_path = torch.zeros(n_agents, n_periods, dtype=torch.long, device=self.device)

        current_states = initial_states.to(self.device)
        states_path[:, 0] = current_states

        for t in range(n_periods):
            # Sample action based on choice probabilities for current states
            current_choice_probs = choice_probs_all_states[current_states]
            chosen_actions = torch.multinomial(current_choice_probs, 1, generator=rng).squeeze(1)
            actions_path[:, t] = chosen_actions

            # Sample next state based on chosen action and transition matrix
            # P(x'|x,a) is self.transition_matrix[current_states, chosen_actions, :]
            next_state_probs = self.transition_matrix[current_states, chosen_actions, :]
            current_states = torch.multinomial(next_state_probs, 1, generator=rng).squeeze(1)
            states_path[:, t+1] = current_states

        return states_path, actions_path

    def counterfactual(
        self,
        data: DynamicChoiceData,
        policy_change: dict,
        rng: Optional[torch.Generator] = None,
    ) -> dict:
        """
        Perform counterfactual policy analysis by changing utility parameters.

        Args:
            data: DynamicChoiceData object containing states for simulation.
            policy_change: Dictionary specifying parameter changes (e.g., {'theta_maintenance': 0.0005})
            rng: Optional[torch.Generator] for reproducibility of simulation

        Returns:
            Dictionary with counterfactual results (e.g., simulated states, actions)
        """
        if self.params is None or self.transition_matrix is None:
            raise ValueError("Model must be fitted and transition matrix set before counterfactual analysis")

        # Store original parameters
        original_params = self.params.clone().detach()
        original_utility_params = self.utility_fn.get_params() if hasattr(self.utility_fn, 'get_params') else None

        # Apply policy change to utility function directly for simplicity for now
        # In a more complex scenario, we'd adjust the 'params' tensor and refit or re-solve
        if isinstance(self.utility_fn, ReplacementUtility):
            # Store original parameters
            original_maintenance_param = self.utility_fn.theta_maintenance.data.clone()
            original_replacement_param = self.utility_fn.theta_replacement_cost.data.clone()

            # Apply policy change
            new_maintenance = policy_change.get(
                "theta_maintenance", original_maintenance_param.item()
            )
            new_replacement_cost = policy_change.get(
                "theta_replacement_cost", original_replacement_param.item()
            )

            counterfactual_flow_utility = self.utility_fn.forward(
                state=torch.arange(self.n_states, device=self.device),
                theta_maintenance=torch.tensor(new_maintenance, device=self.device, dtype=torch.float64),
                theta_replacement_cost=torch.tensor(new_replacement_cost, device=self.device, dtype=torch.float64),
            )

            # Solve value functions under the new policy
            counterfactual_v_bar = self.solve_value_functions(
                flow_utility=counterfactual_flow_utility, tol=self.tol
            )

            # Temporarily update the utility function's parameters for simulation
            self.utility_fn.theta_maintenance.data = torch.tensor(new_maintenance, device=self.device, dtype=torch.float64)
            self.utility_fn.theta_replacement_cost.data = torch.tensor(new_replacement_cost, device=self.device, dtype=torch.float64)

            # Simulate under the new policy
            initial_states_for_sim = data.states.unique() # Or use data.individual_ids unique entries
            counterfactual_sim_states, counterfactual_sim_actions = self.simulate(initial_states_for_sim, n_periods=data.time_periods.max().item(), rng=rng)

            # Restore original parameters
            self.utility_fn.theta_maintenance.data = original_maintenance_param
            self.utility_fn.theta_replacement_cost.data = original_replacement_param

            return {
                "simulated_states": counterfactual_sim_states,
                "simulated_actions": counterfactual_sim_actions,
                "counterfactual_v_bar": counterfactual_v_bar,
            }

        elif isinstance(self.utility_fn, LinearFlowUtility):
            # Store original parameters
            original_theta_param = self.utility_fn.theta.data.clone()

            # Apply policy change
            if "theta" in policy_change and isinstance(policy_change["theta"], torch.Tensor):
                counterfactual_theta = policy_change["theta"].to(self.device, dtype=torch.float64)
            else:
                raise ValueError("Policy change for LinearFlowUtility.theta must be a torch.Tensor.")

            if not hasattr(self, 'all_states_features'):
                raise ValueError("LinearFlowUtility counterfactual requires 'all_states_features' to be set during fit.")

            # Temporarily set utility_fn's theta for counterfactual flow utility computation
            self.utility_fn.theta.data = counterfactual_theta
            counterfactual_flow_utility = self.utility_fn.forward(states=self.all_states_features)

            # Solve value functions under the new policy
            counterfactual_v_bar = self.solve_value_functions(
                flow_utility=counterfactual_flow_utility, tol=self.tol
            )

            # Simulate under the new policy (parameters are already set for counterfactual)
            initial_states_for_sim = data.states.unique()
            counterfactual_sim_states, counterfactual_sim_actions = self.simulate(initial_states_for_sim, n_periods=data.time_periods.max().item(), rng=rng)

            # Restore original parameters
            self.utility_fn.theta.data = original_theta_param

            return {
                "simulated_states": counterfactual_sim_states,
                "simulated_actions": counterfactual_sim_actions,
                "counterfactual_v_bar": counterfactual_v_bar,
            }
        else:
            raise NotImplementedError("Counterfactual not implemented for this utility function.")


# ============================================================================
# Utility Specifications
# ============================================================================


class LinearFlowUtility(nn.Module):
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
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device
        self.theta = nn.Parameter(
            torch.randn(n_features, n_choices, device=self.device, dtype=torch.float64) * 0.01
        )

    def forward(
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


class ReplacementUtility(nn.Module):
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
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if isinstance(device, str) else device

        self.theta_maintenance = nn.Parameter(
            torch.tensor(theta_maintenance, device=self.device, dtype=torch.float64)
        )
        self.theta_replacement_cost = nn.Parameter(
            torch.tensor(theta_replacement_cost, device=self.device, dtype=torch.float64)
        )

    def forward(
        self,
        state: torch.Tensor,
        choice: Optional[int] = None,
        theta_maintenance: Optional[torch.Tensor] = None,
        theta_replacement_cost: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Rust replacement utility.

        Args:
            state: (n_obs,) mileage values
            choice: If 0, return maintain utility. If 1, return replace utility.
                   If None, return both as (n_obs, 2) tensor.
            theta_maintenance: Optional[torch.Tensor], overrides self.theta_maintenance
            theta_replacement_cost: Optional[torch.Tensor], overrides self.theta_replacement_cost

        Returns:
            utilities: (n_obs,) or (n_obs, 2) depending on choice argument
        """
        _theta_maintenance = theta_maintenance if theta_maintenance is not None else self.theta_maintenance
        _theta_replacement_cost = theta_replacement_cost if theta_replacement_cost is not None else self.theta_replacement_cost

        maintain_utility = -_theta_maintenance * state
        replace_utility = -_theta_replacement_cost + torch.zeros_like(state)

        if choice == 0:
            return maintain_utility
        elif choice == 1:
            return replace_utility
        else:
            # Return both
            return torch.stack([maintain_utility, replace_utility], dim=1)

    def get_params(self) -> dict:
        """
        Return current parameter values.
        """
        return {
            "theta_maintenance": self.theta_maintenance.item(),
            "theta_replacement_cost": self.theta_replacement_cost.item(),
        }

    def set_params(self, theta_maintenance: Union[float, torch.Tensor], theta_replacement_cost: Union[float, torch.Tensor]):
        """
        Update parameter values.
        """
        if isinstance(theta_maintenance, torch.Tensor):
            self.theta_maintenance.data = theta_maintenance.to(self.device).to(torch.float64)
        else:
            self.theta_maintenance.data = torch.tensor(theta_maintenance, device=self.device, dtype=torch.float64)

        if isinstance(theta_replacement_cost, torch.Tensor):
            self.theta_replacement_cost.data = theta_replacement_cost.to(self.device).to(torch.float64)
        else:
            self.theta_replacement_cost.data = torch.tensor(theta_replacement_cost, device=self.device, dtype=torch.float64)