"""
Unit tests for dynamic discrete choice models.

Tests cover:
- Data structures and validation
- Value function iteration
- Transition estimation
- State discretization
- Utility specifications
"""

import torch
import pytest

from torchonometrics.choice.dynamic import (
    DynamicChoiceData,
    DynamicChoiceModel,
    LinearFlowUtility,
    ReplacementUtility,
)
from torchonometrics.choice.transitions import (
    estimate_transition_matrix,
    discretize_state,
    DeepValueFunction,
)


# ============================================================================
# Data Structure Tests
# ============================================================================


def test_dynamic_choice_data_creation():
    """Test DynamicChoiceData creation and to_dict conversion."""
    n_obs = 100
    data = DynamicChoiceData(
        states=torch.randint(0, 10, (n_obs,)),
        actions=torch.randint(0, 2, (n_obs,)),
        next_states=torch.randint(0, 10, (n_obs,)),
        individual_ids=torch.arange(n_obs),
        time_periods=torch.arange(n_obs),
    )

    # Test validation
    data.validate()

    # Test to_dict
    data_dict = data.to_dict()
    assert "states" in data_dict
    assert "actions" in data_dict
    assert len(data_dict["states"]) == n_obs


def test_dynamic_choice_data_validation_errors():
    """Test that validation catches inconsistent data."""
    n_obs = 100

    # Inconsistent lengths
    with pytest.raises(ValueError, match="same length"):
        data = DynamicChoiceData(
            states=torch.randint(0, 10, (n_obs,)),
            actions=torch.randint(0, 2, (n_obs - 1,)),  # Wrong length
            next_states=torch.randint(0, 10, (n_obs,)),
            individual_ids=torch.arange(n_obs),
            time_periods=torch.arange(n_obs),
        )
        data.validate()

    # Negative states
    with pytest.raises(ValueError, match="non-negative"):
        data = DynamicChoiceData(
            states=torch.tensor([-1, 0, 1]),
            actions=torch.tensor([0, 1, 0]),
            next_states=torch.tensor([0, 1, 2]),
            individual_ids=torch.tensor([0, 0, 0]),
            time_periods=torch.tensor([0, 1, 2]),
        )
        data.validate()


# ============================================================================
# Helper class for testing abstract DynamicChoiceModel
# ============================================================================


class ConcreteDynamicChoiceModel(DynamicChoiceModel):
    """Concrete implementation for testing purposes."""

    def _negative_log_likelihood(self, params: torch.Tensor, data: dict):
        """Dummy implementation."""
        return torch.tensor(0.0)

    def _compute_fisher_information(self):
        """Dummy implementation."""
        return torch.eye(2)


# ============================================================================
# Value Function Iteration Tests
# ============================================================================


def test_bellman_operator_simple():
    """Test Bellman operator on a simple 2-state, 2-action problem."""
    n_states = 2
    n_choices = 2
    discount_factor = 0.9

    model = ConcreteDynamicChoiceModel(
        n_states=n_states,
        n_choices=n_choices,
        discount_factor=discount_factor,
        device="cpu",
    )

    # Simple deterministic transition: always stay in same state
    P = torch.zeros(n_states, n_choices, n_states)
    for x in range(n_states):
        for a in range(n_choices):
            P[x, a, x] = 1.0  # Stay in same state

    model.set_transition_probabilities(P)

    # Simple flow utilities
    flow_utility = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (n_states, n_choices)

    # Initialize value functions to zero
    v_bar = torch.zeros(n_states, n_choices)

    # Apply Bellman operator once
    T_v_bar = model._bellman_operator(v_bar, flow_utility)

    # Check that operator increases values (since flow utilities are positive)
    assert torch.all(T_v_bar >= flow_utility)


def test_value_function_convergence():
    """Test that value iteration converges to fixed point."""
    n_states = 5
    n_choices = 2
    discount_factor = 0.9

    model = ConcreteDynamicChoiceModel(
        n_states=n_states,
        n_choices=n_choices,
        discount_factor=discount_factor,
        device="cpu",
    )

    # Random transition matrix
    P = torch.rand(n_states, n_choices, n_states)
    P = P / P.sum(dim=2, keepdim=True)  # Normalize
    model.set_transition_probabilities(P)

    # Random flow utilities
    flow_utility = torch.randn(n_states, n_choices)

    # Solve for value functions
    v_bar = model.solve_value_functions(flow_utility, tol=1e-8)

    # Check that we've reached fixed point
    T_v_bar = model._bellman_operator(v_bar, flow_utility)
    assert torch.allclose(v_bar, T_v_bar, atol=1e-6)


def test_value_iteration_properties():
    """Test mathematical properties of value iteration."""
    n_states = 10
    n_choices = 3
    discount_factor = 0.95

    model = ConcreteDynamicChoiceModel(
        n_states=n_states,
        n_choices=n_choices,
        discount_factor=discount_factor,
        device="cpu",
    )

    # Create valid transition matrix
    P = torch.rand(n_states, n_choices, n_states)
    P = P / P.sum(dim=2, keepdim=True)
    model.set_transition_probabilities(P)

    # Two different flow utilities
    flow_u1 = torch.randn(n_states, n_choices)
    flow_u2 = flow_u1 + 5.0  # Shift by constant

    v_bar1 = model.solve_value_functions(flow_u1)
    v_bar2 = model.solve_value_functions(flow_u2)

    # Property: V(u + c) = V(u) + c/(1-β)
    # Approximately, since we have choice-specific values
    mean_diff = (v_bar2 - v_bar1).mean()
    expected_diff = 5.0 / (1 - discount_factor)
    assert torch.abs(mean_diff - expected_diff) < 1.0


# ============================================================================
# Transition Estimation Tests
# ============================================================================


def test_estimate_transition_matrix_frequency():
    """Test nonparametric frequency estimator for transitions."""
    n_states = 3
    n_choices = 2
    n_obs = 1000

    # Generate deterministic transitions for testing
    # State 0 + action 0 -> always state 1
    # State 0 + action 1 -> always state 2
    # etc.
    states = torch.zeros(n_obs, dtype=torch.long)
    actions = torch.randint(0, n_choices, (n_obs,))
    next_states = actions + 1  # Deterministic rule

    P_hat = estimate_transition_matrix(
        states=states,
        actions=actions,
        next_states=next_states,
        n_states=n_states,
        n_choices=n_choices,
        method="frequency",
    )

    # Check that P_hat[0, 0, 1] ≈ 1 (state 0, action 0 -> state 1)
    assert P_hat[0, 0, 1] > 0.9
    # Check that P_hat[0, 1, 2] ≈ 1 (state 0, action 1 -> state 2)
    assert P_hat[0, 1, 2] > 0.9

    # Check probabilities sum to 1
    assert torch.allclose(P_hat.sum(dim=2), torch.ones(n_states, n_choices), atol=1e-5)


def test_estimate_transition_matrix_stochastic():
    """Test transition estimation with stochastic transitions."""
    n_states = 5
    n_choices = 2
    n_obs = 5000

    # Generate data from known stochastic transition
    # Action 0: stay in place (70%) or move up one state (30%)
    states = torch.randint(0, n_states - 1, (n_obs,))
    actions = torch.zeros(n_obs, dtype=torch.long)
    next_states = torch.where(
        torch.rand(n_obs) < 0.7,
        states,  # Stay
        torch.clamp(states + 1, max=n_states - 1),  # Move up
    )

    P_hat = estimate_transition_matrix(
        states=states,
        actions=actions,
        next_states=next_states,
        n_states=n_states,
        n_choices=n_choices,
        method="frequency",
    )

    # Check approximate recovery of transition probabilities
    # For state 2, action 0: should have ~70% staying, ~30% moving to 3
    if torch.sum((states == 2) & (actions == 0)) > 100:  # Enough observations
        assert 0.6 < P_hat[2, 0, 2] < 0.8  # Stay probability
        assert 0.2 < P_hat[2, 0, 3] < 0.4  # Move probability


# ============================================================================
# State Discretization Tests
# ============================================================================


def test_discretize_state_uniform():
    """Test uniform discretization of continuous states."""
    continuous_state = torch.linspace(0, 100, 1000)
    n_bins = 10

    discrete_state = discretize_state(
        continuous_state=continuous_state,
        n_bins=n_bins,
        method="uniform",
        state_min=0,
        state_max=100,
    )

    # Check that we get exactly n_bins unique values
    assert len(torch.unique(discrete_state)) == n_bins

    # Check that first values are in bin 0, last values in bin 9
    assert discrete_state[0] == 0
    assert discrete_state[-1] == n_bins - 1


def test_discretize_state_quantile():
    """Test quantile discretization."""
    # Create skewed distribution
    continuous_state = torch.exp(torch.randn(1000))
    n_bins = 5

    discrete_state = discretize_state(
        continuous_state=continuous_state,
        n_bins=n_bins,
        method="quantile",
    )

    # Check that each bin has approximately equal count
    bin_counts = torch.bincount(discrete_state, minlength=n_bins)
    # Should be roughly 1000/5 = 200 per bin
    assert torch.all(bin_counts > 150)
    assert torch.all(bin_counts < 250)


def test_discretize_state_bounds():
    """Test that discretization handles edge cases correctly."""
    continuous_state = torch.tensor([0.0, 50.0, 100.0, 150.0])
    n_bins = 10

    discrete_state = discretize_state(
        continuous_state=continuous_state,
        n_bins=n_bins,
        method="uniform",
        state_min=0,
        state_max=100,
    )

    # Value beyond max should be clamped to last bin
    assert discrete_state[-1] == n_bins - 1


# ============================================================================
# Utility Specification Tests
# ============================================================================


def test_linear_flow_utility():
    """Test linear utility specification."""
    n_features = 3
    n_choices = 2
    n_obs = 100

    utility = LinearFlowUtility(n_features, n_choices, device="cpu")

    # Test computation
    states = torch.randn(n_obs, n_features)
    utilities = utility.compute(states)

    assert utilities.shape == (n_obs, n_choices)

    # Test single choice
    utility_0 = utility.compute(states, choice=0)
    assert utility_0.shape == (n_obs,)
    assert torch.allclose(utility_0, utilities[:, 0])


def test_replacement_utility():
    """Test Rust (1987) replacement utility."""
    utility = ReplacementUtility(theta_maintenance=0.001, theta_replacement_cost=10.0, device="cpu")

    # Test on some mileage values
    mileage = torch.tensor([0.0, 100.0, 200.0])

    # Maintain utility should decrease with mileage
    maintain = utility.compute(mileage, choice=0)
    assert maintain[0] > maintain[1] > maintain[2]

    # Replace utility should be constant (doesn't depend on mileage)
    replace = utility.compute(mileage, choice=1)
    assert torch.allclose(replace, replace[0] * torch.ones_like(replace))

    # Test getting both
    both = utility.compute(mileage)
    assert both.shape == (3, 2)
    assert torch.allclose(both[:, 0], maintain)
    assert torch.allclose(both[:, 1], replace)


def test_replacement_utility_params():
    """Test parameter getter/setter for replacement utility."""
    utility = ReplacementUtility(theta_maintenance=0.001, theta_replacement_cost=10.0)

    params = utility.get_params()
    assert abs(params["theta_maintenance"] - 0.001) < 1e-6
    assert abs(params["theta_replacement_cost"] - 10.0) < 1e-6

    # Update parameters
    utility.set_params(theta_maintenance=0.002, theta_replacement_cost=15.0)
    params = utility.get_params()
    assert abs(params["theta_maintenance"] - 0.002) < 1e-6
    assert abs(params["theta_replacement_cost"] - 15.0) < 1e-6


# ============================================================================
# Deep Value Function Tests
# ============================================================================


def test_deep_value_function_forward():
    """Test deep value function network forward pass."""
    state_dim = 10
    n_choices = 3
    batch_size = 32

    network = DeepValueFunction(
        state_dim=state_dim, n_choices=n_choices, hidden_dims=[64, 32]
    )

    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    values = network(states)

    assert values.shape == (batch_size, n_choices)


def test_deep_value_function_gradient():
    """Test that deep value function can compute gradients."""
    state_dim = 5
    n_choices = 2

    network = DeepValueFunction(state_dim=state_dim, n_choices=n_choices)

    # Create dummy loss
    states = torch.randn(10, state_dim)
    values = network(states)
    loss = values.mean()

    # Compute gradient
    loss.backward()

    # Check that parameters have gradients
    for param in network.parameters():
        assert param.grad is not None


# ============================================================================
# Choice Probability Tests
# ============================================================================


def test_compute_choice_probs():
    """Test computation of choice probabilities from value functions."""
    n_states = 5
    n_choices = 3
    n_obs = 100

    model = ConcreteDynamicChoiceModel(
        n_states=n_states, n_choices=n_choices, discount_factor=0.9
    )

    # Create dummy value functions
    v_bar = torch.randn(n_states, n_choices)

    # Random states
    states = torch.randint(0, n_states, (n_obs,))

    probs = model._compute_choice_probs(v_bar, states)

    # Check shape
    assert probs.shape == (n_obs, n_choices)

    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=1), torch.ones(n_obs), atol=1e-5)

    # Check probabilities are non-negative
    assert torch.all(probs >= 0)


def test_choice_probs_argmax():
    """Test that choice probs are highest for highest value."""
    n_states = 3
    n_choices = 3

    model = ConcreteDynamicChoiceModel(
        n_states=n_states, n_choices=n_choices, discount_factor=0.9
    )

    # Create value functions where choice 1 is always best for state 0
    v_bar = torch.tensor([[1.0, 10.0, 2.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    states = torch.tensor([0])
    probs = model._compute_choice_probs(v_bar, states)

    # Choice 1 should have highest probability
    assert torch.argmax(probs[0]) == 1
    assert probs[0, 1] > 0.9  # Should be very high


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_value_iteration_pipeline():
    """Integration test: full pipeline from data to value functions."""
    # Setup simple problem
    n_states = 10
    n_choices = 2
    n_obs = 500

    # Generate synthetic data
    states = torch.randint(0, n_states, (n_obs,))
    actions = torch.randint(0, n_choices, (n_obs,))
    next_states = torch.clamp(states + actions, max=n_states - 1)

    # Estimate transitions
    P_hat = estimate_transition_matrix(
        states=states,
        actions=actions,
        next_states=next_states,
        n_states=n_states,
        n_choices=n_choices,
        method="frequency",
    )

    # Create model
    model = ConcreteDynamicChoiceModel(
        n_states=n_states, n_choices=n_choices, discount_factor=0.9, device="cpu"
    )
    model.set_transition_probabilities(P_hat)

    # Solve value functions
    flow_utility = torch.randn(n_states, n_choices)
    v_bar = model.solve_value_functions(flow_utility, tol=1e-7)

    # Compute choice probabilities
    test_states = torch.arange(n_states)
    probs = model._compute_choice_probs(v_bar, test_states)

    # Check output shapes and validity
    assert v_bar.shape == (n_states, n_choices)
    assert probs.shape == (n_states, n_choices)
    assert torch.allclose(probs.sum(dim=1), torch.ones(n_states), atol=1e-5)
