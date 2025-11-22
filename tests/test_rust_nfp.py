import pytest
import torch
from torchonometrics.choice import RustNFP, ReplacementUtility, DynamicChoiceData, estimate_transition_matrix

@pytest.fixture
def rust_nfp_model():
    n_states = 5  # Simplified for testing
    n_choices = 2
    discount_factor = 0.95
    model = RustNFP(
        n_states=n_states,
        n_choices=n_choices,
        discount_factor=discount_factor,
        transition_type="nonparametric", # Only nonparametric supported for now
    )
    return model

@pytest.fixture
def zurcher_data_mini():
    # Mini dataset mimicking Zurcher bus data structure
    # State: mileage (0-4)
    # Action: 0=maintain, 1=replace
    states = torch.tensor([0, 1, 2, 3, 0, 1, 4, 0, 2, 3], dtype=torch.long)
    actions = torch.tensor([0, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.long)
    next_states = torch.tensor([1, 2, 3, 0, 1, 4, 0, 2, 3, 0], dtype=torch.long)
    individual_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    time_periods = torch.tensor([0, 1, 2, 3, 0, 1, 2, 0, 1, 2], dtype=torch.long)

    # For RustNFP with ReplacementUtility, we don't need covariates for flow_utility.
    # But for general cases, we might need all_states_features.
    # Let's create a dummy all_states_features for now, if it's ever needed.
    all_states_features = torch.arange(5).float().unsqueeze(1) # For linear utility, state=mileage

    data = DynamicChoiceData(
        states=states,
        actions=actions,
        next_states=next_states,
        individual_ids=individual_ids,
        time_periods=time_periods,
    )
    return {"data_obj": data, "all_states_features": all_states_features}

def test_rust_nfp_likelihood_computation(rust_nfp_model, zurcher_data_mini):
    model = rust_nfp_model
    data_obj = zurcher_data_mini["data_obj"]
    all_states_features = zurcher_data_mini["all_states_features"]

    # Set utility function (ReplacementUtility)
    model.set_flow_utility(ReplacementUtility(device=model.device))

    # Estimate transition matrix (nonparametric frequency for now)
    transition_matrix = estimate_transition_matrix(
        states=data_obj.states,
        actions=data_obj.actions,
        next_states=data_obj.next_states,
        n_states=model.n_states,
        n_choices=model.n_choices,
        method="frequency",
    )
    model.set_transition_probabilities(transition_matrix)

    # Example parameters for theta_maintenance and theta_replacement_cost
    initial_theta_maintenance = torch.tensor(0.001, requires_grad=True, device=model.device)
    initial_theta_replacement_cost = torch.tensor(10.0, requires_grad=True, device=model.device)
    
    # Concatenate parameters for _negative_log_likelihood
    params = torch.cat([initial_theta_maintenance.unsqueeze(0), initial_theta_replacement_cost.unsqueeze(0)])
    
    # Need to pass all_states_features if utility_fn is LinearFlowUtility,
    # but for ReplacementUtility, it's inferred from n_states.
    # For now, ensure data dict for NLL has it, even if not directly used by ReplacementUtility.
    # This might need refinement based on how _unpack_params and _negative_log_likelihood handle different utilities.
    nll_data = data_obj.to_dict()
    nll_data['all_states_features'] = all_states_features # This will be used if utility_fn were LinearFlowUtility

    # Compute negative log-likelihood
    nll = model._negative_log_likelihood(params, nll_data)

    assert isinstance(nll, torch.Tensor)
    assert nll.ndim == 0  # Should be a scalar
    assert not torch.isnan(nll)
    assert not torch.isinf(nll)
    # Check if gradient can be computed (basic autograd test)
    nll.backward()
    assert initial_theta_maintenance.grad is not None
    assert initial_theta_replacement_cost.grad is not None

def test_rust_nfp_predict_simulate_counterfactual(rust_nfp_model, zurcher_data_mini):
    model = rust_nfp_model
    data_obj = zurcher_data_mini["data_obj"]
    
    # Setup model with utilities and transitions
    model.set_flow_utility(ReplacementUtility(device=model.device))
    transition_matrix = estimate_transition_matrix(
        states=data_obj.states,
        actions=data_obj.actions,
        next_states=data_obj.next_states,
        n_states=model.n_states,
        n_choices=model.n_choices,
        method="frequency",
    )
    model.set_transition_probabilities(transition_matrix)
    
    # Mock fitted parameters
    model.params = torch.tensor([0.001, 10.0], device=model.device)
    
    # Test predict_proba
    probs = model.predict_proba(data_obj.states)
    assert probs.shape == (len(data_obj.states), model.n_choices)
    assert torch.allclose(probs.sum(dim=1), torch.ones(len(data_obj.states), device=model.device))
    
    # Test simulate
    initial_states = torch.tensor([0, 1], device=model.device)
    sim_states, sim_actions = model.simulate(initial_states, n_periods=10)
    assert sim_states.shape == (2, 11)
    assert sim_actions.shape == (2, 10)
    
    # Test counterfactual
    # Reduce replacement cost -> should see more replacements (action 1)
    policy_change = {"theta_replacement_cost": 5.0}
    cf_results = model.counterfactual(data_obj, policy_change)
    
    assert "simulated_states" in cf_results
    assert "simulated_actions" in cf_results
    assert "counterfactual_v_bar" in cf_results
    
    # Check if parameters were restored
    current_params = model.utility_fn.get_params()
    assert current_params["theta_replacement_cost"] == 10.0
