"""
Tests for Hotz-Miller CCP Estimator.
"""

import torch
import pytest
from torchonometrics.choice.dynamic import HotzMillerCCP, ReplacementUtility, DynamicChoiceData

def test_hotz_miller_estimation_recovery():
    # Setup
    n_states = 10
    n_choices = 2
    beta = 0.9
    device = "cpu"
    
    true_theta_maint = 0.1
    true_theta_rep = 2.0
    
    # True transition matrix (deterministic for simplicity)
    # P(x'|x, 0) = 1 if x' = min(x+1, 9)
    # P(x'|x, 1) = 1 if x' = 0
    P = torch.zeros(n_states, n_choices, n_states)
    for x in range(n_states):
        # Action 0: Maintain
        next_x_maint = min(x + 1, n_states - 1)
        P[x, 0, next_x_maint] = 1.0
        # Action 1: Replace
        P[x, 1, 0] = 1.0
        
    # True Utility
    utility_fn = ReplacementUtility(
        theta_maintenance=true_theta_maint, 
        theta_replacement_cost=true_theta_rep,
        device=device
    )
    
    # Solve for true Value Functions and CCPs
    model_true = HotzMillerCCP(n_states, n_choices, beta, device=device)
    model_true.set_transition_probabilities(P)
    model_true.set_flow_utility(utility_fn) 
    
    flow_u = utility_fn.forward(torch.arange(n_states, dtype=torch.float64, device=device))
    v_bar_true = model_true.solve_value_functions(flow_u)
    true_ccps = model_true._compute_choice_probs(v_bar_true, torch.arange(n_states, device=device))
    
    # Simulate Data
    n_agents = 200
    n_periods = 100
    
    states = torch.zeros(n_agents * n_periods, dtype=torch.long)
    actions = torch.zeros(n_agents * n_periods, dtype=torch.long)
    next_states = torch.zeros(n_agents * n_periods, dtype=torch.long)
    
    current_states = torch.randint(0, n_states, (n_agents,))
    
    idx = 0
    for t in range(n_periods):
        # Choose actions
        probs = true_ccps[current_states]
        chosen_actions = torch.multinomial(probs, 1).squeeze()
        
        # Transitions
        # Use P to sample next states
        # Since P is deterministic here, we can just lookup
        # But for generality use multinomial
        # Gather P for (current_states, chosen_actions) -> (n_agents, n_states)
        transition_probs = P[current_states, chosen_actions, :]
        next_s_indices = torch.multinomial(transition_probs, 1).squeeze()
        
        # Store
        start = idx
        end = idx + n_agents
        states[start:end] = current_states
        actions[start:end] = chosen_actions
        next_states[start:end] = next_s_indices
        
        current_states = next_s_indices
        idx = end
        
    data = DynamicChoiceData(
        states=states,
        actions=actions,
        next_states=next_states,
        individual_ids=torch.arange(len(states)), # Dummy
        time_periods=torch.arange(len(states)), # Dummy
    )
    
    # Estimate
    # Initialize with wrong parameters
    est_utility = ReplacementUtility(
        theta_maintenance=0.05, # Start away from 0.1
        theta_replacement_cost=1.0, # Start away from 2.0
        device=device
    )
    
    estimator = HotzMillerCCP(
        n_states=n_states,
        n_choices=n_choices,
        discount_factor=beta,
        optimizer=torch.optim.LBFGS,
        maxiter=100, 
        tol=1e-5,
        device=device
    )
    estimator.set_transition_probabilities(P) # Assume known transitions for clean test
    estimator.set_flow_utility(est_utility)
    
    # Fit
    estimator.fit(data, verbose=True)
    
    # Check results
    params = estimator.utility_fn.get_params()
    print(f"True: {true_theta_maint}, {true_theta_rep}")
    print(f"Est: {params['theta_maintenance']}, {params['theta_replacement_cost']}")
    
    # Tolerances depend on sample size. With 20k obs, should be decent.
    assert abs(params['theta_maintenance'] - true_theta_maint) < 0.05
    assert abs(params['theta_replacement_cost'] - true_theta_rep) < 0.5
