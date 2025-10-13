import torch
import numpy as np

from torchonometrics.choice import BinaryLogit, BinaryProbit, MultinomialLogit, NestedLogit


def test_binary_logit_smoke():
    # Smoke test to ensure the BinaryLogit model can be instantiated and fitted.
    X = torch.randn(100, 2)
    y = (torch.rand(100) < 0.5).to(torch.float32)
    model = BinaryLogit()
    model.fit(X, y)
    assert model.params is not None

    # Test counterfactual
    X_new = torch.randn(100, 2)
    counterfactuals = model.counterfactual(X_new)
    assert "market_share_original" in counterfactuals
    assert "market_share_counterfactual" in counterfactuals
    assert "change_in_market_share" in counterfactuals


def test_binary_logit_dgp():
    # Test with a known data generating process
    n_samples = 1000
    n_features = 2
    true_params = torch.tensor([0.5, -1.5, 1.0])

    # Generate data
    X = torch.randn(n_samples, n_features)
    X = torch.cat([torch.ones(n_samples, 1), X], dim=1)  # Add intercept
    logits = X @ true_params
    probs = torch.sigmoid(logits)
    y = (torch.rand(n_samples) < probs).to(torch.float32)

    # Fit model
    model = BinaryLogit()
    model.fit(X, y)

    # Check if the estimated parameters are close to the true parameters
    estimated_params = model.params["coef"]
    assert torch.allclose(estimated_params, true_params, atol=0.2)


def test_binary_probit_dgp():
    # Test with a known data generating process
    n_samples = 1000
    n_features = 2
    true_params = torch.tensor([0.5, -1.5, 1.0])

    # Generate data
    X = torch.randn(n_samples, n_features)
    X = torch.cat([torch.ones(n_samples, 1), X], dim=1)  # Add intercept
    logits = X @ true_params
    norm_dist = torch.distributions.Normal(0, 1)
    probs = norm_dist.cdf(logits)
    y = (torch.rand(n_samples) < probs).to(torch.float32)

    # Fit model
    model = BinaryProbit()
    model.fit(X, y)

    # Check if the estimated parameters are close to the true parameters
    estimated_params = model.params["coef"]
    assert torch.allclose(estimated_params, true_params, atol=0.2)


def test_multinomial_logit_dgp():
    # Test with a known data generating process
    n_samples = 1000
    n_features = 2
    n_choices = 3
    true_params = torch.tensor([[0.5, -1.5, 0.0], [1.0, -0.5, 0.0], [0.5, 0.5, 0.0]])  # Last choice is base

    # Generate data
    X = torch.randn(n_samples, n_features)
    X = torch.cat([torch.ones(n_samples, 1), X], dim=1)  # Add intercept
    logits = X @ true_params
    probs = torch.nn.functional.softmax(logits, dim=1)
    y = torch.multinomial(probs, 1).squeeze(1)
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=n_choices).to(torch.float32)

    # Fit model
    model = MultinomialLogit()
    # We only need to estimate parameters for n_choices - 1
    init_params = torch.randn(n_features + 1, n_choices - 1) * 0.01
    model.fit(X, y_one_hot, init_params=init_params)

    # Check if the estimated parameters are close to the true parameters
    estimated_params = model.params["coef"]
    # Add back the base choice for comparison
    estimated_params_full = torch.cat([estimated_params, torch.zeros(n_features + 1, 1)], dim=1)
    # The parameters are identified up to a constant, so we check the difference
    diff = estimated_params_full - estimated_params_full[:, -1].unsqueeze(1)
    true_diff = true_params - true_params[:, -1].unsqueeze(1)
    assert torch.allclose(diff, true_diff, atol=0.3)


def test_nested_logit_smoke():
    # Smoke test for NestedLogit
    nesting_structure = {
        'nest1': [0, 1],
        'nest2': [2, 3]
    }
    model = NestedLogit(nesting_structure)
    assert model is not None
