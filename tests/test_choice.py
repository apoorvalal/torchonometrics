import torch
import numpy as np

from torchonometrics.choice import BinaryLogit, BinaryProbit, MultinomialLogit, LowRankLogit


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


def test_low_rank_logit_dgp():
    # Test with a known data generating process for LowRankLogit
    n_users = 50
    n_items = 20
    rank = 3
    n_samples = n_users * 10

    # Generate true parameters
    true_A = torch.randn(n_users, rank)
    true_B = torch.randn(n_items, rank)
    true_theta = true_A @ true_B.T

    # Generate data
    user_indices = torch.randint(0, n_users, (n_samples,))
    
    # For simplicity, assume all items are in the choice set for each user
    probs = torch.nn.functional.softmax(true_theta[user_indices], dim=1)
    item_indices = torch.multinomial(probs, 1).squeeze(1)

    # Fit model
    model = LowRankLogit(rank, n_users, n_items)
    model.fit(user_indices, item_indices)

    # Check if the estimated parameters are close to the true parameters
    # This is a weak test, as the factors A and B are not uniquely identified, and the optimization is non-convex.
    # We can only check if the product is close.
    est_theta = model.params["A"] @ model.params["B"].T
    assert torch.allclose(est_theta, true_theta, atol=2.0) # High tolerance due to non-convexity
