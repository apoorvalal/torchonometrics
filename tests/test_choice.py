import torch

from torchonometrics.choice import BinaryLogit, BinaryProbit, MultinomialLogit, LowRankLogit


def test_binary_logit_smoke():
    # Smoke test to ensure the BinaryLogit model can be instantiated and fitted.
    X = torch.randn(100, 2)
    y = (torch.rand(100) < 0.5).to(torch.float32)
    model = BinaryLogit(device="cpu")
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
    model = BinaryLogit(device="cpu")
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
    model = BinaryProbit(device="cpu")
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
    model = MultinomialLogit(device="cpu")
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

    # Generate true parameters with zero-sum constraint
    true_A = torch.randn(n_users, rank)
    true_B = torch.randn(n_items, rank)
    true_theta = true_A @ true_B.T
    true_theta = true_theta - true_theta.mean(dim=1, keepdim=True)  # Zero-sum constraint

    # Generate data
    user_indices = torch.randint(0, n_users, (n_samples,))

    # For simplicity, assume all items are in the choice set for each user
    probs = torch.nn.functional.softmax(true_theta[user_indices], dim=1)
    item_indices = torch.multinomial(probs, 1).squeeze(1)

    # Fit model with small regularization
    model = LowRankLogit(rank, n_users, n_items, lam=0.01, device="cpu")
    model.fit(user_indices, item_indices)

    # Check model fitted successfully and has expected structure
    assert model.params is not None
    assert "theta" in model.params
    assert model.params["theta"].shape == (n_users, n_items)

    # Check zero-sum constraint is satisfied
    assert torch.allclose(model.params["theta"].mean(dim=1), torch.zeros(n_users), atol=1e-5)


def test_low_rank_logit_varying_assortments():
    # Test LowRankLogit with varying choice sets
    n_users = 30
    n_items = 15
    rank = 2
    n_samples = n_users * 20
    assortment_size = 8

    # Generate true parameters with zero-sum constraint
    true_A = torch.randn(n_users, rank)
    true_B = torch.randn(n_items, rank)
    true_theta = true_A @ true_B.T
    true_theta = true_theta - true_theta.mean(dim=1, keepdim=True)

    # Generate data with varying assortments
    user_indices = torch.randint(0, n_users, (n_samples,))
    assortments = torch.zeros(n_samples, n_items)

    # Each observation has a random subset of items available
    for i in range(n_samples):
        available_items = torch.randperm(n_items)[:assortment_size]
        assortments[i, available_items] = 1

    # Generate choices given assortments
    item_indices = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_samples):
        user = user_indices[i]
        available = assortments[i].bool()
        utilities = true_theta[user, available]
        probs = torch.nn.functional.softmax(utilities, dim=0)
        available_items = torch.where(available)[0]
        chosen_idx = torch.multinomial(probs, 1).item()
        item_indices[i] = available_items[chosen_idx]

    # Fit model with small regularization
    model = LowRankLogit(rank, n_users, n_items, lam=0.01, maxiter=10000, device="cpu")
    model.fit(user_indices, item_indices, assortments)

    # Test prediction on new assortments
    test_users = torch.randint(0, n_users, (10,))
    test_assortments = torch.zeros(10, n_items)
    for i in range(10):
        available_items = torch.randperm(n_items)[:assortment_size]
        test_assortments[i, available_items] = 1

    probs = model.predict_proba(test_users, test_assortments)

    # Check that probabilities sum to 1 and unavailable items have 0 probability
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)
    assert torch.allclose(
        probs[test_assortments == 0],
        torch.zeros(probs[test_assortments == 0].shape),
        atol=1e-5,
    )


def test_low_rank_logit_counterfactual():
    # Test counterfactual analysis with assortment changes
    n_users = 20
    n_items = 10
    rank = 2

    # Fit a simple model
    n_samples = n_users * 15
    true_A = torch.randn(n_users, rank)
    true_B = torch.randn(n_items, rank)
    true_theta = true_A @ true_B.T
    true_theta = true_theta - true_theta.mean(dim=1, keepdim=True)

    user_indices = torch.randint(0, n_users, (n_samples,))
    assortments = torch.ones(n_samples, n_items)
    probs = torch.nn.functional.softmax(true_theta[user_indices], dim=1)
    item_indices = torch.multinomial(probs, 1).squeeze(1)

    model = LowRankLogit(rank, n_users, n_items, lam=0.01, device="cpu")
    model.fit(user_indices, item_indices, assortments)

    # Create counterfactual scenario: remove items 7, 8, 9 from assortment
    test_users = torch.arange(n_users)  # All users
    baseline_assortments = torch.ones(n_users, n_items)
    counterfactual_assortments = torch.ones(n_users, n_items)
    counterfactual_assortments[:, 7:10] = 0  # Remove items 7, 8, 9

    # Test without revenues
    results = model.counterfactual(
        test_users, baseline_assortments, counterfactual_assortments
    )

    # Check required keys are present
    assert "baseline_probs" in results
    assert "counterfactual_probs" in results
    assert "baseline_market_share" in results
    assert "counterfactual_market_share" in results
    assert "market_share_change" in results
    assert "baseline_choices" in results
    assert "counterfactual_choices" in results

    # Check shapes
    assert results["baseline_probs"].shape == (n_users, n_items)
    assert results["counterfactual_probs"].shape == (n_users, n_items)
    assert results["baseline_market_share"].shape == (n_items,)
    assert results["market_share_change"].shape == (n_items,)

    # Check that removed items have zero probability in counterfactual
    assert torch.allclose(
        results["counterfactual_probs"][:, 7:10],
        torch.zeros(n_users, 3),
        atol=1e-5,
    )

    # Check that removed items have negative market share change
    assert torch.all(results["market_share_change"][7:10] < 0)

    # Check that some other items gained market share (substitution effect)
    assert torch.any(results["market_share_change"][:7] > 0)

    # Test with revenues
    item_revenues = torch.tensor([10.0, 15.0, 20.0, 12.0, 18.0, 25.0, 30.0, 5.0, 8.0, 22.0])
    results_with_revenue = model.counterfactual(
        test_users, baseline_assortments, counterfactual_assortments, item_revenues
    )

    # Check revenue keys are present
    assert "baseline_expected_revenue" in results_with_revenue
    assert "counterfactual_expected_revenue" in results_with_revenue
    assert "revenue_change" in results_with_revenue
    assert "revenue_change_pct" in results_with_revenue

    # Revenue should be positive
    assert results_with_revenue["baseline_expected_revenue"] > 0
    assert results_with_revenue["counterfactual_expected_revenue"] > 0

    # Since we removed low-revenue items (5, 8), revenue might increase or decrease
    # depending on substitution patterns
    print(
        f"Revenue change: {results_with_revenue['revenue_change']:.2f} "
        f"({results_with_revenue['revenue_change_pct']:.2%})"
    )
