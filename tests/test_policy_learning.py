import numpy as np
import torch

from trex.policy_learning import policyCATElearner


def test_policy_cate_learner_recovers_linear_signal():
    rng = np.random.default_rng(123)
    n = 1500
    x = rng.normal(size=(n, 2))
    tau = 1.0 + 0.8 * x[:, 0] - 0.5 * x[:, 1]
    baseline = 0.3 - 0.2 * x[:, 0]
    w = rng.binomial(1, 0.5, size=n)
    y = baseline + w * tau + rng.normal(scale=0.5, size=n)

    model = policyCATElearner(cost=0.0, sigma=1.0, distribution="logistic", maxiter=40, tol=1e-10, device="cpu")
    model.fit(x, y, w, propensity=0.5)
    tau_hat = model.predict(x).detach().cpu().numpy()

    corr = np.corrcoef(tau_hat, tau)[0, 1]
    assert corr > 0.9


def test_policy_cate_uniform_matches_transformed_outcome_ols():
    rng = np.random.default_rng(321)
    n = 1000
    x = rng.normal(size=(n, 1))
    tau = 0.5 + 1.2 * x[:, 0]
    w = rng.binomial(1, 0.5, size=n)
    y = 0.1 + w * tau + rng.normal(scale=0.3, size=n)
    y_star = y * (w / 0.5 - (1 - w) / 0.5)

    X = np.column_stack([np.ones(n), x[:, 0]])
    beta_ols, *_ = np.linalg.lstsq(X, y_star, rcond=None)

    model = policyCATElearner(distribution="uniform", sigma=1.0, maxiter=5, tol=0.0, device="cpu")
    model.fit(x, y, w, propensity=0.5)
    beta_trex = model.params["coef"].detach().cpu().numpy()

    assert np.allclose(beta_trex, beta_ols, atol=1e-4)
    assert torch.all(model.predict_policy(x) == (model.predict(x) >= 0).to(torch.int64))
