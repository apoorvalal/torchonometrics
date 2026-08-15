import numpy as np
import torch

from trex import AntitonicScoreMatchingRegression, estimate_antitonic_score


def test_estimate_antitonic_score_is_decreasing():
    rng = np.random.default_rng(0)
    residuals = rng.standard_t(df=3, size=180)

    score = estimate_antitonic_score(
        residuals,
        symmetric=False,
        k=360,
        kernel_grid_size=512,
    )

    grid = np.linspace(np.quantile(residuals, 0.05), np.quantile(residuals, 0.95), 64)
    values = score(grid)

    assert np.all(np.diff(values) <= 1e-8)
    assert np.all(np.isfinite(values))


def test_antitonic_score_matching_regression_smoke_fit_predict_and_infer():
    rng = np.random.default_rng(1)
    n = 160
    X = rng.normal(size=(n, 2))
    beta = np.array([1.5, -0.75])
    y = 0.4 + X @ beta + rng.standard_t(df=3, size=n) * 0.25

    model = AntitonicScoreMatchingRegression(
        pilot="lad",
        alt_iter=1,
        k=360,
        kernel_grid_size=512,
        max_iter=40,
        device="cpu",
    ).fit(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    pred = model.predict(X[:5])
    ci = model.confidence_interval()

    assert pred.shape == (5,)
    assert ci.shape == (3, 2)
    assert np.all(np.isfinite(model.beta_))
    assert np.all(np.isfinite(model.std_errors_))
    assert np.linalg.norm(model.coef_ - beta) < 0.35
