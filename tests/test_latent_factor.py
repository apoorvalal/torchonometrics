import pytest
import torch

from trex import LatentFactorGLM


def _make_gaussian_panel(
    seed: int,
    n_units: int = 18,
    n_times: int = 7,
    n_features: int = 3,
    rank: int = 2,
    noise_scale: float = 0.05,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)

    X = torch.randn(n_units, n_times, n_features, dtype=dtype)
    beta = torch.tensor([0.9, -0.6, 0.35], dtype=dtype)
    unit_factors = torch.randn(n_units, rank, dtype=dtype)
    time_factors = torch.randn(n_times, rank, dtype=dtype)
    offset = 0.15 * torch.randn(n_units, n_times, dtype=dtype)

    latent = unit_factors @ time_factors.T
    mean = torch.einsum("utf,f->ut", X, beta) + latent + offset
    y = mean + noise_scale * torch.randn(n_units, n_times, dtype=dtype)

    unit_ids = torch.repeat_interleave(torch.arange(n_units), n_times)
    time_ids = torch.tile(torch.arange(n_times), (n_units,))

    return {
        "X_dense": X,
        "y_dense": y,
        "mean_dense": mean,
        "offset_dense": offset,
        "X_obs": X.reshape(-1, n_features),
        "y_obs": y.reshape(-1),
        "mean_obs": mean.reshape(-1),
        "offset_obs": offset.reshape(-1),
        "unit_ids": unit_ids,
        "time_ids": time_ids,
        "beta": beta,
        "rank": rank,
        "n_units": n_units,
        "n_times": n_times,
    }


def _make_glm_panel(
    family: str,
    seed: int,
    n_units: int = 20,
    n_times: int = 8,
    n_features: int = 3,
    rank: int = 2,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)

    X = 0.7 * torch.randn(n_units, n_times, n_features, dtype=dtype)
    beta = torch.tensor([0.8, -0.5, 0.3], dtype=dtype)
    unit_factors = 0.8 * torch.randn(n_units, rank, dtype=dtype)
    time_factors = 0.8 * torch.randn(n_times, rank, dtype=dtype)
    offset = 0.15 * torch.randn(n_units, n_times, dtype=dtype)

    eta = torch.einsum("utf,f->ut", X, beta) + unit_factors @ time_factors.T + offset
    if family == "bernoulli":
        eta = eta - 2.2
        mean = torch.sigmoid(eta)
        y = torch.bernoulli(mean)
    elif family == "poisson":
        eta = eta - 0.2
        mean = torch.exp(torch.clamp(eta, max=3.0))
        y = torch.poisson(mean)
    else:
        raise ValueError(f"Unsupported family for test panel: {family}")

    unit_ids = torch.repeat_interleave(torch.arange(n_units), n_times)
    time_ids = torch.tile(torch.arange(n_times), (n_units,))

    return {
        "family": family,
        "X_dense": X,
        "y_dense": y,
        "mean_dense": mean,
        "offset_dense": offset,
        "X_obs": X.reshape(-1, n_features),
        "y_obs": y.reshape(-1),
        "mean_obs": mean.reshape(-1),
        "offset_obs": offset.reshape(-1),
        "unit_ids": unit_ids,
        "time_ids": time_ids,
        "beta": beta,
        "rank": rank,
        "n_units": n_units,
        "n_times": n_times,
    }


def _make_structured_poisson_panel(
    seed: int,
    n_units: int = 60,
    n_times: int = 16,
    rank: int = 1,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)

    X = torch.ones((n_units, n_times, 1), dtype=dtype)
    beta = torch.tensor([-0.2], dtype=dtype)
    unit_factors = 1.2 * torch.randn(n_units, rank, dtype=dtype)
    t = torch.linspace(-1.0, 1.0, n_times, dtype=dtype)
    time_factors = torch.stack([1.6 * t + 0.5 * t.pow(2)], dim=1)[:, :rank]
    offset = 0.02 * torch.randn(n_units, n_times, dtype=dtype)

    eta = torch.einsum("utf,f->ut", X, beta) + unit_factors @ time_factors.T + offset
    mean = torch.exp(torch.clamp(eta, min=-3.0, max=2.5))
    y = torch.poisson(mean)

    return {
        "X_dense": X,
        "y_dense": y,
        "mean_dense": mean,
        "offset_dense": offset,
        "beta": beta,
        "rank": rank,
        "n_units": n_units,
        "n_times": n_times,
    }


def test_latent_factor_glm_recovers_gaussian_coefficients():
    data = _make_gaussian_panel(seed=7)

    model = LatentFactorGLM(
        rank=data["rank"],
        penalty=1e-3,
        beta_penalty=1e-4,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.05, "weight_decay": 0.0},
        maxiter=2000,
        tol=1e-8,
        device="cpu",
    )
    model.fit(
        X=data["X_obs"],
        y=data["y_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        offset=data["offset_obs"],
    )

    pred_mean = model.predict_mean(
        data["X_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        offset=data["offset_obs"],
    )
    pred_index = model.predict_index(
        data["X_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        offset=data["offset_obs"],
    )

    coef_mse = torch.mean((model.params["coef"] - data["beta"]) ** 2)
    mean_mse = torch.mean((pred_mean - data["mean_obs"]) ** 2)

    assert coef_mse < 0.02, f"Latent-factor coefficient MSE too large: {coef_mse}"
    assert mean_mse < 0.03, f"Latent-factor prediction MSE too large: {mean_mse}"
    assert torch.allclose(pred_mean, pred_index)
    assert model.params["unit_factors"].shape == (data["n_units"], data["rank"])
    assert model.params["time_factors"].shape == (data["n_times"], data["rank"])


def test_latent_factor_glm_improves_masked_holdout_prediction():
    data = _make_gaussian_panel(seed=11, n_units=20, n_times=8, noise_scale=0.04)

    torch.manual_seed(12)
    mask = torch.rand(data["n_units"], data["n_times"]) < 0.75
    assert torch.any(~mask)

    fit_kwargs = {
        "penalty": 1e-3,
        "beta_penalty": 1e-4,
        "optimizer": torch.optim.AdamW,
        "optimizer_kwargs": {"lr": 0.05, "weight_decay": 0.0},
        "maxiter": 2200,
        "tol": 1e-8,
        "device": "cpu",
    }

    latent_model = LatentFactorGLM(rank=data["rank"], **fit_kwargs)
    latent_model.fit(
        X=data["X_dense"],
        y=data["y_dense"],
        mask=mask,
        offset=data["offset_dense"],
    )

    baseline_model = LatentFactorGLM(rank=0, **fit_kwargs)
    baseline_model.fit(
        X=data["X_dense"],
        y=data["y_dense"],
        mask=mask,
        offset=data["offset_dense"],
    )

    latent_pred = latent_model.predict_mean(data["X_dense"], offset=data["offset_dense"])
    baseline_pred = baseline_model.predict_mean(data["X_dense"], offset=data["offset_dense"])

    holdout = ~mask
    latent_mse = torch.mean((latent_pred[holdout] - data["mean_dense"][holdout]) ** 2)
    baseline_mse = torch.mean((baseline_pred[holdout] - data["mean_dense"][holdout]) ** 2)

    assert latent_pred.shape == data["y_dense"].shape
    assert latent_mse < baseline_mse * 0.5, (
        f"Latent-factor holdout MSE {latent_mse} did not materially improve on "
        f"rank-0 baseline {baseline_mse}."
    )


def test_latent_factor_glm_rejects_unseen_prediction_levels():
    data = _make_gaussian_panel(seed=19, n_units=8, n_times=5)

    model = LatentFactorGLM(
        rank=data["rank"],
        penalty=1e-3,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.05, "weight_decay": 0.0},
        maxiter=1500,
        tol=1e-8,
        device="cpu",
    )
    model.fit(
        X=data["X_obs"],
        y=data["y_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
    )

    with pytest.raises(ValueError, match="not seen"):
        model.predict_mean(
            data["X_obs"][:1],
            unit_ids=torch.tensor([999]),
            time_ids=torch.tensor([0]),
        )


def test_latent_factor_glm_bernoulli_sparse_matrix_prediction():
    data = _make_glm_panel(family="bernoulli", seed=23, n_units=28, n_times=9)
    assert data["y_dense"].mean() < 0.25

    torch.manual_seed(24)
    train_mask = torch.rand(data["X_obs"].shape[0]) < 0.8
    holdout = ~train_mask

    fit_kwargs = {
        "family": "bernoulli",
        "penalty": 1e-1,
        "beta_penalty": 1e-4,
        "optimizer": torch.optim.AdamW,
        "optimizer_kwargs": {"lr": 0.01, "weight_decay": 0.0},
        "maxiter": 2600,
        "tol": 1e-8,
        "device": "cpu",
    }

    latent_model = LatentFactorGLM(rank=data["rank"], **fit_kwargs)
    latent_model.fit(
        X=data["X_obs"],
        y=data["y_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        mask=train_mask,
        offset=data["offset_obs"],
    )

    baseline_model = LatentFactorGLM(rank=0, **fit_kwargs)
    baseline_model.fit(
        X=data["X_obs"],
        y=data["y_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        mask=train_mask,
        offset=data["offset_obs"],
    )

    latent_prob = latent_model.predict_proba(
        data["X_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        offset=data["offset_obs"],
    )
    baseline_prob = baseline_model.predict_mean(
        data["X_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        offset=data["offset_obs"],
    )
    latent_index = latent_model.predict_index(
        data["X_obs"],
        unit_ids=data["unit_ids"],
        time_ids=data["time_ids"],
        offset=data["offset_obs"],
    )

    latent_brier = torch.mean((latent_prob[holdout] - data["mean_obs"][holdout]) ** 2)
    baseline_brier = torch.mean((baseline_prob[holdout] - data["mean_obs"][holdout]) ** 2)

    assert torch.all((latent_prob >= 0) & (latent_prob <= 1))
    assert torch.allclose(latent_prob, torch.sigmoid(latent_index), atol=1e-6, rtol=1e-6)
    assert latent_brier < baseline_brier * 0.7, (
        f"Bernoulli holdout Brier error {latent_brier} did not materially improve on "
        f"rank-0 baseline {baseline_brier}."
    )


def test_latent_factor_glm_poisson_recovers_count_mean():
    data = _make_structured_poisson_panel(seed=31)

    torch.manual_seed(32)
    mask = torch.rand(data["n_units"], data["n_times"]) < 0.78
    holdout = ~mask

    fit_kwargs = {
        "family": "poisson",
        "penalty": 2e-1,
        "beta_penalty": 1e-4,
        "optimizer": torch.optim.AdamW,
        "optimizer_kwargs": {"lr": 0.01, "weight_decay": 0.0},
        "maxiter": 3200,
        "tol": 1e-8,
        "device": "cpu",
    }

    latent_model = LatentFactorGLM(rank=data["rank"], **fit_kwargs)
    latent_model.fit(
        X=data["X_dense"],
        y=data["y_dense"],
        mask=mask,
        offset=data["offset_dense"],
    )

    baseline_model = LatentFactorGLM(rank=0, **fit_kwargs)
    baseline_model.fit(
        X=data["X_dense"],
        y=data["y_dense"],
        mask=mask,
        offset=data["offset_dense"],
    )

    latent_mean = latent_model.predict_mean(
        data["X_dense"],
        offset=data["offset_dense"],
    )
    baseline_mean = baseline_model.predict_mean(
        data["X_dense"],
        offset=data["offset_dense"],
    )
    latent_index = latent_model.predict_index(
        data["X_dense"],
        offset=data["offset_dense"],
    )

    coef_mse = torch.mean((latent_model.params["coef"] - data["beta"]) ** 2)
    latent_mse = torch.mean((latent_mean[holdout] - data["mean_dense"][holdout]) ** 2)
    baseline_mse = torch.mean((baseline_mean[holdout] - data["mean_dense"][holdout]) ** 2)

    assert torch.all(latent_mean > 0)
    assert torch.allclose(latent_mean, torch.exp(torch.clamp(latent_index, max=20.0)))
    assert coef_mse < 0.08, f"Poisson coefficient MSE too large: {coef_mse}"
    assert latent_mse < baseline_mse * 0.25, (
        f"Poisson holdout MSE {latent_mse} did not materially improve on "
        f"rank-0 baseline {baseline_mse}."
    )


def test_latent_factor_sparse_fit_rejects_ambiguous_dense_prediction():
    data = _make_gaussian_panel(seed=31, n_units=7, n_times=4)
    unit_ids = 100 + 2 * data["unit_ids"]
    time_ids = 20 + 3 * data["time_ids"]

    model = LatentFactorGLM(
        rank=1,
        penalty=1e-3,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"lr": 0.05, "weight_decay": 0.0},
        maxiter=400,
        tol=1e-8,
        device="cpu",
    )
    model.fit(
        X=data["X_obs"],
        y=data["y_obs"],
        unit_ids=unit_ids,
        time_ids=time_ids,
    )

    with pytest.raises(ValueError, match="Dense prediction after sparse fitting"):
        model.predict_mean(data["X_dense"])
