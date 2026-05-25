import numpy as np
import torch

from trex import TabularDiffusion, TabularTransformer, TabularWGAN, distribution_metrics


def _toy_rows(seed: int = 0, n: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    binary = (x + 0.25 * rng.normal(size=n) > 0).astype(float)
    y = np.maximum(0.0, 2.0 + x + 0.5 * rng.normal(size=n))
    return np.column_stack([binary, x, y])


def test_tabular_transformer_rounds_binary_and_clips_nonnegative():
    rows = _toy_rows()
    transformer = TabularTransformer(
        column_names=["d", "x", "y"],
        binary_columns=["d"],
        nonnegative_columns=["y"],
    ).fit(rows)

    transformed = transformer.transform(rows)
    restored = transformer.inverse_transform(transformed)

    assert transformed.shape == rows.shape
    assert set(np.unique(restored[:, 0])) <= {0.0, 1.0}
    assert np.all(restored[:, 2] >= 0.0)


def test_distribution_metrics_are_zero_for_identical_samples():
    rows = _toy_rows()

    metrics = distribution_metrics(rows, rows)

    assert metrics["marginal_w1_mean"] == 0.0
    assert metrics["marginal_ks_mean"] == 0.0
    assert metrics["mean_l2"] == 0.0
    assert metrics["cov_frobenius"] == 0.0
    assert metrics["corr_frobenius"] == 0.0
    assert metrics["sliced_wasserstein"] == 0.0


def test_tabular_wgan_tiny_smoke_fit_and_sample():
    rows = _toy_rows()
    transformer = TabularTransformer(
        column_names=["d", "x", "y"],
        binary_columns=["d"],
        nonnegative_columns=["y"],
    ).fit(rows)
    train = transformer.transform(rows)
    lower, upper = transformer.transformed_bounds()

    model = TabularWGAN(
        hidden_dims=(16,),
        batch_size=16,
        max_steps=2,
        critic_steps=1,
        binary_dims=transformer.binary_indices,
        lower_bounds=lower,
        upper_bounds=upper,
        seed=1,
        device="cpu",
    )
    model.fit(train)
    sample = model.sample(7)

    assert sample.shape == (7, rows.shape[1])
    assert torch.isfinite(sample).all()


def test_tabular_diffusion_tiny_smoke_fit_and_sample():
    rows = _toy_rows()
    transformer = TabularTransformer(
        column_names=["d", "x", "y"],
        binary_columns=["d"],
        nonnegative_columns=["y"],
    ).fit(rows)
    train = transformer.transform(rows)

    model = TabularDiffusion(
        hidden_dims=(16,),
        n_timesteps=4,
        batch_size=16,
        max_steps=2,
        seed=1,
        device="cpu",
    )
    model.fit(train)
    sample = model.sample(7)

    assert sample.shape == (7, rows.shape[1])
    assert torch.isfinite(sample).all()
