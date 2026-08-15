import numpy as np
import torch

from trex import (
    TabularDiffusion,
    TabularPTGAN,
    TabularTransformer,
    TabularWGAN,
    distribution_metrics,
)


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
    callback_steps = []
    model.fit(train, callback=lambda step, _: callback_steps.append(step))
    sample = model.sample(7)

    assert sample.shape == (7, rows.shape[1])
    assert torch.isfinite(sample).all()
    assert callback_steps == [0, 1, 2]


def test_tabular_wgan_supports_optimistic_adam():
    rows = _toy_rows()
    transformer = TabularTransformer(
        column_names=["d", "x", "y"],
        binary_columns=["d"],
        nonnegative_columns=["y"],
    ).fit(rows)
    train = transformer.transform(rows)

    model = TabularWGAN(
        hidden_dims=(16,),
        batch_size=16,
        max_steps=2,
        critic_steps=1,
        optimizer="optimistic_adam",
        seed=1,
        device="cpu",
    )
    model.fit(train)
    sample = model.sample(7)

    assert sample.shape == (7, rows.shape[1])
    assert torch.isfinite(sample).all()


def test_tabular_ptgan_tiny_smoke_fit_and_tempered_sample():
    rows = _toy_rows()
    transformer = TabularTransformer(
        column_names=["d", "x", "y"],
        binary_columns=["d"],
        nonnegative_columns=["y"],
    ).fit(rows)
    train = transformer.transform(rows)
    lower, upper = transformer.transformed_bounds()
    callback_steps = []

    model = TabularPTGAN(
        hidden_dims=(16,),
        batch_size=16,
        max_steps=2,
        critic_steps=1,
        temperature_ratio=0.5,
        coherency_weight=1.0,
        binary_dims=transformer.binary_indices,
        lower_bounds=lower,
        upper_bounds=upper,
        seed=1,
        device="cpu",
    )
    model.fit(train, callback=lambda step, _: callback_steps.append(step))
    target_sample = model.sample(7, alpha=1.0)
    tempered_sample = model.sample(7, alpha=0.5)

    assert target_sample.shape == (7, rows.shape[1])
    assert tempered_sample.shape == target_sample.shape
    assert torch.isfinite(target_sample).all()
    assert torch.isfinite(tempered_sample).all()
    assert callback_steps == [0, 1, 2]
    assert len(model.history["coherency_penalty"]) == 2
    assert np.isfinite(model.history["critic_grad_norm"]).all()


def test_tabular_ptgan_uses_symmetric_temperature_feature():
    alpha = torch.tensor([[0.2], [0.8], [0.5], [1.0]])
    feature = TabularPTGAN._temperature_feature(alpha)

    assert torch.allclose(feature[0], feature[1])
    assert torch.allclose(feature[2], torch.ones(1))
    assert torch.allclose(feature[3], torch.zeros(1))


def test_tabular_ptgan_coherency_penalty_is_directional_derivative():
    class SumCritic(torch.nn.Module):
        def forward(self, x, context):
            return x.sum(dim=1, keepdim=True)

    model = TabularPTGAN(coherency_weight=1.0, device="cpu")
    model.critic = SumCritic()
    q_tilde = torch.randn(5, 3, requires_grad=True)
    q_difference = torch.randn(5, 3)
    context = torch.zeros(5, 0)
    alpha = torch.rand(5, 1)

    penalty = model._coherency_penalty(
        q_tilde,
        q_difference,
        context,
        alpha,
    )

    expected = q_difference.sum(dim=1).pow(2).mean()
    assert torch.allclose(penalty, expected)


def test_tabular_ptgan_temperature_ratio_one_draws_only_target_temperature():
    model = TabularPTGAN(temperature_ratio=1.0, device="cpu")

    alpha = model._draw_training_alpha(20)

    assert torch.equal(alpha, torch.ones_like(alpha))


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
