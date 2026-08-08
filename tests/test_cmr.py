import torch

from trex.cmr import (
    MaximumMomentRestriction,
    SieveMinimumDistance,
    hsic,
    mmr_loss,
    polynomial_sieve,
    rbf_kernel,
)


def _residual_moment(prediction, y):
    return prediction - y


def test_rbf_kernel_and_mmr_loss_are_finite():
    torch.manual_seed(0)
    z = torch.linspace(-1, 1, 12)[:, None]
    moments = torch.randn(12, 2)

    kernel, bandwidth = rbf_kernel(z)
    loss = mmr_loss(moments, z)

    assert kernel.shape == (12, 12)
    assert torch.allclose(kernel, kernel.T)
    assert bandwidth > 0
    assert torch.isfinite(loss)
    assert loss >= 0


def test_hsic_detects_dependence_ordering():
    torch.manual_seed(0)
    x = torch.randn(80, 1)
    y_dependent = x + 0.1 * torch.randn(80, 1)
    y_independent = torch.randn(80, 1)

    assert hsic(x, y_dependent) > hsic(x, y_independent)


def test_polynomial_sieve_shape_with_interactions():
    z = torch.randn(10, 2)
    basis = polynomial_sieve(z, degree=2, include_interactions=True)

    # 1 intercept + z1 + z2 + z1^2 + z1*z2 + z2^2
    assert basis.shape == (10, 6)
    assert torch.allclose(basis[:, 0], torch.ones(10))


def test_mmr_estimator_reduces_conditional_moment_loss():
    torch.manual_seed(0)
    n_obs = 120
    z = torch.linspace(-2, 2, n_obs)[:, None]
    t = z + 0.25 * torch.randn(n_obs, 1)
    y = 1.0 + 2.0 * t + 0.2 * torch.randn(n_obs, 1)

    model = torch.nn.Linear(1, 1)
    estimator = MaximumMomentRestriction(
        model=model,
        moment_function=_residual_moment,
        maxiter=40,
        device="cpu",
    )

    initial_loss = mmr_loss(_residual_moment(model(t), y), z).detach()
    estimator.fit(t, y, z)
    final_loss = estimator.params["mmr_loss"]

    assert final_loss < 0.1 * initial_loss
    assert torch.mean((estimator.predict(t) - y).square()) < 0.1


def test_sieve_minimum_distance_fits_linear_moment():
    torch.manual_seed(1)
    n_obs = 120
    z = torch.randn(n_obs, 1)
    t = z + 0.2 * torch.randn(n_obs, 1)
    y = -0.5 + 1.5 * t + 0.15 * torch.randn(n_obs, 1)

    model = torch.nn.Linear(1, 1)
    estimator = SieveMinimumDistance(
        model=model,
        moment_function=_residual_moment,
        degree=2,
        maxiter=40,
        device="cpu",
    )
    estimator.fit(t, y, z)

    assert estimator.params["smd_loss"] < 1e-3
    assert torch.mean((estimator.predict(t) - y).square()) < 0.1
