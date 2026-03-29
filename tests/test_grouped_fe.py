import torch

from torchonometrics.grouped_fe import (
    KNNGroupedFixedEffects,
    build_panel_embeddings,
    chunked_knn_indices,
)


def test_build_panel_embeddings_shape():
    torch.manual_seed(0)
    n_units, n_times = 5, 4
    n_obs = n_units * n_times

    X = torch.randn(n_obs, 2)
    y = torch.randn(n_obs)
    unit_ids = torch.repeat_interleave(torch.arange(n_units), n_times)
    time_ids = torch.tile(torch.arange(n_times), (n_units,))

    embeddings, unit_levels, time_levels = build_panel_embeddings(
        X=X,
        y=y,
        unit_ids=unit_ids,
        time_ids=time_ids,
    )

    assert embeddings.shape == (n_units, n_times * (X.shape[1] + 1))
    assert unit_levels.shape[0] == n_units
    assert time_levels.shape[0] == n_times


def test_chunked_knn_indices_excludes_self():
    torch.manual_seed(1)
    embeddings = torch.randn(12, 3)
    neighbor_idx, neighbor_dist = chunked_knn_indices(
        embeddings=embeddings,
        n_neighbors=3,
        block_size=5,
        include_self=False,
    )

    assert neighbor_idx.shape == (12, 3)
    assert neighbor_dist.shape == (12, 3)

    row_ids = torch.arange(embeddings.shape[0])[:, None]
    assert not torch.any(neighbor_idx == row_ids)


def test_knn_grouped_fixed_effects_beta_recovery():
    torch.manual_seed(2)
    n_groups, units_per_group, n_times = 3, 15, 6
    n_units = n_groups * units_per_group
    n_obs = n_units * n_times

    unit_group = torch.repeat_interleave(torch.arange(n_groups), units_per_group)
    unit_ids = torch.repeat_interleave(torch.arange(n_units), n_times)
    time_ids = torch.tile(torch.arange(n_times), (n_units,))

    X = torch.randn(n_obs, 2)
    true_beta = torch.tensor([1.0, -0.4])
    group_time_effects = torch.tensor(
        [
            [1.5, 1.0, 0.5, 0.2, -0.1, -0.4],
            [-0.7, -0.2, 0.3, 0.7, 1.1, 1.4],
            [0.2, 0.6, 1.0, 1.3, 0.9, 0.4],
        ],
        dtype=X.dtype,
    )

    obs_group = unit_group[unit_ids]
    y = (
        X @ true_beta
        + group_time_effects[obs_group, time_ids]
        + 0.05 * torch.randn(n_obs)
    )

    classification_features = (
        group_time_effects[unit_group] + 0.05 * torch.randn(n_units, n_times)
    )

    model = KNNGroupedFixedEffects(
        n_groups=n_groups,
        n_neighbors=5,
        knn_block_size=16,
        kmeans_niter=50,
        use_flash_kmeans=False,
        device="cpu",
    )
    model.fit(
        X=X,
        y=y,
        unit_ids=unit_ids,
        time_ids=time_ids,
        classification_features=classification_features,
    )

    coef_mse = torch.mean((model.params["coef"] - true_beta) ** 2)
    assert coef_mse < 0.02, f"High grouped-FE coefficient MSE: {coef_mse}"
    assert model.group_ids_.shape[0] == n_units
    assert model.group_time_ids_.shape[0] == n_obs
    assert model.neighbor_indices_.shape == (n_units, 5)
