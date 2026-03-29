"""
Integration tests for torchonometrics functionality
"""
import numpy as np
import torch
import pytest
from scipy import sparse
from torchonometrics.linear import LinearRegression
from torchonometrics.mle import LogisticRegression, PoissonRegression


class TestLinearRegression:
    """Test LinearRegression functionality"""
    
    def test_basic_regression(self):
        """Test basic linear regression without fixed effects"""
        torch.manual_seed(42)
        n, p = 100, 5
        X = torch.randn(n, p)
        true_coef = torch.randn(p)
        y = X @ true_coef + 0.1 * torch.randn(n)
        
        model = LinearRegression(solver="torch", device="cpu")
        model.fit(X, y)
        
        fitted_coef = model.params["coef"]
        mse = torch.mean((true_coef - fitted_coef)**2)
        
        assert mse < 0.01, f"High MSE: {mse}"
        assert fitted_coef.shape == true_coef.shape
    
    def test_prediction(self):
        """Test prediction functionality"""
        torch.manual_seed(42)
        n, p = 100, 3
        X = torch.randn(n, p)
        true_coef = torch.randn(p)
        y = X @ true_coef + 0.1 * torch.randn(n)
        
        model = LinearRegression(device="cpu")
        model.fit(X, y)

        # Test prediction
        y_pred = model.predict(X)
        r_squared = 1 - torch.var(y - y_pred) / torch.var(y)
        
        assert r_squared > 0.9, f"Low R²: {r_squared}"
        assert y_pred.shape == y.shape
    
    def test_standard_errors(self):
        """Test standard error computation"""
        torch.manual_seed(42)
        n, p = 200, 3
        X = torch.randn(n, p)
        true_coef = torch.randn(p)
        y = X @ true_coef + 0.1 * torch.randn(n)
        
        model = LinearRegression(device="cpu")
        model.fit(X, y, se="HC1")
        
        assert "se" in model.params
        assert model.params["se"].shape == model.params["coef"].shape
        assert torch.all(model.params["se"] > 0)


class TestFixedEffects:
    """Test fixed effects functionality"""
    
    def test_single_fixed_effect(self):
        """Test regression with single fixed effect"""
        torch.manual_seed(42)
        n_groups, n_per_group = 20, 10
        n_obs = n_groups * n_per_group
        
        X = torch.randn(n_obs, 3)
        group_ids = torch.repeat_interleave(torch.arange(n_groups), n_per_group)
        
        # Add intercept
        X_with_intercept = torch.cat([torch.ones(n_obs, 1), X], dim=1)
        true_coef = torch.tensor([2.0, 1.0, -0.5, 0.3])
        group_effects = torch.randn(n_groups)[group_ids]
        
        y = X_with_intercept @ true_coef + group_effects + 0.1 * torch.randn(n_obs)
        
        model = LinearRegression(device="cpu")
        model.fit(X_with_intercept, y, fe=[group_ids])
        
        # Check that coefficients are recovered (except intercept which is absorbed)
        fitted_coef = model.params["coef"]
        assert fitted_coef.shape == true_coef.shape
        
        # Non-intercept coefficients should be close
        # TODO: investigate why MSE is higher than expected with fixed effects
        coef_mse = torch.mean((true_coef[1:] - fitted_coef[1:])**2)
        assert coef_mse < 1.0, f"High coefficient MSE: {coef_mse}"
    
    def test_two_way_fixed_effects(self):
        """Test regression with two-way fixed effects"""
        torch.manual_seed(42)
        n_firms, n_years = 20, 5
        n_obs = n_firms * n_years
        
        X = torch.randn(n_obs, 2)
        firm_ids = torch.repeat_interleave(torch.arange(n_firms), n_years)
        year_ids = torch.tile(torch.arange(n_years), (n_firms,))
        
        X_with_intercept = torch.cat([torch.ones(n_obs, 1), X], dim=1)
        true_coef = torch.tensor([3.0, 1.2, -0.7])
        
        firm_effects = torch.randn(n_firms)[firm_ids]
        year_effects = torch.randn(n_years)[year_ids]
        
        y = X_with_intercept @ true_coef + firm_effects + year_effects + 0.1 * torch.randn(n_obs)
        
        model = LinearRegression(device="cpu")
        model.fit(X_with_intercept, y, fe=[firm_ids, year_ids])
        
        fitted_coef = model.params["coef"]
        assert fitted_coef.shape == true_coef.shape


class TestMaximumLikelihood:
    """Test MLE estimators"""
    
    def test_logistic_regression(self):
        """Test logistic regression"""
        torch.manual_seed(42)
        n, p = 300, 3
        X = torch.randn(n, p)
        X_with_intercept = torch.cat([torch.ones(n, 1), X], dim=1)
        true_coef = torch.tensor([0.5, 1.0, -0.8, 0.3])
        
        logits = X_with_intercept @ true_coef
        probs = torch.sigmoid(logits)
        y = torch.bernoulli(probs).to(torch.float32)
        
        model = LogisticRegression(maxiter=50, device="cpu")
        model.fit(X_with_intercept, y)
        
        assert "coef" in model.params
        assert model.params["coef"].shape == true_coef.shape
        
        # Test prediction methods
        y_pred_proba = model.predict_proba(X_with_intercept)
        y_pred = model.predict(X_with_intercept)
        
        assert torch.all((y_pred_proba >= 0) & (y_pred_proba <= 1))
        assert torch.all((y_pred == 0) | (y_pred == 1))
        
        accuracy = torch.mean((y_pred == y).float())
        assert accuracy > 0.6, f"Low accuracy: {accuracy}"
    
    def test_poisson_regression(self):
        """Test Poisson regression"""
        torch.manual_seed(42)
        n, p = 200, 2
        X = torch.randn(n, p)
        X_with_intercept = torch.cat([torch.ones(n, 1), X], dim=1)
        true_coef = torch.tensor([1.0, 0.5, -0.3])
        
        # Generate Poisson counts
        linear_pred = X_with_intercept @ true_coef
        lambda_true = torch.exp(linear_pred)
        y = torch.poisson(lambda_true)
        
        model = PoissonRegression(maxiter=50, device="cpu")
        model.fit(X_with_intercept, y)
        
        assert "coef" in model.params
        assert model.params["coef"].shape == true_coef.shape
        
        # Test prediction
        y_pred = model.predict(X_with_intercept)
        assert torch.all(y_pred >= 0)  # Predictions should be non-negative

    def test_logistic_regression_with_fixed_effects_and_offset(self):
        """Test FE-aware logistic regression with offset and inference."""
        torch.manual_seed(0)
        n_groups, group_size = 30, 20
        n_obs = n_groups * group_size

        X = torch.randn(n_obs, 2)
        group_ids = torch.repeat_interleave(torch.arange(n_groups), group_size)
        offset = 0.25 * torch.randn(n_obs)
        true_coef = torch.tensor([0.8, -0.5])
        group_effects = 0.6 * torch.randn(n_groups)

        eta = X @ true_coef + group_effects[group_ids] + offset
        y = torch.bernoulli(torch.sigmoid(eta)).to(torch.float32)

        model = LogisticRegression(maxiter=80, device="cpu")
        model.fit(X, y, fe=[group_ids], offset=offset)

        coef_mse = torch.mean((model.params["coef"] - true_coef) ** 2)
        assert coef_mse < 0.02, f"High FE-logit coefficient MSE: {coef_mse}"
        assert "fe_coef" in model.params
        assert model.params["fe_coef"][0].shape[0] == n_groups
        assert torch.all(torch.isfinite(model.params["se"]))
        assert torch.all(torch.isfinite(model.params["vcov"]))

        probs = model.predict_proba(X, fe=[group_ids], offset=offset)
        preds = model.predict(X, fe=[group_ids], offset=offset)
        assert probs.shape == y.shape
        assert torch.all((probs >= 0) & (probs <= 1))
        assert torch.all((preds == 0) | (preds == 1))

    def test_poisson_regression_with_fixed_effects_and_hdfe_inference(self):
        """Test FE-aware Poisson regression and hdfe-style diagonal FE SE output."""
        torch.manual_seed(1)
        n_groups, group_size = 25, 30
        n_obs = n_groups * group_size

        X = torch.randn(n_obs, 2)
        group_ids = torch.repeat_interleave(torch.arange(n_groups), group_size)
        true_coef = torch.tensor([0.4, -0.3])
        group_effects = 0.4 * torch.randn(n_groups)

        eta = X @ true_coef + group_effects[group_ids]
        y = torch.poisson(torch.exp(eta))

        model = PoissonRegression(maxiter=80, device="cpu")
        model.fit(X, y, fe=[group_ids], hdfe_index=0)

        coef_mse = torch.mean((model.params["coef"] - true_coef) ** 2)
        assert coef_mse < 0.01, f"High FE-poisson coefficient MSE: {coef_mse}"
        assert "fe_se_diag" in model.params
        assert model.params["fe_se_diag"].shape[0] == n_groups
        assert torch.all(torch.isfinite(model.params["fe_se_diag"]))
        assert torch.all(torch.isfinite(model.params["se"]))
        assert torch.all(torch.isfinite(model.params["vcov"]))

        y_pred = model.predict(X, fe=[group_ids])
        assert torch.all(y_pred >= 0)

    def test_fe_design_matches_id_path(self):
        """Test sparse FE design input matches the ID-vector path."""
        torch.manual_seed(2)
        n_groups, group_size = 20, 25
        n_obs = n_groups * group_size

        X = torch.randn(n_obs, 2)
        group_ids = torch.repeat_interleave(torch.arange(n_groups), group_size)
        true_coef = torch.tensor([0.7, -0.4])
        group_effects = 0.5 * torch.randn(n_groups)

        eta = X @ true_coef + group_effects[group_ids]
        y = torch.bernoulli(torch.sigmoid(eta)).to(torch.float32)

        row_index = torch.arange(n_obs).numpy()
        csr = sparse.csr_matrix(
            (
                np.ones(n_obs),
                (row_index, group_ids.numpy()),
            ),
            shape=(n_obs, n_groups),
        )

        model_ids = LogisticRegression(maxiter=80, device="cpu")
        model_ids.fit(X, y, fe=[group_ids])

        model_design = LogisticRegression(maxiter=80, device="cpu")
        model_design.fit(X, y, fe_design=[csr])

        assert torch.allclose(
            model_ids.params["coef"],
            model_design.params["coef"],
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            model_ids.params["fe_coef"][0],
            model_design.params["fe_coef"][0],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_minibatch_fe_logit_close_to_full_batch(self):
        """Test mini-batch FE-logit estimates stay close to full-batch Adam."""
        torch.manual_seed(2)
        n_groups, group_size = 20, 30
        n_obs = n_groups * group_size

        X = torch.randn(n_obs, 2)
        group_ids = torch.repeat_interleave(torch.arange(n_groups), group_size)
        true_coef = torch.tensor([0.6, -0.25])
        group_effects = 0.45 * torch.randn(n_groups)

        eta = X @ true_coef + group_effects[group_ids]
        y = torch.bernoulli(torch.sigmoid(eta)).to(torch.float32)

        full_batch = LogisticRegression(
            optimizer=torch.optim.Adam,
            maxiter=800,
            tol=0.0,
            device="cpu",
        )
        full_batch.fit(X, y, fe=[group_ids])

        minibatch = LogisticRegression(
            optimizer=torch.optim.Adam,
            maxiter=800,
            tol=0.0,
            device="cpu",
        )
        minibatch.fit(X, y, fe=[group_ids], batch_size=96)

        assert torch.allclose(
            full_batch.params["coef"],
            minibatch.params["coef"],
            atol=0.08,
            rtol=0.0,
        )


class TestDeviceHandling:
    """Test GPU/CPU device handling"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test that models work on GPU"""
        torch.manual_seed(42)
        device = torch.device('cuda')

        n, p = 100, 3
        X = torch.randn(n, p, device=device)
        y = torch.randn(n, device=device)

        model = LinearRegression(device="cuda")
        model.fit(X, y)

        # Check that parameters are on correct device
        assert model.params["coef"].device.type == "cuda"

        # Test prediction
        y_pred = model.predict(X)
        assert y_pred.device.type == "cuda"
    
    def test_mixed_device_handling(self):
        """Test handling of CPU tensors with explicit device"""
        torch.manual_seed(42)

        n, p = 50, 2
        X_cpu = torch.randn(n, p)
        y_cpu = torch.randn(n)

        model = LinearRegression(device="cpu")
        model.fit(X_cpu, y_cpu)

        # Predictions should be on model's device
        y_pred_cpu = model.predict(X_cpu)
        assert y_pred_cpu.device.type == "cpu"
