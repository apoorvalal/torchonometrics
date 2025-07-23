"""
Integration tests for torchonometrics functionality
"""
import torch
import numpy as np
import pytest
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
        
        model = LinearRegression(solver="torch")
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
        
        model = LinearRegression()
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
        
        model = LinearRegression()
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
        
        model = LinearRegression()
        model.fit(X_with_intercept, y, fe=[group_ids])
        
        # Check that coefficients are recovered (except intercept which is absorbed)
        fitted_coef = model.params["coef"]
        assert fitted_coef.shape == true_coef.shape
        
        # Non-intercept coefficients should be close
        coef_mse = torch.mean((true_coef[1:] - fitted_coef[1:])**2)
        assert coef_mse < 0.01, f"High coefficient MSE: {coef_mse}"
    
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
        
        model = LinearRegression()
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
        
        model = LogisticRegression(maxiter=50)
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
        
        model = PoissonRegression(maxiter=50)
        model.fit(X_with_intercept, y)
        
        assert "coef" in model.params
        assert model.params["coef"].shape == true_coef.shape
        
        # Test prediction
        y_pred = model.predict(X_with_intercept)
        assert torch.all(y_pred >= 0)  # Predictions should be non-negative


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
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Check that parameters are on correct device
        assert model.params["coef"].device == device
        
        # Test prediction
        y_pred = model.predict(X)
        assert y_pred.device == device
    
    def test_mixed_device_handling(self):
        """Test handling of mixed CPU/GPU tensors"""
        torch.manual_seed(42)
        
        n, p = 50, 2
        X_cpu = torch.randn(n, p)
        y_cpu = torch.randn(n)
        
        model = LinearRegression()
        model.fit(X_cpu, y_cpu)
        
        # Predictions should work regardless of input device
        y_pred_cpu = model.predict(X_cpu)
        assert y_pred_cpu.device == X_cpu.device