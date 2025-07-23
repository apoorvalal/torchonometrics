# `torchonometrics`: GPU-accelerated econometrics in PyTorch

`torchonometrics` provides high-performance implementations of standard econometrics routines using PyTorch, with first-class support for GPU acceleration and modern deep learning workflows.

## Features

- **Linear Regression** with multiple solver backends (PyTorch, NumPy)
- **Fixed Effects Regression** with GPU-accelerated alternating projections
- **Maximum Likelihood Estimation** (Logistic, Poisson) with PyTorch optimizers
- **GPU Support** - seamless CPU/GPU operation with proper device handling
- **M-Series Mac Friendly** - no more JAX Metal backend issues
- **Modern PyTorch Integration** - works naturally with existing PyTorch workflows

## Installation

```bash
git clone https://github.com/apoorvalal/torchonometrics
cd torchonometrics
uv venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### Linear Regression

```python
import torch
from torchonometrics import LinearRegression

# Generate synthetic data
n, p = 1000, 5
X = torch.randn(n, p)
true_coef = torch.randn(p)
y = X @ true_coef + 0.1 * torch.randn(n)

# Fit model
model = LinearRegression()
model.fit(X, y, se="HC1")  # Robust standard errors
print(f"Coefficients: {model.params['coef']}")
print(f"Standard Errors: {model.params['se']}")

# Predict
y_pred = model.predict(X)
```

### Fixed Effects Regression

```python
import torch
from torchonometrics import LinearRegression

# Panel data setup
n_firms, n_years = 100, 10
n_obs = n_firms * n_years

# Generate data with firm and year effects
X = torch.randn(n_obs, 3)
firm_ids = torch.repeat_interleave(torch.arange(n_firms), n_years)
year_ids = torch.tile(torch.arange(n_years), (n_firms,))

# Add intercept
X_with_intercept = torch.cat([torch.ones(n_obs, 1), X], dim=1)

# True coefficients and effects
true_coef = torch.tensor([2.0, 1.5, -0.8, 0.3])
firm_effects = torch.randn(n_firms)[firm_ids]
year_effects = torch.randn(n_years)[year_ids]

y = X_with_intercept @ true_coef + firm_effects + year_effects + 0.1 * torch.randn(n_obs)

# Fit with two-way fixed effects
model = LinearRegression()
model.fit(X_with_intercept, y, fe=[firm_ids, year_ids])
print(f"Coefficients: {model.params['coef']}")
```

### Maximum Likelihood Estimation

```python
import torch
from torchonometrics import LogisticRegression

# Binary classification data
n, p = 500, 4
X = torch.randn(n, p)
X_with_intercept = torch.cat([torch.ones(n, 1), X], dim=1)

# Generate binary outcomes
true_coef = torch.tensor([0.5, 1.0, -0.8, 0.3, 0.2])
logits = X_with_intercept @ true_coef
probs = torch.sigmoid(logits)
y = torch.bernoulli(probs)

# Fit logistic regression
model = LogisticRegression(maxiter=100)
model.fit(X_with_intercept, y)

# Predictions
y_pred_proba = model.predict_proba(X_with_intercept)
y_pred = model.predict(X_with_intercept)
```

## 🎛️ GPU Usage

All models automatically detect and use GPU when available:

```python
# Move data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = X.to(device)
y = y.to(device)

# Models automatically run on GPU
model = LinearRegression()
model.fit(X, y)  # Computation happens on GPU
```

## 🔧 Advanced Features

### Custom Optimizers for MLE

```python
from torchonometrics import LogisticRegression
import torch.optim as optim

# Use custom optimizer
model = LogisticRegression(
    optimizer=optim.Adam,  # Instead of default LBFGS
    maxiter=1000
)
model.fit(X, y)
```

### Solver Options

```python
from torchonometrics import LinearRegression

# Different solver backends
model_torch = LinearRegression(solver="torch")    # PyTorch lstsq (default)
model_numpy = LinearRegression(solver="numpy")    # NumPy lstsq fallback
```

## 📊 Performance

`torchonometrics` is designed for performance:

- **GPU Acceleration**: Automatic GPU usage for large datasets
- **Compiled Operations**: Key operations can use `torch.compile` (PyTorch 2.0+)
- **Memory Efficient**: Optimized memory usage for large fixed effects problems
- **Batched Operations**: Vectorized computations throughout

## 🧪 Comparison with JAX Implementation

`torchonometrics` is a PyTorch port of `jaxonometrics`, designed to address M-series Mac compatibility issues while providing similar performance and APIs. Key differences:

| Feature | jaxonometrics | torchonometrics |
|---------|---------------|-----------------|
| Backend | JAX | PyTorch |
| M-Series Mac | ❌ Metal issues | ✅ Native support |
| GPU Support | ✅ CUDA/TPU | ✅ CUDA/MPS |
| API | JAX-style | PyTorch-style |
| Compilation | `jax.jit` | `torch.compile` |
| Ecosystem | JAX/Flax | PyTorch/Lightning |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Citation

If you use `torchonometrics` in your research, please cite:

```bibtex
@software{torchonometrics,
  title = {torchonometrics: GPU-accelerated econometrics in PyTorch},
  author = {Lal, Apoorva},
  year = {2024},
  url = {https://github.com/py-econometrics/torchonometrics}
}
```

## 🔗 Related Projects

- [`jaxonometrics`](https://github.com/py-econometrics/jaxonometrics) - JAX-based econometrics (parent project)
- [`pyfixest`](https://github.com/py-econometrics/pyfixest) - Fast fixed effects estimation in Python
- [`linearmodels`](https://github.com/bashtage/linearmodels) - Additional econometric models
