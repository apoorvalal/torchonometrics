# torchonometrics: GPU-accelerated econometrics in PyTorch

High-performance econometric estimation using PyTorch with first-class GPU support and automatic differentiation. Implements method of moments estimators (GMM, GEL), maximum likelihood models, and discrete choice models with modern deep learning workflows.

## Features

### Core Estimators
- **Linear Regression**: OLS with fixed effects via GPU-accelerated alternating projections
- **Generalized Method of Moments (GMM)**: Two-step efficient GMM with HAC-robust inference
- **Generalized Empirical Likelihood (GEL)**: Empirical likelihood, exponential tilting, and CUE estimators
- **Maximum Likelihood**: Logistic and Poisson regression with PyTorch optimizers
- **Discrete Choice Models**: Multinomial logit, probit, and low-rank logit for large choice sets

### Technical Capabilities
- **Automatic GPU Detection**: Seamless CPU/GPU operation with device management
- **Heteroskedasticity-Robust Inference**: HC0-HC3 standard errors for cross-sectional data
- **HAC-Robust Inference**: Newey-West covariance estimation for time series
- **Custom Optimizers**: Full access to PyTorch optimizer ecosystem (LBFGS, Adam, SGD)
- **Batched Operations**: Memory-efficient estimation for large datasets
- **M-Series Mac Support**: Native MPS backend support (no JAX Metal issues)

## Installation

```bash
git clone https://github.com/apoorvalal/torchonometrics
cd torchonometrics
uv venv
source .venv/bin/activate
uv sync
```

## Quick Start

### Linear Regression with Fixed Effects

```python
import torch
from torchonometrics import LinearRegression

# Panel data: firms × years
n_firms, n_years = 100, 10
n_obs = n_firms * n_years

X = torch.randn(n_obs, 3)
firm_ids = torch.repeat_interleave(torch.arange(n_firms), n_years)
year_ids = torch.tile(torch.arange(n_years), (n_firms,))

# True DGP with fixed effects
true_coef = torch.tensor([1.5, -0.8, 0.3])
firm_effects = torch.randn(n_firms)[firm_ids]
year_effects = torch.randn(n_years)[year_ids]
y = X @ true_coef + firm_effects + year_effects + 0.1 * torch.randn(n_obs)

# Two-way fixed effects regression
model = LinearRegression()
model.fit(X, y, fe=[firm_ids, year_ids], se="HC1")
print(f"Coefficients: {model.params['coef']}")
print(f"Robust SE: {model.params['se']}")
```

### Instrumental Variables via GMM

```python
from torchonometrics.gmm import GMMEstimator

# Define IV moment condition: E[Z'(Y - X'β)] = 0
def iv_moment(Z, Y, X, beta):
    return Z * (Y - X @ beta).unsqueeze(-1)

# Two-step efficient GMM
gmm = GMMEstimator(iv_moment, weighting_matrix="optimal", backend="torch")
gmm.fit(instruments, outcome, endogenous_vars, two_step=True)
print(gmm.summary())
```

### Maximum Likelihood Estimation

```python
from torchonometrics import LogisticRegression

# Binary response model
X = torch.randn(1000, 5)
true_coef = torch.tensor([0.5, 1.0, -0.8, 0.3, 0.2])
y = torch.bernoulli(torch.sigmoid(X @ true_coef))

# MLE with Fisher information-based standard errors
model = LogisticRegression(maxiter=100)
model.fit(X, y)
model.summary()  # Displays coefficients, SE, z-stats, p-values

# Predictions
probs = model.predict_proba(X)
classes = model.predict(X, threshold=0.5)
```

### Discrete Choice: Low-Rank Logit

```python
from torchonometrics.choice import LowRankLogit

# Large-scale choice data with varying assortments
n_users, n_items, rank = 1000, 100, 5
user_indices = torch.randint(0, n_users, (5000,))
chosen_items = torch.randint(0, n_items, (5000,))
assortments = torch.randint(0, 2, (5000, n_items)).float()  # Binary availability

# Factorized utility model: Θ = AB' with zero-sum normalization
model = LowRankLogit(rank=rank, n_users=n_users, n_items=n_items, lam=0.01)
model.fit(user_indices, chosen_items, assortments)

# Counterfactual analysis
baseline = torch.ones(100, n_items)
baseline[:, 50] = 0  # Product 50 unavailable
counterfactual = torch.ones(100, n_items)  # All products available
results = model.counterfactual(user_indices[:100], baseline, counterfactual)
print(f"Market share change: {results['market_share_change'][50]:.3f}")
```

## GPU Usage

All estimators automatically detect and use CUDA/MPS when available:

```python
# Automatic device detection
model = LinearRegression()  # Uses CUDA if available, else CPU
model.fit(X, y)

# Explicit device control
model_cpu = LinearRegression(device='cpu')
model_gpu = LogisticRegression(device='cuda')

# Move fitted models between devices
model.fit(X_cpu, y_cpu)
model.to('cuda')  # Transfer to GPU
predictions = model.predict(X_gpu)  # Input data auto-moved to model device
```

## Mathematical Framework

### GMM Estimation

The library implements Hansen's (1982) GMM framework. Given moment conditions $E[g(Z_i, \theta_0)] = 0$, the estimator minimizes:

$$\hat{\theta}_{GMM} = \arg\min_\theta \bar{g}_n(\theta)' W_n \bar{g}_n(\theta)$$

where $W_n$ is the weighting matrix. The efficient two-step procedure uses:
1. First step: $W_1 = I$ (identity matrix)
2. Second step: $W_2 = \hat{\Omega}^{-1}$ where $\hat{\Omega} = \frac{1}{n}\sum_i g_i g_i'$

Asymptotic distribution with optimal weighting:
$$\sqrt{n}(\hat{\theta} - \theta_0) \xrightarrow{d} N(0, (G'\Omega^{-1}G)^{-1})$$

### HAC-Robust Inference

For time series or spatial dependence, the library implements Newey-West (1987) HAC covariance:

$$\hat{\Omega}_{HAC} = \hat{\Gamma}_0 + \sum_{j=1}^L w_j(\hat{\Gamma}_j + \hat{\Gamma}_j')$$

with Bartlett kernel weights $w_j = 1 - j/(L+1)$ and automatic bandwidth selection $L = \lfloor 4(n/100)^{2/9}\rfloor$.

### Fixed Effects

Multi-way fixed effects are eliminated via alternating projections (Gaure, 2013). For two-way effects:

$$\ddot{y}_{it} = \ddot{x}_{it}'\beta + \ddot{\epsilon}_{it}$$

where $\ddot{z}_{it} = z_{it} - \bar{z}_{i\cdot} - \bar{z}_{\cdot t} + \bar{z}_{\cdot\cdot}$ is the within transformation.

### Discrete Choice

The low-rank logit model (Kallus & Udell, 2016) factorizes user-item utilities as $\Theta = AB'$ with rank $r \ll \min(n_{users}, n_{items})$, enabling scalable estimation for large choice sets with varying assortments.

See [mathematical notes](nb/math.pdf) for detailed exposition.

## Advanced Usage

### Custom Optimizers

```python
import torch.optim as optim

# Use Adam instead of default LBFGS
model = LogisticRegression(
    optimizer=optim.Adam,
    maxiter=1000
)
```

### Solver Backends

```python
# Linear regression solvers
model_torch = LinearRegression(solver="torch")  # PyTorch lstsq (GPU-capable)
model_numpy = LinearRegression(solver="numpy")  # NumPy fallback
```

### Memory-Efficient Batching

For datasets exceeding GPU memory, use DataLoader for batched optimization:

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Custom training loop with gradient accumulation
# See notebooks for complete examples
```

## Performance Characteristics

- **GPU Acceleration**: 10-100× speedup for large datasets (n > 10,000)
- **Fixed Effects**: Memory-efficient alternating projections scale to millions of observations
- **Batched Operations**: Vectorized computations throughout, compatible with `torch.compile`
- **Numerical Stability**: Eigenvalue regularization and pseudo-inverse for ill-conditioned problems

## Comparison with JAX Implementation

torchonometrics is a PyTorch port of jaxonometrics with enhanced device management:

| Feature | jaxonometrics | torchonometrics |
|---------|---------------|-----------------|
| Backend | JAX | PyTorch |
| M-Series Mac | Metal issues | Native MPS support |
| GPU Support | CUDA/TPU | CUDA/MPS/CPU |
| Auto-diff | `jax.grad` | `torch.autograd` |
| Compilation | `jax.jit` | `torch.compile` |
| Device Management | Manual | Automatic with `.to()` |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{torchonometrics,
  title = {torchonometrics: GPU-accelerated econometrics in PyTorch},
  author = {Lal, Apoorva},
  year = {2025},
  url = {https://github.com/apoorvalal/torchonometrics}
}
```

## References

- Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. *Econometrica*, 50(4), 1029-1054.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Gaure, S. (2013). OLS with multiple high dimensional category variables. *Computational Statistics & Data Analysis*, 66, 8-18.
- Kallus, N., & Udell, M. (2016). Dynamic assortment personalization in high dimensions. *arXiv preprint arXiv:1610.05604*.

## Related Projects

- [jaxonometrics](https://github.com/py-econometrics/jaxonometrics) - JAX-based econometrics (parent project)
- [pyfixest](https://github.com/py-econometrics/pyfixest) - Fast fixed effects estimation
- [linearmodels](https://github.com/bashtage/linearmodels) - Panel data models

## License

MIT License. See [LICENSE](LICENSE) for details.
