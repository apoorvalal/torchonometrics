# Specification: Port `fastreg` FE Maximum Likelihood to `torchonometrics`

## Goal
Implement fixed-effects maximum likelihood estimation in Torch with feature parity for the core `fastreg` path:
- Panel GLM point estimates with real + categorical + high-dimensional FE terms
- Batched optimization of mean log-likelihood
- Fisher/OPG-based inference, including the one-HDFE fast block path
- Matrix-first API (scikit-style): user provides `X`, `y`, and FE structures directly; no formula parser work

## Fastreg Reference (What We Are Porting)
- `fastreg/fastreg/general.py:271` `maxlike_panel`
- `fastreg/fastreg/general.py:309` `glm_model`
- `fastreg/fastreg/general.py:337` `glm`
- `fastreg/fastreg/utils.py:219` `tree_fisher`
- `fastreg/fastreg/utils.py:229` `diag_fisher`
- `fastreg/fastreg/tools.py:154` `block_inverse`

## Current Torchonometrics Baseline
- MLE exists for dense `X @ beta` in `torchonometrics/mle.py` (`LogisticRegression`, `PoissonRegression`)
- Fixed effects machinery exists for OLS via alternating projections in `torchonometrics/linear.py` and `torchonometrics/demean.py`
- No FE-aware nonlinear MLE path yet

## Scope
### In Scope
- FE-aware Logistic and Poisson MLE
- Optional offset term
- Mini-batch and full-batch training
- Full OPG covariance for small/medium models
- `hdfe`-style fast inference path for one designated high-dimensional FE block
- Sparse FE inputs (`torch.sparse_csr`/COO or SciPy CSR)

### Out of Scope (Phase 1)
- Formula parser parity with `fastreg`
- Zero-inflated and negative binomial models
- Cluster-robust covariance in FE-MLE
- Incidental parameter bias corrections (beyond documentation warnings)

## Proposed API
Add FE support to existing MLE estimators:

```python
model = LogisticRegression(maxiter=..., optimizer=...)
model.fit(
    X,
    y,
    fe=[firm_id, year_id],      # optional list of ID vectors
    fe_design=[D_firm, D_year], # optional list of sparse FE design matrices (CSR/COO)
    hdfe_index=0,               # optional index into fe list
    offset=offset,              # optional
    batch_size=32768,           # optional
    stderr=True,                # True/False
)
```

Returned params structure:
- `params["coef"]`: real-valued regressors
- `params["fe_coef"]`: list/dict of FE coefficient tensors
- `params["se"]`, `params["vcov"]` for real coefficients
- if `hdfe_index` provided: add `params["fe_se_diag"]` for designated HDFE block

Input rules:
- `X`: dense array/tensor `(n_obs, n_features)`
- FE input must be exactly one of:
  - `fe`: list of 1D factor ID vectors `(n_obs,)`
  - `fe_design`: list of sparse incidence matrices `(n_obs, n_levels_j)` (one active level per row)
- if both are supplied, raise `ValueError`

## Model Parameterization
For each observation `i`:

`eta_i = offset_i + x_i' beta + sum_j alpha_j[fe_j[i]]`

Identification strategy (Phase 1):
- Per FE block, anchor one level to zero (drop/reference level)
- Keep explicit mapping metadata from original IDs to internal contiguous IDs
- For sparse FE design input, enforce one-hot row structure (or document strict fallback behavior)

## Implementation Plan
1. **Core FE-GLM Engine**
- Add internal helper(s) in `torchonometrics/mle.py` to:
  - canonicalize FE input:
    - ID vectors -> contiguous indices per factor
    - sparse FE matrices -> validated sparse structure for fast `sparse @ alpha`
  - pack/unpack `(beta, alpha_1, ..., alpha_J)`
  - compute predictor with:
    - gather/index-add for ID-vector path
    - sparse matvec for sparse design path
- Keep operations device-safe and batch-safe

2. **Training Loop Upgrade**
- Extend `MaximumLikelihoodEstimator.fit` to accept:
  - `fe=None`, `offset=None`, `batch_size=None`, `stderr=True`, `hdfe_index=None`
- Implement batched loss/grad loop for Adam/AdamW family
- Keep LBFGS supported in full-batch mode
- Preserve convergence tracking (`history["loss"]`)

3. **Likelihood Implementations**
- Reuse existing stable forms:
  - Logit: `logsigmoid`
  - Poisson: canonical log-link with `exp(eta)`
- Ensure both dense-only and FE paths share code where possible

4. **Inference (OPG/Fisher)**
- Compute per-observation score vectors for all active parameters
- Path A: full covariance via `(G'G)^{-1}`
- Path B (`hdfe_index`): block-Schur path mirroring `fastreg`:
  - full covariance for non-HDFE params
  - diagonal variances for designated HDFE levels
- Add small ridge regularization fallback for near-singular systems
- Ensure score accumulation works with sparse FE paths without densifying large FE blocks

5. **Model Outputs and Summary**
- Update summaries to display FE diagnostics compactly:
  - number of FE blocks, levels per block
  - whether `hdfe` fast inference was used
- Keep backward compatibility for dense-only models

6. **Tests**
Add/extend tests in `tests/test_integration.py` (and split if needed):
- FE logit coefficient recovery on synthetic panel
- FE poisson coefficient recovery on synthetic panel
- Dense-only regression unchanged behavior
- Sparse FE design input acceptance and equivalence (ID path vs CSR path)
- SE path smoke tests:
  - full OPG
  - `hdfe_index` diagonal FE SE output
- CPU/GPU parity test with FE path
- Mini-batch vs full-batch estimate closeness

7. **Docs**
- Update `README.md` FE-MLE section with one logit and one poisson FE example
- Add note on incidental parameter bias in nonlinear FE models

## Milestones
1. **M1 (Point Estimates)**
- FE logit/poisson fit works; tests for coefficient recovery pass

2. **M2 (Inference)**
- Full OPG and `hdfe` fast path implemented; inference tests pass

3. **M3 (Polish)**
- Docs, summary formatting, and API cleanup complete

## Acceptance Criteria
- FE-enabled `LogisticRegression.fit(..., fe=[...])` and `PoissonRegression.fit(..., fe=[...])` run on CPU and CUDA
- FE-enabled `LogisticRegression.fit(..., fe_design=[...])` and `PoissonRegression.fit(..., fe_design=[...])` run without densifying FE blocks
- Existing dense-only tests remain green
- New FE tests pass with stable tolerances
- `hdfe_index` path returns finite diagonal FE standard errors and finite real-parameter covariance
- README contains runnable FE-MLE examples

## Risks and Mitigations
- **Incidental parameter bias** in nonlinear FE: document clearly, avoid claiming bias correction
- **Memory pressure** from full OPG with many FE levels: default to `hdfe` path when requested; batch score accumulation
- **Identification bugs** with FE indexing: enforce reference-level normalization and add explicit checks/tests
