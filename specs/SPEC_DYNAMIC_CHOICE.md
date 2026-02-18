# Specification: Dynamic Discrete Choice Module (Current Status)

## Last Updated
2026-02-18

## Objective
Track the actual implementation state of dynamic discrete choice in `torchonometrics`, identify rough edges, and define the next set of engineering improvements.

## Reference Material (`ref/`)
- `ref/hj_choice.pdf`: static + dynamic discrete choice foundations, likelihood-based estimation framing.
- `ref/Rawat_and_Rust_2025_-_Structural_Econometrics_and_Reinforcement_Learning.pdf`: RL/DP framing, transition-learning motivation, high-dimensional state considerations.
- `ref/hj_search.pdf`: search-friction framing relevant for future dynamic/search model extensions.

## Current Implementation Snapshot

### Implemented Models and Components
- `torchonometrics/choice/dynamic.py`
  - `DynamicChoiceData`: panel data container with validation.
  - `DynamicChoiceModel`: base class with Bellman operator, value iteration, choice-probability mapping, transition/utility hooks.
  - `RustNFP`: nested fixed-point estimator for full-solution dynamic choice likelihood.
  - `HotzMillerCCP`: CCP-based estimator with inversion precomputation.
  - `LinearFlowUtility`, `ReplacementUtility`: flow-utility parameterizations.
- `torchonometrics/choice/transitions.py`
  - `estimate_transition_matrix` (frequency estimator implemented).
  - `discretize_state` (uniform + quantile binning).
  - `DeepValueFunction` (neural value-function approximator scaffold).
- `torchonometrics/choice/ccp_estimators.py`
  - `estimate_ccps` (frequency estimator implemented).
- Public API exports are wired in `torchonometrics/choice/__init__.py`.

### Testing Coverage (Dynamic Area)
- `tests/test_dynamic_choice.py`: data validation, Bellman/value-iteration behavior, transitions, discretization, utilities, probability mapping.
- `tests/test_rust_nfp.py`: NFP likelihood, predict/simulate/counterfactual smoke tests.
- `tests/test_hotz_miller.py`: Hotz-Miller recovery-style integration test.

## Alignment with Reference Theory
- Rust-style full-solution MLE path exists (`RustNFP`) and uses explicit fixed-point solution of value functions.
- Hotz-Miller CCP inversion path exists (`HotzMillerCCP`) with nonparametric first-stage CCP estimation.
- Transition estimation from data is supported in nonparametric frequency form, consistent with the practical estimation framing discussed in the reference materials.

## Known Rough Edges and Gaps

### Estimation/Modeling Gaps
- Transition estimation methods are incomplete: `kernel` and `parametric` options raise `NotImplementedError`.
- CCP estimation currently only supports frequency; no smoothing/regularized alternatives.
- `RustNFP(estimate_transitions=True)` is not implemented beyond a guard.
- `DynamicChoiceModel._unpack_params()` has an unimplemented generic fallback path for arbitrary utility modules.
- `RustNFP.counterfactual()` only has concrete behavior for `ReplacementUtility`; non-replacement utilities currently fall through to empty output.
- Fisher-information/SE computation is not implemented for `RustNFP` and `HotzMillerCCP`.

### Engineering/UX Gaps
- In `DynamicChoiceModel.fit()`, if utility has no trainable parameters, method returns early and does not set a fitted parameter object (`self.params`) for downstream predict/simulate workflows.
- Dynamic docs currently rely mostly on API docstrings and tests; no dedicated end-to-end notebook or canonical worked example.

## Proposed Next Steps (General Improvements Branch)

### Phase 1: Reliability and API Consistency
1. Finish generic parameter packing/unpacking for arbitrary `nn.Module` utilities.
2. Fix no-parameter fit behavior by setting a consistent fitted state.
3. Complete `RustNFP.counterfactual()` behavior for `LinearFlowUtility` and generic utilities.
4. Add explicit, structured error messages for unsupported dynamic paths.

### Phase 2: Inference and Estimation Depth
1. Implement score-based covariance/Fisher approximations for dynamic estimators.
2. Add at least one smoothed CCP estimator variant for sparse state support.
3. Add a parametric transition model option (minimum viable ordered-state case).

### Phase 3: Documentation and Reproducibility
1. Add one reproducible dynamic example (Rust replacement toy data).
2. Add one CCP inversion walkthrough showing first-stage and second-stage objects.
3. Ensure pdoc landing pages cross-link dynamic components and estimators for developer navigation.

## Acceptance Criteria for “Dynamic Module Stable v1”
- `RustNFP` and `HotzMillerCCP` have consistent fit/predict/simulate behavior across utility types.
- Transition + CCP estimation pathways have at least one robust option beyond raw frequency counting.
- Dynamic estimators expose standard-error or covariance output with documented caveats.
- Documentation includes at least one end-to-end dynamic workflow and explicit model limitations.

## Out of Scope (for this branch)
- Dynamic games with strategic interaction/equilibrium multiplicity.
- Full inverse-RL estimator stack.
- Large-scale deep RL training loops beyond value-function scaffolding.
