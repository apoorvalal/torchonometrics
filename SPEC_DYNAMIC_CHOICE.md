# Specification: Dynamic Discrete Choice Models with CCP Estimation

## Executive Summary

This document specifies the implementation of dynamic discrete choice models with Conditional Choice Probability (CCP) estimation methods for `torchonometrics`. The implementation will extend the existing static choice models to handle forward-looking agents making sequential decisions under uncertainty.

**Reference**: Hotz & Miller (1993), "Conditional Choice Probabilities and the Estimation of Dynamic Models", Review of Economic Studies

## 1. Current State Analysis

### 1.1 Implemented Models

The package currently implements static discrete choice models in `torchonometrics/choice/static.py`:

1. **BinaryLogit**: Binary choice with logistic link function
2. **BinaryProbit**: Binary choice with normal link function
3. **MultinomialLogit**: Multiple alternatives with softmax/IIA property
4. **LowRankLogit**: Matrix completion for choice with varying assortments

### 1.2 Architecture

All models inherit from `ChoiceModel` base class and implement:

- `_negative_log_likelihood()`: Likelihood computation
- `fit()`: Maximum likelihood estimation
- `predict_proba()`: Choice probability prediction
- `simulate()`: Generate simulated choices
- `counterfactual()`: Policy evaluation
- `_compute_fisher_information()`: Standard errors via Fisher information

### 1.3 Key Strengths

- Clean PyTorch-based implementation
- Modular design with base class architecture
- Support for counterfactual analysis
- Automated standard error computation

## 2. Gap Analysis

### 2.1 Missing Components

Dynamic discrete choice models are entirely absent. Key missing features:

1. **Dynamic Programming Framework**
   - No support for sequential decision problems
   - No state transition modeling
   - No value function iteration
   - No forward-looking behavior

2. **Estimation Methods** (from Hotz & Miller, 1993)
   - Full solution method (Rust 1987)
   - CCP inversion method (Hotz & Miller 1993)
   - Nested pseudo-likelihood (Aguirregabiria & Mira 2002)
   - EM algorithm for unobserved states (Arcidiacono & Miller 2011)

3. **Identification Issues**
   - No treatment of discount factor identification
   - No exclusion restriction methods

## 3. Dynamic Choice Model Specification

### 3.1 Mathematical Framework

An agent solves an infinite-horizon dynamic programming problem:

```
max E[Σ_{t=1}^∞ β^{t-1} u(x_t, a_t, ε_t; θ)]
```

where:

- `x_t ∈ X`: Observed state (finite discrete set)
- `ε_t ∈ ℝ^J`: Unobserved utility shocks (Type I extreme value)
- `a_t ∈ {1,...,J}`: Action/choice
- `β ∈ (0,1)`: Discount factor (fixed/known)
- `θ`: Structural parameters to estimate

**Key Assumptions (DDC1-DDC3)**:

1. **Additive Separability**: `u(s_t, a=j; θ) = ū_j(x_t; θ) + ε_{jt}`
2. **i.i.d. errors**: `ε_t ⊥ ε_{t+1}` (conditional independence over time)
3. **Conditional Independence**: `x_{t+1} ⊥ (ε_t, ε_{t+1})` given `(x_t, a_t)`

### 3.2 Bellman Equation

The value function satisfies:

```
v(x_t, ε_t; θ, φ) = max_{a∈J} {ū_a(x_t; θ) + ε_{at} +
                      β ∫ v(x_{t+1}, ε_{t+1}; θ, φ) dF(ε_{t+1}) dF(x_{t+1}|x_t, a; φ)}
```

Under Type I extreme value errors, the choice-specific value function is:

```
v̄_j(x_t; θ, φ) = ū_j(x_t; θ) + β ∫ ln[Σ_k exp(v̄_k(x_{t+1}; θ, φ))] dF(x_{t+1}|x_t, a=j; φ)
```

**Conditional Choice Probability**:

```
P(a_t = j | x_t; θ, φ) = exp(v̄_j(x_t; θ, φ)) / Σ_k exp(v̄_k(x_t; θ, φ))
```

## 4. Implementation Design

### 4.1 Module Structure

```
torchonometrics/choice/
├── __init__.py
├── base.py           # Existing ChoiceModel base class
├── static.py         # Existing static models
├── dynamic.py        # NEW: Dynamic choice models
└── transitions.py    # NEW: State transition estimation
```

### 4.2 State Space and Transition Estimation

**Key Insight from Rawat & Rust (2025)**: Reinforcement learning methods provide a framework for working with estimated transitions rather than requiring known model primitives. As they note: *"Reinforcement learning algorithms can be understood as a generalization of dynamic programming that relaxes the requirement of knowing the transition probabilities and reward functions."*

#### 4.2.1 Empirical Transition Estimation

```python
def estimate_transition_matrix(
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    n_states: int,
    n_choices: int,
    method: str = "frequency",
    bandwidth: float = None,
) -> torch.Tensor:
    """
    Estimate state transition probabilities P(x'|x,a) from panel data.

    Three approaches supported:
    1. "frequency": Nonparametric frequency estimator
    2. "kernel": Kernel smoothing for continuous states
    3. "parametric": Ordered probit/logit for ordinal transitions

    Args:
        states: (n_obs,) observed states x_t
        actions: (n_obs,) observed actions a_t
        next_states: (n_obs,) observed next states x_{t+1}
        n_states: Number of discrete states
        n_choices: Number of actions
        method: Estimation approach
        bandwidth: For kernel methods

    Returns:
        P: (n_states, n_choices, n_states) estimated transition matrix
           where P[x, a, x'] ≈ Pr(x_{t+1}=x' | x_t=x, a_t=a)

    Example:
        >>> # Estimate transitions from bus maintenance data
        >>> P_hat = estimate_transition_matrix(
        ...     states=data.states,
        ...     actions=data.actions,
        ...     next_states=data.next_states,
        ...     n_states=90,
        ...     n_choices=2,
        ...     method="frequency"
        ... )
        >>> # Verify probabilities sum to 1
        >>> assert torch.allclose(P_hat.sum(dim=2), torch.ones(90, 2))
    """
    if method == "frequency":
        # Nonparametric frequency estimator
        P = torch.zeros(n_states, n_choices, n_states)
        for x in range(n_states):
            for a in range(n_choices):
                mask = (states == x) & (actions == a)
                if mask.sum() > 0:
                    next_state_counts = torch.bincount(
                        next_states[mask],
                        minlength=n_states
                    )
                    P[x, a, :] = next_state_counts / next_state_counts.sum()
        return P
    elif method == "kernel":
        # Kernel smoothing for continuous/high-dimensional states
        raise NotImplementedError("Kernel estimation coming in Phase 3")
    elif method == "parametric":
        # Parametric specification (e.g., ordered probit for mileage)
        raise NotImplementedError("Parametric transitions coming in Phase 3")


def discretize_state(
    continuous_state: torch.Tensor,
    n_bins: int,
    method: str = "quantile",
    state_min: float = None,
    state_max: float = None,
) -> torch.Tensor:
    """
    Discretize continuous state variable into bins.

    Args:
        continuous_state: (n_obs,) continuous state values
        n_bins: Number of discrete bins
        method: "quantile" (equal counts) or "uniform" (equal width)
        state_min: Override minimum for uniform binning
        state_max: Override maximum for uniform binning

    Returns:
        discrete_state: (n_obs,) discretized state indices in {0,...,n_bins-1}

    Example:
        >>> # Discretize mileage into 90 bins
        >>> mileage_discrete = discretize_state(
        ...     continuous_state=mileage,
        ...     n_bins=90,
        ...     method="uniform",
        ...     state_min=0,
        ...     state_max=450000
        ... )
    """
```

#### 4.2.2 High-Dimensional State Spaces with Function Approximation

For state spaces where traditional tabular methods become intractable, we leverage **Deep RL** insights from Rawat & Rust (2025, Section 3.2-3.4):

*"Deep RL can indeed overcome the curse of dimensionality that limits traditional Dynamic Programming... DQN scales efficiently and maintains high solution quality"* (p. 16-17)

```python
class DeepValueFunction(nn.Module):
    """
    Neural network value function approximation for high-dimensional states.

    Enables dynamic choice estimation when |X| > 10,000 makes tabular
    methods intractable.
    """

    def __init__(
        self,
        state_dim: int,
        n_choices: int,
        hidden_dims: list = [128, 64],
    ):
        """
        Args:
            state_dim: Dimension of continuous/high-dim state space
            n_choices: Number of discrete actions
            hidden_dims: Architecture of hidden layers
        """
        super().__init__()
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_choices))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute value function for each choice at given states.

        Args:
            state: (batch, state_dim) continuous states

        Returns:
            values: (batch, n_choices) choice-specific values v̄(s,a)
        """
        return self.network(state)
```

### 4.2 Core Classes

#### 4.2.1 `DynamicChoiceModel` (Base Class)

```python
class DynamicChoiceModel(ChoiceModel):
    """
    Base class for dynamic discrete choice models.

    Implements common infrastructure for:
    - State space management
    - Transition probability specification
    - Value function iteration
    - Choice probability computation
    """

    def __init__(
        self,
        n_states: int,
        n_choices: int,
        discount_factor: float,
        transition_type: str = "parametric",  # or "nonparametric"
        optimizer: torch.optim.Optimizer = torch.optim.LBFGS,
        maxiter: int = 1000,
        tol: float = 1e-6,
    ):
        """
        Args:
            n_states: Number of discrete states in X
            n_choices: Number of discrete actions in J
            discount_factor: β ∈ (0,1), typically 0.95
            transition_type: How to model P(x'|x,a)
        """

    def set_transition_probabilities(
        self,
        transition_matrix: torch.Tensor,
        transition_params: dict = None,
    ):
        """
        Specify state transition mechanism P(x_{t+1} | x_t, a_t).

        Args:
            transition_matrix: (n_states, n_choices, n_states) tensor
                              P[x, a, x'] = Pr(x_{t+1}=x' | x_t=x, a_t=a)
            transition_params: Parameters φ if transitions are parametric
        """

    def set_flow_utility(
        self,
        utility_fn: Callable,
        utility_params: dict = None,
    ):
        """
        Specify per-period utility ū(x, a; θ).

        Args:
            utility_fn: Function computing u(x, a; θ)
            utility_params: Initial values for θ
        """
```

#### 4.2.2 `RustNFP` (Full Solution via Nested Fixed Point)

```python
class RustNFP(DynamicChoiceModel):
    """
    Rust (1987) full maximum likelihood via nested fixed point (NFP).

    At each parameter guess (θ, φ):
    1. Solve for value functions v̄_j(x; θ, φ) via contraction mapping
    2. Compute choice probabilities P(a|x; θ, φ)
    3. Evaluate likelihood L(θ, φ; data)
    4. Update parameters

    References:
        Rust, J. (1987). "Optimal replacement of GMC bus engines:
        An empirical model of Harold Zurcher." Econometrica, 55(5), 999-1033.
    """

    def solve_value_functions(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        tol: float = 1e-8,
        max_iter: int = 10000,
    ) -> torch.Tensor:
        """
        Solve value function fixed point via successive approximations.

        Iterates: v̄^{k+1} = T(v̄^k; θ, φ) until convergence, where T is
        the Bellman operator.

        Args:
            theta: Utility parameters
            phi: Transition parameters
            tol: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            v_bar: (n_states, n_choices) value functions
        """

    def _negative_log_likelihood(
        self,
        params: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        """
        Compute -log L(θ, φ | data) via nested fixed point.

        Args:
            params: Concatenated [theta; phi]
            data: Dict with keys 'states', 'actions', 'next_states'

        Returns:
            Negative log-likelihood
        """
        theta, phi = self._unpack_params(params)

        # Inner loop: solve for value functions
        v_bar = self.solve_value_functions(theta, phi)

        # Compute choice probabilities
        choice_probs = self._compute_choice_probs(v_bar, data['states'])

        # Likelihood of observed actions
        nll = -torch.sum(torch.log(
            choice_probs[range(len(data['actions'])), data['actions']]
        ))

        # Add transition likelihood if φ unknown
        if self.estimate_transitions:
            trans_ll = self._transition_likelihood(phi, data)
            nll -= trans_ll

        return nll
```

#### 4.2.3 `HotzMillerCCP` (CCP Inversion Method)

```python
class HotzMillerCCP(DynamicChoiceModel):
    """
    Hotz & Miller (1993) CCP inversion estimator.

    Two-stage procedure:
    1. Estimate conditional choice probabilities P̂(a|x) nonparametrically
    2. Invert CCPs to recover value function differences
    3. Estimate structural parameters via matching/MLE

    Avoids nested fixed point by using inverted CCPs directly.

    References:
        Hotz, V. J., & Miller, R. A. (1993). "Conditional choice probabilities
        and the estimation of dynamic models." Review of Economic Studies,
        60(3), 497-529.
    """

    def estimate_ccps(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        method: str = "frequency",  # or "kernel", "sieve"
    ) -> torch.Tensor:
        """
        First stage: Estimate P(a|x) from data nonparametrically.

        Args:
            states: Observed states (n_obs,)
            actions: Observed actions (n_obs,)
            method: Estimation method for CCPs

        Returns:
            ccp_hat: (n_states, n_choices) estimated CCPs
        """

    def invert_ccps(
        self,
        ccp_hat: torch.Tensor,
        phi_hat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invert CCPs to recover choice-specific value differences.

        Uses the formula:

            v̄_j(x) - v̄_J(x) = log(P(a=j|x) / P(a=J|x))
                                + β [E[w(x')|x,a=j] - E[w(x')|x,a=J]]

        where w(x) = log Σ_k exp(v̄_k(x)) and J is reference alternative.

        Args:
            ccp_hat: Estimated CCPs (n_states, n_choices)
            phi_hat: Estimated transition probabilities

        Returns:
            v_diff: (n_states, n_choices-1) value differences relative to alt J
        """

    def _negative_log_likelihood(
        self,
        theta: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        """
        Pseudo-likelihood using inverted CCPs.

        Matches model-implied value differences to data-inverted differences:

            min_θ Σ_x Σ_j [v̄_j(x; θ, φ̂) - v̄_J(x; θ, φ̂) - v̂_j(x) + v̂_J(x)]^2

        Or uses full likelihood with fixed value functions.
        """
```

#### 4.2.4 `AguirregabiraMiraNPL` (Nested Pseudo-Likelihood)

```python
class AguirregabiraNPL(DynamicChoiceModel):
    """
    Aguirregabiria & Mira (2002) nested pseudo-likelihood estimator.

    K-stage policy iteration:
    1. Start with initial CCP guess P^0(a|x)
    2. For k = 1, ..., K:
        a. Update parameters: θ^k = argmax L(θ | P^{k-1})
        b. Update policy: P^k(a|x; θ^k)
    3. Iterate until convergence

    Bridges Hotz-Miller and Rust: avoids inner loop but iterates CCPs.

    References:
        Aguirregabiria, V., & Mira, P. (2002). "Swapping the nested fixed
        point algorithm: A class of estimators for discrete Markov decision
        models." Econometrica, 70(4), 1519-1543.
    """

    def policy_iteration(
        self,
        ccp_init: torch.Tensor,
        max_stages: int = 10,
    ) -> tuple[torch.Tensor, list]:
        """
        K-stage policy iteration.

        Args:
            ccp_init: Initial CCP guess (n_states, n_choices)
            max_stages: Maximum K iterations

        Returns:
            theta_final: Converged parameter estimates
            history: List of (θ^k, P^k) for each iteration
        """
```

#### 4.2.5 `ArcidiaconoMillerEM` (Unobserved Heterogeneity)

```python
class ArcidiaconoMillerEM(DynamicChoiceModel):
    """
    Arcidiacono & Miller (2011) EM algorithm for unobserved state variables.

    Handles persistent unobserved heterogeneity h_t via EM:
    - E-step: Compute P(h_t | data, θ^{old})
    - M-step: Update θ^{new} = argmax E[log L(θ; data, h) | data, θ^{old}]

    Allows for:
    - Time-invariant types (finite mixture)
    - First-order Markov h_t transitions

    References:
        Arcidiacono, P., & Miller, R. A. (2011). "Conditional choice
        probability estimation of dynamic discrete choice models with
        unobserved heterogeneity." Econometrica, 79(6), 1823-1867.
    """

    def __init__(
        self,
        n_states: int,
        n_choices: int,
        n_types: int,  # Number of unobserved types
        discount_factor: float,
        type_transition: str = "fixed",  # "fixed" or "markov"
        **kwargs,
    ):
        """
        Args:
            n_types: Number of discrete unobserved types H = {1,...,H}
            type_transition: "fixed" for permanent types, "markov" for transitions
        """

    def e_step(
        self,
        theta_old: torch.Tensor,
        phi_old: torch.Tensor,
        data: dict,
    ) -> torch.Tensor:
        """
        Expectation step: compute posterior type probabilities.

        Returns:
            type_probs: (n_obs, n_types) P(h_t | x_t, a_t; θ^{old})
        """

    def m_step(
        self,
        type_probs: torch.Tensor,
        data: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maximization step: update parameters given type posteriors.

        Returns:
            theta_new: Updated utility parameters
            phi_new: Updated transition/type parameters
        """
```

### 4.3 Data Structure

#### 4.3.1 Panel Data Format

```python
@dataclass
class DynamicChoiceData:
    """
    Container for dynamic choice panel data.

    Attributes:
        states: (n_obs,) observed states x_t
        actions: (n_obs,) observed actions a_t
        next_states: (n_obs,) observed next states x_{t+1}
        individual_ids: (n_obs,) panel identifier
        time_periods: (n_obs,) time index
        covariates: Optional (n_obs, n_features) for x_t
    """
    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    individual_ids: torch.Tensor
    time_periods: torch.Tensor
    covariates: torch.Tensor = None

    def validate(self):
        """Check data integrity and stationarity."""

    def to_dict(self) -> dict:
        """Convert to dict for compatibility."""
```

### 4.4 Utility Specifications

#### 4.4.1 Linear Utility

```python
class LinearFlowUtility:
    """
    Linear per-period utility: ū_j(x; θ) = x'θ_j

    Common specification with state variables entering linearly.
    """

    def __init__(self, n_features: int, n_choices: int):
        self.theta = torch.randn(n_features, n_choices)

    def compute(
        self,
        states: torch.Tensor,  # (n_obs, n_features)
        choice: int,
    ) -> torch.Tensor:
        """Compute ū_j(x) for all observations."""
        return states @ self.theta[:, choice]
```

#### 4.4.2 Replacement Cost Utility

```python
class ReplacementUtility:
    """
    Rust (1987) bus engine replacement utility.

    ū_1(x; θ) = θ_1 x           # Maintain (x = mileage)
    ū_2(x; θ) = -θ_2 - θ_1 x    # Replace (pay RC + maintain)
    """

    def __init__(self):
        self.theta_maintenance = torch.tensor(0.001)  # θ_1
        self.theta_replacement_cost = torch.tensor(10.0)  # θ_2 (RC)

    def compute(
        self,
        state: torch.Tensor,  # Mileage
        action: int,  # 0=maintain, 1=replace
    ) -> torch.Tensor:
        """Rust replacement utility."""
        if action == 0:  # Maintain
            return -self.theta_maintenance * state
        else:  # Replace
            return -self.theta_replacement_cost - self.theta_maintenance * 0.0
```

## 5. Empirical Examples

### 5.1 Harold Zurcher Bus Engine Problem (Rust 1987)

**Setting**: Bus maintenance superintendent decides when to replace bus engines

**State Space**:
- `x_t` = cumulative mileage since last replacement (discretized, e.g., 90 bins)

**Action Space**:
- `a=0`: Continue operating (regular maintenance)
- `a=1`: Replace engine (pay replacement cost RC)

**Utility**:
```
u(x_t, a=0, ε_t) = -θ_1 x_t + ε_{0t}
u(x_t, a=1, ε_t) = -RC - θ_1 * 0 + ε_{1t}
```

**Transition**:
```
x_{t+1} = {
    x_t + Δx_t  if a_t = 0 (maintain)
    0           if a_t = 1 (replace)
}
where Δx_t ~ G(·; φ) is mileage increment
```

**Data**: Historical records of bus mileage and replacement decisions

**Identification**:
- Discount factor β fixed (e.g., 0.9999 monthly)
- RC and θ_1 identified from replacement timing conditional on mileage
- Transition G(·) identified from mileage evolution

**Implementation**:
```python
# Example usage
from torchonometrics.choice import RustNFP, ReplacementUtility

# Load Rust's bus data
data = load_zurcher_data()  # Returns DynamicChoiceData

# Define model
model = RustNFP(
    n_states=90,  # Mileage bins
    n_choices=2,  # Maintain vs Replace
    discount_factor=0.9999,
    transition_type="parametric",
)

# Set utility function
model.set_flow_utility(ReplacementUtility())

# Estimate
model.fit(data, method="NFP")

# Results
print(f"Estimated RC: {model.params['replacement_cost']:.2f}")
print(f"Maintenance cost: {model.params['theta_1']:.4f}")

# Counterfactual: What if RC decreased by 20%?
results = model.counterfactual(
    policy_change={"replacement_cost": model.params['replacement_cost'] * 0.8}
)
print(f"New avg replacement mileage: {results['avg_replacement_mileage']:.0f}")
```

### 5.2 Female Labor Force Participation (Keane & Wolpin 1997)

**Setting**: Women choose among work, home, and school over life cycle

**State Space**:
- Experience: `x1_t` (years of work experience)
- Education: `x2_t` (years of schooling)
- Age: `x3_t`
- Children: `x4_t` (number of young children)

**Action Space**:
- `a=1`: Work full-time
- `a=2`: Stay at home
- `a=3`: Attend school

**Utility**:
```
u(x_t, a=1) = wage(x_t) - work_cost(x4_t) + ε_1t
u(x_t, a=2) = home_value(x4_t) + ε_2t
u(x_t, a=3) = -tuition(x2_t) + ε_3t
```

**Transitions**:
```
x1_{t+1} = x1_t + 1{a_t=1}  # Experience accumulates if working
x2_{t+1} = x2_t + 1{a_t=3}  # Education increases if schooling
x3_{t+1} = x3_t + 1         # Age always increases
x4_{t+1} ~ fertility_process(x_t)
```

**Implementation**:
```python
from torchonometrics.choice import HotzMillerCCP

model = HotzMillerCCP(
    n_states=1000,  # Discretized (experience, education, age, children)
    n_choices=3,
    discount_factor=0.95,
)

# First stage: estimate CCPs
model.estimate_ccps(data.states, data.actions, method="kernel")

# Second stage: invert and estimate θ
model.fit(data, method="two_stage")

# Counterfactual: childcare subsidy
results = model.counterfactual(
    policy_change={"childcare_subsidy": 5000}
)
```

### 5.3 Firm Entry/Exit Decisions

**Setting**: Firms decide to enter, stay, or exit a market

**State Space**:
- Market size: `x1_t`
- Number of competitors: `x2_t`
- Own productivity: `x3_t` (unobserved heterogeneity)

**Action Space**:
- `a=0`: Not in market (inactive)
- `a=1`: Enter market (pay entry cost)
- `a=2`: Continue operating
- `a=3`: Exit market (get scrap value)

**Utility**:
```
u(x_t, a=0) = 0
u(x_t, a=1) = -EC + π(x_t) + ε_1t  # Entry cost
u(x_t, a=2) = π(x_t) + ε_2t         # Operating profit
u(x_t, a=3) = SV + ε_3t             # Scrap value
```

**Unobserved Heterogeneity**:
- Firm productivity `x3_t` unobserved to econometrician
- Requires Arcidiacono-Miller EM approach

**Implementation**:
```python
from torchonometrics.choice import ArcidiaconoMillerEM

model = ArcidiaconoMillerEM(
    n_states=50,  # Market conditions
    n_choices=4,
    n_types=3,    # Low/Medium/High productivity types
    discount_factor=0.95,
    type_transition="markov",
)

model.fit(data, max_em_iter=100)
```

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goals**: Core infrastructure and basic dynamic model

**Deliverables**:
1. `DynamicChoiceModel` base class
2. Data structures (`DynamicChoiceData`)
3. Basic utility specifications (linear, Rust replacement)
4. Value function iteration utilities
5. Unit tests for Bellman operator

**Key Files**:
- `torchonometrics/choice/dynamic.py`
- `torchonometrics/choice/utils.py` (new)
- `tests/test_dynamic_choice.py`

### Phase 2: Full Solution Method (Weeks 3-4)

**Goals**: Rust NFP estimator operational

**Deliverables**:
1. `RustNFP` class with nested fixed point
2. Contraction mapping solver
3. Transition probability estimation
4. Likelihood computation
5. Zurcher bus engine example + notebook
6. Integration tests with simulated data

**Key Files**:
- `torchonometrics/choice/dynamic.py` (RustNFP)
- `tests/test_rust_nfp.py`
- `examples/zurcher_bus_engine.py`
- `nb/rust_replication.ipynb`

**Validation**:
- Replicate Rust (1987) Table V estimates
- Compare to Rust's Fortran code results

### Phase 3: CCP Inversion (Weeks 5-6)

**Goals**: Hotz-Miller estimator operational

**Deliverables**:
1. `HotzMillerCCP` class
2. Nonparametric CCP estimation (frequency, kernel)
3. CCP inversion formula implementation
4. Two-stage estimation procedure
5. Comparison with NFP on simulated data
6. Computational speedup benchmarks

**Key Files**:
- `torchonometrics/choice/dynamic.py` (HotzMillerCCP)
- `torchonometrics/choice/ccp_estimators.py` (new)
- `tests/test_hotz_miller.py`
- `benchmarks/ccp_vs_nfp.py`

**Validation**:
- Monte Carlo: compare finite-sample properties vs NFP
- Speed comparison on varying state space sizes

### Phase 4: NPL and Extensions (Weeks 7-8)

**Goals**: Advanced estimation methods

**Deliverables**:
1. `AguirregabiraMiraNPL` class
2. Policy iteration algorithm
3. Convergence diagnostics
4. `ArcidiaconoMillerEM` class (basic)
5. EM algorithm for finite mixture
6. Documentation and examples

**Key Files**:
- `torchonometrics/choice/dynamic.py` (NPL, EM)
- `tests/test_npl.py`
- `tests/test_em_unobserved.py`
- `examples/labor_supply.py`

### Phase 5: Documentation and Paper Replications (Weeks 9-10)

**Goals**: Publication-ready package

**Deliverables**:
1. Complete API documentation
2. Tutorial notebooks:
   - Intro to dynamic choice
   - Rust (1987) replication
   - Keane-Wolpin (1997) style example
3. Performance benchmarks
4. Comparison table: NFP vs CCP vs NPL
5. Identification discussion and tools
6. Release prep (versioning, CI/CD)

**Key Files**:
- `docs/dynamic_choice.md`
- `nb/tutorial_dynamic_choice.ipynb`
- `nb/rust_replication.ipynb`
- `nb/female_labor_supply.ipynb`
- `README.md` (update with new features)

## 7. Testing Strategy

### 7.1 Unit Tests

For each class, test:
1. **Initialization**: Proper setup of state/action spaces
2. **Value iteration**: Convergence to fixed point
3. **Likelihood**: Correct computation at known parameters
4. **Gradient**: Autograd correctness
5. **Edge cases**: Single state, two choices, etc.

### 7.2 Integration Tests

1. **Simulated DGP**: Generate data from known model, recover parameters
2. **NFP vs CCP**: Should yield similar estimates on same data
3. **Counterfactuals**: Consistent policy evaluation across methods

### 7.3 Replication Tests

1. **Rust (1987)**: Match published Table V estimates
2. **Computational**: Compare speed with existing implementations (e.g., Julia's DDC.jl)

## 8. Performance Considerations

### 8.1 Computational Bottlenecks

1. **Value iteration**: O(|X|² |J|) per iteration
2. **Nested optimization**: NFP repeats value iteration at each θ guess
3. **Likelihood evaluation**: O(n_obs |X| |J|)

### 8.2 Optimization Strategies

1. **Vectorization**: Batch operations over state space using PyTorch
2. **GPU acceleration**: Move value iteration to GPU for large state spaces
3. **Sparse transitions**: Exploit sparsity in P(x'|x,a) when possible
4. **Parallel CCP**: Estimate CCPs independently per state
5. **Warm starts**: Initialize optimizer with previous iteration results

### 8.3 Scalability Targets

- **Small problems**: |X| ≤ 100, real-time estimation
- **Medium problems**: |X| ≤ 1000, estimation in minutes
- **Large problems**: |X| > 1000, feasible with GPU + CCP methods

## 9. Documentation Requirements

### 9.1 Docstring Standard

All classes and methods must have:
- One-line summary
- Extended description with model details
- Mathematical formulation in LaTeX (using ```math```)
- Args with types and shapes
- Returns with types and shapes
- Example usage code
- References to papers

Example:
```python
def solve_value_functions(
    self,
    theta: torch.Tensor,
    phi: torch.Tensor,
    tol: float = 1e-8,
) -> torch.Tensor:
    """
    Solve value function fixed point via contraction mapping.

    Iterates the Bellman operator until convergence:

    ```math
    v̄_j^{k+1}(x) = ū_j(x; θ) + β Σ_{x'} P(x'|x,a=j; φ)
                    log[Σ_k exp(v̄_k^k(x'))]
    ```

    Args:
        theta: Utility parameters, shape (n_features, n_choices)
        phi: Transition parameters, shape (n_trans_params,)
        tol: Convergence tolerance for sup-norm

    Returns:
        Value functions v̄(x,a), shape (n_states, n_choices)

    Example:
        >>> model = RustNFP(n_states=90, n_choices=2, discount_factor=0.95)
        >>> v_bar = model.solve_value_functions(theta, phi)
        >>> print(v_bar.shape)
        torch.Size([90, 2])

    References:
        Rust (1987), Equation (3.4)
    """
```

### 9.2 Tutorial Notebooks

1. **Introduction to Dynamic Choice Models**
   - Motivating example
   - Bellman equation intuition
   - Simple 2-state, 2-action problem
   - Comparison with static logit

2. **Replicating Rust (1987)**
   - Load Zurcher bus data
   - Specify replacement utility
   - Estimate via NFP
   - Interpret results
   - Counterfactual: lower replacement costs

3. **CCP Methods Explained**
   - Why avoid nested fixed point?
   - Two-stage procedure walkthrough
   - Comparison with full solution
   - When to use CCP vs NFP

4. **Unobserved Heterogeneity with EM**
   - Motivation: persistent unobservables
   - EM algorithm intuition
   - Application to entry/exit with firm types
   - Type probability posterior analysis

## 10. Open Questions and Future Extensions

### 10.1 Open Questions

1. **Discount Factor Identification**
   - Implement Magnac-Thesmar (2002) exclusion restrictions?
   - Add Abbring-Daljord (2020) invertibility methods?

2. **Continuous States**
   - Discretization schemes (equal-width vs quantiles)
   - Approximation methods (value function interpolation)

3. **Finite Horizon**
   - Backward induction for T-period problems
   - Initial condition issues

4. **Multiple Agents**
   - Extension to dynamic games (Aguirregabiria & Mira 2007)
   - Equilibrium computation

### 10.2 Future Extensions

1. **MPEC (Su & Judd 2012)**
   - Constrained optimization reformulation
   - Comparison with NFP

2. **Forward Simulation (Hotz et al 1994)**
   - Monte Carlo integration for complex transitions
   - Reduce curse of dimensionality

3. **Bayesian Estimation (Imai, Jain, Ching 2009)**
   - MCMC for posterior inference
   - Incorporate priors on structural parameters

4. **Machine Learning Integration**
   - Neural network value function approximation
   - Deep reinforcement learning connections

5. **Non-standard Preferences**
   - Hyperbolic discounting (β-δ models)
   - Risk aversion (non-expected utility)
   - Ambiguity aversion

## 11. References

### Core Papers

1. **Rust, J. (1987)**. "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher." *Econometrica*, 55(5), 999-1033.

2. **Hotz, V. J., & Miller, R. A. (1993)**. "Conditional choice probabilities and the estimation of dynamic models." *Review of Economic Studies*, 60(3), 497-529.

3. **Aguirregabiria, V., & Mira, P. (2002)**. "Swapping the nested fixed point algorithm: A class of estimators for discrete Markov decision models." *Econometrica*, 70(4), 1519-1543.

4. **Arcidiacono, P., & Miller, R. A. (2011)**. "Conditional choice probability estimation of dynamic discrete choice models with unobserved heterogeneity." *Econometrica*, 79(6), 1823-1867.

### Identification

5. **Magnac, T., & Thesmar, D. (2002)**. "Identifying dynamic discrete decision processes." *Econometrica*, 70(2), 801-816.

6. **Abbring, J. H., & Daljord, Ø. (2020)**. "Identifying the discount factor in dynamic discrete choice models." *Quantitative Economics*, 11(2), 471-501.

### Computational Methods

7. **Su, C. L., & Judd, K. L. (2012)**. "Constrained optimization approaches to estimation of structural models." *Econometrica*, 80(5), 2213-2230.

8. **Iskhakov, F., Lee, J., Rust, J., Schjerning, B., & Seo, K. (2016)**. "Comment on 'Constrained optimization approaches to estimation of structural models'." *Econometrica*, 84(1), 365-370.

### Surveys

9. **Aguirregabiria, V., & Mira, P. (2010)**. "Dynamic discrete choice structural models: A survey." *Journal of Econometrics*, 156(1), 38-67.

10. **Rust, J. (1994)**. "Structural estimation of Markov decision processes." *Handbook of Econometrics*, 4, 3081-3143.

### Applications

11. **Keane, M. P., & Wolpin, K. I. (1997)**. "The career decisions of young men." *Journal of Political Economy*, 105(3), 473-522.

12. **Pakes, A. (1986)**. "Patents as options: Some estimates of the value of holding European patent stocks." *Econometrica*, 54(4), 755-784.

13. **Rawat, P., & Rust, J. (2025)**. "Structural Econometrics and Reinforcement Learning." *Handbook of Reinforcement Learning and Control* (forthcoming).

---

## 12. Implementation Log

### Phase 1: Foundation (Completed 2025-10-27)

**Status**: ✅ Complete
**Commits**: Initial implementation of dynamic choice infrastructure

**Deliverables Completed**:

1. **`torchonometrics/choice/transitions.py`** (206 lines)
   - ✅ `estimate_transition_matrix()`: Nonparametric frequency estimator for P(x'|x,a)
   - ✅ `discretize_state()`: Quantile and uniform discretization methods
   - ✅ `DeepValueFunction`: Neural network for high-dimensional state spaces (DQN-style)
   - Implements insights from Rawat & Rust (2025) on RL connections

2. **`torchonometrics/choice/dynamic.py`** (401 lines)
   - ✅ `DynamicChoiceData`: Panel data container with validation
   - ✅ `DynamicChoiceModel`: Abstract base class with:
     - Bellman operator implementation
     - Contraction mapping value function solver
     - Choice probability computation (Type I EV)
     - Transition probability management
   - ✅ `LinearFlowUtility`: Linear per-period utility ū(x,a) = x'θ_a
   - ✅ `ReplacementUtility`: Rust (1987) bus engine specification

3. **`tests/test_dynamic_choice.py`** (530 lines)
   - ✅ 18 unit tests covering:
     - Data structure validation
     - Bellman operator correctness
     - Value iteration convergence
     - Transition matrix estimation
     - State discretization (uniform/quantile)
     - Utility specifications
     - Choice probability computation
     - Full integration pipeline
   - All tests passing (18/18)

4. **Module Structure**
   - ✅ Updated `torchonometrics/choice/__init__.py` to export dynamic models
   - ✅ Removed outdated `spec.md` (superseded by this document)
   - ✅ Backward compatibility: all existing static model tests pass (7/7)

**Key Technical Achievements**:
- Contraction mapping solver with convergence guarantees (sup-norm tolerance)
- Handles unobserved (state, action) pairs with uniform prior
- Full type hints and comprehensive docstrings with examples
- Integration of deep RL methods for curse of dimensionality

**Test Coverage**: 25/25 tests passing (18 dynamic + 7 static)

**Next Steps**: Phase 2 (Weeks 3-4) - Implement RustNFP full solution estimator

---

**Document Version**: 1.1
**Last Updated**: 2025-10-27
**Authors**: Implementation by Claude Code based on specification from Hotz & Miller (1993), Rust (1987), and Rawat & Rust (2025)
