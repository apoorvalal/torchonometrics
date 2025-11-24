# Structural Estimation of Dynamic Discrete Choice Models

## Theory, Methods, and Connections to Inverse Reinforcement Learning

**Date:** November 21, 2025

References: Hortacsu and Joo (2023), Rust (1987), Hotz & Miller (1993), Arcidiacono & Miller (2011), Ziebart et al. (2008), Rawat & Rust (2025)


## 1. The Structural Framework

Dynamic Discrete Choice (DDC) models provide a rigorous framework for analyzing agents who make sequential decisions under uncertainty, balancing current utility against future consequences. Unlike reduced-form approaches, DDC estimates the "deep structural parameters" (preferences and technology), allowing for counterfactual policy analysis.

### 1.1 The Primitives

We consider an infinite-horizon discrete time environment ($t = 0, 1, ... , \infty$) with a forward-looking agent.

*   **State Space ($S$):** At time $t$, the agent observes a state vector $s_t \in S$.
*   **Action Space ($A$):** The agent chooses an action $a_t \in A = \{0, 1, \dots, J\}$.
*   **Unobservables ($\varepsilon_t$):** A vector of utility shocks $\varepsilon_t \in \mathbb{R}^{J+1}$, observed by the agent but not the econometrician.
*   **Transition:** The state evolves according to a Markov process $P(s_{t+1} | s_t, a_t)$.
*   **Discount Factor:** $\beta \in (0, 1)$.

### 1.2 The Bellman Equation

The agent maximizes the expected discounted sum of per-period utilities. Under the standard assumption of **Additive Separability (AS)**, the utility function is:

$$ U(s_t, a_t, \varepsilon_t) = u(s_t, a_t; \theta) + \varepsilon_{t}(a_t) $$

Where $u(\cdot)$ is the deterministic flow utility parameterized by $\theta$. The agent's value function $V(s_t, \varepsilon_t)$ satisfies the Bellman equation:

$$ V(s_t, \varepsilon_t) = \max_{a \in A} \left\{ u(s_t, a; \theta) + \varepsilon_t(a) + \beta \mathbb{E} \left[ V(s_{t+1}, \varepsilon_{t+1}) \mid s_t, a \right] \right\} $$

### 1.3 The Rust Assumptions

To make this tractable, Rust (1987) introduced the "Conditional Independence" (CI) assumption:

1.  **Additive Separability:** (As above).
2.  **I.I.D. Errors:** $\varepsilon_t$ are i.i.d. across time and distributed Multivariate Extreme Value Type I (Gumbel).
3.  **Conditional Independence:** $P(s_{t+1}, \varepsilon_{t+1} | s_t, \varepsilon_t, a_t) = P(\varepsilon_{t+1}) P(s_{t+1} | s_t, a_t)$. The errors do not predict future states once $(s_t, a_t)$ are controlled for.

### 1.4 Integrated and Choice-Specific Value Functions

Under these assumptions, we define the **Choice-Specific Value Function** $v(s, a)$:

$$ v(s, a) = u(s, a; \theta) + \beta \sum_{s'} \log \left( \sum_{k \in A} \exp(v(s', k)) \right) P(s' | s, a) $$

This equation encapsulates the core computational challenge: $v(s, a)$ appears on both sides. It is a fixed point problem.

The probability of choosing action $j$ given state $s$ takes the familiar Multinomial Logit form:

$$ P(j | s; \theta) = \frac{\exp(v(s, j))}{\sum_{k \in A} \exp(v(s, k))} $$


## 2. Direct Estimation: The Nested Fixed Point (NFP) Algorithm

John Rust's original approach (implemented in `RustNFP`) treats the problem as a constrained maximum likelihood estimation.

### 2.1 The Likelihood Function

Given a panel dataset $\{ (s_{it}, a_{it}) \}_{i=1, t=1}^{N, T}$, the log-likelihood is:

$$ \mathcal{L}(\theta) = \sum_{i=1}^N \sum_{t=1}^T \log P(a_{it} | s_{it}; \theta) $$

However, computing $P(a|s)$ requires knowing $v(s, a)$, which depends on $\theta$.

### 2.2 The Algorithm

The NFP algorithm nests a dynamic programming solver *inside* an optimization loop:

1.  **Outer Loop (Optimization):** Search over parameters $\theta$ to maximize $\mathcal{L}(\theta)$.
2.  **Inner Loop (Fixed Point):** For a candidate $\theta$:
    *   Initialize guess $v^0$.
    *   Iterate the Bellman operator $T(v)$ until convergence: $v = T(v)$.
    $$ T(v)(s, a) = u(s, a; \theta) + \beta \mathbb{E}_{s'|s,a} \left[ \log \sum_k \exp(v(s', k)) \right] $$
    *   Output the converged $v_\theta$.
3.  **Compute Likelihood:** Use $v_\theta$ to compute probabilities and the likelihood value.

**Computational Cost:** The inner loop requires solving a fixed point for every likelihood evaluation. For a state space of size $|S|$, this is roughly $O(|S|^2)$ or $O(|S|^3)$ depending on sparsity, repeated thousands of times.



## 3. Conditional Choice Probabilities (CCP): The Hotz-Miller Approach

Hotz & Miller (1993) observed that the unique mapping from value functions to choice probabilities implies a reverse mapping: **Choice probabilities contain all necessary information about value differences.**

### 3.1 The Inversion Theorem

From the logit formula, the difference in value between action $j$ and a reference action $0$ is:

$$ v(s, j) - v(s, 0) = \log \left( \frac{P(j|s)}{P(0|s)} \right) $$

This allows us to express the future value term (the "Emax" function) purely as a function of current probabilities and the current flow utility, *without* solving the fixed point.

### 3.2 The Estimator

The expected value function $\bar{V}(s)$ can be written in matrix notation (as implemented in `HotzMillerCCP`):

$$ \mathbf{\bar{V}} = (\mathbf{I} - \beta \mathbf{M})^{-1} \left[ \sum_{a} P(a) \odot (u(a) + e(a)) \right] $$

Where:
*   $\mathbf{M}$ is the state transition matrix induced by the observed policies: $M(s, s') = \sum_a P(a|s) P(s'|s,a)$.
*   $e(a)$ is the Euler-constant correction for the expected value of the error term (entropy).

### 3.3 Two-Stage Estimation

1.  **First Stage:** Estimate choice probabilities $\hat{P}(a|s)$ and transitions $\hat{P}(s'|s, a)$ directly from the data (e.g., using frequency counts or kernel regression). This is computationally free.
2.  **Second Stage:** Estimate $\theta$.
    *   Using $\hat{P}$, compute the inversion matrix $(\mathbf{I} - \beta \mathbf{M})^{-1}$.
    *   For any candidate $\theta$, compute implied values $v(s, a)$ via simple matrix algebra (no recursion).
    *   Maximize the pseudo-likelihood.

**Trade-off:** CCP is orders of magnitude faster than NFP but can suffer from finite-sample bias if the first-stage probabilities $\hat{P}$ are estimated poorly (e.g., in states with few observations).



## 4. Handling Heterogeneity

A major limitation of the basic model is assuming all agents have identical preferences $\theta$.

### 4.1 Observed Heterogeneity
If agents differ by observable traits $z_i$ (e.g., education, location), we simply expand the state space $s_{it} \rightarrow (s_{it}, z_i)$. This increases computational burden but requires no new theory.

### 4.2 Unobserved Heterogeneity (Finite Mixtures)
If agents differ by unobservable types $k \in \{1, \dots, K\}$, the likelihood becomes a mixture:

$$ \mathcal{L}(\theta) = \sum_{i=1}^N \log \left( \sum_{k=1}^K \pi_k \prod_{t=1}^T P(a_{it} | s_{it}; \theta_k) \right) $$

**Estimation via EM Algorithm (Arcidiacono & Miller, 2011):**
1.  **E-Step:** Compute the posterior probability that agent $i$ is type $k$, given their history of choices and current parameters.
2.  **M-Step:** Maximize the expected log-likelihood weighted by these posterior probabilities. This separates the maximization problem, allowing us to estimate $\theta_k$ for each type almost independently.



## 5. Connections to Inverse Reinforcement Learning (IRL)

Structural econometrics and Machine Learning have converged on the same problem from different angles. DDC is formally equivalent to **Inverse Reinforcement Learning (IRL)**.

### 5.1 The IRL Problem
In RL, given Reward $R$ and Dynamics $P$, we find the Policy $\pi$.
In **Inverse RL**, given Policy $\pi$ (observed behavior) and Dynamics $P$, we find the Reward $R$ (Utility).

### 5.2 Maximum Entropy IRL (Ziebart et al., 2008)
MaxEnt IRL assumes agents act somewhat randomly but proportionally to the exponentiated value of actions.
$$ P(\tau) \propto \exp\left(\sum_{t} R(s_t, a_t)\right) $$
This is mathematically identical to the Logit assumption in DDC.

### 5.3 Deep IRL and Adversarial Methods
Modern extensions (like GAIL - Generative Adversarial Imitation Learning) replace the structural utility function $u(s, a; \theta)$ with a Neural Network $R_\psi(s, a)$.

*   **Econometrics:** Focuses on *identifiability* and interpreting $\theta$ (e.g., "What is the dollar cost of engine replacement?").
*   **Deep IRL:** Focuses on *prediction* and high-dimensional states (e.g., "Replicate the driving style of a human").

### 5.4 The Frontier: Deep Structural Estimation

Recent work combines these:

*   Using Deep Neural Networks to approximate the Value Function $V(s)$ (DQN style) to handle the "Curse of Dimensionality" in NFP.
*   Using Neural Networks for the first-stage CCPs in Hotz-Miller to handle continuous state spaces.

## 6. Summary for `torchonometrics` Users

| Method | Class | Complexity | Best For |
| :--- | :--- | :--- | :--- |
| **NFP** | `RustNFP` | High | Small state spaces; when precision is paramount; calculating standard errors. |
| **CCP** | `HotzMillerCCP` | Low | Large state spaces; complex dynamics; when "good enough" first-stage estimates exist. |
| **Deep DDC** | `DeepValueFunction` | Variable | Continuous/High-dim states where tabular methods fail. |

The `torchonometrics` library implements these using PyTorch's automatic differentiation, removing the need for deriving complex analytical gradients for the likelihood functions.
