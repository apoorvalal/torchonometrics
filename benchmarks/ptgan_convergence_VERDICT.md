# PTGAN convergence spike

**Verdict: PARTIAL.** The implementation validates the paper's stabilization
mechanism on its canonical multimodal target, but this benchmark does not yet
establish gains on real tabular econometric data.

## Question

Does the parallel tempering proposal in Sohn and Song,
*Parallelly Tempered Generative Adversarial Nets: Toward Stabilized Gradients*
([arXiv:2411.11786v2](https://arxiv.org/abs/2411.11786)), improve convergence of
Trex's tabular WGAN?

## Implementation

`TabularPTGAN` follows Algorithm 3 and its minibatch construction:

- learn the family `alpha * X1 + (1 - alpha) * X2` jointly;
- draw `alpha` from `r * delta_1 + (1 - r) * Uniform(0, 1)`;
- condition both networks on the symmetric feature
  `1 - 2 * abs(alpha - 0.5)`;
- optionally interpolate reference noise as
  `alpha * Z1 + (1 - alpha) * Z2`;
- regularize the critic with the paper's squared directional-derivative
  coherency penalty; and
- sample at `alpha=1` for the original data distribution.

The implementation also records the critic loss, generator loss, coherency
penalty, optional gradient penalty, and critic gradient norm. Both PTGAN and
the existing WGAN accept a callback so convergence can be measured during
training without changing the optimization path.

## Evidence

The equal-budget benchmark uses the paper's eight-component Gaussian ring:
radius 1.5 and component variance 0.01. Each method receives the same 4,000
training rows, 4,000 evaluation rows, seed, 64-by-64 MLPs, batch size 100,
1,500 critic/generator steps, Adam learning rate `1e-4`, and betas `(0, 0.9)`.
PTGAN uses `r=0.9` and the paper default coherency weight 100. Results are 10
paired replications evaluated at `alpha=1`.

| Method | SW at 1,500 | Modes at 1,500 | Mode TV | Full coverage | Train seconds |
|---|---:|---:|---:|---:|---:|
| PTGAN-CP | **0.219** | **8.0 / 8** | **0.068** | **10 / 10** | **1.092** |
| PTGAN-GP | 0.728 | 2.4 / 8 | 0.587 | 0 / 10 | 1.318 |
| WGAN-GP | 0.808 | 2.5 / 8 | 0.553 | 1 / 10 | 1.260 |

At step 800, PTGAN-CP already has mean sliced Wasserstein 0.407 and covers 4.3
modes, versus 0.715 and 1.9 modes for WGAN-GP. At step 1,200 it reaches 0.265
and 7.7 modes, versus 0.739 and 2.4 modes. At the final step, the paired
PTGAN-minus-WGAN sliced-Wasserstein difference is -0.588 (95% t interval
[-0.814, -0.362], paired p=0.00023). Absolute CPU timings are machine-specific,
but the full PTGAN is about 13% faster here because its directional penalty is
cheaper than the WGAN gradient-norm penalty.

The PTGAN-GP ablation is the important failure case: convex tempering plus an
ordinary gradient penalty does not stabilize the run. The paper's coherency
penalty, rather than interpolation by itself, drives the result.

## Edge case

On a single Gaussian with the same variance, both methods converge completely.
After 1,000 steps PTGAN-CP has sliced Wasserstein 0.0147 and WGAN-GP 0.0194;
both cover the sole mode in all 10 replications. This does not reproduce the
paper's warning that vanilla training may be more efficient on simple targets,
but it does show that the large multimodal advantage becomes practically
minor when there is no mode-collapse problem to solve.

## What worked

- The exact coherency penalty produces stable, monotone late-stage improvement.
- All ten PTGAN runs recover all eight modes; the comparison methods do not.
- Training at all temperatures does not slow target-temperature convergence in
  this small CPU benchmark.

## What remains unvalidated

- Performance on mixed continuous/binary tabular datasets and downstream
  econometric estimands.
- Sensitivity to the temperature ratio `r` and coherency weight outside this
  paper-aligned setup.
- Large-network and GPU behavior.
- The paper's broader gradient-variance and statistical-rate claims; this spike
  measures distributional convergence, not its theorem assumptions.

## Reproduction

```bash
python benchmarks/ptgan_convergence.py \
  --steps 1500 \
  --checkpoints 0,50,100,200,400,800,1200,1500 \
  --seeds 10 \
  --n-train 4000 \
  --n-eval 4000 \
  --hidden-dims 64 64 \
  --batch-size 100 \
  --coherency-weight 100 \
  --temperature-ratio 0.9 \
  --include-ablation \
  --output-dir tmp/ptgan-convergence-main
```

