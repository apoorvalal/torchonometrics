# Lalonde SimDGP Benchmark

This benchmark compares tabular simulators on the Lalonde-Dehejia-Wahba samples
from `evanmunro/dswgan-paper`. It assumes that repository is cloned next to
Trex as `../dswgan-paper`.

Use the existing conda environment for this work:

```bash
uv pip install --python /home/alal/miniforge3/envs/torch/bin/python -e .
conda run -n torch python benchmarks/lalonde_simdgp.py \
  --paper-repo ../dswgan-paper \
  --dataset exp \
  --methods wgan diffusion \
  --steps 1000 \
  --sample-size 445
```

The benchmark writes synthetic samples and metric summaries to
`tmp/simdgp_benchmark` by default. Metrics are reported on both raw columns and
standardized columns:

- marginal Wasserstein distance and Kolmogorov-Smirnov distance by column
- mean-vector distance
- covariance and correlation matrix discrepancy
- sliced Wasserstein distance for joint distribution fit

LLM methods require a local safetensors causal language model:

```bash
conda run -n torch python benchmarks/lalonde_simdgp.py \
  --methods llm-icl \
  --llm-model /path/to/local/model
```

QLoRA can either train a new adapter or load an existing PEFT adapter:

```bash
conda run -n torch python benchmarks/lalonde_simdgp.py \
  --methods llm-qlora \
  --llm-model /path/to/base/model \
  --fit-adapter \
  --adapter-output tmp/simdgp_lora/lalonde-exp

conda run -n torch python benchmarks/lalonde_simdgp.py \
  --methods llm-qlora \
  --llm-model /path/to/base/model \
  --adapter-path tmp/simdgp_lora/lalonde-exp
```
