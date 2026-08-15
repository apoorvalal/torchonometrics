# Repo-Specific Agent Notes

These notes refine the higher-level workflow for this repository only.

## Local Environment

- Use the shared conda environment named `torch` for this checkout on this machine.
- Do not create or maintain a repo-local `.venv` for Trex work unless the user explicitly asks for one.
- Run Python commands through `conda run -n torch ...` or an already activated `torch` shell.
- Install editable Trex into that environment when needed:

```bash
uv pip install --python /home/alal/miniforge3/envs/torch/bin/python -e .
```

- Heavy GPU work, including Torch, Hugging Face, LoRA, bitsandbytes, wandb, and llama.cpp CUDA checks, belongs in the shared `torch` environment.
- If a stale `.venv` appears in the repo, treat it as disposable local state unless the user says otherwise.

## Simulation-DGP Work

- The Trex package should keep the general simulator code: WGAN, diffusion, tabular transforms, scoring utilities, and focused benchmark CLI support.
- llama.cpp endpoint ICL is research exploration, not Trex API surface. Keep endpoint-serving scripts, prompt experiments, and one-off ICL reproductions in `/home/alal/Dropbox/1_Research/doodles/lalonde_simdgp_benchmark`, not in package modules.
- The Athey-Imbens-Metzger-Munro numerical repo is expected at `../dswgan-paper` when running local Lalonde benchmarks from this checkout.
- Use `tmp/` for local benchmark artifacts; it is ignored and should not be staged.
- When comparing generators on Lalonde, report standardized distribution metrics as the primary comparison because raw earnings scale dominates raw distances.

## Quarto Reports

- For Trex reports, use Quarto only when a rendered report is the deliverable.
- Render with the conda `torch` Python:

```bash
QUARTO_PYTHON=/home/alal/miniforge3/envs/torch/bin/python quarto render <report>.qmd --to html
```

- Use `embed-resources: true`, `page-layout: full`, and folded code for HTML reports.
- Keep math in `$...$` and `$$...$$` form.
- Render and inspect math-heavy reports before handing them off.

## Notebook Format

- In this repository, notebooks under `nb/` should be authored as `.ipynb` by default.
- Prefer `.ipynb` over `qmd` for exploratory notes, demos, and review notebooks because GitHub renders notebooks natively.
- Use `qmd` only when the user explicitly asks for a Quarto report or when a rendered HTML artifact is the primary deliverable.

## Notebook Outputs

- For notebooks intended for review in GitHub, keep the executed outputs in the `.ipynb` file when they materially improve readability.
- Visual diagnostics are part of the deliverable for synthetic-control and counterfactual-prediction notebooks. Do not substitute tables when the core object is path geometry over time.
