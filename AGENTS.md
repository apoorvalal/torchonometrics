# Repo-Specific Agent Notes

These notes refine the higher-level workflow for this repository only.

## Notebook Format

- In this repository, notebooks under `nb/` should be authored as `.ipynb` by default.
- Prefer `.ipynb` over `qmd` for exploratory notes, demos, and review notebooks because GitHub renders notebooks natively.
- Use `qmd` only when the user explicitly asks for a Quarto report or when a rendered HTML artifact is the primary deliverable.

## Notebook Outputs

- For notebooks intended for review in GitHub, keep the executed outputs in the `.ipynb` file when they materially improve readability.
- Visual diagnostics are part of the deliverable for synthetic-control and counterfactual-prediction notebooks. Do not substitute tables when the core object is path geometry over time.
