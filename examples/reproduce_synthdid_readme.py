"""Reproduce the synthdid README panel-estimator point-estimate table.

Run from the repository root:

    uv run python examples/reproduce_synthdid_readme.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from trex.panel import panel_estimates


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHDID_DATA = REPO_ROOT.parent / "_refs" / "synthdid" / "data" / "california_prop99.csv"


def load_california_prop99(path: Path = SYNTHDID_DATA):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(
                {
                    "state": row["State"],
                    "year": int(row["Year"]),
                    "packs": float(row["PacksPerCapita"]),
                    "treated": int(row["treated"]),
                }
            )
    states = sorted({r["state"] for r in rows})
    years = sorted({r["year"] for r in rows})
    treated_states = sorted({r["state"] for r in rows if r["treated"] == 1})
    control_states = [s for s in states if s not in treated_states]
    ordered_states = control_states + treated_states
    index = {(r["state"], r["year"]): r for r in rows}
    Y = torch.empty((len(ordered_states), len(years)), dtype=torch.float64)
    W = torch.empty_like(Y, dtype=torch.bool)
    for i, state in enumerate(ordered_states):
        for j, year in enumerate(years):
            r = index[(state, year)]
            Y[i, j] = r["packs"]
            W[i, j] = bool(r["treated"])
    treated_unit = W.any(dim=1)
    treated_time = W.any(dim=0)
    N0 = int((~treated_unit).sum().item())
    T0 = int((~treated_time).sum().item())
    return Y, N0, T0, ordered_states, years


def main() -> None:
    Y, N0, T0, states, years = load_california_prop99()
    estimates = panel_estimates(
        Y,
        N0,
        T0,
        # Keep MC reasonably fast and deterministic for the README reproduction.
        mc_kwargs={"lambda_fraction": 0.15, "maxiter": 400, "tol": 1e-7},
        sdid_kwargs={"maxiter": 10_000, "min_decrease": 1e-5, "sparsify": True},
    )
    print(f"California Prop 99: N0={N0}, T0={T0}, treated={states[N0:]}, years={years[0]}-{years[-1]}")
    print("\nEstimator                         Estimate")
    print("------------------------------------------")
    for name, value in estimates.items():
        print(f"{name:<34} {float(value):9.3f}")


if __name__ == "__main__":
    main()
