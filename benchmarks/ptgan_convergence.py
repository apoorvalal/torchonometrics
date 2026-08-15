#!/usr/bin/env python3
"""Equal-budget convergence benchmark for TabularPTGAN.

The target is the eight-component Gaussian ring used in Sohn and Song
(arXiv:2411.11786v2): radius 1.5 and component variance 0.01. The benchmark
tracks target-distribution quality at alpha=1 by training iteration and wall
time, using identical data, architecture, optimizer, and random seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trex import TabularPTGAN, TabularWGAN, sliced_wasserstein_distance


def gaussian_ring(
    seed: int,
    n: int,
    n_modes: int = 8,
    radius: float = 1.5,
    component_sd: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    angles = 2.0 * np.pi * np.arange(n_modes) / n_modes
    centers = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    labels = rng.integers(0, len(centers), size=n)
    rows = centers[labels] + rng.normal(scale=component_sd, size=(n, 2))
    return rows.astype(np.float32), centers


def mode_metrics(
    fake: np.ndarray,
    centers: np.ndarray,
    capture_radius: float,
) -> dict[str, float]:
    distances = np.linalg.norm(fake[:, None, :] - centers[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(centers))
    captured = np.bincount(
        nearest[distances.min(axis=1) <= capture_radius],
        minlength=len(centers),
    )
    frequencies = counts / counts.sum()
    target = np.full(len(centers), 1.0 / len(centers))
    return {
        "mode_coverage": float(np.sum(captured >= max(5, int(0.005 * len(fake))))),
        "mode_tv": float(0.5 * np.abs(frequencies - target).sum()),
        "nearest_mode_distance": float(distances.min(axis=1).mean()),
    }


def evaluate(
    model: Any,
    real_eval: np.ndarray,
    centers: np.ndarray,
    n_eval: int,
    seed: int,
    capture_radius: float,
) -> dict[str, float]:
    # Evaluation must not consume the RNG stream used by subsequent training.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fake = model.sample(n_eval).numpy()
    metrics = mode_metrics(fake, centers, capture_radius)
    metrics["sliced_wasserstein"] = sliced_wasserstein_distance(
        real_eval,
        fake,
        n_projections=256,
        seed=seed,
    )
    return metrics


def run_one(
    method: str,
    seed: int,
    train: np.ndarray,
    real_eval: np.ndarray,
    centers: np.ndarray,
    checkpoints: set[int],
    args: argparse.Namespace,
) -> list[dict[str, float | int | str]]:
    common = dict(
        hidden_dims=tuple(args.hidden_dims),
        critic_hidden_dims=tuple(reversed(args.hidden_dims)),
        noise_dim=2,
        batch_size=args.batch_size,
        max_steps=args.steps,
        critic_steps=1,
        lr=args.lr,
        betas=(0.0, 0.9),
        generator_dropout=0.0,
        critic_dropout=0.0,
        seed=seed,
        device=args.device,
    )
    if method == "WGAN-GP":
        model = TabularWGAN(gp_weight=args.gp_weight, **common)
    elif method == "PTGAN-CP":
        model = TabularPTGAN(
            temperature_ratio=args.temperature_ratio,
            coherency_weight=args.coherency_weight,
            gp_weight=0.0,
            interpolate_noise=True,
            **common,
        )
    elif method == "PTGAN-GP":
        model = TabularPTGAN(
            temperature_ratio=args.temperature_ratio,
            coherency_weight=0.0,
            gp_weight=args.gp_weight,
            interpolate_noise=True,
            **common,
        )
    else:
        raise ValueError(method)

    rows: list[dict[str, float | int | str]] = []
    training_started: float | None = None
    excluded_eval_time = 0.0

    def callback(step: int, fitted: Any) -> None:
        nonlocal training_started, excluded_eval_time
        if step not in checkpoints:
            return
        eval_started = time.perf_counter()
        if training_started is None:
            elapsed = 0.0
        else:
            elapsed = eval_started - training_started - excluded_eval_time
        row: dict[str, float | int | str] = {
            "method": method,
            "seed": seed,
            "step": step,
            "elapsed_sec": elapsed,
        }
        row.update(
            evaluate(
                fitted,
                real_eval,
                centers,
                args.n_eval,
                seed + step,
                3.0 * args.component_sd,
            )
        )
        rows.append(row)
        print(
            f"{method} seed={seed} step={step}: "
            f"SW={row['sliced_wasserstein']:.4f}, "
            f"coverage={row['mode_coverage']:.0f}",
            flush=True,
        )
        eval_finished = time.perf_counter()
        if training_started is None:
            training_started = eval_finished
        else:
            excluded_eval_time += eval_finished - eval_started

    model.fit(train, callback=callback)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    metrics = [
        "elapsed_sec",
        "sliced_wasserstein",
        "mode_coverage",
        "mode_tv",
        "nearest_mode_distance",
    ]
    keys = sorted({(row["method"], row["step"]) for row in rows})
    for method, step in keys:
        selected = [
            row for row in rows if row["method"] == method and row["step"] == step
        ]
        summary: dict[str, Any] = {
            "method": method,
            "step": step,
            "replications": len(selected),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        result.append(summary)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    summary: list[dict[str, Any]],
    path: Path,
    n_modes: int,
    x_axis: str = "step",
) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("sliced_wasserstein", "Sliced Wasserstein", False),
        ("mode_coverage", f"Modes covered (of {n_modes})", True),
        ("mode_tv", "Mode-mass total variation", False),
    ]
    methods = sorted({row["method"] for row in summary})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for method in methods:
        selected = sorted(
            [row for row in summary if row["method"] == method],
            key=lambda row: row["step"],
        )
        x_key = "step" if x_axis == "step" else "elapsed_sec_mean"
        x = np.asarray([row[x_key] for row in selected])
        for ax, (metric, title, integer_axis) in zip(axes, metrics):
            mean = np.asarray([row[f"{metric}_mean"] for row in selected])
            sd = np.asarray([row[f"{metric}_sd"] for row in selected])
            ax.plot(x, mean, marker="o", label=method)
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.16)
            ax.set_title(title)
            ax.grid(alpha=0.25)
            if integer_axis:
                ax.set_ylim(0, n_modes + 0.3)
            ax.set_xlabel("Training step" if x_axis == "step" else "Wall seconds")
    axes[0].legend(frameon=False)
    target = "Gaussian" if n_modes == 1 else f"{n_modes}-component Gaussian mixture"
    fig.suptitle(f"{target} convergence at alpha=1")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--checkpoints", default="0,50,100,200,400,800,1200")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-eval", type=int, default=4000)
    parser.add_argument("--n-modes", type=int, default=8)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--component-sd", type=float, default=0.1)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gp-weight", type=float, default=10.0)
    parser.add_argument("--temperature-ratio", type=float, default=0.9)
    parser.add_argument("--coherency-weight", type=float, default=100.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tmp/ptgan-convergence")
    )
    args = parser.parse_args()
    checkpoints = {int(value) for value in args.checkpoints.split(",")}
    checkpoints.add(0)
    checkpoints.add(args.steps)
    if max(checkpoints) > args.steps:
        raise ValueError("A checkpoint exceeds --steps.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["WGAN-GP", "PTGAN-CP"]
    if args.include_ablation:
        methods.insert(1, "PTGAN-GP")
    rows: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        train, centers = gaussian_ring(
            10_000 + seed,
            args.n_train,
            args.n_modes,
            args.radius,
            args.component_sd,
        )
        real_eval, _ = gaussian_ring(
            20_000 + seed,
            args.n_eval,
            args.n_modes,
            args.radius,
            args.component_sd,
        )
        for method in methods:
            rows.extend(
                run_one(
                    method,
                    seed,
                    train,
                    real_eval,
                    centers,
                    checkpoints,
                    args,
                )
            )

    summary = summarize(rows)
    write_csv(args.output_dir / "convergence_draws.csv", rows)
    write_csv(args.output_dir / "convergence_summary.csv", summary)
    make_plot(
        summary,
        args.output_dir / "convergence_by_step.png",
        args.n_modes,
        x_axis="step",
    )
    make_plot(
        summary,
        args.output_dir / "convergence_by_time.png",
        args.n_modes,
        x_axis="time",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
