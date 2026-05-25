"""Compare tabular DGP simulators on Lalonde-Dehejia-Wahba data.

Examples
--------
Run a quick CPU/GPU smoke benchmark from the cloned paper replication repo:

    conda run -n torch python benchmarks/lalonde_simdgp.py \
        --paper-repo ../dswgan-paper --dataset exp \
        --methods wgan diffusion --steps 50 --sample-size 445

Run ICL from a local safetensors causal LM:

    conda run -n torch python benchmarks/lalonde_simdgp.py \
        --methods llm-icl --llm-model /path/to/model
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from trex import (
    SafetensorsLLMInContextGenerator,
    SafetensorsQLORAGenerator,
    TabularDiffusion,
    TabularTransformer,
    TabularWGAN,
    distribution_metrics,
)


LDW_COLUMNS = [
    "t",
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
    "re78",
]
BINARY_COLUMNS = ["t", "black", "hispanic", "married", "nodegree"]
NONNEGATIVE_COLUMNS = ["age", "education", "re74", "re75", "re78"]


@dataclass
class BenchmarkOutput:
    dataset: str
    method: str
    n_real: int
    n_fake: int
    fit_seconds: float
    sample_seconds: float
    metrics: dict[str, float]

    def as_flat_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "dataset": self.dataset,
            "method": self.method,
            "n_real": self.n_real,
            "n_fake": self.n_fake,
            "fit_seconds": self.fit_seconds,
            "sample_seconds": self.sample_seconds,
        }
        row.update(self.metrics)
        return row


def load_lalonde_dataset(paper_repo: Path, dataset: str) -> pd.DataFrame:
    data_path = paper_repo / "data" / "original_data" / f"{dataset}_merged.feather"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find Lalonde data file: {data_path}")
    df = pd.read_feather(data_path)
    missing = [column for column in LDW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df[LDW_COLUMNS].astype(float)


def fit_and_sample(
    method: str,
    real: pd.DataFrame,
    args: argparse.Namespace,
    transformer: TabularTransformer,
) -> tuple[np.ndarray, float, float]:
    if method == "wgan":
        train = transformer.transform(real)
        lower, upper = transformer.transformed_bounds()
        model = TabularWGAN(
            hidden_dims=tuple(args.hidden_dims),
            critic_hidden_dims=tuple(reversed(args.hidden_dims)),
            batch_size=args.batch_size,
            max_steps=args.steps,
            critic_steps=args.critic_steps,
            lr=args.lr,
            gp_weight=args.gp_weight,
            binary_dims=transformer.binary_indices,
            lower_bounds=lower,
            upper_bounds=upper,
            seed=args.seed,
            device=args.device,
        )
        fit_start = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - fit_start
        sample_start = time.perf_counter()
        fake_z = model.sample(args.sample_size).numpy()
        sample_seconds = time.perf_counter() - sample_start
        fake = transformer.inverse_transform(fake_z, random_state=args.seed)
        return fake, fit_seconds, sample_seconds

    if method == "diffusion":
        train = transformer.transform(real)
        model = TabularDiffusion(
            hidden_dims=tuple(args.hidden_dims),
            n_timesteps=args.diffusion_timesteps,
            batch_size=args.batch_size,
            max_steps=args.steps,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
        )
        fit_start = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - fit_start
        sample_start = time.perf_counter()
        fake_z = model.sample(args.sample_size).numpy()
        sample_seconds = time.perf_counter() - sample_start
        fake = transformer.inverse_transform(fake_z, random_state=args.seed)
        return fake, fit_seconds, sample_seconds

    if method == "llm-icl":
        _require_safetensors_model(args.llm_model)
        model = SafetensorsLLMInContextGenerator(
            model_path=args.llm_model,
            tokenizer_path=args.tokenizer_path,
            examples=args.icl_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.llm_greedy,
            load_in_4bit=args.llm_4bit,
            max_attempts=args.llm_max_attempts,
            rows_per_prompt=args.llm_rows_per_prompt,
            progress_path=str(args.output_dir / f"{args.dataset}_{method}_progress.csv"),
            device=args.device,
        )
        fit_start = time.perf_counter()
        model.fit(real.to_numpy(), column_names=list(real.columns))
        fit_seconds = time.perf_counter() - fit_start
        sample_start = time.perf_counter()
        fake = model.sample(args.sample_size)
        sample_seconds = time.perf_counter() - sample_start
        return postprocess_rows(fake, real.columns), fit_seconds, sample_seconds

    if method == "llm-qlora":
        _require_safetensors_model(args.llm_model)
        model = SafetensorsQLORAGenerator(
            model_path=args.llm_model,
            tokenizer_path=args.tokenizer_path,
            examples=args.icl_examples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=not args.llm_greedy,
            load_in_4bit=args.llm_4bit,
            max_attempts=args.llm_max_attempts,
            rows_per_prompt=args.llm_rows_per_prompt,
            progress_path=str(args.output_dir / f"{args.dataset}_{method}_progress.csv"),
            device=args.device,
        )
        fit_start = time.perf_counter()
        model.fit(real.to_numpy(), column_names=list(real.columns))
        if args.fit_adapter:
            if args.adapter_output is None:
                raise ValueError("--adapter-output is required with --fit-adapter.")
            model.fit_adapter(
                output_dir=args.adapter_output,
                num_train_epochs=args.qlora_epochs,
                learning_rate=args.qlora_lr,
                per_device_train_batch_size=args.qlora_batch_size,
                gradient_accumulation_steps=args.qlora_grad_accumulation,
                rows_per_completion=args.qlora_rows_per_completion,
                train_samples=args.qlora_train_samples,
                max_length=args.qlora_max_length,
                lora_r=args.qlora_r,
                lora_alpha=args.qlora_alpha,
                lora_dropout=args.qlora_dropout,
                seed=args.seed,
            )
        elif args.adapter_path is not None:
            _require_adapter_path(args.adapter_path)
            model.adapter_path = args.adapter_path
            model.tokenizer_path = args.adapter_path
        fit_seconds = time.perf_counter() - fit_start
        sample_start = time.perf_counter()
        fake = model.sample(args.sample_size)
        sample_seconds = time.perf_counter() - sample_start
        return postprocess_rows(fake, real.columns), fit_seconds, sample_seconds

    raise ValueError(f"Unknown method: {method}")


def score_fake(
    real: pd.DataFrame,
    fake: np.ndarray,
    transformer: TabularTransformer,
    seed: int,
) -> dict[str, float]:
    fake_df = pd.DataFrame(fake, columns=real.columns)
    raw_metrics = distribution_metrics(real.to_numpy(), fake_df.to_numpy(), seed=seed)
    standardized_metrics = distribution_metrics(
        transformer.transform(real),
        transformer.transform(fake_df),
        seed=seed,
    )
    return {
        **{f"raw_{key}": value for key, value in raw_metrics.items()},
        **{f"z_{key}": value for key, value in standardized_metrics.items()},
    }


def postprocess_rows(fake: np.ndarray, columns: pd.Index) -> np.ndarray:
    clipped = np.asarray(fake, dtype=np.float64).copy()
    column_index = {column: idx for idx, column in enumerate(columns)}
    for column in BINARY_COLUMNS:
        idx = column_index[column]
        clipped[:, idx] = (np.clip(clipped[:, idx], 0.0, 1.0) >= 0.5).astype(float)
    for column in NONNEGATIVE_COLUMNS:
        idx = column_index[column]
        clipped[:, idx] = np.maximum(clipped[:, idx], 0.0)
    return clipped


def _require_safetensors_model(model_path: str | None) -> None:
    if model_path is None:
        raise ValueError("--llm-model is required for LLM methods.")
    path = Path(model_path)
    if not path.exists():
        if "/" in model_path and not model_path.startswith(("/", ".")):
            return
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if path.is_dir() and not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"No safetensors weights found in {path}")
    if path.is_file() and path.suffix != ".safetensors":
        raise ValueError("Expected a .safetensors file or directory containing safetensors.")


def _require_adapter_path(adapter_path: str) -> None:
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {path}")
    if not (path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"No adapter_model.safetensors found in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo", type=Path, default=Path("../dswgan-paper"))
    parser.add_argument("--dataset", choices=["exp", "cps", "psid"], default="exp")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["wgan", "diffusion", "llm-icl", "llm-qlora"],
        default=["wgan", "diffusion"],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/simdgp_benchmark"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gp-weight", type=float, default=5.0)
    parser.add_argument("--diffusion-timesteps", type=int, default=100)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--icl-examples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--llm-greedy", action="store_true")
    parser.add_argument("--llm-4bit", action="store_true")
    parser.add_argument("--llm-max-attempts", type=int, default=20)
    parser.add_argument("--llm-rows-per-prompt", type=int, default=None)
    parser.add_argument("--fit-adapter", action="store_true")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--adapter-output", default=None)
    parser.add_argument("--qlora-epochs", type=float, default=1.0)
    parser.add_argument("--qlora-lr", type=float, default=2e-4)
    parser.add_argument("--qlora-batch-size", type=int, default=1)
    parser.add_argument("--qlora-grad-accumulation", type=int, default=8)
    parser.add_argument("--qlora-rows-per-completion", type=int, default=8)
    parser.add_argument("--qlora-train-samples", type=int, default=None)
    parser.add_argument("--qlora-max-length", type=int, default=1024)
    parser.add_argument("--qlora-r", type=int, default=16)
    parser.add_argument("--qlora-alpha", type=int, default=32)
    parser.add_argument("--qlora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    real = load_lalonde_dataset(args.paper_repo, args.dataset)
    if args.sample_size is None:
        args.sample_size = len(real)

    transformer = TabularTransformer(
        column_names=list(real.columns),
        binary_columns=BINARY_COLUMNS,
        nonnegative_columns=NONNEGATIVE_COLUMNS,
    ).fit(real)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[BenchmarkOutput] = []
    for method in args.methods:
        fake, fit_seconds, sample_seconds = fit_and_sample(method, real, args, transformer)
        fake = postprocess_rows(fake, real.columns)
        fake_df = pd.DataFrame(fake, columns=real.columns)
        fake_path = args.output_dir / f"{args.dataset}_{method}_fake.csv"
        fake_df.to_csv(fake_path, index=False)
        metrics = score_fake(real, fake, transformer, args.seed)
        result = BenchmarkOutput(
            dataset=args.dataset,
            method=method,
            n_real=len(real),
            n_fake=len(fake_df),
            fit_seconds=fit_seconds,
            sample_seconds=sample_seconds,
            metrics=metrics,
        )
        results.append(result)
        print(json.dumps(result.as_flat_row(), sort_keys=True))

    metrics_path = args.output_dir / f"{args.dataset}_metrics.csv"
    pd.DataFrame([result.as_flat_row() for result in results]).to_csv(metrics_path, index=False)
    summary_path = args.output_dir / f"{args.dataset}_metrics.json"
    summary_path.write_text(
        json.dumps([result.as_flat_row() for result in results], indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
