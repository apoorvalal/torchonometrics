"""Simulation DGPs for tabular econometric Monte Carlo designs.

The estimators here share a small tensor-first interface: fit on an empirical
sample and draw synthetic rows from the fitted distribution. Optional wrappers
for LLM-based row generation keep model-loading dependencies out of the core
package import path.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseEstimator


def _as_tensor(
    value: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def _as_2d_tensor(
    value: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    tensor = _as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError("Expected a two-dimensional tabular array.")
    return tensor


def _column_indices(
    columns: Optional[list[str]],
    selected: Optional[Iterable[int | str]],
) -> list[int]:
    if selected is None:
        return []
    result: list[int] = []
    for item in selected:
        if isinstance(item, str):
            if columns is None:
                raise ValueError("Column names are required for string column selectors.")
            result.append(columns.index(item))
        else:
            result.append(int(item))
    return result


@dataclass
class TabularTransformer:
    """Standardize continuous columns while preserving binary columns.

    Parameters
    ----------
    column_names : list of str, optional
        Names used when accepting or returning pandas data frames.
    binary_columns : iterable of str or int, optional
        Columns treated as Bernoulli indicators. They are left on the original
        0/1 scale during training and rounded after generation.
    nonnegative_columns : iterable of str or int, optional
        Columns clipped at zero after inverse transformation.
    eps : float, default=1e-6
        Added to standard deviations for numerical stability.
    """

    column_names: Optional[list[str]] = None
    binary_columns: Optional[Iterable[int | str]] = None
    nonnegative_columns: Optional[Iterable[int | str]] = None
    eps: float = 1e-6

    def fit(self, data: Any) -> "TabularTransformer":
        values, columns = self._values_and_columns(data)
        if self.column_names is None:
            self.column_names = columns

        self.binary_indices = _column_indices(self.column_names, self.binary_columns)
        self.nonnegative_indices = _column_indices(
            self.column_names,
            self.nonnegative_columns,
        )
        self.continuous_indices = [
            j for j in range(values.shape[1]) if j not in set(self.binary_indices)
        ]
        self.mean_ = values.mean(axis=0)
        self.std_ = values.std(axis=0) + self.eps
        self.mean_[self.binary_indices] = 0.0
        self.std_[self.binary_indices] = 1.0
        return self

    def transform(self, data: Any) -> np.ndarray:
        self._check_is_fitted()
        values, _ = self._values_and_columns(data)
        return ((values - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, data: Any) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(
        self,
        values: Any,
        *,
        sample_binary: bool = False,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        array = np.asarray(values, dtype=np.float64) * self.std_ + self.mean_

        if self.binary_indices:
            probs = np.clip(array[:, self.binary_indices], 0.0, 1.0)
            if sample_binary:
                rng = np.random.default_rng(random_state)
                array[:, self.binary_indices] = rng.binomial(1, probs)
            else:
                array[:, self.binary_indices] = (probs >= 0.5).astype(np.float64)

        if self.nonnegative_indices:
            array[:, self.nonnegative_indices] = np.maximum(
                array[:, self.nonnegative_indices],
                0.0,
            )
        return array

    def transformed_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_is_fitted()
        lower = np.full_like(self.mean_, -np.inf, dtype=np.float64)
        upper = np.full_like(self.mean_, np.inf, dtype=np.float64)
        if self.binary_indices:
            lower[self.binary_indices] = 0.0
            upper[self.binary_indices] = 1.0
        if self.nonnegative_indices:
            lower[self.nonnegative_indices] = np.maximum(
                lower[self.nonnegative_indices],
                0.0,
            )
        return (
            ((lower - self.mean_) / self.std_).astype(np.float32),
            ((upper - self.mean_) / self.std_).astype(np.float32),
        )

    def _values_and_columns(self, data: Any) -> tuple[np.ndarray, Optional[list[str]]]:
        if hasattr(data, "columns") and hasattr(data, "to_numpy"):
            return data.to_numpy(dtype=np.float64), list(data.columns)
        return np.asarray(data, dtype=np.float64), self.column_names

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "mean_"):
            raise RuntimeError("TabularTransformer must be fitted before use.")


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(din, dout), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _WGANGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
        lower_bounds: Optional[torch.Tensor],
        upper_bounds: Optional[torch.Tensor],
        binary_dims: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.noise_dim = int(noise_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.mlp = _MLP(noise_dim + context_dim, output_dim, hidden_dims, dropout)
        self.binary_dims = tuple(binary_dims)
        if lower_bounds is not None:
            self.register_buffer("lower_bounds", lower_bounds)
        else:
            self.lower_bounds = None
        if upper_bounds is not None:
            self.register_buffer("upper_bounds", upper_bounds)
        else:
            self.upper_bounds = None

    def forward(self, context: Optional[torch.Tensor] = None, n: Optional[int] = None) -> torch.Tensor:
        if context is None:
            if n is None:
                raise ValueError("Pass either context or n.")
            context = torch.zeros(n, 0, device=next(self.parameters()).device)
        noise = torch.randn(context.shape[0], self.noise_dim, device=context.device)
        output = self.mlp(torch.cat([noise, context], dim=1))
        if self.binary_dims:
            output[:, self.binary_dims] = torch.sigmoid(output[:, self.binary_dims])
        if self.lower_bounds is not None:
            output = torch.maximum(output, self.lower_bounds)
        if self.upper_bounds is not None:
            output = torch.minimum(output, self.upper_bounds)
        return output


class _WGANCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.mlp = _MLP(input_dim + context_dim, 1, hidden_dims, dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if context is None:
            context = torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)
        return self.mlp(torch.cat([x, context], dim=1))


class TabularWGAN(BaseEstimator):
    """Wasserstein GAN with gradient penalty for tabular simulation."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
        critic_hidden_dims: Optional[tuple[int, ...]] = None,
        noise_dim: Optional[int] = None,
        batch_size: int = 128,
        max_steps: int = 1000,
        critic_steps: int = 5,
        lr: float = 1e-4,
        gp_weight: float = 5.0,
        generator_dropout: float = 0.1,
        critic_dropout: float = 0.0,
        binary_dims: Iterable[int] = (),
        lower_bounds: Optional[Any] = None,
        upper_bounds: Optional[Any] = None,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(device=device)
        self.hidden_dims = tuple(hidden_dims)
        self.critic_hidden_dims = (
            tuple(critic_hidden_dims) if critic_hidden_dims is not None else tuple(reversed(hidden_dims))
        )
        self.noise_dim = noise_dim
        self.batch_size = int(batch_size)
        self.max_steps = int(max_steps)
        self.critic_steps = int(critic_steps)
        self.lr = float(lr)
        self.gp_weight = float(gp_weight)
        self.generator_dropout = float(generator_dropout)
        self.critic_dropout = float(critic_dropout)
        self.binary_dims = tuple(int(j) for j in binary_dims)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.seed = seed
        self.history: dict[str, list[float]] = {"critic_loss": [], "generator_loss": []}

    def fit(self, X: Any, context: Optional[Any] = None) -> "TabularWGAN":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        x = _as_2d_tensor(X, self.device)
        c = self._context_tensor(context, x.shape[0])
        input_dim = x.shape[1]
        context_dim = c.shape[1]
        noise_dim = int(self.noise_dim or input_dim)
        lower = self._bound_tensor(self.lower_bounds, input_dim)
        upper = self._bound_tensor(self.upper_bounds, input_dim)

        self.generator = _WGANGenerator(
            noise_dim,
            context_dim,
            input_dim,
            self.hidden_dims,
            self.generator_dropout,
            lower,
            upper,
            self.binary_dims,
        ).to(self.device)
        self.critic = _WGANCritic(
            input_dim,
            context_dim,
            self.critic_hidden_dims,
            self.critic_dropout,
        ).to(self.device)

        loader = DataLoader(
            TensorDataset(x, c),
            batch_size=min(self.batch_size, x.shape[0]),
            shuffle=True,
            drop_last=False,
        )
        generator_opt = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9))
        critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.5, 0.9))

        data_iter = iter(loader)
        for step in range(self.max_steps):
            for _ in range(self.critic_steps):
                try:
                    real_x, real_c = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    real_x, real_c = next(data_iter)
                fake_x = self.generator(real_c).detach()
                critic_loss = (
                    self.critic(fake_x, real_c).mean()
                    - self.critic(real_x, real_c).mean()
                    + self.gp_weight * self._gradient_penalty(real_x, fake_x, real_c)
                )
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

            try:
                real_x, real_c = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                real_x, real_c = next(data_iter)
            fake_x = self.generator(real_c)
            generator_loss = -self.critic(fake_x, real_c).mean()
            generator_opt.zero_grad()
            generator_loss.backward()
            generator_opt.step()

            self.history["critic_loss"].append(float(critic_loss.detach().cpu()))
            self.history["generator_loss"].append(float(generator_loss.detach().cpu()))

        self.params = {
            "generator_state": {
                name: value.detach().cpu() for name, value in self.generator.state_dict().items()
            },
            "critic_state": {
                name: value.detach().cpu() for name, value in self.critic.state_dict().items()
            },
        }
        return self

    def sample(self, n: int, context: Optional[Any] = None) -> torch.Tensor:
        if not hasattr(self, "generator"):
            raise RuntimeError("TabularWGAN must be fitted before sampling.")
        self.generator.eval()
        with torch.no_grad():
            c = self._context_tensor(context, n)
            return self.generator(c, n=n).detach().cpu()

    def _context_tensor(self, context: Optional[Any], n: int) -> torch.Tensor:
        if context is None:
            return torch.zeros(n, 0, device=self.device)
        c = _as_2d_tensor(context, self.device)
        if c.shape[0] != n:
            raise ValueError("Context rows must match X rows or requested sample size.")
        return c

    def _bound_tensor(self, bound: Optional[Any], dim: int) -> Optional[torch.Tensor]:
        if bound is None:
            return None
        tensor = _as_tensor(bound, self.device)
        if tensor.numel() != dim:
            raise ValueError("Bounds must have one value per generated column.")
        return tensor.reshape(1, dim)

    def _gradient_penalty(
        self,
        real_x: torch.Tensor,
        fake_x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.rand(real_x.shape[0], 1, device=real_x.device)
        mixed = (alpha * real_x + (1.0 - alpha) * fake_x).requires_grad_(True)
        critic_mixed = self.critic(mixed, context)
        gradients = torch.autograd.grad(
            outputs=critic_mixed,
            inputs=mixed,
            grad_outputs=torch.ones_like(critic_mixed),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return F.relu(gradients.norm(2, dim=1) - 1.0).pow(2).mean()


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0, np.log(10000.0), half, device=t.device) * -1.0
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class _Denoiser(nn.Module):
    def __init__(
        self,
        data_dim: int,
        context_dim: int,
        hidden_dims: tuple[int, ...],
        time_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_embedding = _TimeEmbedding(time_dim)
        self.net = _MLP(data_dim + context_dim + time_dim, data_dim, hidden_dims, dropout)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context is None:
            context = torch.zeros(x_t.shape[0], 0, device=x_t.device, dtype=x_t.dtype)
        return self.net(torch.cat([x_t, context, self.time_embedding(t)], dim=1))


class TabularDiffusion(BaseEstimator):
    """Small DDPM-style diffusion model for continuous tabular rows."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
        time_dim: int = 32,
        n_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        batch_size: int = 128,
        max_steps: int = 1000,
        lr: float = 1e-3,
        dropout: float = 0.0,
        seed: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(device=device)
        self.hidden_dims = tuple(hidden_dims)
        self.time_dim = int(time_dim)
        self.n_timesteps = int(n_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.batch_size = int(batch_size)
        self.max_steps = int(max_steps)
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.seed = seed
        self.history: dict[str, list[float]] = {"loss": []}

    def fit(self, X: Any, context: Optional[Any] = None) -> "TabularDiffusion":
        if self.seed is not None:
            torch.manual_seed(self.seed)
        x = _as_2d_tensor(X, self.device)
        c = self._context_tensor(context, x.shape[0])
        self._setup_schedule()
        self.denoiser = _Denoiser(
            x.shape[1],
            c.shape[1],
            self.hidden_dims,
            self.time_dim,
            self.dropout,
        ).to(self.device)
        opt = torch.optim.AdamW(self.denoiser.parameters(), lr=self.lr)
        loader = DataLoader(
            TensorDataset(x, c),
            batch_size=min(self.batch_size, x.shape[0]),
            shuffle=True,
            drop_last=False,
        )
        data_iter = iter(loader)
        for _ in range(self.max_steps):
            try:
                x0, ctx = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x0, ctx = next(data_iter)
            t = torch.randint(0, self.n_timesteps, (x0.shape[0],), device=self.device)
            noise = torch.randn_like(x0)
            sqrt_alpha_bar = self.sqrt_alpha_bars[t].unsqueeze(1)
            sqrt_one_minus = self.sqrt_one_minus_alpha_bars[t].unsqueeze(1)
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
            pred_noise = self.denoiser(x_t, t, ctx)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.history["loss"].append(float(loss.detach().cpu()))

        self.params = {
            "denoiser_state": {
                name: value.detach().cpu() for name, value in self.denoiser.state_dict().items()
            }
        }
        return self

    def sample(self, n: int, context: Optional[Any] = None) -> torch.Tensor:
        if not hasattr(self, "denoiser"):
            raise RuntimeError("TabularDiffusion must be fitted before sampling.")
        self.denoiser.eval()
        c = self._context_tensor(context, n)
        data_dim = self.denoiser.net.net[-1].out_features
        x = torch.randn(n, data_dim, device=self.device)
        with torch.no_grad():
            for step in reversed(range(self.n_timesteps)):
                t = torch.full((n,), step, device=self.device, dtype=torch.long)
                beta = self.betas[t].unsqueeze(1)
                alpha = self.alphas[t].unsqueeze(1)
                alpha_bar = self.alpha_bars[t].unsqueeze(1)
                pred_noise = self.denoiser(x, t, c)
                mean = (x - beta / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
                if step > 0:
                    x = mean + torch.sqrt(beta) * torch.randn_like(x)
                else:
                    x = mean
        return x.detach().cpu()

    def _setup_schedule(self) -> None:
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.n_timesteps,
            device=self.device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def _context_tensor(self, context: Optional[Any], n: int) -> torch.Tensor:
        if context is None:
            return torch.zeros(n, 0, device=self.device)
        c = _as_2d_tensor(context, self.device)
        if c.shape[0] != n:
            raise ValueError("Context rows must match X rows or requested sample size.")
        return c


class SafetensorsLLMInContextGenerator(BaseEstimator):
    """Generate CSV-like tabular rows from a local safetensors language model.

    This wrapper is intentionally thin. It keeps model choice, prompt budget, and
    parsing policy visible to the benchmark instead of hiding them behind a
    broad synthetic-data abstraction.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        examples: int = 32,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        load_in_4bit: bool = False,
        max_attempts: int = 20,
        rows_per_prompt: Optional[int] = None,
        progress_path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__(device=device)
        self.model_path = model_path
        self.base_model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.examples = int(examples)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        self.load_in_4bit = bool(load_in_4bit)
        self.max_attempts = int(max_attempts)
        self.rows_per_prompt = None if rows_per_prompt is None else int(rows_per_prompt)
        self.progress_path = progress_path

    def fit(self, X: Any, column_names: Optional[list[str]] = None) -> "SafetensorsLLMInContextGenerator":
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("Expected a two-dimensional tabular array.")
        self.training_rows = values
        self.column_names = column_names or [f"x{j}" for j in range(values.shape[1])]
        self.params = {"n_examples": torch.tensor(values.shape[0])}
        return self

    def sample(self, n: int) -> np.ndarray:
        if not hasattr(self, "training_rows"):
            raise RuntimeError("Fit the generator before sampling.")
        try:
            from transformers import AutoTokenizer, StoppingCriteriaList
        except ImportError as exc:
            raise ImportError(
                "Install transformers and safetensors to use LLM row generation."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        model = self._load_model()

        rows: list[list[float]] = []
        attempts = 0
        while len(rows) < n and attempts < self.max_attempts:
            attempts += 1
            rows_requested = n - len(rows)
            if self.rows_per_prompt is not None:
                rows_requested = min(rows_requested, self.rows_per_prompt)
            prompt = self._prompt(rows_requested)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            start_length = inputs["input_ids"].shape[1]
            output = model.generate(
                **inputs,
                do_sample=self.do_sample,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [_StopOnStrings(tokenizer, ("<END>",), start_length)]
                ),
                **({"temperature": self.temperature} if self.do_sample else {}),
            )
            text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            rows.extend(self._parse_rows(text))
            self._write_progress(rows[:n], n)
            print(f"Parsed {min(len(rows), n)} / {n} rows after {attempts} attempts", flush=True)
        if len(rows) < n:
            raise RuntimeError(
                f"Parsed {len(rows)} valid rows after {attempts} LLM generation attempts; "
                "increase max_attempts or adjust the prompt/model."
            )
        return np.asarray(rows[:n], dtype=np.float64)

    def _write_progress(self, rows: list[list[float]], n: int) -> None:
        if self.progress_path is None:
            return
        with open(self.progress_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.column_names)
            writer.writerows(rows[:n])

    def _prompt(self, rows_requested: int) -> str:
        n_examples = min(self.examples, self.training_rows.shape[0])
        idx = np.random.choice(self.training_rows.shape[0], n_examples, replace=False)
        header = ",".join(self.column_names)
        examples = "\n".join(
            ",".join(f"{value:.6g}" for value in self.training_rows[i]) for i in idx
        )
        return (
            "You are generating realistic synthetic rows from an economic dataset.\n"
            "Return only CSV rows with the same columns and numeric formats.\n"
            f"Columns: {header}\n"
            "Examples:\n"
            f"{examples}\n"
            f"Generate exactly {rows_requested} new rows.\n"
            "After the final row, write <END> on its own line.\n"
        )

    def _parse_rows(self, text: str) -> list[list[float]]:
        text = text.split("<END>", 1)[0]
        rows: list[list[float]] = []
        width = len(self.column_names)
        for line in text.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != width:
                continue
            try:
                rows.append([float(part) for part in parts])
            except ValueError:
                continue
        return rows

    def _load_model(self) -> Any:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        load_kwargs = {
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs.pop("torch_dtype", None)
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        except ValueError:
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                **load_kwargs,
            )
        if self.device.type != "cuda":
            model.to(self.device)
        return model


class SafetensorsQLORAGenerator(SafetensorsLLMInContextGenerator):
    """QLoRA-fine-tuned LLM row generator.

    The benchmark calls ``fit_adapter`` explicitly because fine-tuning can be
    expensive and model-specific. After an adapter exists, sampling follows the
    same CSV parsing path as the in-context generator.
    """

    def fit_adapter(
        self,
        output_dir: str,
        num_train_epochs: float = 1.0,
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        rows_per_completion: int = 8,
        train_samples: Optional[int] = None,
        max_length: int = 1024,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        seed: int = 123,
    ) -> "SafetensorsQLORAGenerator":
        try:
            import transformers
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Install transformers, peft, accelerate, and bitsandbytes for QLoRA."
            ) from exc

        if not hasattr(self, "training_rows"):
            raise RuntimeError("Call fit before fit_adapter.")

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            device_map="auto",
        )
        model.config.use_cache = False
        _prepare_model_for_lora_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

        examples = self._adapter_training_examples(
            rows_per_completion=rows_per_completion,
            train_samples=train_samples,
            seed=seed,
        )
        dataset = _CompletionDataset(examples, tokenizer, max_length=max_length)
        args = transformers.TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            bf16=self.device.type == "cuda",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=[],
        )
        trainer = transformers.Trainer(model=model, args=args, train_dataset=dataset)
        trainer.train()
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        del trainer
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.adapter_path = output_dir
        self.tokenizer_path = output_dir
        return self

    def _adapter_training_examples(
        self,
        *,
        rows_per_completion: int,
        train_samples: Optional[int],
        seed: int,
    ) -> list[tuple[str, str]]:
        rng = np.random.default_rng(seed)
        n_rows = self.training_rows.shape[0]
        n_examples = min(self.examples, n_rows)
        block = max(1, min(int(rows_per_completion), n_rows))
        n_samples = int(train_samples) if train_samples is not None else max(128, n_rows)
        examples: list[tuple[str, str]] = []
        for _ in range(n_samples):
            prompt_idx = rng.choice(n_rows, n_examples, replace=False)
            completion_idx = rng.choice(n_rows, block, replace=block > n_rows)
            prompt = self._adapter_prompt(prompt_idx, block)
            completion = self._adapter_completion(completion_idx)
            examples.append((prompt, completion))
        return examples

    def _adapter_prompt(self, prompt_idx: np.ndarray, rows_requested: int) -> str:
        header = ",".join(self.column_names)
        examples = "\n".join(
            ",".join(f"{value:.6g}" for value in self.training_rows[i]) for i in prompt_idx
        )
        return (
            "You are generating realistic synthetic rows from an economic dataset.\n"
            "Return only CSV rows with the same columns and numeric formats.\n"
            f"Columns: {header}\n"
            "Examples:\n"
            f"{examples}\n"
            f"Generate exactly {rows_requested} new rows.\n"
            "After the final row, write <END> on its own line.\n"
        )

    def _adapter_completion(self, completion_idx: np.ndarray) -> str:
        rows = "\n".join(
            ",".join(f"{value:.6g}" for value in self.training_rows[i])
            for i in completion_idx
        )
        return f"{rows}\n<END>"

    def _row_training_text(self, row: np.ndarray) -> str:
        header = ",".join(self.column_names)
        values = ",".join(f"{value:.6g}" for value in row)
        return f"Columns: {header}\nRow: {values}"

    def _load_model(self) -> Any:
        if not hasattr(self, "adapter_path"):
            return super()._load_model()

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        load_kwargs = {
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            "device_map": "auto" if self.device.type == "cuda" else None,
        }
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            load_kwargs.pop("torch_dtype", None)
        try:
            base = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                **load_kwargs,
            )
        except ValueError:
            base = AutoModelForImageTextToText.from_pretrained(
                self.base_model_path,
                **load_kwargs,
            )
        model = PeftModel.from_pretrained(base, self.adapter_path)
        if self.device.type != "cuda":
            model.to(self.device)
        return model


class _StopOnStrings:
    def __init__(
        self,
        tokenizer: Any,
        stop_strings: tuple[str, ...],
        start_length: int,
    ) -> None:
        self.stop_token_ids = [
            tokenizer.encode(stop_string, add_special_tokens=False)
            for stop_string in stop_strings
        ]
        self.start_length = int(start_length)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: Optional[torch.FloatTensor],
        **kwargs: Any,
    ) -> bool:
        generated = input_ids[0, self.start_length :]
        if generated.numel() == 0:
            return False
        generated_ids = generated.tolist()
        for stop_ids in self.stop_token_ids:
            if len(generated_ids) >= len(stop_ids) and generated_ids[-len(stop_ids) :] == stop_ids:
                return True
        return False


class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int = 512) -> None:
        self.examples = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.examples["labels"] = self.examples["input_ids"].clone()

    def __len__(self) -> int:
        return self.examples["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {key: value[idx] for key, value in self.examples.items()}


class _CompletionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: list[tuple[str, str]],
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        for prompt, completion in examples:
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = tokenizer(
                prompt + completion,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            labels = full["input_ids"].clone()
            prompt_length = min(len(prompt_ids), labels.shape[1])
            labels[:, :prompt_length] = -100
            labels[full["attention_mask"] == 0] = -100
            self.items.append(
                {
                    "input_ids": full["input_ids"][0],
                    "attention_mask": full["attention_mask"][0],
                    "labels": labels[0],
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.items[idx]


def _prepare_model_for_lora_training(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module: Any, input: Any, output: torch.Tensor) -> None:
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()


def distribution_metrics(
    real: Any,
    fake: Any,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Compute marginal and joint distribution discrepancy metrics."""

    real_array = np.asarray(real, dtype=np.float64)
    fake_array = np.asarray(fake, dtype=np.float64)
    if real_array.ndim != 2 or fake_array.ndim != 2:
        raise ValueError("Distribution metrics expect two-dimensional arrays.")
    if real_array.shape[1] != fake_array.shape[1]:
        raise ValueError("Real and fake arrays must have the same number of columns.")

    try:
        from scipy.stats import ks_2samp, wasserstein_distance
    except ImportError as exc:
        raise ImportError("Install scipy to compute distribution metrics.") from exc

    marginal_w1 = [
        wasserstein_distance(real_array[:, j], fake_array[:, j])
        for j in range(real_array.shape[1])
    ]
    marginal_ks = [
        ks_2samp(real_array[:, j], fake_array[:, j]).statistic
        for j in range(real_array.shape[1])
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_real = np.corrcoef(real_array, rowvar=False)
        corr_fake = np.corrcoef(fake_array, rowvar=False)
    corr_real = np.nan_to_num(corr_real)
    corr_fake = np.nan_to_num(corr_fake)
    return {
        "marginal_w1_mean": float(np.mean(marginal_w1)),
        "marginal_w1_max": float(np.max(marginal_w1)),
        "marginal_ks_mean": float(np.mean(marginal_ks)),
        "marginal_ks_max": float(np.max(marginal_ks)),
        "mean_l2": float(np.linalg.norm(real_array.mean(axis=0) - fake_array.mean(axis=0))),
        "cov_frobenius": float(np.linalg.norm(np.cov(real_array, rowvar=False) - np.cov(fake_array, rowvar=False))),
        "corr_frobenius": float(np.linalg.norm(corr_real - corr_fake)),
        "sliced_wasserstein": sliced_wasserstein_distance(
            real_array,
            fake_array,
            n_projections=n_projections,
            seed=seed,
        ),
    }


def sliced_wasserstein_distance(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    dim = real.shape[1]
    directions = rng.normal(size=(n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    real_proj = np.sort(real @ directions.T, axis=0)
    fake_proj = np.sort(fake @ directions.T, axis=0)
    n = min(real_proj.shape[0], fake_proj.shape[0])
    return float(np.mean(np.abs(real_proj[:n] - fake_proj[:n])))
