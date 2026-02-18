#!/usr/bin/env python3
"""Generate pdoc API docs with lightweight dependency stubs for CI."""

from __future__ import annotations

import importlib.util
import pkgutil
import sys
import types
from pathlib import Path

# Ensure local package import works in isolated/no-project runs.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_torch_stub() -> None:
    """Install a minimal torch/torchmin stub when torch is unavailable."""
    if importlib.util.find_spec("torch") is not None:
        return

    torch = types.ModuleType("torch")

    class Tensor:
        pass

    class Device:
        def __init__(self, spec: str | None = None):
            self.spec = spec or "cpu"

        def __str__(self) -> str:
            return self.spec

    class Generator:
        pass

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class _MPS:
        @staticmethod
        def is_available() -> bool:
            return False

    torch.Tensor = Tensor
    torch.device = Device
    torch.Generator = Generator
    torch.cuda = _Cuda()
    torch.backends = types.SimpleNamespace(mps=_MPS())
    torch.float32 = "float32"
    torch.float64 = "float64"
    torch.int64 = "int64"
    torch.int32 = "int32"
    torch.long = "int64"

    nn = types.ModuleType("torch.nn")

    class Module:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def parameters(self):
            return []

    class Parameter:
        def __init__(self, value=None):
            self.data = value

    nn.Module = Module
    nn.Parameter = Parameter
    nn.functional = types.SimpleNamespace(logsigmoid=lambda x: x)

    optim = types.ModuleType("torch.optim")

    class Optimizer:
        pass

    class LBFGS(Optimizer):
        pass

    optim.Optimizer = Optimizer
    optim.LBFGS = LBFGS

    torch.nn = nn
    torch.optim = optim

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.optim"] = optim

    if importlib.util.find_spec("torchmin") is None:
        sys.modules["torchmin"] = types.ModuleType("torchmin")


def _all_modules() -> list[str]:
    """Return full module tree under torchonometrics."""
    import torchonometrics

    modules = [torchonometrics.__name__]
    modules.extend(
        sorted(
            m.name
            for m in pkgutil.walk_packages(
                torchonometrics.__path__, torchonometrics.__name__ + "."
            )
        )
    )
    return modules


def main() -> None:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/api")
    _ensure_torch_stub()

    import pdoc
    import pdoc.render

    pdoc.render.configure(docformat="google", math=True)
    modules = _all_modules()
    pdoc.pdoc(*modules, output_directory=output_dir)
    print(f"API docs generated at {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
