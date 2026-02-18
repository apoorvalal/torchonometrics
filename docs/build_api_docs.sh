#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-docs/api}"

cd "${ROOT_DIR}"

# Discover all package modules so pdoc renders choice/gmm subpackages too.
mapfile -t MODULES < <(
  uv run python - <<'PY'
import pkgutil
import torchonometrics

print(torchonometrics.__name__)
for mod in sorted(
    m.name for m in pkgutil.walk_packages(
        torchonometrics.__path__,
        torchonometrics.__name__ + ".",
    )
):
    print(mod)
PY
)

uv run pdoc "${MODULES[@]}" \
  --docformat google \
  --math \
  --output-directory "${OUTPUT_DIR}"

echo "API docs generated at ${OUTPUT_DIR}/index.html"
