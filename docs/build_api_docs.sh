#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-docs/api}"

cd "${ROOT_DIR}"

uv run python docs/generate_api_docs.py "${OUTPUT_DIR}"
