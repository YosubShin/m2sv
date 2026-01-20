#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG:-configs/molmo2_m2sv.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "Launching Molmo2 evaluation with config: ${CONFIG_PATH}"
uv run python evaluate_molmo2.py --config "${CONFIG_PATH}" --output-dir "${OUTPUT_DIR}" "$@"
