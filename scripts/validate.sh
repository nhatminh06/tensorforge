#!/usr/bin/env bash
# Run the full deterministic validation suite: pytest, then a few
# high-level CLI smoke checks that exercise every major mode.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== pytest ==="
pytest -q

echo
echo "=== CLI: --help ==="
python -m tensorforge --help > /dev/null

echo
echo "=== CLI: --list-presets ==="
python -m tensorforge --list-presets > /dev/null

echo
echo "=== CLI: one preset experiment ==="
python -m tensorforge --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64 --tile-n-values 32,64 --tile-k-values 32,64 > /dev/null

echo
echo "All validation checks passed."
