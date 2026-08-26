#!/usr/bin/env bash
# Reproducible TensorForge Core demo. Runs a handful of small/medium
# examples using only the project CLI. Does not modify repository state,
# does not require network access, and finishes in a few seconds.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== 1. GEMM (single-configuration mode) ==="
python -m tensorforge --m 128 --n 128 --k 128 --dtype fp16 \
    --peak-tflops 10 --bandwidth-gbps 200 --pe-rows 16 --pe-cols 16

echo
echo "=== 2. Transformer-like block ==="
python -m tensorforge --transformer-block \
    --batch-size 1 --seq-len 128 --d-model 512 --num-heads 8 --d-ff 2048 \
    --dtype fp16 --peak-tflops 10 --bandwidth-gbps 200 --sram-kib 256 --clock-ghz 1 \
    --pe-rows 16 --pe-cols 16 \
    --tile-m-values 32,64 --tile-n-values 32,64 --tile-k-values 32,64

echo
echo "=== 3. Conv2D (materialized-im2col lowering) ==="
python -m tensorforge --conv2d \
    --batch-size 1 --in-channels 64 --input-height 56 --input-width 56 \
    --out-channels 128 --kernel-h 3 --kernel-w 3 --padding-h 1 --padding-w 1 \
    --dtype fp16 --peak-tflops 10 --bandwidth-gbps 200 --sram-kib 256 --clock-ghz 1 \
    --pe-rows 16 --pe-cols 16 \
    --tile-m-values 32,64 --tile-n-values 32,64 --tile-k-values 32,64

echo
echo "=== 4. Preset listing ==="
python -m tensorforge --list-presets

echo
echo "=== 5. Reproducible preset experiment (deterministic JSON to stdout summary) ==="
python -m tensorforge --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128

echo
echo "Demo complete."
