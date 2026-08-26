# Real benchmark runner reference

## Purpose

TensorForge Core's `ExperimentResult` is an **analytical prediction**:
`perfect_overlap_time_seconds`, `compute_time_seconds`,
`dram_time_seconds`, etc. are all derived from closed-form roofline/
tiling/timing formulas, never measured on real hardware.

`tensorforge_ops.benchmark` and `tensorforge_ops.benchmark_pytorch` add a
second, independent data source: **measured** wall-clock latency from a
real framework (PyTorch) actually executing the same workload preset.
Nothing here adjusts, corrects, or "calibrates" the analytical result —
the two live side by side, under distinct field-name prefixes, so they
can be compared (a future milestone) without ever being confused.

```
predicted_*  ->  from tensorforge.experiments.run_experiment() (Core, analytical)
measured_*   ->  from tensorforge_ops.benchmark_pytorch.run_pytorch_benchmark() (Ops, real execution)
```

Core's own metric names (`perfect_overlap_time_seconds`, ...) are never
overwritten by a benchmark run; the measured statistics use their own
`measured_*` names in both the JSON schema and the MLflow metrics logged
by `log_benchmark_result()`.

## Core / Ops boundary

`src/tensorforge/` still imports nothing beyond the standard library.
`grep -RIn "import torch\|onnxruntime\|mlflow" src/tensorforge/` is
empty — verified by `tests/test_core_boundary.py` (see below). PyTorch is
only imported inside `tensorforge_ops/benchmark_pytorch.py`, and only
when that module is actually imported (the CLI imports it lazily, inside
the `benchmark` command handler), so `pip install -e .` (Core alone) and
`pip install -e ".[ops]"` (Core + MLflow tracking, no benchmarking)
continue to work with zero PyTorch installed.

## Supported workloads

| Core preset kind | PyTorch operation                                   |
|-------------------|-----------------------------------------------------|
| `gemm`            | `torch.matmul(a, b)`                                |
| `conv2d`           | `torch.nn.functional.conv2d(x, weight, bias=None, ...)` |
| `transformer`      | GEMM-only decomposition, see below                  |
| `cnn`              | not supported by this backend (raises `ValueError`) |

### Transformer: GEMM-only, not "Transformer inference"

Core's Transformer model (Milestone 8) only counts the GEMM operations of
a Transformer block: Q/K/V projections, the per-head attention-score
matmul (`QK^T`), the per-head attention-value matmul, the output
projection, and the MLP up/down projections. It does **not** model
softmax, layer normalization, residual connections, or dropout, and
neither does this benchmark.

The per-head matmuls are executed in an explicit Python loop over
`(batch, head)` pairs — not a single batched/vectorized matmul — because
Core's cost model treats attention as `batch * num_heads` sequential
repetitions of a per-head GEMM pair. Benchmarking a single fused batched
matmul would measure a different computation than the one Core's formula
counts. This makes the PyTorch benchmark slower than an optimized
attention kernel would be; that is intentional, not a bug.

This is why the workload is always called `transformer_gemm_only` (or
"the Transformer GEMM decomposition") in code, output, and this
document — never "Transformer inference" or "running a Transformer."

## Device semantics

`--device cpu` or `--device cuda[:N]`. If `cuda` is requested and
`torch.cuda.is_available()` is `False`, the benchmark fails immediately
with a clear `ValueError` — it never silently falls back to CPU.

TensorForge's generic accelerator presets (`small`, `balanced`, ...) used
by Core are **not** a description of the physical machine the benchmark
runs on. A benchmark result's `device` and `runtime_metadata` (torch
version, CUDA device name, CUDA runtime version) describe the real
machine; the `workload_preset`/`core_result_fingerprint` fields describe
which Core analytical run it is being compared against. These are two
independent axes and must not be conflated.

## Warmup and timing methodology

- `warmup_iterations` (default 10) run first and are excluded from every
  statistic. Warmup lets PyTorch/cuDNN kernel selection, CUDA context/
  memory-pool setup, and Python-level lazy work settle before measurement.
- Inputs and weights are allocated on the target device **before** timing
  starts (outside both warmup and measured loops), so measured latency
  never includes tensor construction or host→device upload.
- The whole timed region runs under `torch.inference_mode()`: no autograd
  graph, no gradients, no training-mode layer behavior.
- CPU timing uses `time.perf_counter()` around each iteration.
- CUDA timing additionally calls `torch.cuda.synchronize(device)` both
  immediately before starting the clock and immediately before stopping
  it (for every warmup and measured iteration), so the measured interval
  reflects actual device completion, not just kernel-launch/enqueue time.
  This is the one CUDA timing methodology used throughout — CUDA events
  are not used, so all CUDA numbers in this project are directly
  comparable to each other.

## Latency statistics

`compute_latency_statistics()` (in `tensorforge_ops/benchmark.py`, no
torch dependency) computes, from the raw per-iteration sample list:

- `count`, `mean_seconds` (`statistics.fmean`)
- `p50_seconds` (linear-interpolation percentile, always available)
- `p95_seconds` — only computed once `count >= 20`, else `None`
- `p99_seconds` — only computed once `count >= 100`, else `None`
- `min_seconds`, `max_seconds`
- `throughput_per_second = 1 / mean_seconds`

The percentile method is the standard "linear interpolation between
closest ranks" (the same convention as NumPy's default `percentile`):
for a sorted array of length `n`, `p95` interpolates between the floor
and ceiling of `(n-1) * 0.95`. It is verified against hand-computable
known arrays in `tests/test_benchmark.py`
(`test_statistics_p95_known_value_small_array`, etc.).

A `None` p95/p99 is never fabricated or extrapolated from fewer samples
— run more iterations (`--iterations`) if you need it.

## Memory measurement

- **CUDA**: `torch.cuda.reset_peak_memory_stats(device)` is called right
  before the timed region, and `torch.cuda.max_memory_allocated(device)`
  is read right after, giving peak allocator bytes across the whole
  warmup+measured run (allocator bytes, not raw device memory — includes
  PyTorch's caching allocator overhead).
- **CPU**: peak host memory is explicitly **not measured** this
  milestone (`peak_memory_allocated_bytes` is `None` for `device=cpu`).
  This is a stated gap, not a silent omission.

## Benchmark result schema

`BenchmarkResult.to_dict()` / `.to_json()` produce a `benchmark_schema_version`-
tagged structure, deliberately **separate** from Core's
`schema_version` — Core's JSON is byte-for-byte deterministic given the
same inputs; a benchmark's JSON is not (it embeds real, non-deterministic
timing samples) and must never be compared for exact equality across runs.

```json
{
  "benchmark_schema_version": 1,
  "core_result_fingerprint": "...",
  "workload_preset": "gemm_tiny",
  "workload_kind": "gemm",
  "backend": "pytorch",
  "device": "cpu",
  "dtype": "fp16",
  "warmup_iterations": 10,
  "measured_iterations": 50,
  "latency_samples_seconds": [...],
  "statistics": { "count": 50, "mean_seconds": ..., "p50_seconds": ..., ... },
  "peak_memory_allocated_bytes": null,
  "runtime_metadata": { "torch_version": "...", "device_type": "cpu" }
}
```

## Linking a benchmark to its Core result

Every `BenchmarkResult` carries `core_result_fingerprint`, the same
SHA-256 fingerprint (`compute_result_fingerprint()`, Milestone 11)
computed from the Core `ExperimentResult` it was benchmarked against.
`python -m tensorforge_ops benchmark` computes the fingerprint once, from
the Core run it just executed, and stamps it onto the `BenchmarkResult`
it produces in the same invocation.

## MLflow integration

`log_benchmark_result(tracking, run_id, benchmark)` attaches measured
data to an **already-tracked** MLflow run (the same run the analytical
result was logged to), using `MlflowClient` directly rather than
`mlflow.start_run()` — this lets it append params/metrics/tags/an
artifact to a run that `track_result()` already closed, instead of
opening a second, separate run for the same experiment.

- params: `benchmark_backend`, `benchmark_device`, `benchmark_dtype`,
  `warmup_iterations`, `measured_iterations`, `runtime_<key>` (one per
  `runtime_metadata` entry, e.g. `runtime_device_name`)
- metrics: `measured_mean_latency_seconds`, `measured_p50_latency_seconds`,
  `measured_p95_latency_seconds` (only if computed),
  `measured_p99_latency_seconds` (only if computed),
  `measured_min_latency_seconds`, `measured_max_latency_seconds`,
  `measured_throughput_per_second`,
  `measured_peak_memory_allocated_bytes` (only if not `None`)
- tags: `tensorforge.measurement=real`,
  `tensorforge.core_result_fingerprint`,
  `tensorforge.benchmark_schema_version`, `benchmark.backend`,
  `benchmark.device_type`
- artifact: `benchmark-result.json` (the full `BenchmarkResult.to_json()`)

`python -m tensorforge_ops benchmark ... --no-track` skips MLflow
entirely and prints the summary only.

## CLI

```bash
python -m tensorforge_ops benchmark \
    --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128 \
    --backend pytorch --device cpu --warmup 10 --iterations 50 \
    --experiment tensorforge-local
```

`--device cuda` runs the same benchmark on the first CUDA device.
`--backend onnxruntime` is currently **DEFERRED** — see below.

## ONNX Runtime status: DEFERRED

This milestone implements the PyTorch backend only.
`--backend onnxruntime` is accepted by the CLI's `--backend` choices (so
`--help` documents the intended future surface) but fails clearly with a
"DEFERRED" message rather than attempting a partial/broken
implementation. The `onnx` extra in `pyproject.toml` is left declared for
a future milestone to build against; no ONNX code exists yet.

## What this milestone does not do

- No prediction-vs-measured error/accuracy computation (Milestone 13).
- No real-hardware accelerator presets (e.g. "RTX 4090") — Core's
  `small`/`balanced`/... presets remain generic and are not tied to the
  physical benchmarking machine.
- No CPU peak memory measurement.
- No ONNX Runtime backend (see above).
- No int8/quantized dtypes (PyTorch backend supports fp32/fp16 only,
  matching Core's `DType`).
