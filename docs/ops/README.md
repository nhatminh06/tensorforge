# TensorForge Ops (Phase 2)

TensorForge Core (`src/tensorforge/`) is a frozen, deterministic
analytical modeling engine. TensorForge Ops (`src/tensorforge_ops/`) is a
separate package built *around* Core, never inside it.

```
TensorForge Core (src/tensorforge/)
        │
        │ ExperimentResult (deterministic, immutable)
        ▼
tensorforge_ops.tracking
        │
        ▼
MLflow Tracking Server
   ├── parameters   (filterable configuration)
   ├── metrics      (numeric analytical outputs)
   ├── tags         (categorical metadata, fingerprint, Git info)
   └── artifacts    (exact core-result.json + tracking-metadata.json)
```

The arrow never points back into Core: nothing under `src/tensorforge/`
imports `mlflow` or `tensorforge_ops`, and Core continues to run and pass
its full test suite with no MLflow installed at all.

## Scope so far

- **Milestone 11** added **reproducible MLflow experiment tracking** for
  already-deterministic Core results. It is not model training and not a
  regression gate or hardware right-sizing tool — see
  [mlflow-tracking.md](mlflow-tracking.md) and
  [docs/limitations.md](../limitations.md).
- **Milestone 12** added a **real PyTorch benchmark runner**
  (`tensorforge_ops.benchmark_pytorch`) that actually executes GEMM/
  Conv2D/Transformer-GEMM-only workloads and measures real latency,
  throughput, and (on CUDA) peak allocated memory, attached to the same
  MLflow run as the analytical result under explicit `measured_*` field
  names — see [benchmarking.md](benchmarking.md). This is not a
  prediction-accuracy comparison (that is a later milestone) and does not
  include an ONNX Runtime backend yet (DEFERRED, see that doc).

## Package layout

```
src/tensorforge_ops/
    __init__.py         public exports
    tracking.py          TrackingConfig, TrackedRun, compute_result_fingerprint(),
                          track_result(), track_experiment(), list_runs(),
                          log_benchmark_result()
    benchmark.py          BenchmarkConfig, BenchmarkResult, latency statistics
                          (no torch dependency)
    benchmark_pytorch.py  run_pytorch_benchmark() -- imports torch, imported
                          lazily by the CLI so torch stays optional
    cli.py               python -m tensorforge_ops track / list-runs / benchmark
    __main__.py
```

## Quick start

```bash
pip install -e ".[ops]"

# start a local MLflow server (localhost only, no auth added here)
mlflow server --host 127.0.0.1 --port 5000 &

export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python -m tensorforge_ops track \
    --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128 \
    --experiment tensorforge-local
```

Open `http://127.0.0.1:5000` in a browser to inspect the run.

See [mlflow-tracking.md](mlflow-tracking.md) for the full parameter/metric/
tag reference, fingerprint semantics, and comparison workflow.

## Real benchmark quick start

```bash
pip install -e ".[ops,benchmark]"

python -m tensorforge_ops benchmark \
    --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128 \
    --backend pytorch --device cpu --warmup 10 --iterations 50 \
    --experiment tensorforge-local
```

This runs the same Core experiment as `track` above, logs the analytical
result to MLflow, then actually executes the workload in PyTorch and logs
the measured latency/throughput/memory to the **same** run under
`measured_*` names. See [benchmarking.md](benchmarking.md) for the full
methodology.
