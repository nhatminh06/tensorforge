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

## Milestone 11 scope

This milestone adds **reproducible MLflow experiment tracking** for
already-deterministic Core results. It is not model training, not real
GPU benchmarking, not a regression gate, and not hardware right-sizing —
see [mlflow-tracking.md](mlflow-tracking.md) for the full detail and
[docs/limitations.md](../limitations.md) for what is not modeled.

## Package layout

```
src/tensorforge_ops/
    __init__.py     public exports
    tracking.py      TrackingConfig, TrackedRun, compute_result_fingerprint(),
                      track_result(), track_experiment(), list_runs()
    cli.py           python -m tensorforge_ops track / list-runs
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
