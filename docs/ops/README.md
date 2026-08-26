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
  prediction-accuracy comparison (that was this next milestone) and does
  not include an ONNX Runtime backend yet (DEFERRED, see that doc).
- **Milestone 13** added **physical-device calibration and
  prediction-vs-measurement validation**
  (`tensorforge_ops.calibration`/`calibration_pytorch`): a dedicated
  compute probe and memory-copy probe measure this device's real
  sustained compute rate and memory bandwidth (never a fake mapping of
  GPU cores to TensorForge PEs), an "empirical roofline prediction" is
  built from those rates plus each workload's existing Core FLOPs/
  baseline-DRAM-bytes properties, and that prediction is compared against
  Milestone 12's measured p50 latency — see
  [calibration.md](calibration.md) and [validation.md](validation.md).
  This still does not add regression gating, GPU telemetry, or hardware
  cost modeling (all later milestones).
- **Milestone 14** added a **PR performance regression guard**
  (`tensorforge_ops.regression`): a pure, torch-free comparison of two
  already-measured `BenchmarkResult`s against an explicit policy, gating
  on measured p50/p95 latency, throughput, and (CUDA) peak memory --
  never on Core's predicted/calibrated latency. A GitHub Actions workflow
  runs base and candidate benchmarks in the same job on GitHub-hosted CPU
  and posts a report — see [regression-guard.md](regression-guard.md).
  This does not add GPU telemetry, hardware cost modeling, or automated
  root-cause diagnosis (all later milestones).
- **Milestone 15** added **GPU telemetry correlation**
  (`tensorforge_ops.telemetry`/`telemetry_nvml`): NVML-based GPU
  utilization/memory-activity/VRAM/power/temperature/clock telemetry,
  collected in a separate phase from latency measurement (never
  perturbing Milestone-12 timing), summarized deterministically, and
  compared baseline-vs-candidate as cautious, evidence-worded diagnostic
  signals — never a gate condition, never merged into `BenchmarkResult`
  — see [gpu-telemetry.md](gpu-telemetry.md). DCGM Exporter support is
  DEFERRED (not implementable/verifiable on this project's development
  GPU+OS combination). This does not add automated root-cause diagnosis,
  Nsight/CUPTI integration, hardware cost modeling, or right-sizing (all
  later milestones).
- **Milestone 16** added a **hardware right-sizing + inference SLO/cost
  planner** (`tensorforge_ops.sizing`): pure arithmetic over already-
  measured `BenchmarkResult` evidence (steady-state throughput/p95 as
  explicit capacity/latency proxies, never claimed to be production
  QPS or end-to-end latency) determines the cheapest user-supplied
  deployment candidate satisfying an explicit SLO — replica count,
  headroom, and cost are transparent formulas, not a queueing-theory
  model, and TelemetrySummary/ValidationResult attach only as
  diagnostic context that never changes feasibility or ranking — see
  [right-sizing.md](right-sizing.md). No cloud prices are hard-coded or
  fetched; pricing is entirely user-supplied. This does not add a
  model-change impact report (a later milestone).

## Package layout

```
src/tensorforge_ops/
    __init__.py           public exports
    tracking.py            TrackingConfig, TrackedRun, compute_result_fingerprint(),
                            track_result(), track_experiment(), list_runs(),
                            log_benchmark_result(), log_calibration_profile(),
                            log_telemetry(), log_telemetry_correlation(),
                            log_validation_result(), log_validation_summary()
    benchmark.py            BenchmarkConfig, BenchmarkResult, latency statistics
                            (no torch dependency)
    benchmark_pytorch.py    run_pytorch_benchmark(), run_pytorch_telemetry_window() --
                            imports torch, imported lazily by the CLI
    calibration.py          CalibrationConfig, DeviceCalibrationProfile, predict(),
                            validate_prediction(), ValidationSummary (no torch dependency)
    calibration_pytorch.py  run_pytorch_calibration() -- imports torch, imported
                            lazily by the CLI
    regression.py           RegressionPolicy, compare_benchmark_results(),
                            compare_regression_suite(), render_markdown_report()
                            (no torch dependency, no MLflow dependency)
    telemetry.py             TelemetryConfig, TelemetryTrace, TelemetrySummary,
                            correlate_telemetry() (no torch/NVML/MLflow dependency)
    telemetry_nvml.py        NvmlProvider, run_telemetry_window(), probe_capabilities() --
                            imports pynvml (nvidia-ml-py), imported lazily by the CLI
    sizing.py                DeploymentCandidate, DeploymentCatalog, SloPolicy,
                            build_sizing_plan() (no torch/NVML/MLflow dependency)
    cli.py                 python -m tensorforge_ops track / list-runs / benchmark /
                            calibrate / validate / validate-suite / regression / telemetry /
                            right-size
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

## Calibration + validation quick start

```bash
python -m tensorforge_ops calibrate \
    --backend pytorch --device cuda --dtype fp16 \
    --compute-m 2048 --compute-n 2048 --compute-k 2048 --memory-probe-mib 64 \
    --output calibration.json

python -m tensorforge_ops validate \
    --calibration calibration.json \
    --workload-preset gemm_large_square --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128 \
    --backend pytorch --device cuda \
    --experiment tensorforge-local
```

`calibrate` measures this device's real sustained compute/memory rates
(never a fake TensorForge-PE mapping of a physical GPU). `validate` runs
the Core experiment, runs the real benchmark, builds an empirical
roofline prediction from the calibration profile, and compares it against
the measured p50 latency -- logging `core-result.json`,
`benchmark-result.json`, `calibration-profile.json`, and
`validation-result.json` to the same MLflow run. `validate-suite` runs
this across several workload presets and prints/saves a grouped error
summary. See [calibration.md](calibration.md) and
[validation.md](validation.md) for the full methodology, including why
this is not a hardware-accurate or pass/fail-graded prediction.

## PR performance regression guard (Milestone 14)

```bash
python -m tensorforge_ops regression \
    --baseline base-benchmark.json --candidate candidate-benchmark.json \
    --policy perf/cpu-ci-policy.json \
    --output-json regression-result.json --output-markdown regression-report.md
```

Compares two already-measured `BenchmarkResult`s (never TensorForge's
predicted/calibrated latency) against an explicit policy and exits `0`
(pass), `1` (operational/comparison error), or `2` (regression). The
`.github/workflows/performance-regression.yml` workflow runs this on
GitHub-hosted CPU for every PR touching `src/**`/`pyproject.toml`/
`perf/**`, benchmarking base and head commits in the same job so the
comparison is meaningful. See
[regression-guard.md](regression-guard.md) for the full policy/threshold
semantics, same-runner requirements, and CUDA self-hosted-runner
limitations.

## GPU telemetry (Milestone 15)

```bash
python -m tensorforge_ops telemetry probe --device-index 0

python -m tensorforge_ops benchmark \
    --workload-preset gemm_large_square --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128 \
    --backend pytorch --device cuda --telemetry nvml --telemetry-duration 5
```

Collects real NVML telemetry (GPU/memory activity, VRAM, power,
temperature, clocks) in a separate phase after latency measurement, so
Milestone 12's timing is untouched. `python -m tensorforge_ops
regression ... --baseline-telemetry-summary ... --candidate-telemetry-summary ...`
appends a diagnostic "GPU Telemetry Context" section to a regression
report -- evidence only, never part of the PASS/FAIL decision. See
[gpu-telemetry.md](gpu-telemetry.md) for metric semantics, sampling
methodology, and live RTX 3050 results.

## Hardware right-sizing / SLO-cost planning (Milestone 16)

```bash
python -m tensorforge_ops right-size \
    --catalog deployment-catalog.json --slo inference-slo.json \
    --output-json sizing-result.json --output-markdown sizing-report.md
```

Given a catalog of measured `BenchmarkResult` candidates and explicit
user-supplied hourly pricing, determines the cheapest candidate that
satisfies an explicit latency/throughput SLO -- replica count and cost
are transparent arithmetic over measured steady-state throughput (never
claimed to be production QPS), never a queueing model. No cloud price is
fetched or hard-coded. See [right-sizing.md](right-sizing.md) for the
full methodology and a live demo combining real RTX 3050 evidence with
labeled synthetic peers.

## Current end-to-end flow

```
Core analytical result
      |
   MLflow (Milestone 11)
      |
real benchmark (Milestone 12)
      |
device calibration (Milestone 13)
      |
prediction-vs-measurement validation (Milestone 13)
      |
PR performance regression guard (Milestone 14)
      |
GPU telemetry correlation (Milestone 15)
      |
hardware right-sizing / SLO-cost planning (Milestone 16)
```

A model-change impact report is a future milestone, not yet implemented.
