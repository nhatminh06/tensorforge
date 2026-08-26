# GPU telemetry reference

## Purpose and scope

TensorForge Ops can now observe the physical GPU while a workload
executes, and can compare that telemetry between a baseline and a
candidate benchmark run. The goal is **not** "automatically discover the
true root cause of a regression." The goal is **"attach physical GPU
evidence to observed performance behavior."** Every report this module
produces is worded as evidence/context, never as a proven cause -- see
"Diagnostic language" below.

```
real benchmark
    |
    +-- latency samples (Milestone 12, unchanged)
    +-- telemetry phase (this milestone, separate)
            |
            v
     TelemetryTrace -> TelemetrySummary
            |
            v
  correlated with BenchmarkResult / RegressionResult / ValidationResult
            |
            v
  TelemetryCorrelationResult -> evidence-backed signals
```

## Why NVML, not DCGM, is the primary backend

The development environment this project is built and verified against
is an **RTX 3050 Laptop GPU on CachyOS (Arch-family Linux)**. NVIDIA's
own DCGM documentation states advanced profiling metrics are limited to
supported datacenter GPUs (Volta and newer), and the officially
documented DCGM Linux distribution list does not include Arch/CachyOS.
So this project uses **NVML (`nvidia-ml-py`, NVIDIA's own maintained
Python bindings)** as the mandatory local/live telemetry backend --
NVML's basic monitoring API works broadly across Maxwell-and-newer NVIDIA
GPUs, including consumer/laptop parts. An optional DCGM Exporter backend
exists for operators who have that infrastructure (see below), but it is
never required, and this project never claims DCGM profiling metrics
(SM_ACTIVE, DRAM_ACTIVE, TENSOR_ACTIVE) are validated on this GPU --
because they are not: NVML's own GPM capability probe
(`nvmlGpmQueryDeviceSupport`) proves `isSupportedDevice = 0` on this
RTX 3050 (see "Live capability report" below). Unsupported means the
four advanced fields stay `None` on every sample, never `0`, never
fabricated.

## Latency-vs-telemetry separation

`BenchmarkResult` (Milestone 12) is a regression contract -- the guard in
`docs/ops/regression-guard.md` depends on its timing being exactly what
it was before this milestone. So telemetry is **never** sampled
concurrently with the timed latency loop; a background sampler
thread/NVML calls could perturb that timing. Instead:

- **Phase A** (unchanged): `run_pytorch_benchmark()` measures latency
  exactly as it did in Milestone 12 -- no telemetry involved.
- **Phase B** (new, opt-in via `--telemetry nvml`): the SAME workload
  construction (same preset, device, dtype -- literally the same
  `_build_gemm_callable`/`_build_conv_callable`/
  `_build_transformer_gemm_callable` helpers `run_pytorch_benchmark`
  uses) is executed repeatedly for `--telemetry-duration` seconds (default
  5s) while a background thread polls NVML every
  `--telemetry-sample-interval` seconds (default 0.15s).

A single TensorForge microbenchmark invocation can complete in
microseconds; NVML's own internal utilization sampling period is itself
on the order of hundreds of milliseconds to seconds. A multi-second
telemetry window with the workload looping continuously is what makes
NVML's utilization numbers mean anything for these workloads -- sampling
around one tiny operation would not.

## Telemetry sample fields and semantics

| Field | Meaning | Never call it... |
|---|---|---|
| `gpu_utilization_percent` | fraction of the recent sampling period during which *any* kernel was executing | "SM occupancy", "SM saturation", "compute utilization" |
| `memory_activity_percent` | fraction of the recent sampling period during which device memory was read/written | "memory bandwidth utilization" (NVML's basic API does not report bytes/sec vs. peak) |
| `memory_used_bytes`/`memory_free_bytes`/`memory_total_bytes` | **device-level**, may include other processes | "TensorForge process memory" (that remains `BenchmarkResult.peak_memory_allocated_bytes`, the PyTorch allocator's per-process peak) |
| `power_watts` | instantaneous device power draw | -- |
| `temperature_celsius` | GPU die temperature | -- |
| `sm_clock_mhz`/`memory_clock_mhz` | sampled physical clocks | not TensorForge Core's modeled `clock_hz` |
| `performance_state` | NVML P-state (0 = highest performance) | not interpreted as "P0 = perfect performance" -- contextual evidence only |
| `clock_event_reasons` | decoded NVML clocks-event-reason bit names | evidence a reason bit was set, not proof of its performance impact |
| `sm_activity_percent`/`sm_occupancy_percent`/`tensor_activity_percent`/`dram_bandwidth_utilization_percent` | NVML GPM advanced metrics | only reported when `nvmlGpmQueryDeviceSupport` proves the device supports them -- **`None` on this RTX 3050** |

Every field is `None`, never `0`, when NVML could not read it for that
sample. Only a failure to initialize NVML itself, or to get a device
handle, is a hard error (`NvmlUnavailableError`) -- a single unsupported
metric never fails the whole trace.

## Sampling window and overhead

Defaults: `sample_interval_seconds=0.15`, `telemetry_duration_seconds=5.0`
(configurable per the `TelemetryConfig` validation: both must be `> 0`).
Telemetry collection has real overhead (a background thread, repeated
Python/NVML calls) that is **not claimed to be zero** -- this is why
telemetry latency numbers are never used as a substitute for Phase A's
measurement. Telemetry's purpose is diagnosis, not exact timing.

## TelemetryTrace / TelemetrySummary

`TelemetryTrace` (`telemetry_schema_version = 1`, distinct from Core's,
`BenchmarkResult`'s, calibration's, validation's, and regression's schema
versions) carries backend/device identity, sample interval, requested
vs. actual duration, workload identity (preset/kind/backend/dtype/Core
fingerprint), and the raw sample list.

`summarize_telemetry_trace()` is a pure, deterministic function producing
`TelemetrySummary`: mean/min/max (and a sample count) for every numeric
metric, computed **only over the samples where that metric was actually
present** -- a metric absent from every sample gets an all-`None`
`MetricStat` (never `0`) and is listed in `unavailable_metrics` rather
than `available_metrics`. `throttle_reasons_observed` is the union of
decoded reason names across all samples in the trace.

## Baseline-vs-candidate correlation

`correlate_telemetry(baseline_summary, candidate_summary, regression_status=...)`
is pure and never re-collects telemetry. It **never** reads or modifies a
`RegressionResult.status` -- `regression_status` is a plain string carried
through purely as report context. See
[regression-guard.md](regression-guard.md#telemetry-context) for how the
CLI wires this into a regression report.

Deltas use the semantically correct unit for each metric:

- Utilization/activity metrics: **percentage points** (`70% -> 80%` is
  reported as `+10 pp`, never `+14.3%`).
- Temperature/power/clocks/memory: **absolute units** (°C, W, MHz,
  bytes) -- a relative percentage on a temperature reading is not a
  clarifying number.

### Diagnostic signals and language

Signals are only emitted when a delta crosses an explicit, named
threshold (`_MATERIAL_UTILIZATION_DELTA_PP = 10`,
`_MATERIAL_CLOCK_RELATIVE_DELTA = 5%`,
`_MATERIAL_TEMPERATURE_DELTA_C = 5`,
`_MATERIAL_VRAM_FRACTION_DELTA_PP = 10` -- these gate report *wording*
only, never PASS/FAIL):

- **lower_gpu_activity**: candidate mean GPU activity dropped materially.
- **higher_memory_activity**: candidate mean memory activity rose materially.
- **clock_reduction**: candidate mean SM clock dropped materially.
- **thermal_or_power_limit_observed**: emitted **only** when an actual
  NVML clock-event reason (`HwPowerBrakeSlowdown`, `HwThermalSlowdown`,
  `SwPowerCap`, `SwThermalSlowdown`) was decoded from a sample -- never
  inferred merely because clocks changed.
- **higher_vram_pressure**: candidate peak VRAM usage, as a fraction of
  total device memory, rose materially.

Every signal is worded with "evidence suggests", "consistent with",
"observed alongside" -- never "proves", "root cause", "definitely caused
by", "is memory-bound", or "is compute-bound". Basic NVML utilization/
activity telemetry cannot establish those stronger claims on its own; a
future milestone's deeper profiling (Nsight, used manually, not
integrated here) would be needed for that.

A separate, single-trace `low_utilization_note()` flags a workload whose
mean GPU activity is below 20% -- worded as "consistent with
underutilization (small kernels, launch overhead, or serialization) --
basic telemetry cannot distinguish which," never a specific cause.

## Live capability report: RTX 3050 Laptop GPU

`python -m tensorforge_ops telemetry probe --device-index 0` (real
output from this development machine, nvidia-ml-py 13.610.43, driver
610.43.03):

```
Available:
  gpu_utilization_percent
  memory_activity_percent
  memory_used_bytes
  power_watts
  temperature_celsius
  sm_clock_mhz
  memory_clock_mhz
  performance_state
  clock_event_reasons

Unavailable:
  sm_activity_percent
  sm_occupancy_percent
  tensor_activity_percent
  dram_bandwidth_utilization_percent
```

`power.limit` is also unsupported via NVML on this device
(`nvmlDeviceGetPowerManagementLimit` raises `NotSupported`) --
consistent with `nvidia-smi`'s own `Pwr:Usage/Cap` column showing `N/A`
for the cap on this laptop GPU.

## Live workload telemetry (RTX 3050, `--telemetry-duration 4`)

| Workload | mean GPU activity | max GPU activity | throttle reasons |
|---|---|---|---|
| `gemm_tiny` | 12% | 17% | none |
| `gemm_large_square` | 82% | 100% | SwPowerCap |
| `conv_pointwise` | 25% | 36% | none |
| `conv_spatial` | 85% | 100% | GpuIdle, SwPowerCap |
| `transformer_small` (GEMM-only) | 20% | 26% | none |

This is real, evidence-backed corroboration of the Milestone-13
validation hypothesis: `gemm_tiny` showed ~99% APE (predicted vs.
measured) there, and here shows only 12% mean GPU activity when the
identical workload is looped continuously for 4 seconds -- consistent
with the earlier "unmodeled dispatch/launch overhead" explanation, not
proof of it. `gemm_large_square` (25% APE in Milestone 13) shows 82%
mean activity here, a large, consistent gap in the same direction.
`transformer_small`'s GEMM-only benchmark -- which deliberately executes
attention heads in an explicit per-head Python loop, multiplying the
number of small, separately-dispatched kernel launches -- shows only 20%
mean activity, consistent with (not proof of) the "many small serialized
launches" explanation for its ~91% APE in Milestone 13.

`SwPowerCap` (software power-cap throttling) was observed repeatedly
during sustained GEMM/Conv telemetry windows on this laptop GPU under
its default power limit -- this is an actual decoded NVML clocks-event
reason, not an inference from clock deltas alone.

## MLflow artifacts and metrics

`log_telemetry(tracking, run_id, trace, summary)` attaches, to the
**same** run already holding `core-result.json`/`benchmark-result.json`:

- artifacts: `telemetry-trace.json`, `telemetry-summary.json`
- metrics (only for available metrics -- never a `0` standing in for
  `None`): `telemetry_gpu_util_percent_{mean,max}`,
  `telemetry_memory_activity_percent_{mean,max}`,
  `telemetry_vram_used_max_bytes`, `telemetry_power_watts_{mean,max}`,
  `telemetry_temperature_max_celsius`,
  `telemetry_sm_clock_mhz_{mean,min,max}`,
  `telemetry_memory_clock_mhz_{mean,min,max}`, and the four advanced
  metrics' mean/max only when actually available
- tags: `tensorforge.telemetry`, `tensorforge.telemetry_schema_version`,
  `tensorforge.telemetry_device`, `tensorforge.telemetry_advanced_metrics`
  (`available`/`unavailable`)

`log_telemetry_correlation(tracking, run_id, correlation)` attaches
`telemetry-correlation.json` plus a `tensorforge.telemetry_correlation =
"diagnostic-context"` tag -- it never touches any Milestone-14 regression
metric or tag.

## CLI

```bash
python -m tensorforge_ops telemetry probe --device-index 0

python -m tensorforge_ops benchmark \
    --workload-preset gemm_large_square --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128 \
    --backend pytorch --device cuda --warmup 10 --iterations 50 \
    --telemetry nvml --telemetry-duration 5 --telemetry-sample-interval 0.15 \
    --telemetry-output-trace telemetry-trace.json --telemetry-output-summary telemetry-summary.json
```

`--telemetry` requires `--device cuda[:N]` (NVML observes physical
devices; there is nothing to observe for `--device cpu`, so that
combination fails clearly rather than silently skipping telemetry).
Output files use the same overwrite-protection convention as the rest of
this CLI (`--force` required to replace an existing file).

## Optional DCGM Exporter backend

Not implemented in this milestone. Rationale: this project's development
GPU is a consumer laptop GPU on Arch-family Linux, a combination NVIDIA's
own DCGM distribution/hardware support documentation does not cover for
profiling metrics -- attempting to install/run DCGM here would mean
claiming support this project cannot actually verify. **DCGM Exporter
status: DEFERRED.** A future contributor with access to a supported
datacenter GPU and Linux distribution could add a
`telemetry_dcgm.py` backend that scrapes a `dcgm-exporter` HTTP endpoint
(e.g. `http://localhost:9400/metrics`) for the same coarse metrics plus,
where the endpoint actually exposes them, the advanced profiling fields
-- following the same "runtime-proven, `None` if absent" discipline this
module already uses for NVML/GPM. This repository does not install,
configure, or auto-start DCGM/DCGM Exporter, Prometheus, or Grafana.

## Limitations

- Telemetry overhead is real and unmeasured precisely -- it is a
  diagnostic window, not a timing-accurate one.
- NVML's coarse utilization/activity percentages cannot, by themselves,
  prove a specific bottleneck (compute-bound vs. memory-bound); a future
  milestone's deeper profiling would be needed for that.
- Advanced GPM metrics (SM/tensor/DRAM activity) are proven unavailable
  on this project's development GPU; the corresponding code paths are
  therefore unverified on real hardware and remain `None` everywhere.
- DCGM Exporter support is deferred entirely (see above).
- Telemetry never modifies power limits, clocks, fan curves, or
  performance states -- it is observation-only.
- No automated root-cause diagnosis, no Nsight/CUPTI integration, no GPU
  kernel-level attribution.
