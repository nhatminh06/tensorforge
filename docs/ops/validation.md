# Prediction-vs-measurement validation reference

## What is being compared

```
DeviceCalibrationProfile (empirical compute/memory ceilings)
    + TensorForge workload's modeled FLOPs / baseline DRAM bytes
CalibratedPrediction ("empirical roofline prediction")
    vs.
BenchmarkResult.statistics.p50_seconds (Milestone 12, real PyTorch execution)
ValidationResult (signed/absolute/relative error, APE, ratio)
```

This is **not** a hardware-accurate, cycle-accurate, or simulated
prediction, and it is **not** the same thing as Core's tile/schedule
analytical result (`perfect_overlap_time_seconds`). It is a simple
empirical roofline lower bound built from two real, measured device
ceilings -- see [calibration.md](calibration.md) for how those ceilings
are measured.

## Prediction formula

```
predicted_compute_seconds = modeled_flops / effective_compute_flops_per_second
predicted_memory_seconds  = modeled_baseline_dram_bytes / effective_memory_bandwidth_bytes_per_second
predicted_latency_seconds = max(predicted_compute_seconds, predicted_memory_seconds)
```

No additive overhead term, no fitted correction coefficient, no
workload-specific fudge factor. `modeled_flops` / `modeled_baseline_dram_bytes`
come directly from existing Core objects (`Gemm.flops`/`Gemm.dram_bytes`,
the materialized-im2col `lower_conv2d_to_gemm(...).gemm`, and the
Transformer block's per-op `Gemm.dram_bytes * repetitions` sum) --
**not** `ExperimentResult.primary_metrics["total_dram_bytes"]`, which
reflects one specific Core tile/schedule mapping that cuBLAS/cuDNN/
PyTorch never actually use.

`predicted_bottleneck` is "compute-bound" / "memory-bound" / "balanced"
(equal within a documented `1e-9` relative tolerance) based on which
term above is larger. This is an **analytical** classification of the
prediction formula, not a measured claim -- see below.

## Why "predicted bottleneck," never "measured bottleneck"

A single wall-clock latency number cannot prove whether the real,
measured GPU execution was actually compute-bound or memory-bound --
that requires device telemetry (SM occupancy, memory controller
utilization, etc.), which this milestone does not have. Every place this
codebase reports a bottleneck derived from `predict()`/`ValidationResult`
is prefixed "predicted"; nothing here ever asserts a measured bottleneck.
(Milestone 15's planned GPU telemetry integration is the earliest point
that claim could become supportable.)

## Validation target: measured p50

The comparison target is `BenchmarkResult.statistics.p50_seconds` --
steady-state central-tendency latency, matching what a `max(compute,
memory)` analytical lower bound is trying to predict. p95 (tail latency)
remains operationally important but is not the quantity this formula
predicts, so it is not used as the primary error target (it is still
reported alongside p50/mean in every `ValidationResult` for reference).

## Error formulas

```
signed_error_seconds        = predicted - measured        (Core convention: negative = underestimate)
absolute_error_seconds      = abs(signed_error_seconds)
relative_error               = signed_error_seconds / measured
absolute_percentage_error    = abs(relative_error) * 100
measured_to_predicted_ratio  = measured / predicted        (1.0 = exact; >1 = measured slower than predicted)
```

## Device/runtime matching is mandatory

`validate_prediction()` raises `ValueError` before computing any error if
the calibration profile's backend/device/dtype/PyTorch-version/(CUDA
device identity) does not match the benchmark result's. See
[calibration.md](calibration.md#deviceruntime-matching-why-validation-can-refuse-to-run).

## Holdout principle

Calibration profiles are built **only** from the compute/memory probes.
The workloads validated below (`gemm_tiny`, `gemm_large_square`,
`conv_pointwise`, `conv_spatial`, `transformer_small`) were never used to
derive `effective_compute_flops_per_second` /
`effective_memory_bandwidth_bytes_per_second` -- they are a genuine
holdout set, and no coefficient here was tuned to make any of their
errors look smaller.

## No pass/fail threshold

This milestone measures and reports model error. It does not decide
"APE < X% => pass." A large error is a legitimate, useful validation
result, not a bug to be hidden or a reason to change Core's formulas.
(A regression-gate policy, if ever added, is a distinct future
milestone.)

## Live results: RTX 3050 Laptop GPU, FP16

Device: NVIDIA GeForce RTX 3050 Laptop GPU (laptop, 4 GB VRAM) &nbsp;|&nbsp;
PyTorch 2.13.0+cu126 &nbsp;|&nbsp; CUDA runtime 12.6 &nbsp;|&nbsp; dtype fp16

Calibration (compute probe 2048x2048x2048, memory probe 64 MiB,
warmup 10 / iterations 50):

| | |
|---|---|
| effective compute rate | 11.597 TFLOP/s |
| effective memory bandwidth | 177.936 GB/s |

These are this specific run's measured empirical ceilings -- not a
marketing spec, and not guaranteed to reproduce exactly on a rerun (see
[calibration.md](calibration.md)). One physical laptop's numbers here do
not generalize to "RTX 3050 Laptop GPUs" as a class -- see
[calibration.md](calibration.md#limitations).

Holdout validation (`accelerator-preset balanced`, `warmup 10` /
`iterations 50`, `--device cuda`):

| Workload | Predicted | Measured p50 | APE | Predicted bound |
|---|---|---|---|---|
| gemm_tiny | 552 ns | 48.18 us | 98.9% | memory-bound |
| gemm_large_square | 185.18 us | 245.97 us | 24.7% | compute-bound |
| conv_pointwise | 6.86 us | 54.19 us | 87.3% | memory-bound |
| conv_spatial | 39.87 us | 115.66 us | 65.5% | compute-bound |
| transformer_small | 72.33 us | 835.58 us | 91.3% | compute-bound |

Overall: median APE 87.3%, mean APE 73.6%, max APE 98.9%.

By workload kind (median APE): GEMM 61.8% (n=2), Conv2D 76.4% (n=2),
Transformer GEMM-only 91.3% (n=1).

By predicted bound (median APE): compute-bound 65.5% (n=3),
memory-bound 93.1% (n=2).

### Small vs. large GEMM

`gemm_tiny` (128x128x128) has a ~99% APE; `gemm_large_square`
(1024x1024x1024) has a ~25% APE -- more than a 4x reduction. **Possible
explanation:** the empirical roofline formula has no term for kernel
dispatch/launch or PyTorch-level framework overhead, and that fixed
per-call overhead is a much larger fraction of `gemm_tiny`'s tiny
predicted runtime than of `gemm_large_square`'s. This matches the
Milestone-12 observation that `gemm_tiny`'s ~2us analytical Core
prediction vs. ~56us measured latency was mostly unmodeled overhead, not
a hardware-calibration error.

### Conv2D: materialized-im2col mismatch

Both Conv2D presets show large error (87.3% / 65.5% APE). **Possible
explanation:** TensorForge Core models Conv2D as a materialized-im2col
GEMM (explicit im2col tensor read/write baked into the baseline byte
count), while PyTorch/cuDNN's actual convolution kernel very likely does
not materialize im2col at all for these shapes -- it uses a fused
convolution algorithm with a different, and probably smaller, real
memory-traffic pattern. This is model-semantics evidence, not merely a
hardware-calibration gap, and Core's Conv2D model is intentionally left
unchanged in this milestone (see [calibration.md](calibration.md) and
`CLAUDE.md`'s scope-discipline rule).

### Transformer GEMM-only

`transformer_small`'s predicted latency (72.33us, the sum of eight small
sequential GEMM groups' compute/memory times) is roughly 11.5x smaller
than the measured p50 (835.58us) -- the largest gap of any workload here.
**Possible explanation:** the benchmark deliberately executes attention
heads in an explicit sequential Python loop to match Core's B*H-repetition
model (see `benchmark_pytorch.py`), which multiplies the number of small,
separately-dispatched CUDA kernel launches far beyond a single GEMM --
so unmodeled per-launch overhead very plausibly compounds across many
more kernel invocations than either GEMM preset incurs. This result
characterizes the GEMM-only decomposition benchmark, not a validation of
full Transformer inference (softmax/normalization/residuals are excluded
from both the Core model and this benchmark).

### Predicted-bottleneck grouping

Both predicted-memory-bound workloads (`gemm_tiny`, `conv_pointwise`)
show APE well above 85%; predicted-compute-bound workloads span 24.7%-
91.3%. With only n=2 and n=3 per group this is far too small a sample to
generalize, but it is consistent with the small-workload/launch-overhead
explanation above: both memory-bound-predicted workloads here also
happen to be small, low-computation presets.

## Limitations

- Five workload presets, one physical device, one calibration run: not a
  statistically powered study. Every "possible explanation" above is
  exactly that -- plausible, not proven by this data alone.
- No measured-bottleneck claim is made anywhere (see above).
- No accuracy pass/fail gate exists yet.
- Results are not claimed to generalize across machines, thermal states,
  driver versions, or other RTX 3050 units.
