# Physical-device calibration reference

## Why calibration exists, and why there is no PE mapping

TensorForge Core's accelerator presets describe an educational PE-array
abstraction: a grid of PEs, each doing `1 MAC/cycle`, fed by an SRAM/DRAM
hierarchy. A real NVIDIA GPU is not that abstraction wearing a different
name -- it has CUDA cores, Tensor Cores, SMs, register files, shared
memory, an L2 cache, DRAM, several clock domains, and vendor-library
kernels (cuBLAS/cuDNN) with their own internal tiling and mapping
decisions that TensorForge does not (and, at this milestone, cannot)
model.

So this milestone does **not** map "RTX 3050" to some number of
TensorForge PE rows/columns, does not equate CUDA-core or Tensor-Core
counts with PE counts, and does not add a "RTX 3050" accelerator preset
built from marketing specifications. Instead, `tensorforge_ops.calibration`
*measures* two empirical ceilings directly on the real runtime/device --
sustained compute throughput and sustained memory-copy bandwidth -- and
uses only those measured numbers, never a Core accelerator preset's PE
geometry/clock, as the basis for a physical-device prediction.

```
physical device
    -> compute probe (torch.mm)     -> effective_compute_flops_per_second
    -> memory probe (tensor copy)   -> effective_memory_bandwidth_bytes_per_second
DeviceCalibrationProfile (schema-versioned, backend/device/dtype/runtime-specific)
```

## Compute probe

A dense `M x N x K` GEMM (`torch.mm(a, b, out=c)`, defaults 2048x2048x2048)
with `a`, `b`, `c` preallocated on the target device before timing starts,
run under `torch.inference_mode()`. Warmup iterations are excluded.
CUDA timing is synchronized with `torch.cuda.synchronize()` around every
warmup and measured iteration (the same timing loop `benchmark.py` uses
for Milestone-12 benchmarks -- `run_timed_iterations()` is reused
directly, not reimplemented).

`effective_compute_flops_per_second = (2*M*N*K) / p50_latency`. p50 (not
min or max) is used because it is a steady-state central measurement,
less sensitive to a single fast/slow outlier iteration. This value is
called "measured sustained compute rate" / "effective compute ceiling"
everywhere -- never "GPU peak FLOPs," since it reflects one probe
shape/dtype/runtime, not a vendor peak-throughput claim.

## Memory probe

`dst.copy_(src)` on two preallocated same-size device tensors (default
64 MiB payload). One copy of an X-byte tensor logically moves X bytes
read + X bytes written, so modeled traffic is `2 * payload_bytes`.
`effective_memory_bandwidth_bytes_per_second = (2*payload_bytes) /
p50_latency`. This is an empirical memory-throughput proxy -- kernel
launch, runtime, and cache behavior all influence it -- so it is called
"effective sustained memory-copy bandwidth," never "peak DRAM bandwidth."

## Device calibration profile

`DeviceCalibrationProfile` (`calibration_schema_version = 1`, a distinct
schema from both Core's `schema_version` and Milestone-12's
`benchmark_schema_version`) records: `backend`, `device_type`,
`device_index`, `device_name`, `dtype`, `runtime_metadata` (torch
version, CUDA runtime version, device name/index when CUDA),
`device_metadata` (compute capability, total device memory when CUDA),
both probe results, and the two effective rates.

**Re-running calibration is not deterministic.** Unlike Core's experiment
fingerprint, `compute_calibration_fingerprint()` (SHA-256 of the profile's
canonical JSON) legitimately changes across runs of the identical
`CalibrationConfig`, because measured rates vary run to run (thermal
state, background load, driver scheduling). This is expected, not a bug.

## Device/runtime matching (why validation can refuse to run)

A calibration profile identifies one specific backend + device + dtype +
runtime combination via its `device_signature()`. Before comparing a
profile's prediction against a Milestone-12 `BenchmarkResult`,
`validate_prediction()` requires ALL of: `backend`, `device_type`,
`dtype`, PyTorch version, and (when CUDA) `device_index`, `device_name`,
`cuda_runtime_version` to match exactly. Any mismatch raises `ValueError`
with a specific reason -- a profile calibrated on CUDA FP16 never
silently validates a CPU or FP32 benchmark, and a profile from one
PyTorch/CUDA build never silently validates a benchmark run under a
different one. Recalibrate after a runtime change rather than reusing an
old profile.

## Calibration set vs. validation set (holdout)

Calibration uses **only** the synthetic compute/memory probes above.
Validation uses TensorForge workload presets (`gemm_tiny`,
`conv_spatial`, `transformer_small`, ...). The two are never mixed:
effective rates are computed once, from the probes, and never adjusted
to reduce error on any specific validation workload. See
[validation.md](validation.md) for the comparison methodology.

## Profile lifecycle

```bash
python -m tensorforge_ops calibrate \
    --backend pytorch --device cuda --dtype fp16 \
    --compute-m 2048 --compute-n 2048 --compute-k 2048 \
    --memory-probe-mib 64 --warmup 10 --iterations 50 \
    --output calibration-rtx3050-fp16.json
```

`--output` refuses to silently overwrite an existing file -- pass
`--force` to intentionally replace one. The saved JSON is the profile
alone (no envelope); `python -m tensorforge_ops validate
--calibration <path>` reloads it and recomputes the same fingerprint
deterministically from its contents.

## Limitations

- The compute/memory probes measure one shape/payload size; a workload
  far outside that regime (much smaller, launch-overhead-dominated, or
  much larger, capacity-bound) is not guaranteed to see the same
  effective rate -- this is part of what validation quantifies.
- Calibration and validation results are not stable across machines,
  power states, thermal states, background load, or driver/runtime
  versions. A profile measured on one run of one physical laptop GPU
  does not generalize to "RTX 3050 Laptop GPUs" as a class.
- CPU calibration is supported (useful for CI/manual testing) but is not
  this milestone's primary evidence; the live GPU results in
  [validation.md](validation.md) come from the actual RTX 3050 Laptop
  GPU available in this environment.
- No DCGM/Nsight telemetry is used or required by calibration -- the
  probes are plain PyTorch timing, nothing more.
