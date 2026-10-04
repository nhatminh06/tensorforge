# Canonical TensorForge hardware evidence

This bundle is the canonical portfolio capture from TensorForge commit
`37871229d6bc44991a4900ad00449fe390065b88`. It is distinct from the historical RTX 3050
validation study in `docs/ops/validation.md`; values may naturally differ.

## Device

Real `NVIDIA GeForce RTX 3050 Laptop GPU` using the PyTorch CUDA backend and `fp16`.

## Runtime

- PyTorch: `2.14.1+cu130`
- CUDA runtime: `13.0`
- warmups: 10
- measured iterations: 50

## Calibration

Measured empirical ceilings from a 2048 x 2048 x 2048 compute probe and a
64 MiB memory-copy probe:

- effective compute rate: 11.687860 TFLOP/s
- effective memory-copy bandwidth: 180.748794 GB/s

These are probe-derived empirical ceilings, not vendor specifications or
theoretical Tensor Core peak values.

## Primary workload

`gemm_large_square`:

- predicted latency: 0.000183736250165 seconds
- measured p50: 0.000195164502657 seconds
- measured p95: 0.00022512414871 seconds
- APE: 5.855702%
- measured/predicted: 1.062199x
- predicted bottleneck: `compute-bound`

The bottleneck label is analytical, not a measured-bottleneck claim.

## Contrast workload

`gemm_tiny`:

- predicted latency: 5.4387084969e-07 seconds
- measured p50: 1.22790006571e-05 seconds
- measured p95: 1.37907518365e-05 seconds
- APE: 95.570724%
- measured/predicted: 22.577052x

Any larger small-workload error is consistent with fixed framework/kernel-launch
overhead absent from the analytical lower-bound model; this capture does not
establish that explanation as a cause.

## Telemetry

Real NVML evidence was captured in a separate phase (35
samples). Telemetry is diagnostic context and does not gate a verdict.
Unsupported metrics remain unavailable rather than being converted to zero.

- supported fields: clock_event_reasons, gpu_utilization_percent, memory_activity_percent, memory_clock_mhz, memory_used_bytes, performance_state, power_watts, sm_clock_mhz, temperature_celsius
- unavailable fields: dram_bandwidth_utilization_percent, sm_activity_percent, sm_occupancy_percent, tensor_activity_percent

## Regression

**NOT EVALUATED**

No defensible performance-changing historical baseline was available.

## Right-sizing

**OMITTED**

Only one physical device was measured.

## Impact

**NOT EVALUATED**

A measured regression result is required.

## Limitations

- One physical GPU and one capture session.
- Laptop thermal and power state can affect measurements.
- The model is analytical and a lower bound, not cycle accurate or a GPU simulator.
- No measured-bottleneck claim is made.
- Results do not generalize to every RTX 3050 Laptop GPU.
- TensorForge Ops does not assess model quality or correctness. The recommendation
  covers measured performance and deployment/infrastructure evidence only.
