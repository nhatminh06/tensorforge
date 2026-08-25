# Milestone 1 model: analytical GEMM + roofline

## Operation convention

```
MACs  = M * N * K
FLOPs = 2 * MACs
```

1 multiply-accumulate is counted as 2 floating-point operations (one
multiply, one add). This is a counting convention, not a hardware claim.

## Memory-traffic assumption

Milestone 1 assumes each input tensor is fetched once from the modeled
memory boundary and each output tensor is written once:

```
DRAM bytes = bytes(A) + bytes(B) + bytes(C)
```

This is an idealized, lower-bound-style analytical assumption. It does not
model tiling, SRAM capacity, cache behavior, or dataflow-dependent reuse.
Real hardware may transfer more or less data than this depending on how the
GEMM is actually mapped. Milestone 4 (tiling) and Milestone 5 (dataflows)
refine this.

## Arithmetic intensity

```
arithmetic_intensity = FLOPs / DRAM bytes     [FLOP/byte]
```

The denominator is the Milestone-1 modeled DRAM boundary defined above —
not SRAM or PE-local traffic, which do not exist yet in this model.

## Roofline

```
ridge_point     = peak_compute / memory_bandwidth        [FLOP/byte]
memory_ceiling  = arithmetic_intensity * memory_bandwidth  [FLOP/s]
compute_ceiling = peak_compute                                  [FLOP/s]
attainable_performance = min(compute_ceiling, memory_ceiling)
```

Classification, with equality handled via a relative tolerance:

```
memory_ceiling < compute_ceiling  -> memory-bound
memory_ceiling > compute_ceiling  -> compute-bound
memory_ceiling == compute_ceiling -> balanced
```

Equivalently, comparing arithmetic intensity to the ridge point:
`AI < ridge_point` implies memory-bound, `AI > ridge_point` implies
compute-bound.

## Time estimate

```
compute_time   = FLOPs / peak_compute
memory_time    = DRAM bytes / memory_bandwidth
estimated_time = max(compute_time, memory_time)
```

This is an analytical lower-bound time under a perfect-overlap assumption
(compute and memory transfers fully overlap, and one of them dominates).
It is not a measured latency, and it does not model queueing, cache
misses, synchronization, pipeline startup, or network-on-chip contention.

## Limitations

- No tiling.
- No cache/SRAM capacity model.
- No PE utilization.
- No dataflow.
- No pipeline overhead.
- No network-on-chip.
- No contention.
- No actual measured hardware latency.
