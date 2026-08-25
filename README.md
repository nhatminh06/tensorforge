# TensorForge

TensorForge is an educational simulator for studying how neural-network
workloads interact with accelerator compute, memory bandwidth, and memory
capacity. It starts from analytical models and adds architectural detail
incrementally (PE arrays, memory hierarchy, tiling, dataflows, convolution,
Transformer operations).

## Current model (Milestone 2)

Analytical GEMM + roofline + idealized rectangular PE-array mapping. There
is no SRAM/tiling model, no dataflow, no systolic timing, and no cycle
accuracy. All numbers come from closed-form formulas over a `Gemm`, a
`HardwareConfig`, and (optionally) a `PeArray`.

## Equations

```
C[M,N] = A[M,K] x B[K,N]

MACs  = M * N * K
FLOPs = 2 * MACs                (1 MAC = 2 FLOPs)

DRAM bytes = bytes(A) + bytes(B) + bytes(C)
             (A read once, B read once, C written once — idealized)

arithmetic_intensity = FLOPs / DRAM bytes            [FLOP/byte]

ridge_point    = peak_compute / memory_bandwidth      [FLOP/byte]
memory_ceiling = arithmetic_intensity * memory_bandwidth   [FLOP/s]
compute_ceiling = peak_compute                              [FLOP/s]
attainable_performance = min(compute_ceiling, memory_ceiling)

compute_time   = FLOPs / peak_compute        [s]
memory_time    = DRAM bytes / memory_bandwidth  [s]
estimated_time = max(compute_time, memory_time)  (perfect-overlap assumption)
```

Classification: `memory_ceiling < compute_ceiling` -> memory-bound,
`memory_ceiling > compute_ceiling` -> compute-bound, equal (within
tolerance) -> balanced.

See [docs/model.md](docs/model.md) for full detail and assumptions.

### PE array (optional)

```
M -> PE rows, N -> PE columns
row_waves    = ceil(M / rows)
column_waves = ceil(N / columns)
waves        = row_waves * column_waves

spatial_utilization = (M * N) / (waves * rows * columns)

compute_cycles = waves * K       (1 MAC/PE/cycle)
```

An idealized spatial mapping of GEMM output onto a finite PE array — not a
systolic-array timing model. See [docs/pe-array.md](docs/pe-array.md).

## Build / run

```bash
pip install -e .
pytest -q
```

## Example

```bash
python -m tensorforge --m 1024 --n 1024 --k 1024 --dtype fp16 \
    --peak-tflops 10 --bandwidth-gbps 200
```

Prints GEMM dimensions, MAC/FLOP counts, tensor byte sizes, modeled DRAM
traffic, arithmetic intensity, roofline ceilings, estimated time, and the
compute-bound/memory-bound classification.

Add `--pe-rows`/`--pe-cols` (both required together) to also print the
PE-array mapping section:

```bash
python -m tensorforge --m 128 --n 128 --k 128 --dtype fp16 \
    --peak-tflops 1 --bandwidth-gbps 100 --pe-rows 16 --pe-cols 16
```

## Limitations

- GEMM only; no convolution or Transformer operations yet.
- DRAM traffic assumes each tensor is fetched/written exactly once — no
  tiling, no SRAM capacity, no reuse modeling.
- PE-array model is a spatial-mapping + idealized compute-cycle model, not
  a systolic-array timing model: no fill/drain latency, operand
  propagation, memory access, NoC, or synchronization. No clock frequency,
  so results are cycles, not seconds.
- No dataflow modeling.
- Time estimates are analytical lower bounds under perfect compute/memory
  overlap, not measured latency. No queueing, contention, pipeline startup,
  or synchronization overhead is modeled.
- Not validated against real hardware; not cycle-accurate.

## Roadmap

PE array -> memory hierarchy -> tiling -> dataflows -> convolution ->
Transformer operations -> architecture sweeps -> validation.
