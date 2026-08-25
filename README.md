# TensorForge

TensorForge is an educational simulator for studying how neural-network
workloads interact with accelerator compute, memory bandwidth, and memory
capacity. It starts from analytical models and adds architectural detail
incrementally (PE arrays, memory hierarchy, tiling, dataflows, convolution,
Transformer operations).

## Current model (Milestone 5)

Analytical GEMM + roofline + rectangular PE mapping + SRAM capacity +
explicit GEMM tiling traffic under three explicit loop-order/residency
schedules (`c-resident`, `a-resident`, `b-resident`). Tiled DRAM traffic
is exact only for the specific modeled schedule used — not a claim of
optimal, universal, or real-hardware GEMM traffic, and not yet a claim
that these schedules implement general "output/input/weight stationary"
hardware dataflows. There is no automatic tile/schedule search, no
systolic timing, and no cycle accuracy. All numbers come from closed-form
formulas over a `Gemm`, a `HardwareConfig`, an optional `PeArray`, an
optional `MemoryHierarchy`, and an optional `GemmTile` + `GemmSchedule`.

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

### SRAM capacity (optional)

```
working_set_bytes = a_bytes + b_bytes + c_bytes
full_working_set_fits = working_set_bytes <= sram_bytes
tiling_required = not full_working_set_fits
```

Reports whether A, B, C individually and the full working set fit in a
modeled SRAM capacity — a capacity fact, not a reuse guarantee. Baseline
DRAM traffic (A+B read, C written) is unchanged regardless of fit; the
exact extra traffic caused by an SRAM-limited working set requires tiling.
See [docs/memory.md](docs/memory.md).

### GEMM tiling (optional, requires SRAM)

```
m_tiles = ceil(M/Tm), n_tiles = ceil(N/Tn), k_tiles = ceil(K/Tk)

tile_working_set_bytes = (Tm*Tk + Tk*Tn + Tm*Tn) * bytes_per_element
                          (rejected if it exceeds SRAM capacity)

A reads = a_bytes * n_tiles      (C stays resident, not A)
B reads = b_bytes * m_tiles      (C stays resident, not B)
C writes = c_bytes                (written once, no C read)

traffic_amplification = total_dram_bytes / ideal_baseline_dram_bytes
effective_arithmetic_intensity = flops / total_dram_bytes
```

For one explicit user-supplied tile shape. See [docs/tiling.md](docs/tiling.md).

### Schedule comparison (optional, requires tiling)

```
c-resident: A reads = a_bytes*n_tiles, B reads = b_bytes*m_tiles, C reads = 0
a-resident: A reads = a_bytes,         B reads = b_bytes*m_tiles, C reads = c_bytes*(k_tiles-1)
b-resident: A reads = a_bytes*n_tiles, B reads = b_bytes,         C reads = c_bytes*(k_tiles-1)
```

Three explicit loop-order/residency schedules over the *same* GEMM, tile,
and SRAM — only loop order changes. `compare_schedules` reports every
schedule tied for lowest traffic among these three (`best_among_modeled_schedules`,
not a claim of global optimality). No automatic tile or schedule search.
See [docs/schedules.md](docs/schedules.md).

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

Add `--sram-kib` (independently of `--pe-rows`/`--pe-cols`) to also print
the SRAM capacity section:

```bash
python -m tensorforge --m 128 --n 128 --k 128 --dtype fp16 \
    --peak-tflops 1 --bandwidth-gbps 100 --sram-kib 128
```

Add `--tile-m`/`--tile-n`/`--tile-k` (all three required together, and
requiring `--sram-kib`) to also print exact tiled DRAM traffic under the
fixed loop schedule:

```bash
python -m tensorforge --m 4 --n 4 --k 4 --dtype fp32 \
    --peak-tflops 1 --bandwidth-gbps 100 --sram-kib 0.0625 \
    --tile-m 2 --tile-n 2 --tile-k 2
```

Add `--schedule {c-resident,a-resident,b-resident}` (defaults to
`c-resident`, matching Milestone-4 behavior) to pick a residency schedule,
or `--compare-schedules` to print all three side by side:

```bash
python -m tensorforge --m 4 --n 4 --k 4 --dtype fp32 \
    --peak-tflops 1 --bandwidth-gbps 100 --sram-kib 0.0625 \
    --tile-m 2 --tile-n 2 --tile-k 2 --compare-schedules
```

## Limitations

- GEMM only; no convolution or Transformer operations yet.
- DRAM traffic assumes each tensor is fetched/written exactly once — no
  reuse modeling. This baseline traffic does not change even when the
  working set does not fit in the modeled SRAM (that's what tiling
  traffic, below, is for).
- PE-array model is a spatial-mapping + idealized compute-cycle model, not
  a systolic-array timing model: no fill/drain latency, operand
  propagation, memory access, NoC, or synchronization. No clock frequency,
  so results are cycles, not seconds.
- SRAM model is capacity-only: fit/no-fit and headroom/deficit, not an
  exact refetch-traffic estimate on its own.
- Tiling traffic is exact only under the specific modeled schedule used —
  not optimal, not universal, not real-hardware traffic. Only three
  explicit schedules are modeled (`c-resident`, `a-resident`,
  `b-resident`); tile shape and schedule are always user-supplied, with
  no automatic search. No PE-local memory, NoC, SRAM-to-PE traffic,
  memory latency, or PE/tile coupling.
- Schedule names describe the modeled loop order/residency literally, not
  a verified claim of matching general "output/input/weight stationary"
  hardware dataflow definitions.
- Time estimates are analytical lower bounds under perfect compute/memory
  overlap, not measured latency. No queueing, contention, pipeline startup,
  or synchronization overhead is modeled.
- Not validated against real hardware; not cycle-accurate.

## Roadmap

PE array -> memory hierarchy -> tiling -> dataflows -> convolution ->
Transformer operations -> architecture sweeps -> validation.
