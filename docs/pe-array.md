# Milestone 2 model: rectangular PE-array mapping

## What this models

An idealized spatial mapping of a GEMM output C[M,N] onto a rectangular
array of processing elements, plus an idealized compute-cycle count. It
does **not** model a systolic array, memory access, or a pipeline.

## Mapping assumption

```
M -> PE rows
N -> PE columns
```

Each active PE is responsible for exactly one output element C[m,n] during
a given wave, and computes it as a length-K dot product at a fixed
`1 MAC/PE/cycle`. If M or N does not evenly divide the array dimensions,
the array is swept in multiple waves, and the final row/column of waves is
only partially occupied.

## Wave model

```
row_waves    = ceil(M / rows)
column_waves = ceil(N / columns)
waves        = row_waves * column_waves
```

## Spatial utilization

```
useful_pe_slots    = M * N
available_pe_slots = waves * rows * columns
unused_pe_slots    = available_pe_slots - useful_pe_slots

spatial_utilization = useful_pe_slots / available_pe_slots
```

This measures PE occupancy caused only by M/N not filling the array
exactly. It does not include memory stalls, pipeline bubbles, data
dependencies, SRAM misses, or NoC stalls.

## Compute-cycle model

```
cycles_per_wave = K                 (1 MAC/PE/cycle)
compute_cycles  = waves * cycles_per_wave
```

`compute_cycles` is reported in cycles, not seconds — no clock frequency
is modeled in this milestone.

## What this is NOT

This is **not** a systolic-array cycle count. It ignores:

- array fill/drain latency
- operand propagation
- pipeline latency
- memory access (DRAM/SRAM)
- network-on-chip effects
- synchronization
- partial-sum movement

Call these numbers "analytical compute cycles" or "idealized compute
cycles" — never "measured" or "cycle-accurate."

## Relationship to the roofline model

The roofline model (`docs/model.md`) and this PE-array model are
intentionally kept separate. Roofline answers "compute- or memory-bound
under analytical FLOP/s and bandwidth ceilings?" The PE-array model
answers "how effectively does this finite compute array spatially map
this GEMM, and how many idealized cycles does that take?" Connecting PE
geometry to the roofline's peak FLOP/s would require additional
assumptions (clock frequency, MACs per PE, dtype throughput) that are not
introduced yet.

## Limitations

- Fixed 1 MAC/PE/cycle (no wider PEs yet).
- No clock frequency — cycles, not seconds.
- No memory stalls of any kind.
- No SRAM or tiling.
- No dataflow / operand-movement model.
- No systolic fill/drain or pipeline model.
