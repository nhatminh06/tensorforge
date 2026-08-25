# Milestone 3 model: DRAM/SRAM capacity analysis

## Modeled hierarchy

```
DRAM
  |
Global SRAM        (capacity_bytes)
  |
compute array       (PE mapping, see docs/pe-array.md)
```

This milestone models only SRAM **capacity** and the same idealized
baseline DRAM traffic already used for the Milestone-1 roofline model. It
does not model SRAM latency, SRAM bandwidth, SRAM-to-PE traffic, bank
conflicts, or any cache behavior.

## Working set

```
working_set_bytes = a_bytes + b_bytes + c_bytes
```

reusing `Gemm.a_bytes` / `b_bytes` / `c_bytes` directly.

## Fit is a capacity fact, not a reuse guarantee

A tensor "fitting" (`bytes <= sram_bytes`) means it *could* reside in SRAM
simultaneously with the others under this simplified model — not that the
compute array only ever reads it once from SRAM. Actual reuse depends on
tiling, loop order, and dataflow, none of which exist yet. It is entirely
possible (and demonstrated in tests) for A, B, and C to each fit
individually while the full working set does not fit simultaneously.

## Capacity accounting

```
full_working_set_fits = working_set_bytes <= sram_bytes

capacity_headroom_bytes = max(0, sram_bytes - working_set_bytes)
capacity_deficit_bytes  = max(0, working_set_bytes - sram_bytes)
working_set_to_capacity_ratio = working_set_bytes / sram_bytes
```

`working_set_to_capacity_ratio` may exceed 1.0; it is a ratio, not a
percentage-bounded "utilization."

## Tiling-required flag

```
tiling_required = not full_working_set_fits
```

This means only: the full A/B/C working set cannot reside simultaneously
in the modeled SRAM. It does **not** define a tiling strategy, tile
shape, or loop order — that is a future milestone.

## Baseline DRAM traffic (unchanged from Milestone 1)

```
dram_read_bytes  = a_bytes + b_bytes
dram_write_bytes = c_bytes
total_dram_bytes = dram_read_bytes + dram_write_bytes
```

This is identical in total to `Gemm.dram_bytes` and is **not** adjusted
when the working set fails to fit. Increased DRAM traffic caused by
refetching is a real effect of insufficient SRAM, but its exact size
depends on tile shape, loop order, and which tensor(s) are kept resident —
none of which this milestone models. Reporting a traffic number now would
be an unjustified guess, so we deliberately report only the capacity fact
(`tiling_required`), not a refetch estimate.

## What this milestone does NOT do

- No exact refetch/extra-traffic estimation.
- No GEMM tiling (no tile M/N/K).
- No dataflow (weight-stationary, output-stationary, ...).
- No SRAM-to-PE traffic.
- No SRAM/DRAM latency or bandwidth modeling.
- No cache behavior, bank conflicts, or NoC.
- PE mapping (`pe_array.py`) and roofline (`roofline.py`) are untouched by
  SRAM capacity — they remain independent analytical views.
