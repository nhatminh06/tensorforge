# Milestone 4 model: capacity-constrained GEMM tiling traffic

## What this models

Given a user-chosen tile shape `(Tm, Tn, Tk)`, this checks whether one
tile's working set fits the modeled SRAM and computes the **exact** DRAM
traffic that results from one specific loop schedule. It does not search
for a good tile shape. This document describes the original
`c-resident` schedule in detail; two additional explicit schedules
(`a-resident`, `b-resident`) and a side-by-side comparison are covered in
[docs/schedules.md](schedules.md).

## Tile dimensions and counts

```
m_tiles = ceil(M / Tm)
n_tiles = ceil(N / Tn)
k_tiles = ceil(K / Tk)

output_tiles = m_tiles * n_tiles
tile_steps   = m_tiles * n_tiles * k_tiles
```

`Tm`/`Tn`/`Tk` may exceed the GEMM's own dimensions; actual tile
dimensions are clipped to the workload (`min(Tm, M)`, etc.) — no padded
elements are charged as traffic or capacity.

## Fixed schedule: C-output-tile-resident

```
for mi in m_tiles:
    for ni in n_tiles:
        C_tile = zero                      # resident in SRAM
        for ki in k_tiles:
            A_tile = load(A[mi, ki])        # from DRAM
            B_tile = load(B[ki, ni])        # from DRAM
            C_tile += A_tile @ B_tile
        store(C_tile)                       # to DRAM
```

This is one explicit, fixed loop schedule — not a general "output
stationary accelerator dataflow," and not yet compared against
alternative loop orders or residency choices. That comparison is future
work.

## SRAM residency and capacity requirement

During one K-tile step, SRAM must simultaneously hold one A tile, one B
tile, and the resident (in-progress) C tile:

```
tile_working_set_bytes =
    (Tm*Tk + Tk*Tn + Tm*Tn) * bytes_per_element
```

using the largest actual tile (index 0 along each axis, clipped to the
workload). If this exceeds SRAM capacity, the tiling configuration is
**rejected outright** — `analyze_tiling` raises `ValueError`. No implicit
shrinking, no silent proceeding.

## Traffic

- **A reads**: `A[mi, ki]` is reloaded once per `ni`, because C — not A —
  stays resident. Total: `a_bytes * n_tiles`.
- **B reads**: `B[ki, ni]` is reloaded once per `mi`, because C — not B —
  stays resident. Total: `b_bytes * m_tiles`.
- **C writes**: each `C[mi, ni]` tile starts at zero, accumulates across
  the K loop while resident, and is written exactly once. There is no C
  read.

```
total_dram_bytes =
    a_bytes * n_tiles
  + b_bytes * m_tiles
  + c_bytes
```

This holds exactly even with edge tiles, because summing the useful
bytes of all sub-tiles within a fixed reuse context always equals the
full tensor's bytes — edge clipping changes individual tile sizes, not
the total.

### Why K-tiling alone doesn't change total traffic

Splitting K only changes `k_tiles` (and hence `tile_steps` and the tile
working set) — it does not introduce any additional M/N reuse repetition
under this schedule. With `Tm`/`Tn` fixed, changing `Tk` changes capacity
requirements and step count but **not** `total_dram_bytes`.

## Traffic amplification and effective arithmetic intensity

```
traffic_amplification = total_dram_bytes / ideal_baseline_dram_bytes
```

where `ideal_baseline_dram_bytes` is the unchanged Milestone-1 baseline
(`Gemm.dram_bytes`). Amplification is `>= 1.0` for this schedule, and
equals exactly `1.0` when `m_tiles == n_tiles == 1` regardless of K
tiling.

```
effective_arithmetic_intensity = Gemm.flops / total_dram_bytes   [FLOP/byte]
```

This is distinct from — and typically lower than — the Milestone-1 ideal
arithmetic intensity, since `total_dram_bytes >= ideal_baseline_dram_bytes`.
The existing roofline model (`roofline.py`) is untouched; it continues to
use the ideal baseline.

## The core tradeoff

- Larger `Tn` reduces `n_tiles`, which reduces repeated **A** traffic.
- Larger `Tm` reduces `m_tiles`, which reduces repeated **B** traffic.
- Larger tiles require more SRAM (`tile_working_set_bytes` grows).

Two tile shapes with the same PE-independent SRAM capacity requirement
can produce meaningfully different total DRAM traffic — tile *shape*,
not just SRAM *capacity*, determines reuse.

## Terminology

Traffic numbers here are **exact under the modeled schedule** — not
"exact hardware traffic." A real accelerator's traffic depends on loop
order, dataflow, and residency choices not modeled yet.

## What this milestone does NOT do

- No automatic tile search (tile shape is always user-supplied).
- No alternative loop orders or dataflow comparison.
- No SRAM-to-PE traffic, memory latency, or cycle accounting.
- No coupling between tile shape and `PeArray`/`map_gemm` — PE mapping is
  unaffected by tiling in this milestone.
- No claim of optimal or universal GEMM traffic.
