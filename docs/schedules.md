# Milestone 5 model: tile-residency schedule comparison

## What this models

For the same GEMM, the same tile shape, and the same SRAM capacity, three
explicit loop orders are compared. Each keeps a different operand tile
resident in SRAM across the innermost reuse loop; the traffic difference
between them is caused *only* by loop order and residency, not by any
change to the arithmetic, the tile shape, or the SRAM capacity.

These are literal descriptions of the modeled loop nest and which tile
stays resident. They are **not** yet asserted to implement the general
hardware dataflow concepts "output stationary," "input stationary," or
"weight stationary" — that would require the operand roles and movement
to be checked against those definitions precisely, which is future work.

## Comparison table

| Schedule     | Resident tensor | A behavior                         | B behavior                         | C behavior                                  |
|--------------|------------------|-------------------------------------|-------------------------------------|----------------------------------------------|
| `c-resident` | C                | reloaded once per `n_tiles`         | reloaded once per `m_tiles`         | no partial-sum traffic (fully accumulated resident) |
| `a-resident` | A                | read once (`a_bytes`, no repeat)    | reloaded once per `m_tiles`         | partial sums cross DRAM across K tiles      |
| `b-resident` | B                | reloaded once per `n_tiles`         | read once (`b_bytes`, no repeat)    | partial sums cross DRAM across K tiles      |

## c-resident (the original Milestone-4 schedule, unchanged)

```
for mi in m_tiles:
    for ni in n_tiles:
        C_tile = zero                      # resident
        for ki in k_tiles:
            A_tile = load(A[mi, ki])
            B_tile = load(B[ki, ni])
            C_tile += A_tile @ B_tile
        store(C_tile)
```

```
A reads  = a_bytes * n_tiles
B reads  = b_bytes * m_tiles
C reads  = 0
C writes = c_bytes
```

## a-resident

```
for mi in m_tiles:
    for ki in k_tiles:
        A_tile = load(A[mi, ki])           # resident
        for ni in n_tiles:
            B_tile = load(B[ki, ni])
            C_tile = load(C[mi, ni]) or zero  # zero only on the first K tile
            C_tile += A_tile @ B_tile
            store(C_tile)                     # partial or final
```

```
A reads  = a_bytes
B reads  = b_bytes * m_tiles
C reads  = c_bytes * (k_tiles - 1)
C writes = c_bytes * k_tiles
```

## b-resident (symmetric)

```
for ni in n_tiles:
    for ki in k_tiles:
        B_tile = load(B[ki, ni])           # resident
        for mi in m_tiles:
            A_tile = load(A[mi, ki])
            C_tile = load(C[mi, ni]) or zero
            C_tile += A_tile @ B_tile
            store(C_tile)
```

```
A reads  = a_bytes * n_tiles
B reads  = b_bytes
C reads  = c_bytes * (k_tiles - 1)
C writes = c_bytes * k_tiles
```

## Why partial-sum traffic appears in a-resident/b-resident

In `c-resident`, K is the innermost loop, so C stays resident across the
*entire* K reduction for one output tile — it only ever leaves SRAM once,
fully summed. In `a-resident`/`b-resident`, K is no longer innermost, so
each output tile `C[mi,ni]` is visited once per K tile, and its partial
sum must be written out and read back in between visits (except the very
first visit, which starts from zero, and the last, which needs no further
read-back). Summed over all output tiles (which exactly partition C):

```
C writes = c_bytes * k_tiles
C reads  = c_bytes * (k_tiles - 1)
```

This is why K-tiling has opposite effects across the schedules:
- `c-resident`: splitting K (Tm/Tn fixed) does **not** change total DRAM
  bytes — only tile_steps and capacity change.
- `a-resident`/`b-resident`: splitting K directly **increases** C traffic.

At `k_tiles = 1` there are no intermediate partial sums, so
`a-resident`/`b-resident` reduce to `C reads = 0`, `C writes = c_bytes` —
identical in form to `c-resident`.

## Full residency

When `m_tiles = n_tiles = k_tiles = 1`, all three schedules collapse to
the same traffic: `A + B + C`, matching the Milestone-1 ideal baseline
exactly (amplification = 1.0). Schedule choice only matters once tiling
introduces genuine reuse decisions.

## Why a-resident and b-resident are not equivalent

`a-resident` trades A reuse for repeated B loads (`* m_tiles`);
`b-resident` trades B reuse for repeated A loads (`* n_tiles`). For a
square, symmetric workload these can tie, but for an asymmetric shape
(e.g. `M != N`, or `m_tiles != n_tiles`) they diverge — whichever
schedule repeats the *smaller* reuse multiplier wins. There is no
universal winner; it depends on the workload shape and chosen tile.

## Comparison mode

`compare_schedules(gemm, tile, hierarchy)` analyzes the same tile under
all three schedules and reports `best_among_modeled_schedules` — every
schedule tied for the lowest `total_dram_bytes` among the three modeled
here. This is explicitly **not** a claim of global optimality: only these
three schedules are considered, and the tile shape is not searched.

## Terminology

These traffic totals are **exact under the specified analytical
loop/residency schedules**. They are NOT measured hardware traffic, a
cache simulation, NoC traffic, PE-local traffic, or cycle-level memory
behavior.

## What this milestone does NOT do

- No automatic tile search or automatic schedule search.
- No general dataflow framework — exactly three fixed schedules.
- No "output/input/weight stationary" hardware-dataflow claims.
- No PE-local memory, NoC, memory latency, or SRAM bandwidth.
- No coupling between schedule choice and PE mapping or the roofline
  model — both remain unchanged from prior milestones.
