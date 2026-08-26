# Milestone 7 model: bounded configuration-space exploration

## What this models

A deterministic search over a **finite, user-supplied** Cartesian product
of tile shapes, residency schedules, and PE-array shapes. It introduces
no new performance model — it only drives the existing analytical models
(`analyze_tiling`, `map_gemm`, `estimate_execution_time`) across many
configurations and ranks the results.

## Search dimensions (varied)

- Tile shape: `tile_m_values x tile_n_values x tile_k_values`
- Residency schedule: any subset of `c-resident`, `a-resident`, `b-resident`
- PE-array shape: `pe_rows_values x pe_cols_values`

## Fixed inputs (held constant across one exploration)

- GEMM (`M`, `N`, `K`) and dtype
- SRAM capacity
- PE clock frequency
- DRAM bandwidth

Clock, bandwidth, SRAM capacity, and dtype are **not** searched this
milestone — sweeping them would make the search space and its
interpretation much harder to reason about. Tile candidates are always
user-supplied; there is no automatic candidate generation.

## Candidate representation

Each candidate (`CandidateConfig`) identifies exactly:

```
tile_m, tile_n, tile_k
schedule
pe_rows, pe_cols
```

No hidden parameters — everything that varies between candidates is a
field on this dataclass.

## Candidate count and search bound

```
candidate_count =
    len(tile_m) * len(tile_n) * len(tile_k)
    * len(pe_rows) * len(pe_cols)
    * len(schedules)
```

using deduplicated candidate-value lists (duplicates in the input, e.g.
`tile_m=[16,16,32]`, are silently reduced to `{16,32}` before counting —
each candidate is evaluated exactly once regardless of input repetition).

If `candidate_count` exceeds `MAX_CANDIDATES` (100,000), the search is
rejected outright with a clear error **before evaluating anything** — no
silent truncation.

## Feasibility

A candidate's tile is feasible exactly under the existing Milestone-4
capacity rule (`analyze_tiling`'s `ValueError` on tile-working-set
overflow): `(Tm*Tk + Tk*Tn + Tm*Tn) * bytes_per_element <= sram_bytes`
(using workload-clipped tile dimensions). Since tile capacity does not
depend on PE shape, an infeasible tile is recorded once and its
infeasibility applies to every PE candidate that would have paired with
it — the search does not crash, it continues past infeasible candidates
and reports both feasible and infeasible counts.

## Evaluation pipeline

```
CandidateConfig
   |
GemmTile(tile_m, tile_n, tile_k)
   |
analyze_tiling(gemm, tile, hierarchy, schedule)   -> TilingResult (or infeasible)
   |
PeArray(pe_rows, pe_cols)
   |
map_gemm(gemm, pe_array)                          -> PeMappingResult
   |
estimate_execution_time(pe_mapping, tiling_result, hardware, timing_config)
   -> ExecutionTimingResult
```

No formula from `tiling.py`, `pe_array.py`, or `timing.py` is duplicated
in `explore.py` — it only calls them.

## Ranking objective

Primary key: **`perfect_overlap_time_seconds`, ascending** (smaller is
better). Chosen because it is the single number that already combines
compute and memory time under the existing ideal-overlap analytical
model (Milestone 6) — the same model used for one configuration. It is
**not** measured runtime and **not** a claim of universal hardware
optimality.

## Deterministic tie-breakers (ascending)

When `perfect_overlap_time_seconds` ties, break by, in order:

1. `serialized_time_seconds`
2. `total_dram_bytes` (`ExecutionTimingResult.dram_bytes`)
3. `compute_cycles`
4. PE count (`pe_rows * pe_cols`)
5. candidate fields, lexicographically (`tile_m, tile_n, tile_k, schedule value, pe_rows, pe_cols`)

This order is deterministic and documented, not arbitrary: it prefers
the candidate that also wins the no-overlap bound, then the one moving
less data, then the one needing fewer compute cycles, then the one using
less PE hardware, and only falls back to raw field order as a final,
meaningless-but-deterministic tiebreak. `ExplorationResult.tied_for_best`
counts how many feasible candidates share the exact best
`perfect_overlap_time_seconds` — a genuine analytical tie is surfaced,
never silently broken and presented as a real performance difference.

## Top-K

`top_k` (validated `> 0`) selects how many ranked candidates to display.
If fewer feasible candidates exist than `top_k`, all feasible candidates
are shown — `ExplorationResult.ranked` always holds the complete sorted
feasible list; truncation to `top_k` happens only at display time.

## Terminology

The result of a search is always reported as **"best among searched
candidates"** — never "optimal accelerator," "global optimum," or "best
possible configuration." Searching more configurations does not make the
underlying model more accurate; it only samples more points from the
same analytical model with the same assumptions and limitations as a
single-configuration run.

## Winner explanation

After ranking, a short, deterministic explanation is generated from
already-computed metrics (no free-text generation): the winner is
compared against (when such a candidate exists in the searched set) the
next-ranked candidate, the same tile/schedule with a smaller PE array,
and the same tile/PE with a different schedule — showing concretely
whether smaller PE arrays would have tied or lost, and whether other
schedules moved more DRAM traffic.

## Limitations

- Bounded discrete search only — user-supplied candidate sets, no
  automatic tile/PE candidate generation.
- Only `perfect_overlap_time_seconds` is the objective — no power/energy
  objective, no multi-objective optimization.
- No clock, bandwidth, or SRAM sweep this milestone.
- No continuous optimization, no scipy/Optuna/genetic/Bayesian search.
- No parallel/multiprocessing search.
- Inherits every limitation of the underlying models it drives: no
  DRAM/SRAM latency, no NoC, no cycle accuracy, no real-hardware
  validation.
