"""Bounded, deterministic exploration of tile/schedule/PE-array configurations.

This module does not implement any new performance model. It generates a
finite, user-supplied Cartesian product of candidates (tile shape x
residency schedule x PE-array shape) with GEMM, dtype, SRAM capacity,
clock, and DRAM bandwidth held fixed, and evaluates each candidate by
calling the existing analytical models:

    GemmTile + schedule -> analyze_tiling()      (tiling.py, Milestone 4/5)
    PeArray             -> map_gemm()             (pe_array.py, Milestone 2)
    both + hardware      -> estimate_execution_time()  (timing.py, Milestone 6)

Candidates whose tile does not fit the modeled SRAM are marked infeasible
(analyze_tiling's ValueError is caught, not propagated) and excluded from
ranking, but the search continues.

Ranking objective: `perfect_overlap_time_seconds`, ascending (smaller is
better). This is chosen because it combines both compute and memory time
into a single number under the existing ideal-overlap analytical model —
the same model already used for a single configuration in Milestone 6.
It is NOT measured runtime, and a candidate ranked first is only the
"best among searched candidates," never a claim of global or real-hardware
optimality.

Deterministic tie-break order (ascending) when perfect_overlap_time_seconds
ties:
    1. serialized_time_seconds
    2. total_dram_bytes
    3. compute_cycles
    4. PE count (rows * columns)
    5. candidate fields, lexicographically (tile_m, tile_n, tile_k,
       schedule value, pe_rows, pe_cols)

This order is deterministic and documented, not arbitrary: it prefers the
candidate that also wins the no-overlap bound, then the one that moves
less data, then the one needing fewer compute cycles, then the one using
less PE hardware, and only falls back to raw field order as a final,
meaningless-but-deterministic tiebreak. Candidates sharing the exact best
perfect_overlap_time_seconds are counted in `tied_for_best` so a genuine
tie is surfaced rather than silently picking one as "the" winner.
"""

from dataclasses import dataclass

from tensorforge.gemm import Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray, PeMappingResult, map_gemm
from tensorforge.tiling import GemmSchedule, GemmTile, TilingResult, analyze_tiling
from tensorforge.timing import ExecutionTimingResult, TimingConfig, estimate_execution_time

MAX_CANDIDATES = 100_000

_ALL_SCHEDULES = (GemmSchedule.C_RESIDENT, GemmSchedule.A_RESIDENT, GemmSchedule.B_RESIDENT)


@dataclass(frozen=True)
class CandidateConfig:
    tile_m: int
    tile_n: int
    tile_k: int
    schedule: GemmSchedule
    pe_rows: int
    pe_cols: int


@dataclass(frozen=True)
class CandidateResult:
    """A fully evaluated, feasible candidate. Infeasible candidates are not
    materialized as CandidateResult objects — they are only counted, plus
    their tile-capacity reason is retained in ExplorationResult.infeasible_reasons.
    """

    candidate: CandidateConfig
    tiling_result: TilingResult
    pe_mapping: PeMappingResult
    timing_result: ExecutionTimingResult


@dataclass(frozen=True)
class ExplorationResult:
    total_candidates: int
    feasible_candidates: int
    infeasible_candidates: int
    top_k: int
    tied_for_best: int
    ranked: tuple[CandidateResult, ...]
    infeasible_reasons: tuple[str, ...]


def _validate_positive_int_list(name: str, values: list[int]) -> tuple[int, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} entries must be positive ints, got {value!r}")
    return tuple(sorted(set(values)))


def _validate_schedules(schedules: list[GemmSchedule] | None) -> tuple[GemmSchedule, ...]:
    if schedules is None:
        return _ALL_SCHEDULES
    if len(schedules) == 0:
        raise ValueError("schedules must not be empty")
    unique = set(schedules)
    return tuple(s for s in _ALL_SCHEDULES if s in unique)


def _rank_key(result: CandidateResult):
    c = result.candidate
    t = result.timing_result
    return (
        t.perfect_overlap_time_seconds,
        t.serialized_time_seconds,
        t.dram_bytes,
        t.compute_cycles,
        c.pe_rows * c.pe_cols,
        c.tile_m,
        c.tile_n,
        c.tile_k,
        c.schedule.value,
        c.pe_rows,
        c.pe_cols,
    )


def explore(
    gemm: Gemm,
    hierarchy: MemoryHierarchy,
    hardware: HardwareConfig,
    timing_config: TimingConfig,
    tile_m_values: list[int],
    tile_n_values: list[int],
    tile_k_values: list[int],
    pe_rows_values: list[int],
    pe_cols_values: list[int],
    schedules: list[GemmSchedule] | None = None,
    top_k: int = 5,
    max_candidates: int = MAX_CANDIDATES,
) -> ExplorationResult:
    """Evaluate the Cartesian product of the given candidate sets.

    GEMM, dtype (via gemm), SRAM (hierarchy), clock (timing_config), and
    DRAM bandwidth (hardware) are fixed across the whole search. Candidate
    value lists are deduplicated and sorted ascending before use, so
    duplicate inputs do not evaluate the same candidate twice.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    tile_m = _validate_positive_int_list("tile_m_values", tile_m_values)
    tile_n = _validate_positive_int_list("tile_n_values", tile_n_values)
    tile_k = _validate_positive_int_list("tile_k_values", tile_k_values)
    pe_rows = _validate_positive_int_list("pe_rows_values", pe_rows_values)
    pe_cols = _validate_positive_int_list("pe_cols_values", pe_cols_values)
    schedule_set = _validate_schedules(schedules)

    total_candidates = (
        len(tile_m) * len(tile_n) * len(tile_k) * len(pe_rows) * len(pe_cols) * len(schedule_set)
    )
    if total_candidates > max_candidates:
        raise ValueError(
            f"candidate_count {total_candidates} exceeds the limit of {max_candidates}; "
            "narrow the search (fewer tile/PE candidate values or schedules)"
        )

    feasible_results: list[CandidateResult] = []
    infeasible_count = 0
    infeasible_reasons: list[str] = []

    for tm in tile_m:
        for tn in tile_n:
            for tk in tile_k:
                tile = GemmTile(tile_m=tm, tile_n=tn, tile_k=tk)
                for schedule in schedule_set:
                    try:
                        tiling_result = analyze_tiling(gemm, tile, hierarchy, schedule=schedule)
                    except ValueError as exc:
                        # This tile/schedule is infeasible for every PE
                        # candidate that shares it (PE shape does not affect
                        # tile capacity), so count all of them at once.
                        infeasible_count += len(pe_rows) * len(pe_cols)
                        infeasible_reasons.append(
                            f"tile {tm}x{tn}x{tk} ({schedule.value}): {exc}"
                        )
                        continue

                    for rows in pe_rows:
                        for cols in pe_cols:
                            candidate = CandidateConfig(
                                tile_m=tm,
                                tile_n=tn,
                                tile_k=tk,
                                schedule=schedule,
                                pe_rows=rows,
                                pe_cols=cols,
                            )
                            pe_mapping = map_gemm(gemm, PeArray(rows=rows, columns=cols))
                            timing_result = estimate_execution_time(
                                pe_mapping, tiling_result, hardware, timing_config
                            )
                            feasible_results.append(
                                CandidateResult(
                                    candidate=candidate,
                                    tiling_result=tiling_result,
                                    pe_mapping=pe_mapping,
                                    timing_result=timing_result,
                                )
                            )

    feasible_results.sort(key=_rank_key)

    tied_for_best = 0
    if feasible_results:
        best_time = feasible_results[0].timing_result.perfect_overlap_time_seconds
        tied_for_best = sum(
            1
            for r in feasible_results
            if r.timing_result.perfect_overlap_time_seconds == best_time
        )

    return ExplorationResult(
        total_candidates=total_candidates,
        feasible_candidates=len(feasible_results),
        infeasible_candidates=infeasible_count,
        top_k=top_k,
        tied_for_best=tied_for_best,
        ranked=tuple(feasible_results),
        infeasible_reasons=tuple(infeasible_reasons),
    )
