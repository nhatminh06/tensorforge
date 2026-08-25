"""Capacity-constrained GEMM tiling under one of three explicit loop schedules.

Tile dimensions split the GEMM into Tm x Tn x Tk blocks:

    m_tiles = ceil(M / Tm)
    n_tiles = ceil(N / Tn)
    k_tiles = ceil(K / Tk)

Three explicit residency schedules are modeled (GemmSchedule). Each keeps
a different operand tile resident in SRAM across the innermost reuse loop;
the other two tensors are loaded from DRAM. These are literal loop-order
descriptions, not claims that they implement general "output stationary" /
"input stationary" / "weight stationary" hardware dataflows. See
docs/schedules.md for the full derivation and a comparison table.

C_RESIDENT (the original Milestone-4 schedule):

    for mi in m_tiles:
        for ni in n_tiles:
            C_tile = zero                      # resident in SRAM
            for ki in k_tiles:
                A_tile = load(A[mi, ki])        # from DRAM
                B_tile = load(B[ki, ni])        # from DRAM
                C_tile += A_tile @ B_tile
            store(C_tile)                       # to DRAM

    A reads = a_bytes * n_tiles   (A reloaded once per ni; C, not A, is resident)
    B reads = b_bytes * m_tiles   (B reloaded once per mi; C, not B, is resident)
    C reads = 0                   (C accumulates fully resident, no partial spill)
    C writes = c_bytes            (each output tile written once, when complete)

A_RESIDENT:

    for mi in m_tiles:
        for ki in k_tiles:
            A_tile = load(A[mi, ki])            # resident in SRAM
            for ni in n_tiles:
                B_tile = load(B[ki, ni])        # from DRAM
                C_tile = load(C[mi, ni]) or zero  # from DRAM, unless first K step
                C_tile += A_tile @ B_tile
                store(C_tile)                    # partial or final, to DRAM

B_RESIDENT (symmetric to A_RESIDENT):

    for ni in n_tiles:
        for ki in k_tiles:
            B_tile = load(B[ki, ni])            # resident in SRAM
            for mi in m_tiles:
                A_tile = load(A[mi, ki])        # from DRAM
                C_tile = load(C[mi, ni]) or zero  # from DRAM, unless first K step
                C_tile += A_tile @ B_tile
                store(C_tile)                    # partial or final, to DRAM

Because K is no longer the innermost loop in A_RESIDENT/B_RESIDENT, each
output tile C[mi,ni] is visited once per K tile rather than staying
resident across all of K — its partial sum must cross DRAM between
visits. Per output tile: k_tiles writes (every visit, partial or final),
and k_tiles - 1 reads (every visit after the first). Summed over all
output tiles (which exactly partition C):

    C writes = c_bytes * k_tiles
    C reads  = c_bytes * (k_tiles - 1)

A_RESIDENT reads A once (a_bytes total, no repetition) but repeats B
across m_tiles; B_RESIDENT reads B once but repeats A across n_tiles.
Both trade input-tensor reuse for partial-sum DRAM traffic — which
schedule wins depends on the workload's M/N/K shape.

Capacity requirement (same for all three schedules — one A tile, one B
tile, and one C tile must be simultaneously resident during a compute
step):

    tile_working_set_bytes = (Tm*Tk + Tk*Tn + Tm*Tn) * bytes_per_element

using tile dimensions clipped to the workload: min(Tm, M), min(Tn, N),
min(Tk, K). Edge tiles transfer only their useful elements — no padding
is modeled or charged. These closed forms hold exactly regardless of edge
tiling, because summing the useful bytes of all sub-tiles for a fixed
reuse context always equals the full tensor's bytes.

For C_RESIDENT, splitting K changes tile_steps and the tile working set
but — with Tm/Tn fixed — does NOT change total DRAM bytes. For
A_RESIDENT/B_RESIDENT, splitting K directly increases C partial-sum
traffic (more k_tiles means more read/write round-trips per output tile).

This traffic model is exact only under the modeled schedules above. It is
NOT a claim about optimal, universal, or real-hardware GEMM traffic, and
it does not model memory latency, SRAM->PE traffic, or an exhaustive set
of loop orders/dataflows.
"""

from dataclasses import dataclass
from enum import Enum

from tensorforge.gemm import Gemm
from tensorforge.memory import MemoryHierarchy


class GemmSchedule(Enum):
    """Which operand tile stays resident in SRAM across the innermost reuse loop."""

    C_RESIDENT = "c-resident"
    A_RESIDENT = "a-resident"
    B_RESIDENT = "b-resident"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class GemmTile:
    """A GEMM tile shape (Tm, Tn, Tk). May exceed the GEMM's own dimensions."""

    tile_m: int
    tile_n: int
    tile_k: int

    def __post_init__(self) -> None:
        for name, value in (
            ("tile_m", self.tile_m),
            ("tile_n", self.tile_n),
            ("tile_k", self.tile_k),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class TilingResult:
    schedule: "GemmSchedule"

    m_tiles: int
    n_tiles: int
    k_tiles: int
    output_tiles: int
    tile_steps: int

    max_tile_working_set_bytes: int
    tile_fits: bool

    a_dram_read_bytes: int
    b_dram_read_bytes: int
    c_dram_read_bytes: int
    c_dram_write_bytes: int
    total_dram_bytes: int

    ideal_baseline_dram_bytes: int
    traffic_amplification: float

    effective_arithmetic_intensity: float


def analyze_tiling(
    gemm: Gemm,
    tile: GemmTile,
    hierarchy: MemoryHierarchy,
    schedule: GemmSchedule = GemmSchedule.C_RESIDENT,
) -> TilingResult:
    """Analyze one explicit tile shape under one explicit residency schedule.

    Defaults to C_RESIDENT (the original Milestone-4 schedule) so existing
    callers are unaffected. Raises ValueError if the tile working set does
    not fit in the modeled SRAM — an invalid tiling configuration is
    rejected outright, not silently shrunk or proceeded with.
    """
    bytes_per_element = gemm.dtype.bytes_per_element

    m_tiles = _ceil_div(gemm.m, tile.tile_m)
    n_tiles = _ceil_div(gemm.n, tile.tile_n)
    k_tiles = _ceil_div(gemm.k, tile.tile_k)
    output_tiles = m_tiles * n_tiles
    tile_steps = output_tiles * k_tiles

    # Largest tile occurs at index 0 along each axis: its dimensions are
    # the tile dimensions clipped to the workload (no padding charged).
    max_tm = min(tile.tile_m, gemm.m)
    max_tn = min(tile.tile_n, gemm.n)
    max_tk = min(tile.tile_k, gemm.k)
    max_tile_working_set_bytes = (
        max_tm * max_tk + max_tk * max_tn + max_tm * max_tn
    ) * bytes_per_element

    tile_fits = max_tile_working_set_bytes <= hierarchy.sram_bytes
    if not tile_fits:
        raise ValueError(
            f"tile working set requires {max_tile_working_set_bytes} B "
            f"but SRAM capacity is {hierarchy.sram_bytes} B"
        )

    if schedule is GemmSchedule.C_RESIDENT:
        a_dram_read_bytes = gemm.a_bytes * n_tiles
        b_dram_read_bytes = gemm.b_bytes * m_tiles
        c_dram_read_bytes = 0
        c_dram_write_bytes = gemm.c_bytes
    elif schedule is GemmSchedule.A_RESIDENT:
        a_dram_read_bytes = gemm.a_bytes
        b_dram_read_bytes = gemm.b_bytes * m_tiles
        c_dram_read_bytes = gemm.c_bytes * (k_tiles - 1)
        c_dram_write_bytes = gemm.c_bytes * k_tiles
    elif schedule is GemmSchedule.B_RESIDENT:
        a_dram_read_bytes = gemm.a_bytes * n_tiles
        b_dram_read_bytes = gemm.b_bytes
        c_dram_read_bytes = gemm.c_bytes * (k_tiles - 1)
        c_dram_write_bytes = gemm.c_bytes * k_tiles
    else:
        raise ValueError(f"unknown schedule: {schedule!r}")

    total_dram_bytes = (
        a_dram_read_bytes + b_dram_read_bytes + c_dram_read_bytes + c_dram_write_bytes
    )

    ideal_baseline_dram_bytes = gemm.dram_bytes
    traffic_amplification = total_dram_bytes / ideal_baseline_dram_bytes
    effective_arithmetic_intensity = gemm.flops / total_dram_bytes

    return TilingResult(
        schedule=schedule,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_tiles=k_tiles,
        output_tiles=output_tiles,
        tile_steps=tile_steps,
        max_tile_working_set_bytes=max_tile_working_set_bytes,
        tile_fits=tile_fits,
        a_dram_read_bytes=a_dram_read_bytes,
        b_dram_read_bytes=b_dram_read_bytes,
        c_dram_read_bytes=c_dram_read_bytes,
        c_dram_write_bytes=c_dram_write_bytes,
        total_dram_bytes=total_dram_bytes,
        ideal_baseline_dram_bytes=ideal_baseline_dram_bytes,
        traffic_amplification=traffic_amplification,
        effective_arithmetic_intensity=effective_arithmetic_intensity,
    )


@dataclass(frozen=True)
class ScheduleComparison:
    c_resident: TilingResult
    a_resident: TilingResult
    b_resident: TilingResult
    best_among_modeled_schedules: tuple[GemmSchedule, ...]


def compare_schedules(
    gemm: Gemm, tile: GemmTile, hierarchy: MemoryHierarchy
) -> ScheduleComparison:
    """Analyze the same GEMM/tile/SRAM under all three modeled schedules.

    `best_among_modeled_schedules` lists every schedule tied for the
    lowest total_dram_bytes among the three modeled here — it is not a
    claim of global optimality, only the best among these three.
    """
    results = {
        schedule: analyze_tiling(gemm, tile, hierarchy, schedule=schedule)
        for schedule in (
            GemmSchedule.C_RESIDENT,
            GemmSchedule.A_RESIDENT,
            GemmSchedule.B_RESIDENT,
        )
    }
    min_traffic = min(result.total_dram_bytes for result in results.values())
    best = tuple(
        schedule for schedule, result in results.items() if result.total_dram_bytes == min_traffic
    )

    return ScheduleComparison(
        c_resident=results[GemmSchedule.C_RESIDENT],
        a_resident=results[GemmSchedule.A_RESIDENT],
        b_resident=results[GemmSchedule.B_RESIDENT],
        best_among_modeled_schedules=best,
    )
