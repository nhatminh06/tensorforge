"""Capacity-constrained GEMM tiling under one fixed loop schedule.

Tile dimensions split the GEMM into Tm x Tn x Tk blocks:

    m_tiles = ceil(M / Tm)
    n_tiles = ceil(N / Tn)
    k_tiles = ceil(K / Tk)

Fixed schedule modeled (a "C-output-tile-resident" schedule — not a
general accelerator dataflow, and not yet compared against alternatives):

    for mi in m_tiles:
        for ni in n_tiles:
            C_tile = zero                      # resident in SRAM
            for ki in k_tiles:
                A_tile = load(A[mi, ki])        # from DRAM
                B_tile = load(B[ki, ni])        # from DRAM
                C_tile += A_tile @ B_tile
            store(C_tile)                       # to DRAM

During one K-tile step, SRAM must simultaneously hold one A tile, one B
tile, and the resident C tile:

    tile_working_set_bytes =
        (Tm*Tk + Tk*Tn + Tm*Tn) * bytes_per_element

using the actual (possibly edge-clipped) tile dimensions of the largest
tile, i.e. tile dimensions clipped to the workload: min(Tm, M), min(Tn,
N), min(Tk, K). Edge tiles transfer only their useful elements — no
padding is modeled or charged.

Traffic under this schedule:
    - Each A[mi,ki] tile is reloaded once per ni (C, not A, stays
      resident), so total A DRAM reads = a_bytes * n_tiles.
    - Each B[ki,ni] tile is reloaded once per mi, so total B DRAM reads =
      b_bytes * m_tiles.
    - Each C[mi,ni] tile is written exactly once (it starts at zero and
      accumulates across the K loop while resident) — there is no C read.

This holds exactly regardless of edge tiling, because summing the useful
bytes of all A (or B) sub-tiles for a fixed reuse context always equals
the full tensor's bytes.

Splitting K changes tile_steps and the tile working set, but — under this
fixed schedule, with Tm/Tn unchanged — does NOT change total DRAM bytes,
because K-tiling does not introduce any additional M/N reuse repetition.

This traffic model is exact only under the modeled schedule above. It is
NOT a claim about optimal, universal, or real-hardware GEMM traffic, and
it does not model memory latency, SRAM->PE traffic, or alternative loop
orders/dataflows.
"""

from dataclasses import dataclass

from tensorforge.gemm import Gemm
from tensorforge.memory import MemoryHierarchy


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
    m_tiles: int
    n_tiles: int
    k_tiles: int
    output_tiles: int
    tile_steps: int

    max_tile_working_set_bytes: int
    tile_fits: bool

    a_dram_read_bytes: int
    b_dram_read_bytes: int
    c_dram_write_bytes: int
    total_dram_bytes: int

    ideal_baseline_dram_bytes: int
    traffic_amplification: float

    effective_arithmetic_intensity: float


def analyze_tiling(gemm: Gemm, tile: GemmTile, hierarchy: MemoryHierarchy) -> TilingResult:
    """Analyze the fixed output-tile-resident schedule for one explicit tile shape.

    Raises ValueError if the tile working set does not fit in the modeled
    SRAM — an invalid tiling configuration is rejected outright, not
    silently shrunk or proceeded with.
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

    a_dram_read_bytes = gemm.a_bytes * n_tiles
    b_dram_read_bytes = gemm.b_bytes * m_tiles
    c_dram_write_bytes = gemm.c_bytes
    total_dram_bytes = a_dram_read_bytes + b_dram_read_bytes + c_dram_write_bytes

    ideal_baseline_dram_bytes = gemm.dram_bytes
    traffic_amplification = total_dram_bytes / ideal_baseline_dram_bytes
    effective_arithmetic_intensity = gemm.flops / total_dram_bytes

    return TilingResult(
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_tiles=k_tiles,
        output_tiles=output_tiles,
        tile_steps=tile_steps,
        max_tile_working_set_bytes=max_tile_working_set_bytes,
        tile_fits=tile_fits,
        a_dram_read_bytes=a_dram_read_bytes,
        b_dram_read_bytes=b_dram_read_bytes,
        c_dram_write_bytes=c_dram_write_bytes,
        total_dram_bytes=total_dram_bytes,
        ideal_baseline_dram_bytes=ideal_baseline_dram_bytes,
        traffic_amplification=traffic_amplification,
        effective_arithmetic_intensity=effective_arithmetic_intensity,
    )
