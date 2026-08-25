"""Capacity analysis for an explicit DRAM -> SRAM -> compute hierarchy.

Modeled hierarchy for this milestone:

    DRAM
      |
    Global SRAM        (capacity_bytes, this module)
      |
    compute array       (PE mapping, see pe_array.py)

This milestone models only SRAM *capacity* and the same idealized baseline
DRAM traffic already used by Gemm.dram_bytes (read A once, read B once,
write C once). It does NOT model SRAM latency/bandwidth, SRAM-to-PE
traffic, bank conflicts, or any cache behavior.

Working set:
    working_set_bytes = a_bytes + b_bytes + c_bytes

Fit is a capacity fact, not a reuse guarantee: a tensor "fitting" means it
could theoretically reside in SRAM simultaneously with the others under
this simplified model, not that it is only ever read once from SRAM by
the compute array (that depends on tiling/dataflow, not yet modeled).

If the full working set does not fit, the exact extra DRAM traffic caused
by refetching is NOT calculable yet: it depends on tile shape, loop
order, and which tensor(s) are kept resident, none of which exist in this
milestone. We only expose a `tiling_required` fact, not a traffic number.
"""

from dataclasses import dataclass

from tensorforge.gemm import Gemm


@dataclass(frozen=True)
class MemoryHierarchy:
    """Capacity of the modeled global SRAM boundary between DRAM and compute."""

    sram_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.sram_bytes, int) or isinstance(self.sram_bytes, bool):
            raise ValueError(f"sram_bytes must be an int, got {self.sram_bytes!r}")
        if self.sram_bytes <= 0:
            raise ValueError(f"sram_bytes must be > 0, got {self.sram_bytes}")


@dataclass(frozen=True)
class MemoryResult:
    sram_bytes: int

    a_bytes: int
    b_bytes: int
    c_bytes: int
    working_set_bytes: int

    a_fits: bool
    b_fits: bool
    c_fits: bool
    full_working_set_fits: bool

    capacity_headroom_bytes: int
    capacity_deficit_bytes: int
    working_set_to_capacity_ratio: float

    tiling_required: bool

    dram_read_bytes: int
    dram_write_bytes: int
    total_dram_bytes: int


def analyze_memory(gemm: Gemm, hierarchy: MemoryHierarchy) -> MemoryResult:
    """Compare the GEMM working set against modeled SRAM capacity.

    Also reports the same idealized baseline DRAM traffic as
    Gemm.dram_bytes (A+B read, C written) — this milestone does not change
    that traffic based on SRAM fit. See module docstring for why.
    """
    sram_bytes = hierarchy.sram_bytes
    a_bytes, b_bytes, c_bytes = gemm.a_bytes, gemm.b_bytes, gemm.c_bytes
    working_set_bytes = a_bytes + b_bytes + c_bytes

    full_working_set_fits = working_set_bytes <= sram_bytes

    capacity_headroom_bytes = max(0, sram_bytes - working_set_bytes)
    capacity_deficit_bytes = max(0, working_set_bytes - sram_bytes)

    dram_read_bytes = a_bytes + b_bytes
    dram_write_bytes = c_bytes

    return MemoryResult(
        sram_bytes=sram_bytes,
        a_bytes=a_bytes,
        b_bytes=b_bytes,
        c_bytes=c_bytes,
        working_set_bytes=working_set_bytes,
        a_fits=a_bytes <= sram_bytes,
        b_fits=b_bytes <= sram_bytes,
        c_fits=c_bytes <= sram_bytes,
        full_working_set_fits=full_working_set_fits,
        capacity_headroom_bytes=capacity_headroom_bytes,
        capacity_deficit_bytes=capacity_deficit_bytes,
        working_set_to_capacity_ratio=working_set_bytes / sram_bytes,
        tiling_required=not full_working_set_fits,
        dram_read_bytes=dram_read_bytes,
        dram_write_bytes=dram_write_bytes,
        total_dram_bytes=dram_read_bytes + dram_write_bytes,
    )
