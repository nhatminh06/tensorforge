"""Analytical mapping of a GEMM output onto a rectangular PE array.

Mapping assumption (documented, not a real dataflow):
    GEMM output C[M,N] is mapped so that
        M -> PE rows
        N -> PE columns
    Each active PE is responsible for exactly one output element C[m,n]
    during a given wave, and computes it as a length-K dot product at
    1 MAC/cycle. If M or N does not evenly divide the array, the array is
    swept in multiple "waves" (ceil(M/rows) x ceil(N/columns) of them),
    and the final row/column of waves is only partially occupied.

This is an idealized spatial-mapping and compute-cycle model. It is NOT a
systolic-array timing model: it ignores array fill/drain latency, operand
propagation, pipeline behavior, memory access, network-on-chip effects,
synchronization, and partial-sum movement. It also does not include any
clock frequency, so results are reported in cycles, not seconds.
"""

from dataclasses import dataclass

from tensorforge.gemm import Gemm


@dataclass(frozen=True)
class PeArray:
    """A rectangular array of processing elements, each 1 MAC/cycle."""

    rows: int
    columns: int

    def __post_init__(self) -> None:
        for name, value in (("rows", self.rows), ("columns", self.columns)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

    @property
    def pe_count(self) -> int:
        return self.rows * self.columns


@dataclass(frozen=True)
class PeMappingResult:
    pe_count: int
    row_waves: int
    column_waves: int
    waves: int
    useful_pe_slots: int
    available_pe_slots: int
    unused_pe_slots: int
    spatial_utilization: float
    cycles_per_wave: int
    compute_cycles: int


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def map_gemm(gemm: Gemm, pe_array: PeArray) -> PeMappingResult:
    """Map GEMM output C[M,N] onto pe_array (M->rows, N->columns).

    See module docstring for the mapping assumption and its limitations.
    """
    row_waves = _ceil_div(gemm.m, pe_array.rows)
    column_waves = _ceil_div(gemm.n, pe_array.columns)
    waves = row_waves * column_waves

    useful_pe_slots = gemm.m * gemm.n
    available_pe_slots = waves * pe_array.pe_count
    unused_pe_slots = available_pe_slots - useful_pe_slots

    spatial_utilization = useful_pe_slots / available_pe_slots

    cycles_per_wave = gemm.k  # 1 MAC/PE/cycle, so K MACs take K cycles
    compute_cycles = waves * cycles_per_wave

    return PeMappingResult(
        pe_count=pe_array.pe_count,
        row_waves=row_waves,
        column_waves=column_waves,
        waves=waves,
        useful_pe_slots=useful_pe_slots,
        available_pe_slots=available_pe_slots,
        unused_pe_slots=unused_pe_slots,
        spatial_utilization=spatial_utilization,
        cycles_per_wave=cycles_per_wave,
        compute_cycles=compute_cycles,
    )
