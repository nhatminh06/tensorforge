"""Analytical end-to-end timing: PE compute time + schedule-specific DRAM transfer time.

This combines two already-modeled quantities that previously had no time
units attached:

    PeMappingResult.compute_cycles   (Milestone 2, pe_array.py)
    TilingResult.total_dram_bytes    (Milestone 4/5, tiling.py — schedule-specific)

Compute time:
    compute_time_seconds = compute_cycles / clock_hz

An idealized PE compute time. It ignores pipeline fill/drain, memory
stalls, NoC, synchronization, and control overhead. `clock_hz` is a new,
separate timing parameter (TimingConfig) — it is not merged into
`PeArray`, which continues to model only array geometry.

Memory time:
    memory_time_seconds = total_dram_bytes / memory_bandwidth_bytes_per_second

`total_dram_bytes` is deliberately the *schedule-specific* traffic from
`TilingResult` (which already reflects tile shape, edge tiles, and the
chosen residency schedule) — not the Milestone-1 idealized `Gemm.dram_bytes`.
This is the first milestone where schedule choice measurably changes an
execution-time estimate. Memory time here is a bandwidth-only transfer
time: it ignores DRAM latency, bank conflicts, burst inefficiency,
controller overhead, and queueing.

Two analytical bounds are reported, not one:
    serialized_time_seconds       = compute_time + memory_time     (no overlap)
    perfect_overlap_time_seconds  = max(compute_time, memory_time)  (full overlap)

Neither is a claim about real hardware scheduling — they bracket the
range between "no overlap at all" and "perfect overlap," with the true
behavior of a specific accelerator's memory/compute scheduler expected to
fall somewhere between them. No partial-overlap model is introduced.

Bottleneck classification compares compute_time and memory_time directly
(analogous in spirit to the Milestone-1 roofline classification, but
using actual modeled cycles/traffic rather than idealized ceilings).

This module does not modify `compute_roofline()`, `PeArray`/`map_gemm()`,
`MemoryHierarchy`/`analyze_memory()`, or the tiling/schedule traffic
formulas — it only combines their outputs.
"""

import math
from dataclasses import dataclass

from tensorforge.hardware import HardwareConfig
from tensorforge.pe_array import PeMappingResult
from tensorforge.tiling import TilingResult

DEFAULT_RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TimingConfig:
    """PE clock frequency, in Hz. Kept separate from PeArray (geometry only)."""

    clock_hz: float

    def __post_init__(self) -> None:
        if not isinstance(self.clock_hz, (int, float)) or isinstance(self.clock_hz, bool):
            raise ValueError(f"clock_hz must be a real number, got {self.clock_hz!r}")
        if math.isnan(self.clock_hz) or math.isinf(self.clock_hz):
            raise ValueError(f"clock_hz must be finite, got {self.clock_hz}")
        if self.clock_hz <= 0:
            raise ValueError(f"clock_hz must be > 0, got {self.clock_hz}")


@dataclass(frozen=True)
class ExecutionTimingResult:
    compute_cycles: int
    clock_hz: float
    compute_time_seconds: float

    dram_bytes: int
    memory_bandwidth_bytes_per_second: float
    memory_time_seconds: float

    serialized_time_seconds: float
    perfect_overlap_time_seconds: float

    bottleneck: str
    memory_to_compute_ratio: float

    pe_implied_peak_flops_per_second: float
    configured_roofline_peak_flops_per_second: float


def estimate_execution_time(
    pe_mapping: PeMappingResult,
    tiling_result: TilingResult,
    hardware: HardwareConfig,
    timing: TimingConfig,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> ExecutionTimingResult:
    compute_cycles = pe_mapping.compute_cycles
    compute_time_seconds = compute_cycles / timing.clock_hz

    dram_bytes = tiling_result.total_dram_bytes
    memory_time_seconds = dram_bytes / hardware.memory_bandwidth_bytes_per_second

    serialized_time_seconds = compute_time_seconds + memory_time_seconds
    perfect_overlap_time_seconds = max(compute_time_seconds, memory_time_seconds)

    if math.isclose(compute_time_seconds, memory_time_seconds, rel_tol=relative_tolerance):
        bottleneck = "balanced"
    elif compute_time_seconds > memory_time_seconds:
        bottleneck = "compute-bound"
    else:
        bottleneck = "memory-bound"

    memory_to_compute_ratio = memory_time_seconds / compute_time_seconds

    # 1 MAC/PE/cycle, 1 MAC = 2 FLOPs (same convention as gemm.py/pe_array.py).
    pe_implied_peak_flops_per_second = 2 * pe_mapping.pe_count * timing.clock_hz

    return ExecutionTimingResult(
        compute_cycles=compute_cycles,
        clock_hz=timing.clock_hz,
        compute_time_seconds=compute_time_seconds,
        dram_bytes=dram_bytes,
        memory_bandwidth_bytes_per_second=hardware.memory_bandwidth_bytes_per_second,
        memory_time_seconds=memory_time_seconds,
        serialized_time_seconds=serialized_time_seconds,
        perfect_overlap_time_seconds=perfect_overlap_time_seconds,
        bottleneck=bottleneck,
        memory_to_compute_ratio=memory_to_compute_ratio,
        pe_implied_peak_flops_per_second=pe_implied_peak_flops_per_second,
        configured_roofline_peak_flops_per_second=hardware.peak_compute_flops_per_second,
    )
