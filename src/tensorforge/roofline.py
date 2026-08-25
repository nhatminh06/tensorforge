"""Analytical roofline model.

    arithmetic_intensity = FLOPs / DRAM bytes                  [FLOP/byte]
    ridge_point           = peak_compute / memory_bandwidth     [FLOP/byte]
    memory_ceiling        = arithmetic_intensity * memory_bandwidth   [FLOP/s]
    compute_ceiling       = peak_compute                              [FLOP/s]
    attainable_performance = min(compute_ceiling, memory_ceiling)     [FLOP/s]

Classification (equality handled with a relative tolerance):
    memory_ceiling < compute_ceiling  -> "memory-bound"
    memory_ceiling > compute_ceiling  -> "compute-bound"
    otherwise                          -> "balanced"

Time estimate (analytical lower bound, perfect overlap assumed):
    compute_time   = FLOPs / peak_compute                 [s]
    memory_time    = DRAM bytes / memory_bandwidth         [s]
    estimated_time = max(compute_time, memory_time)        [s]

This is NOT a measured latency. It does not model queueing, cache misses,
synchronization, pipeline startup, or any other real hardware overhead.
"""

import math
from dataclasses import dataclass

from tensorforge.gemm import Gemm
from tensorforge.hardware import HardwareConfig

DEFAULT_RELATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RooflineResult:
    arithmetic_intensity: float
    ridge_point: float
    compute_ceiling_flops_per_second: float
    memory_ceiling_flops_per_second: float
    attainable_flops_per_second: float
    compute_time_seconds: float
    memory_time_seconds: float
    estimated_time_seconds: float
    classification: str


def compute_roofline(
    gemm: Gemm,
    hardware: HardwareConfig,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
) -> RooflineResult:
    flops = gemm.flops
    dram_bytes = gemm.dram_bytes

    arithmetic_intensity = flops / dram_bytes

    compute_ceiling = hardware.peak_compute_flops_per_second
    ridge_point = compute_ceiling / hardware.memory_bandwidth_bytes_per_second
    memory_ceiling = arithmetic_intensity * hardware.memory_bandwidth_bytes_per_second
    attainable = min(compute_ceiling, memory_ceiling)

    compute_time = flops / compute_ceiling
    memory_time = dram_bytes / hardware.memory_bandwidth_bytes_per_second
    estimated_time = max(compute_time, memory_time)

    if math.isclose(memory_ceiling, compute_ceiling, rel_tol=relative_tolerance):
        classification = "balanced"
    elif memory_ceiling < compute_ceiling:
        classification = "memory-bound"
    else:
        classification = "compute-bound"

    return RooflineResult(
        arithmetic_intensity=arithmetic_intensity,
        ridge_point=ridge_point,
        compute_ceiling_flops_per_second=compute_ceiling,
        memory_ceiling_flops_per_second=memory_ceiling,
        attainable_flops_per_second=attainable,
        compute_time_seconds=compute_time,
        memory_time_seconds=memory_time,
        estimated_time_seconds=estimated_time,
        classification=classification,
    )
