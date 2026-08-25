"""Minimal hardware model needed for analytical roofline analysis.

Milestone 1 only needs the two limits a roofline model compares against:
    peak_compute_flops_per_second
    memory_bandwidth_bytes_per_second

PE-array geometry, SRAM capacity, clock frequency, and other architectural
details are introduced in later milestones once they actually participate
in a calculation.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareConfig:
    """A generic modeled accelerator, defined only by its roofline limits."""

    peak_compute_flops_per_second: float
    memory_bandwidth_bytes_per_second: float
    name: str = "Example Accelerator"

    def __post_init__(self) -> None:
        _validate_positive_finite(
            "peak_compute_flops_per_second", self.peak_compute_flops_per_second
        )
        _validate_positive_finite(
            "memory_bandwidth_bytes_per_second", self.memory_bandwidth_bytes_per_second
        )


def _validate_positive_finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
