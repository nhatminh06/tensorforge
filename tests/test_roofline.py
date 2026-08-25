import math

import pytest

from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.roofline import compute_roofline

# Shared workload: M=2, N=3, K=4, FP32
# FLOPs = 48, DRAM bytes = 104 (see test_gemm.py for the hand derivation)
# AI = 48 / 104 = 6/13 FLOP/byte
GEMM = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
EXPECTED_AI = 48 / 104


def test_arithmetic_intensity_known_value():
    hardware = HardwareConfig(
        peak_compute_flops_per_second=1e12,
        memory_bandwidth_bytes_per_second=1e9,
    )
    result = compute_roofline(GEMM, hardware)
    assert math.isclose(result.arithmetic_intensity, EXPECTED_AI, rel_tol=1e-12)


def test_memory_bound_classification():
    # ridge point = 1e12 / 1e9 = 1000 FLOP/byte, far above AI (~0.4615)
    hardware = HardwareConfig(
        peak_compute_flops_per_second=1e12,
        memory_bandwidth_bytes_per_second=1e9,
    )
    result = compute_roofline(GEMM, hardware)

    assert result.classification == "memory-bound"
    expected_memory_ceiling = EXPECTED_AI * 1e9
    assert math.isclose(result.memory_ceiling_flops_per_second, expected_memory_ceiling)
    assert result.attainable_flops_per_second < result.compute_ceiling_flops_per_second

    expected_memory_time = GEMM.dram_bytes / 1e9
    expected_compute_time = GEMM.flops / 1e12
    assert math.isclose(result.memory_time_seconds, expected_memory_time)
    assert math.isclose(result.compute_time_seconds, expected_compute_time)
    assert result.estimated_time_seconds == max(expected_compute_time, expected_memory_time)
    assert result.estimated_time_seconds == result.memory_time_seconds


def test_compute_bound_classification():
    # ridge point = 100 / 1e12 = 1e-10 FLOP/byte, far below AI (~0.4615)
    hardware = HardwareConfig(
        peak_compute_flops_per_second=100.0,
        memory_bandwidth_bytes_per_second=1e12,
    )
    result = compute_roofline(GEMM, hardware)

    assert result.classification == "compute-bound"
    assert result.attainable_flops_per_second == pytest.approx(100.0)

    expected_memory_time = GEMM.dram_bytes / 1e12
    expected_compute_time = GEMM.flops / 100.0
    assert result.estimated_time_seconds == max(expected_compute_time, expected_memory_time)
    assert result.estimated_time_seconds == result.compute_time_seconds


def test_ridge_point_formula():
    hardware = HardwareConfig(
        peak_compute_flops_per_second=2e12,
        memory_bandwidth_bytes_per_second=4e8,
    )
    result = compute_roofline(GEMM, hardware)
    assert math.isclose(result.ridge_point, 2e12 / 4e8)
    assert result.arithmetic_intensity < result.ridge_point
    assert result.classification == "memory-bound"


def test_balanced_classification_at_equality():
    # memory_ceiling = AI * bandwidth = compute_ceiling exactly
    hardware = HardwareConfig(
        peak_compute_flops_per_second=6.0,
        memory_bandwidth_bytes_per_second=13.0,
    )
    result = compute_roofline(GEMM, hardware)
    assert math.isclose(result.memory_ceiling_flops_per_second, 6.0)
    assert result.classification == "balanced"


@pytest.mark.parametrize(
    "peak,bandwidth",
    [(0, 1e9), (-1, 1e9), (1e12, 0), (1e12, -1), (float("nan"), 1e9), (float("inf"), 1e9)],
)
def test_hardware_rejects_invalid_values(peak, bandwidth):
    with pytest.raises(ValueError):
        HardwareConfig(peak_compute_flops_per_second=peak, memory_bandwidth_bytes_per_second=bandwidth)
