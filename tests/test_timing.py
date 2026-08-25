import math

import pytest

from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray, map_gemm
from tensorforge.tiling import GemmSchedule, GemmTile, analyze_tiling
from tensorforge.timing import TimingConfig, estimate_execution_time


@pytest.mark.parametrize("clock_hz", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_clock(clock_hz):
    with pytest.raises(ValueError):
        TimingConfig(clock_hz=clock_hz)


def test_known_compute_time():
    # M=2, N=3, K=4 on a 1x1 PE array: waves=6, compute_cycles=24 (Milestone 2).
    gemm = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    pe_array = PeArray(rows=1, columns=1)
    mapping = map_gemm(gemm, pe_array)
    assert mapping.compute_cycles == 24

    tile = GemmTile(tile_m=2, tile_n=3, tile_k=4)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    tiling = analyze_tiling(gemm, tile, hierarchy)

    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1e12)
    timing = TimingConfig(clock_hz=2e9)  # 2 GHz

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    # 24 cycles / 2e9 Hz = 12 ns
    assert math.isclose(result.compute_time_seconds, 12e-9)


def test_known_memory_time():
    # M=N=K=4, FP32, tile 2x2x2, C-resident -> 320 B (Milestone 4/5 regression contract).
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)
    tiling = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.C_RESIDENT)
    assert tiling.total_dram_bytes == 320

    pe_array = PeArray(rows=2, columns=2)
    mapping = map_gemm(gemm, pe_array)

    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=100e9)
    timing = TimingConfig(clock_hz=1e9)

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    # 320 B / 100e9 B/s = 3.2 ns
    assert math.isclose(result.memory_time_seconds, 3.2e-9)


def test_combined_known_value_serialized_and_perfect_overlap():
    gemm = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    pe_array = PeArray(rows=1, columns=1)
    mapping = map_gemm(gemm, pe_array)
    assert mapping.compute_cycles == 24

    tile = GemmTile(tile_m=2, tile_n=3, tile_k=4)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    tiling = analyze_tiling(gemm, tile, hierarchy)
    assert tiling.total_dram_bytes == gemm.dram_bytes == 104

    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1e9)
    timing = TimingConfig(clock_hz=2e9)

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    expected_compute = 24 / 2e9  # 12 ns
    expected_memory = 104 / 1e9  # 104 ns

    assert math.isclose(result.compute_time_seconds, expected_compute)
    assert math.isclose(result.memory_time_seconds, expected_memory)
    assert math.isclose(result.serialized_time_seconds, expected_compute + expected_memory)
    assert math.isclose(result.perfect_overlap_time_seconds, max(expected_compute, expected_memory))
    assert result.bottleneck == "memory-bound"


def test_compute_bound_classification():
    gemm = Gemm(m=64, n=64, k=64, dtype=DType.FP32)
    pe_array = PeArray(rows=1, columns=1)  # tiny array -> huge compute_cycles
    mapping = map_gemm(gemm, pe_array)

    tile = GemmTile(tile_m=64, tile_n=64, tile_k=64)
    hierarchy = MemoryHierarchy(sram_bytes=10_000_000)
    tiling = analyze_tiling(gemm, tile, hierarchy)

    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1e15)
    timing = TimingConfig(clock_hz=1e12)  # fast clock, but tiny PE array dominates via cycles

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    assert result.compute_time_seconds > result.memory_time_seconds
    assert result.bottleneck == "compute-bound"
    assert result.perfect_overlap_time_seconds == result.compute_time_seconds


def test_memory_bound_classification():
    gemm = Gemm(m=64, n=64, k=64, dtype=DType.FP32)
    pe_array = PeArray(rows=64, columns=64)  # huge array -> few compute cycles
    mapping = map_gemm(gemm, pe_array)

    tile = GemmTile(tile_m=64, tile_n=64, tile_k=64)
    hierarchy = MemoryHierarchy(sram_bytes=10_000_000)
    tiling = analyze_tiling(gemm, tile, hierarchy)

    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1e3)
    timing = TimingConfig(clock_hz=1e12)

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    assert result.memory_time_seconds > result.compute_time_seconds
    assert result.bottleneck == "memory-bound"
    assert result.perfect_overlap_time_seconds == result.memory_time_seconds


def test_balanced_classification():
    gemm = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    pe_array = PeArray(rows=1, columns=1)
    mapping = map_gemm(gemm, pe_array)
    assert mapping.compute_cycles == 24

    tile = GemmTile(tile_m=2, tile_n=3, tile_k=4)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    tiling = analyze_tiling(gemm, tile, hierarchy)
    assert tiling.total_dram_bytes == 104

    # Choose bandwidth/clock so compute_time == memory_time exactly:
    # compute_time = 24/clock_hz, memory_time = 104/bandwidth.
    # Pick clock_hz=24, bandwidth=104 -> both equal 1 second.
    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=104.0)
    timing = TimingConfig(clock_hz=24.0)

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    assert math.isclose(result.compute_time_seconds, result.memory_time_seconds)
    assert result.bottleneck == "balanced"


def test_schedule_changes_memory_time_not_compute_time():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)
    pe_array = PeArray(rows=2, columns=2)
    mapping = map_gemm(gemm, pe_array)

    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=100e9)
    timing = TimingConfig(clock_hz=1e9)

    c_tiling = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.C_RESIDENT)
    a_tiling = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.A_RESIDENT)

    c_result = estimate_execution_time(mapping, c_tiling, hardware, timing)
    a_result = estimate_execution_time(mapping, a_tiling, hardware, timing)

    assert c_result.compute_cycles == a_result.compute_cycles
    assert c_result.compute_time_seconds == a_result.compute_time_seconds
    assert c_result.memory_time_seconds != a_result.memory_time_seconds
    assert c_result.dram_bytes == 320
    assert a_result.dram_bytes == 384


def test_pe_array_changes_compute_time_not_schedule_traffic():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)
    tiling = analyze_tiling(gemm, tile, hierarchy)  # schedule-independent of PE array

    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=100e9)
    timing = TimingConfig(clock_hz=1e9)

    small_array = map_gemm(gemm, PeArray(rows=1, columns=1))
    large_array = map_gemm(gemm, PeArray(rows=4, columns=4))

    small_result = estimate_execution_time(small_array, tiling, hardware, timing)
    large_result = estimate_execution_time(large_array, tiling, hardware, timing)

    assert small_result.dram_bytes == large_result.dram_bytes == tiling.total_dram_bytes
    assert small_result.compute_cycles != large_result.compute_cycles
    assert small_result.compute_time_seconds != large_result.compute_time_seconds


def test_bandwidth_scaling_inverse():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)
    tiling = analyze_tiling(gemm, tile, hierarchy)
    pe_array = PeArray(rows=2, columns=2)
    mapping = map_gemm(gemm, pe_array)
    timing = TimingConfig(clock_hz=1e9)

    results = []
    for bandwidth_gbps in (50e9, 100e9, 200e9):
        hardware = HardwareConfig(
            peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=bandwidth_gbps
        )
        results.append(estimate_execution_time(mapping, tiling, hardware, timing))

    for r in results:
        assert math.isclose(r.memory_time_seconds * (r.memory_bandwidth_bytes_per_second), tiling.total_dram_bytes)

    assert results[0].memory_time_seconds > results[1].memory_time_seconds > results[2].memory_time_seconds
    # Compute time is unaffected by bandwidth.
    assert results[0].compute_time_seconds == results[1].compute_time_seconds == results[2].compute_time_seconds


def test_clock_scaling_inverse():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)
    tiling = analyze_tiling(gemm, tile, hierarchy)
    pe_array = PeArray(rows=2, columns=2)
    mapping = map_gemm(gemm, pe_array)
    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=100e9)

    results = []
    for clock_hz in (0.5e9, 1e9, 2e9):
        timing = TimingConfig(clock_hz=clock_hz)
        results.append(estimate_execution_time(mapping, tiling, hardware, timing))

    assert results[0].compute_time_seconds > results[1].compute_time_seconds > results[2].compute_time_seconds
    # Memory time is unaffected by clock.
    assert results[0].memory_time_seconds == results[1].memory_time_seconds == results[2].memory_time_seconds


def test_traffic_result_consistency():
    gemm = Gemm(m=6, n=10, k=9, dtype=DType.FP16)
    tile = GemmTile(tile_m=3, tile_n=4, tile_k=3)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    tiling = analyze_tiling(gemm, tile, hierarchy)
    pe_array = PeArray(rows=3, columns=4)
    mapping = map_gemm(gemm, pe_array)
    hardware = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=100e9)
    timing = TimingConfig(clock_hz=1e9)

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    assert result.compute_cycles == mapping.compute_cycles
    assert result.dram_bytes == tiling.total_dram_bytes
    assert result.serialized_time_seconds == result.compute_time_seconds + result.memory_time_seconds
    assert result.perfect_overlap_time_seconds == max(
        result.compute_time_seconds, result.memory_time_seconds
    )
    assert math.isclose(
        result.memory_to_compute_ratio, result.memory_time_seconds / result.compute_time_seconds
    )


def test_pe_implied_peak_and_configured_roofline_peak_independent():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)
    tiling = analyze_tiling(gemm, tile, hierarchy)
    pe_array = PeArray(rows=4, columns=4)
    mapping = map_gemm(gemm, pe_array)

    # Deliberately mismatched: PE-implied peak != configured roofline peak.
    hardware = HardwareConfig(peak_compute_flops_per_second=999e12, memory_bandwidth_bytes_per_second=100e9)
    timing = TimingConfig(clock_hz=1e9)

    result = estimate_execution_time(mapping, tiling, hardware, timing)

    # pe_implied = 2 * 16 PEs * 1e9 Hz = 3.2e10 FLOP/s
    assert math.isclose(result.pe_implied_peak_flops_per_second, 3.2e10)
    assert result.configured_roofline_peak_flops_per_second == 999e12
    assert result.pe_implied_peak_flops_per_second != result.configured_roofline_peak_flops_per_second
