import math

import pytest

from tensorforge.gemm import DType, Gemm
from tensorforge.memory import MemoryHierarchy
from tensorforge.tiling import GemmSchedule, GemmTile, analyze_tiling, compare_schedules


def test_c_resident_default_matches_milestone_4():
    # Explicit schedule=C_RESIDENT must match the no-schedule-arg default
    # exactly (Milestone-4 regression contract: 4x4x4/2x2x2 -> 320B).
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)

    default_result = analyze_tiling(gemm, tile, hierarchy)
    explicit_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.C_RESIDENT)

    assert default_result == explicit_result
    assert default_result.a_dram_read_bytes == 128
    assert default_result.b_dram_read_bytes == 128
    assert default_result.c_dram_read_bytes == 0
    assert default_result.c_dram_write_bytes == 64
    assert default_result.total_dram_bytes == 320


def test_known_value_a_resident_4x4x4_tile_2x2x2():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)

    result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.A_RESIDENT)

    assert result.a_dram_read_bytes == 64  # A read once, no repetition
    assert result.b_dram_read_bytes == 128  # B bytes * m_tiles(2)
    assert result.c_dram_read_bytes == 64  # C bytes * (k_tiles(2) - 1)
    assert result.c_dram_write_bytes == 128  # C bytes * k_tiles(2)
    assert result.total_dram_bytes == 384

    assert math.isclose(result.effective_arithmetic_intensity, 128 / 384)
    assert math.isclose(result.effective_arithmetic_intensity, 1 / 3)


def test_known_value_b_resident_4x4x4_tile_2x2x2():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)

    result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.B_RESIDENT)

    assert result.a_dram_read_bytes == 128  # A bytes * n_tiles(2)
    assert result.b_dram_read_bytes == 64  # B read once, no repetition
    assert result.c_dram_read_bytes == 64  # C bytes * (k_tiles(2) - 1)
    assert result.c_dram_write_bytes == 128  # C bytes * k_tiles(2)
    assert result.total_dram_bytes == 384

    assert math.isclose(result.effective_arithmetic_intensity, 1 / 3)


@pytest.mark.parametrize(
    "schedule",
    [GemmSchedule.C_RESIDENT, GemmSchedule.A_RESIDENT, GemmSchedule.B_RESIDENT],
)
def test_traffic_component_sum_invariant(schedule):
    gemm = Gemm(m=6, n=10, k=9, dtype=DType.FP16)
    tile = GemmTile(tile_m=3, tile_n=4, tile_k=3)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result = analyze_tiling(gemm, tile, hierarchy, schedule=schedule)

    assert result.total_dram_bytes == (
        result.a_dram_read_bytes
        + result.b_dram_read_bytes
        + result.c_dram_read_bytes
        + result.c_dram_write_bytes
    )


@pytest.mark.parametrize(
    "schedule",
    [GemmSchedule.C_RESIDENT, GemmSchedule.A_RESIDENT, GemmSchedule.B_RESIDENT],
)
def test_schedule_does_not_change_gemm_operations(schedule):
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)

    analyze_tiling(gemm, tile, hierarchy, schedule=schedule)

    assert gemm.macs == 64
    assert gemm.flops == 128


def test_full_residency_equal_across_schedules():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=4)  # m_tiles=n_tiles=k_tiles=1
    hierarchy = MemoryHierarchy(sram_bytes=192)

    c_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.C_RESIDENT)
    a_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.A_RESIDENT)
    b_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.B_RESIDENT)

    for result in (c_result, a_result, b_result):
        assert result.total_dram_bytes == gemm.dram_bytes == 192
        assert result.c_dram_read_bytes == 0
        assert result.traffic_amplification == 1.0


def test_k_tiles_one_no_partial_sum_traffic():
    gemm = Gemm(m=8, n=8, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=4)  # k_tiles = 1
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    for schedule in (GemmSchedule.A_RESIDENT, GemmSchedule.B_RESIDENT):
        result = analyze_tiling(gemm, tile, hierarchy, schedule=schedule)
        assert result.k_tiles == 1
        assert result.c_dram_read_bytes == 0
        assert result.c_dram_write_bytes == gemm.c_bytes


def test_increasing_k_tiles_increases_partial_sum_traffic():
    gemm = Gemm(m=8, n=8, k=32, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    few_k_tiles = analyze_tiling(
        gemm, GemmTile(tile_m=4, tile_n=4, tile_k=16), hierarchy, schedule=GemmSchedule.A_RESIDENT
    )
    many_k_tiles = analyze_tiling(
        gemm, GemmTile(tile_m=4, tile_n=4, tile_k=4), hierarchy, schedule=GemmSchedule.A_RESIDENT
    )

    assert many_k_tiles.k_tiles > few_k_tiles.k_tiles
    assert many_k_tiles.c_dram_read_bytes > few_k_tiles.c_dram_read_bytes
    assert many_k_tiles.c_dram_write_bytes > few_k_tiles.c_dram_write_bytes
    assert many_k_tiles.total_dram_bytes > few_k_tiles.total_dram_bytes


def test_c_resident_tk_invariance_still_holds():
    # Regression: C_RESIDENT traffic must stay independent of Tk (Milestone-4 result).
    gemm = Gemm(m=8, n=8, k=32, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result_tk8 = analyze_tiling(
        gemm, GemmTile(tile_m=4, tile_n=4, tile_k=8), hierarchy, schedule=GemmSchedule.C_RESIDENT
    )
    result_tk16 = analyze_tiling(
        gemm, GemmTile(tile_m=4, tile_n=4, tile_k=16), hierarchy, schedule=GemmSchedule.C_RESIDENT
    )

    assert result_tk8.k_tiles != result_tk16.k_tiles
    assert result_tk8.total_dram_bytes == result_tk16.total_dram_bytes


def test_asymmetric_workload_a_resident_and_b_resident_differ():
    # M != N: A-resident (repeats B across m_tiles) and B-resident (repeats
    # A across n_tiles) are not equivalent.
    gemm = Gemm(m=8, n=64, k=16, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=8, tile_k=8)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    a_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.A_RESIDENT)
    b_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.B_RESIDENT)

    assert a_result.total_dram_bytes != b_result.total_dram_bytes

    m_tiles = 2  # ceil(8/4)
    n_tiles = 8  # ceil(64/8)
    k_tiles = 2  # ceil(16/8)

    assert a_result.a_dram_read_bytes == gemm.a_bytes
    assert a_result.b_dram_read_bytes == gemm.b_bytes * m_tiles
    assert a_result.c_dram_read_bytes == gemm.c_bytes * (k_tiles - 1)
    assert a_result.c_dram_write_bytes == gemm.c_bytes * k_tiles

    assert b_result.a_dram_read_bytes == gemm.a_bytes * n_tiles
    assert b_result.b_dram_read_bytes == gemm.b_bytes
    assert b_result.c_dram_read_bytes == gemm.c_bytes * (k_tiles - 1)
    assert b_result.c_dram_write_bytes == gemm.c_bytes * k_tiles


def test_edge_tile_across_schedules():
    # Non-divisible dims exercised under all three schedules.
    gemm = Gemm(m=5, n=3, k=7, dtype=DType.FP16)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=3)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    m_tiles, n_tiles, k_tiles = 3, 2, 3  # ceil(5/2), ceil(3/2), ceil(7/3)

    c_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.C_RESIDENT)
    assert c_result.a_dram_read_bytes == gemm.a_bytes * n_tiles
    assert c_result.b_dram_read_bytes == gemm.b_bytes * m_tiles
    assert c_result.c_dram_write_bytes == gemm.c_bytes

    a_result = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.A_RESIDENT)
    assert a_result.a_dram_read_bytes == gemm.a_bytes
    assert a_result.b_dram_read_bytes == gemm.b_bytes * m_tiles
    assert a_result.c_dram_read_bytes == gemm.c_bytes * (k_tiles - 1)
    assert a_result.c_dram_write_bytes == gemm.c_bytes * k_tiles


def test_traffic_amplification_at_least_one_for_all_schedules():
    gemm = Gemm(m=6, n=10, k=9, dtype=DType.FP16)
    tile = GemmTile(tile_m=3, tile_n=4, tile_k=3)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    for schedule in (GemmSchedule.C_RESIDENT, GemmSchedule.A_RESIDENT, GemmSchedule.B_RESIDENT):
        result = analyze_tiling(gemm, tile, hierarchy, schedule=schedule)
        assert result.traffic_amplification >= 1.0
        assert math.isclose(
            result.traffic_amplification, result.total_dram_bytes / gemm.dram_bytes
        )


def test_compare_schedules_known_value():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)

    comparison = compare_schedules(gemm, tile, hierarchy)

    assert comparison.c_resident.total_dram_bytes == 320
    assert comparison.a_resident.total_dram_bytes == 384
    assert comparison.b_resident.total_dram_bytes == 384

    assert comparison.best_among_modeled_schedules == (GemmSchedule.C_RESIDENT,)


def test_compare_schedules_tie_reported():
    # Full-residency: all three schedules produce identical minimum traffic.
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=4)
    hierarchy = MemoryHierarchy(sram_bytes=192)

    comparison = compare_schedules(gemm, tile, hierarchy)

    assert set(comparison.best_among_modeled_schedules) == {
        GemmSchedule.C_RESIDENT,
        GemmSchedule.A_RESIDENT,
        GemmSchedule.B_RESIDENT,
    }


def test_invalid_schedule_rejected():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)

    with pytest.raises(ValueError):
        analyze_tiling(gemm, tile, hierarchy, schedule="not-a-schedule")
