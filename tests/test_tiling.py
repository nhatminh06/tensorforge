import math

import pytest

from tensorforge.gemm import DType, Gemm
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray, map_gemm
from tensorforge.tiling import GemmTile, analyze_tiling


@pytest.mark.parametrize("tm,tn,tk", [(0, 2, 2), (2, 0, 2), (2, 2, 0), (-1, 2, 2)])
def test_rejects_invalid_tile_dimensions(tm, tn, tk):
    with pytest.raises(ValueError):
        GemmTile(tile_m=tm, tile_n=tn, tile_k=tk)


def test_known_value_4x4x4_tile_2x2x2():
    # M=N=K=4, FP32: A=B=C=64B, ideal baseline=192B, FLOPs=128.
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=64)

    result = analyze_tiling(gemm, tile, hierarchy)

    assert result.m_tiles == 2
    assert result.n_tiles == 2
    assert result.k_tiles == 2
    assert result.output_tiles == 4
    assert result.tile_steps == 8

    # A tile = 2x2 = 4 elem, B tile = 4 elem, C tile = 4 elem -> 12 elem * 4B = 48B
    assert result.max_tile_working_set_bytes == 48
    assert result.tile_fits is True

    assert result.a_dram_read_bytes == 64 * 2  # A bytes * n_tiles
    assert result.b_dram_read_bytes == 64 * 2  # B bytes * m_tiles
    assert result.c_dram_write_bytes == 64
    assert result.total_dram_bytes == 320

    assert result.ideal_baseline_dram_bytes == 192
    assert math.isclose(result.traffic_amplification, 320 / 192)

    assert gemm.flops == 128
    assert math.isclose(result.effective_arithmetic_intensity, 128 / 320)
    assert math.isclose(result.effective_arithmetic_intensity, 0.4)


def test_full_residency_tile_matches_ideal_baseline():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=4)
    hierarchy = MemoryHierarchy(sram_bytes=192)

    result = analyze_tiling(gemm, tile, hierarchy)

    assert result.m_tiles == 1
    assert result.n_tiles == 1
    assert result.k_tiles == 1

    assert result.total_dram_bytes == 192
    assert result.total_dram_bytes == gemm.dram_bytes
    assert result.traffic_amplification == 1.0


def test_k_only_tiling_preserves_ideal_traffic():
    # M=4,N=4 full tile, K split into 4 -> amplification stays 1.0.
    gemm = Gemm(m=4, n=4, k=8, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result = analyze_tiling(gemm, tile, hierarchy)

    assert result.m_tiles == 1
    assert result.n_tiles == 1
    assert result.k_tiles == 4

    assert result.total_dram_bytes == gemm.dram_bytes
    assert result.traffic_amplification == 1.0


def test_changing_tk_preserves_total_traffic_but_changes_capacity_and_steps():
    gemm = Gemm(m=8, n=8, k=32, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result_tk8 = analyze_tiling(gemm, GemmTile(tile_m=4, tile_n=4, tile_k=8), hierarchy)
    result_tk16 = analyze_tiling(gemm, GemmTile(tile_m=4, tile_n=4, tile_k=16), hierarchy)

    assert result_tk8.k_tiles != result_tk16.k_tiles
    assert result_tk8.max_tile_working_set_bytes != result_tk16.max_tile_working_set_bytes
    assert result_tk8.tile_steps != result_tk16.tile_steps

    assert result_tk8.total_dram_bytes == result_tk16.total_dram_bytes


def test_smaller_tn_increases_a_refetch():
    gemm = Gemm(m=8, n=8, k=8, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    wide_tn = analyze_tiling(gemm, GemmTile(tile_m=4, tile_n=8, tile_k=4), hierarchy)
    narrow_tn = analyze_tiling(gemm, GemmTile(tile_m=4, tile_n=4, tile_k=4), hierarchy)

    assert narrow_tn.n_tiles > wide_tn.n_tiles
    assert narrow_tn.a_dram_read_bytes > wide_tn.a_dram_read_bytes
    # B refetch is unaffected by Tn.
    assert narrow_tn.b_dram_read_bytes == wide_tn.b_dram_read_bytes


def test_smaller_tm_increases_b_refetch():
    gemm = Gemm(m=8, n=8, k=8, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    wide_tm = analyze_tiling(gemm, GemmTile(tile_m=8, tile_n=4, tile_k=4), hierarchy)
    narrow_tm = analyze_tiling(gemm, GemmTile(tile_m=4, tile_n=4, tile_k=4), hierarchy)

    assert narrow_tm.m_tiles > wide_tm.m_tiles
    assert narrow_tm.b_dram_read_bytes > wide_tm.b_dram_read_bytes
    # A refetch is unaffected by Tm.
    assert narrow_tm.a_dram_read_bytes == wide_tm.a_dram_read_bytes


def test_edge_tile_traffic():
    # M=5,N=3,K=7, FP16, tile 2x2x3: dimensions do not divide evenly.
    gemm = Gemm(m=5, n=3, k=7, dtype=DType.FP16)
    tile = GemmTile(tile_m=2, tile_n=2, tile_k=3)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result = analyze_tiling(gemm, tile, hierarchy)

    assert result.m_tiles == 3  # ceil(5/2)
    assert result.n_tiles == 2  # ceil(3/2)
    assert result.k_tiles == 3  # ceil(7/3)

    # A = 5*7=35 elem * 2B = 70B; B = 7*3=21 elem*2B=42B; C=5*3=15 elem*2B=30B
    assert gemm.a_bytes == 70
    assert gemm.b_bytes == 42
    assert gemm.c_bytes == 30

    assert result.a_dram_read_bytes == 70 * 2
    assert result.b_dram_read_bytes == 42 * 3
    assert result.c_dram_write_bytes == 30
    assert result.total_dram_bytes == 140 + 126 + 30

    # Largest tile: min(2,5)=2, min(2,3)=2, min(3,7)=3 -> (2*3+3*2+2*2)*2B = 16*2=32B
    assert result.max_tile_working_set_bytes == 32


def test_invalid_tile_capacity_rejected():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=4)  # requires 192B
    hierarchy = MemoryHierarchy(sram_bytes=64)

    with pytest.raises(ValueError):
        analyze_tiling(gemm, tile, hierarchy)


def test_exact_capacity_boundary_fits():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=4)
    hierarchy = MemoryHierarchy(sram_bytes=192)  # exactly the required 192B

    result = analyze_tiling(gemm, tile, hierarchy)
    assert result.tile_fits is True
    assert result.max_tile_working_set_bytes == 192


def test_tile_larger_than_workload_clips_to_workload():
    gemm = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    tile = GemmTile(tile_m=16, tile_n=16, tile_k=16)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result = analyze_tiling(gemm, tile, hierarchy)

    assert result.m_tiles == 1
    assert result.n_tiles == 1
    assert result.k_tiles == 1

    # Clipped to workload dims: A tile=2*4=8, B tile=4*3=12, C tile=2*3=6 -> 26*4B=104B
    assert result.max_tile_working_set_bytes == 104
    assert result.total_dram_bytes == gemm.dram_bytes
    assert result.traffic_amplification == 1.0


def test_traffic_consistency_and_lower_bound():
    gemm = Gemm(m=6, n=10, k=5, dtype=DType.FP16)
    tile = GemmTile(tile_m=3, tile_n=4, tile_k=2)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result = analyze_tiling(gemm, tile, hierarchy)

    assert result.total_dram_bytes == (
        result.a_dram_read_bytes + result.b_dram_read_bytes + result.c_dram_write_bytes
    )
    assert result.total_dram_bytes >= gemm.dram_bytes


def test_tile_shape_does_not_change_gemm_operations():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    analyze_tiling(gemm, GemmTile(tile_m=1, tile_n=1, tile_k=1), hierarchy)
    analyze_tiling(gemm, GemmTile(tile_m=4, tile_n=4, tile_k=4), hierarchy)

    assert gemm.macs == 64
    assert gemm.flops == 128


def test_tile_shape_does_not_change_pe_mapping():
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    pe_array = PeArray(rows=2, columns=2)
    baseline = map_gemm(gemm, pe_array)

    for tile in (GemmTile(1, 1, 1), GemmTile(4, 4, 4), GemmTile(2, 2, 2)):
        analyze_tiling(gemm, tile, hierarchy)
        assert map_gemm(gemm, pe_array) == baseline
