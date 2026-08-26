"""Milestone 10: a small, focused set of independent cross-checks that were
not already covered by the per-module test suites. This intentionally does
NOT duplicate the extensive existing coverage in test_gemm.py, test_pe_array.py,
test_tiling.py, test_schedules.py, test_timing.py, test_transformer.py, and
test_convolution.py -- see docs/validation.md for the full validation summary.
"""

import math

import pytest

from tensorforge.convolution import Conv2DSpec, lower_conv2d_to_gemm
from tensorforge.gemm import DType, Gemm
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray, map_gemm
from tensorforge.tiling import GemmSchedule, GemmTile, analyze_tiling


@pytest.mark.parametrize(
    "m,n,k",
    [(2, 3, 4), (1, 1, 1), (7, 5, 3), (64, 64, 64), (100, 1, 50)],
)
def test_gemm_mac_flop_formula_holds_across_shapes(m, n, k):
    gemm = Gemm(m=m, n=n, k=k, dtype=DType.FP32)
    assert gemm.macs == m * n * k
    assert gemm.flops == 2 * m * n * k
    assert gemm.a_bytes == m * k * 4
    assert gemm.b_bytes == k * n * 4
    assert gemm.c_bytes == m * n * 4


@pytest.mark.parametrize("m,n,k", [(2, 3, 4), (17, 9, 5), (64, 64, 64), (1, 100, 1)])
def test_1x1_pe_array_compute_cycles_equals_mac_count(m, n, k):
    # Independent invariant: with a single PE (no spatial parallelism at
    # all), the analytical compute-cycle count must equal the raw MAC
    # count, since 1 MAC/PE/cycle and every output element is visited
    # exactly once, sequentially.
    gemm = Gemm(m=m, n=n, k=k, dtype=DType.FP32)
    mapping = map_gemm(gemm, PeArray(rows=1, columns=1))
    assert mapping.compute_cycles == gemm.macs


@pytest.mark.parametrize(
    "m,n,k,rows,cols",
    [(16, 16, 8, 16, 16), (32, 32, 4, 8, 8), (64, 32, 2, 16, 4)],
)
def test_perfect_fit_pe_count_times_cycles_equals_macs(m, n, k, rows, cols):
    # When M/N divide evenly by rows/cols (no spatial tail waste), the
    # available PE-slot capacity should exactly match useful MACs, i.e.
    # compute_cycles * pe_count == macs (no idle PE-cycles anywhere).
    gemm = Gemm(m=m, n=n, k=k, dtype=DType.FP32)
    pe_array = PeArray(rows=rows, columns=cols)
    mapping = map_gemm(gemm, pe_array)
    assert mapping.spatial_utilization == 1.0
    assert mapping.compute_cycles * pe_array.pe_count == gemm.macs


@pytest.mark.parametrize(
    "m,n,k,tm,tn,tk",
    [(4, 4, 4, 2, 2, 2), (8, 8, 8, 4, 4, 4), (6, 10, 5, 3, 4, 2)],
)
def test_all_schedules_traffic_at_least_ideal_baseline(m, n, k, tm, tn, tk):
    gemm = Gemm(m=m, n=n, k=k, dtype=DType.FP16)
    tile = GemmTile(tile_m=tm, tile_n=tn, tile_k=tk)
    hierarchy = MemoryHierarchy(sram_bytes=1_000_000)
    for schedule in (GemmSchedule.C_RESIDENT, GemmSchedule.A_RESIDENT, GemmSchedule.B_RESIDENT):
        result = analyze_tiling(gemm, tile, hierarchy, schedule=schedule)
        assert result.total_dram_bytes >= gemm.dram_bytes
        assert result.traffic_amplification >= 1.0


def test_a_resident_b_resident_symmetric_for_square_workload_and_tile():
    # For a perfectly square GEMM with a square tile (so m_tiles == n_tiles
    # and A/B tensor byte sizes match), A-resident and B-resident traffic
    # should be exactly symmetric -- swapping the roles of A and B changes
    # nothing when A and B are themselves symmetric in shape/size.
    gemm = Gemm(m=8, n=8, k=8, dtype=DType.FP32)
    tile = GemmTile(tile_m=4, tile_n=4, tile_k=4)
    hierarchy = MemoryHierarchy(sram_bytes=1_000_000)

    a_resident = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.A_RESIDENT)
    b_resident = analyze_tiling(gemm, tile, hierarchy, schedule=GemmSchedule.B_RESIDENT)

    assert a_resident.total_dram_bytes == b_resident.total_dram_bytes
    assert a_resident.a_dram_read_bytes == b_resident.b_dram_read_bytes
    assert a_resident.b_dram_read_bytes == b_resident.a_dram_read_bytes
    assert a_resident.c_dram_read_bytes == b_resident.c_dram_read_bytes
    assert a_resident.c_dram_write_bytes == b_resident.c_dram_write_bytes


@pytest.mark.parametrize("kernel,expected_ratio", [(1, 1.0), (3, 9.0), (5, 25.0)])
def test_conv_im2col_expansion_scales_with_kernel_area_at_fixed_output(kernel, expected_ratio):
    # Independent cross-check of the kernel-size study in docs/experiments.md:
    # with padding chosen so output resolution stays fixed, expansion ratio
    # should equal exactly kernel_height * kernel_width.
    padding = (kernel - 1) // 2
    spec = Conv2DSpec(
        batch_size=1, in_channels=8, input_height=16, input_width=16,
        out_channels=4, kernel_height=kernel, kernel_width=kernel,
        stride_height=1, stride_width=1, padding_height=padding, padding_width=padding,
        dtype=DType.FP32,
    )
    result = lower_conv2d_to_gemm(spec)
    assert result.output_height == 16
    assert result.output_width == 16
    assert math.isclose(result.im2col_expansion_ratio, expected_ratio)
