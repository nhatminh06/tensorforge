import math

import pytest

from tensorforge.gemm import DType, Gemm
from tensorforge.pe_array import PeArray, map_gemm


@pytest.mark.parametrize("rows,cols", [(0, 4), (-1, 4), (4, 0), (4, -1)])
def test_rejects_nonpositive_dimensions(rows, cols):
    with pytest.raises(ValueError):
        PeArray(rows=rows, columns=cols)


def test_perfect_one_wave_fit():
    # M=16, N=16, K=32 onto a 16x16 array: output fits in exactly one wave.
    gemm = Gemm(m=16, n=16, k=32, dtype=DType.FP32)
    result = map_gemm(gemm, PeArray(rows=16, columns=16))

    assert result.row_waves == 1
    assert result.column_waves == 1
    assert result.waves == 1

    assert result.useful_pe_slots == 256
    assert result.available_pe_slots == 256
    assert result.unused_pe_slots == 0

    assert result.spatial_utilization == 1.0
    assert result.compute_cycles == 32


def test_multi_wave_perfect_fit():
    # M=32, N=32, K=8 onto a 16x16 array: 2x2 = 4 waves, each fully occupied.
    gemm = Gemm(m=32, n=32, k=8, dtype=DType.FP32)
    result = map_gemm(gemm, PeArray(rows=16, columns=16))

    assert result.row_waves == 2
    assert result.column_waves == 2
    assert result.waves == 4

    assert result.useful_pe_slots == 1024
    assert result.available_pe_slots == 1024
    assert result.unused_pe_slots == 0
    assert result.spatial_utilization == 1.0

    assert result.compute_cycles == 32


def test_edge_utilization():
    # M=17, N=17, K=8 onto a 16x16 array: partial edge wave wastes slots.
    gemm = Gemm(m=17, n=17, k=8, dtype=DType.FP32)
    result = map_gemm(gemm, PeArray(rows=16, columns=16))

    assert result.row_waves == 2
    assert result.column_waves == 2
    assert result.waves == 4

    assert result.useful_pe_slots == 289
    assert result.available_pe_slots == 1024
    assert result.unused_pe_slots == 735

    assert math.isclose(result.spatial_utilization, 289 / 1024)
    assert result.compute_cycles == 32


def test_rectangular_array():
    # M=10, N=20, K=16 onto a 4x8 (rectangular) array.
    gemm = Gemm(m=10, n=20, k=16, dtype=DType.FP32)
    pe_array = PeArray(rows=4, columns=8)
    result = map_gemm(gemm, pe_array)

    assert result.row_waves == 3
    assert result.column_waves == 3
    assert result.waves == 9

    assert pe_array.pe_count == 32
    assert result.available_pe_slots == 288
    assert result.useful_pe_slots == 200
    assert math.isclose(result.spatial_utilization, 200 / 288)
    assert result.compute_cycles == 9 * 16


def test_array_larger_than_workload():
    gemm = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    result = map_gemm(gemm, PeArray(rows=16, columns=16))

    assert result.waves == 1
    assert result.useful_pe_slots == 6
    assert result.available_pe_slots == 256
    assert math.isclose(result.spatial_utilization, 6 / 256)
    assert result.compute_cycles == 4


def test_single_pe_array():
    gemm = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    result = map_gemm(gemm, PeArray(rows=1, columns=1))

    assert result.waves == 6
    assert result.spatial_utilization == 1.0
    assert result.compute_cycles == 24
    # 1 MAC/PE/cycle with full spatial utilization: cycles == total MACs.
    assert result.compute_cycles == gemm.macs


def test_same_pe_count_different_shape_gives_different_utilization():
    # Both arrays have 64 PEs, but different utilization for this workload.
    gemm = Gemm(m=17, n=64, k=128, dtype=DType.FP32)

    wide = map_gemm(gemm, PeArray(rows=4, columns=16))
    narrow = map_gemm(gemm, PeArray(rows=16, columns=4))

    assert wide.pe_count == narrow.pe_count == 64
    assert wide.spatial_utilization != narrow.spatial_utilization


def test_perfect_fit_compute_cycles_consistent_with_gemm_macs():
    # No spatial tail and 1 MAC/PE/cycle: compute_cycles * pe_count == MACs.
    gemm = Gemm(m=16, n=16, k=32, dtype=DType.FP32)
    pe_array = PeArray(rows=16, columns=16)
    result = map_gemm(gemm, pe_array)

    assert result.compute_cycles * pe_array.pe_count == gemm.macs
