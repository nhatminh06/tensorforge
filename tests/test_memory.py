import math

import pytest

from tensorforge.gemm import DType, Gemm
from tensorforge.memory import MemoryHierarchy, analyze_memory
from tensorforge.pe_array import PeArray, map_gemm

# Shared workload: M=2, N=3, K=4, FP32
# A=32B, B=48B, C=24B, working set = 104B (see test_gemm.py for derivation)
GEMM = Gemm(m=2, n=3, k=4, dtype=DType.FP32)


@pytest.mark.parametrize("sram_bytes", [0, -1, 0.5])
def test_rejects_invalid_sram(sram_bytes):
    with pytest.raises(ValueError):
        MemoryHierarchy(sram_bytes=sram_bytes)


def test_full_working_set_fits():
    result = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=128))

    assert result.a_bytes == 32
    assert result.b_bytes == 48
    assert result.c_bytes == 24
    assert result.working_set_bytes == 104

    assert result.a_fits is True
    assert result.b_fits is True
    assert result.c_fits is True
    assert result.full_working_set_fits is True

    assert result.capacity_headroom_bytes == 24
    assert result.capacity_deficit_bytes == 0
    assert math.isclose(result.working_set_to_capacity_ratio, 104 / 128)
    assert result.tiling_required is False


def test_full_working_set_does_not_fit():
    result = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=64))

    assert result.full_working_set_fits is False
    assert result.capacity_deficit_bytes == 40
    assert result.capacity_headroom_bytes == 0
    assert math.isclose(result.working_set_to_capacity_ratio, 104 / 64)
    assert result.tiling_required is True

    # Every individual tensor still fits even though the full set does not.
    assert result.a_fits is True
    assert result.b_fits is True
    assert result.c_fits is True


def test_individual_tensor_does_not_fit():
    # B (48B) exceeds a 40B SRAM, though the workload is otherwise tiny.
    result = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=40))

    assert result.a_fits is True
    assert result.b_fits is False
    assert result.c_fits is True
    assert result.full_working_set_fits is False


def test_exact_capacity_boundary():
    result = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=104))

    assert result.full_working_set_fits is True
    assert result.capacity_headroom_bytes == 0
    assert result.capacity_deficit_bytes == 0
    assert result.working_set_to_capacity_ratio == 1.0
    assert result.tiling_required is False


def test_large_sram():
    result = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=1_000_000))

    assert result.full_working_set_fits is True
    assert result.capacity_headroom_bytes == 1_000_000 - 104
    assert result.working_set_to_capacity_ratio < 1.0


def test_dram_traffic_matches_gemm_baseline():
    result = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=128))

    assert result.dram_read_bytes == 80  # A + B = 32 + 48
    assert result.dram_write_bytes == 24  # C
    assert result.total_dram_bytes == 104

    assert result.total_dram_bytes == GEMM.dram_bytes


def test_dtype_changes_fit_but_not_operations():
    fp32 = Gemm(m=2, n=3, k=4, dtype=DType.FP32)
    fp16 = Gemm(m=2, n=3, k=4, dtype=DType.FP16)

    sram = MemoryHierarchy(sram_bytes=64)
    fp32_result = analyze_memory(fp32, sram)
    fp16_result = analyze_memory(fp16, sram)

    # FP32 working set (104B) does not fit in 64B SRAM; FP16 (52B) does.
    assert fp32_result.full_working_set_fits is False
    assert fp16_result.full_working_set_fits is True

    assert fp32.macs == fp16.macs
    assert fp32.flops == fp16.flops


def test_sram_capacity_does_not_change_gemm_operations():
    small = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=1))
    large = analyze_memory(GEMM, MemoryHierarchy(sram_bytes=1_000_000))

    assert GEMM.macs == 24
    assert GEMM.flops == 48
    # Analyzing memory at either capacity must not touch the workload.
    assert small.a_bytes == large.a_bytes == GEMM.a_bytes


def test_sram_capacity_does_not_change_pe_mapping():
    pe_array = PeArray(rows=1, columns=1)
    baseline = map_gemm(GEMM, pe_array)

    # SRAM capacity is unrelated input; PE mapping must be identical regardless.
    for sram_bytes in (1, 64, 128, 1_000_000):
        analyze_memory(GEMM, MemoryHierarchy(sram_bytes=sram_bytes))
        mapping = map_gemm(GEMM, pe_array)
        assert mapping == baseline
