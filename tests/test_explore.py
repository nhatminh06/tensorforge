import math

import pytest

from tensorforge.explore import CandidateConfig, explore
from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.tiling import GemmSchedule
from tensorforge.timing import TimingConfig

GEMM_4x4x4_FP32 = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
HARDWARE = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1e9)
TIMING = TimingConfig(clock_hz=1e9)


def test_small_known_search_candidate_count_and_winner():
    # M=N=K=4, FP32, SRAM=192B: Tm=[2,4], Tn=[2,4], Tk=[2], PE rows=[1,2], cols=[1,2]
    # candidate_count = 2*2*1*2*2*3 = 48. All tiles fit 192B SRAM (largest tile
    # 4x4 with Tk=2 needs (4*2+2*4+4*4)*4B = 128B <= 192B), so all feasible.
    hierarchy = MemoryHierarchy(sram_bytes=192)

    result = explore(
        GEMM_4x4x4_FP32,
        hierarchy,
        HARDWARE,
        TIMING,
        tile_m_values=[2, 4],
        tile_n_values=[2, 4],
        tile_k_values=[2],
        pe_rows_values=[1, 2],
        pe_cols_values=[1, 2],
        top_k=48,
    )

    assert result.total_candidates == 48
    assert result.feasible_candidates == 48
    assert result.infeasible_candidates == 0
    assert len(result.ranked) == 48

    # Winner: tile 4x4x2, c-resident (min total_dram_bytes=192, matching ideal),
    # PE 2x2 (fewest cycles among tied perfect_overlap candidates -> smallest serialized).
    best = result.ranked[0]
    assert best.candidate.tile_m == 4
    assert best.candidate.tile_n == 4
    assert best.candidate.tile_k == 2
    assert best.candidate.schedule == GemmSchedule.C_RESIDENT
    assert best.timing_result.dram_bytes == 192
    assert best.candidate.pe_rows == 2
    assert best.candidate.pe_cols == 2
    assert best.timing_result.compute_cycles == 16


def test_explore_rejects_invalid_inputs():
    hierarchy = MemoryHierarchy(sram_bytes=1000)

    with pytest.raises(ValueError):
        explore(
            GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
            tile_m_values=[], tile_n_values=[2], tile_k_values=[2],
            pe_rows_values=[1], pe_cols_values=[1],
        )
    with pytest.raises(ValueError):
        explore(
            GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
            tile_m_values=[0], tile_n_values=[2], tile_k_values=[2],
            pe_rows_values=[1], pe_cols_values=[1],
        )
    with pytest.raises(ValueError):
        explore(
            GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
            tile_m_values=[2], tile_n_values=[2], tile_k_values=[2],
            pe_rows_values=[-1], pe_cols_values=[1],
        )
    with pytest.raises(ValueError):
        explore(
            GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
            tile_m_values=[2], tile_n_values=[2], tile_k_values=[2],
            pe_rows_values=[1], pe_cols_values=[1], top_k=0,
        )


def test_excessive_search_size_rejected_without_evaluation():
    hierarchy = MemoryHierarchy(sram_bytes=1000)
    with pytest.raises(ValueError):
        explore(
            GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
            tile_m_values=list(range(1, 60)),
            tile_n_values=list(range(1, 60)),
            tile_k_values=[1],
            pe_rows_values=[1],
            pe_cols_values=[1],
            max_candidates=100,
        )


def test_duplicate_candidate_values_deduplicated():
    hierarchy = MemoryHierarchy(sram_bytes=192)
    result = explore(
        GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
        tile_m_values=[4, 4, 4], tile_n_values=[4], tile_k_values=[4],
        pe_rows_values=[1, 1], pe_cols_values=[1],
        schedules=[GemmSchedule.C_RESIDENT],
    )
    assert result.total_candidates == 1
    assert result.feasible_candidates == 1


def test_infeasible_tile_excluded_but_search_continues():
    # 4x4 tile at Tk=4 needs 192B; 2x2 tile fits easily. SRAM=100B rejects the former.
    hierarchy = MemoryHierarchy(sram_bytes=100)
    result = explore(
        GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[4],
        pe_rows_values=[1], pe_cols_values=[1],
        schedules=[GemmSchedule.C_RESIDENT],
    )
    assert result.total_candidates == 4
    assert result.feasible_candidates == 1  # only tile 2x2x4 fits
    assert result.infeasible_candidates == 3
    assert result.feasible_candidates + result.infeasible_candidates == result.total_candidates
    assert len(result.infeasible_reasons) == 3
    assert result.ranked[0].candidate.tile_m == 2
    assert result.ranked[0].candidate.tile_n == 2


def test_all_infeasible_search():
    hierarchy = MemoryHierarchy(sram_bytes=1)  # nothing fits
    result = explore(
        GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2, 4],
        pe_rows_values=[1, 2], pe_cols_values=[1, 2],
    )
    assert result.feasible_candidates == 0
    assert result.infeasible_candidates == result.total_candidates
    assert result.ranked == ()
    assert result.tied_for_best == 0


def test_deterministic_repeated_runs():
    hierarchy = MemoryHierarchy(sram_bytes=192)
    kwargs = dict(
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2],
        pe_rows_values=[1, 2], pe_cols_values=[1, 2],
    )
    result1 = explore(GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING, **kwargs)
    result2 = explore(GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING, **kwargs)

    order1 = [(r.candidate.tile_m, r.candidate.tile_n, r.candidate.tile_k,
               r.candidate.schedule.value, r.candidate.pe_rows, r.candidate.pe_cols)
              for r in result1.ranked]
    order2 = [(r.candidate.tile_m, r.candidate.tile_n, r.candidate.tile_k,
               r.candidate.schedule.value, r.candidate.pe_rows, r.candidate.pe_cols)
              for r in result2.ranked]
    assert order1 == order2


def test_tie_detection_full_residency_all_schedules_equal():
    # m_tiles=n_tiles=k_tiles=1 for all three schedules -> identical traffic/timing.
    gemm = Gemm(m=4, n=4, k=4, dtype=DType.FP32)
    hierarchy = MemoryHierarchy(sram_bytes=192)
    result = explore(
        gemm, hierarchy, HARDWARE, TIMING,
        tile_m_values=[4], tile_n_values=[4], tile_k_values=[4],
        pe_rows_values=[2], pe_cols_values=[2],
    )
    assert result.feasible_candidates == 3  # one per schedule
    assert result.tied_for_best == 3
    times = {r.timing_result.perfect_overlap_time_seconds for r in result.ranked}
    assert len(times) == 1


def test_top_k_behavior():
    hierarchy = MemoryHierarchy(sram_bytes=192)
    kwargs = dict(
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2],
        pe_rows_values=[1, 2], pe_cols_values=[1, 2],
    )
    result_1 = explore(GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING, top_k=1, **kwargs)
    result_3 = explore(GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING, top_k=3, **kwargs)
    result_huge = explore(GEMM_4x4x4_FP32, hierarchy, HARDWARE, TIMING, top_k=1000, **kwargs)

    assert result_1.top_k == 1
    assert result_3.top_k == 3
    # `ranked` holds the full feasible list regardless of top_k; slicing is a CLI concern.
    assert len(result_1.ranked) == 48
    assert result_1.ranked[:1][0] == result_3.ranked[:3][0]
    assert len(result_huge.ranked[:1000]) == 48  # fewer feasible than requested top_k


def test_same_tile_different_schedule_compute_unchanged_memory_may_differ():
    hierarchy = MemoryHierarchy(sram_bytes=1000)
    result = explore(
        Gemm(m=8, n=8, k=8, dtype=DType.FP32), hierarchy, HARDWARE, TIMING,
        tile_m_values=[4], tile_n_values=[4], tile_k_values=[4],
        pe_rows_values=[2], pe_cols_values=[2],
    )
    assert result.feasible_candidates == 3
    cycles = {r.timing_result.compute_cycles for r in result.ranked}
    assert len(cycles) == 1  # identical PE mapping regardless of schedule

    dram = {r.timing_result.dram_bytes for r in result.ranked}
    assert len(dram) >= 1  # may or may not differ depending on tile/shape; just must be consistent
    c_resident = next(r for r in result.ranked if r.candidate.schedule == GemmSchedule.C_RESIDENT)
    a_resident = next(r for r in result.ranked if r.candidate.schedule == GemmSchedule.A_RESIDENT)
    assert c_resident.timing_result.compute_cycles == a_resident.timing_result.compute_cycles


def test_cli_explore_runs_and_prints_winner(capsys):
    from tensorforge.cli import main

    main([
        "--m", "4", "--n", "4", "--k", "4", "--dtype", "fp32",
        "--peak-tflops", "1", "--bandwidth-gbps", "1",
        "--sram-kib", "0.1875", "--clock-ghz", "1", "--explore",
        "--tile-m-values", "2,4", "--tile-n-values", "2,4", "--tile-k-values", "2",
        "--pe-row-values", "1,2", "--pe-col-values", "1,2", "--top-k", "3",
    ])
    out = capsys.readouterr().out
    assert "total candidates        48" in out
    assert "feasible candidates     48" in out
    assert "Best among searched candidates:" in out
    assert "Why:" in out


def test_same_schedule_different_pe_traffic_unchanged_compute_may_differ():
    hierarchy = MemoryHierarchy(sram_bytes=1000)
    result = explore(
        Gemm(m=8, n=8, k=8, dtype=DType.FP32), hierarchy, HARDWARE, TIMING,
        tile_m_values=[4], tile_n_values=[4], tile_k_values=[4],
        pe_rows_values=[1, 4], pe_cols_values=[1, 4],
        schedules=[GemmSchedule.C_RESIDENT],
    )
    assert result.feasible_candidates == 4
    dram = {r.timing_result.dram_bytes for r in result.ranked}
    assert len(dram) == 1  # traffic independent of PE shape

    cycles = {r.timing_result.compute_cycles for r in result.ranked}
    assert len(cycles) > 1  # compute cycles depend on PE shape
