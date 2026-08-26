import math

import pytest

from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray
from tensorforge.tiling import GemmSchedule
from tensorforge.timing import TimingConfig
from tensorforge.transformer import (
    TransformerBlockSpec,
    block_total_flops,
    block_total_macs,
    derive_transformer_gemms,
    evaluate_transformer_block,
    explore_transformer_architectures,
)

HARDWARE = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1e9)
TIMING = TimingConfig(clock_hz=1e9)


def tiny_spec() -> TransformerBlockSpec:
    return TransformerBlockSpec(
        batch_size=1, sequence_length=2, d_model=4, num_heads=2, d_ff=8, dtype=DType.FP32
    )


# --- spec validation --------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_size=0, sequence_length=2, d_model=4, num_heads=2, d_ff=8),
        dict(batch_size=1, sequence_length=0, d_model=4, num_heads=2, d_ff=8),
        dict(batch_size=1, sequence_length=2, d_model=0, num_heads=2, d_ff=8),
        dict(batch_size=1, sequence_length=2, d_model=4, num_heads=0, d_ff=8),
        dict(batch_size=1, sequence_length=2, d_model=4, num_heads=2, d_ff=0),
    ],
)
def test_rejects_nonpositive_dimensions(kwargs):
    with pytest.raises(ValueError):
        TransformerBlockSpec(dtype=DType.FP32, **kwargs)


def test_rejects_d_model_not_divisible_by_heads():
    with pytest.raises(ValueError):
        TransformerBlockSpec(
            batch_size=1, sequence_length=2, d_model=5, num_heads=2, d_ff=8, dtype=DType.FP32
        )


def test_valid_spec_head_dim():
    spec = TransformerBlockSpec(
        batch_size=1, sequence_length=2, d_model=4, num_heads=2, d_ff=8, dtype=DType.FP32
    )
    assert spec.head_dim == 2


# --- operation shape derivation ---------------------------------------------

def test_tiny_block_operation_shapes_and_order():
    spec = tiny_spec()
    ops = derive_transformer_gemms(spec)

    expected = [
        ("q_projection", 2, 4, 4, 1),
        ("k_projection", 2, 4, 4, 1),
        ("v_projection", 2, 4, 4, 1),
        ("attention_scores", 2, 2, 2, 2),
        ("attention_value", 2, 2, 2, 2),
        ("output_projection", 2, 4, 4, 1),
        ("mlp_up", 2, 8, 4, 1),
        ("mlp_down", 2, 4, 8, 1),
    ]
    assert len(ops) == len(expected)
    for op, (name, m, n, k, rep) in zip(ops, expected):
        assert op.name == name
        assert (op.gemm.m, op.gemm.n, op.gemm.k) == (m, n, k)
        assert op.repetitions == rep
        assert op.gemm.dtype == DType.FP32


# --- FLOP/MAC totals ---------------------------------------------------------

def test_tiny_block_flop_total_known_value():
    # Hand-derived: 4 projections (Q,K,V,output) @ 64 FLOPs each = 256
    # MLP up+down @ 128 FLOPs each = 256
    # attention scores/value @ 16 FLOPs * 2 heads each = 32 + 32 = 64
    # total = 256 + 256 + 64 = 576
    spec = tiny_spec()
    ops = derive_transformer_gemms(spec)
    assert block_total_flops(ops) == 576
    assert block_total_macs(ops) == 288


def test_closed_form_flop_formula():
    # 8*B*S*D^2 + 4*B*S*D*F + 4*B*S^2*D
    for b, s, d, h, f in [(1, 2, 4, 2, 8), (2, 4, 8, 4, 16), (1, 16, 64, 8, 256)]:
        spec = TransformerBlockSpec(
            batch_size=b, sequence_length=s, d_model=d, num_heads=h, d_ff=f, dtype=DType.FP32
        )
        ops = derive_transformer_gemms(spec)
        expected = 8 * b * s * d**2 + 4 * b * s * d * f + 4 * b * s**2 * d
        assert block_total_flops(ops) == expected


def test_head_count_changes_attention_repetitions_not_projection():
    for h in (1, 2, 4):
        spec = TransformerBlockSpec(
            batch_size=1, sequence_length=8, d_model=8, num_heads=h, d_ff=16, dtype=DType.FP32
        )
        ops = derive_transformer_gemms(spec)
        by_name = {op.name: op for op in ops}
        assert by_name["attention_scores"].repetitions == 1 * h
        assert by_name["attention_value"].repetitions == 1 * h
        assert by_name["q_projection"].repetitions == 1
        # head_dim changes with H; attention shape uses head_dim, not D.
        assert by_name["attention_scores"].gemm.k == spec.head_dim


def test_batch_changes_m_and_attention_repetitions():
    for b in (1, 2, 3):
        spec = TransformerBlockSpec(
            batch_size=b, sequence_length=8, d_model=8, num_heads=2, d_ff=16, dtype=DType.FP32
        )
        ops = derive_transformer_gemms(spec)
        by_name = {op.name: op for op in ops}
        assert by_name["q_projection"].gemm.m == b * 8
        assert by_name["attention_scores"].repetitions == b * 2
        # per-head attention shape (S x S x head_dim) doesn't depend on B.
        assert by_name["attention_scores"].gemm.m == 8
        assert by_name["attention_scores"].gemm.n == 8


def test_sequence_length_scaling_quadratic_attention_linear_projection():
    def flops_for(s):
        spec = TransformerBlockSpec(
            batch_size=1, sequence_length=s, d_model=8, num_heads=2, d_ff=16, dtype=DType.FP32
        )
        ops = derive_transformer_gemms(spec)
        by_name = {op.name: op for op in ops}
        projection_flops = by_name["q_projection"].gemm.flops  # scales with S (M=B*S)
        attention_flops = (
            by_name["attention_scores"].gemm.flops * by_name["attention_scores"].repetitions
        )
        return projection_flops, attention_flops

    proj_s, attn_s = flops_for(4)
    proj_2s, attn_2s = flops_for(8)

    assert math.isclose(proj_2s, proj_s * 2)  # linear in S
    assert math.isclose(attn_2s, attn_s * 4)  # quadratic in S


def test_d_ff_changes_only_mlp_shapes():
    for f in (8, 16, 32):
        spec = TransformerBlockSpec(
            batch_size=1, sequence_length=4, d_model=8, num_heads=2, d_ff=f, dtype=DType.FP32
        )
        ops = derive_transformer_gemms(spec)
        by_name = {op.name: op for op in ops}
        assert by_name["mlp_up"].gemm.n == f
        assert by_name["mlp_down"].gemm.k == f
        assert by_name["q_projection"].gemm.n == 8  # unaffected by F
        assert by_name["attention_scores"].gemm.m == 4  # unaffected by F


# --- fixed-hardware block evaluation ----------------------------------------

def test_fixed_hardware_block_evaluation():
    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    pe_array = PeArray(rows=2, columns=2)

    result = evaluate_transformer_block(
        spec, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4, 8], tile_n_values=[2, 4, 8], tile_k_values=[2, 4, 8],
    )

    assert len(result.operation_results) == 8
    names = [r.name for r in result.operation_results]
    assert names == [
        "q_projection", "k_projection", "v_projection",
        "attention_scores", "attention_value",
        "output_projection", "mlp_up", "mlp_down",
    ]
    assert result.total_macs == 288
    assert result.total_flops == 576
    assert result.pe_array == pe_array
    assert result.unmodeled_operations == (
        "softmax", "normalization", "activation", "residual additions"
    )


def test_block_time_and_dram_are_sums_of_aggregate_contributions():
    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    pe_array = PeArray(rows=2, columns=2)

    result = evaluate_transformer_block(
        spec, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4, 8], tile_n_values=[2, 4, 8], tile_k_values=[2, 4, 8],
    )

    expected_perfect_overlap = sum(
        r.aggregate_perfect_overlap_time_seconds for r in result.operation_results
    )
    expected_serialized = sum(r.aggregate_serialized_time_seconds for r in result.operation_results)
    expected_dram = sum(r.aggregate_dram_bytes for r in result.operation_results)

    assert result.perfect_overlap_time_seconds == expected_perfect_overlap
    assert result.serialized_time_seconds == expected_serialized
    assert result.total_dram_bytes == expected_dram

    # Repetitions applied: attention aggregate = per-execution * 2 (heads).
    attn = next(r for r in result.operation_results if r.name == "attention_scores")
    assert attn.aggregate_perfect_overlap_time_seconds == (
        attn.mapping.timing_result.perfect_overlap_time_seconds * 2
    )
    assert attn.aggregate_dram_bytes == attn.mapping.timing_result.dram_bytes * 2


def test_largest_contributor_identification():
    # Make MLP dominate by using a much larger d_ff relative to everything else.
    spec = TransformerBlockSpec(
        batch_size=1, sequence_length=4, d_model=4, num_heads=2, d_ff=256, dtype=DType.FP32
    )
    hierarchy = MemoryHierarchy(sram_bytes=1_000_000)
    pe_array = PeArray(rows=4, columns=4)

    result = evaluate_transformer_block(
        spec, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[4], tile_n_values=[256], tile_k_values=[256],
    )
    assert result.largest_time_contributor in ("mlp_up", "mlp_down")
    assert result.largest_dram_contributor in ("mlp_up", "mlp_down")


def test_qkv_and_output_projection_share_cached_mapping():
    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    pe_array = PeArray(rows=2, columns=2)

    result = evaluate_transformer_block(
        spec, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2, 4],
    )
    by_name = {r.name: r for r in result.operation_results}
    q, k, v, o = by_name["q_projection"], by_name["k_projection"], by_name["v_projection"], by_name["output_projection"]
    assert q.mapping.candidate == k.mapping.candidate == v.mapping.candidate == o.mapping.candidate


def test_repetition_does_not_change_best_mapping():
    # Compare the chosen mapping for attention_scores (rep=2) against a
    # standalone single-execution exploration of the same GEMM shape.
    from tensorforge.explore import explore

    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    pe_array = PeArray(rows=2, columns=2)

    result = evaluate_transformer_block(
        spec, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2, 4],
    )
    attn = next(r for r in result.operation_results if r.name == "attention_scores")

    single = explore(
        attn.gemm, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2, 4],
        pe_rows_values=[2], pe_cols_values=[2], top_k=1,
    )
    assert attn.mapping.candidate == single.ranked[0].candidate


def test_infeasible_operation_raises_clean_error():
    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=1)  # nothing fits
    pe_array = PeArray(rows=1, columns=1)

    with pytest.raises(ValueError):
        evaluate_transformer_block(
            spec, pe_array, hierarchy, HARDWARE, TIMING,
            tile_m_values=[2], tile_n_values=[2], tile_k_values=[2],
        )


# --- PE architecture exploration across the block ---------------------------

def test_architecture_comparison_1x1_vs_2x2():
    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result = explore_transformer_architectures(
        spec, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2, 4],
        pe_rows_values=[1, 2], pe_cols_values=[1, 2],
    )

    assert result.total_architectures == 4
    assert result.feasible_architectures == 4
    pe_counts = {(a.pe_rows, a.pe_cols): a for a in result.ranked}
    assert (1, 1) in pe_counts and (2, 2) in pe_counts

    a11 = pe_counts[(1, 1)]
    a22 = pe_counts[(2, 2)]
    assert a22.block_result.perfect_overlap_time_seconds <= a11.block_result.perfect_overlap_time_seconds

    # deterministic ranking check: run twice, same order
    result2 = explore_transformer_architectures(
        spec, hierarchy, HARDWARE, TIMING,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2, 4],
        pe_rows_values=[1, 2], pe_cols_values=[1, 2],
    )
    order1 = [(a.pe_rows, a.pe_cols) for a in result.ranked]
    order2 = [(a.pe_rows, a.pe_cols) for a in result2.ranked]
    assert order1 == order2


def test_architecture_tie_break_order_matches_documented_key():
    # Extremely memory-bound: tiny bandwidth makes memory dominate for all
    # PE sizes, so every architecture ties exactly on perfect_overlap_time.
    # (Their serialized_time/compute_time still differ minutely per PE
    # shape's actual cycle count, so lower-priority tie-break keys are
    # what actually decide the order here -- verify that order is exactly
    # the documented key, not that PE count wins outright.)
    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    slow_hardware = HardwareConfig(
        peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1.0
    )
    fast_timing = TimingConfig(clock_hz=1e15)

    result = explore_transformer_architectures(
        spec, hierarchy, slow_hardware, fast_timing,
        tile_m_values=[2, 4], tile_n_values=[2, 4], tile_k_values=[2, 4],
        pe_rows_values=[1, 2, 4], pe_cols_values=[1, 2, 4],
    )
    assert result.feasible_architectures == 9
    best = result.ranked[0]
    tied = [
        a for a in result.ranked
        if a.block_result.perfect_overlap_time_seconds == best.block_result.perfect_overlap_time_seconds
    ]
    assert len(tied) == 9  # every architecture ties on the primary objective here

    expected_order = sorted(
        result.ranked,
        key=lambda a: (
            a.block_result.perfect_overlap_time_seconds,
            a.block_result.serialized_time_seconds,
            a.block_result.total_dram_bytes,
            a.pe_count,
            a.pe_rows,
            a.pe_cols,
        ),
    )
    assert list(result.ranked) == expected_order


def test_architecture_tie_prefers_smaller_pe_count_when_genuinely_tied():
    # Degenerate but valid all-1s spec: every derived GEMM is 1x1x1, so a
    # 1x1 output needs exactly one wave regardless of PE array size --
    # compute_cycles (and therefore serialized_time/dram) are IDENTICAL
    # across every PE shape here. This isolates PE count as the deciding
    # tie-break with no floating-point noise from unequal compute times.
    spec = TransformerBlockSpec(
        batch_size=1, sequence_length=1, d_model=1, num_heads=1, d_ff=1, dtype=DType.FP32
    )
    hierarchy = MemoryHierarchy(sram_bytes=10_000)

    result = explore_transformer_architectures(
        spec, hierarchy, HARDWARE, TIMING,
        tile_m_values=[1], tile_n_values=[1], tile_k_values=[1],
        pe_rows_values=[1, 2, 4], pe_cols_values=[1, 2, 4],
    )
    assert result.feasible_architectures == 9
    best = result.ranked[0]
    assert (best.pe_rows, best.pe_cols) == (1, 1)
    assert best.pe_count == 1
    assert best.pe_count == min(a.pe_count for a in result.ranked)

    # among the pe_count=2 tier, (1,2) sorts before (2,1) -- rows ascending.
    tier_two = [a for a in result.ranked if a.pe_count == 2]
    assert [(a.pe_rows, a.pe_cols) for a in tier_two] == [(1, 2), (2, 1)]


def test_architecture_search_size_limit():
    spec = tiny_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    with pytest.raises(ValueError):
        explore_transformer_architectures(
            spec, hierarchy, HARDWARE, TIMING,
            tile_m_values=list(range(1, 30)),
            tile_n_values=list(range(1, 30)),
            tile_k_values=list(range(1, 30)),
            pe_rows_values=list(range(1, 30)),
            pe_cols_values=list(range(1, 30)),
        )
