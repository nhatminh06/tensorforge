import math

import pytest

from tensorforge.convolution import (
    CnnLayerSpec,
    CnnWorkload,
    Conv2DSpec,
    derive_cnn_conv_specs,
    evaluate_cnn_workload,
    evaluate_conv2d,
    explore_conv2d_architectures,
    lower_conv2d_to_gemm,
)
from tensorforge.explore import explore
from tensorforge.gemm import DType
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray
from tensorforge.timing import TimingConfig

HARDWARE = HardwareConfig(peak_compute_flops_per_second=1e12, memory_bandwidth_bytes_per_second=1e9)
TIMING = TimingConfig(clock_hz=1e9)


def make_spec(**overrides):
    defaults = dict(
        batch_size=1, in_channels=3, input_height=5, input_width=5,
        out_channels=4, kernel_height=3, kernel_width=3,
        stride_height=1, stride_width=1, padding_height=0, padding_width=0,
        dtype=DType.FP32,
    )
    defaults.update(overrides)
    return Conv2DSpec(**defaults)


# --- spec validation ---------------------------------------------------------

@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0), ("in_channels", 0), ("input_height", 0), ("input_width", 0),
        ("out_channels", 0), ("kernel_height", 0), ("kernel_width", 0),
        ("stride_height", 0), ("stride_width", 0),
    ],
)
def test_rejects_nonpositive_dimensions(field, value):
    with pytest.raises(ValueError):
        make_spec(**{field: value})


@pytest.mark.parametrize("field", ["padding_height", "padding_width"])
def test_rejects_negative_padding(field):
    with pytest.raises(ValueError):
        make_spec(**{field: -1})


def test_zero_padding_allowed():
    spec = make_spec(padding_height=0, padding_width=0)
    assert spec.padding_height == 0


def test_invalid_output_size_rejected():
    # 2x2 input, 5x5 kernel, no padding -> non-positive output.
    with pytest.raises(ValueError):
        make_spec(input_height=2, input_width=2, kernel_height=5, kernel_width=5, padding_height=0, padding_width=0)


# --- output dimensions --------------------------------------------------------

def test_padding_preserves_spatial_size():
    # H=W=5, kernel=3, stride=1, padding=1 -> Hout=Wout=5
    spec = make_spec(input_height=5, input_width=5, kernel_height=3, kernel_width=3,
                      stride_height=1, stride_width=1, padding_height=1, padding_width=1)
    assert spec.output_height == 5
    assert spec.output_width == 5


def test_stride_reduces_output():
    # H=W=7, kernel=3, stride=2, padding=1 -> ((7+2-3)//2)+1 = 4
    spec = make_spec(input_height=7, input_width=7, kernel_height=3, kernel_width=3,
                      stride_height=2, stride_width=2, padding_height=1, padding_width=1)
    assert spec.output_height == 4
    assert spec.output_width == 4


# --- known-value 1x1 and 3x3 lowering -----------------------------------------

def test_known_1x1_conv_lowering():
    spec = Conv2DSpec(
        batch_size=1, in_channels=3, input_height=4, input_width=4,
        out_channels=8, kernel_height=1, kernel_width=1,
        stride_height=1, stride_width=1, padding_height=0, padding_width=0,
        dtype=DType.FP32,
    )
    result = lower_conv2d_to_gemm(spec)

    assert result.output_height == 4
    assert result.output_width == 4
    assert result.gemm.m == 16
    assert result.gemm.n == 8
    assert result.gemm.k == 3

    assert result.gemm.macs == 384
    assert result.gemm.flops == 768
    assert spec.macs == 384
    assert spec.flops == 768

    assert result.input_bytes == 48 * 4  # 1*3*4*4 elements * 4B
    assert result.im2col_bytes == 16 * 3 * 4  # M*K elements * 4B
    assert result.im2col_bytes == 48 * 4
    assert math.isclose(result.im2col_expansion_ratio, 1.0)


def test_known_3x3_conv_lowering():
    spec = Conv2DSpec(
        batch_size=1, in_channels=3, input_height=5, input_width=5,
        out_channels=4, kernel_height=3, kernel_width=3,
        stride_height=1, stride_width=1, padding_height=0, padding_width=0,
        dtype=DType.FP32,
    )
    result = lower_conv2d_to_gemm(spec)

    assert result.output_height == 3
    assert result.output_width == 3
    assert result.gemm.m == 9
    assert result.gemm.n == 4
    assert result.gemm.k == 27

    assert result.gemm.macs == 972
    assert result.gemm.flops == 1944
    assert spec.macs == 972

    assert result.input_bytes == 75 * 4
    assert result.im2col_bytes == 243 * 4
    assert math.isclose(result.im2col_expansion_ratio, 243 / 75)
    assert math.isclose(result.im2col_expansion_ratio, 3.24)


# --- MAC consistency and byte correspondence ----------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_size=1, in_channels=3, input_height=5, input_width=5, out_channels=4,
             kernel_height=3, kernel_width=3, stride_height=1, stride_width=1,
             padding_height=0, padding_width=0),
        dict(batch_size=2, in_channels=8, input_height=16, input_width=16, out_channels=16,
             kernel_height=5, kernel_width=5, stride_height=2, stride_width=2,
             padding_height=2, padding_width=2),
        dict(batch_size=1, in_channels=1, input_height=8, input_width=8, out_channels=1,
             kernel_height=1, kernel_width=1, stride_height=1, stride_width=1,
             padding_height=0, padding_width=0),
    ],
)
def test_direct_macs_equal_lowered_gemm_macs(kwargs):
    spec = Conv2DSpec(dtype=DType.FP16, **kwargs)
    result = lower_conv2d_to_gemm(spec)
    assert spec.macs == result.gemm.macs
    assert spec.flops == result.gemm.flops


def test_weight_bytes_equal_gemm_b_bytes():
    spec = make_spec()
    result = lower_conv2d_to_gemm(spec)
    assert result.weight_bytes == result.gemm.b_bytes


def test_output_bytes_equal_gemm_c_bytes():
    spec = make_spec()
    result = lower_conv2d_to_gemm(spec)
    assert result.output_bytes == result.gemm.c_bytes


def test_im2col_bytes_equal_gemm_a_bytes():
    spec = make_spec()
    result = lower_conv2d_to_gemm(spec)
    assert result.im2col_bytes == result.gemm.a_bytes


# --- dtype behavior ------------------------------------------------------------

def test_dtype_does_not_change_shapes_or_operations():
    kwargs = dict(
        batch_size=1, in_channels=3, input_height=5, input_width=5, out_channels=4,
        kernel_height=3, kernel_width=3, stride_height=1, stride_width=1,
        padding_height=0, padding_width=0,
    )
    fp32 = lower_conv2d_to_gemm(Conv2DSpec(dtype=DType.FP32, **kwargs))
    fp16 = lower_conv2d_to_gemm(Conv2DSpec(dtype=DType.FP16, **kwargs))
    int8 = lower_conv2d_to_gemm(Conv2DSpec(dtype=DType.INT8, **kwargs))

    for r in (fp32, fp16, int8):
        assert (r.output_height, r.output_width) == (3, 3)
        assert (r.gemm.m, r.gemm.n, r.gemm.k) == (9, 4, 27)
        assert r.gemm.macs == 972


def test_dtype_changes_byte_footprints():
    kwargs = dict(
        batch_size=1, in_channels=3, input_height=5, input_width=5, out_channels=4,
        kernel_height=3, kernel_width=3, stride_height=1, stride_width=1,
        padding_height=0, padding_width=0,
    )
    fp32 = lower_conv2d_to_gemm(Conv2DSpec(dtype=DType.FP32, **kwargs))
    int8 = lower_conv2d_to_gemm(Conv2DSpec(dtype=DType.INT8, **kwargs))

    assert fp32.im2col_bytes == int8.im2col_bytes * 4
    assert fp32.input_bytes == int8.input_bytes * 4
    # ratio is dtype-independent (both numerator/denominator scale together)
    assert math.isclose(fp32.im2col_expansion_ratio, int8.im2col_expansion_ratio)


# --- single Conv2D evaluation --------------------------------------------------

def test_conv2d_evaluation_matches_direct_gemm_exploration():
    spec = make_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    pe_array = PeArray(rows=2, columns=2)
    tile_kwargs = dict(tile_m_values=[3, 9], tile_n_values=[2, 4], tile_k_values=[9, 27])

    conv_eval = evaluate_conv2d(spec, pe_array, hierarchy, HARDWARE, TIMING, **tile_kwargs)

    lowering = lower_conv2d_to_gemm(spec)
    direct = explore(
        lowering.gemm, hierarchy, HARDWARE, TIMING,
        pe_rows_values=[2], pe_cols_values=[2], top_k=1, **tile_kwargs,
    )

    assert conv_eval.mapping.candidate == direct.ranked[0].candidate
    assert conv_eval.mapping.timing_result.dram_bytes == direct.ranked[0].timing_result.dram_bytes
    assert (
        conv_eval.mapping.timing_result.perfect_overlap_time_seconds
        == direct.ranked[0].timing_result.perfect_overlap_time_seconds
    )


def test_conv2d_evaluator_uses_materialized_im2col_bytes_not_logical_input():
    # A must reflect im2col_bytes (972B for 3x3 case), not the smaller
    # logical input_bytes (300B) -- verifying the schedule model operates
    # on the materialized lowering, not a substituted "cheaper" value.
    spec = make_spec()
    lowering = lower_conv2d_to_gemm(spec)
    assert lowering.im2col_bytes == 972
    assert lowering.input_bytes == 300
    assert lowering.gemm.a_bytes == 972  # NOT 300

    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    pe_array = PeArray(rows=1, columns=1)
    conv_eval = evaluate_conv2d(
        spec, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[9], tile_n_values=[4], tile_k_values=[27],
    )
    # C-resident (default-ish choice among candidates) A-read traffic must
    # be derived from the 972B im2col matrix, not the 300B logical input.
    assert conv_eval.mapping.tiling_result.a_dram_read_bytes >= 972


# --- Conv2D PE architecture exploration ----------------------------------------

def test_conv2d_architecture_exploration_deterministic():
    spec = make_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    kwargs = dict(tile_m_values=[3, 9], tile_n_values=[2, 4], tile_k_values=[9, 27])

    r1 = explore_conv2d_architectures(
        spec, hierarchy, HARDWARE, TIMING, pe_rows_values=[1, 2], pe_cols_values=[1, 2], **kwargs
    )
    r2 = explore_conv2d_architectures(
        spec, hierarchy, HARDWARE, TIMING, pe_rows_values=[1, 2], pe_cols_values=[1, 2], **kwargs
    )

    assert r1.total_architectures == 4
    assert r1.feasible_architectures == 4
    order1 = [(a.pe_rows, a.pe_cols) for a in r1.ranked]
    order2 = [(a.pe_rows, a.pe_cols) for a in r2.ranked]
    assert order1 == order2


def test_conv2d_architecture_search_size_limit():
    spec = make_spec()
    hierarchy = MemoryHierarchy(sram_bytes=10_000)
    with pytest.raises(ValueError):
        explore_conv2d_architectures(
            spec, hierarchy, HARDWARE, TIMING,
            tile_m_values=list(range(1, 30)), tile_n_values=list(range(1, 30)),
            tile_k_values=list(range(1, 30)),
            pe_rows_values=list(range(1, 30)), pe_cols_values=list(range(1, 30)),
        )


# --- tiny CNN workload ----------------------------------------------------------

def tiny_cnn_workload() -> CnnWorkload:
    return CnnWorkload(
        batch_size=1, in_channels=3, input_height=8, input_width=8, dtype=DType.FP32,
        layers=(
            CnnLayerSpec("conv1", out_channels=4, kernel_height=3, kernel_width=3,
                         stride_height=1, stride_width=1, padding_height=1, padding_width=1),
            CnnLayerSpec("conv2", out_channels=8, kernel_height=3, kernel_width=3,
                         stride_height=2, stride_width=2, padding_height=1, padding_width=1),
        ),
    )


def test_tiny_cnn_layer_shapes():
    workload = tiny_cnn_workload()
    specs = derive_cnn_conv_specs(workload)

    assert len(specs) == 2
    # conv1: 8x8 -> 8x8 (padding=1, stride=1, kernel=3)
    assert specs[0].in_channels == 3
    assert (specs[0].output_height, specs[0].output_width) == (8, 8)
    assert specs[0].out_channels == 4

    # conv2: input is conv1's output (4 channels, 8x8) -> stride 2 halves it
    assert specs[1].in_channels == 4
    assert specs[1].input_height == 8
    assert specs[1].input_width == 8
    assert (specs[1].output_height, specs[1].output_width) == (4, 4)
    assert specs[1].out_channels == 8


def test_tiny_cnn_aggregate_macs_flops():
    workload = tiny_cnn_workload()
    specs = derive_cnn_conv_specs(workload)

    # conv1: batch=1,Hout=8,Wout=8,Cout=4,Cin=3,Kh=Kw=3 -> 1*8*8*4*3*3*3
    expected_macs_1 = 1 * 8 * 8 * 4 * 3 * 3 * 3
    # conv2: batch=1,Hout=4,Wout=4,Cout=8,Cin=4,Kh=Kw=3 -> 1*4*4*8*4*3*3
    expected_macs_2 = 1 * 4 * 4 * 8 * 4 * 3 * 3

    assert specs[0].macs == expected_macs_1
    assert specs[1].macs == expected_macs_2

    total_expected_macs = expected_macs_1 + expected_macs_2
    total_expected_flops = 2 * total_expected_macs
    assert sum(s.macs for s in specs) == total_expected_macs
    assert sum(s.flops for s in specs) == total_expected_flops


def test_cnn_workload_evaluation_aggregation():
    workload = tiny_cnn_workload()
    hierarchy = MemoryHierarchy(sram_bytes=1_000_000)
    pe_array = PeArray(rows=2, columns=2)

    result = evaluate_cnn_workload(
        workload, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[4, 8, 16], tile_n_values=[4, 8], tile_k_values=[4, 8, 16, 27, 36],
    )

    assert len(result.layer_results) == 2
    assert [lr.name for lr in result.layer_results] == ["conv1", "conv2"]

    specs = derive_cnn_conv_specs(workload)
    assert result.total_macs == sum(s.macs for s in specs)
    assert result.total_flops == sum(s.flops for s in specs)

    expected_dram = sum(
        lr.evaluation.mapping.timing_result.dram_bytes for lr in result.layer_results
    )
    expected_serialized = sum(
        lr.evaluation.mapping.timing_result.serialized_time_seconds for lr in result.layer_results
    )
    expected_perfect_overlap = sum(
        lr.evaluation.mapping.timing_result.perfect_overlap_time_seconds for lr in result.layer_results
    )
    assert result.total_dram_bytes == expected_dram
    assert result.serialized_time_seconds == expected_serialized
    assert result.perfect_overlap_time_seconds == expected_perfect_overlap

    assert result.unmodeled_operations == (
        "activation", "normalization", "pooling", "residual additions",
        "fully connected/classifier layers (unless represented manually as a GEMM)",
    )


def test_cnn_fixed_pe_but_per_layer_mapping_may_differ():
    workload = tiny_cnn_workload()
    hierarchy = MemoryHierarchy(sram_bytes=1_000_000)
    pe_array = PeArray(rows=2, columns=2)

    result = evaluate_cnn_workload(
        workload, pe_array, hierarchy, HARDWARE, TIMING,
        tile_m_values=[4, 8, 16], tile_n_values=[4, 8], tile_k_values=[4, 8, 16, 27, 36],
    )

    for lr in result.layer_results:
        assert lr.evaluation.pe_array == pe_array  # fixed hardware across layers

    # Different GEMM shapes are free to select different tile/schedule mappings.
    candidates = [lr.evaluation.mapping.candidate for lr in result.layer_results]
    assert len(candidates) == 2  # just confirms both layers produced a mapping


def test_cnn_empty_layers_rejected():
    with pytest.raises(ValueError):
        CnnWorkload(batch_size=1, in_channels=3, input_height=8, input_width=8, dtype=DType.FP32, layers=())
