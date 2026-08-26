"""Conv2D workload: output-shape derivation + materialized-im2col GEMM lowering.

This models Conv2D by deriving its output geometry, then lowering it to a
GEMM via **materialized im2col** and delegating everything else (PE
mapping, SRAM capacity, tiling, residency schedules, timing, bounded
exploration) to the existing GEMM accelerator model. No new PE, tiling,
schedule, or timing formula is introduced here.

Materialized im2col: the convolution's input is conceptually flattened
into an "activation-patch matrix" A of shape (batch*Hout*Wout) x
(Cin*Kh*Kw), where each row holds one output position's full receptive
field, laid out contiguously. This is one explicit, simple lowering
choice among several real-hardware options (direct convolution, implicit
im2col, Winograd, FFT-based convolution) — none of which are modeled
here. Materializing im2col literally means: overlapping receptive fields
duplicate the same input pixels into multiple rows of A, so A can be
substantially larger than the logical input tensor it was derived from
(see `im2col_expansion_ratio`). Padded (zero) positions are also counted
at full size — no sparse compression of padding is modeled.

Because the existing tiling/schedule model operates on `Gemm.a_bytes`
(and its tiled variants), applying it to this lowering means: **DRAM
traffic attributed to the convolution's "A" operand is materialized
im2col traffic, not direct-convolution input traffic.** A real direct-
convolution accelerator reading overlapping windows straight from the
input tensor would generally move far less data — that is a different,
unmodeled dataflow. This distinction is central to this milestone and
must not be described as "actual CNN accelerator traffic."

Output dimensions (dilation=1, groups=1 — the only case modeled):

    output_height = (input_height + 2*padding_height - kernel_height) // stride_height + 1
    output_width  = (input_width  + 2*padding_width  - kernel_width)  // stride_width  + 1

Direct-convolution compute (independently derived, used only to
cross-check the lowered GEMM's MAC count — see tests):

    MACs  = batch_size * output_height * output_width
            * out_channels * in_channels * kernel_height * kernel_width
    FLOPs = 2 * MACs

Im2col lowering:

    M = batch_size * output_height * output_width   (one row per output position)
    N = out_channels
    K = in_channels * kernel_height * kernel_width   (flattened receptive field)

    Gemm(M, N, K)  with  Gemm.a_bytes == materialized im2col bytes
                         Gemm.b_bytes == weight bytes
                         Gemm.c_bytes == output bytes

Not modeled this milestone: dilation != 1, groups != 1 (depthwise/grouped
convolution), transposed convolution, direct/implicit convolution
dataflows, Winograd/FFT convolution, activation/normalization/pooling,
and (for CnnWorkload) cross-layer SRAM residency — each convolution layer
is evaluated independently at the DRAM boundary.
"""

from dataclasses import dataclass

from tensorforge.explore import MAX_CANDIDATES, CandidateResult, explore
from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray
from tensorforge.tiling import GemmSchedule
from tensorforge.timing import TimingConfig

UNMODELED_CNN_OPERATIONS = (
    "activation",
    "normalization",
    "pooling",
    "residual additions",
    "fully connected/classifier layers (unless represented manually as a GEMM)",
)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class Conv2DSpec:
    """A single Conv2D layer's shape. dilation=1 and groups=1 only."""

    batch_size: int
    in_channels: int
    input_height: int
    input_width: int
    out_channels: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    padding_height: int
    padding_width: int
    dtype: DType

    def __post_init__(self) -> None:
        for name in (
            "batch_size", "in_channels", "input_height", "input_width",
            "out_channels", "kernel_height", "kernel_width",
            "stride_height", "stride_width",
        ):
            _validate_positive_int(name, getattr(self, name))
        for name in ("padding_height", "padding_width"):
            _validate_nonnegative_int(name, getattr(self, name))
        # Access now so invalid (non-positive) output geometry fails at
        # construction time rather than silently producing empty output.
        _ = self.output_height
        _ = self.output_width

    @property
    def output_height(self) -> int:
        oh = (
            self.input_height + 2 * self.padding_height - self.kernel_height
        ) // self.stride_height + 1
        if oh <= 0:
            raise ValueError(f"resulting output_height must be > 0, got {oh}")
        return oh

    @property
    def output_width(self) -> int:
        ow = (
            self.input_width + 2 * self.padding_width - self.kernel_width
        ) // self.stride_width + 1
        if ow <= 0:
            raise ValueError(f"resulting output_width must be > 0, got {ow}")
        return ow

    @property
    def macs(self) -> int:
        """Direct-convolution MAC count, derived independently of the GEMM lowering."""
        return (
            self.batch_size * self.output_height * self.output_width
            * self.out_channels * self.in_channels * self.kernel_height * self.kernel_width
        )

    @property
    def flops(self) -> int:
        return 2 * self.macs


@dataclass(frozen=True)
class Conv2DLoweringResult:
    spec: Conv2DSpec
    output_height: int
    output_width: int
    gemm: Gemm

    input_bytes: int
    weight_bytes: int
    output_bytes: int
    im2col_bytes: int
    im2col_expansion_ratio: float


def lower_conv2d_to_gemm(spec: Conv2DSpec) -> Conv2DLoweringResult:
    """Pure, deterministic Conv2D -> materialized-im2col Gemm lowering.

    No tensor is actually allocated; only shapes and byte counts are
    derived.
    """
    oh, ow = spec.output_height, spec.output_width
    m = spec.batch_size * oh * ow
    n = spec.out_channels
    k = spec.in_channels * spec.kernel_height * spec.kernel_width
    gemm = Gemm(m=m, n=n, k=k, dtype=spec.dtype)

    bpe = spec.dtype.bytes_per_element
    input_bytes = spec.batch_size * spec.in_channels * spec.input_height * spec.input_width * bpe
    weight_bytes = spec.out_channels * spec.in_channels * spec.kernel_height * spec.kernel_width * bpe
    output_bytes = spec.batch_size * spec.out_channels * oh * ow * bpe
    im2col_bytes = m * k * bpe

    return Conv2DLoweringResult(
        spec=spec,
        output_height=oh,
        output_width=ow,
        gemm=gemm,
        input_bytes=input_bytes,
        weight_bytes=weight_bytes,
        output_bytes=output_bytes,
        im2col_bytes=im2col_bytes,
        im2col_expansion_ratio=im2col_bytes / input_bytes,
    )


@dataclass(frozen=True)
class Conv2DEvaluationResult:
    spec: Conv2DSpec
    lowering: Conv2DLoweringResult
    pe_array: PeArray
    mapping: CandidateResult  # candidate (tile/schedule/PE) + tiling/PE/timing results


def evaluate_conv2d(
    spec: Conv2DSpec,
    pe_array: PeArray,
    hierarchy: MemoryHierarchy,
    hardware: HardwareConfig,
    timing_config: TimingConfig,
    tile_m_values: list[int],
    tile_n_values: list[int],
    tile_k_values: list[int],
    schedules: list[GemmSchedule] | None = None,
) -> Conv2DEvaluationResult:
    """Lower Conv2D to a Gemm, then reuse the existing explorer to pick the
    best tile/schedule mapping for one fixed PE array. Raises ValueError if
    no candidate mapping fits the modeled SRAM.
    """
    lowering = lower_conv2d_to_gemm(spec)
    exploration = explore(
        lowering.gemm,
        hierarchy,
        hardware,
        timing_config,
        tile_m_values=tile_m_values,
        tile_n_values=tile_n_values,
        tile_k_values=tile_k_values,
        pe_rows_values=[pe_array.rows],
        pe_cols_values=[pe_array.columns],
        schedules=schedules,
        top_k=1,
    )
    if exploration.feasible_candidates == 0:
        raise ValueError(
            f"no feasible tile/schedule mapping for this Conv2D's materialized-im2col "
            f"GEMM ({lowering.gemm.m}x{lowering.gemm.n}x{lowering.gemm.k}) under the "
            "given SRAM capacity and tile candidates"
        )
    return Conv2DEvaluationResult(
        spec=spec, lowering=lowering, pe_array=pe_array, mapping=exploration.ranked[0]
    )


def _validate_positive_int_list(name: str, values: list[int]) -> tuple[int, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} entries must be positive ints, got {value!r}")
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class Conv2DArchitectureResult:
    pe_rows: int
    pe_cols: int
    pe_count: int
    evaluation: Conv2DEvaluationResult


@dataclass(frozen=True)
class Conv2DArchitectureExplorationResult:
    total_architectures: int
    feasible_architectures: int
    top_k: int
    ranked: tuple[Conv2DArchitectureResult, ...]


def explore_conv2d_architectures(
    spec: Conv2DSpec,
    hierarchy: MemoryHierarchy,
    hardware: HardwareConfig,
    timing_config: TimingConfig,
    tile_m_values: list[int],
    tile_n_values: list[int],
    tile_k_values: list[int],
    pe_rows_values: list[int],
    pe_cols_values: list[int],
    schedules: list[GemmSchedule] | None = None,
    top_k: int = 5,
) -> Conv2DArchitectureExplorationResult:
    """Compare candidate PE-array shapes for one Conv2D's lowered GEMM.

    Ranked by perfect_overlap_time_seconds ascending; ties broken by
    serialized_time_seconds, then DRAM traffic, then PE count, then PE
    rows, then PE columns (same deterministic order as Milestone 7/8).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    pe_rows = _validate_positive_int_list("pe_rows_values", pe_rows_values)
    pe_cols = _validate_positive_int_list("pe_cols_values", pe_cols_values)

    num_schedules = len(schedules) if schedules else 3
    per_shape_candidates = (
        len(set(tile_m_values)) * len(set(tile_n_values)) * len(set(tile_k_values)) * num_schedules
    )
    total_architectures = len(pe_rows) * len(pe_cols)
    estimated_evaluations = total_architectures * per_shape_candidates
    if estimated_evaluations > MAX_CANDIDATES:
        raise ValueError(
            f"estimated evaluation count {estimated_evaluations} exceeds the limit of "
            f"{MAX_CANDIDATES}; narrow the PE/tile/schedule candidate sets"
        )

    results: list[Conv2DArchitectureResult] = []
    for rows in pe_rows:
        for cols in pe_cols:
            pe_array = PeArray(rows=rows, columns=cols)
            try:
                evaluation = evaluate_conv2d(
                    spec, pe_array, hierarchy, hardware, timing_config,
                    tile_m_values, tile_n_values, tile_k_values, schedules,
                )
            except ValueError:
                continue
            results.append(
                Conv2DArchitectureResult(pe_rows=rows, pe_cols=cols, pe_count=rows * cols, evaluation=evaluation)
            )

    results.sort(
        key=lambda a: (
            a.evaluation.mapping.timing_result.perfect_overlap_time_seconds,
            a.evaluation.mapping.timing_result.serialized_time_seconds,
            a.evaluation.mapping.timing_result.dram_bytes,
            a.pe_count,
            a.pe_rows,
            a.pe_cols,
        )
    )

    return Conv2DArchitectureExplorationResult(
        total_architectures=total_architectures,
        feasible_architectures=len(results),
        top_k=top_k,
        ranked=tuple(results),
    )


# --- Conv-only CNN workload (chained layer geometry) ------------------------


@dataclass(frozen=True)
class CnnLayerSpec:
    """One convolution layer's configuration. Input shape/channels are
    derived automatically by chaining from the previous layer (or the
    workload's input shape, for the first layer) -- see CnnWorkload.
    """

    name: str
    out_channels: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    padding_height: int
    padding_width: int


@dataclass(frozen=True)
class CnnWorkload:
    """An ordered, Conv-only CNN: each layer's input is the previous
    layer's output (or the workload's input shape, for the first layer).
    This is a deliberately simple chained-geometry model -- there is no
    support yet for branches, pooling-induced resizing, or independently
    specified per-layer input shapes.
    """

    batch_size: int
    in_channels: int
    input_height: int
    input_width: int
    dtype: DType
    layers: tuple[CnnLayerSpec, ...]

    def __post_init__(self) -> None:
        if len(self.layers) == 0:
            raise ValueError("CnnWorkload.layers must not be empty")


def derive_cnn_conv_specs(workload: CnnWorkload) -> tuple[Conv2DSpec, ...]:
    """Chain each layer's Conv2DSpec from the previous layer's output shape."""
    specs = []
    in_channels = workload.in_channels
    height, width = workload.input_height, workload.input_width
    for layer in workload.layers:
        spec = Conv2DSpec(
            batch_size=workload.batch_size,
            in_channels=in_channels,
            input_height=height,
            input_width=width,
            out_channels=layer.out_channels,
            kernel_height=layer.kernel_height,
            kernel_width=layer.kernel_width,
            stride_height=layer.stride_height,
            stride_width=layer.stride_width,
            padding_height=layer.padding_height,
            padding_width=layer.padding_width,
            dtype=workload.dtype,
        )
        specs.append(spec)
        in_channels = layer.out_channels
        height, width = spec.output_height, spec.output_width
    return tuple(specs)


@dataclass(frozen=True)
class CnnLayerResult:
    name: str
    spec: Conv2DSpec
    evaluation: Conv2DEvaluationResult


@dataclass(frozen=True)
class CnnWorkloadResult:
    workload: CnnWorkload
    pe_array: PeArray
    layer_results: tuple[CnnLayerResult, ...]

    total_macs: int
    total_flops: int
    total_dram_bytes: int

    serialized_time_seconds: float
    perfect_overlap_time_seconds: float

    largest_time_contributor: str
    largest_dram_contributor: str

    unmodeled_operations: tuple[str, ...]


def evaluate_cnn_workload(
    workload: CnnWorkload,
    pe_array: PeArray,
    hierarchy: MemoryHierarchy,
    hardware: HardwareConfig,
    timing_config: TimingConfig,
    tile_m_values: list[int],
    tile_n_values: list[int],
    tile_k_values: list[int],
    schedules: list[GemmSchedule] | None = None,
) -> CnnWorkloadResult:
    """Evaluate every layer of a CnnWorkload on one fixed PE array, letting
    each layer choose its own best tile/schedule mapping. Layers execute
    sequentially with no cross-layer SRAM residency: each is evaluated
    independently at the DRAM boundary, so block traffic is the sum of
    independently-evaluated layers (a layer's output being kept resident
    in SRAM for the next layer's input is NOT modeled).
    """
    conv_specs = derive_cnn_conv_specs(workload)

    layer_results = []
    for layer, spec in zip(workload.layers, conv_specs):
        evaluation = evaluate_conv2d(
            spec, pe_array, hierarchy, hardware, timing_config,
            tile_m_values, tile_n_values, tile_k_values, schedules,
        )
        layer_results.append(CnnLayerResult(name=layer.name, spec=spec, evaluation=evaluation))

    total_macs = sum(lr.spec.macs for lr in layer_results)
    total_flops = sum(lr.spec.flops for lr in layer_results)
    total_dram_bytes = sum(lr.evaluation.mapping.timing_result.dram_bytes for lr in layer_results)
    serialized_time_seconds = sum(
        lr.evaluation.mapping.timing_result.serialized_time_seconds for lr in layer_results
    )
    perfect_overlap_time_seconds = sum(
        lr.evaluation.mapping.timing_result.perfect_overlap_time_seconds for lr in layer_results
    )

    largest_time = max(
        layer_results, key=lambda lr: lr.evaluation.mapping.timing_result.perfect_overlap_time_seconds
    )
    largest_dram = max(layer_results, key=lambda lr: lr.evaluation.mapping.timing_result.dram_bytes)

    return CnnWorkloadResult(
        workload=workload,
        pe_array=pe_array,
        layer_results=tuple(layer_results),
        total_macs=total_macs,
        total_flops=total_flops,
        total_dram_bytes=total_dram_bytes,
        serialized_time_seconds=serialized_time_seconds,
        perfect_overlap_time_seconds=perfect_overlap_time_seconds,
        largest_time_contributor=largest_time.name,
        largest_dram_contributor=largest_dram.name,
        unmodeled_operations=UNMODELED_CNN_OPERATIONS,
    )


@dataclass(frozen=True)
class CnnArchitectureResult:
    pe_rows: int
    pe_cols: int
    pe_count: int
    workload_result: CnnWorkloadResult


@dataclass(frozen=True)
class CnnArchitectureExplorationResult:
    total_architectures: int
    feasible_architectures: int
    top_k: int
    ranked: tuple[CnnArchitectureResult, ...]


def explore_cnn_architectures(
    workload: CnnWorkload,
    hierarchy: MemoryHierarchy,
    hardware: HardwareConfig,
    timing_config: TimingConfig,
    tile_m_values: list[int],
    tile_n_values: list[int],
    tile_k_values: list[int],
    pe_rows_values: list[int],
    pe_cols_values: list[int],
    schedules: list[GemmSchedule] | None = None,
    top_k: int = 5,
) -> CnnArchitectureExplorationResult:
    """Compare candidate PE-array shapes across the whole CNN workload.
    The PE array is fixed across all layers of one candidate; ranking uses
    the same deterministic key as explore_conv2d_architectures/Transformer
    architecture exploration.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    pe_rows = _validate_positive_int_list("pe_rows_values", pe_rows_values)
    pe_cols = _validate_positive_int_list("pe_cols_values", pe_cols_values)

    num_schedules = len(schedules) if schedules else 3
    per_shape_candidates = (
        len(set(tile_m_values)) * len(set(tile_n_values)) * len(set(tile_k_values)) * num_schedules
    )
    total_architectures = len(pe_rows) * len(pe_cols)
    estimated_evaluations = total_architectures * len(workload.layers) * per_shape_candidates
    if estimated_evaluations > MAX_CANDIDATES:
        raise ValueError(
            f"estimated evaluation count {estimated_evaluations} exceeds the limit of "
            f"{MAX_CANDIDATES}; narrow the PE/tile/schedule candidate sets"
        )

    results: list[CnnArchitectureResult] = []
    for rows in pe_rows:
        for cols in pe_cols:
            pe_array = PeArray(rows=rows, columns=cols)
            try:
                workload_result = evaluate_cnn_workload(
                    workload, pe_array, hierarchy, hardware, timing_config,
                    tile_m_values, tile_n_values, tile_k_values, schedules,
                )
            except ValueError:
                continue
            results.append(
                CnnArchitectureResult(
                    pe_rows=rows, pe_cols=cols, pe_count=rows * cols, workload_result=workload_result
                )
            )

    results.sort(
        key=lambda a: (
            a.workload_result.perfect_overlap_time_seconds,
            a.workload_result.serialized_time_seconds,
            a.workload_result.total_dram_bytes,
            a.pe_count,
            a.pe_rows,
            a.pe_cols,
        )
    )

    return CnnArchitectureExplorationResult(
        total_architectures=total_architectures,
        feasible_architectures=len(results),
        top_k=top_k,
        ranked=tuple(results),
    )
