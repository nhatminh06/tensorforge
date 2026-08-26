"""Transformer GEMM-block workload: decomposition + fixed-hardware evaluation.

This module derives the GEMM operations that make up the matrix-multiply
portion of one Transformer block, and evaluates them on one shared,
fixed accelerator (PE array, SRAM, clock, DRAM bandwidth) using the
existing bounded explorer (explore.py) to pick a per-operation tile/
schedule mapping. It introduces no new GEMM, tiling, PE, or timing
formulas — it only derives GEMM shapes and orchestrates the existing
models across several of them.

The resulting timing is a **GEMM-only Transformer block estimate**, never
"complete Transformer block latency": softmax, normalization (LayerNorm/
RMSNorm), activation functions, and residual additions are explicitly
UNMODELED (not "free" — simply absent from the aggregated numbers).

Derived operations, in block execution order:

    X: (B*S) x D, Wq/Wk/Wv/Wo: D x D, W_up: D x F, W_down: F x D

    q_projection       M=B*S, N=D,        K=D,        repetitions=1
    k_projection       M=B*S, N=D,        K=D,        repetitions=1
    v_projection       M=B*S, N=D,        K=D,        repetitions=1
    attention_scores   M=S,   N=S,        K=head_dim, repetitions=B*H
    attention_value    M=S,   N=head_dim, K=S,        repetitions=B*H
    output_projection  M=B*S, N=D,        K=D,        repetitions=1
    mlp_up             M=B*S, N=F,        K=D,        repetitions=1
    mlp_down           M=B*S, N=D,        K=F,        repetitions=1

Attention repetitions (B*H) model B batches x H independent attention
heads. This milestone assumes they execute SEQUENTIALLY on the modeled
accelerator: aggregate time/traffic for a repeated operation is exactly
(single-execution value) x repetitions. No head parallelism, no cross-op
SRAM residency (each GEMM is evaluated independently at the DRAM
boundary — e.g. Q projection's output being kept resident for the
attention-score GEMM is NOT modeled), and no fusion (QKV stays three
separate GEMMs; attention stays two separate GEMMs with softmax excluded
between them) are modeled.

Modeled GEMM FLOPs simplify (since H * head_dim = D) to:

    8*B*S*D^2   (Q + K + V + output projections, 4 * 2*B*S*D^2)
  + 4*B*S*D*F   (MLP up + down, 2*B*S*D*F + 2*B*S*F*D)
  + 4*B*S^2*D   (attention scores + value, each 2*B*S^2*D)

Fixed hardware, per-operation mapping: within one block evaluation the
PE array is identical for every operation (it represents the physical
accelerator), but tile shape and residency schedule may be chosen
independently per operation from the same user-supplied candidate sets,
since Transformer GEMM shapes differ substantially (e.g. attention's
S x head_dim x S vs MLP's B*S x D x F). Each unique GEMM shape (by
M, N, K, dtype) is explored via `explore.explore()` exactly once and the
result reused for every operation sharing that shape (Q/K/V/output
projections share M=B*S, N=D, K=D) or repetition — repetition does not
change which mapping is best, since it multiplies every candidate's time
by the same positive constant.
"""

from dataclasses import dataclass

from tensorforge.explore import MAX_CANDIDATES, CandidateResult, explore
from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray
from tensorforge.tiling import GemmSchedule
from tensorforge.timing import TimingConfig

UNMODELED_OPERATIONS = ("softmax", "normalization", "activation", "residual additions")

MAX_ARCHITECTURE_EVALUATIONS = MAX_CANDIDATES


@dataclass(frozen=True)
class TransformerBlockSpec:
    """B/S/D/H/F dimensions of one Transformer block's GEMM workload."""

    batch_size: int
    sequence_length: int
    d_model: int
    num_heads: int
    d_ff: int
    dtype: DType

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("sequence_length", self.sequence_length),
            ("d_model", self.d_model),
            ("num_heads", self.num_heads),
            ("d_ff", self.d_ff),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


@dataclass(frozen=True)
class TransformerGemmOp:
    name: str
    gemm: Gemm
    repetitions: int


def derive_transformer_gemms(spec: TransformerBlockSpec) -> tuple[TransformerGemmOp, ...]:
    """Derive the 8 GEMM operation groups for one block, in execution order."""
    bs = spec.batch_size * spec.sequence_length
    d, f, s, head_dim = spec.d_model, spec.d_ff, spec.sequence_length, spec.head_dim
    dtype = spec.dtype
    head_repetitions = spec.batch_size * spec.num_heads

    return (
        TransformerGemmOp("q_projection", Gemm(m=bs, n=d, k=d, dtype=dtype), 1),
        TransformerGemmOp("k_projection", Gemm(m=bs, n=d, k=d, dtype=dtype), 1),
        TransformerGemmOp("v_projection", Gemm(m=bs, n=d, k=d, dtype=dtype), 1),
        TransformerGemmOp(
            "attention_scores", Gemm(m=s, n=s, k=head_dim, dtype=dtype), head_repetitions
        ),
        TransformerGemmOp(
            "attention_value", Gemm(m=s, n=head_dim, k=s, dtype=dtype), head_repetitions
        ),
        TransformerGemmOp("output_projection", Gemm(m=bs, n=d, k=d, dtype=dtype), 1),
        TransformerGemmOp("mlp_up", Gemm(m=bs, n=f, k=d, dtype=dtype), 1),
        TransformerGemmOp("mlp_down", Gemm(m=bs, n=d, k=f, dtype=dtype), 1),
    )


def block_total_macs(ops: tuple[TransformerGemmOp, ...]) -> int:
    return sum(op.gemm.macs * op.repetitions for op in ops)


def block_total_flops(ops: tuple[TransformerGemmOp, ...]) -> int:
    return sum(op.gemm.flops * op.repetitions for op in ops)


@dataclass(frozen=True)
class OperationResult:
    name: str
    gemm: Gemm
    repetitions: int
    mapping: CandidateResult  # per-execution: candidate (tile/schedule/PE) + tiling/PE/timing results

    aggregate_perfect_overlap_time_seconds: float
    aggregate_serialized_time_seconds: float
    aggregate_dram_bytes: int
    aggregate_flops: int


@dataclass(frozen=True)
class TransformerBlockResult:
    spec: TransformerBlockSpec
    pe_array: PeArray
    operation_results: tuple[OperationResult, ...]

    total_macs: int
    total_flops: int
    total_dram_bytes: int

    serialized_time_seconds: float
    perfect_overlap_time_seconds: float

    largest_time_contributor: str
    largest_dram_contributor: str

    unmodeled_operations: tuple[str, ...]


def evaluate_transformer_block(
    spec: TransformerBlockSpec,
    pe_array: PeArray,
    hierarchy: MemoryHierarchy,
    hardware: HardwareConfig,
    timing_config: TimingConfig,
    tile_m_values: list[int],
    tile_n_values: list[int],
    tile_k_values: list[int],
    schedules: list[GemmSchedule] | None = None,
) -> TransformerBlockResult:
    """Evaluate every derived GEMM on one fixed PE array, letting each pick
    its own best tile/schedule mapping from the shared candidate sets.

    Raises ValueError if any operation's GEMM shape has no feasible
    mapping under the given SRAM/tile candidates (propagated from
    explore()) — the block result is only meaningful if every operation
    can actually run.
    """
    ops = derive_transformer_gemms(spec)

    cache: dict[tuple[int, int, int, DType], CandidateResult] = {}
    operation_results = []
    for op in ops:
        key = (op.gemm.m, op.gemm.n, op.gemm.k, op.gemm.dtype)
        if key not in cache:
            exploration = explore(
                op.gemm,
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
                    f"no feasible tile/schedule mapping for operation {op.name!r} "
                    f"(shape {op.gemm.m}x{op.gemm.n}x{op.gemm.k}) under the given "
                    "SRAM capacity and tile candidates"
                )
            cache[key] = exploration.ranked[0]

        best = cache[key]
        t = best.timing_result
        operation_results.append(
            OperationResult(
                name=op.name,
                gemm=op.gemm,
                repetitions=op.repetitions,
                mapping=best,
                aggregate_perfect_overlap_time_seconds=t.perfect_overlap_time_seconds
                * op.repetitions,
                aggregate_serialized_time_seconds=t.serialized_time_seconds * op.repetitions,
                aggregate_dram_bytes=t.dram_bytes * op.repetitions,
                aggregate_flops=op.gemm.flops * op.repetitions,
            )
        )

    total_macs = block_total_macs(ops)
    total_flops = sum(r.aggregate_flops for r in operation_results)
    total_dram_bytes = sum(r.aggregate_dram_bytes for r in operation_results)
    serialized_time_seconds = sum(r.aggregate_serialized_time_seconds for r in operation_results)
    perfect_overlap_time_seconds = sum(
        r.aggregate_perfect_overlap_time_seconds for r in operation_results
    )

    # max() is stable: ties resolve to the first operation in block execution order.
    largest_time = max(operation_results, key=lambda r: r.aggregate_perfect_overlap_time_seconds)
    largest_dram = max(operation_results, key=lambda r: r.aggregate_dram_bytes)

    return TransformerBlockResult(
        spec=spec,
        pe_array=pe_array,
        operation_results=tuple(operation_results),
        total_macs=total_macs,
        total_flops=total_flops,
        total_dram_bytes=total_dram_bytes,
        serialized_time_seconds=serialized_time_seconds,
        perfect_overlap_time_seconds=perfect_overlap_time_seconds,
        largest_time_contributor=largest_time.name,
        largest_dram_contributor=largest_dram.name,
        unmodeled_operations=UNMODELED_OPERATIONS,
    )


@dataclass(frozen=True)
class ArchitectureResult:
    pe_rows: int
    pe_cols: int
    pe_count: int
    block_result: TransformerBlockResult


@dataclass(frozen=True)
class ArchitectureExplorationResult:
    total_architectures: int
    feasible_architectures: int
    top_k: int
    ranked: tuple[ArchitectureResult, ...]


def _validate_positive_int_list(name: str, values: list[int]) -> tuple[int, ...]:
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} entries must be positive ints, got {value!r}")
    return tuple(sorted(set(values)))


def explore_transformer_architectures(
    spec: TransformerBlockSpec,
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
) -> ArchitectureExplorationResult:
    """Compare candidate PE-array shapes across the whole Transformer block.

    For each candidate PE array, every block GEMM independently picks its
    own best tile/schedule mapping (same candidate sets, same PE array).
    Ranked by block perfect_overlap_time_seconds ascending; ties broken by
    block serialized time, then block DRAM traffic, then PE count, then
    PE rows, then PE columns — the same style of deterministic tie order
    used by explore.py's single-GEMM ranking.

    Raises ValueError before evaluating anything if the estimated total
    work (architectures x unique GEMM shapes x per-shape candidates)
    exceeds MAX_ARCHITECTURE_EVALUATIONS.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    pe_rows = _validate_positive_int_list("pe_rows_values", pe_rows_values)
    pe_cols = _validate_positive_int_list("pe_cols_values", pe_cols_values)

    num_unique_shapes = len({(op.gemm.m, op.gemm.n, op.gemm.k) for op in derive_transformer_gemms(spec)})
    num_schedules = len(schedules) if schedules else 3
    per_shape_candidates = (
        len(set(tile_m_values)) * len(set(tile_n_values)) * len(set(tile_k_values)) * num_schedules
    )
    total_architectures = len(pe_rows) * len(pe_cols)
    estimated_evaluations = total_architectures * num_unique_shapes * per_shape_candidates
    if estimated_evaluations > MAX_ARCHITECTURE_EVALUATIONS:
        raise ValueError(
            f"estimated evaluation count {estimated_evaluations} exceeds the limit of "
            f"{MAX_ARCHITECTURE_EVALUATIONS}; narrow the PE/tile/schedule candidate sets"
        )

    results: list[ArchitectureResult] = []
    for rows in pe_rows:
        for cols in pe_cols:
            pe_array = PeArray(rows=rows, columns=cols)
            try:
                block_result = evaluate_transformer_block(
                    spec, pe_array, hierarchy, hardware, timing_config,
                    tile_m_values, tile_n_values, tile_k_values, schedules,
                )
            except ValueError:
                continue
            results.append(
                ArchitectureResult(pe_rows=rows, pe_cols=cols, pe_count=rows * cols, block_result=block_result)
            )

    results.sort(
        key=lambda a: (
            a.block_result.perfect_overlap_time_seconds,
            a.block_result.serialized_time_seconds,
            a.block_result.total_dram_bytes,
            a.pe_count,
            a.pe_rows,
            a.pe_cols,
        )
    )

    return ArchitectureExplorationResult(
        total_architectures=total_architectures,
        feasible_architectures=len(results),
        top_k=top_k,
        ranked=tuple(results),
    )
