"""Backend-agnostic real-benchmark configuration, timing loop, statistics,
and result schema.

This module imports neither PyTorch nor ONNX Runtime -- it is safe to
import even when no benchmark runtime is installed (see
`tensorforge_ops.cli`, which only imports `benchmark_pytorch` /
`benchmark_onnx` lazily, inside the `benchmark` subcommand handler).
Actual device execution lives in `benchmark_pytorch.py` /
`benchmark_onnx.py`, which call back into the pure functions here.

Predicted vs. measured: TensorForge Core (`tensorforge.experiments`)
produces a *predicted/analytical* result -- no code here ever executes
real computation to produce `predicted_*` fields, and nothing in
`tensorforge.experiments` executes real computation either. Every field
this module produces is explicitly named `measured_*` downstream (see
`tracking.py`) to keep the two domains unambiguous.

Timing methodology (documented once, used identically by every backend):
    for warmup_iterations:
        fn()                       # + sync_fn() if the device is async
    for measured_iterations:
        sync_fn()                  # drain any async work left by warmup/prior iteration
        start = time.perf_counter()
        fn()
        sync_fn()                  # force completion before stopping the clock
        end = time.perf_counter()
        samples.append(end - start)

`sync_fn` is `None` for CPU (a synchronous call already means "done" when
it returns) and `torch.cuda.synchronize` for CUDA (execution there is
asynchronous by default; without this, `perf_counter()` would only
measure kernel *launch* time, not kernel *completion* time). Warmup
samples are discarded entirely -- they exist to let lazy runtime
initialization, kernel loading, allocator setup, and (for CUDA
convolution) algorithm selection settle before any timed measurement.
"""

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

BENCHMARK_SCHEMA_VERSION = 1

# Minimum sample counts before a percentile is considered meaningful
# enough to report. Below these thresholds the field is left as None --
# never fabricated from too few samples.
_P95_MIN_SAMPLES = 20
_P99_MIN_SAMPLES = 100


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str
    device: str
    warmup_iterations: int = 10
    measured_iterations: int = 50

    def __post_init__(self) -> None:
        if self.backend not in ("pytorch", "onnxruntime"):
            raise ValueError(f"backend must be 'pytorch' or 'onnxruntime', got {self.backend!r}")
        if not self.device:
            raise ValueError("device must not be empty")
        if not isinstance(self.warmup_iterations, int) or self.warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be an int >= 0, got {self.warmup_iterations!r}")
        if not isinstance(self.measured_iterations, int) or self.measured_iterations <= 0:
            raise ValueError(f"measured_iterations must be an int > 0, got {self.measured_iterations!r}")


def _percentile(sorted_values: list, pct: float) -> float:
    """Linear interpolation between closest ranks (the common definition
    also used by e.g. NumPy's default `percentile`). `sorted_values` must
    already be sorted ascending. Deterministic and hand-verifiable:
    for [1,2,3,4,5], p50 -> 3.0 exactly, p95 -> 4.8 exactly.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * (pct / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


@dataclass(frozen=True)
class LatencyStatistics:
    count: int
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float | None
    p99_seconds: float | None
    min_seconds: float
    max_seconds: float
    throughput_per_second: float


def compute_latency_statistics(samples) -> LatencyStatistics:
    """Pure statistics over already-measured latency samples (seconds).
    p95 requires >= 20 samples, p99 requires >= 100 samples -- below that,
    the field is None rather than a statistically meaningless estimate.
    """
    if len(samples) == 0:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    count = len(ordered)
    mean = statistics.fmean(ordered)
    return LatencyStatistics(
        count=count,
        mean_seconds=mean,
        p50_seconds=_percentile(ordered, 50),
        p95_seconds=_percentile(ordered, 95) if count >= _P95_MIN_SAMPLES else None,
        p99_seconds=_percentile(ordered, 99) if count >= _P99_MIN_SAMPLES else None,
        min_seconds=ordered[0],
        max_seconds=ordered[-1],
        throughput_per_second=1.0 / mean,
    )


def run_timed_iterations(
    fn: Callable[[], None],
    warmup_iterations: int,
    measured_iterations: int,
    sync_fn: Callable[[], None] | None = None,
) -> tuple[float, ...]:
    """Warmup + measured timing loop, backend-agnostic. `fn()` executes one
    steady-state iteration; inputs/weights must already be allocated on
    the target device before this is called (allocation itself is never
    timed). Returns exactly `measured_iterations` latency samples in
    seconds; warmup samples never enter the returned tuple.
    """
    for _ in range(warmup_iterations):
        fn()
        if sync_fn is not None:
            sync_fn()

    samples = []
    for _ in range(measured_iterations):
        if sync_fn is not None:
            sync_fn()  # drain any leftover async work before starting the clock
        start = time.perf_counter()
        fn()
        if sync_fn is not None:
            sync_fn()  # force completion before stopping the clock
        end = time.perf_counter()
        samples.append(end - start)
    return tuple(samples)


@dataclass(frozen=True)
class BenchmarkResult:
    core_result_fingerprint: str
    workload_preset: str
    workload_kind: str
    backend: str
    device: str
    dtype: str
    warmup_iterations: int
    measured_iterations: int
    latency_samples_seconds: tuple[float, ...]
    statistics: LatencyStatistics
    peak_memory_allocated_bytes: int | None
    runtime_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        stats = self.statistics
        return {
            "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
            "core_result_fingerprint": self.core_result_fingerprint,
            "workload": {"preset": self.workload_preset, "kind": self.workload_kind},
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations,
            "latency_samples_seconds": list(self.latency_samples_seconds),
            "measured_latency_statistics_seconds": {
                "count": stats.count,
                "mean": stats.mean_seconds,
                "p50": stats.p50_seconds,
                "p95": stats.p95_seconds,
                "p99": stats.p99_seconds,
                "min": stats.min_seconds,
                "max": stats.max_seconds,
            },
            "measured_throughput_per_second": stats.throughput_per_second,
            "measured_peak_memory_allocated_bytes": self.peak_memory_allocated_bytes,
            "runtime_metadata": dict(self.runtime_metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def benchmark_result_from_dict(d: dict) -> BenchmarkResult:
    """Reconstruct a BenchmarkResult from BenchmarkResult.to_dict()'s output.

    Requires an exact benchmark_schema_version match -- a version mismatch
    means measurement semantics may have changed, so results are not
    directly comparable and this refuses to silently convert.
    """
    schema_version = d.get("benchmark_schema_version")
    if schema_version != BENCHMARK_SCHEMA_VERSION:
        raise ValueError(
            f"benchmark schema version mismatch: expected {BENCHMARK_SCHEMA_VERSION}, "
            f"got {schema_version!r} -- benchmark results are not directly comparable "
            "across schema versions"
        )

    stats_dict = d["measured_latency_statistics_seconds"]
    statistics = LatencyStatistics(
        count=stats_dict["count"],
        mean_seconds=stats_dict["mean"],
        p50_seconds=stats_dict["p50"],
        p95_seconds=stats_dict["p95"],
        p99_seconds=stats_dict["p99"],
        min_seconds=stats_dict["min"],
        max_seconds=stats_dict["max"],
        throughput_per_second=d["measured_throughput_per_second"],
    )
    workload = d["workload"]
    return BenchmarkResult(
        core_result_fingerprint=d["core_result_fingerprint"],
        workload_preset=workload["preset"],
        workload_kind=workload["kind"],
        backend=d["backend"],
        device=d["device"],
        dtype=d["dtype"],
        warmup_iterations=d["warmup_iterations"],
        measured_iterations=d["measured_iterations"],
        latency_samples_seconds=tuple(d["latency_samples_seconds"]),
        statistics=statistics,
        peak_memory_allocated_bytes=d["measured_peak_memory_allocated_bytes"],
        runtime_metadata=dict(d["runtime_metadata"]),
    )


def load_benchmark_result(path: str) -> BenchmarkResult:
    import json

    with open(path, encoding="utf-8") as f:
        return benchmark_result_from_dict(json.load(f))
