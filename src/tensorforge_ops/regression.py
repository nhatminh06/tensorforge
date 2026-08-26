"""PR performance-regression guard: compares two already-measured
BenchmarkResults (baseline vs. candidate) against an explicit policy.

Torch-free and MLflow-free -- pure comparison over already-produced
BenchmarkResult JSON files, so this module works offline in CI with no
tracking server and no GPU/torch install beyond what already produced
the two benchmark JSON files being compared.

The gate answers one narrow question: "did measured performance get
meaningfully worse from baseline to candidate, on this benchmark
environment, under this explicit policy?" It does NOT answer "is this
implementation fast" (a candidate can PASS regression while still being
objectively slow), and it does NOT decide the gate using TensorForge's
predicted/calibrated analytical latency -- Milestone 13 showed Core
prediction error can be large (small GEMMs, materialized-im2col Conv2D,
multi-launch Transformer workloads), so an analytical prediction is
diagnostic context here, never the gating signal.

This module never reruns a benchmark -- both BenchmarkResults must
already be computed (see benchmark_pytorch.py / the `benchmark` CLI
command). Comparison is pure and deterministic for fixed inputs.
"""

import math
from dataclasses import dataclass, field

from tensorforge_ops.benchmark import BENCHMARK_SCHEMA_VERSION, BenchmarkResult

REGRESSION_POLICY_SCHEMA_VERSION = 1
REGRESSION_RESULT_SCHEMA_VERSION = 1

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_ERROR = "ERROR"
STATUS_NOT_COMPARABLE = "NOT_COMPARABLE"

# Fixed metric vocabulary. "higher is worse" for latency/memory,
# "lower is worse" for throughput -- this asymmetry is deliberate (see
# module docstring section on metric direction) and must never be
# implemented with one shared sign rule.
_LATENCY_METRICS = ("p50_latency_seconds", "p95_latency_seconds", "mean_latency_seconds", "p99_latency_seconds")
_MEMORY_METRICS = ("peak_memory_allocated_bytes",)
_THROUGHPUT_METRICS = ("throughput_per_second",)
KNOWN_METRICS = _LATENCY_METRICS + _MEMORY_METRICS + _THROUGHPUT_METRICS

_HIGHER_IS_WORSE = set(_LATENCY_METRICS) | set(_MEMORY_METRICS)
_LOWER_IS_WORSE = set(_THROUGHPUT_METRICS)


def _extract_metric_value(benchmark: BenchmarkResult, metric: str) -> float | None:
    stats = benchmark.statistics
    if metric == "p50_latency_seconds":
        return stats.p50_seconds
    if metric == "p95_latency_seconds":
        return stats.p95_seconds
    if metric == "p99_latency_seconds":
        return stats.p99_seconds
    if metric == "mean_latency_seconds":
        return stats.mean_seconds
    if metric == "throughput_per_second":
        return stats.throughput_per_second
    if metric == "peak_memory_allocated_bytes":
        value = benchmark.peak_memory_allocated_bytes
        return float(value) if value is not None else None
    raise ValueError(f"unknown metric name {metric!r} (known: {', '.join(KNOWN_METRICS)})")


def _is_finite_nonnegative(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class MetricPolicy:
    """Threshold configuration for one metric. At least one allowance must
    be set -- there is no hidden/implicit default threshold anywhere in
    this module.

    max_relative_regression: fraction of |baseline| (e.g. 0.10 = 10%).
    max_absolute_regression_seconds: used for latency metrics.
    max_absolute_regression_bytes: used for the memory metric.
    (Neither absolute field applies to throughput; see module docs.)
    """

    max_relative_regression: float | None = None
    max_absolute_regression_seconds: float | None = None
    max_absolute_regression_bytes: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_relative_regression", self.max_relative_regression),
            ("max_absolute_regression_seconds", self.max_absolute_regression_seconds),
            ("max_absolute_regression_bytes", self.max_absolute_regression_bytes),
        ):
            if value is not None and not _is_finite_nonnegative(value):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        if (
            self.max_relative_regression is None
            and self.max_absolute_regression_seconds is None
            and self.max_absolute_regression_bytes is None
        ):
            raise ValueError(
                "MetricPolicy must set at least one of max_relative_regression / "
                "max_absolute_regression_seconds / max_absolute_regression_bytes"
            )

    def to_dict(self) -> dict:
        d = {}
        if self.max_relative_regression is not None:
            d["max_relative_regression"] = self.max_relative_regression
        if self.max_absolute_regression_seconds is not None:
            d["max_absolute_regression_seconds"] = self.max_absolute_regression_seconds
        if self.max_absolute_regression_bytes is not None:
            d["max_absolute_regression_bytes"] = self.max_absolute_regression_bytes
        return d


def metric_policy_from_dict(d: dict) -> MetricPolicy:
    return MetricPolicy(
        max_relative_regression=d.get("max_relative_regression"),
        max_absolute_regression_seconds=d.get("max_absolute_regression_seconds"),
        max_absolute_regression_bytes=d.get("max_absolute_regression_bytes"),
    )


def _absolute_allowance_for_metric(metric: str, policy: MetricPolicy) -> float | None:
    if metric in _LATENCY_METRICS:
        return policy.max_absolute_regression_seconds
    if metric in _MEMORY_METRICS:
        return policy.max_absolute_regression_bytes
    return None  # throughput has no absolute-allowance concept in this policy


@dataclass(frozen=True)
class RegressionPolicy:
    metrics: dict  # metric name -> MetricPolicy
    regression_policy_schema_version: int = REGRESSION_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.regression_policy_schema_version != REGRESSION_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"regression_policy_schema_version must be {REGRESSION_POLICY_SCHEMA_VERSION}, "
                f"got {self.regression_policy_schema_version!r}"
            )
        if not self.metrics:
            raise ValueError("policy must define at least one metric")
        for name, policy in self.metrics.items():
            if name not in KNOWN_METRICS:
                raise ValueError(f"unknown metric name {name!r} in policy (known: {', '.join(KNOWN_METRICS)})")
            if not isinstance(policy, MetricPolicy):
                raise ValueError(f"policy for metric {name!r} must be a MetricPolicy, got {type(policy)!r}")

    def to_dict(self) -> dict:
        return {
            "regression_policy_schema_version": self.regression_policy_schema_version,
            "metrics": {name: policy.to_dict() for name, policy in self.metrics.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def regression_policy_from_dict(d: dict) -> RegressionPolicy:
    if "metrics" not in d:
        raise ValueError("policy JSON must have a 'metrics' object")
    metrics = {name: metric_policy_from_dict(spec) for name, spec in d["metrics"].items()}
    return RegressionPolicy(
        metrics=metrics,
        regression_policy_schema_version=d.get("regression_policy_schema_version", REGRESSION_POLICY_SCHEMA_VERSION),
    )


def load_regression_policy(path: str) -> RegressionPolicy:
    import json

    with open(path, encoding="utf-8") as f:
        return regression_policy_from_dict(json.load(f))


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    baseline_value: float | None
    candidate_value: float | None
    absolute_delta: float | None       # candidate - baseline, raw (metric's own units)
    relative_delta: float | None       # (candidate - baseline) / baseline, raw
    allowed_regression: float | None   # in the metric's own units; None if not policy-gated
    status: str                        # PASS / FAIL / NOT_COMPARABLE / ERROR
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "allowed_regression": self.allowed_regression,
            "status": self.status,
            "detail": self.detail,
        }


def _compare_metric(metric: str, baseline: BenchmarkResult, candidate: BenchmarkResult, policy: MetricPolicy | None) -> MetricComparison:
    baseline_value = _extract_metric_value(baseline, metric)
    candidate_value = _extract_metric_value(candidate, metric)

    if baseline_value is None and candidate_value is None:
        return MetricComparison(
            metric=metric, baseline_value=None, candidate_value=None,
            absolute_delta=None, relative_delta=None, allowed_regression=None,
            status=STATUS_NOT_COMPARABLE,
            detail="metric not available on this device/backend for either run",
        )

    if baseline_value is None or candidate_value is None:
        # Coverage changed between baseline and candidate -- never silently
        # treated as 0; this is a comparison-infrastructure problem.
        return MetricComparison(
            metric=metric, baseline_value=baseline_value, candidate_value=candidate_value,
            absolute_delta=None, relative_delta=None, allowed_regression=None,
            status=STATUS_ERROR,
            detail="metric present on one side but not the other (measurement coverage changed)",
        )

    absolute_delta = candidate_value - baseline_value
    relative_delta = (absolute_delta / baseline_value) if baseline_value != 0 else None

    if policy is None:
        return MetricComparison(
            metric=metric, baseline_value=baseline_value, candidate_value=candidate_value,
            absolute_delta=absolute_delta, relative_delta=relative_delta, allowed_regression=None,
            status=STATUS_NOT_COMPARABLE, detail="no policy configured for this metric",
        )

    # regression_amount > 0 always means "worse", regardless of metric direction.
    if metric in _HIGHER_IS_WORSE:
        regression_amount = candidate_value - baseline_value
    else:
        regression_amount = baseline_value - candidate_value

    allowances = []
    if policy.max_relative_regression is not None:
        allowances.append(abs(baseline_value) * policy.max_relative_regression)
    absolute_allowance = _absolute_allowance_for_metric(metric, policy)
    if absolute_allowance is not None:
        allowances.append(absolute_allowance)
    allowed_regression = max(allowances)

    status = STATUS_FAIL if regression_amount > allowed_regression else STATUS_PASS

    return MetricComparison(
        metric=metric, baseline_value=baseline_value, candidate_value=candidate_value,
        absolute_delta=absolute_delta, relative_delta=relative_delta,
        allowed_regression=allowed_regression, status=status,
    )


def _identity_mismatch_reason(baseline: BenchmarkResult, candidate: BenchmarkResult) -> str | None:
    """Everything that must match for the two results to describe the
    same mathematical workload measured the same way. Iteration counts
    are deliberately NOT required to match -- a different --iterations
    value does not change workload identity, and both counts are still
    recorded in the RegressionResult for transparency.
    """
    if baseline.workload_preset != candidate.workload_preset:
        return f"workload preset mismatch: baseline={baseline.workload_preset!r} candidate={candidate.workload_preset!r}"
    if baseline.workload_kind != candidate.workload_kind:
        return f"workload kind mismatch: baseline={baseline.workload_kind!r} candidate={candidate.workload_kind!r}"
    if baseline.backend != candidate.backend:
        return f"backend mismatch: baseline={baseline.backend!r} candidate={candidate.backend!r}"
    if baseline.dtype != candidate.dtype:
        return f"dtype mismatch: baseline={baseline.dtype!r} candidate={candidate.dtype!r}"

    baseline_device_type = baseline.device.split(":")[0]
    candidate_device_type = candidate.device.split(":")[0]
    if baseline_device_type != candidate_device_type:
        return f"device type mismatch: baseline={baseline_device_type!r} candidate={candidate_device_type!r}"

    if baseline_device_type == "cuda":
        for key in ("device_index", "device_name"):
            bv = baseline.runtime_metadata.get(key)
            cv = candidate.runtime_metadata.get(key)
            if bv != cv:
                return f"CUDA device mismatch on {key}: baseline={bv!r} candidate={cv!r}"

    return None


@dataclass(frozen=True)
class RegressionResult:
    status: str  # PASS / FAIL / ERROR
    error_message: str | None

    workload_preset: str | None
    workload_kind: str | None
    backend: str | None
    device: str | None
    dtype: str | None

    baseline_core_fingerprint: str | None
    candidate_core_fingerprint: str | None
    baseline_warmup_iterations: int | None
    baseline_measured_iterations: int | None
    candidate_warmup_iterations: int | None
    candidate_measured_iterations: int | None
    baseline_runtime_metadata: dict
    candidate_runtime_metadata: dict

    metric_comparisons: tuple

    regression_result_schema_version: int = REGRESSION_RESULT_SCHEMA_VERSION

    @property
    def failed_metrics(self) -> tuple:
        return tuple(c.metric for c in self.metric_comparisons if c.status == STATUS_FAIL)

    def to_dict(self) -> dict:
        return {
            "regression_result_schema_version": self.regression_result_schema_version,
            "status": self.status,
            "error_message": self.error_message,
            "workload": {"preset": self.workload_preset, "kind": self.workload_kind},
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "core_fingerprints": {
                "baseline": self.baseline_core_fingerprint,
                "candidate": self.candidate_core_fingerprint,
            },
            "iterations": {
                "baseline": {"warmup": self.baseline_warmup_iterations, "measured": self.baseline_measured_iterations},
                "candidate": {"warmup": self.candidate_warmup_iterations, "measured": self.candidate_measured_iterations},
            },
            "runtime_metadata": {
                "baseline": dict(self.baseline_runtime_metadata),
                "candidate": dict(self.candidate_runtime_metadata),
            },
            "metrics": [c.to_dict() for c in self.metric_comparisons],
            "failed_metrics": list(self.failed_metrics),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def _error_result(message: str, baseline: BenchmarkResult | None = None, candidate: BenchmarkResult | None = None) -> RegressionResult:
    return RegressionResult(
        status=STATUS_ERROR, error_message=message,
        workload_preset=baseline.workload_preset if baseline else None,
        workload_kind=baseline.workload_kind if baseline else None,
        backend=baseline.backend if baseline else None,
        device=baseline.device if baseline else None,
        dtype=baseline.dtype if baseline else None,
        baseline_core_fingerprint=baseline.core_result_fingerprint if baseline else None,
        candidate_core_fingerprint=candidate.core_result_fingerprint if candidate else None,
        baseline_warmup_iterations=baseline.warmup_iterations if baseline else None,
        baseline_measured_iterations=baseline.measured_iterations if baseline else None,
        candidate_warmup_iterations=candidate.warmup_iterations if candidate else None,
        candidate_measured_iterations=candidate.measured_iterations if candidate else None,
        baseline_runtime_metadata=dict(baseline.runtime_metadata) if baseline else {},
        candidate_runtime_metadata=dict(candidate.runtime_metadata) if candidate else {},
        metric_comparisons=(),
    )


def compare_benchmark_results(baseline: BenchmarkResult, candidate: BenchmarkResult, policy: RegressionPolicy) -> RegressionResult:
    """Pure, deterministic comparison of two already-measured
    BenchmarkResults under an explicit policy. Never reruns a benchmark.

    Returns a RegressionResult with status ERROR (not a raised exception)
    when the two results cannot be validly compared (workload/backend/
    device/dtype mismatch) -- this lets the caller still produce a
    diagnostic report explaining why, rather than losing all context to
    an exception. A malformed `policy` (caught earlier at construction)
    is a genuine configuration error and is not this function's concern.
    """
    mismatch = _identity_mismatch_reason(baseline, candidate)
    if mismatch is not None:
        return _error_result(
            f"baseline and candidate do not describe the same benchmark, cannot compare: {mismatch}",
            baseline, candidate,
        )

    comparisons = tuple(
        _compare_metric(metric, baseline, candidate, policy.metrics.get(metric))
        for metric in KNOWN_METRICS
        if metric in policy.metrics
    )

    if any(c.status == STATUS_ERROR for c in comparisons):
        bad = next(c for c in comparisons if c.status == STATUS_ERROR)
        return _error_result(
            f"metric {bad.metric!r} could not be compared: {bad.detail}", baseline, candidate,
        )

    status = STATUS_FAIL if any(c.status == STATUS_FAIL for c in comparisons) else STATUS_PASS

    return RegressionResult(
        status=status, error_message=None,
        workload_preset=baseline.workload_preset, workload_kind=baseline.workload_kind,
        backend=baseline.backend, device=baseline.device, dtype=baseline.dtype,
        baseline_core_fingerprint=baseline.core_result_fingerprint,
        candidate_core_fingerprint=candidate.core_result_fingerprint,
        baseline_warmup_iterations=baseline.warmup_iterations,
        baseline_measured_iterations=baseline.measured_iterations,
        candidate_warmup_iterations=candidate.warmup_iterations,
        candidate_measured_iterations=candidate.measured_iterations,
        baseline_runtime_metadata=dict(baseline.runtime_metadata),
        candidate_runtime_metadata=dict(candidate.runtime_metadata),
        metric_comparisons=comparisons,
    )


# --- suite ------------------------------------------------------------------

@dataclass(frozen=True)
class RegressionSuiteResult:
    status: str  # PASS / FAIL / ERROR (ERROR > FAIL > PASS precedence)
    results: tuple
    failed_workloads: tuple
    errored_workloads: tuple
    regression_result_schema_version: int = REGRESSION_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "regression_result_schema_version": self.regression_result_schema_version,
            "status": self.status,
            "results": [r.to_dict() for r in self.results],
            "failed_workloads": list(self.failed_workloads),
            "errored_workloads": list(self.errored_workloads),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


# --- Markdown report ---------------------------------------------------------

_METRIC_LABELS = {
    "p50_latency_seconds": "p50 latency",
    "p95_latency_seconds": "p95 latency",
    "p99_latency_seconds": "p99 latency",
    "mean_latency_seconds": "mean latency",
    "throughput_per_second": "throughput",
    "peak_memory_allocated_bytes": "peak memory",
}

_STATUS_MARKS = {STATUS_PASS: "PASS", STATUS_FAIL: "FAIL", STATUS_NOT_COMPARABLE: "n/a", STATUS_ERROR: "ERROR"}


def _format_metric_value(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if metric in _LATENCY_METRICS:
        if value < 1e-3:
            return f"{value * 1e6:.2f} us"
        if value < 1:
            return f"{value * 1e3:.2f} ms"
        return f"{value:.2f} s"
    if metric in _MEMORY_METRICS:
        return f"{value / (1024 * 1024):.1f} MiB"
    if metric in _THROUGHPUT_METRICS:
        return f"{value:.1f}/s"
    return str(value)


def _format_relative(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%"


def _format_allowed(metric: str, comparison: MetricComparison) -> str:
    if comparison.allowed_regression is None:
        return "n/a"
    if metric in _LATENCY_METRICS:
        return f"+{_format_metric_value(metric, comparison.allowed_regression)}"
    if metric in _MEMORY_METRICS:
        return f"+{_format_metric_value(metric, comparison.allowed_regression)}"
    if metric in _THROUGHPUT_METRICS:
        # allowed_regression here is an absolute drop in the metric's own
        # units; express it as the equivalent relative-drop percentage.
        if comparison.baseline_value:
            return f"-{comparison.allowed_regression / comparison.baseline_value * 100:.1f}%"
        return f"-{comparison.allowed_regression:.2f}/s"
    return str(comparison.allowed_regression)


def render_markdown_report(result: RegressionResult) -> str:
    """Deterministic Markdown report for one RegressionResult. Plain
    PASS/FAIL/ERROR text, no emoji, matching this project's style.
    """
    lines = ["## TensorForge Performance Regression Guard", ""]

    if result.status == STATUS_ERROR:
        lines.append("**Result: ERROR**")
        lines.append("")
        lines.append(f"{result.error_message}")
        lines.append("")
        lines.append(
            "No performance verdict is available -- baseline and candidate "
            "could not be validly compared."
        )
        return "\n".join(lines) + "\n"

    lines.append(f"**Result: {result.status}**")
    lines.append("")
    lines.append(f"Workload: `{result.workload_preset}` ({result.workload_kind})")
    lines.append("")
    lines.append("| Metric | Base | Candidate | Change | Allowed | Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for c in result.metric_comparisons:
        label = _METRIC_LABELS.get(c.metric, c.metric)
        base = _format_metric_value(c.metric, c.baseline_value)
        candidate = _format_metric_value(c.metric, c.candidate_value)
        change = _format_relative(c.relative_delta)
        allowed = _format_allowed(c.metric, c)
        status = _STATUS_MARKS.get(c.status, c.status)
        lines.append(f"| {label} | {base} | {candidate} | {change} | {allowed} | {status} |")
    lines.append("")

    lines.append("### Environment")
    lines.append("")
    lines.append(f"Backend: {result.backend}")
    lines.append(f"Device: {result.device}")
    lines.append(f"Dtype: {result.dtype}")
    lines.append("")

    baseline_torch = result.baseline_runtime_metadata.get("torch_version")
    candidate_torch = result.candidate_runtime_metadata.get("torch_version")
    if baseline_torch != candidate_torch:
        lines.append("### Runtime change")
        lines.append("")
        lines.append(f"PyTorch: {baseline_torch} -> {candidate_torch}")
        lines.append("")
        lines.append(
            "This PR changed the PyTorch runtime used to measure baseline vs. "
            "candidate; a resulting performance change may be caused by that "
            "runtime change rather than by code in this PR."
        )
        lines.append("")

    lines.append("### Analytical context")
    lines.append("")
    if result.baseline_core_fingerprint == result.candidate_core_fingerprint:
        lines.append("Core analytical fingerprint unchanged between baseline and candidate.")
    else:
        lines.append(
            f"Core analytical fingerprint changed: `{result.baseline_core_fingerprint}` -> "
            f"`{result.candidate_core_fingerprint}`. This means the PR changed the modeled "
            "workload/accelerator configuration, Core code, or both -- it may help explain "
            "a measured change above, but this section is diagnostic context only and never "
            "decides the PASS/FAIL result."
        )
    lines.append("")

    if result.failed_metrics:
        lines.append("### Failed metrics")
        lines.append("")
        for metric in result.failed_metrics:
            lines.append(f"- {_METRIC_LABELS.get(metric, metric)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def compare_regression_suite(pairs, policy: RegressionPolicy) -> RegressionSuiteResult:
    """pairs: iterable of (baseline: BenchmarkResult, candidate: BenchmarkResult).
    Precedence for overall status: ERROR > FAIL > PASS -- any infrastructure
    error anywhere in the suite takes priority over reporting it as a plain
    pass/fail.
    """
    results = tuple(compare_benchmark_results(b, c, policy) for b, c in pairs)
    if not results:
        raise ValueError("pairs must not be empty")

    failed = tuple(r.workload_preset for r in results if r.status == STATUS_FAIL)
    errored = tuple(r.workload_preset for r in results if r.status == STATUS_ERROR)

    if errored:
        status = STATUS_ERROR
    elif failed:
        status = STATUS_FAIL
    else:
        status = STATUS_PASS

    return RegressionSuiteResult(status=status, results=results, failed_workloads=failed, errored_workloads=errored)
