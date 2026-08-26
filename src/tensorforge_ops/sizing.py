"""Hardware right-sizing + inference SLO/cost planner.

Torch-free, MLflow-free, NVML-free: consumes already-serialized
BenchmarkResult/TelemetrySummary/ValidationResult JSON via the existing
loaders in benchmark.py/telemetry.py/calibration.py, and does pure
arithmetic from there. Keeps planning lightweight -- no accelerator
runtime or tracking server is needed once benchmark/telemetry/validation
JSON already exists.

The central question this module answers:

    Given the performance we actually measured, what is the cheapest
    deployment configuration among the supplied candidates that
    satisfies the explicit SLO assumptions?

Measured performance (BenchmarkResult) determines feasibility.
TelemetrySummary and ValidationResult are attached only as diagnostic
context -- they never change a candidate's FEASIBLE/SLO_VIOLATION/
INSUFFICIENT_EVIDENCE status, and they never change the ranking.

Critical honesty rules (see docs/ops/right-sizing.md for the full
rationale):

- `BenchmarkResult.statistics.throughput_per_second` is used as a
  STEADY-STATE BENCHMARK CAPACITY PROXY -- it is repeated steady-state
  invocation throughput from Milestone 12, not guaranteed production
  requests/sec, not load-tested HTTP throughput, and not a
  queueing-theory result. No M/M/1, M/M/c, Little's Law, request
  concurrency, dynamic batching, or autoscaler dynamics are modeled here
  or planned for this milestone.
- Benchmark p95 is the measured workload's own operator/workload
  latency, not full end-to-end service latency (no network, queueing,
  preprocessing, or response serialization time included).
- Replicas are assumed to scale capacity perfectly linearly
  (total_capacity = replicas * per_replica_capacity). Shared databases,
  network bottlenecks, load-balancer overhead, storage bottlenecks,
  inter-replica contention, and cluster scheduling effects are not
  modeled -- this is stated, not hidden.
- No cloud price is ever fetched, hard-coded, or verified over the
  network. Pricing is entirely user-supplied; only its provenance
  (provider/region/source/as-of) is recorded, never validated.
- The "recommended" candidate is the cheapest FEASIBLE candidate among
  the SUPPLIED candidates only -- never a claim of a global/theoretical
  optimum.

Money: hourly/monthly costs are parsed and computed using `Decimal`
(via `Decimal(str(value))`, which avoids the classic
`Decimal(0.1) != 0.1` binary-float artifact) to avoid floating-point
cent-rounding surprises in cost comparisons and budget checks. JSON
output converts Decimal -> float for readability; the computation itself
never rounds arbitrarily.
"""

import math
import os
from dataclasses import dataclass, field, replace
from decimal import Decimal

from tensorforge_ops.benchmark import BenchmarkResult, load_benchmark_result
from tensorforge_ops.calibration import ValidationResult, load_validation_result
from tensorforge_ops.telemetry import TelemetrySummary, load_telemetry_summary

DEPLOYMENT_CATALOG_SCHEMA_VERSION = 1
SLO_POLICY_SCHEMA_VERSION = 1
SIZING_RESULT_SCHEMA_VERSION = 1

STATUS_FEASIBLE = "FEASIBLE"
STATUS_SLO_VIOLATION = "SLO_VIOLATION"
STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

PLAN_STATUS_OK = "OK"
PLAN_STATUS_ERROR = "ERROR"

_ASSUMPTIONS = (
    "measured_capacity_per_replica is BenchmarkResult.statistics.throughput_per_second -- a steady-state "
    "benchmark capacity proxy from repeated invocation, not measured production request throughput; no "
    "queueing, batching, or concurrency model is applied.",
    "measured p95 is the benchmark's own workload/operator latency, not full end-to-end service latency "
    "(no network, queueing, preprocessing, or response-serialization time is included).",
    "replicas are assumed to scale capacity perfectly linearly (total_capacity = replicas * "
    "per_replica_capacity); shared databases, network bottlenecks, load-balancer overhead, storage "
    "bottlenecks, inter-replica contention, and cluster scheduling effects are not modeled.",
    "capacity_headroom_fraction defaults to 0.0 -- no operational safety margin is invented; the caller "
    "must request one explicitly.",
    "pricing is entirely user-supplied; no cloud price is fetched, hard-coded, or verified over the network, "
    "and no currency conversion is performed.",
    "the recommended candidate is the cheapest FEASIBLE candidate among the supplied candidates only -- "
    "never a claim of global/theoretical optimality.",
    "no queueing-theory model (M/M/1, M/M/c, Little's Law) is used -- this is transparent arithmetic over "
    "already-measured steady-state benchmark results.",
    "TelemetrySummary/ValidationResult attached to a candidate are diagnostic context only; they never "
    "change a candidate's feasibility status or the ranking.",
)


def _is_finite_nonnegative_decimal(value: Decimal) -> bool:
    return value.is_finite() and value >= 0


# --- deployment candidate / catalog ------------------------------------------------

@dataclass(frozen=True)
class DeploymentCandidate:
    candidate_id: str
    benchmark_result: BenchmarkResult
    hourly_cost: Decimal
    currency: str

    provider: str | None = None
    region: str | None = None
    hardware_label: str | None = None
    price_source: str | None = None
    price_as_of: str | None = None

    telemetry_summary: TelemetrySummary | None = None
    validation_result: ValidationResult | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if not self.currency:
            raise ValueError(f"candidate {self.candidate_id!r}: currency must not be empty")
        if not isinstance(self.hourly_cost, Decimal) or not _is_finite_nonnegative_decimal(self.hourly_cost):
            raise ValueError(f"candidate {self.candidate_id!r}: hourly_cost must be a finite Decimal >= 0, got {self.hourly_cost!r}")


@dataclass(frozen=True)
class DeploymentCatalog:
    deployment_catalog_schema_version: int
    currency: str
    candidates: tuple

    def __post_init__(self) -> None:
        if self.deployment_catalog_schema_version != DEPLOYMENT_CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"deployment_catalog_schema_version must be {DEPLOYMENT_CATALOG_SCHEMA_VERSION}, "
                f"got {self.deployment_catalog_schema_version!r}"
            )
        if not self.currency:
            raise ValueError("catalog currency must not be empty")
        if not self.candidates:
            raise ValueError("catalog must contain at least one candidate")
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate candidate_id(s) in catalog: {duplicates}")
        mismatched = [c.candidate_id for c in self.candidates if c.currency != self.currency]
        if mismatched:
            raise ValueError(
                f"mixed currency in catalog: catalog currency is {self.currency!r}, but candidate(s) "
                f"{mismatched} use a different currency -- no currency conversion is performed"
            )


def _resolve_path(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def deployment_candidate_from_dict(d: dict, base_dir: str, catalog_currency: str) -> DeploymentCandidate:
    benchmark_result = load_benchmark_result(_resolve_path(d["benchmark_result"], base_dir))

    telemetry_summary = None
    if d.get("telemetry_summary") is not None:
        telemetry_summary = load_telemetry_summary(_resolve_path(d["telemetry_summary"], base_dir))

    validation_result = None
    if d.get("validation_result") is not None:
        validation_result = load_validation_result(_resolve_path(d["validation_result"], base_dir))

    return DeploymentCandidate(
        candidate_id=d["id"],
        benchmark_result=benchmark_result,
        hourly_cost=Decimal(str(d["hourly_cost"])),
        currency=d.get("currency", catalog_currency),
        provider=d.get("provider"),
        region=d.get("region"),
        hardware_label=d.get("hardware_label"),
        price_source=d.get("price_source"),
        price_as_of=d.get("price_as_of"),
        telemetry_summary=telemetry_summary,
        validation_result=validation_result,
    )


def deployment_catalog_from_dict(d: dict, base_dir: str) -> DeploymentCatalog:
    schema_version = d.get("deployment_catalog_schema_version")
    if schema_version != DEPLOYMENT_CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"deployment_catalog_schema_version must be {DEPLOYMENT_CATALOG_SCHEMA_VERSION}, got {schema_version!r}"
        )
    currency = d["currency"]
    candidates = tuple(
        deployment_candidate_from_dict(c, base_dir, currency) for c in d["candidates"]
    )
    return DeploymentCatalog(deployment_catalog_schema_version=schema_version, currency=currency, candidates=candidates)


def load_deployment_catalog(path: str) -> DeploymentCatalog:
    import json

    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(path))
    return deployment_catalog_from_dict(d, base_dir)


# --- SLO policy ---------------------------------------------------------------

@dataclass(frozen=True)
class SloPolicy:
    required_invocations_per_second: float
    max_p95_latency_seconds: float
    capacity_headroom_fraction: float = 0.0
    monthly_hours: float | None = None
    max_monthly_cost: Decimal | None = None
    max_replicas: int | None = None
    max_peak_memory_allocated_bytes: int | None = None
    slo_policy_schema_version: int = SLO_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.slo_policy_schema_version != SLO_POLICY_SCHEMA_VERSION:
            raise ValueError(f"slo_policy_schema_version must be {SLO_POLICY_SCHEMA_VERSION}, got {self.slo_policy_schema_version!r}")
        if not math.isfinite(self.required_invocations_per_second) or self.required_invocations_per_second <= 0:
            raise ValueError(f"required_invocations_per_second must be a finite number > 0, got {self.required_invocations_per_second!r}")
        if not math.isfinite(self.max_p95_latency_seconds) or self.max_p95_latency_seconds <= 0:
            raise ValueError(f"max_p95_latency_seconds must be a finite number > 0, got {self.max_p95_latency_seconds!r}")
        if not math.isfinite(self.capacity_headroom_fraction) or not (0 <= self.capacity_headroom_fraction < 1):
            raise ValueError(f"capacity_headroom_fraction must satisfy 0 <= x < 1, got {self.capacity_headroom_fraction!r}")
        if self.monthly_hours is not None and (not math.isfinite(self.monthly_hours) or self.monthly_hours <= 0):
            raise ValueError(f"monthly_hours must be a finite number > 0, got {self.monthly_hours!r}")
        if self.max_monthly_cost is not None:
            if self.monthly_hours is None:
                raise ValueError("max_monthly_cost requires monthly_hours to also be supplied")
            if not isinstance(self.max_monthly_cost, Decimal) or not _is_finite_nonnegative_decimal(self.max_monthly_cost):
                raise ValueError(f"max_monthly_cost must be a finite Decimal >= 0, got {self.max_monthly_cost!r}")
        if self.max_replicas is not None and (not isinstance(self.max_replicas, int) or isinstance(self.max_replicas, bool) or self.max_replicas < 1):
            raise ValueError(f"max_replicas must be an int >= 1, got {self.max_replicas!r}")
        if self.max_peak_memory_allocated_bytes is not None and (
            not isinstance(self.max_peak_memory_allocated_bytes, int) or self.max_peak_memory_allocated_bytes <= 0
        ):
            raise ValueError(f"max_peak_memory_allocated_bytes must be an int > 0, got {self.max_peak_memory_allocated_bytes!r}")


def slo_policy_from_dict(d: dict) -> SloPolicy:
    max_monthly_cost = d.get("max_monthly_cost")
    return SloPolicy(
        required_invocations_per_second=d["required_invocations_per_second"],
        max_p95_latency_seconds=d["max_p95_latency_seconds"],
        capacity_headroom_fraction=d.get("capacity_headroom_fraction", 0.0),
        monthly_hours=d.get("monthly_hours"),
        max_monthly_cost=Decimal(str(max_monthly_cost)) if max_monthly_cost is not None else None,
        max_replicas=d.get("max_replicas"),
        max_peak_memory_allocated_bytes=d.get("max_peak_memory_allocated_bytes"),
        slo_policy_schema_version=d.get("slo_policy_schema_version", SLO_POLICY_SCHEMA_VERSION),
    )


def load_slo_policy(path: str) -> SloPolicy:
    import json

    with open(path, encoding="utf-8") as f:
        return slo_policy_from_dict(json.load(f))


@dataclass(frozen=True)
class DemandScenario:
    name: str
    required_invocations_per_second: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must not be empty")
        if not math.isfinite(self.required_invocations_per_second) or self.required_invocations_per_second <= 0:
            raise ValueError(f"required_invocations_per_second must be a finite number > 0, got {self.required_invocations_per_second!r}")


# --- candidate plan -------------------------------------------------------------

def _telemetry_context(summary: TelemetrySummary | None) -> dict | None:
    if summary is None:
        return None
    return {
        "gpu_utilization_percent_mean": summary.gpu_utilization_percent.mean,
        "gpu_utilization_percent_max": summary.gpu_utilization_percent.max,
        "memory_activity_percent_mean": summary.memory_activity_percent.mean,
        "memory_used_bytes_max": summary.memory_used_bytes.max,
        "power_watts_mean": summary.power_watts.mean,
        "power_watts_max": summary.power_watts.max,
        "temperature_celsius_max": summary.temperature_celsius.max,
        "throttle_reasons_observed": list(summary.throttle_reasons_observed),
    }


def _validation_context(result: ValidationResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "absolute_percentage_error": result.absolute_percentage_error,
        "measured_to_predicted_ratio": result.measured_to_predicted_ratio,
        "predicted_bottleneck": result.predicted_bottleneck,
    }


@dataclass(frozen=True)
class CandidatePlan:
    candidate_id: str
    status: str
    reasons: tuple

    device: str
    backend: str
    dtype: str
    runtime_metadata: dict

    measured_p95_latency_seconds: float | None
    measured_capacity_per_replica: float

    capacity_headroom_fraction: float
    usable_capacity_per_replica: float

    required_replicas: int | None
    total_measured_capacity: float | None
    total_usable_capacity: float | None
    spare_usable_capacity: float | None
    planned_utilization_fraction: float | None

    hourly_cost_per_replica: Decimal
    total_hourly_cost: Decimal | None
    monthly_cost: Decimal | None
    cost_per_million_required_invocations: Decimal | None

    peak_memory_allocated_bytes: int | None

    provider: str | None
    region: str | None
    hardware_label: str | None
    price_source: str | None
    price_as_of: str | None

    telemetry_context: dict | None
    validation_context: dict | None

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "device": self.device,
            "backend": self.backend,
            "dtype": self.dtype,
            "runtime_metadata": dict(self.runtime_metadata),
            "measured_p95_latency_seconds": self.measured_p95_latency_seconds,
            "measured_capacity_per_replica": self.measured_capacity_per_replica,
            "capacity_headroom_fraction": self.capacity_headroom_fraction,
            "usable_capacity_per_replica": self.usable_capacity_per_replica,
            "required_replicas": self.required_replicas,
            "total_measured_capacity": self.total_measured_capacity,
            "total_usable_capacity": self.total_usable_capacity,
            "spare_usable_capacity": self.spare_usable_capacity,
            "planned_utilization_fraction": self.planned_utilization_fraction,
            "hourly_cost_per_replica": float(self.hourly_cost_per_replica),
            "total_hourly_cost": float(self.total_hourly_cost) if self.total_hourly_cost is not None else None,
            "monthly_cost": float(self.monthly_cost) if self.monthly_cost is not None else None,
            "cost_per_million_required_invocations": (
                float(self.cost_per_million_required_invocations)
                if self.cost_per_million_required_invocations is not None else None
            ),
            "peak_memory_allocated_bytes": self.peak_memory_allocated_bytes,
            "provider": self.provider,
            "region": self.region,
            "hardware_label": self.hardware_label,
            "price_source": self.price_source,
            "price_as_of": self.price_as_of,
            "telemetry_context": self.telemetry_context,
            "validation_context": self.validation_context,
        }


def _evaluate_candidate(candidate: DeploymentCandidate, slo: SloPolicy) -> CandidatePlan:
    benchmark = candidate.benchmark_result
    stats = benchmark.statistics
    reasons = []

    p95 = stats.p95_seconds
    if p95 is None:
        reasons.append("missing_p95")
        latency_ok = False
    elif p95 > slo.max_p95_latency_seconds:
        reasons.append("latency_slo_exceeded")
        latency_ok = False
    else:
        latency_ok = True

    measured_capacity_per_replica = stats.throughput_per_second
    usable_capacity_per_replica = measured_capacity_per_replica * (1 - slo.capacity_headroom_fraction)

    required_replicas = total_measured_capacity = total_usable_capacity = None
    spare_usable_capacity = planned_utilization_fraction = None
    total_hourly_cost = monthly_cost = cost_per_million = None

    if latency_ok:
        required_replicas = max(1, math.ceil(slo.required_invocations_per_second / usable_capacity_per_replica))
        total_measured_capacity = required_replicas * measured_capacity_per_replica
        total_usable_capacity = required_replicas * usable_capacity_per_replica
        spare_usable_capacity = total_usable_capacity - slo.required_invocations_per_second
        planned_utilization_fraction = slo.required_invocations_per_second / total_measured_capacity

        total_hourly_cost = required_replicas * candidate.hourly_cost
        if slo.monthly_hours is not None:
            monthly_cost = total_hourly_cost * Decimal(str(slo.monthly_hours))

        hourly_invocations = Decimal(str(slo.required_invocations_per_second)) * Decimal(3600)
        cost_per_million = (total_hourly_cost / hourly_invocations) * Decimal(1_000_000)

        if slo.max_replicas is not None and required_replicas > slo.max_replicas:
            reasons.append("max_replicas_exceeded")
        if slo.max_monthly_cost is not None and monthly_cost is not None and monthly_cost > slo.max_monthly_cost:
            reasons.append("monthly_budget_exceeded")

    if slo.max_peak_memory_allocated_bytes is not None:
        if benchmark.peak_memory_allocated_bytes is None:
            reasons.append("missing_peak_memory")
        elif benchmark.peak_memory_allocated_bytes > slo.max_peak_memory_allocated_bytes:
            reasons.append("peak_memory_exceeded")

    if any(r.startswith("missing_") for r in reasons):
        status = STATUS_INSUFFICIENT_EVIDENCE
    elif reasons:
        status = STATUS_SLO_VIOLATION
    else:
        status = STATUS_FEASIBLE

    return CandidatePlan(
        candidate_id=candidate.candidate_id, status=status, reasons=tuple(reasons),
        device=benchmark.device, backend=benchmark.backend, dtype=benchmark.dtype,
        runtime_metadata=dict(benchmark.runtime_metadata),
        measured_p95_latency_seconds=p95, measured_capacity_per_replica=measured_capacity_per_replica,
        capacity_headroom_fraction=slo.capacity_headroom_fraction, usable_capacity_per_replica=usable_capacity_per_replica,
        required_replicas=required_replicas, total_measured_capacity=total_measured_capacity,
        total_usable_capacity=total_usable_capacity, spare_usable_capacity=spare_usable_capacity,
        planned_utilization_fraction=planned_utilization_fraction,
        hourly_cost_per_replica=candidate.hourly_cost, total_hourly_cost=total_hourly_cost,
        monthly_cost=monthly_cost, cost_per_million_required_invocations=cost_per_million,
        peak_memory_allocated_bytes=benchmark.peak_memory_allocated_bytes,
        provider=candidate.provider, region=candidate.region, hardware_label=candidate.hardware_label,
        price_source=candidate.price_source, price_as_of=candidate.price_as_of,
        telemetry_context=_telemetry_context(candidate.telemetry_summary),
        validation_context=_validation_context(candidate.validation_result),
    )


def _identity_mismatch_reason(reference: DeploymentCandidate, candidate: DeploymentCandidate) -> str | None:
    """Workload preset/kind, backend, and dtype must match across every
    candidate in one plan -- device is deliberately ALLOWED to differ
    (that is the entire point of right-sizing across hardware).
    """
    rb, cb = reference.benchmark_result, candidate.benchmark_result
    if rb.workload_preset != cb.workload_preset or rb.workload_kind != cb.workload_kind:
        return (
            f"workload mismatch: {reference.candidate_id!r} benchmarks {rb.workload_preset!r} "
            f"({rb.workload_kind}), {candidate.candidate_id!r} benchmarks {cb.workload_preset!r} ({cb.workload_kind})"
        )
    if rb.backend != cb.backend:
        return f"backend mismatch: {reference.candidate_id!r}={rb.backend!r}, {candidate.candidate_id!r}={cb.backend!r}"
    if rb.dtype != cb.dtype:
        return f"dtype mismatch: {reference.candidate_id!r}={rb.dtype!r}, {candidate.candidate_id!r}={cb.dtype!r}"
    return None


def _rank_key(plan: CandidatePlan):
    memory = plan.peak_memory_allocated_bytes if plan.peak_memory_allocated_bytes is not None else math.inf
    return (
        plan.total_hourly_cost,
        plan.required_replicas,
        plan.measured_p95_latency_seconds,
        -plan.spare_usable_capacity,
        memory,
        plan.candidate_id,
    )


@dataclass(frozen=True)
class SizingPlanResult:
    status: str
    error_message: str | None

    workload_preset: str | None
    workload_kind: str | None
    backend: str | None
    dtype: str | None
    currency: str | None

    required_invocations_per_second: float | None
    max_p95_latency_seconds: float | None
    capacity_headroom_fraction: float | None
    monthly_hours: float | None

    candidate_plans: tuple
    feasible_candidate_count: int
    ranking: tuple
    recommended_candidate_id: str | None

    assumptions: tuple = field(default_factory=lambda: _ASSUMPTIONS)
    sizing_result_schema_version: int = SIZING_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "sizing_result_schema_version": self.sizing_result_schema_version,
            "status": self.status,
            "error_message": self.error_message,
            "workload": {"preset": self.workload_preset, "kind": self.workload_kind},
            "backend": self.backend,
            "dtype": self.dtype,
            "currency": self.currency,
            "slo": {
                "required_invocations_per_second": self.required_invocations_per_second,
                "max_p95_latency_seconds": self.max_p95_latency_seconds,
                "capacity_headroom_fraction": self.capacity_headroom_fraction,
                "monthly_hours": self.monthly_hours,
            },
            "candidate_plans": [c.to_dict() for c in self.candidate_plans],
            "feasible_candidate_count": self.feasible_candidate_count,
            "ranking": list(self.ranking),
            "recommended_candidate_id": self.recommended_candidate_id,
            "assumptions": list(self.assumptions),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def _error_plan(message: str, catalog: DeploymentCatalog | None = None) -> SizingPlanResult:
    return SizingPlanResult(
        status=PLAN_STATUS_ERROR, error_message=message,
        workload_preset=None, workload_kind=None, backend=None, dtype=None,
        currency=catalog.currency if catalog is not None else None,
        required_invocations_per_second=None, max_p95_latency_seconds=None,
        capacity_headroom_fraction=None, monthly_hours=None,
        candidate_plans=(), feasible_candidate_count=0, ranking=(), recommended_candidate_id=None,
    )


def build_sizing_plan(catalog: DeploymentCatalog, slo: SloPolicy) -> SizingPlanResult:
    """Pure, deterministic: never re-runs a benchmark, never fetches
    pricing, never mutates `catalog`/`slo`. Returns a SizingPlanResult
    with status=ERROR (not a raised exception) when candidates in the
    catalog do not describe comparable work -- consistent with
    Milestone 14's RegressionResult ERROR-as-status pattern, so a
    diagnostic report can still be produced.
    """
    reference = catalog.candidates[0]
    for candidate in catalog.candidates[1:]:
        mismatch = _identity_mismatch_reason(reference, candidate)
        if mismatch is not None:
            return _error_plan(
                f"catalog candidates do not benchmark the same workload/backend/dtype, cannot compare: {mismatch}",
                catalog,
            )

    candidate_plans = tuple(_evaluate_candidate(c, slo) for c in catalog.candidates)
    feasible = tuple(p for p in candidate_plans if p.status == STATUS_FEASIBLE)
    ranking = tuple(p.candidate_id for p in sorted(feasible, key=_rank_key))
    recommended = ranking[0] if ranking else None

    rb = reference.benchmark_result
    return SizingPlanResult(
        status=PLAN_STATUS_OK, error_message=None,
        workload_preset=rb.workload_preset, workload_kind=rb.workload_kind, backend=rb.backend, dtype=rb.dtype,
        currency=catalog.currency,
        required_invocations_per_second=slo.required_invocations_per_second,
        max_p95_latency_seconds=slo.max_p95_latency_seconds,
        capacity_headroom_fraction=slo.capacity_headroom_fraction,
        monthly_hours=slo.monthly_hours,
        candidate_plans=candidate_plans, feasible_candidate_count=len(feasible),
        ranking=ranking, recommended_candidate_id=recommended,
    )


def build_sizing_plans_for_scenarios(catalog: DeploymentCatalog, base_slo: SloPolicy, scenarios) -> dict:
    """Evaluate the same catalog/pricing/latency/headroom policy at
    several demand levels -- only `required_invocations_per_second`
    changes per scenario. Static snapshots only: no time-varying
    traffic, scale-up/down delay, or autoscaler behavior is simulated.
    """
    return {
        scenario.name: build_sizing_plan(catalog, replace(base_slo, required_invocations_per_second=scenario.required_invocations_per_second))
        for scenario in scenarios
    }


# --- Markdown report -----------------------------------------------------------

def _format_seconds(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _format_money(value: Decimal | None, currency: str | None) -> str:
    if value is None:
        return "--"
    return f"{value:.2f} {currency}" if currency else f"{value:.2f}"


def render_sizing_markdown_report(plan: SizingPlanResult) -> str:
    lines = ["# TensorForge Right-Sizing Plan", ""]

    if plan.status == PLAN_STATUS_ERROR:
        lines.append("**Status: ERROR**")
        lines.append("")
        lines.append(plan.error_message or "")
        return "\n".join(lines) + "\n"

    lines.append(f"Workload: `{plan.workload_preset}` ({plan.workload_kind}, {plan.backend}, {plan.dtype})")
    lines.append("")
    lines.append(f"Demand: {plan.required_invocations_per_second:g} invocations/sec")
    lines.append("")
    lines.append(
        f"SLO: p95 <= {_format_seconds(plan.max_p95_latency_seconds)}, "
        f"capacity headroom = {plan.capacity_headroom_fraction * 100:.0f}%"
    )
    lines.append("")
    lines.append("| Candidate | p95 | Measured cap. | Usable/replica | Replicas | Hourly | Monthly | Status |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for c in plan.candidate_plans:
        p95 = _format_seconds(c.measured_p95_latency_seconds) if c.measured_p95_latency_seconds is not None else "n/a"
        replicas = str(c.required_replicas) if c.required_replicas is not None else "--"
        hourly = _format_money(c.total_hourly_cost, plan.currency)
        monthly = _format_money(c.monthly_cost, plan.currency)
        status = c.status if not c.reasons else f"{c.status} ({', '.join(c.reasons)})"
        lines.append(
            f"| {c.candidate_id} | {p95} | {c.measured_capacity_per_replica:.1f}/s | "
            f"{c.usable_capacity_per_replica:.1f}/s | {replicas} | {hourly} | {monthly} | {status} |"
        )
    lines.append("")

    if plan.recommended_candidate_id is not None:
        recommended = next(c for c in plan.candidate_plans if c.candidate_id == plan.recommended_candidate_id)
        lines.append(f"Recommendation: **{plan.recommended_candidate_id}** (cheapest feasible candidate among those supplied)")
        lines.append("")
        lines.append("Why:")
        lines.append(
            f"- measured p95 {_format_seconds(recommended.measured_p95_latency_seconds)} satisfies the "
            f"{_format_seconds(plan.max_p95_latency_seconds)} requirement"
        )
        lines.append(
            f"- {recommended.required_replicas} replica(s) provide "
            f"{recommended.total_usable_capacity:.1f} invocations/sec usable capacity after "
            f"{plan.capacity_headroom_fraction * 100:.0f}% headroom"
        )
        lines.append(f"- required demand is {plan.required_invocations_per_second:g} invocations/sec")
        lines.append(f"- total cost is {_format_money(recommended.total_hourly_cost, plan.currency)}/hour")
        for c in plan.candidate_plans:
            if c.candidate_id == plan.recommended_candidate_id:
                continue
            if c.status == STATUS_FEASIBLE:
                lines.append(
                    f"- {c.candidate_id} is also feasible but requires {c.required_replicas} replica(s) "
                    f"and costs {_format_money(c.total_hourly_cost, plan.currency)}/hour"
                )
            else:
                lines.append(f"- {c.candidate_id} does not qualify: {', '.join(c.reasons) or c.status}")
        if recommended.telemetry_context is not None:
            tc = recommended.telemetry_context
            gpu_mean = tc.get("gpu_utilization_percent_mean")
            if gpu_mean is not None:
                lines.append(
                    f"- telemetry context: mean GPU activity {gpu_mean:.0f}% during the separate telemetry "
                    "window (diagnostic evidence, not a factor in this recommendation)"
                )
            if tc.get("throttle_reasons_observed"):
                lines.append(f"- telemetry context: throttle reasons observed: {', '.join(tc['throttle_reasons_observed'])}")
        if recommended.validation_context is not None:
            vc = recommended.validation_context
            lines.append(
                f"- calibration context: analytical prediction APE was {vc['absolute_percentage_error']:.1f}% "
                "for this workload (diagnostic context, not a factor in this recommendation)"
            )
    else:
        lines.append("Recommendation: **NO FEASIBLE CANDIDATE**")
        lines.append("")
        lines.append("None of the supplied candidates satisfy the stated SLO assumptions:")
        for c in plan.candidate_plans:
            lines.append(f"- {c.candidate_id}: {c.status} ({', '.join(c.reasons) or 'no reason recorded'})")
    lines.append("")

    lines.append("Assumptions:")
    for a in plan.assumptions:
        lines.append(f"- {a}")

    return "\n".join(lines) + "\n"
