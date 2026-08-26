"""Model-change performance and deployment impact report.

Torch-free, MLflow-free, NVML-free: composes already-computed
RegressionResult/SizingPlanResult/TelemetrySummary/ValidationResult/Core
evidence into one deterministic report. Never re-runs a benchmark,
telemetry window, calibration, or sizing plan -- this is a fast, offline
composition layer over existing artifacts.

Scope discipline (read before changing readiness logic):

The final recommendation covers PERFORMANCE AND INFRASTRUCTURE
READINESS ONLY. It never evaluates model accuracy, F1, mAP, perplexity,
BLEU, correctness, business value, safety, or user experience --
TensorForge Ops has no evidence about any of those. Every report states
this explicitly. There is no composite/weighted score (no "impact score
= 87/100"); evidence stays separate and the policy that turns it into a
decision stays explicit and inspectable.

Decision hierarchy:

    RegressionResult (measured base-vs-candidate performance)
    SizingPlanResult (measured deployment feasibility/cost)
        --> together with an explicit ImpactPolicy, decide readiness

    Core result / ValidationResult / TelemetrySummary
        --> context only; NEVER override a measured decision

A cheaper deployment never overrides a failed regression. A better
analytical prediction never overrides a failed regression. Telemetry
never gates anything in this milestone.
"""

import math
from dataclasses import dataclass, field
from decimal import Decimal

from tensorforge_ops.benchmark import BenchmarkResult, load_benchmark_result
from tensorforge_ops.calibration import ValidationResult, load_validation_result
from tensorforge_ops.regression import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_PASS,
    RegressionResult,
    load_regression_result,
)
from tensorforge_ops.sizing import (
    PLAN_STATUS_ERROR,
    PLAN_STATUS_OK,
    SizingPlanResult,
    load_sizing_plan_result,
)
from tensorforge_ops.telemetry import (
    TelemetryCorrelationResult,
    TelemetrySummary,
    correlate_telemetry,
    load_telemetry_summary,
)

IMPACT_MANIFEST_SCHEMA_VERSION = 1
IMPACT_POLICY_SCHEMA_VERSION = 1
IMPACT_RESULT_SCHEMA_VERSION = 1

READINESS_READY = "PERFORMANCE_READY"
READINESS_BLOCKED = "PERFORMANCE_BLOCKED"
READINESS_REVIEW = "REVIEW_REQUIRED"

_MODEL_QUALITY_DISCLAIMER = (
    "TensorForge Ops does not assess model quality or correctness. This recommendation "
    "covers measured performance and deployment/infrastructure evidence only."
)

_LIMITATIONS = (
    "This is a performance-and-infrastructure readiness report only -- it does not evaluate "
    "model accuracy, correctness, safety, or business value.",
    "There is no composite/weighted score; evidence is kept separate and the readiness "
    "decision comes from an explicit, inspectable ImpactPolicy.",
    "Measured RegressionResult is the source of truth for performance; measured "
    "SizingPlanResult is the source of truth for deployment feasibility/cost. Analytical "
    "(Core), calibration (ValidationResult), and telemetry evidence are context only and "
    "never override a measured decision.",
    "This report composes already-computed evidence; it never re-runs a benchmark, "
    "telemetry window, calibration probe, or sizing plan.",
    "Cost/replica policy thresholds, when configured, are explicit user policy -- there is "
    "no default numeric threshold anywhere in this module.",
)

# Analytical metrics worth comparing, taken directly from Core's own
# ExperimentResult.primary_metrics -- never re-derived. Deliberately a
# small whitelist, not a mechanical diff of every JSON field.
_ANALYTICAL_METRIC_WHITELIST = (
    "total_flops", "total_macs", "total_dram_bytes",
    "serialized_time_seconds", "perfect_overlap_time_seconds",
    "effective_arithmetic_intensity",
)


def _is_finite_nonnegative(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


# --- ImpactPolicy ---------------------------------------------------------------

@dataclass(frozen=True)
class ImpactPolicy:
    """Design choice: the two boolean requirements default to the common
    team expectation (measured regression must pass; candidate deployment
    must be feasible) -- these are explicit boolean semantics, not hidden
    numeric thresholds. The cost/replica ceilings default to `None`
    (unenforced) because there is no defensible universal number for
    "how much cost increase is acceptable" -- that is a business tradeoff
    each team must state explicitly. The CLI never ships or falls back to
    a bundled default policy file; `--policy` is always required, so the
    decision criteria are always visible and auditable in the invocation
    itself, not buried in this module.
    """

    require_regression_pass: bool = True
    require_feasible_candidate_deployment: bool = True
    max_hourly_cost_increase_fraction: float | None = None
    max_monthly_cost_increase_fraction: float | None = None
    max_replica_increase: int | None = None
    impact_policy_schema_version: int = IMPACT_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.impact_policy_schema_version != IMPACT_POLICY_SCHEMA_VERSION:
            raise ValueError(f"impact_policy_schema_version must be {IMPACT_POLICY_SCHEMA_VERSION}, got {self.impact_policy_schema_version!r}")
        if not isinstance(self.require_regression_pass, bool):
            raise ValueError(f"require_regression_pass must be a bool, got {self.require_regression_pass!r}")
        if not isinstance(self.require_feasible_candidate_deployment, bool):
            raise ValueError(f"require_feasible_candidate_deployment must be a bool, got {self.require_feasible_candidate_deployment!r}")
        for name, value in (
            ("max_hourly_cost_increase_fraction", self.max_hourly_cost_increase_fraction),
            ("max_monthly_cost_increase_fraction", self.max_monthly_cost_increase_fraction),
        ):
            if value is not None and not _is_finite_nonnegative(value):
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        if self.max_replica_increase is not None and (
            not isinstance(self.max_replica_increase, int) or isinstance(self.max_replica_increase, bool) or self.max_replica_increase < 0
        ):
            raise ValueError(f"max_replica_increase must be an int >= 0, got {self.max_replica_increase!r}")

    def to_dict(self) -> dict:
        return {
            "impact_policy_schema_version": self.impact_policy_schema_version,
            "require_regression_pass": self.require_regression_pass,
            "require_feasible_candidate_deployment": self.require_feasible_candidate_deployment,
            "max_hourly_cost_increase_fraction": self.max_hourly_cost_increase_fraction,
            "max_monthly_cost_increase_fraction": self.max_monthly_cost_increase_fraction,
            "max_replica_increase": self.max_replica_increase,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def impact_policy_from_dict(d: dict) -> ImpactPolicy:
    return ImpactPolicy(
        require_regression_pass=d.get("require_regression_pass", True),
        require_feasible_candidate_deployment=d.get("require_feasible_candidate_deployment", True),
        max_hourly_cost_increase_fraction=d.get("max_hourly_cost_increase_fraction"),
        max_monthly_cost_increase_fraction=d.get("max_monthly_cost_increase_fraction"),
        max_replica_increase=d.get("max_replica_increase"),
        impact_policy_schema_version=d.get("impact_policy_schema_version", IMPACT_POLICY_SCHEMA_VERSION),
    )


def load_impact_policy(path: str) -> ImpactPolicy:
    import json

    with open(path, encoding="utf-8") as f:
        return impact_policy_from_dict(json.load(f))


# --- manifest / evidence loading ------------------------------------------------

def _resolve_path(path: str, base_dir: str) -> str:
    import os

    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _load_core_result_summary(path: str) -> dict:
    """Reads core-result.json directly as JSON (no Core import, no
    ExperimentResult reconstruction needed) and extracts a small
    whitelist of analytical metrics plus workload identity. The Core
    fingerprint shown elsewhere in this report always comes from
    RegressionResult's own baseline/candidate fingerprint fields, never
    recomputed from this file -- re-hashing a saved JSON file is fragile
    (e.g. ExperimentResult.save() appends a trailing newline that
    compute_result_fingerprint()'s canonical hash does not include).
    """
    import json

    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    primary_metrics = d.get("primary_metrics", {})
    metrics = {
        k: primary_metrics[k] for k in _ANALYTICAL_METRIC_WHITELIST
        if k in primary_metrics and isinstance(primary_metrics[k], (int, float)) and not isinstance(primary_metrics[k], bool)
    }
    bottleneck = primary_metrics.get("bottleneck") or primary_metrics.get("largest_time_contributor")
    workload = d.get("workload", {})
    return {
        "workload_preset": workload.get("preset"),
        "workload_kind": workload.get("kind"),
        "metrics": metrics,
        "bottleneck": bottleneck,
    }


@dataclass(frozen=True)
class SideEvidence:
    core_result_summary: dict | None
    benchmark_result: BenchmarkResult | None
    validation_result: ValidationResult | None
    telemetry_summary: TelemetrySummary | None
    sizing_result: SizingPlanResult | None


def _load_side_evidence(d: dict | None, base_dir: str) -> SideEvidence:
    if not d:
        return SideEvidence(None, None, None, None, None)
    return SideEvidence(
        core_result_summary=_load_core_result_summary(_resolve_path(d["core_result"], base_dir)) if d.get("core_result") else None,
        benchmark_result=load_benchmark_result(_resolve_path(d["benchmark_result"], base_dir)) if d.get("benchmark_result") else None,
        validation_result=load_validation_result(_resolve_path(d["validation_result"], base_dir)) if d.get("validation_result") else None,
        telemetry_summary=load_telemetry_summary(_resolve_path(d["telemetry_summary"], base_dir)) if d.get("telemetry_summary") else None,
        sizing_result=load_sizing_plan_result(_resolve_path(d["sizing_result"], base_dir)) if d.get("sizing_result") else None,
    )


@dataclass(frozen=True)
class ImpactManifest:
    regression_result: RegressionResult
    baseline: SideEvidence
    candidate: SideEvidence


def load_impact_manifest(path: str) -> ImpactManifest:
    import json
    import os

    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    schema_version = d.get("impact_manifest_schema_version")
    if schema_version != IMPACT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"impact_manifest_schema_version must be {IMPACT_MANIFEST_SCHEMA_VERSION}, got {schema_version!r}")
    if not d.get("regression_result"):
        raise ValueError(
            "manifest must specify regression_result -- it is the required source of measured performance truth"
        )
    base_dir = os.path.dirname(os.path.abspath(path))
    regression_result = load_regression_result(_resolve_path(d["regression_result"], base_dir))
    baseline = _load_side_evidence(d.get("baseline"), base_dir)
    candidate = _load_side_evidence(d.get("candidate"), base_dir)
    return ImpactManifest(regression_result=regression_result, baseline=baseline, candidate=candidate)


# --- analytical impact -----------------------------------------------------------

@dataclass(frozen=True)
class AnalyticalMetricDelta:
    metric: str
    baseline_value: float
    candidate_value: float
    absolute_delta: float
    relative_delta: float | None

    def to_dict(self) -> dict:
        return {
            "metric": self.metric, "baseline_value": self.baseline_value, "candidate_value": self.candidate_value,
            "absolute_delta": self.absolute_delta, "relative_delta": self.relative_delta,
        }


def _analytical_deltas(baseline_summary: dict | None, candidate_summary: dict | None) -> tuple:
    if baseline_summary is None or candidate_summary is None:
        return ()
    common = sorted(set(baseline_summary["metrics"]) & set(candidate_summary["metrics"]))
    deltas = []
    for metric in common:
        b = baseline_summary["metrics"][metric]
        c = candidate_summary["metrics"][metric]
        absolute = c - b
        relative = (absolute / b) if b != 0 else None
        deltas.append(AnalyticalMetricDelta(metric, b, c, absolute, relative))
    return tuple(deltas)


# --- validation impact -------------------------------------------------------------

@dataclass(frozen=True)
class ValidationImpact:
    baseline_absolute_percentage_error: float | None
    candidate_absolute_percentage_error: float | None
    ape_delta_percentage_points: float | None
    baseline_measured_to_predicted_ratio: float | None
    candidate_measured_to_predicted_ratio: float | None
    baseline_predicted_bottleneck: str | None
    candidate_predicted_bottleneck: str | None

    def to_dict(self) -> dict:
        return {
            "baseline_absolute_percentage_error": self.baseline_absolute_percentage_error,
            "candidate_absolute_percentage_error": self.candidate_absolute_percentage_error,
            "ape_delta_percentage_points": self.ape_delta_percentage_points,
            "baseline_measured_to_predicted_ratio": self.baseline_measured_to_predicted_ratio,
            "candidate_measured_to_predicted_ratio": self.candidate_measured_to_predicted_ratio,
            "baseline_predicted_bottleneck": self.baseline_predicted_bottleneck,
            "candidate_predicted_bottleneck": self.candidate_predicted_bottleneck,
        }


def _validation_impact(baseline_vr: ValidationResult | None, candidate_vr: ValidationResult | None) -> ValidationImpact | None:
    if baseline_vr is None and candidate_vr is None:
        return None
    b_ape = baseline_vr.absolute_percentage_error if baseline_vr else None
    c_ape = candidate_vr.absolute_percentage_error if candidate_vr else None
    delta = (c_ape - b_ape) if (b_ape is not None and c_ape is not None) else None
    return ValidationImpact(
        baseline_absolute_percentage_error=b_ape, candidate_absolute_percentage_error=c_ape,
        ape_delta_percentage_points=delta,
        baseline_measured_to_predicted_ratio=baseline_vr.measured_to_predicted_ratio if baseline_vr else None,
        candidate_measured_to_predicted_ratio=candidate_vr.measured_to_predicted_ratio if candidate_vr else None,
        baseline_predicted_bottleneck=baseline_vr.predicted_bottleneck if baseline_vr else None,
        candidate_predicted_bottleneck=candidate_vr.predicted_bottleneck if candidate_vr else None,
    )


# --- deployment/cost impact ---------------------------------------------------------

@dataclass(frozen=True)
class DeploymentImpact:
    baseline_recommended_candidate_id: str | None
    candidate_recommended_candidate_id: str | None
    candidate_deployment_feasible: bool | None

    baseline_replicas: int | None
    candidate_replicas: int | None
    replica_delta: int | None

    baseline_hourly_cost: Decimal | None
    candidate_hourly_cost: Decimal | None
    hourly_cost_absolute_delta: Decimal | None
    hourly_cost_relative_delta: float | None

    baseline_monthly_cost: Decimal | None
    candidate_monthly_cost: Decimal | None
    monthly_cost_absolute_delta: Decimal | None
    monthly_cost_relative_delta: float | None

    def to_dict(self) -> dict:
        def _f(x):
            return float(x) if x is not None else None

        return {
            "baseline_recommended_candidate_id": self.baseline_recommended_candidate_id,
            "candidate_recommended_candidate_id": self.candidate_recommended_candidate_id,
            "candidate_deployment_feasible": self.candidate_deployment_feasible,
            "baseline_replicas": self.baseline_replicas, "candidate_replicas": self.candidate_replicas,
            "replica_delta": self.replica_delta,
            "baseline_hourly_cost": _f(self.baseline_hourly_cost), "candidate_hourly_cost": _f(self.candidate_hourly_cost),
            "hourly_cost_absolute_delta": _f(self.hourly_cost_absolute_delta),
            "hourly_cost_relative_delta": self.hourly_cost_relative_delta,
            "baseline_monthly_cost": _f(self.baseline_monthly_cost), "candidate_monthly_cost": _f(self.candidate_monthly_cost),
            "monthly_cost_absolute_delta": _f(self.monthly_cost_absolute_delta),
            "monthly_cost_relative_delta": self.monthly_cost_relative_delta,
        }


def _recommended_plan(sizing_result: SizingPlanResult | None):
    if sizing_result is None or sizing_result.recommended_candidate_id is None:
        return None
    return next((c for c in sizing_result.candidate_plans if c.candidate_id == sizing_result.recommended_candidate_id), None)


def _deployment_impact(baseline_sizing: SizingPlanResult | None, candidate_sizing: SizingPlanResult | None) -> DeploymentImpact:
    baseline_plan = _recommended_plan(baseline_sizing)
    candidate_plan = _recommended_plan(candidate_sizing)

    candidate_feasible = None
    if candidate_sizing is not None and candidate_sizing.status == PLAN_STATUS_OK:
        candidate_feasible = candidate_sizing.recommended_candidate_id is not None

    replica_delta = None
    if baseline_plan is not None and candidate_plan is not None and baseline_plan.required_replicas is not None and candidate_plan.required_replicas is not None:
        replica_delta = candidate_plan.required_replicas - baseline_plan.required_replicas

    hourly_abs = hourly_rel = None
    if baseline_plan is not None and candidate_plan is not None and baseline_plan.total_hourly_cost is not None and candidate_plan.total_hourly_cost is not None:
        hourly_abs = candidate_plan.total_hourly_cost - baseline_plan.total_hourly_cost
        hourly_rel = float(hourly_abs / baseline_plan.total_hourly_cost) if baseline_plan.total_hourly_cost != 0 else None

    monthly_abs = monthly_rel = None
    if baseline_plan is not None and candidate_plan is not None and baseline_plan.monthly_cost is not None and candidate_plan.monthly_cost is not None:
        monthly_abs = candidate_plan.monthly_cost - baseline_plan.monthly_cost
        monthly_rel = float(monthly_abs / baseline_plan.monthly_cost) if baseline_plan.monthly_cost != 0 else None

    return DeploymentImpact(
        baseline_recommended_candidate_id=baseline_sizing.recommended_candidate_id if baseline_sizing else None,
        candidate_recommended_candidate_id=candidate_sizing.recommended_candidate_id if candidate_sizing else None,
        candidate_deployment_feasible=candidate_feasible,
        baseline_replicas=baseline_plan.required_replicas if baseline_plan else None,
        candidate_replicas=candidate_plan.required_replicas if candidate_plan else None,
        replica_delta=replica_delta,
        baseline_hourly_cost=baseline_plan.total_hourly_cost if baseline_plan else None,
        candidate_hourly_cost=candidate_plan.total_hourly_cost if candidate_plan else None,
        hourly_cost_absolute_delta=hourly_abs, hourly_cost_relative_delta=hourly_rel,
        baseline_monthly_cost=baseline_plan.monthly_cost if baseline_plan else None,
        candidate_monthly_cost=candidate_plan.monthly_cost if candidate_plan else None,
        monthly_cost_absolute_delta=monthly_abs, monthly_cost_relative_delta=monthly_rel,
    )


# --- readiness decision --------------------------------------------------------

def _identity_mismatch_reasons(regression_result: RegressionResult, baseline: SideEvidence, candidate: SideEvidence) -> list:
    reasons = []
    for label, evidence in (("baseline", baseline), ("candidate", candidate)):
        summary = evidence.core_result_summary
        if summary is not None and (
            summary["workload_preset"] != regression_result.workload_preset
            or summary["workload_kind"] != regression_result.workload_kind
        ):
            reasons.append("artifact_identity_mismatch")
        bench = evidence.benchmark_result
        if bench is not None and (
            bench.workload_preset != regression_result.workload_preset or bench.workload_kind != regression_result.workload_kind
        ):
            reasons.append("artifact_identity_mismatch")
    return reasons


def _decide(
    policy: ImpactPolicy,
    regression_result: RegressionResult,
    candidate_sizing: SizingPlanResult | None,
    deployment_impact: DeploymentImpact,
    identity_mismatch_reasons: list,
) -> tuple:
    reasons: list = list(dict.fromkeys(identity_mismatch_reasons))  # dedupe, preserve order
    review = bool(identity_mismatch_reasons)
    blocked = False

    if regression_result.status == STATUS_ERROR:
        reasons.append("regression_error")
        review = True
    elif regression_result.status == STATUS_FAIL and policy.require_regression_pass:
        reasons.append("measured_regression_failed")
        blocked = True

    if policy.require_feasible_candidate_deployment:
        if candidate_sizing is None:
            reasons.append("missing_candidate_sizing")
            review = True
        elif candidate_sizing.status == PLAN_STATUS_ERROR:
            reasons.append("candidate_sizing_invalid")
            review = True
        elif candidate_sizing.recommended_candidate_id is None:
            reasons.append("candidate_deployment_infeasible")
            blocked = True

    if policy.max_hourly_cost_increase_fraction is not None:
        if deployment_impact.hourly_cost_relative_delta is None:
            reasons.append("missing_cost_evidence")
            review = True
        elif deployment_impact.hourly_cost_relative_delta > policy.max_hourly_cost_increase_fraction:
            reasons.append("hourly_cost_increase_exceeded")
            blocked = True

    if policy.max_monthly_cost_increase_fraction is not None:
        if deployment_impact.monthly_cost_relative_delta is None:
            reasons.append("missing_cost_evidence")
            review = True
        elif deployment_impact.monthly_cost_relative_delta > policy.max_monthly_cost_increase_fraction:
            reasons.append("monthly_cost_increase_exceeded")
            blocked = True

    if policy.max_replica_increase is not None:
        if deployment_impact.replica_delta is None:
            reasons.append("missing_replica_evidence")
            review = True
        elif deployment_impact.replica_delta > policy.max_replica_increase:
            reasons.append("replica_increase_exceeded")
            blocked = True

    if review:
        readiness = READINESS_REVIEW
    elif blocked:
        readiness = READINESS_BLOCKED
    else:
        readiness = READINESS_READY

    return readiness, tuple(dict.fromkeys(reasons))


def _missing_evidence(baseline: SideEvidence, candidate: SideEvidence) -> tuple:
    missing = []
    for label, evidence in (("baseline", baseline), ("candidate", candidate)):
        for field_name, value in (
            ("core_result", evidence.core_result_summary), ("validation_result", evidence.validation_result),
            ("telemetry_summary", evidence.telemetry_summary), ("sizing_result", evidence.sizing_result),
        ):
            if value is None:
                missing.append(f"{label} {field_name}: NOT PROVIDED")
    return tuple(missing)


# --- ImpactResult ----------------------------------------------------------------

@dataclass(frozen=True)
class ImpactResult:
    readiness: str
    decision_reasons: tuple

    workload_preset: str | None
    workload_kind: str | None
    backend: str | None
    dtype: str | None

    baseline_core_fingerprint: str | None
    candidate_core_fingerprint: str | None
    core_fingerprint_changed: bool | None

    regression_status: str
    regression_error_message: str | None
    measured_metric_deltas: tuple

    analytical_deltas: tuple
    baseline_predicted_bottleneck: str | None
    candidate_predicted_bottleneck: str | None

    validation_impact: ValidationImpact | None
    telemetry_correlation: TelemetryCorrelationResult | None

    deployment_impact: DeploymentImpact

    missing_evidence: tuple
    limitations: tuple = field(default_factory=lambda: _LIMITATIONS)
    model_quality_disclaimer: str = _MODEL_QUALITY_DISCLAIMER
    impact_result_schema_version: int = IMPACT_RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "impact_result_schema_version": self.impact_result_schema_version,
            "readiness": self.readiness,
            "decision_reasons": list(self.decision_reasons),
            "workload": {"preset": self.workload_preset, "kind": self.workload_kind},
            "backend": self.backend,
            "dtype": self.dtype,
            "core_fingerprints": {
                "baseline": self.baseline_core_fingerprint,
                "candidate": self.candidate_core_fingerprint,
                "changed": self.core_fingerprint_changed,
            },
            "regression": {
                "status": self.regression_status,
                "error_message": self.regression_error_message,
                "metric_deltas": [c.to_dict() for c in self.measured_metric_deltas],
            },
            "analytical_context": {
                "deltas": [d.to_dict() for d in self.analytical_deltas],
                "baseline_predicted_bottleneck": self.baseline_predicted_bottleneck,
                "candidate_predicted_bottleneck": self.candidate_predicted_bottleneck,
            },
            "validation_context": self.validation_impact.to_dict() if self.validation_impact is not None else None,
            "telemetry_context": self.telemetry_correlation.to_dict() if self.telemetry_correlation is not None else None,
            "deployment_impact": self.deployment_impact.to_dict(),
            "missing_evidence": list(self.missing_evidence),
            "limitations": list(self.limitations),
            "model_quality_disclaimer": self.model_quality_disclaimer,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def build_impact_result(manifest: ImpactManifest, policy: ImpactPolicy) -> ImpactResult:
    """Pure, deterministic composition. Never reruns any evidence-producing
    step; never mutates the manifest's loaded objects. Given the same
    already-loaded evidence and policy, always returns byte-identical JSON.
    """
    regression_result = manifest.regression_result
    baseline, candidate = manifest.baseline, manifest.candidate

    identity_mismatches = _identity_mismatch_reasons(regression_result, baseline, candidate)
    deployment_impact = _deployment_impact(baseline.sizing_result, candidate.sizing_result)
    readiness, reasons = _decide(policy, regression_result, candidate.sizing_result, deployment_impact, identity_mismatches)

    telemetry_correlation = None
    if baseline.telemetry_summary is not None and candidate.telemetry_summary is not None:
        telemetry_correlation = correlate_telemetry(
            baseline.telemetry_summary, candidate.telemetry_summary,
            regression_status=regression_result.status,
            workload_preset=regression_result.workload_preset, workload_kind=regression_result.workload_kind,
        )

    baseline_fp = regression_result.baseline_core_fingerprint
    candidate_fp = regression_result.candidate_core_fingerprint
    fingerprint_changed = (baseline_fp != candidate_fp) if (baseline_fp is not None and candidate_fp is not None) else None

    return ImpactResult(
        readiness=readiness, decision_reasons=reasons,
        workload_preset=regression_result.workload_preset, workload_kind=regression_result.workload_kind,
        backend=regression_result.backend, dtype=regression_result.dtype,
        baseline_core_fingerprint=baseline_fp, candidate_core_fingerprint=candidate_fp,
        core_fingerprint_changed=fingerprint_changed,
        regression_status=regression_result.status, regression_error_message=regression_result.error_message,
        measured_metric_deltas=regression_result.metric_comparisons,
        analytical_deltas=_analytical_deltas(baseline.core_result_summary, candidate.core_result_summary),
        baseline_predicted_bottleneck=baseline.core_result_summary["bottleneck"] if baseline.core_result_summary else None,
        candidate_predicted_bottleneck=candidate.core_result_summary["bottleneck"] if candidate.core_result_summary else None,
        validation_impact=_validation_impact(baseline.validation_result, candidate.validation_result),
        telemetry_correlation=telemetry_correlation,
        deployment_impact=deployment_impact,
        missing_evidence=_missing_evidence(baseline, candidate),
    )


# --- Markdown report -----------------------------------------------------------

def _format_seconds(seconds) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _format_pct(fraction) -> str:
    return f"{fraction * 100:+.1f}%" if fraction is not None else "n/a"


def _format_money(value, currency=None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} {currency}" if currency else f"{value:.2f}"


def render_impact_markdown_report(result: ImpactResult) -> str:
    lines = ["# TensorForge Model-Change Impact", ""]

    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**{result.readiness}**")
    lines.append("")
    if result.readiness == READINESS_READY:
        lines.append("Ready from the measured performance and deployment perspective.")
    elif result.readiness == READINESS_BLOCKED:
        lines.append("Blocked from the measured performance and deployment perspective.")
    else:
        lines.append("Evidence is missing or invalid -- a decision cannot be made automatically.")
    if result.decision_reasons:
        lines.append("")
        lines.append("Reasons: " + ", ".join(result.decision_reasons))
    lines.append("")
    lines.append(result.model_quality_disclaimer)
    lines.append("")

    lines.append("## Measured performance")
    lines.append("")
    lines.append(f"Regression gate: **{result.regression_status}**" + (f" ({result.regression_error_message})" if result.regression_error_message else ""))
    lines.append("")
    if result.measured_metric_deltas:
        lines.append("| Metric | Base | Candidate | Change | Status |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for c in result.measured_metric_deltas:
            base = _format_seconds(c.baseline_value) if "latency" in c.metric else (f"{c.baseline_value:.2f}" if c.baseline_value is not None else "n/a")
            cand = _format_seconds(c.candidate_value) if "latency" in c.metric else (f"{c.candidate_value:.2f}" if c.candidate_value is not None else "n/a")
            change = _format_pct(c.relative_delta)
            lines.append(f"| {c.metric} | {base} | {cand} | {change} | {c.status} |")
    lines.append("")

    lines.append("## Deployment impact")
    lines.append("")
    di = result.deployment_impact
    lines.append(f"Baseline: {di.baseline_recommended_candidate_id or 'n/a'}, replicas={di.baseline_replicas if di.baseline_replicas is not None else 'n/a'}, hourly={_format_money(di.baseline_hourly_cost)}")
    lines.append(f"Candidate: {di.candidate_recommended_candidate_id or 'n/a'}, replicas={di.candidate_replicas if di.candidate_replicas is not None else 'n/a'}, hourly={_format_money(di.candidate_hourly_cost)}")
    lines.append(f"Candidate deployment feasible: {di.candidate_deployment_feasible if di.candidate_deployment_feasible is not None else 'unknown (no sizing evidence)'}")
    lines.append(f"Replica change: {di.replica_delta if di.replica_delta is not None else 'n/a'}")
    lines.append(f"Hourly cost change: {_format_money(di.hourly_cost_absolute_delta)} ({_format_pct(di.hourly_cost_relative_delta)})")
    if di.baseline_monthly_cost is not None or di.candidate_monthly_cost is not None:
        lines.append(f"Monthly cost change: {_format_money(di.monthly_cost_absolute_delta)} ({_format_pct(di.monthly_cost_relative_delta)})")
    lines.append("")

    lines.append("## Analytical context")
    lines.append("")
    lines.append(
        f"Core fingerprint: {'unchanged' if result.core_fingerprint_changed is False else ('changed' if result.core_fingerprint_changed else 'unknown')}"
    )
    if result.analytical_deltas:
        lines.append("")
        for d in result.analytical_deltas:
            lines.append(f"- {d.metric}: {d.baseline_value:g} -> {d.candidate_value:g} ({_format_pct(d.relative_delta)})")
    if result.baseline_predicted_bottleneck or result.candidate_predicted_bottleneck:
        lines.append(f"- predicted bottleneck: {result.baseline_predicted_bottleneck or 'n/a'} -> {result.candidate_predicted_bottleneck or 'n/a'}")
    lines.append("")

    lines.append("## GPU telemetry evidence")
    lines.append("")
    if result.telemetry_correlation is not None:
        if result.telemetry_correlation.signals:
            for s in result.telemetry_correlation.signals:
                lines.append(f"- {s}")
        else:
            lines.append("No material telemetry signals observed between baseline and candidate.")
        lines.append("")
        lines.append("Telemetry is diagnostic context and does not affect readiness.")
    else:
        lines.append("Telemetry evidence not provided.")
    lines.append("")

    lines.append("## Model-validation context")
    lines.append("")
    vi = result.validation_impact
    if vi is not None:
        lines.append(f"Prediction APE: {vi.baseline_absolute_percentage_error if vi.baseline_absolute_percentage_error is not None else 'n/a'}% -> {vi.candidate_absolute_percentage_error if vi.candidate_absolute_percentage_error is not None else 'n/a'}%")
        if vi.ape_delta_percentage_points is not None:
            lines.append(f"APE change: {vi.ape_delta_percentage_points:+.1f} percentage points")
        lines.append("This describes analytical-model-vs-hardware agreement, not deployment reliability.")
    else:
        lines.append("Analytical-vs-measured validation evidence not provided.")
    lines.append("")

    lines.append("## Missing evidence")
    lines.append("")
    if result.missing_evidence:
        for m in result.missing_evidence:
            lines.append(f"- {m}")
    else:
        lines.append("All optional evidence was provided.")
    lines.append("")

    lines.append("## Important limitations")
    lines.append("")
    for limitation in result.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append(result.model_quality_disclaimer)

    return "\n".join(lines) + "\n"
