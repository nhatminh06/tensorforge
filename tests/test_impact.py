"""Tests for tensorforge_ops.impact: pure composition of already-computed
RegressionResult/SizingPlanResult/TelemetrySummary/ValidationResult/Core
evidence into a deterministic performance-and-infrastructure readiness
report. No torch/mlflow/pynvml dependency.
"""

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from tensorforge_ops.benchmark import BenchmarkResult, LatencyStatistics
from tensorforge_ops.calibration import ValidationResult
from tensorforge_ops.regression import MetricPolicy, RegressionPolicy, compare_benchmark_results
from tensorforge_ops.sizing import DeploymentCandidate, DeploymentCatalog, SloPolicy, build_sizing_plan
from tensorforge_ops.telemetry import MetricStat, TelemetrySummary
from tensorforge_ops.impact import (
    READINESS_BLOCKED,
    READINESS_READY,
    READINESS_REVIEW,
    ImpactManifest,
    ImpactPolicy,
    SideEvidence,
    build_impact_result,
    impact_policy_from_dict,
    load_impact_manifest,
    load_impact_policy,
    render_impact_markdown_report,
)


def make_bench(p50=0.00025, throughput=4000.0, mem=None, preset="gemm_large_square", kind="gemm", backend="pytorch", dtype="fp16"):
    stats = LatencyStatistics(
        count=50, mean_seconds=p50, p50_seconds=p50, p95_seconds=p50 * 1.1, p99_seconds=None,
        min_seconds=p50 * 0.9, max_seconds=p50 * 1.2, throughput_per_second=throughput,
    )
    return BenchmarkResult(
        core_result_fingerprint="sha256:" + "0" * 64, workload_preset=preset, workload_kind=kind,
        backend=backend, device="cuda:0", dtype=dtype, warmup_iterations=10, measured_iterations=50,
        latency_samples_seconds=(p50,), statistics=stats, peak_memory_allocated_bytes=mem,
        runtime_metadata={"torch_version": "2.13.0+cu126"},
    )


def make_regression(baseline_p50=0.00025, candidate_p50=0.00022, baseline_tput=4000.0, candidate_tput=4500.0, relative=0.10, **overrides):
    policy = RegressionPolicy(metrics={
        "p50_latency_seconds": MetricPolicy(max_relative_regression=relative),
        "throughput_per_second": MetricPolicy(max_relative_regression=relative),
    })
    baseline = make_bench(p50=baseline_p50, throughput=baseline_tput)
    candidate = make_bench(p50=candidate_p50, throughput=candidate_tput, **overrides)
    return compare_benchmark_results(baseline, candidate, policy)


def make_sizing_plan(p50, throughput, demand=3800.0, max_p95=0.001, hourly_cost="0.80", headroom=0.1, monthly_hours=730):
    bench = make_bench(p50=p50, throughput=throughput)
    catalog = DeploymentCatalog(1, "USD", (DeploymentCandidate("gpu-a", bench, Decimal(hourly_cost), "USD"),))
    slo = SloPolicy(required_invocations_per_second=demand, max_p95_latency_seconds=max_p95, capacity_headroom_fraction=headroom, monthly_hours=monthly_hours)
    return build_sizing_plan(catalog, slo)


def make_manifest(regression_result, baseline_sizing=None, candidate_sizing=None, baseline_validation=None, candidate_validation=None, baseline_telemetry=None, candidate_telemetry=None):
    return ImpactManifest(
        regression_result=regression_result,
        baseline=SideEvidence(None, None, baseline_validation, baseline_telemetry, baseline_sizing),
        candidate=SideEvidence(None, None, candidate_validation, candidate_telemetry, candidate_sizing),
    )


# --- ImpactPolicy validation --------------------------------------------------

def test_policy_defaults():
    policy = ImpactPolicy()
    assert policy.require_regression_pass is True
    assert policy.require_feasible_candidate_deployment is True
    assert policy.max_hourly_cost_increase_fraction is None
    assert policy.max_monthly_cost_increase_fraction is None
    assert policy.max_replica_increase is None


def test_policy_rejects_negative_cost_fraction():
    with pytest.raises(ValueError):
        ImpactPolicy(max_hourly_cost_increase_fraction=-0.1)


def test_policy_rejects_nan_cost_fraction():
    with pytest.raises(ValueError):
        ImpactPolicy(max_monthly_cost_increase_fraction=float("nan"))


def test_policy_rejects_inf_cost_fraction():
    with pytest.raises(ValueError):
        ImpactPolicy(max_hourly_cost_increase_fraction=float("inf"))


def test_policy_rejects_negative_replica_increase():
    with pytest.raises(ValueError):
        ImpactPolicy(max_replica_increase=-1)


def test_policy_rejects_non_bool_require_flags():
    with pytest.raises(ValueError):
        ImpactPolicy(require_regression_pass="yes")


def test_policy_json_roundtrip():
    policy = ImpactPolicy(max_hourly_cost_increase_fraction=0.2, max_replica_increase=1)
    restored = impact_policy_from_dict(json.loads(policy.to_json()))
    assert restored.to_json() == policy.to_json()


def test_load_impact_policy_from_file(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"impact_policy_schema_version": 1, "require_regression_pass": True, "require_feasible_candidate_deployment": False}))
    policy = load_impact_policy(str(path))
    assert policy.require_feasible_candidate_deployment is False


def test_policy_rejects_bad_schema_version(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"impact_policy_schema_version": 99}))
    with pytest.raises(ValueError):
        load_impact_policy(str(path))


# --- manifest loading (relative paths) ------------------------------------------

def test_manifest_loading_resolves_relative_paths(tmp_path):
    (tmp_path / "results").mkdir()
    regression_result = make_regression()
    (tmp_path / "results" / "regression-result.json").write_text(regression_result.to_json())

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "impact_manifest_schema_version": 1,
        "regression_result": "results/regression-result.json",
    }))
    manifest = load_impact_manifest(str(manifest_path))
    assert manifest.regression_result.status == regression_result.status


def test_manifest_requires_regression_result(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"impact_manifest_schema_version": 1}))
    with pytest.raises(ValueError, match="regression_result"):
        load_impact_manifest(str(path))


def test_manifest_missing_file_raises_not_traceback(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"impact_manifest_schema_version": 1, "regression_result": "does-not-exist.json"}))
    with pytest.raises(OSError):
        load_impact_manifest(str(path))


def test_manifest_rejects_bad_schema_version(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"impact_manifest_schema_version": 99, "regression_result": "x.json"}))
    with pytest.raises(ValueError):
        load_impact_manifest(str(path))


# --- readiness: known scenarios (hard-coded) --------------------------------------

def test_ready_no_cost_or_replica_limits():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0)
    candidate_sizing = make_sizing_plan(0.00022, 4500.0)
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)
    result = build_impact_result(manifest, ImpactPolicy())
    assert result.readiness == READINESS_READY
    assert result.decision_reasons == ()


def test_blocked_by_regression_failure():
    regression = make_regression(candidate_p50=0.00040)  # much slower -> FAIL
    candidate_sizing = make_sizing_plan(0.00040, 4000.0)
    manifest = make_manifest(regression, candidate_sizing=candidate_sizing)
    result = build_impact_result(manifest, ImpactPolicy())
    assert result.readiness == READINESS_BLOCKED
    assert "measured_regression_failed" in result.decision_reasons


def test_blocked_by_deployment_infeasibility():
    regression = make_regression()
    infeasible_sizing = make_sizing_plan(0.00022, 4500.0, max_p95=0.0001)  # impossible SLO
    manifest = make_manifest(regression, candidate_sizing=infeasible_sizing)
    result = build_impact_result(manifest, ImpactPolicy())
    assert result.readiness == READINESS_BLOCKED
    assert "candidate_deployment_infeasible" in result.decision_reasons


def test_review_on_regression_error():
    baseline = make_bench(preset="gemm_tiny", kind="gemm")
    candidate = make_bench(preset="conv_spatial", kind="conv2d")
    policy = RegressionPolicy(metrics={"p50_latency_seconds": MetricPolicy(max_relative_regression=0.1)})
    error_regression = compare_benchmark_results(baseline, candidate, policy)
    manifest = make_manifest(error_regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.readiness == READINESS_REVIEW
    assert "regression_error" in result.decision_reasons


def test_review_on_missing_required_sizing():
    regression = make_regression()
    manifest = make_manifest(regression)  # no sizing at all
    result = build_impact_result(manifest, ImpactPolicy())
    assert result.readiness == READINESS_REVIEW
    assert "missing_candidate_sizing" in result.decision_reasons


def test_sizing_optional_when_policy_says_so():
    regression = make_regression()
    manifest = make_manifest(regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.readiness == READINESS_READY
    assert any("candidate sizing_result: NOT PROVIDED" in m for m in result.missing_evidence)


# --- cost policy -----------------------------------------------------------------------

def test_hourly_cost_increase_pass():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0, hourly_cost="1.00", demand=3600.0)
    candidate_sizing = make_sizing_plan(0.00022, 4000.0, hourly_cost="1.10", demand=3600.0)
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)
    result = build_impact_result(manifest, ImpactPolicy(max_hourly_cost_increase_fraction=0.20))
    assert result.readiness == READINESS_READY


def test_hourly_cost_increase_fail():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0, hourly_cost="1.00", demand=3600.0)
    candidate_sizing = make_sizing_plan(0.00022, 4000.0, hourly_cost="1.30", demand=3600.0)
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)
    result = build_impact_result(manifest, ImpactPolicy(max_hourly_cost_increase_fraction=0.20))
    assert result.readiness == READINESS_BLOCKED
    assert "hourly_cost_increase_exceeded" in result.decision_reasons


def test_zero_baseline_cost_relative_delta_is_none_and_reviews_if_policy_needs_it():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0, hourly_cost="0", demand=3600.0)
    candidate_sizing = make_sizing_plan(0.00022, 4000.0, hourly_cost="0.50", demand=3600.0)
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)

    result_no_policy = build_impact_result(manifest, ImpactPolicy())
    assert result_no_policy.deployment_impact.hourly_cost_relative_delta is None
    assert result_no_policy.readiness == READINESS_READY  # no cost policy configured -> fine

    result_with_policy = build_impact_result(manifest, ImpactPolicy(max_hourly_cost_increase_fraction=0.5))
    assert result_with_policy.readiness == READINESS_REVIEW
    assert "missing_cost_evidence" in result_with_policy.decision_reasons


def test_monthly_cost_pass_and_fail():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0, hourly_cost="1.00", demand=3600.0, monthly_hours=730)
    candidate_ok = make_sizing_plan(0.00022, 4000.0, hourly_cost="1.10", demand=3600.0, monthly_hours=730)
    candidate_fail = make_sizing_plan(0.00022, 4000.0, hourly_cost="1.30", demand=3600.0, monthly_hours=730)

    manifest_ok = make_manifest(regression, baseline_sizing, candidate_ok)
    result_ok = build_impact_result(manifest_ok, ImpactPolicy(max_monthly_cost_increase_fraction=0.20))
    assert result_ok.readiness == READINESS_READY

    manifest_fail = make_manifest(regression, baseline_sizing, candidate_fail)
    result_fail = build_impact_result(manifest_fail, ImpactPolicy(max_monthly_cost_increase_fraction=0.20))
    assert result_fail.readiness == READINESS_BLOCKED
    assert "monthly_cost_increase_exceeded" in result_fail.decision_reasons


def test_monthly_comparison_unavailable_when_only_one_side_has_it():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0, hourly_cost="1.00", demand=3600.0, monthly_hours=730)
    candidate_sizing = make_sizing_plan(0.00022, 4000.0, hourly_cost="1.10", demand=3600.0, monthly_hours=None)
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)
    result = build_impact_result(manifest, ImpactPolicy())
    assert result.deployment_impact.monthly_cost_relative_delta is None


# --- replica policy --------------------------------------------------------------------

def test_replica_increase_blocks_when_over_limit():
    regression = make_regression()
    # baseline: 1 replica; candidate: force 3 replicas via low throughput + high demand
    baseline_sizing = make_sizing_plan(0.00025, 4000.0, demand=3600.0, headroom=0.0)
    candidate_sizing = make_sizing_plan(0.00022, 1200.0, demand=3600.0, headroom=0.0)  # needs 3 replicas
    assert baseline_sizing.candidate_plans[0].required_replicas == 1
    assert candidate_sizing.candidate_plans[0].required_replicas == 3
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)
    result = build_impact_result(manifest, ImpactPolicy(max_replica_increase=1))
    assert result.readiness == READINESS_BLOCKED
    assert "replica_increase_exceeded" in result.decision_reasons


# --- analytical deltas / fingerprints ------------------------------------------------

def test_analytical_deltas_from_core_result_summaries():
    from tensorforge_ops.impact import _analytical_deltas

    baseline_summary = {"metrics": {"total_flops": 1000.0, "total_dram_bytes": 0.0}, "workload_preset": "x", "workload_kind": "gemm", "bottleneck": "compute-bound"}
    candidate_summary = {"metrics": {"total_flops": 1200.0, "total_dram_bytes": 500.0}, "workload_preset": "x", "workload_kind": "gemm", "bottleneck": "memory-bound"}
    deltas = {d.metric: d for d in _analytical_deltas(baseline_summary, candidate_summary)}
    assert deltas["total_flops"].absolute_delta == pytest.approx(200.0)
    assert deltas["total_flops"].relative_delta == pytest.approx(0.2)
    # zero baseline handled safely, never divides by zero
    assert deltas["total_dram_bytes"].relative_delta is None


def test_analytical_deltas_empty_when_either_side_missing():
    from tensorforge_ops.impact import _analytical_deltas

    assert _analytical_deltas(None, {"metrics": {}}) == ()
    assert _analytical_deltas({"metrics": {}}, None) == ()


def test_fingerprint_unchanged_reported():
    regression = make_regression()  # baseline and candidate share the same fingerprint by construction
    manifest = make_manifest(regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.core_fingerprint_changed is False


def test_fingerprint_changed_reported_without_gating():
    baseline = make_bench()
    candidate = replace(make_bench(), core_result_fingerprint="sha256:" + "1" * 64)
    policy = RegressionPolicy(metrics={"p50_latency_seconds": MetricPolicy(max_relative_regression=0.5)})
    regression = compare_benchmark_results(baseline, candidate, policy)
    manifest = make_manifest(regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.core_fingerprint_changed is True
    assert result.readiness == READINESS_READY  # fingerprint change never gates


# --- validation context (non-gating) ----------------------------------------------------

def make_validation_result(ape, ratio=1.5, bottleneck="compute-bound"):
    return ValidationResult(
        core_result_fingerprint="sha256:" + "0" * 64, calibration_fingerprint="sha256:" + "1" * 64,
        workload_preset="gemm_large_square", workload_kind="gemm", backend="pytorch", device="cuda:0", dtype="fp16",
        predicted_compute_seconds=0.0001, predicted_memory_seconds=0.00005, predicted_latency_seconds=0.0001,
        predicted_bottleneck=bottleneck, measured_p50_latency_seconds=0.00025, measured_mean_latency_seconds=0.00025,
        measured_p95_latency_seconds=0.0003, signed_error_seconds=-0.00015, absolute_error_seconds=0.00015,
        relative_error=-0.6, absolute_percentage_error=ape, measured_to_predicted_ratio=ratio,
    )


def test_validation_context_present_and_non_gating():
    regression = make_regression()
    baseline_validation = make_validation_result(ape=20.0)
    candidate_validation = make_validation_result(ape=40.0)  # much worse analytical agreement
    manifest = make_manifest(regression, baseline_validation=baseline_validation, candidate_validation=candidate_validation)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.validation_impact.ape_delta_percentage_points == pytest.approx(20.0)
    assert result.readiness == READINESS_READY  # worse calibration agreement never blocks


def test_missing_validation_reported_not_fabricated():
    regression = make_regression()
    manifest = make_manifest(regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.validation_impact is None
    assert any("validation_result: NOT PROVIDED" in m for m in result.missing_evidence)


# --- telemetry context (non-gating) -----------------------------------------------------

def make_telemetry_summary(gpu_mean):
    stat = MetricStat(sample_count=1, mean=gpu_mean, min=gpu_mean, max=gpu_mean)
    empty = MetricStat(sample_count=0, mean=None, min=None, max=None)
    return TelemetrySummary(
        telemetry_schema_version=1, sample_count=1, duration_seconds=1.0,
        gpu_utilization_percent=stat, memory_activity_percent=empty, memory_used_bytes=empty,
        memory_total_bytes=None, power_watts=empty, temperature_celsius=empty,
        sm_clock_mhz=empty, memory_clock_mhz=empty, sm_activity_percent=empty, sm_occupancy_percent=empty,
        tensor_activity_percent=empty, dram_bandwidth_utilization_percent=empty,
        available_metrics=("gpu_utilization_percent",), unavailable_metrics=(), throttle_reasons_observed=(),
    )


def test_telemetry_correlation_computed_and_non_gating():
    regression = make_regression()
    baseline_telemetry = make_telemetry_summary(90.0)
    candidate_telemetry = make_telemetry_summary(10.0)  # much worse utilization
    manifest = make_manifest(regression, baseline_telemetry=baseline_telemetry, candidate_telemetry=candidate_telemetry)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.telemetry_correlation is not None
    assert len(result.telemetry_correlation.signals) > 0
    assert result.readiness == READINESS_READY  # telemetry never gates in this milestone


def test_missing_telemetry_reported_not_fabricated():
    regression = make_regression()
    manifest = make_manifest(regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.telemetry_correlation is None
    assert any("telemetry_summary: NOT PROVIDED" in m for m in result.missing_evidence)


# --- artifact identity mismatch ---------------------------------------------------------

def test_artifact_identity_mismatch_triggers_review():
    from tensorforge_ops.impact import SideEvidence

    regression = make_regression()  # workload_preset = gemm_large_square
    mismatched_benchmark = make_bench(preset="conv_spatial", kind="conv2d")
    manifest = ImpactManifest(
        regression_result=regression,
        baseline=SideEvidence(None, mismatched_benchmark, None, None, None),
        candidate=SideEvidence(None, None, None, None, None),
    )
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    assert result.readiness == READINESS_REVIEW
    assert "artifact_identity_mismatch" in result.decision_reasons


# --- Markdown report --------------------------------------------------------------------

def test_markdown_report_contains_required_sections():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0)
    candidate_sizing = make_sizing_plan(0.00022, 4500.0)
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)
    result = build_impact_result(manifest, ImpactPolicy())
    report = render_impact_markdown_report(result)
    assert "# TensorForge Model-Change Impact" in report
    assert "## Recommendation" in report
    assert "PERFORMANCE_READY" in report
    assert "## Measured performance" in report
    assert "## Deployment impact" in report
    assert "## Analytical context" in report
    assert "## GPU telemetry evidence" in report
    assert "## Model-validation context" in report
    assert "## Missing evidence" in report
    assert "## Important limitations" in report
    assert "does not assess model quality or correctness" in report


def test_markdown_report_never_dumps_raw_latency_samples():
    regression = make_regression()
    manifest = make_manifest(regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    report = render_impact_markdown_report(result)
    assert "latency_samples_seconds" not in report


# --- serialization -----------------------------------------------------------------------

def test_impact_result_json_deterministic_for_fixed_inputs():
    regression = make_regression()
    baseline_sizing = make_sizing_plan(0.00025, 4000.0)
    candidate_sizing = make_sizing_plan(0.00022, 4500.0)
    manifest = make_manifest(regression, baseline_sizing, candidate_sizing)
    result_a = build_impact_result(manifest, ImpactPolicy())
    result_b = build_impact_result(manifest, ImpactPolicy())
    assert result_a.to_json() == result_b.to_json()


def test_impact_result_json_structure():
    regression = make_regression()
    manifest = make_manifest(regression)
    result = build_impact_result(manifest, ImpactPolicy(require_feasible_candidate_deployment=False))
    d = json.loads(result.to_json())
    assert d["impact_result_schema_version"] == 1
    assert d["readiness"] == "PERFORMANCE_READY"
    assert "model_quality_disclaimer" in d
    assert d["model_quality_disclaimer"]
