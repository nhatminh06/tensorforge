"""Tests for tensorforge_ops.sizing: pure arithmetic over already-measured
BenchmarkResult/TelemetrySummary/ValidationResult evidence. No torch/
mlflow/pynvml dependency.
"""

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from tensorforge_ops.benchmark import BenchmarkResult, LatencyStatistics
from tensorforge_ops.telemetry import MetricStat, TelemetrySummary
from tensorforge_ops.sizing import (
    PLAN_STATUS_ERROR,
    PLAN_STATUS_OK,
    STATUS_FEASIBLE,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_SLO_VIOLATION,
    DemandScenario,
    DeploymentCandidate,
    DeploymentCatalog,
    SloPolicy,
    build_sizing_plan,
    build_sizing_plans_for_scenarios,
    load_deployment_catalog,
    load_slo_policy,
    render_sizing_markdown_report,
)


def make_bench(
    p95=0.020, throughput=100.0, peak_mem=None,
    preset="gemm_large_square", kind="gemm", backend="pytorch", device="cuda:0", dtype="fp16",
):
    stats = LatencyStatistics(
        count=50, mean_seconds=(p95 * 0.9 if p95 is not None else 0.01),
        p50_seconds=(p95 * 0.85 if p95 is not None else 0.009),
        p95_seconds=p95, p99_seconds=None,
        min_seconds=(p95 * 0.7 if p95 is not None else 0.008),
        max_seconds=(p95 * 1.1 if p95 is not None else 0.012),
        throughput_per_second=throughput,
    )
    return BenchmarkResult(
        core_result_fingerprint="sha256:" + "0" * 64, workload_preset=preset, workload_kind=kind,
        backend=backend, device=device, dtype=dtype, warmup_iterations=10, measured_iterations=50,
        latency_samples_seconds=(p95 or 0.01,), statistics=stats, peak_memory_allocated_bytes=peak_mem,
        runtime_metadata={"torch_version": "2.13.0+cu126"},
    )


def make_candidate(candidate_id, hourly_cost, currency="USD", **bench_kwargs):
    return DeploymentCandidate(candidate_id, make_bench(**bench_kwargs), Decimal(str(hourly_cost)), currency)


def make_catalog(*candidates, currency="USD"):
    return DeploymentCatalog(1, currency, tuple(candidates))


def default_slo(**overrides):
    kwargs = dict(required_invocations_per_second=100.0, max_p95_latency_seconds=0.025)
    kwargs.update(overrides)
    return SloPolicy(**kwargs)


# --- DeploymentCandidate / catalog validation --------------------------------------

def test_candidate_rejects_empty_id():
    with pytest.raises(ValueError):
        DeploymentCandidate("", make_bench(), Decimal("1"), "USD")


def test_candidate_rejects_negative_cost():
    with pytest.raises(ValueError):
        DeploymentCandidate("a", make_bench(), Decimal("-1"), "USD")


def test_candidate_allows_zero_cost_for_owned_hardware():
    candidate = DeploymentCandidate("local", make_bench(), Decimal("0"), "USD")
    assert candidate.hourly_cost == Decimal("0")


def test_candidate_rejects_nonfinite_cost():
    with pytest.raises(ValueError):
        DeploymentCandidate("a", make_bench(), Decimal("Infinity"), "USD")


def test_catalog_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        make_catalog(make_candidate("a", 1), make_candidate("a", 2))


def test_catalog_rejects_empty():
    with pytest.raises(ValueError):
        DeploymentCatalog(1, "USD", ())


def test_catalog_rejects_mixed_currency():
    with pytest.raises(ValueError, match="mixed currency"):
        make_catalog(make_candidate("a", 1, currency="USD"), make_candidate("b", 1, currency="EUR"))


# --- SloPolicy validation -----------------------------------------------------------

def test_slo_rejects_nonpositive_demand():
    with pytest.raises(ValueError):
        SloPolicy(required_invocations_per_second=0, max_p95_latency_seconds=0.02)


def test_slo_rejects_nonpositive_latency():
    with pytest.raises(ValueError):
        SloPolicy(required_invocations_per_second=10, max_p95_latency_seconds=0)


def test_slo_rejects_headroom_out_of_range():
    with pytest.raises(ValueError):
        SloPolicy(required_invocations_per_second=10, max_p95_latency_seconds=0.02, capacity_headroom_fraction=1.0)
    with pytest.raises(ValueError):
        SloPolicy(required_invocations_per_second=10, max_p95_latency_seconds=0.02, capacity_headroom_fraction=-0.1)


def test_slo_rejects_nan_headroom():
    with pytest.raises(ValueError):
        SloPolicy(required_invocations_per_second=10, max_p95_latency_seconds=0.02, capacity_headroom_fraction=float("nan"))


def test_slo_max_monthly_cost_requires_monthly_hours():
    with pytest.raises(ValueError, match="monthly_hours"):
        SloPolicy(required_invocations_per_second=10, max_p95_latency_seconds=0.02, max_monthly_cost=Decimal("100"))


def test_slo_rejects_bad_max_replicas():
    with pytest.raises(ValueError):
        SloPolicy(required_invocations_per_second=10, max_p95_latency_seconds=0.02, max_replicas=0)


# --- replica formula (known values, hard-coded) --------------------------------------

def test_known_replica_count():
    candidate = make_candidate("gpu", 1, throughput=100.0, p95=0.01)
    catalog = make_catalog(candidate)
    slo = default_slo(required_invocations_per_second=150.0, capacity_headroom_fraction=0.20)
    plan = build_sizing_plan(catalog, slo)
    cp = plan.candidate_plans[0]
    assert cp.usable_capacity_per_replica == pytest.approx(80.0)
    assert cp.required_replicas == 2


def test_exact_division_replica_count():
    candidate = make_candidate("gpu", 1, throughput=75.0, p95=0.01)
    catalog = make_catalog(candidate)
    slo = default_slo(required_invocations_per_second=150.0, capacity_headroom_fraction=0.0)
    plan = build_sizing_plan(catalog, slo)
    assert plan.candidate_plans[0].required_replicas == 2  # not 3


# --- latency SLO ---------------------------------------------------------------------

def test_latency_pass():
    candidate = make_candidate("gpu", 1, p95=0.020)
    plan = build_sizing_plan(make_catalog(candidate), default_slo(max_p95_latency_seconds=0.025))
    assert plan.candidate_plans[0].status == STATUS_FEASIBLE


def test_latency_boundary_pass():
    candidate = make_candidate("gpu", 1, p95=0.025)
    plan = build_sizing_plan(make_catalog(candidate), default_slo(max_p95_latency_seconds=0.025))
    assert plan.candidate_plans[0].status == STATUS_FEASIBLE


def test_latency_fail():
    candidate = make_candidate("gpu", 1, p95=0.025001)
    plan = build_sizing_plan(make_catalog(candidate), default_slo(max_p95_latency_seconds=0.025))
    cp = plan.candidate_plans[0]
    assert cp.status == STATUS_SLO_VIOLATION
    assert "latency_slo_exceeded" in cp.reasons
    assert cp.required_replicas is None
    assert cp.total_hourly_cost is None


def test_missing_p95_is_insufficient_evidence_not_p50_fallback():
    candidate = make_candidate("gpu", 1, p95=None)
    plan = build_sizing_plan(make_catalog(candidate), default_slo())
    cp = plan.candidate_plans[0]
    assert cp.status == STATUS_INSUFFICIENT_EVIDENCE
    assert "missing_p95" in cp.reasons
    assert cp.required_replicas is None


# --- cost / monthly / budget ----------------------------------------------------------

def test_hourly_cost():
    candidate = make_candidate("gpu", "0.40", throughput=50.0, p95=0.01)
    slo = default_slo(required_invocations_per_second=140.0, capacity_headroom_fraction=0.0)  # -> 3 replicas
    plan = build_sizing_plan(make_catalog(candidate), slo)
    cp = plan.candidate_plans[0]
    assert cp.required_replicas == 3
    assert cp.total_hourly_cost == Decimal("1.20")


def test_monthly_cost():
    candidate = make_candidate("gpu", "0.40", throughput=50.0, p95=0.01)
    slo = default_slo(required_invocations_per_second=140.0, capacity_headroom_fraction=0.0, monthly_hours=730)
    plan = build_sizing_plan(make_catalog(candidate), slo)
    cp = plan.candidate_plans[0]
    assert cp.monthly_cost == Decimal("876.00")


def test_budget_pass():
    candidate = make_candidate("gpu", "0.40", throughput=50.0, p95=0.01)
    slo = default_slo(
        required_invocations_per_second=140.0, capacity_headroom_fraction=0.0,
        monthly_hours=730, max_monthly_cost=Decimal("900"),
    )
    plan = build_sizing_plan(make_catalog(candidate), slo)
    assert plan.candidate_plans[0].status == STATUS_FEASIBLE


def test_budget_fail():
    candidate = make_candidate("gpu", "0.40", throughput=50.0, p95=0.01)
    slo = default_slo(
        required_invocations_per_second=140.0, capacity_headroom_fraction=0.0,
        monthly_hours=730, max_monthly_cost=Decimal("800"),
    )
    plan = build_sizing_plan(make_catalog(candidate), slo)
    cp = plan.candidate_plans[0]
    assert cp.status == STATUS_SLO_VIOLATION
    assert "monthly_budget_exceeded" in cp.reasons
    # cost is still reported, not blanked out, for a budget (soft) violation
    assert cp.monthly_cost == Decimal("876.00")


def test_max_replicas_violation():
    candidate = make_candidate("gpu", 1, throughput=40.0, p95=0.01)
    slo = default_slo(required_invocations_per_second=150.0, capacity_headroom_fraction=0.0, max_replicas=3)
    plan = build_sizing_plan(make_catalog(candidate), slo)
    cp = plan.candidate_plans[0]
    assert cp.required_replicas == 4
    assert cp.status == STATUS_SLO_VIOLATION
    assert "max_replicas_exceeded" in cp.reasons


# --- memory constraint -----------------------------------------------------------------

def test_memory_constraint_pass():
    candidate = make_candidate("gpu", 1, peak_mem=1_000_000, p95=0.01)
    slo = default_slo(max_peak_memory_allocated_bytes=2_000_000)
    plan = build_sizing_plan(make_catalog(candidate), slo)
    assert plan.candidate_plans[0].status == STATUS_FEASIBLE


def test_memory_constraint_fail():
    candidate = make_candidate("gpu", 1, peak_mem=3_000_000, p95=0.01)
    slo = default_slo(max_peak_memory_allocated_bytes=2_000_000)
    plan = build_sizing_plan(make_catalog(candidate), slo)
    cp = plan.candidate_plans[0]
    assert cp.status == STATUS_SLO_VIOLATION
    assert "peak_memory_exceeded" in cp.reasons


def test_memory_constraint_missing_metric_is_insufficient_evidence():
    candidate = make_candidate("gpu", 1, peak_mem=None, p95=0.01)
    slo = default_slo(max_peak_memory_allocated_bytes=2_000_000)
    plan = build_sizing_plan(make_catalog(candidate), slo)
    cp = plan.candidate_plans[0]
    assert cp.status == STATUS_INSUFFICIENT_EVIDENCE
    assert "missing_peak_memory" in cp.reasons


# --- ranking -----------------------------------------------------------------------------

def test_ranking_total_cost_beats_unit_count():
    a = make_candidate("a", "0.90", throughput=200.0, p95=0.01)  # 1 replica, $0.90
    b = make_candidate("b", "0.35", throughput=60.0, p95=0.01)  # 2 replicas, $0.70
    slo = default_slo(required_invocations_per_second=100.0, capacity_headroom_fraction=0.0)
    plan = build_sizing_plan(make_catalog(a, b), slo)
    assert plan.recommended_candidate_id == "b"


def test_ranking_tie_break_by_replicas_then_p95_then_id():
    # same total cost, different replicas -> fewer replicas wins
    a = make_candidate("a", "0.50", throughput=100.0, p95=0.01)  # 1 replica * 0.50 = 0.50
    b = make_candidate("b", "0.25", throughput=60.0, p95=0.01)  # 2 replicas * 0.25 = 0.50
    slo = default_slo(required_invocations_per_second=100.0, capacity_headroom_fraction=0.0)
    plan = build_sizing_plan(make_catalog(a, b), slo)
    assert plan.recommended_candidate_id == "a"  # fewer replicas


def test_cheapest_unit_trap():
    a = make_candidate("a", "0.30", throughput=40.0, p95=0.01)  # 4 replicas * 0.30 = 1.20
    b = make_candidate("b", "0.80", throughput=150.0, p95=0.01)  # 1 replica * 0.80 = 0.80
    slo = default_slo(required_invocations_per_second=150.0, capacity_headroom_fraction=0.0)
    plan = build_sizing_plan(make_catalog(a, b), slo)
    assert plan.recommended_candidate_id == "b"


def test_fastest_hardware_trap():
    fast = make_candidate("fast", "2.00", p95=0.005, throughput=100.0)
    cheap = make_candidate("cheap", "0.80", p95=0.020, throughput=100.0)
    slo = default_slo(required_invocations_per_second=50.0, max_p95_latency_seconds=0.025, capacity_headroom_fraction=0.0)
    plan = build_sizing_plan(make_catalog(fast, cheap), slo)
    assert plan.recommended_candidate_id == "cheap"


def test_no_feasible_candidate():
    candidate = make_candidate("gpu", 1, p95=0.1)  # too slow
    plan = build_sizing_plan(make_catalog(candidate), default_slo(max_p95_latency_seconds=0.025))
    assert plan.recommended_candidate_id is None
    assert plan.feasible_candidate_count == 0


# --- identity checks ---------------------------------------------------------------------

def test_workload_mismatch_is_plan_error():
    a = make_candidate("a", 1, preset="gemm_large_square", kind="gemm")
    b = make_candidate("b", 1, preset="conv_spatial", kind="conv2d")
    plan = build_sizing_plan(make_catalog(a, b), default_slo())
    assert plan.status == PLAN_STATUS_ERROR
    assert plan.candidate_plans == ()


def test_dtype_mismatch_is_plan_error():
    a = make_candidate("a", 1, dtype="fp16")
    b = make_candidate("b", 1, dtype="fp32")
    plan = build_sizing_plan(make_catalog(a, b), default_slo())
    assert plan.status == PLAN_STATUS_ERROR


def test_backend_mismatch_is_plan_error():
    a = make_candidate("a", 1, backend="pytorch")
    b = make_candidate("b", 1, backend="onnxruntime")
    plan = build_sizing_plan(make_catalog(a, b), default_slo())
    assert plan.status == PLAN_STATUS_ERROR


def test_different_devices_allowed():
    a = make_candidate("a", 1, device="cuda:0", p95=0.01)
    b = make_candidate("b", 1, device="cpu", p95=0.01)
    plan = build_sizing_plan(make_catalog(a, b), default_slo())
    assert plan.status == PLAN_STATUS_OK


# --- telemetry/validation context are diagnostic only -------------------------------------

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


def test_telemetry_does_not_change_recommendation():
    a = make_candidate("a", "0.50", throughput=100.0, p95=0.01)
    b = DeploymentCandidate("b", make_bench(p95=0.01, throughput=100.0), Decimal("0.80"), "USD")
    slo = default_slo(required_invocations_per_second=50.0, capacity_headroom_fraction=0.0)
    plan_without = build_sizing_plan(make_catalog(a, b), slo)

    b_with_bad_telemetry = replace(b, telemetry_summary=make_telemetry_summary(gpu_mean=1.0))
    plan_with = build_sizing_plan(make_catalog(a, b_with_bad_telemetry), slo)

    assert plan_without.recommended_candidate_id == plan_with.recommended_candidate_id == "a"


def test_validation_does_not_change_recommendation():
    from tensorforge_ops.calibration import ValidationResult

    a = make_candidate("a", "0.50", throughput=100.0, p95=0.01)
    b = DeploymentCandidate("b", make_bench(p95=0.01, throughput=100.0), Decimal("0.80"), "USD")
    slo = default_slo(required_invocations_per_second=50.0, capacity_headroom_fraction=0.0)
    plan_without = build_sizing_plan(make_catalog(a, b), slo)

    bad_validation = ValidationResult(
        core_result_fingerprint="sha256:" + "0" * 64, calibration_fingerprint="sha256:" + "1" * 64,
        workload_preset="gemm_large_square", workload_kind="gemm", backend="pytorch", device="cuda:0", dtype="fp16",
        predicted_compute_seconds=0.001, predicted_memory_seconds=0.0005, predicted_latency_seconds=0.001,
        predicted_bottleneck="compute-bound", measured_p50_latency_seconds=0.01, measured_mean_latency_seconds=0.01,
        measured_p95_latency_seconds=0.01, signed_error_seconds=-0.009, absolute_error_seconds=0.009,
        relative_error=-0.9, absolute_percentage_error=90.0, measured_to_predicted_ratio=10.0,
    )
    b_with_validation = replace(b, validation_result=bad_validation)
    plan_with = build_sizing_plan(make_catalog(a, b_with_validation), slo)

    assert plan_without.recommended_candidate_id == plan_with.recommended_candidate_id == "a"


# --- demand scenarios -----------------------------------------------------------------------

def test_demand_scenario_changes_recommendation():
    # A: cheap per unit but low throughput -- wins at low demand
    # B: pricier per unit but high throughput -- wins at high demand once A needs many replicas
    a = make_candidate("a", "0.20", throughput=20.0, p95=0.01)
    b = make_candidate("b", "3.00", throughput=500.0, p95=0.01)
    base_slo = default_slo(capacity_headroom_fraction=0.0)

    low_demand = build_sizing_plan(make_catalog(a, b), replace(base_slo, required_invocations_per_second=15.0))
    high_demand = build_sizing_plan(make_catalog(a, b), replace(base_slo, required_invocations_per_second=500.0))

    assert low_demand.recommended_candidate_id == "a"
    assert high_demand.recommended_candidate_id == "b"


def test_build_sizing_plans_for_scenarios():
    a = make_candidate("a", "0.20", throughput=20.0, p95=0.01)
    b = make_candidate("b", "3.00", throughput=500.0, p95=0.01)
    catalog = make_catalog(a, b)
    base_slo = default_slo(required_invocations_per_second=15.0, capacity_headroom_fraction=0.0)
    scenarios = (DemandScenario("1x", 15.0), DemandScenario("5x", 75.0), DemandScenario("30x", 500.0))
    results = build_sizing_plans_for_scenarios(catalog, base_slo, scenarios)
    assert set(results) == {"1x", "5x", "30x"}
    assert results["1x"].recommended_candidate_id == "a"
    assert results["30x"].recommended_candidate_id == "b"


def test_demand_scenario_rejects_nonpositive_rate():
    with pytest.raises(ValueError):
        DemandScenario("bad", 0)


# --- headroom sensitivity ---------------------------------------------------------------------

def test_headroom_changes_replica_count():
    candidate = make_candidate("gpu", 1, throughput=100.0, p95=0.01)
    slo_no_headroom = default_slo(required_invocations_per_second=95.0, capacity_headroom_fraction=0.0)
    slo_with_headroom = default_slo(required_invocations_per_second=95.0, capacity_headroom_fraction=0.20)

    plan_no_headroom = build_sizing_plan(make_catalog(candidate), slo_no_headroom)
    plan_with_headroom = build_sizing_plan(make_catalog(candidate), slo_with_headroom)

    assert plan_no_headroom.candidate_plans[0].required_replicas == 1
    assert plan_with_headroom.candidate_plans[0].required_replicas == 2


# --- pricing provenance ------------------------------------------------------------------------

def test_pricing_provenance_survives_into_plan():
    candidate = DeploymentCandidate(
        "gpu", make_bench(p95=0.01), Decimal("1"), "USD",
        provider="example-provider", region="example-region", hardware_label="Example GPU",
        price_source="manual", price_as_of="2026-08-26",
    )
    plan = build_sizing_plan(make_catalog(candidate), default_slo())
    cp = plan.candidate_plans[0]
    assert cp.provider == "example-provider"
    assert cp.region == "example-region"
    assert cp.hardware_label == "Example GPU"
    assert cp.price_source == "manual"
    assert cp.price_as_of == "2026-08-26"


# --- report ----------------------------------------------------------------------------------

def test_markdown_report_contains_key_content():
    a = make_candidate("gpu-a", "0.45", throughput=90.0, p95=0.015)
    b = make_candidate("gpu-b", "0.80", throughput=160.0, p95=0.020)
    slo = default_slo(required_invocations_per_second=120.0, max_p95_latency_seconds=0.025, capacity_headroom_fraction=0.20)
    plan = build_sizing_plan(make_catalog(a, b), slo)
    report = render_sizing_markdown_report(plan)
    assert "gpu-a" in report
    assert "gpu-b" in report
    assert "Recommendation" in report
    assert "Assumptions" in report
    assert "steady-state benchmark capacity proxy" in report


def test_markdown_report_no_feasible_candidate():
    candidate = make_candidate("gpu", 1, p95=0.1)
    plan = build_sizing_plan(make_catalog(candidate), default_slo(max_p95_latency_seconds=0.025))
    report = render_sizing_markdown_report(plan)
    assert "NO FEASIBLE CANDIDATE" in report


# --- serialization -----------------------------------------------------------------------------

def test_sizing_plan_result_json_roundtrip_structure():
    candidate = make_candidate("gpu", "0.50", p95=0.01)
    plan = build_sizing_plan(make_catalog(candidate), default_slo())
    d = json.loads(plan.to_json())
    assert d["status"] == "OK"
    assert d["sizing_result_schema_version"] == 1
    assert d["candidate_plans"][0]["candidate_id"] == "gpu"
    assert isinstance(d["assumptions"], list) and len(d["assumptions"]) > 0


# --- catalog / SLO file loading (relative paths, schema versions) -------------------------------

def _write_benchmark_json(path, **bench_kwargs):
    path.write_text(make_bench(**bench_kwargs).to_json())


def test_load_deployment_catalog_resolves_relative_benchmark_paths(tmp_path):
    (tmp_path / "results").mkdir()
    _write_benchmark_json(tmp_path / "results" / "gpu-a.json", p95=0.015, throughput=90.0)

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({
        "deployment_catalog_schema_version": 1,
        "currency": "USD",
        "candidates": [
            {"id": "gpu-a", "benchmark_result": "results/gpu-a.json", "hourly_cost": 0.45},
        ],
    }))

    catalog = load_deployment_catalog(str(catalog_path))
    assert catalog.currency == "USD"
    assert catalog.candidates[0].candidate_id == "gpu-a"
    assert catalog.candidates[0].benchmark_result.statistics.p95_seconds == pytest.approx(0.015)
    assert catalog.candidates[0].hourly_cost == Decimal("0.45")


def test_load_deployment_catalog_rejects_bad_schema_version(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"deployment_catalog_schema_version": 99, "currency": "USD", "candidates": []}))
    with pytest.raises(ValueError, match="deployment_catalog_schema_version"):
        load_deployment_catalog(str(catalog_path))


def test_load_slo_policy_from_file(tmp_path):
    slo_path = tmp_path / "slo.json"
    slo_path.write_text(json.dumps({
        "slo_policy_schema_version": 1,
        "required_invocations_per_second": 120.0,
        "max_p95_latency_seconds": 0.025,
        "capacity_headroom_fraction": 0.2,
    }))
    slo = load_slo_policy(str(slo_path))
    assert slo.required_invocations_per_second == 120.0
    assert slo.capacity_headroom_fraction == 0.2
