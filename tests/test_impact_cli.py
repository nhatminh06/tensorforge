"""CLI-level exit-code contract tests for `python -m tensorforge_ops impact`.

Requires the "ops" extra (mlflow) because tensorforge_ops.cli imports
tracking.py unconditionally at module load time, even though `impact`
itself never talks to MLflow unless --track is passed.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("mlflow")

from tensorforge_ops.benchmark import BenchmarkResult, compute_latency_statistics
from tensorforge_ops.regression import MetricPolicy, RegressionPolicy, compare_benchmark_results


def _bench(p50, throughput=None, preset="gemm_tiny", kind="gemm"):
    samples = (p50,) * 5
    stats = compute_latency_statistics(samples)
    from dataclasses import replace

    stats = replace(stats, p50_seconds=p50, mean_seconds=p50, throughput_per_second=throughput or (1.0 / p50))
    return BenchmarkResult(
        core_result_fingerprint="sha256:" + "0" * 64, workload_preset=preset, workload_kind=kind,
        backend="pytorch", device="cpu", dtype="fp16", warmup_iterations=1, measured_iterations=5,
        latency_samples_seconds=samples, statistics=stats, peak_memory_allocated_bytes=None,
        runtime_metadata={"torch_version": "2.13.0"},
    )


def _write_regression_result(path, baseline_p50=0.100, candidate_p50=0.103, relative=0.10, preset="gemm_tiny"):
    policy = RegressionPolicy(metrics={"p50_latency_seconds": MetricPolicy(max_relative_regression=relative)})
    result = compare_benchmark_results(_bench(baseline_p50, preset=preset), _bench(candidate_p50, preset=preset), policy)
    path.write_text(result.to_json())
    return result


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "tensorforge_ops", "impact", *args],
        capture_output=True, text=True,
    )


def test_cli_ready_returns_zero(tmp_path):
    regression_path = tmp_path / "regression-result.json"
    _write_regression_result(regression_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "impact_policy_schema_version": 1, "require_regression_pass": True,
        "require_feasible_candidate_deployment": False,
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"impact_manifest_schema_version": 1, "regression_result": "regression-result.json"}))

    proc = _run_cli("--manifest", str(manifest_path), "--policy", str(policy_path))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PERFORMANCE_READY" in proc.stdout


def test_cli_blocked_returns_two(tmp_path):
    regression_path = tmp_path / "regression-result.json"
    _write_regression_result(regression_path, baseline_p50=0.100, candidate_p50=0.130, relative=0.10)  # FAIL
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "impact_policy_schema_version": 1, "require_regression_pass": True,
        "require_feasible_candidate_deployment": False,
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"impact_manifest_schema_version": 1, "regression_result": "regression-result.json"}))

    proc = _run_cli("--manifest", str(manifest_path), "--policy", str(policy_path))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "PERFORMANCE_BLOCKED" in proc.stdout


def test_cli_review_required_returns_one(tmp_path):
    regression_path = tmp_path / "regression-result.json"
    policy = RegressionPolicy(metrics={"p50_latency_seconds": MetricPolicy(max_relative_regression=0.1)})
    error_result = compare_benchmark_results(
        _bench(0.1, preset="gemm_tiny", kind="gemm"), _bench(0.1, preset="conv_spatial", kind="conv2d"), policy,
    )
    regression_path.write_text(error_result.to_json())
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "impact_policy_schema_version": 1, "require_regression_pass": True,
        "require_feasible_candidate_deployment": False,
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"impact_manifest_schema_version": 1, "regression_result": "regression-result.json"}))

    proc = _run_cli("--manifest", str(manifest_path), "--policy", str(policy_path))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "REVIEW_REQUIRED" in proc.stdout


def test_cli_missing_manifest_file_returns_one(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"impact_policy_schema_version": 1}))
    proc = _run_cli("--manifest", str(tmp_path / "does-not-exist.json"), "--policy", str(policy_path))
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_cli_output_overwrite_protection_returns_one_not_two(tmp_path):
    regression_path = tmp_path / "regression-result.json"
    _write_regression_result(regression_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "impact_policy_schema_version": 1, "require_regression_pass": True,
        "require_feasible_candidate_deployment": False,
    }))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"impact_manifest_schema_version": 1, "regression_result": "regression-result.json"}))
    output_json = tmp_path / "impact-result.json"

    proc1 = _run_cli("--manifest", str(manifest_path), "--policy", str(policy_path), "--output-json", str(output_json))
    assert proc1.returncode == 0

    proc2 = _run_cli("--manifest", str(manifest_path), "--policy", str(policy_path), "--output-json", str(output_json))
    assert proc2.returncode == 1  # operational error, distinct from BLOCKED (2)
    assert "already exists" in proc2.stderr
