"""CLI-level exit-code contract tests for `python -m tensorforge_ops regression`.

Requires the "ops" extra (mlflow) because tensorforge_ops.cli imports
tracking.py unconditionally at module load time, even though the
`regression` command itself never talks to MLflow.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("mlflow")

from tensorforge_ops.benchmark import BenchmarkResult, compute_latency_statistics


def _benchmark_json(p50, preset="gemm_tiny", kind="gemm", device="cpu", dtype="fp16"):
    samples = (p50,) * 5
    stats = compute_latency_statistics(samples)
    from dataclasses import replace

    stats = replace(stats, p50_seconds=p50, mean_seconds=p50)
    result = BenchmarkResult(
        core_result_fingerprint="sha256:" + "0" * 64,
        workload_preset=preset, workload_kind=kind, backend="pytorch", device=device, dtype=dtype,
        warmup_iterations=1, measured_iterations=5, latency_samples_seconds=samples,
        statistics=stats, peak_memory_allocated_bytes=None, runtime_metadata={"torch_version": "2.13.0+cu126"},
    )
    return result.to_json()


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "tensorforge_ops", "regression", *args],
        capture_output=True, text=True,
    )


def test_cli_pass_returns_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    policy = tmp_path / "policy.json"
    baseline.write_text(_benchmark_json(0.100))
    candidate.write_text(_benchmark_json(0.103))
    policy.write_text(json.dumps({
        "regression_policy_schema_version": 1,
        "metrics": {"p50_latency_seconds": {"max_relative_regression": 0.10}},
    }))

    proc = _run_cli("--baseline", str(baseline), "--candidate", str(candidate), "--policy", str(policy))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASS" in proc.stdout


def test_cli_fail_returns_two(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    policy = tmp_path / "policy.json"
    baseline.write_text(_benchmark_json(0.100))
    candidate.write_text(_benchmark_json(0.130))
    policy.write_text(json.dumps({
        "regression_policy_schema_version": 1,
        "metrics": {"p50_latency_seconds": {"max_relative_regression": 0.10}},
    }))

    proc = _run_cli("--baseline", str(baseline), "--candidate", str(candidate), "--policy", str(policy))
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout


def test_cli_error_returns_one_for_missing_file(tmp_path):
    baseline = tmp_path / "baseline.json"
    policy = tmp_path / "policy.json"
    baseline.write_text(_benchmark_json(0.100))
    policy.write_text(json.dumps({
        "regression_policy_schema_version": 1,
        "metrics": {"p50_latency_seconds": {"max_relative_regression": 0.10}},
    }))

    proc = _run_cli(
        "--baseline", str(baseline), "--candidate", str(tmp_path / "does-not-exist.json"), "--policy", str(policy),
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_cli_error_returns_one_for_workload_mismatch(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    policy = tmp_path / "policy.json"
    baseline.write_text(_benchmark_json(0.100, preset="gemm_tiny", kind="gemm"))
    candidate.write_text(_benchmark_json(0.100, preset="conv_spatial", kind="conv2d"))
    policy.write_text(json.dumps({
        "regression_policy_schema_version": 1,
        "metrics": {"p50_latency_seconds": {"max_relative_regression": 0.10}},
    }))

    proc = _run_cli("--baseline", str(baseline), "--candidate", str(candidate), "--policy", str(policy))
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "ERROR" in proc.stdout


def test_cli_writes_output_files_with_overwrite_protection(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    policy = tmp_path / "policy.json"
    output_json = tmp_path / "regression-result.json"
    output_md = tmp_path / "regression-report.md"
    baseline.write_text(_benchmark_json(0.100))
    candidate.write_text(_benchmark_json(0.103))
    policy.write_text(json.dumps({
        "regression_policy_schema_version": 1,
        "metrics": {"p50_latency_seconds": {"max_relative_regression": 0.10}},
    }))

    proc = _run_cli(
        "--baseline", str(baseline), "--candidate", str(candidate), "--policy", str(policy),
        "--output-json", str(output_json), "--output-markdown", str(output_md),
    )
    assert proc.returncode == 0
    assert output_json.exists()
    assert output_md.exists()

    proc_again = _run_cli(
        "--baseline", str(baseline), "--candidate", str(candidate), "--policy", str(policy),
        "--output-json", str(output_json),
    )
    assert proc_again.returncode == 1  # operational error, distinct from regression FAIL (2)
    assert "already exists" in proc_again.stderr
