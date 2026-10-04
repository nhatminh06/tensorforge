import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_capture_module():
    spec = importlib.util.spec_from_file_location("tensorforge_demo_capture", ROOT / "tools/demo/capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_measurement_bundle(bundle):
    bundle.mkdir()
    manifest = {
        "schema_version": 2, "tensorforge_commit": "a" * 40,
        "device": {"name": "Test GPU", "dtype": "fp16"},
        "runtime": {"pytorch_version": "test", "cuda_runtime_version": "test"},
        "measurement": {"warmup_iterations": 10, "measured_iterations": 50},
        "calibration": {"effective_compute_tflops": 1.0, "effective_memory_gbps": 2.0},
        "primary_workload": "gemm_large_square", "contrast_workload": "gemm_tiny",
        "validation": {"predicted_seconds": 0.001, "measured_p50_seconds": 0.002, "measured_p95_seconds": 0.003, "ape_percent": 50.0},
        "regression": {"status": "not_evaluated", "reason": "No defensible baseline."},
        "telemetry": {"status": "captured", "sample_count": 3},
        "right_sizing": {"status": "omitted", "reason": "Only one device."},
        "impact": {"status": "not_evaluated", "reason": "Regression required."},
    }
    benchmark = {"benchmark_schema_version": 1}
    validation = {
        "validation_schema_version": 1,
        "prediction": {"latency_seconds": 0.000001, "bottleneck": "compute-bound"},
        "measurement": {"p50_latency_seconds": 0.00001},
        "error": {"absolute_percentage_error": 90.0},
    }
    stat = {"sample_count": 0, "mean": None, "min": None, "max": None}
    telemetry = {
        "telemetry_schema_version": 1,
        "sm_activity_percent": stat, "sm_occupancy_percent": stat,
        "tensor_activity_percent": stat, "dram_bandwidth_utilization_percent": stat,
    }
    write_json(bundle / "manifest.json", manifest)
    write_json(bundle / "calibration.json", {"calibration_schema_version": 1})
    write_json(bundle / "primary-core.json", {"schema_version": 1})
    write_json(bundle / "contrast-core.json", {"schema_version": 1})
    write_json(bundle / "primary-benchmark.json", benchmark)
    write_json(bundle / "candidate-benchmark.json", benchmark)
    write_json(bundle / "contrast-benchmark.json", benchmark)
    write_json(bundle / "primary-validation.json", validation)
    write_json(bundle / "contrast-validation.json", validation)
    write_json(bundle / "telemetry-summary.json", telemetry)
    write_json(bundle / "telemetry-trace.json", {"telemetry_schema_version": 1})
    write_json(bundle / "telemetry-capabilities.json", {"capabilities": {}})
    (bundle / "README.md").write_text("# Evidence\n", encoding="utf-8")
    return manifest


def run_tool(name, bundle):
    return subprocess.run(
        [sys.executable, str(ROOT / f"tools/demo/{name}.py"), str(bundle)],
        cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


def test_no_baseline_selects_measurement_only_mode():
    capture = load_capture_module()
    assert capture.baseline_mode(None, None) == "measurement_only"


@pytest.mark.parametrize("benchmark,telemetry", [(pathlib.Path("baseline.json"), None), (None, pathlib.Path("telemetry.json"))])
def test_exactly_one_baseline_artifact_is_rejected(benchmark, telemetry):
    capture = load_capture_module()
    with pytest.raises(SystemExit, match="must either both be provided or both be omitted"):
        capture.baseline_mode(benchmark, telemetry)


def test_both_baseline_artifacts_preserve_regression_mode():
    capture = load_capture_module()
    assert capture.baseline_mode(pathlib.Path("baseline.json"), pathlib.Path("telemetry.json")) == "regression"


def test_summarize_handles_not_evaluated(tmp_path):
    bundle = tmp_path / "evidence"
    make_measurement_bundle(bundle)
    result = run_tool("summarize", bundle)
    assert result.returncode == 0, result.stderr
    assert "REGRESSION\nNOT EVALUATED" in result.stdout
    assert "RIGHT-SIZING\nOMITTED" in result.stdout
    assert "IMPACT\nNOT EVALUATED" in result.stdout


def test_validator_accepts_measurement_only_bundle(tmp_path):
    bundle = tmp_path / "evidence"
    make_measurement_bundle(bundle)
    result = run_tool("validate", bundle)
    assert result.returncode == 0, result.stderr
    assert (bundle / "SHA256SUMS").is_file()
    second = run_tool("validate", bundle)
    assert second.returncode == 0, second.stderr


def test_validator_rejects_inconsistent_measurement_only_artifacts(tmp_path):
    bundle = tmp_path / "evidence"
    make_measurement_bundle(bundle)
    write_json(bundle / "regression-result.json", {"regression_result_schema_version": 1})
    result = run_tool("validate", bundle)
    assert result.returncode != 0
    assert "contains regression/impact artifacts" in result.stderr
