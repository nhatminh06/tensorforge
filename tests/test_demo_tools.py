import json
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_capture_rejects_missing_real_baseline_before_creating_output(tmp_path):
    output = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "tools/demo/capture.py"), str(output),
            "--baseline-benchmark", str(tmp_path / "missing-benchmark.json"),
            "--baseline-telemetry", str(tmp_path / "missing-telemetry.json"),
        ],
        cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode != 0
    assert "baseline benchmark and telemetry must both be existing files" in result.stderr
    assert not output.exists()


def test_summarize_uses_manifest_and_contrast_artifact(tmp_path):
    manifest = {
        "device": {"name": "Test GPU", "dtype": "fp16"},
        "runtime": {"pytorch_version": "test", "cuda_runtime_version": "test"},
        "calibration": {"effective_compute_tflops": 1.0, "effective_memory_gbps": 2.0},
        "primary_workload": "gemm_large_square", "contrast_workload": "gemm_tiny",
        "validation": {"predicted_seconds": 0.001, "measured_p50_seconds": 0.002, "measured_p95_seconds": 0.003, "ape_percent": 50.0},
        "regression": {"status": "PASS"}, "telemetry": {"status": "captured", "sample_count": 3},
        "impact": {"status": "PERFORMANCE_READY"},
    }
    contrast = {
        "prediction": {"latency_seconds": 0.000001},
        "measurement": {"p50_latency_seconds": 0.00001},
        "error": {"absolute_percentage_error": 90.0},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "contrast-validation.json").write_text(json.dumps(contrast), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/demo/summarize.py"), str(tmp_path)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    assert "Test GPU" in result.stdout
    assert "gemm_large_square" in result.stdout
    assert "PERFORMANCE_READY" in result.stdout
