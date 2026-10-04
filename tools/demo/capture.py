#!/usr/bin/env python3
"""Capture one honest TensorForge Core + Ops evidence bundle.

This is orchestration only: all modeling, measurement, calibration,
validation, regression, telemetry, and impact decisions use existing APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TILES = (32, 64, 128)
PRIMARY = "gemm_large_square"
CONTRAST = "gemm_tiny"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"capture: {message}")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def commit_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def core_result(preset_name: str):
    from tensorforge.experiments import ExperimentSpec, run_experiment

    return run_experiment(
        ExperimentSpec(
            name=f"{preset_name}-canonical-evidence",
            workload_preset=preset_name,
            accelerator_preset="balanced",
            tile_m_values=TILES,
            tile_n_values=TILES,
            tile_k_values=TILES,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new evidence directory")
    parser.add_argument(
        "--baseline-benchmark", type=Path, required=True,
        help="real gemm_large_square CUDA BenchmarkResult from the baseline revision",
    )
    parser.add_argument(
        "--baseline-telemetry", type=Path, required=True,
        help="matching baseline TelemetrySummary from a separate telemetry phase",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--telemetry-duration", type=float, default=5.0)
    parser.add_argument("--telemetry-sample-interval", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        fail(f"output already exists: {args.output}")
    if not args.baseline_benchmark.is_file() or not args.baseline_telemetry.is_file():
        fail("baseline benchmark and telemetry must both be existing files")
    if not args.device.startswith("cuda"):
        fail("canonical evidence requires CUDA; CPU fallback is intentionally disabled")

    try:
        import torch
        import pynvml  # noqa: F401 - prerequisite check before creating output
    except ImportError as exc:
        fail(f"missing canonical GPU prerequisite: {exc}")
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False; canonical capture was not converted to CPU")

    from tensorforge.presets import get_workload_preset
    from tensorforge_ops.benchmark import BenchmarkConfig, load_benchmark_result
    from tensorforge_ops.benchmark_pytorch import run_pytorch_benchmark, run_pytorch_telemetry_window
    from tensorforge_ops.calibration import predict, validate_prediction
    from tensorforge_ops.calibration_pytorch import run_pytorch_calibration
    from tensorforge_ops.calibration import CalibrationConfig
    from tensorforge_ops.impact import ImpactPolicy, build_impact_result, load_impact_manifest, render_impact_markdown_report
    from tensorforge_ops.regression import (
        compare_benchmark_results, load_regression_policy, render_markdown_report,
    )
    from tensorforge_ops.telemetry import TelemetryConfig, load_telemetry_summary, summarize_telemetry_trace
    from tensorforge_ops.tracking import compute_result_fingerprint

    baseline = load_benchmark_result(str(args.baseline_benchmark))
    baseline_telemetry = load_telemetry_summary(str(args.baseline_telemetry))
    if baseline.workload_preset != PRIMARY:
        fail(f"baseline workload must be {PRIMARY!r}, got {baseline.workload_preset!r}")
    if not baseline.device.startswith("cuda"):
        fail(f"baseline must be CUDA evidence, got {baseline.device!r}")

    out = args.output.resolve()
    out.mkdir(parents=True)
    try:
        sha = commit_sha()
        primary_core = core_result(PRIMARY)
        contrast_core = core_result(CONTRAST)
        primary_core.save(str(out / "primary-core.json"))
        contrast_core.save(str(out / "contrast-core.json"))

        calibration = run_pytorch_calibration(CalibrationConfig(
            backend="pytorch", device=args.device, dtype="fp16",
            compute_m=2048, compute_n=2048, compute_k=2048,
            memory_probe_mib=64, warmup_iterations=args.warmup,
            measured_iterations=args.iterations,
        ))
        (out / "calibration.json").write_text(calibration.to_json() + "\n", encoding="utf-8")

        bconfig = BenchmarkConfig("pytorch", args.device, args.warmup, args.iterations)
        primary_fp = compute_result_fingerprint(primary_core)
        contrast_fp = compute_result_fingerprint(contrast_core)
        candidate = run_pytorch_benchmark(get_workload_preset(PRIMARY), primary_fp, bconfig)
        contrast = run_pytorch_benchmark(get_workload_preset(CONTRAST), contrast_fp, bconfig)
        (out / "baseline-benchmark.json").write_text(baseline.to_json() + "\n", encoding="utf-8")
        (out / "candidate-benchmark.json").write_text(candidate.to_json() + "\n", encoding="utf-8")
        (out / "primary-benchmark.json").write_text(candidate.to_json() + "\n", encoding="utf-8")
        (out / "contrast-benchmark.json").write_text(contrast.to_json() + "\n", encoding="utf-8")

        primary_validation = validate_prediction(
            calibration, predict(get_workload_preset(PRIMARY), calibration), primary_fp, candidate
        )
        contrast_validation = validate_prediction(
            calibration, predict(get_workload_preset(CONTRAST), calibration), contrast_fp, contrast
        )
        (out / "primary-validation.json").write_text(primary_validation.to_json() + "\n", encoding="utf-8")
        (out / "contrast-validation.json").write_text(contrast_validation.to_json() + "\n", encoding="utf-8")

        telemetry_config = TelemetryConfig(
            backend="nvml", device_index=int(args.device.split(":", 1)[1]) if ":" in args.device else 0,
            sample_interval_seconds=args.telemetry_sample_interval,
            telemetry_duration_seconds=args.telemetry_duration,
        )
        trace = run_pytorch_telemetry_window(
            get_workload_preset(PRIMARY), telemetry_config, core_result_fingerprint=primary_fp
        )
        telemetry = summarize_telemetry_trace(trace)
        (out / "telemetry-summary.json").write_text(telemetry.to_json() + "\n", encoding="utf-8")
        (out / "baseline-telemetry-summary.json").write_text(baseline_telemetry.to_json() + "\n", encoding="utf-8")

        policy_path = ROOT / "perf" / "gpu-regression-policy.json"
        policy = load_regression_policy(str(policy_path))
        shutil.copyfile(policy_path, out / "regression-policy.json")
        regression = compare_benchmark_results(baseline, candidate, policy)
        (out / "regression-result.json").write_text(regression.to_json() + "\n", encoding="utf-8")
        (out / "regression-report.md").write_text(render_markdown_report(regression), encoding="utf-8")

        impact_manifest = {
            "impact_manifest_schema_version": 1,
            "regression_result": "regression-result.json",
            "baseline": {"benchmark_result": "baseline-benchmark.json", "telemetry_summary": "baseline-telemetry-summary.json"},
            "candidate": {"core_result": "primary-core.json", "benchmark_result": "candidate-benchmark.json", "validation_result": "primary-validation.json", "telemetry_summary": "telemetry-summary.json"},
        }
        impact_policy = {
            "impact_policy_schema_version": 1,
            "require_regression_pass": True,
            "require_feasible_candidate_deployment": False,
            "max_hourly_cost_increase_fraction": None,
            "max_monthly_cost_increase_fraction": None,
            "max_replica_increase": None,
        }
        write_json(out / "impact-manifest.json", impact_manifest)
        write_json(out / "impact-policy.json", impact_policy)
        loaded_manifest = load_impact_manifest(str(out / "impact-manifest.json"))
        impact = build_impact_result(loaded_manifest, ImpactPolicy(**{k: v for k, v in impact_policy.items() if k != "impact_policy_schema_version"}))
        (out / "impact-result.json").write_text(impact.to_json() + "\n", encoding="utf-8")
        (out / "impact-report.md").write_text(render_impact_markdown_report(impact), encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "tensorforge_commit": sha,
            "device": {"backend": "pytorch", "device": args.device, "name": calibration.device_name, "dtype": calibration.dtype},
            "runtime": {"pytorch_version": calibration.runtime_metadata.get("torch_version"), "cuda_runtime_version": calibration.runtime_metadata.get("cuda_runtime_version")},
            "primary_workload": PRIMARY,
            "contrast_workload": CONTRAST,
            "measurement": {"warmup_iterations": args.warmup, "measured_iterations": args.iterations},
            "calibration": {
                "compute_probe_shape": [2048, 2048, 2048], "memory_probe_mib": 64,
                "effective_compute_tflops": calibration.effective_compute_flops_per_second / 1e12,
                "effective_memory_gbps": calibration.effective_memory_bandwidth_bytes_per_second / 1e9,
            },
            "validation": {
                "predicted_seconds": primary_validation.predicted_latency_seconds,
                "measured_p50_seconds": primary_validation.measured_p50_latency_seconds,
                "measured_p95_seconds": primary_validation.measured_p95_latency_seconds,
                "ape_percent": primary_validation.absolute_percentage_error,
                "predicted_bottleneck": primary_validation.predicted_bottleneck,
            },
            "regression": {"status": regression.status},
            "telemetry": {"status": "captured", "sample_count": telemetry.sample_count},
            "right_sizing": {"status": "omitted", "reason": "only one physical device was measured"},
            "impact": {"status": impact.readiness},
        }
        write_json(out / "manifest.json", manifest)
        subprocess.run([sys.executable, str(ROOT / "tools/demo/validate.py"), str(out)], check=True)
        subprocess.run([sys.executable, str(ROOT / "tools/demo/summarize.py"), str(out)], check=True)
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
