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
from typing import NoReturn


ROOT = Path(__file__).resolve().parents[2]
TILES = (32, 64, 128)
PRIMARY = "gemm_large_square"
CONTRAST = "gemm_tiny"


NO_BASELINE_REASON = "No defensible performance-changing historical baseline was available."
NO_SIZING_REASON = "Only one physical device was measured."
NO_IMPACT_REASON = "A measured regression result is required."


def fail(message: str) -> NoReturn:
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
        "--baseline-benchmark", type=Path,
        help="real gemm_large_square CUDA BenchmarkResult from the baseline revision",
    )
    parser.add_argument(
        "--baseline-telemetry", type=Path,
        help="matching baseline TelemetrySummary from a separate telemetry phase",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--telemetry-duration", type=float, default=5.0)
    parser.add_argument("--telemetry-sample-interval", type=float, default=0.15)
    return parser.parse_args()


def baseline_mode(baseline_benchmark: Path | None, baseline_telemetry: Path | None) -> str:
    if (baseline_benchmark is None) != (baseline_telemetry is None):
        fail("baseline benchmark and baseline telemetry must either both be provided or both be omitted")
    return "regression" if baseline_benchmark is not None else "measurement_only"


def render_evidence_readme(manifest: dict, primary: dict, contrast: dict) -> str:
    device = manifest["device"]
    runtime = manifest["runtime"]
    calibration = manifest["calibration"]
    regression = manifest["regression"]
    sizing = manifest["right_sizing"]
    impact = manifest["impact"]
    telemetry = manifest["telemetry"]
    p = primary
    c = contrast
    return f"""# Canonical TensorForge hardware evidence

This bundle is the canonical portfolio capture from TensorForge commit
`{manifest['tensorforge_commit']}`. It is distinct from the historical RTX 3050
validation study in `docs/ops/validation.md`; values may naturally differ.

## Device

Real `{device['name']}` using the PyTorch CUDA backend and `{device['dtype']}`.

## Runtime

- PyTorch: `{runtime['pytorch_version']}`
- CUDA runtime: `{runtime['cuda_runtime_version']}`
- warmups: {manifest['measurement']['warmup_iterations']}
- measured iterations: {manifest['measurement']['measured_iterations']}

## Calibration

Measured empirical ceilings from a 2048 x 2048 x 2048 compute probe and a
64 MiB memory-copy probe:

- effective compute rate: {calibration['effective_compute_tflops']:.6f} TFLOP/s
- effective memory-copy bandwidth: {calibration['effective_memory_gbps']:.6f} GB/s

These are probe-derived empirical ceilings, not vendor specifications or
theoretical Tensor Core peak values.

## Primary workload

`{manifest['primary_workload']}`:

- predicted latency: {p['prediction']['latency_seconds']:.12g} seconds
- measured p50: {p['measurement']['p50_latency_seconds']:.12g} seconds
- measured p95: {p['measurement']['p95_latency_seconds']:.12g} seconds
- APE: {p['error']['absolute_percentage_error']:.6f}%
- measured/predicted: {p['error']['measured_to_predicted_ratio']:.6f}x
- predicted bottleneck: `{p['prediction']['bottleneck']}`

The bottleneck label is analytical, not a measured-bottleneck claim.

## Contrast workload

`{manifest['contrast_workload']}`:

- predicted latency: {c['prediction']['latency_seconds']:.12g} seconds
- measured p50: {c['measurement']['p50_latency_seconds']:.12g} seconds
- measured p95: {c['measurement']['p95_latency_seconds']:.12g} seconds
- APE: {c['error']['absolute_percentage_error']:.6f}%
- measured/predicted: {c['error']['measured_to_predicted_ratio']:.6f}x

Any larger small-workload error is consistent with fixed framework/kernel-launch
overhead absent from the analytical lower-bound model; this capture does not
establish that explanation as a cause.

## Telemetry

Real NVML evidence was captured in a separate phase ({telemetry['sample_count']}
samples). Telemetry is diagnostic context and does not gate a verdict.
Unsupported metrics remain unavailable rather than being converted to zero.

- supported fields: {', '.join(telemetry['supported_fields'])}
- unavailable fields: {', '.join(telemetry['unavailable_fields']) or 'none'}

## Regression

**{regression['status'].replace('_', ' ').upper()}**

{regression.get('reason', 'See regression-result.json for measured policy evidence.')}

## Right-sizing

**{sizing['status'].replace('_', ' ').upper()}**

{sizing['reason']}

## Impact

**{impact['status'].replace('_', ' ').upper()}**

{impact.get('reason', 'See impact-result.json for the composed decision.')}

## Limitations

- One physical GPU and one capture session.
- Laptop thermal and power state can affect measurements.
- The model is analytical and a lower bound, not cycle accurate or a GPU simulator.
- No measured-bottleneck claim is made.
- Results do not generalize to every RTX 3050 Laptop GPU.
- TensorForge Ops does not assess model quality or correctness. The recommendation
  covers measured performance and deployment/infrastructure evidence only.
"""


def main() -> int:
    args = parse_args()
    mode = baseline_mode(args.baseline_benchmark, args.baseline_telemetry)
    if args.output.exists():
        fail(f"output already exists: {args.output}")
    if mode == "regression" and (
        not args.baseline_benchmark.is_file() or not args.baseline_telemetry.is_file()
    ):
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
    from tensorforge_ops.telemetry_nvml import probe_capabilities
    from tensorforge_ops.tracking import compute_result_fingerprint

    baseline = None
    baseline_telemetry = None
    if mode == "regression":
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
        capabilities = probe_capabilities(telemetry_config.device_index)
        (out / "telemetry-trace.json").write_text(trace.to_json() + "\n", encoding="utf-8")
        (out / "telemetry-summary.json").write_text(telemetry.to_json() + "\n", encoding="utf-8")
        write_json(out / "telemetry-capabilities.json", capabilities)
        regression = None
        impact = None
        if mode == "regression":
            (out / "baseline-benchmark.json").write_text(baseline.to_json() + "\n", encoding="utf-8")
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
            "schema_version": 2,
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
            "regression": ({"status": regression.status} if regression is not None else {"status": "not_evaluated", "reason": NO_BASELINE_REASON}),
            "telemetry": {
                "status": "captured", "sample_count": telemetry.sample_count,
                "supported_fields": sorted(k for k, v in capabilities["capabilities"].items() if v),
                "unavailable_fields": sorted(k for k, v in capabilities["capabilities"].items() if not v),
            },
            "right_sizing": {"status": "omitted", "reason": NO_SIZING_REASON},
            "impact": ({"status": impact.readiness} if impact is not None else {"status": "not_evaluated", "reason": NO_IMPACT_REASON}),
        }
        write_json(out / "manifest.json", manifest)
        (out / "README.md").write_text(
            render_evidence_readme(manifest, primary_validation.to_dict(), contrast_validation.to_dict()),
            encoding="utf-8",
        )
        subprocess.run([sys.executable, str(ROOT / "tools/demo/validate.py"), str(out)], check=True)
        subprocess.run([sys.executable, str(ROOT / "tools/demo/summarize.py"), str(out)], check=True)
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
