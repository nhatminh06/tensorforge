"""Deterministic TensorForge Core results -> MLflow experiment tracking.

Dependency direction: tensorforge_ops -> tensorforge, never the reverse.
Nothing under src/tensorforge/ imports mlflow or tensorforge_ops; Core
continues to run without an MLflow tracking server. Tracking is a side
effect layered on top of an already-complete, already-deterministic Core
result -- it never re-derives or adjusts any analytical value.

    TensorForge Core result (ExperimentResult, immutable/deterministic)
            |
    compute_result_fingerprint()   -- SHA-256 of the canonical Core JSON
            |
    track_result()                  -- logs params/metrics/tags + the
            |                           exact Core JSON to MLflow
    MLflow Tracking Server (params, metrics, tags, artifacts)

MLflow parameters/metrics/tags are an INDEX for filtering and comparing
runs -- never a second, independently-computed result. The logged
`core-result.json` artifact is always byte-identical to
`ExperimentResult.to_json()`; it remains the authoritative full
analytical result. The fingerprint identifies a deterministic Core
analytical result, not a machine environment, an MLflow run, or (later)
a real hardware benchmark -- two tracked runs of the identical
ExperimentSpec have different MLflow run IDs but the same fingerprint.
"""

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from tensorforge.experiments import SCHEMA_VERSION, ExperimentResult, ExperimentSpec, run_experiment

from tensorforge_ops.benchmark import BENCHMARK_SCHEMA_VERSION, BenchmarkResult
from tensorforge_ops.telemetry import (
    TELEMETRY_CORRELATION_SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    TelemetryCorrelationResult,
    TelemetrySummary,
    TelemetryTrace,
)
from tensorforge_ops.calibration import (
    VALIDATION_SCHEMA_VERSION,
    DeviceCalibrationProfile,
    ValidationResult,
    ValidationSummary,
    compute_calibration_fingerprint,
)
from tensorforge_ops.sizing import SizingPlanResult

DEFAULT_EXPERIMENT_NAME = "tensorforge"


@dataclass(frozen=True)
class TrackingConfig:
    tracking_uri: str | None
    experiment_name: str = DEFAULT_EXPERIMENT_NAME

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")


def resolve_tracking_uri(explicit: str | None) -> str | None:
    """Precedence: explicit value > MLFLOW_TRACKING_URI env var > MLflow's
    own default (None -- do not invent a path; MLflow's default is the
    well-documented local ./mlruns directory). This never silently
    substitutes a different server than the one the user asked for.
    """
    if explicit:
        return explicit
    env_value = os.environ.get("MLFLOW_TRACKING_URI")
    if env_value:
        return env_value
    return None


@dataclass(frozen=True)
class TrackedRun:
    run_id: str
    experiment_id: str
    experiment_name: str
    tracking_uri: str | None
    result_fingerprint: str
    artifact_uri: str | None


def compute_result_fingerprint(result: ExperimentResult) -> str:
    """SHA-256 of the canonical deterministic Core JSON (result.to_json())."""
    canonical_bytes = result.to_json().encode("utf-8")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return f"sha256:{digest}"


def _stringify(values: tuple) -> str:
    return ",".join(str(v) for v in values)


def _git_metadata() -> dict:
    """Best-effort Git metadata for the repository containing this file.
    Never raises -- returns "unknown" fields if Git is unavailable or this
    file is not inside a Git working tree. Uses a fixed argv list (never
    shell=True, never user-supplied arguments), so there is no shell
    injection surface.
    """
    repo_dir = os.path.dirname(os.path.abspath(__file__))

    def _run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    commit = _run(["rev-parse", "HEAD"]) or "unknown"
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    status = _run(["status", "--porcelain"])
    dirty = "unknown" if status is None else ("true" if status else "false")
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _build_params(result: ExperimentResult) -> dict:
    spec = result.spec
    accelerator = result.accelerator_summary
    return {
        "workload_preset": spec.workload_preset,
        "workload_kind": result.workload_kind,
        "accelerator_preset": spec.accelerator_preset,
        "tile_m_values": _stringify(spec.tile_m_values),
        "tile_n_values": _stringify(spec.tile_n_values),
        "tile_k_values": _stringify(spec.tile_k_values),
        "schedules": _stringify(spec.schedule_names)
        if spec.schedule_names is not None
        else "c-resident,a-resident,b-resident",
        "pe_rows": accelerator["pe_rows"],
        "pe_cols": accelerator["pe_cols"],
        "sram_bytes": accelerator["sram_bytes"],
        "clock_hz": accelerator["clock_hz"],
        "bandwidth_bytes_per_second": accelerator["bandwidth_bytes_per_second"],
        "schema_version": SCHEMA_VERSION,
    }


# primary_metrics keys that are numeric across workload kinds -> MLflow metric key.
_NUMERIC_METRIC_KEYS = {
    "total_macs": "modeled_macs",
    "total_flops": "modeled_flops",
    "total_dram_bytes": "modeled_dram_bytes",
    "serialized_time_seconds": "serialized_time_seconds",
    "perfect_overlap_time_seconds": "perfect_overlap_time_seconds",
    "effective_arithmetic_intensity": "effective_arithmetic_intensity",
}
# primary_metrics keys that are categorical (string) -> tag key. Logged as
# tags, never as MLflow metrics (which are numeric only), and never
# invented when the workload kind doesn't produce them.
_CATEGORICAL_METRIC_KEYS = {
    "bottleneck": "tensorforge.bottleneck",
    "largest_time_contributor": "tensorforge.largest_time_contributor",
    "largest_dram_contributor": "tensorforge.largest_dram_contributor",
}


def _build_metrics(result: ExperimentResult) -> dict:
    metrics = {}
    for source_key, metric_key in _NUMERIC_METRIC_KEYS.items():
        if source_key in result.primary_metrics:
            metrics[metric_key] = float(result.primary_metrics[source_key])
    # Conv2D-specific: im2col expansion ratio lives in workload_summary, not
    # primary_metrics, but is a genuinely useful numeric comparison metric.
    if "im2col_expansion_ratio" in result.workload_summary:
        metrics["im2col_expansion_ratio"] = float(result.workload_summary["im2col_expansion_ratio"])
    return metrics


def _log_text_artifact(text: str, artifact_file: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, artifact_file)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        mlflow.log_artifact(path)


def track_result(
    result: ExperimentResult,
    tracking: TrackingConfig,
    run_name: str | None = None,
) -> TrackedRun:
    """Log an already-computed, immutable Core result to MLflow.

    Kept separate from track_experiment() so a result produced elsewhere
    (a CI job, a benchmark worker, a saved JSON file re-loaded later) can
    be tracked without re-running the Core model.
    """
    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_experiment(tracking.experiment_name)

    fingerprint = compute_result_fingerprint(result)
    git_meta = _git_metadata()
    core_json_text = result.to_json()

    default_run_name = f"{result.spec.workload_preset}__{result.spec.accelerator_preset}"

    with mlflow.start_run(run_name=run_name or default_run_name) as run:
        mlflow.log_params(_build_params(result))
        mlflow.log_metrics(_build_metrics(result))

        tags = {
            "tensorforge.phase": "ops",
            "tensorforge.core_schema_version": str(SCHEMA_VERSION),
            "tensorforge.workload_kind": result.workload_kind,
            "tensorforge.result_fingerprint": fingerprint,
            "git.commit": git_meta["commit"],
            "git.branch": git_meta["branch"],
            "git.dirty": git_meta["dirty"],
            "source": "tensorforge-core",
        }
        for source_key, tag_key in _CATEGORICAL_METRIC_KEYS.items():
            if source_key in result.primary_metrics:
                tags[tag_key] = str(result.primary_metrics[source_key])
        mlflow.set_tags(tags)

        _log_text_artifact(core_json_text, "core-result.json")

        tracking_metadata = {
            "mlflow_experiment_name": tracking.experiment_name,
            "mlflow_experiment_id": run.info.experiment_id,
            "git": git_meta,
            "tracking_adapter_schema_version": 1,
        }
        _log_text_artifact(json.dumps(tracking_metadata, sort_keys=True, indent=2), "tracking-metadata.json")

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        artifact_uri = run.info.artifact_uri

    return TrackedRun(
        run_id=run_id,
        experiment_id=experiment_id,
        experiment_name=tracking.experiment_name,
        tracking_uri=tracking.tracking_uri,
        result_fingerprint=fingerprint,
        artifact_uri=artifact_uri,
    )


def track_experiment(
    spec: ExperimentSpec,
    tracking: TrackingConfig,
    run_name: str | None = None,
) -> TrackedRun:
    """Run a Core experiment, then track its result. A ValueError raised
    here (unknown preset, infeasible search, etc.) is a Core failure and
    occurs before any MLflow interaction begins; any exception raised
    after that point is a tracking failure, not a Core failure.
    """
    result = run_experiment(spec)
    return track_result(result, tracking, run_name=run_name)


def _build_benchmark_params(benchmark: BenchmarkResult) -> dict:
    params = {
        "benchmark_backend": benchmark.backend,
        "benchmark_device": benchmark.device,
        "benchmark_dtype": benchmark.dtype,
        "warmup_iterations": benchmark.warmup_iterations,
        "measured_iterations": benchmark.measured_iterations,
    }
    # Small, flat runtime metadata (torch/CUDA version, device name, ...) --
    # never huge machine inventories, never latency samples.
    for key, value in benchmark.runtime_metadata.items():
        params[f"runtime_{key}"] = value
    return params


def _build_benchmark_metrics(benchmark: BenchmarkResult) -> dict:
    stats = benchmark.statistics
    metrics = {
        "measured_mean_latency_seconds": stats.mean_seconds,
        "measured_p50_latency_seconds": stats.p50_seconds,
        "measured_min_latency_seconds": stats.min_seconds,
        "measured_max_latency_seconds": stats.max_seconds,
        "measured_throughput_per_second": stats.throughput_per_second,
    }
    # Percentiles below the sample-count threshold are None on the result
    # (never fabricated) -- and are correspondingly just omitted here,
    # never logged as a misleading 0.0.
    if stats.p95_seconds is not None:
        metrics["measured_p95_latency_seconds"] = stats.p95_seconds
    if stats.p99_seconds is not None:
        metrics["measured_p99_latency_seconds"] = stats.p99_seconds
    if benchmark.peak_memory_allocated_bytes is not None:
        metrics["measured_peak_memory_allocated_bytes"] = float(benchmark.peak_memory_allocated_bytes)
    return metrics


def _build_benchmark_tags(benchmark: BenchmarkResult) -> dict:
    return {
        "tensorforge.measurement": "real",
        "tensorforge.core_result_fingerprint": benchmark.core_result_fingerprint,
        "tensorforge.benchmark_schema_version": str(BENCHMARK_SCHEMA_VERSION),
        "benchmark.backend": benchmark.backend,
        "benchmark.device_type": benchmark.device.split(":")[0],
    }


def log_benchmark_result(tracking: TrackingConfig, run_id: str, benchmark: BenchmarkResult) -> None:
    """Attach a measured BenchmarkResult to an existing MLflow run -- by
    strong preference, the SAME run already created by track_result()/
    track_experiment() for the analytical Core result this benchmark
    measures, so predicted and measured metrics sit side by side in one
    place for easy comparison in the MLflow UI.

    Uses MlflowClient directly (log_param/log_metric/set_tag/log_artifact
    by run_id) rather than mlflow.start_run(), so an already-closed
    analytical run can be safely appended to without reopening or
    altering its existing data. Never mutates or overwrites any
    predicted_*/analytical metric already logged on that run -- every key
    here is prefixed measured_/benchmark_/runtime_ to keep the two
    domains unambiguous.
    """
    client = MlflowClient(tracking_uri=tracking.tracking_uri)

    for key, value in _build_benchmark_params(benchmark).items():
        client.log_param(run_id, key, value)
    for key, value in _build_benchmark_metrics(benchmark).items():
        client.log_metric(run_id, key, value)
    for key, value in _build_benchmark_tags(benchmark).items():
        client.set_tag(run_id, key, value)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "benchmark-result.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(benchmark.to_json())
        client.log_artifact(run_id, path)


def _build_calibration_params(profile: DeviceCalibrationProfile) -> dict:
    params = {
        "calibration_backend": profile.backend,
        "calibration_device_type": profile.device_type,
        "calibration_dtype": profile.dtype,
        "calibration_compute_probe_shape": (
            f"{profile.compute_probe.shape_m}x{profile.compute_probe.shape_n}x{profile.compute_probe.shape_k}"
        ),
        "calibration_memory_probe_payload_bytes": profile.memory_probe.payload_bytes,
    }
    for key, value in profile.runtime_metadata.items():
        params[f"calibration_runtime_{key}"] = value
    for key, value in profile.device_metadata.items():
        params[f"calibration_device_{key}"] = value
    return params


def _build_calibration_metrics(profile: DeviceCalibrationProfile) -> dict:
    return {
        "calibration_effective_compute_flops_per_second": profile.effective_compute_flops_per_second,
        "calibration_effective_memory_bandwidth_bytes_per_second": profile.effective_memory_bandwidth_bytes_per_second,
    }


def log_calibration_profile(tracking: TrackingConfig, run_id: str, profile: DeviceCalibrationProfile) -> str:
    """Attach a DeviceCalibrationProfile to an existing MLflow run. Returns
    the calibration fingerprint that was logged, so callers can pass it on
    to log_validation_result() without recomputing it.
    """
    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    fingerprint = compute_calibration_fingerprint(profile)

    for key, value in _build_calibration_params(profile).items():
        client.log_param(run_id, key, value)
    for key, value in _build_calibration_metrics(profile).items():
        client.log_metric(run_id, key, value)

    tags = {
        "tensorforge.calibration": "empirical",
        "tensorforge.calibration_fingerprint": fingerprint,
        "tensorforge.calibration_schema_version": str(profile.calibration_schema_version),
        "calibration.device_name": profile.device_name or "",
        "calibration.dtype": profile.dtype,
        "calibration.backend": profile.backend,
    }
    for key, value in tags.items():
        client.set_tag(run_id, key, value)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "calibration-profile.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(profile.to_json())
        client.log_artifact(run_id, path)

    return fingerprint


def _build_validation_metrics(validation: ValidationResult) -> dict:
    return {
        "calibrated_predicted_compute_seconds": validation.predicted_compute_seconds,
        "calibrated_predicted_memory_seconds": validation.predicted_memory_seconds,
        "calibrated_predicted_latency_seconds": validation.predicted_latency_seconds,
        "validation_signed_error_seconds": validation.signed_error_seconds,
        "validation_absolute_error_seconds": validation.absolute_error_seconds,
        "validation_absolute_percentage_error": validation.absolute_percentage_error,
        "validation_measured_to_predicted_ratio": validation.measured_to_predicted_ratio,
    }


def log_validation_result(tracking: TrackingConfig, run_id: str, validation: ValidationResult) -> None:
    """Attach a ValidationResult (predicted vs. measured p50) to an
    existing MLflow run -- by strong preference the same run already
    holding the Core, benchmark, and calibration artifacts for this
    workload. Never overwrites Core's predicted_*/analytical metrics or
    Milestone-12's measured_* metrics; every key here is prefixed
    calibrated_predicted_/validation_ to keep the three domains distinct.
    """
    client = MlflowClient(tracking_uri=tracking.tracking_uri)

    for key, value in _build_validation_metrics(validation).items():
        client.log_metric(run_id, key, value)

    tags = {
        "tensorforge.validation": "prediction-vs-measurement",
        "tensorforge.calibration_fingerprint": validation.calibration_fingerprint,
        "tensorforge.predicted_bottleneck": validation.predicted_bottleneck,
        "tensorforge.validation_schema_version": str(VALIDATION_SCHEMA_VERSION),
    }
    for key, value in tags.items():
        client.set_tag(run_id, key, value)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "validation-result.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(validation.to_json())
        client.log_artifact(run_id, path)


def log_validation_summary(
    tracking: TrackingConfig,
    summary: ValidationSummary,
    run_name: str | None = None,
) -> str:
    """Log a ValidationSummary (across several workloads) as its own new
    MLflow run -- a rollup, not tied to any single Core/benchmark run.
    Individual per-workload validation runs remain separate; this does not
    create nested runs. Returns the created run_id.
    """
    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_experiment(tracking.experiment_name)

    with mlflow.start_run(run_name=run_name or "validation-suite-summary") as run:
        mlflow.log_metrics(
            {
                "validation_summary_count": summary.count,
                "validation_summary_mean_ape": summary.mean_absolute_percentage_error,
                "validation_summary_median_ape": summary.median_absolute_percentage_error,
                "validation_summary_max_ape": summary.max_absolute_percentage_error,
                "validation_summary_median_ratio": summary.median_measured_to_predicted_ratio,
            }
        )
        mlflow.set_tags(
            {
                "tensorforge.validation": "prediction-vs-measurement-summary",
                "tensorforge.validation_schema_version": str(VALIDATION_SCHEMA_VERSION),
            }
        )
        _log_text_artifact(summary.to_json(), "validation-summary.json")
        run_id = run.info.run_id

    return run_id


def _build_telemetry_summary_metrics(summary: TelemetrySummary) -> dict:
    metrics = {}

    def _add(prefix, stat, *, want_min=False):
        if stat.mean is not None:
            metrics[f"{prefix}_mean"] = stat.mean
        if stat.max is not None:
            metrics[f"{prefix}_max"] = stat.max
        if want_min and stat.min is not None:
            metrics[f"{prefix}_min"] = stat.min

    _add("telemetry_gpu_util_percent", summary.gpu_utilization_percent)
    _add("telemetry_memory_activity_percent", summary.memory_activity_percent)
    if summary.memory_used_bytes.max is not None:
        metrics["telemetry_vram_used_max_bytes"] = float(summary.memory_used_bytes.max)
    _add("telemetry_power_watts", summary.power_watts)
    if summary.temperature_celsius.max is not None:
        metrics["telemetry_temperature_max_celsius"] = summary.temperature_celsius.max
    _add("telemetry_sm_clock_mhz", summary.sm_clock_mhz, want_min=True)
    _add("telemetry_memory_clock_mhz", summary.memory_clock_mhz, want_min=True)
    for name, stat in (
        ("telemetry_sm_activity_percent", summary.sm_activity_percent),
        ("telemetry_sm_occupancy_percent", summary.sm_occupancy_percent),
        ("telemetry_tensor_activity_percent", summary.tensor_activity_percent),
        ("telemetry_dram_bandwidth_utilization_percent", summary.dram_bandwidth_utilization_percent),
    ):
        _add(name, stat)
    return metrics


def log_telemetry(
    tracking: TrackingConfig,
    run_id: str,
    trace: TelemetryTrace,
    summary: TelemetrySummary,
) -> None:
    """Attach a TelemetryTrace + its TelemetrySummary to an existing
    MLflow run -- by strong preference the same run already holding the
    Core/benchmark artifacts for this workload. Never logs an
    unavailable (None) metric as 0 -- see _build_telemetry_summary_metrics.
    """
    client = MlflowClient(tracking_uri=tracking.tracking_uri)

    for key, value in _build_telemetry_summary_metrics(summary).items():
        client.log_metric(run_id, key, value)

    advanced_metric_names = {
        "sm_activity_percent", "sm_occupancy_percent", "tensor_activity_percent", "dram_bandwidth_utilization_percent",
    }
    tags = {
        "tensorforge.telemetry": trace.backend,
        "tensorforge.telemetry_schema_version": str(TELEMETRY_SCHEMA_VERSION),
        "tensorforge.telemetry_device": trace.device_name or "",
        "tensorforge.telemetry_advanced_metrics": (
            "available" if advanced_metric_names & set(summary.available_metrics) else "unavailable"
        ),
    }
    for key, value in tags.items():
        client.set_tag(run_id, key, value)

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_path = os.path.join(tmpdir, "telemetry-trace.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            f.write(trace.to_json())
        client.log_artifact(run_id, trace_path)

        summary_path = os.path.join(tmpdir, "telemetry-summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary.to_json())
        client.log_artifact(run_id, summary_path)


def log_telemetry_correlation(tracking: TrackingConfig, run_id: str, correlation: TelemetryCorrelationResult) -> None:
    """Attach a TelemetryCorrelationResult to an existing MLflow run.
    Diagnostic evidence only -- never touches Milestone 14's regression
    metrics/tags, and never implies a gate decision.
    """
    client = MlflowClient(tracking_uri=tracking.tracking_uri)

    client.set_tag(run_id, "tensorforge.telemetry_correlation", "diagnostic-context")
    client.set_tag(run_id, "tensorforge.telemetry_correlation_schema_version", str(TELEMETRY_CORRELATION_SCHEMA_VERSION))
    if correlation.signals:
        client.set_tag(run_id, "tensorforge.telemetry_signal_count", str(len(correlation.signals)))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "telemetry-correlation.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(correlation.to_json())
        client.log_artifact(run_id, path)


def log_sizing_plan(
    tracking: TrackingConfig,
    plan: SizingPlanResult,
    report_markdown: str | None = None,
    run_name: str | None = None,
) -> str:
    """Log a SizingPlanResult (Milestone 16) as its own new MLflow run --
    a sizing plan spans multiple candidates/benchmark runs, so it does
    not belong to any single existing run. Never modifies
    core-result.json/benchmark-result.json/validation-result.json/
    telemetry artifacts on any other run. Returns the created run_id.
    """
    mlflow.set_tracking_uri(tracking.tracking_uri)
    mlflow.set_experiment(tracking.experiment_name)

    with mlflow.start_run(run_name=run_name or "sizing-plan") as run:
        metrics = {}
        if plan.required_invocations_per_second is not None:
            metrics["sizing_required_invocations_per_second"] = plan.required_invocations_per_second
        metrics["sizing_feasible_candidate_count"] = plan.feasible_candidate_count

        recommended = None
        if plan.recommended_candidate_id is not None:
            recommended = next(c for c in plan.candidate_plans if c.candidate_id == plan.recommended_candidate_id)
            metrics["sizing_recommended_replicas"] = recommended.required_replicas
            metrics["sizing_recommended_hourly_cost"] = float(recommended.total_hourly_cost)
            if recommended.monthly_cost is not None:
                metrics["sizing_recommended_monthly_cost"] = float(recommended.monthly_cost)
            if recommended.measured_p95_latency_seconds is not None:
                metrics["sizing_recommended_p95_seconds"] = recommended.measured_p95_latency_seconds
            metrics["sizing_recommended_usable_capacity_per_second"] = recommended.usable_capacity_per_replica
        mlflow.log_metrics(metrics)

        tags = {
            "tensorforge.sizing": "measured",
            "tensorforge.sizing_result_schema_version": str(plan.sizing_result_schema_version),
            "tensorforge.recommended_candidate": plan.recommended_candidate_id or "",
        }
        if plan.currency is not None:
            tags["currency"] = plan.currency
        mlflow.set_tags(tags)

        _log_text_artifact(plan.to_json(), "sizing-result.json")
        if report_markdown is not None:
            _log_text_artifact(report_markdown, "sizing-report.md")

        run_id = run.info.run_id

    return run_id


def list_runs(tracking: TrackingConfig, max_results: int = 20) -> tuple[dict, ...]:
    """Small optional helper: list recent tracked runs in one MLflow
    experiment as plain dicts. Not a dashboard -- just enough to compare a
    handful of runs from a script or the CLI.
    """
    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    experiment = client.get_experiment_by_name(tracking.experiment_name)
    if experiment is None:
        return ()
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=max_results
    )
    return tuple(
        {
            "run_id": r.info.run_id,
            "run_name": r.data.tags.get("mlflow.runName", ""),
            "result_fingerprint": r.data.tags.get("tensorforge.result_fingerprint", ""),
            "workload_preset": r.data.params.get("workload_preset", ""),
            "accelerator_preset": r.data.params.get("accelerator_preset", ""),
            "perfect_overlap_time_seconds": r.data.metrics.get("perfect_overlap_time_seconds"),
        }
        for r in runs
    )
