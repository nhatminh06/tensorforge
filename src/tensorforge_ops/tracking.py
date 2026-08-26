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
