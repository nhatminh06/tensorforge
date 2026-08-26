import json
import subprocess

import pytest

mlflow = pytest.importorskip("mlflow")

from mlflow.tracking import MlflowClient

from tensorforge.experiments import ExperimentSpec, run_experiment
from tensorforge_ops.benchmark import BenchmarkConfig, BenchmarkResult, LatencyStatistics, compute_latency_statistics
from tensorforge_ops.calibration import (
    compute_calibration_fingerprint,
    predict,
    summarize_validation_results,
    validate_prediction,
)
from tensorforge_ops.telemetry import (
    TelemetryConfig,
    TelemetrySample,
    TelemetryTrace,
    correlate_telemetry,
    summarize_telemetry_trace,
)
from tensorforge_ops.sizing import (
    DeploymentCandidate,
    DeploymentCatalog,
    SloPolicy,
    build_sizing_plan,
    render_sizing_markdown_report,
)
from tensorforge_ops.tracking import (
    TrackingConfig,
    compute_result_fingerprint,
    list_runs,
    log_benchmark_result,
    log_calibration_profile,
    log_sizing_plan,
    log_telemetry,
    log_telemetry_correlation,
    log_validation_result,
    log_validation_summary,
    resolve_tracking_uri,
    track_experiment,
    track_result,
)


def make_spec(**overrides):
    defaults = dict(
        name="test-experiment",
        workload_preset="gemm_tiny",
        accelerator_preset="balanced",
        tile_m_values=(32, 64),
        tile_n_values=(32, 64),
        tile_k_values=(32, 64),
    )
    defaults.update(overrides)
    return ExperimentSpec(**defaults)


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    # MLflow's sqlite backend store defaults artifact storage to ./mlruns
    # relative to the CWD unless an artifact location is set explicitly.
    # Running each test from inside tmp_path keeps every MLflow-created
    # file (backend DB, artifact store) out of the repository entirely.
    monkeypatch.chdir(tmp_path)


def tracking_config(tmp_path, experiment_name="tensorforge-test"):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    return TrackingConfig(tracking_uri=uri, experiment_name=experiment_name)


# --- TrackingConfig validation --------------------------------------------------

def test_tracking_config_rejects_empty_experiment_name():
    with pytest.raises(ValueError):
        TrackingConfig(tracking_uri="sqlite:///x.db", experiment_name="")


def test_resolve_tracking_uri_precedence(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri("http://explicit:5000") == "http://explicit:5000"

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env:5000")
    assert resolve_tracking_uri(None) == "http://from-env:5000"
    assert resolve_tracking_uri("http://explicit:5000") == "http://explicit:5000"

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri(None) is None


# --- fingerprint -----------------------------------------------------------------

def test_fingerprint_format():
    result = run_experiment(make_spec())
    fingerprint = compute_result_fingerprint(result)
    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == len("sha256:") + 64


def test_identical_spec_gives_identical_fingerprint():
    spec = make_spec()
    result1 = run_experiment(spec)
    result2 = run_experiment(spec)
    assert compute_result_fingerprint(result1) == compute_result_fingerprint(result2)


def test_different_accelerator_gives_different_fingerprint():
    result_balanced = run_experiment(make_spec(accelerator_preset="balanced"))
    result_compute_heavy = run_experiment(make_spec(accelerator_preset="compute_heavy"))
    assert compute_result_fingerprint(result_balanced) != compute_result_fingerprint(result_compute_heavy)


# --- local temporary MLflow backend: tracking one run ----------------------------

def test_track_known_gemm_run(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec()
    result = run_experiment(spec)

    tracked = track_result(result, tracking)

    assert tracked.experiment_name == tracking.experiment_name
    assert tracked.tracking_uri == tracking.tracking_uri
    assert tracked.result_fingerprint == compute_result_fingerprint(result)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    experiment = client.get_experiment_by_name(tracking.experiment_name)
    assert experiment is not None

    run = client.get_run(tracked.run_id)
    assert run.data.params["workload_preset"] == "gemm_tiny"
    assert run.data.params["accelerator_preset"] == "balanced"
    assert run.data.params["workload_kind"] == "gemm"
    assert "modeled_flops" in run.data.metrics
    assert "perfect_overlap_time_seconds" in run.data.metrics
    assert run.data.metrics["modeled_flops"] == result.primary_metrics["total_flops"]
    assert run.data.tags["tensorforge.result_fingerprint"] == tracked.result_fingerprint
    assert run.data.tags["tensorforge.workload_kind"] == "gemm"
    assert run.data.tags["tensorforge.core_schema_version"] == "1"


def test_track_experiment_matches_track_result(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec()

    tracked = track_experiment(spec, tracking)
    result = run_experiment(spec)  # same spec, independently re-run

    assert tracked.result_fingerprint == compute_result_fingerprint(result)


# --- fingerprint identity vs. MLflow run identity --------------------------------

def test_identical_experiment_tracked_twice_has_different_run_ids_same_fingerprint(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec()

    tracked1 = track_experiment(spec, tracking)
    tracked2 = track_experiment(spec, tracking)

    assert tracked1.run_id != tracked2.run_id
    assert tracked1.result_fingerprint == tracked2.result_fingerprint


def test_changed_accelerator_changes_fingerprint_and_tracked_param(tmp_path):
    tracking = tracking_config(tmp_path)

    tracked_balanced = track_experiment(make_spec(accelerator_preset="balanced"), tracking)
    tracked_heavy = track_experiment(make_spec(accelerator_preset="compute_heavy"), tracking)

    assert tracked_balanced.result_fingerprint != tracked_heavy.result_fingerprint

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run_balanced = client.get_run(tracked_balanced.run_id)
    run_heavy = client.get_run(tracked_heavy.run_id)
    assert run_balanced.data.params["accelerator_preset"] == "balanced"
    assert run_heavy.data.params["accelerator_preset"] == "compute_heavy"
    assert run_balanced.data.params["pe_rows"] != run_heavy.data.params["pe_rows"]


# --- artifact exactness -----------------------------------------------------------

def test_logged_artifact_is_byte_identical_to_core_json(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec()
    result = run_experiment(spec)

    tracked = track_result(result, tracking)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    local_path = client.download_artifacts(tracked.run_id, "core-result.json", str(tmp_path))
    with open(local_path, encoding="utf-8") as f:
        artifact_text = f.read()

    assert artifact_text == result.to_json()
    # no MLflow metadata was injected into the artifact
    parsed = json.loads(artifact_text)
    assert set(parsed.keys()) == {
        "schema_version", "name", "workload", "accelerator", "search",
        "primary_metrics", "selected_mappings", "limitations",
    }


def test_tracking_metadata_artifact_is_separate_from_core_result(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    local_path = client.download_artifacts(tracked.run_id, "tracking-metadata.json", str(tmp_path))
    with open(local_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert "git" in metadata
    assert "mlflow_experiment_name" in metadata
    # the deterministic core result fields must NOT appear in this file
    assert "primary_metrics" not in metadata


# --- Core result unchanged by tracking -------------------------------------------

def test_tracking_does_not_mutate_core_result(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec()
    result = run_experiment(spec)
    before = result.to_json()

    track_result(result, tracking)

    assert result.to_json() == before


# --- experiment separation --------------------------------------------------------

def test_separate_experiments_stay_separate(tmp_path):
    tracking_a = tracking_config(tmp_path, experiment_name="tensorforge-a")
    tracking_b = tracking_config(tmp_path, experiment_name="tensorforge-b")

    tracked_a = track_experiment(make_spec(), tracking_a)
    tracked_b = track_experiment(make_spec(), tracking_b)

    assert tracked_a.experiment_id != tracked_b.experiment_id

    runs_a = list_runs(tracking_a)
    runs_b = list_runs(tracking_b)
    assert {r["run_id"] for r in runs_a} == {tracked_a.run_id}
    assert {r["run_id"] for r in runs_b} == {tracked_b.run_id}


# --- multi-accelerator comparison --------------------------------------------------

def test_three_accelerator_comparison(tmp_path):
    tracking = tracking_config(tmp_path)
    tracked = {
        accel: track_experiment(make_spec(accelerator_preset=accel), tracking)
        for accel in ("small", "balanced", "compute_heavy")
    }
    runs = list_runs(tracking, max_results=10)
    assert len(runs) == 3
    fingerprints = {t.result_fingerprint for t in tracked.values()}
    assert len(fingerprints) == 3  # all analytically different


# --- Transformer / Conv tracking ---------------------------------------------------

def test_track_transformer_workload(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec(workload_preset="transformer_small")
    tracked = track_experiment(spec, tracking)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert run.data.params["workload_kind"] == "transformer"
    assert "modeled_flops" in run.data.metrics
    assert run.data.tags["tensorforge.largest_time_contributor"]


def test_track_conv_workload_logs_im2col_expansion(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec(workload_preset="conv_spatial")
    tracked = track_experiment(spec, tracking)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert run.data.params["workload_kind"] == "conv2d"
    assert "im2col_expansion_ratio" in run.data.metrics
    assert run.data.metrics["im2col_expansion_ratio"] > 1.0


# --- unknown presets ---------------------------------------------------------------

def test_unknown_workload_preset_fails_before_tracking(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec(workload_preset="does-not-exist")
    with pytest.raises(ValueError):
        track_experiment(spec, tracking)


def test_unknown_accelerator_preset_fails_before_tracking(tmp_path):
    tracking = tracking_config(tmp_path)
    spec = make_spec(accelerator_preset="does-not-exist")
    with pytest.raises(ValueError):
        track_experiment(spec, tracking)


# --- attaching a measured benchmark to an already-tracked analytical run --------

def make_benchmark_result(fingerprint, samples=(0.01, 0.011, 0.009, 0.0105, 0.0095), peak_memory=None):
    return BenchmarkResult(
        core_result_fingerprint=fingerprint,
        workload_preset="gemm_tiny",
        workload_kind="gemm",
        backend="pytorch",
        device="cpu",
        dtype="fp16",
        warmup_iterations=2,
        measured_iterations=len(samples),
        latency_samples_seconds=tuple(samples),
        statistics=compute_latency_statistics(samples),
        peak_memory_allocated_bytes=peak_memory,
        runtime_metadata={"torch_version": "2.4.0", "device_type": "cpu"},
    )


def test_log_benchmark_result_attaches_to_same_run_as_analytical_result(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    benchmark = make_benchmark_result(tracked.result_fingerprint)
    log_benchmark_result(tracking, tracked.run_id, benchmark)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)

    # analytical metrics from track_result() are untouched
    assert "perfect_overlap_time_seconds" in run.data.metrics
    # measured metrics sit alongside them, never overwriting predicted names
    assert run.data.metrics["measured_mean_latency_seconds"] == pytest.approx(benchmark.statistics.mean_seconds)
    assert run.data.metrics["measured_throughput_per_second"] == pytest.approx(benchmark.statistics.throughput_per_second)
    assert "measured_p95_latency_seconds" not in run.data.metrics  # below the 20-sample threshold
    assert run.data.tags["tensorforge.core_result_fingerprint"] == tracked.result_fingerprint
    assert run.data.tags["tensorforge.measurement"] == "real"
    assert run.data.params["benchmark_backend"] == "pytorch"
    assert run.data.params["runtime_torch_version"] == "2.4.0"


def test_log_benchmark_result_omits_none_peak_memory_metric(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    benchmark = make_benchmark_result(tracked.result_fingerprint, peak_memory=None)
    log_benchmark_result(tracking, tracked.run_id, benchmark)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert "measured_peak_memory_allocated_bytes" not in run.data.metrics


def test_log_benchmark_result_logs_positive_peak_memory_metric(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    benchmark = make_benchmark_result(tracked.result_fingerprint, peak_memory=123456)
    log_benchmark_result(tracking, tracked.run_id, benchmark)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert run.data.metrics["measured_peak_memory_allocated_bytes"] == 123456.0


def test_log_benchmark_result_artifact_is_byte_identical_to_benchmark_json(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    benchmark = make_benchmark_result(tracked.result_fingerprint)
    log_benchmark_result(tracking, tracked.run_id, benchmark)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    local_path = client.download_artifacts(tracked.run_id, "benchmark-result.json", str(tmp_path))
    with open(local_path, encoding="utf-8") as f:
        artifact_text = f.read()
    assert artifact_text == benchmark.to_json()


# --- attaching calibration/validation to an already-tracked analytical run ------

def make_calibration_profile(effective_compute=1e9, effective_memory=1e9):
    from tensorforge_ops.calibration import ComputeProbeResult, DeviceCalibrationProfile, MemoryProbeResult

    compute_probe = ComputeProbeResult(
        shape_m=8, shape_n=8, shape_k=8, flops=1024,
        latency_samples_seconds=(0.001,), p50_seconds=0.001,
        effective_compute_flops_per_second=effective_compute,
    )
    memory_probe = MemoryProbeResult(
        payload_bytes=1024, modeled_copy_traffic_bytes=2048,
        latency_samples_seconds=(0.001,), p50_seconds=0.001,
        effective_memory_bandwidth_bytes_per_second=effective_memory,
    )
    return DeviceCalibrationProfile(
        calibration_schema_version=1,
        backend="pytorch", device_type="cpu", device_index=None, device_name=None,
        dtype="fp16",
        runtime_metadata={"torch_version": "2.4.0", "device_type": "cpu"},
        device_metadata={},
        compute_probe=compute_probe, memory_probe=memory_probe,
        effective_compute_flops_per_second=effective_compute,
        effective_memory_bandwidth_bytes_per_second=effective_memory,
    )


def test_log_calibration_profile_attaches_metrics_and_artifact(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    profile = make_calibration_profile()
    fingerprint = log_calibration_profile(tracking, tracked.run_id, profile)

    assert fingerprint == compute_calibration_fingerprint(profile)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert run.data.metrics["calibration_effective_compute_flops_per_second"] == 1e9
    assert run.data.tags["tensorforge.calibration"] == "empirical"
    assert run.data.tags["tensorforge.calibration_fingerprint"] == fingerprint
    # analytical metrics from track_result() are untouched
    assert "perfect_overlap_time_seconds" in run.data.metrics

    local_path = client.download_artifacts(tracked.run_id, "calibration-profile.json", str(tmp_path))
    with open(local_path, encoding="utf-8") as f:
        assert f.read() == profile.to_json()


def test_log_validation_result_attaches_metrics_and_artifact(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    profile = make_calibration_profile()
    log_calibration_profile(tracking, tracked.run_id, profile)

    benchmark = make_benchmark_result(tracked.result_fingerprint, samples=(0.01, 0.011, 0.009, 0.0105, 0.0095))
    prediction = predict(get_workload_preset_for_spec(), profile)
    validation = validate_prediction(profile, prediction, tracked.result_fingerprint, benchmark)
    log_validation_result(tracking, tracked.run_id, validation)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert run.data.metrics["validation_absolute_percentage_error"] == pytest.approx(validation.absolute_percentage_error)
    assert run.data.tags["tensorforge.validation"] == "prediction-vs-measurement"
    assert run.data.tags["tensorforge.predicted_bottleneck"] == validation.predicted_bottleneck
    # measured_* and predicted_* Core/benchmark metrics remain untouched
    assert "measured_mean_latency_seconds" not in run.data.metrics  # benchmark metrics were never logged here
    assert "perfect_overlap_time_seconds" in run.data.metrics

    local_path = client.download_artifacts(tracked.run_id, "validation-result.json", str(tmp_path))
    with open(local_path, encoding="utf-8") as f:
        assert f.read() == validation.to_json()


def get_workload_preset_for_spec():
    from tensorforge.presets import get_workload_preset

    return get_workload_preset("gemm_tiny")


def test_log_validation_summary_creates_its_own_run(tmp_path):
    tracking = tracking_config(tmp_path)
    profile = make_calibration_profile()
    result = run_experiment(make_spec())
    fingerprint = compute_result_fingerprint(result)
    benchmark = make_benchmark_result(fingerprint, samples=(0.01, 0.011, 0.009, 0.0105, 0.0095))
    prediction = predict(get_workload_preset_for_spec(), profile)
    validation = validate_prediction(profile, prediction, fingerprint, benchmark)
    summary = summarize_validation_results((validation,))

    run_id = log_validation_summary(tracking, summary)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.metrics["validation_summary_count"] == 1
    assert run.data.tags["tensorforge.validation"] == "prediction-vs-measurement-summary"

    local_path = client.download_artifacts(run_id, "validation-summary.json", str(tmp_path))
    with open(local_path, encoding="utf-8") as f:
        assert f.read() == summary.to_json()


# --- attaching telemetry / telemetry correlation to an already-tracked run -------

def make_telemetry_trace(gpu_values=(60.0, 70.0, 80.0), power_watts=None):
    samples = tuple(
        TelemetrySample(
            timestamp_seconds=i * 0.15, gpu_utilization_percent=v, memory_activity_percent=10.0,
            memory_used_bytes=1_000_000, memory_free_bytes=3_000_000, memory_total_bytes=4_000_000,
            power_watts=power_watts, temperature_celsius=60.0, sm_clock_mhz=1500, memory_clock_mhz=6001,
            performance_state=0, clock_event_reasons_bitmask=0, clock_event_reasons=(),
        )
        for i, v in enumerate(gpu_values)
    )
    return TelemetryTrace(
        telemetry_schema_version=1, backend="nvml", device_index=0, device_name="Fake GPU",
        sample_interval_seconds=0.15, requested_duration_seconds=0.45, actual_duration_seconds=0.46,
        samples=samples, workload_preset="gemm_tiny", workload_kind="gemm", benchmark_backend="pytorch",
        dtype="fp16", core_result_fingerprint="sha256:" + "0" * 64, runtime_metadata={"nvml_driver_version": "610.43.03"},
    )


def test_log_telemetry_attaches_metrics_and_artifacts(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    trace = make_telemetry_trace()
    summary = summarize_telemetry_trace(trace)
    log_telemetry(tracking, tracked.run_id, trace, summary)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert run.data.metrics["telemetry_gpu_util_percent_mean"] == pytest.approx(70.0)
    assert run.data.metrics["telemetry_gpu_util_percent_max"] == pytest.approx(80.0)
    assert run.data.tags["tensorforge.telemetry"] == "nvml"
    # unavailable metric (power) never logged as 0
    assert "telemetry_power_watts_mean" not in run.data.metrics
    # analytical metrics from track_result() are untouched
    assert "perfect_overlap_time_seconds" in run.data.metrics

    trace_path = client.download_artifacts(tracked.run_id, "telemetry-trace.json", str(tmp_path))
    with open(trace_path, encoding="utf-8") as f:
        assert f.read() == trace.to_json()
    summary_path = client.download_artifacts(tracked.run_id, "telemetry-summary.json", str(tmp_path))
    with open(summary_path, encoding="utf-8") as f:
        assert f.read() == summary.to_json()


def test_log_telemetry_logs_available_advanced_metrics_tag_correctly(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    trace = make_telemetry_trace()
    summary = summarize_telemetry_trace(trace)
    log_telemetry(tracking, tracked.run_id, trace, summary)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    # this fixture never sets sm_activity/tensor_activity/etc -- unavailable
    assert run.data.tags["tensorforge.telemetry_advanced_metrics"] == "unavailable"


def test_log_telemetry_correlation_attaches_artifact_and_tags(tmp_path):
    tracking = tracking_config(tmp_path)
    result = run_experiment(make_spec())
    tracked = track_result(result, tracking)

    baseline_summary = summarize_telemetry_trace(make_telemetry_trace(gpu_values=(90.0, 90.0)))
    candidate_summary = summarize_telemetry_trace(make_telemetry_trace(gpu_values=(50.0, 50.0)))
    correlation = correlate_telemetry(baseline_summary, candidate_summary, regression_status="FAIL", workload_preset="gemm_tiny", workload_kind="gemm")

    log_telemetry_correlation(tracking, tracked.run_id, correlation)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(tracked.run_id)
    assert run.data.tags["tensorforge.telemetry_correlation"] == "diagnostic-context"
    assert int(run.data.tags["tensorforge.telemetry_signal_count"]) >= 1
    # regression status/metrics on the run remain whatever track_result() set --
    # this function never touches Milestone-14 regression fields.

    local_path = client.download_artifacts(tracked.run_id, "telemetry-correlation.json", str(tmp_path))
    with open(local_path, encoding="utf-8") as f:
        assert f.read() == correlation.to_json()


# --- logging a right-sizing plan (Milestone 16) as its own MLflow run -------------

def make_sizing_plan():
    from decimal import Decimal

    # compute_latency_statistics() only derives a p95 with >= 20 samples,
    # so build LatencyStatistics directly with an explicit p95 here --
    # this fixture only needs a FEASIBLE candidate, not a real timing run.
    stats = LatencyStatistics(
        count=5, mean_seconds=0.0098, p50_seconds=0.0098, p95_seconds=0.011, p99_seconds=None,
        min_seconds=0.009, max_seconds=0.011, throughput_per_second=102.0,
    )
    bench = BenchmarkResult(
        core_result_fingerprint="sha256:" + "0" * 64, workload_preset="gemm_tiny", workload_kind="gemm",
        backend="pytorch", device="cuda:0", dtype="fp16", warmup_iterations=1, measured_iterations=5,
        latency_samples_seconds=(0.01, 0.011, 0.009, 0.0105, 0.0095), statistics=stats,
        peak_memory_allocated_bytes=None, runtime_metadata={"torch_version": "2.13.0"},
    )
    candidate = DeploymentCandidate("gpu-a", bench, Decimal("0.50"), "USD")
    catalog = DeploymentCatalog(1, "USD", (candidate,))
    slo = SloPolicy(required_invocations_per_second=10.0, max_p95_latency_seconds=0.02)
    return build_sizing_plan(catalog, slo)


def test_log_sizing_plan_creates_its_own_run(tmp_path):
    tracking = tracking_config(tmp_path)
    plan = make_sizing_plan()
    report = render_sizing_markdown_report(plan)

    run_id = log_sizing_plan(tracking, plan, report_markdown=report)

    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    run = client.get_run(run_id)
    assert run.data.tags["tensorforge.sizing"] == "measured"
    assert run.data.tags["tensorforge.recommended_candidate"] == "gpu-a"
    assert run.data.metrics["sizing_feasible_candidate_count"] == 1
    assert run.data.metrics["sizing_recommended_hourly_cost"] == pytest.approx(0.50)

    result_path = client.download_artifacts(run_id, "sizing-result.json", str(tmp_path))
    with open(result_path, encoding="utf-8") as f:
        assert f.read() == plan.to_json()
    report_path = client.download_artifacts(run_id, "sizing-report.md", str(tmp_path))
    with open(report_path, encoding="utf-8") as f:
        assert f.read() == report


# --- Core boundary: no MLflow import anywhere under src/tensorforge/ --------------

def test_core_package_does_not_import_mlflow():
    output = subprocess.run(
        ["grep", "-rl", "mlflow", "src/tensorforge/"],
        cwd=__file__.rsplit("/tests/", 1)[0],
        capture_output=True, text=True,
    )
    # grep exit code 1 means "no matches" -- that's the expected, passing state.
    assert output.returncode == 1, f"unexpected mlflow reference(s) in Core:\n{output.stdout}"
