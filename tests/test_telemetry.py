"""Tests for tensorforge_ops.telemetry: pure dataclasses, summaries, and
baseline-vs-candidate correlation. No NVML/torch/mlflow dependency.
"""

import json

import pytest

from tensorforge_ops.telemetry import (
    TELEMETRY_CORRELATION_SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    TelemetryConfig,
    TelemetrySample,
    TelemetryTrace,
    correlate_telemetry,
    low_utilization_note,
    render_telemetry_markdown_section,
    summarize_telemetry_trace,
    load_telemetry_summary,
    telemetry_sample_from_dict,
    telemetry_summary_from_dict,
    telemetry_trace_from_dict,
)


def make_sample(
    t=0.0, gpu=50.0, mem_act=10.0, mem_used=1_000_000, mem_free=3_000_000, mem_total=4_000_000,
    power=30.0, temp=60.0, sm_clock=1500, mem_clock=6001, pstate=0, reasons_bitmask=0, reasons=(),
    sm_activity=None, sm_occupancy=None, tensor_activity=None, dram_bw=None,
):
    return TelemetrySample(
        timestamp_seconds=t, gpu_utilization_percent=gpu, memory_activity_percent=mem_act,
        memory_used_bytes=mem_used, memory_free_bytes=mem_free, memory_total_bytes=mem_total,
        power_watts=power, temperature_celsius=temp, sm_clock_mhz=sm_clock, memory_clock_mhz=mem_clock,
        performance_state=pstate, clock_event_reasons_bitmask=reasons_bitmask, clock_event_reasons=reasons,
        sm_activity_percent=sm_activity, sm_occupancy_percent=sm_occupancy,
        tensor_activity_percent=tensor_activity, dram_bandwidth_utilization_percent=dram_bw,
    )


def make_trace(samples, **overrides):
    kwargs = dict(
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION, backend="nvml", device_index=0,
        device_name="Fake GPU", sample_interval_seconds=0.15, requested_duration_seconds=3.0,
        actual_duration_seconds=3.1, samples=tuple(samples),
        workload_preset="gemm_large_square", workload_kind="gemm", benchmark_backend="pytorch", dtype="fp16",
        core_result_fingerprint="sha256:" + "0" * 64, runtime_metadata={"nvml_driver_version": "610.43.03"},
    )
    kwargs.update(overrides)
    return TelemetryTrace(**kwargs)


# --- TelemetryConfig validation --------------------------------------------------

def test_config_rejects_bad_backend():
    with pytest.raises(ValueError):
        TelemetryConfig(backend="not-a-backend", device_index=0)


def test_config_rejects_negative_device_index():
    with pytest.raises(ValueError):
        TelemetryConfig(backend="nvml", device_index=-1)


def test_config_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        TelemetryConfig(backend="nvml", device_index=0, sample_interval_seconds=0)


def test_config_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        TelemetryConfig(backend="nvml", device_index=0, telemetry_duration_seconds=-1)


def test_config_valid_defaults():
    config = TelemetryConfig(backend="nvml", device_index=0)
    assert config.sample_interval_seconds > 0
    assert config.telemetry_duration_seconds >= 3.0


# --- serialization roundtrips ------------------------------------------------------

def test_sample_roundtrip():
    sample = make_sample(reasons_bitmask=1, reasons=("GpuIdle",))
    restored = telemetry_sample_from_dict(sample.to_dict())
    assert restored == sample


def test_trace_roundtrip():
    trace = make_trace([make_sample(t=0.0), make_sample(t=0.15, gpu=90.0)])
    restored = telemetry_trace_from_dict(json.loads(trace.to_json()))
    assert restored.to_json() == trace.to_json()


def test_trace_from_dict_rejects_schema_mismatch():
    trace = make_trace([make_sample()])
    d = json.loads(trace.to_json())
    d["telemetry_schema_version"] = 999
    with pytest.raises(ValueError, match="schema version mismatch"):
        telemetry_trace_from_dict(d)


# --- summary: known values (hard-coded) ---------------------------------------------

def test_summary_known_values_gpu_utilization():
    trace = make_trace([make_sample(t=0.0, gpu=20.0), make_sample(t=0.15, gpu=40.0), make_sample(t=0.3, gpu=60.0)])
    summary = summarize_telemetry_trace(trace)
    assert summary.gpu_utilization_percent.mean == pytest.approx(40.0)
    assert summary.gpu_utilization_percent.min == 20.0
    assert summary.gpu_utilization_percent.max == 60.0
    assert summary.gpu_utilization_percent.sample_count == 3


def test_summary_known_values_multiple_metrics():
    samples = [
        make_sample(t=0.0, power=10.0, temp=50.0, sm_clock=1000, mem_clock=5000),
        make_sample(t=0.15, power=20.0, temp=60.0, sm_clock=1500, mem_clock=6000),
        make_sample(t=0.3, power=30.0, temp=70.0, sm_clock=2000, mem_clock=6001),
    ]
    trace = make_trace(samples)
    summary = summarize_telemetry_trace(trace)
    assert summary.power_watts.mean == pytest.approx(20.0)
    assert summary.temperature_celsius.max == 70.0
    assert summary.sm_clock_mhz.min == 1000
    assert summary.sm_clock_mhz.max == 2000
    assert summary.memory_clock_mhz.mean == pytest.approx(5667.0, abs=1.0)


def test_summarize_rejects_empty_trace():
    trace = make_trace([])
    with pytest.raises(ValueError):
        summarize_telemetry_trace(trace)


# --- missing / partial metric behavior --------------------------------------------

def test_summary_all_missing_metric_is_none_not_zero():
    samples = [make_sample(power=None), make_sample(power=None), make_sample(power=None)]
    trace = make_trace(samples)
    summary = summarize_telemetry_trace(trace)
    assert summary.power_watts.mean is None
    assert summary.power_watts.min is None
    assert summary.power_watts.max is None
    assert summary.power_watts.sample_count == 0
    assert "power_watts" in summary.unavailable_metrics
    assert "power_watts" not in summary.available_metrics


def test_summary_partial_missing_metric_averages_only_available():
    samples = [make_sample(power=10.0), make_sample(power=None), make_sample(power=20.0)]
    trace = make_trace(samples)
    summary = summarize_telemetry_trace(trace)
    assert summary.power_watts.sample_count == 2
    assert summary.power_watts.mean == pytest.approx(15.0)
    assert summary.power_watts.min == 10.0
    assert summary.power_watts.max == 20.0
    assert "power_watts" in summary.available_metrics


def test_summary_roundtrip():
    trace = make_trace([make_sample(t=0.0), make_sample(t=0.15, gpu=90.0)])
    summary = summarize_telemetry_trace(trace)
    restored = telemetry_summary_from_dict(json.loads(summary.to_json()))
    assert restored.to_json() == summary.to_json()


def test_load_telemetry_summary_from_file(tmp_path):
    trace = make_trace([make_sample(t=0.0), make_sample(t=0.15, gpu=90.0)])
    summary = summarize_telemetry_trace(trace)
    path = tmp_path / "telemetry-summary.json"
    path.write_text(summary.to_json())
    loaded = load_telemetry_summary(str(path))
    assert loaded.to_json() == summary.to_json()


def test_throttle_reasons_observed_union_across_samples():
    samples = [
        make_sample(reasons_bitmask=1, reasons=("GpuIdle",)),
        make_sample(reasons_bitmask=4, reasons=("SwPowerCap",)),
        make_sample(reasons_bitmask=0, reasons=()),
    ]
    trace = make_trace(samples)
    summary = summarize_telemetry_trace(trace)
    assert summary.throttle_reasons_observed == ("GpuIdle", "SwPowerCap")


# --- low utilization note -----------------------------------------------------------

def test_low_utilization_note_present_when_mean_below_threshold():
    trace = make_trace([make_sample(gpu=5.0), make_sample(gpu=8.0)])
    summary = summarize_telemetry_trace(trace)
    note = low_utilization_note(summary)
    assert note is not None
    assert "underutilization" in note
    assert "root cause" not in note.lower()


def test_low_utilization_note_absent_when_mean_high():
    trace = make_trace([make_sample(gpu=90.0), make_sample(gpu=95.0)])
    summary = summarize_telemetry_trace(trace)
    assert low_utilization_note(summary) is None


def test_low_utilization_note_none_when_metric_unavailable():
    trace = make_trace([make_sample(gpu=None), make_sample(gpu=None)])
    summary = summarize_telemetry_trace(trace)
    assert low_utilization_note(summary) is None


# --- correlation: percentage-point delta, not relative ------------------------------

def test_utilization_delta_is_percentage_points_not_relative():
    baseline = summarize_telemetry_trace(make_trace([make_sample(gpu=70.0), make_sample(gpu=70.0)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(gpu=50.0), make_sample(gpu=50.0)]))
    correlation = correlate_telemetry(baseline, candidate)
    delta = next(d for d in correlation.metric_deltas if d.metric == "gpu_utilization_percent_mean")
    assert delta.absolute_delta == pytest.approx(-20.0)
    assert delta.delta_kind == "percentage_points"


def test_correlation_schema_and_roundtrip():
    baseline = summarize_telemetry_trace(make_trace([make_sample(gpu=70.0)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(gpu=50.0)]))
    correlation = correlate_telemetry(baseline, candidate, regression_status="FAIL", workload_preset="gemm_tiny", workload_kind="gemm")
    assert correlation.telemetry_correlation_schema_version == TELEMETRY_CORRELATION_SCHEMA_VERSION
    d = json.loads(correlation.to_json())
    assert d["regression_status"] == "FAIL"
    assert d["workload"]["preset"] == "gemm_tiny"


# --- diagnostic signals: cautious language, evidence-backed --------------------------

def test_lower_gpu_activity_signal_emitted_when_material():
    baseline = summarize_telemetry_trace(make_trace([make_sample(gpu=90.0), make_sample(gpu=90.0)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(gpu=60.0), make_sample(gpu=60.0)]))
    correlation = correlate_telemetry(baseline, candidate)
    assert any("lower GPU activity" in s for s in correlation.signals)


def test_no_signal_when_delta_not_material():
    baseline = summarize_telemetry_trace(make_trace([make_sample(gpu=90.0)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(gpu=85.0)]))  # -5pp, below 10pp threshold
    correlation = correlate_telemetry(baseline, candidate)
    assert correlation.signals == ()


def test_higher_memory_activity_signal():
    baseline = summarize_telemetry_trace(make_trace([make_sample(mem_act=10.0)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(mem_act=30.0)]))
    correlation = correlate_telemetry(baseline, candidate)
    assert any("greater device-memory activity" in s for s in correlation.signals)


def test_clock_reduction_signal():
    baseline = summarize_telemetry_trace(make_trace([make_sample(sm_clock=1800)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(sm_clock=1500)]))  # -16.7%
    correlation = correlate_telemetry(baseline, candidate)
    assert any("lower SM clocks" in s for s in correlation.signals)


def test_thermal_or_power_limit_signal_requires_actual_reason():
    baseline = summarize_telemetry_trace(make_trace([make_sample(reasons_bitmask=0, reasons=())]))
    candidate_no_reason = summarize_telemetry_trace(make_trace([make_sample(reasons_bitmask=0, reasons=())]))
    assert correlate_telemetry(baseline, candidate_no_reason).signals == ()

    candidate_with_reason = summarize_telemetry_trace(
        make_trace([make_sample(reasons_bitmask=4, reasons=("SwPowerCap",))])
    )
    correlation = correlate_telemetry(baseline, candidate_with_reason)
    assert any("throttle event was observed" in s for s in correlation.signals)


def test_higher_vram_pressure_signal():
    baseline = summarize_telemetry_trace(make_trace([make_sample(mem_used=1_000_000, mem_total=10_000_000)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(mem_used=8_000_000, mem_total=10_000_000)]))
    correlation = correlate_telemetry(baseline, candidate)
    assert any("higher peak VRAM usage" in s for s in correlation.signals)


def test_signals_never_use_root_cause_language():
    baseline = summarize_telemetry_trace(make_trace([make_sample(gpu=90.0, mem_act=5.0, sm_clock=1800)]))
    candidate = summarize_telemetry_trace(
        make_trace([make_sample(gpu=40.0, mem_act=50.0, sm_clock=1400, reasons_bitmask=4, reasons=("SwPowerCap",))])
    )
    correlation = correlate_telemetry(baseline, candidate)
    assert len(correlation.signals) > 0
    forbidden = ("root cause", "proves", "definitely", "is memory-bound", "is compute-bound")
    for s in correlation.signals:
        lowered = s.lower()
        for word in forbidden:
            assert word not in lowered, f"signal used overclaiming language: {s!r}"


def test_regression_status_never_derived_by_telemetry_module():
    # correlate_telemetry only carries regression_status through as context;
    # it must never compute or alter a regression verdict itself.
    baseline = summarize_telemetry_trace(make_trace([make_sample(gpu=90.0)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(gpu=10.0)]))
    correlation = correlate_telemetry(baseline, candidate, regression_status="PASS")
    assert correlation.regression_status == "PASS"  # unchanged, even though telemetry clearly worsened


# --- markdown report -----------------------------------------------------------------

def test_markdown_report_states_no_gate_effect():
    baseline = summarize_telemetry_trace(make_trace([make_sample(gpu=90.0)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(gpu=60.0)]))
    correlation = correlate_telemetry(baseline, candidate, regression_status="FAIL")
    report = render_telemetry_markdown_section(correlation)
    assert "does not affect the regression gate result" in report
    assert "GPU Telemetry Context" in report


def test_markdown_report_lists_unavailable_metrics():
    baseline = summarize_telemetry_trace(make_trace([make_sample(power=None)]))
    candidate = summarize_telemetry_trace(make_trace([make_sample(power=None)]))
    correlation = correlate_telemetry(baseline, candidate)
    report = render_telemetry_markdown_section(correlation)
    assert "power_watts" in report
