"""Tests for the NVML telemetry backend. Requires the "telemetry" extra
(nvidia-ml-py); skipped entirely otherwise. Fake-provider tests exercise
sampler thread lifecycle without requiring a live NVIDIA GPU; live tests
are gated separately behind actual NVML availability.
"""

import threading

import pytest

pynvml = pytest.importorskip("pynvml")

from tensorforge_ops.telemetry import TelemetryConfig
from tensorforge_ops.telemetry_nvml import (
    NvmlProvider,
    NvmlUnavailableError,
    decode_clock_event_reasons,
    probe_capabilities,
    run_telemetry_window,
)


def _nvml_actually_available() -> bool:
    try:
        NvmlProvider(0).shutdown()
        return True
    except NvmlUnavailableError:
        return False


_NVML_AVAILABLE = _nvml_actually_available()


# --- throttle reason decoding (pure, no GPU needed) ----------------------------------

def test_decode_clock_event_reasons_none_bitmask():
    assert decode_clock_event_reasons(None) == ()


def test_decode_clock_event_reasons_zero_bitmask():
    assert decode_clock_event_reasons(0) == ()


def test_decode_clock_event_reasons_single_bit():
    assert decode_clock_event_reasons(pynvml.nvmlClocksEventReasonGpuIdle) == ("GpuIdle",)


def test_decode_clock_event_reasons_multiple_bits():
    bitmask = pynvml.nvmlClocksEventReasonSwPowerCap | pynvml.nvmlClocksEventReasonHwThermalSlowdown
    decoded = set(decode_clock_event_reasons(bitmask))
    assert decoded == {"SwPowerCap", "HwThermalSlowdown"}


# --- fake provider: sampler thread lifecycle ------------------------------------------

class _FakeProvider:
    """Minimal stand-in for NvmlProvider -- no real NVML calls."""

    def __init__(self, device_index):
        self.device_index = device_index
        self.device_name = "Fake GPU"
        self._sample_count = 0
        self.shutdown_called = False

    def sample(self, timestamp_seconds):
        from tensorforge_ops.telemetry import TelemetrySample

        self._sample_count += 1
        return TelemetrySample(
            timestamp_seconds=timestamp_seconds, gpu_utilization_percent=50.0, memory_activity_percent=10.0,
            memory_used_bytes=1000, memory_free_bytes=2000, memory_total_bytes=3000,
            power_watts=20.0, temperature_celsius=55.0, sm_clock_mhz=1500, memory_clock_mhz=6000,
            performance_state=0, clock_event_reasons_bitmask=0, clock_event_reasons=(),
        )

    def runtime_metadata(self):
        return {"fake": True}

    def shutdown(self):
        self.shutdown_called = True


def test_sampler_thread_lifecycle(monkeypatch):
    fake_instances = []

    def _factory(device_index):
        provider = _FakeProvider(device_index)
        fake_instances.append(provider)
        return provider

    monkeypatch.setattr("tensorforge_ops.telemetry_nvml.NvmlProvider", _factory)

    threads_before = set(threading.enumerate())
    config = TelemetryConfig(backend="nvml", device_index=0, sample_interval_seconds=0.05, telemetry_duration_seconds=0.3)
    trace = run_telemetry_window(lambda: None, config)

    assert len(trace.samples) >= 2
    assert fake_instances[0].shutdown_called is True

    threads_after = set(threading.enumerate())
    assert threads_after == threads_before, "sampler thread was not cleaned up after run_telemetry_window returned"


def test_sampler_stops_cleanly_when_workload_raises(monkeypatch):
    monkeypatch.setattr("tensorforge_ops.telemetry_nvml.NvmlProvider", _FakeProvider)

    def _raising_workload():
        raise RuntimeError("synthetic workload failure")

    threads_before = set(threading.enumerate())
    config = TelemetryConfig(backend="nvml", device_index=0, sample_interval_seconds=0.05, telemetry_duration_seconds=1.0)
    with pytest.raises(RuntimeError, match="synthetic workload failure"):
        run_telemetry_window(_raising_workload, config)

    threads_after = set(threading.enumerate())
    assert threads_after == threads_before, "sampler thread leaked after workload raised"


def test_trace_carries_workload_identity(monkeypatch):
    monkeypatch.setattr("tensorforge_ops.telemetry_nvml.NvmlProvider", _FakeProvider)

    config = TelemetryConfig(backend="nvml", device_index=0, sample_interval_seconds=0.05, telemetry_duration_seconds=0.2)
    trace = run_telemetry_window(
        lambda: None, config,
        workload_preset="gemm_tiny", workload_kind="gemm", benchmark_backend="pytorch", dtype="fp16",
        core_result_fingerprint="sha256:" + "0" * 64,
    )
    assert trace.workload_preset == "gemm_tiny"
    assert trace.workload_kind == "gemm"
    assert trace.device_name == "Fake GPU"


def test_run_telemetry_window_rejects_non_nvml_backend():
    config = TelemetryConfig(backend="dcgm-exporter", device_index=0)
    with pytest.raises(ValueError, match="only supports backend='nvml'"):
        run_telemetry_window(lambda: None, config)


# --- live NVML (RTX 3050 in this development environment) ----------------------------

@pytest.mark.skipif(not _NVML_AVAILABLE, reason="NVML not available on this machine")
def test_live_nvml_initializes_and_samples():
    provider = NvmlProvider(0)
    try:
        assert provider.device_name is not None
        sample = provider.sample(0.0)
        assert sample.gpu_utilization_percent is not None
        assert sample.memory_used_bytes is not None
        assert sample.temperature_celsius is not None
    finally:
        provider.shutdown()


@pytest.mark.skipif(not _NVML_AVAILABLE, reason="NVML not available on this machine")
def test_live_probe_capabilities_reports_basic_metrics_supported():
    report = probe_capabilities(0)
    assert report["capabilities"]["gpu_utilization_percent"] is True
    assert report["capabilities"]["memory_used_bytes"] is True
    assert report["capabilities"]["temperature_celsius"] is True


@pytest.mark.skipif(not _NVML_AVAILABLE, reason="NVML not available on this machine")
def test_live_short_telemetry_window():
    config = TelemetryConfig(backend="nvml", device_index=0, sample_interval_seconds=0.1, telemetry_duration_seconds=0.5)
    trace = run_telemetry_window(lambda: None, config)
    assert len(trace.samples) >= 2
    assert trace.device_name is not None
