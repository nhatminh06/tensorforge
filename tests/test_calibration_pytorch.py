"""Tests for the PyTorch calibration probes. Requires the "benchmark"
extra (torch); skipped entirely otherwise.
"""

import pytest

torch = pytest.importorskip("torch")

from tensorforge_ops.calibration import CalibrationConfig
from tensorforge_ops.calibration_pytorch import run_pytorch_calibration


def _config(**overrides):
    kwargs = dict(
        backend="pytorch", device="cpu", dtype="fp16",
        compute_m=32, compute_n=32, compute_k=32,
        memory_probe_mib=1, warmup_iterations=1, measured_iterations=3,
    )
    kwargs.update(overrides)
    return CalibrationConfig(**kwargs)


def test_cpu_compute_probe_positive_rate():
    profile = run_pytorch_calibration(_config())
    probe = profile.compute_probe
    assert len(probe.latency_samples_seconds) == 3
    assert probe.p50_seconds > 0
    assert probe.effective_compute_flops_per_second > 0
    assert probe.flops == 2 * 32 * 32 * 32


def test_cpu_memory_probe_positive_bandwidth():
    profile = run_pytorch_calibration(_config())
    probe = profile.memory_probe
    assert len(probe.latency_samples_seconds) == 3
    assert probe.p50_seconds > 0
    assert probe.effective_memory_bandwidth_bytes_per_second > 0
    assert probe.modeled_copy_traffic_bytes == 2 * probe.payload_bytes


def test_cpu_profile_metadata():
    profile = run_pytorch_calibration(_config())
    assert profile.backend == "pytorch"
    assert profile.device_type == "cpu"
    assert profile.device_index is None
    assert profile.device_name is None
    assert profile.dtype == "fp16"
    assert profile.runtime_metadata["device_type"] == "cpu"
    assert "torch_version" in profile.runtime_metadata
    assert profile.device_metadata == {}


def test_cuda_explicitly_requested_but_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="refusing to silently fall back to CPU"):
        run_pytorch_calibration(_config(device="cuda"))


def test_resolve_calibration_dtype_rejects_unsupported_dtype():
    # CalibrationConfig itself only accepts fp16/fp32, so exercise the
    # backend-level dtype guard directly for a dtype string it doesn't know.
    from tensorforge_ops.calibration_pytorch import _resolve_calibration_dtype

    with pytest.raises(ValueError, match="not supported by the PyTorch calibration backend"):
        _resolve_calibration_dtype("int8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_calibration_positive_rates_and_metadata():
    profile = run_pytorch_calibration(_config(device="cuda", compute_m=256, compute_n=256, compute_k=256, memory_probe_mib=8))
    assert profile.device_type == "cuda"
    assert profile.device_index == 0
    assert profile.device_name
    assert profile.compute_probe.effective_compute_flops_per_second > 0
    assert profile.memory_probe.effective_memory_bandwidth_bytes_per_second > 0
    assert "compute_capability" in profile.device_metadata
    assert "total_device_memory_bytes" in profile.device_metadata
    assert profile.runtime_metadata.get("cuda_runtime_version") is not None
