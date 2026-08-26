"""Tests for the PyTorch benchmark backend.

Requires the "benchmark" extra (torch) to be installed; skipped entirely
otherwise so the rest of the suite stays runnable without torch.
"""

import pytest

torch = pytest.importorskip("torch")

from tensorforge.presets import get_workload_preset
from tensorforge_ops.benchmark import BenchmarkConfig
from tensorforge_ops.benchmark_pytorch import run_pytorch_benchmark

_FINGERPRINT = "0" * 64


def _config(**overrides) -> BenchmarkConfig:
    kwargs = dict(backend="pytorch", device="cpu", warmup_iterations=2, measured_iterations=5)
    kwargs.update(overrides)
    return BenchmarkConfig(**kwargs)


def test_gemm_cpu_benchmark_shape_and_fields():
    preset = get_workload_preset("gemm_tiny")
    result = run_pytorch_benchmark(preset, _FINGERPRINT, _config())

    assert result.core_result_fingerprint == _FINGERPRINT
    assert result.workload_preset == "gemm_tiny"
    assert result.workload_kind == "gemm"
    assert result.backend == "pytorch"
    assert result.device == "cpu"
    assert result.dtype == "fp16"
    assert result.warmup_iterations == 2
    assert result.measured_iterations == 5
    assert len(result.latency_samples_seconds) == 5
    assert all(s >= 0 for s in result.latency_samples_seconds)
    assert result.statistics.count == 5
    assert result.peak_memory_allocated_bytes is None  # CPU: not measured this milestone
    assert result.runtime_metadata["device_type"] == "cpu"


def test_conv2d_cpu_benchmark_runs():
    preset = get_workload_preset("conv_pointwise")
    result = run_pytorch_benchmark(preset, _FINGERPRINT, _config())
    assert result.workload_kind == "conv2d"
    assert result.statistics.count == 5


def test_transformer_gemm_only_cpu_benchmark_runs():
    preset = get_workload_preset("transformer_small")
    result = run_pytorch_benchmark(preset, _FINGERPRINT, _config())
    assert result.workload_kind == "transformer"
    assert result.statistics.count == 5


def test_cnn_kind_is_unsupported():
    preset = get_workload_preset("cnn_like_small")
    with pytest.raises(ValueError, match="does not support workload kind 'cnn'"):
        run_pytorch_benchmark(preset, _FINGERPRINT, _config())


def test_cuda_explicitly_requested_but_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    preset = get_workload_preset("gemm_tiny")
    with pytest.raises(ValueError, match="refusing to silently fall back to CPU"):
        run_pytorch_benchmark(preset, _FINGERPRINT, _config(device="cuda"))


def test_core_preset_untouched_by_benchmark_execution():
    preset = get_workload_preset("gemm_tiny")
    gemm_before = preset.gemm()
    run_pytorch_benchmark(preset, _FINGERPRINT, _config())
    gemm_after = preset.gemm()
    assert gemm_before == gemm_after


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_gemm_cuda_benchmark_reports_positive_peak_memory():
    preset = get_workload_preset("gemm_large_square")
    result = run_pytorch_benchmark(preset, _FINGERPRINT, _config(device="cuda", measured_iterations=10))

    assert result.device.startswith("cuda")
    assert result.peak_memory_allocated_bytes is not None
    assert result.peak_memory_allocated_bytes > 0
    assert "device_name" in result.runtime_metadata
    assert result.statistics.count == 10
