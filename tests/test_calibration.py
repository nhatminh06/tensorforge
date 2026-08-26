"""Tests for tensorforge_ops.calibration: pure formulas, feature extraction,
and prediction/validation error math. No torch/mlflow dependency.
"""

import json

import pytest

from tensorforge.presets import get_workload_preset
from tensorforge_ops.benchmark import BenchmarkConfig, BenchmarkResult, compute_latency_statistics
from tensorforge_ops.calibration import (
    CalibratedPrediction,
    CalibrationConfig,
    ComputeProbeResult,
    DeviceCalibrationProfile,
    MemoryProbeResult,
    calibration_profile_from_dict,
    compute_calibration_fingerprint,
    compute_probe_rate,
    extract_workload_features,
    memory_probe_bandwidth,
    memory_probe_traffic_bytes,
    predict,
    summarize_validation_results,
    validate_prediction,
)


def make_profile(
    effective_compute_flops_per_second,
    effective_memory_bandwidth_bytes_per_second,
    backend="pytorch", device_type="cuda", device_index=0, device_name="Fake GPU",
    dtype="fp16", torch_version="2.13.0+cu126", cuda_runtime_version="12.6",
):
    compute_probe = ComputeProbeResult(
        shape_m=8, shape_n=8, shape_k=8, flops=1024,
        latency_samples_seconds=(0.001,), p50_seconds=0.001,
        effective_compute_flops_per_second=effective_compute_flops_per_second,
    )
    memory_probe = MemoryProbeResult(
        payload_bytes=1024, modeled_copy_traffic_bytes=2048,
        latency_samples_seconds=(0.001,), p50_seconds=0.001,
        effective_memory_bandwidth_bytes_per_second=effective_memory_bandwidth_bytes_per_second,
    )
    runtime_metadata = {"torch_version": torch_version, "device_type": device_type}
    if device_type == "cuda":
        runtime_metadata.update(
            device_index=device_index, device_name=device_name, cuda_runtime_version=cuda_runtime_version
        )
    return DeviceCalibrationProfile(
        calibration_schema_version=1,
        backend=backend, device_type=device_type, device_index=device_index if device_type == "cuda" else None,
        device_name=device_name if device_type == "cuda" else None,
        dtype=dtype,
        runtime_metadata=runtime_metadata,
        device_metadata={},
        compute_probe=compute_probe, memory_probe=memory_probe,
        effective_compute_flops_per_second=effective_compute_flops_per_second,
        effective_memory_bandwidth_bytes_per_second=effective_memory_bandwidth_bytes_per_second,
    )


def make_benchmark_result(p50_seconds, mean_seconds=None, backend="pytorch", device="cuda:0", dtype="fp16",
                           torch_version="2.13.0+cu126", device_index=0, device_name="Fake GPU", cuda_runtime_version="12.6"):
    samples = (p50_seconds,)
    stats = compute_latency_statistics(samples)
    # override p50/mean directly since a single-sample statistics object
    # sets p50==mean==the sample -- construct a LatencyStatistics-shaped
    # object with the exact values the test wants.
    from dataclasses import replace
    stats = replace(stats, p50_seconds=p50_seconds, mean_seconds=mean_seconds if mean_seconds is not None else p50_seconds)
    runtime_metadata = {"torch_version": torch_version}
    if device.startswith("cuda"):
        runtime_metadata.update(device_index=device_index, device_name=device_name, cuda_runtime_version=cuda_runtime_version)
    return BenchmarkResult(
        core_result_fingerprint="sha256:" + "0" * 64,
        workload_preset="gemm_tiny", workload_kind="gemm",
        backend=backend, device=device, dtype=dtype,
        warmup_iterations=1, measured_iterations=1,
        latency_samples_seconds=samples, statistics=stats,
        peak_memory_allocated_bytes=None, runtime_metadata=runtime_metadata,
    )


# --- CalibrationConfig validation --------------------------------------------------

def test_config_rejects_non_pytorch_backend():
    with pytest.raises(ValueError):
        CalibrationConfig(backend="onnxruntime", device="cpu", dtype="fp16")


def test_config_rejects_bad_dtype():
    with pytest.raises(ValueError):
        CalibrationConfig(backend="pytorch", device="cpu", dtype="int8")


def test_config_rejects_nonpositive_compute_dims():
    with pytest.raises(ValueError):
        CalibrationConfig(backend="pytorch", device="cpu", dtype="fp16", compute_m=0)


def test_config_rejects_nonpositive_memory_probe_mib():
    with pytest.raises(ValueError):
        CalibrationConfig(backend="pytorch", device="cpu", dtype="fp16", memory_probe_mib=0)


# --- probe rate formulas (known values, hard-coded) --------------------------------

def test_compute_probe_rate_known_value():
    # FLOPs = 1000, p50 = 0.01s -> 100,000 FLOP/s
    assert compute_probe_rate(1000, 0.01) == 100_000.0


def test_memory_probe_traffic_is_double_payload():
    assert memory_probe_traffic_bytes(1000) == 2000


def test_memory_probe_bandwidth_known_value():
    # payload = 1000 B -> traffic = 2000 B, p50 = 0.02s -> 100,000 B/s
    assert memory_probe_bandwidth(1000, 0.02) == 100_000.0


# --- calibrated prediction (known values, hard-coded) -------------------------------

def test_calibrated_prediction_known_values_memory_bound():
    profile = make_profile(effective_compute_flops_per_second=100_000.0, effective_memory_bandwidth_bytes_per_second=25_000.0)
    fingerprint = compute_calibration_fingerprint(profile)
    prediction = CalibratedPrediction(
        workload_preset="fake", workload_kind="gemm",
        modeled_flops=1000, modeled_baseline_dram_bytes=500,
        effective_compute_flops_per_second=profile.effective_compute_flops_per_second,
        effective_memory_bandwidth_bytes_per_second=profile.effective_memory_bandwidth_bytes_per_second,
        predicted_compute_seconds=1000 / 100_000.0,
        predicted_memory_seconds=500 / 25_000.0,
        predicted_latency_seconds=max(1000 / 100_000.0, 500 / 25_000.0),
        predicted_bottleneck="memory-bound",
        calibration_fingerprint=fingerprint,
    )
    assert prediction.predicted_compute_seconds == pytest.approx(0.01)
    assert prediction.predicted_memory_seconds == pytest.approx(0.02)
    assert prediction.predicted_latency_seconds == pytest.approx(0.02)
    assert prediction.predicted_bottleneck == "memory-bound"


def test_predict_uses_max_of_compute_and_memory_time():
    profile = make_profile(effective_compute_flops_per_second=100_000.0, effective_memory_bandwidth_bytes_per_second=25_000.0)

    class FakeGemmPreset:
        kind = "gemm"
        name = "fake"

        def gemm(self):
            class G:
                flops = 1000
                dram_bytes = 500

            return G()

    prediction = predict(FakeGemmPreset(), profile)
    assert prediction.predicted_compute_seconds == pytest.approx(0.01)
    assert prediction.predicted_memory_seconds == pytest.approx(0.02)
    assert prediction.predicted_latency_seconds == pytest.approx(0.02)
    assert prediction.predicted_bottleneck == "memory-bound"


# --- error formulas (known values, hard-coded) -------------------------------------

def make_validation_result(predicted_seconds, measured_p50_seconds):
    profile = make_profile(1e15, 1e12)
    fingerprint = compute_calibration_fingerprint(profile)
    prediction = CalibratedPrediction(
        workload_preset="gemm_tiny", workload_kind="gemm",
        modeled_flops=1, modeled_baseline_dram_bytes=1,
        effective_compute_flops_per_second=1e15, effective_memory_bandwidth_bytes_per_second=1e12,
        predicted_compute_seconds=predicted_seconds, predicted_memory_seconds=0.0,
        predicted_latency_seconds=predicted_seconds, predicted_bottleneck="compute-bound",
        calibration_fingerprint=fingerprint,
    )
    benchmark_result = make_benchmark_result(measured_p50_seconds, device="cuda:0")
    return validate_prediction(profile, prediction, "sha256:" + "1" * 64, benchmark_result)


def test_error_known_values_underestimate():
    # prediction 0.020s, measured p50 0.025s
    result = make_validation_result(0.020, 0.025)
    assert result.signed_error_seconds == pytest.approx(-0.005)
    assert result.absolute_error_seconds == pytest.approx(0.005)
    assert result.relative_error == pytest.approx(-0.2)
    assert result.absolute_percentage_error == pytest.approx(20.0)
    assert result.measured_to_predicted_ratio == pytest.approx(1.25)


def test_error_known_values_overestimate():
    # prediction 0.030s, measured 0.025s -> signed positive, ratio < 1
    result = make_validation_result(0.030, 0.025)
    assert result.signed_error_seconds == pytest.approx(0.005)
    assert result.signed_error_seconds > 0
    assert result.measured_to_predicted_ratio == pytest.approx(0.025 / 0.030)
    assert result.measured_to_predicted_ratio < 1.0


# --- device / runtime mismatch rejection --------------------------------------------

def test_validate_prediction_rejects_device_type_mismatch():
    profile = make_profile(1e9, 1e9, device_type="cuda")
    prediction = predict(get_workload_preset("gemm_tiny"), profile)
    cpu_result = make_benchmark_result(0.01, device="cpu", torch_version="2.13.0+cu126")
    with pytest.raises(ValueError, match="device type mismatch"):
        validate_prediction(profile, prediction, "sha256:x", cpu_result)


def test_validate_prediction_rejects_dtype_mismatch():
    profile = make_profile(1e9, 1e9, dtype="fp16")
    prediction = predict(get_workload_preset("gemm_tiny"), profile)
    fp32_result = make_benchmark_result(0.01, dtype="fp32")
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_prediction(profile, prediction, "sha256:x", fp32_result)


def test_validate_prediction_rejects_torch_version_mismatch():
    profile = make_profile(1e9, 1e9, torch_version="2.13.0+cu126")
    prediction = predict(get_workload_preset("gemm_tiny"), profile)
    other_version_result = make_benchmark_result(0.01, torch_version="2.12.0+cu126")
    with pytest.raises(ValueError, match="PyTorch runtime mismatch"):
        validate_prediction(profile, prediction, "sha256:x", other_version_result)


def test_validate_prediction_rejects_cuda_device_name_mismatch():
    profile = make_profile(1e9, 1e9, device_name="RTX 3050 Laptop GPU")
    prediction = predict(get_workload_preset("gemm_tiny"), profile)
    other_gpu_result = make_benchmark_result(0.01, device_name="RTX 4090")
    with pytest.raises(ValueError, match="CUDA device/runtime mismatch"):
        validate_prediction(profile, prediction, "sha256:x", other_gpu_result)


def test_validate_prediction_accepts_matching_device():
    profile = make_profile(1e9, 1e9)
    prediction = predict(get_workload_preset("gemm_tiny"), profile)
    matching_result = make_benchmark_result(0.01)
    result = validate_prediction(profile, prediction, "sha256:x", matching_result)
    assert result.workload_preset == "gemm_tiny"


# --- Core feature extraction ---------------------------------------------------------

def test_extract_features_gemm_matches_core_gemm_object():
    preset = get_workload_preset("gemm_tiny")
    flops, baseline_bytes = extract_workload_features(preset)
    gemm = preset.gemm()
    assert flops == gemm.flops
    assert baseline_bytes == gemm.dram_bytes


def test_extract_features_conv2d_uses_materialized_im2col_baseline():
    from tensorforge.convolution import lower_conv2d_to_gemm

    preset = get_workload_preset("conv_spatial")
    flops, baseline_bytes = extract_workload_features(preset)
    lowering = lower_conv2d_to_gemm(preset.spec())
    assert flops == lowering.gemm.flops
    assert baseline_bytes == lowering.gemm.dram_bytes


def test_extract_features_transformer_sums_baseline_bytes_by_repetition():
    from tensorforge.transformer import block_total_flops, derive_transformer_gemms

    preset = get_workload_preset("transformer_small")
    flops, baseline_bytes = extract_workload_features(preset)
    ops = derive_transformer_gemms(preset.spec())
    assert flops == block_total_flops(ops)
    assert baseline_bytes == sum(op.gemm.dram_bytes * op.repetitions for op in ops)


def test_extract_features_rejects_cnn_kind():
    preset = get_workload_preset("cnn_like_small")
    with pytest.raises(ValueError, match="does not support workload kind 'cnn'"):
        extract_workload_features(preset)


# --- calibration profile serialization ------------------------------------------------

def test_calibration_profile_roundtrip_and_fingerprint_stable():
    profile = make_profile(1e9, 1e9)
    fingerprint = compute_calibration_fingerprint(profile)
    restored = calibration_profile_from_dict(json.loads(profile.to_json()))
    assert restored.to_json() == profile.to_json()
    assert compute_calibration_fingerprint(restored) == fingerprint


def test_calibration_fingerprint_changes_with_measured_rate():
    profile_a = make_profile(1e9, 1e9)
    profile_b = make_profile(2e9, 1e9)  # different measured compute rate
    assert compute_calibration_fingerprint(profile_a) != compute_calibration_fingerprint(profile_b)


# --- validation summary ---------------------------------------------------------------

def test_summarize_validation_results_groups_by_kind_and_bottleneck():
    r1 = make_validation_result(0.020, 0.025)  # APE 20%
    r2 = make_validation_result(0.030, 0.025)  # some other APE
    summary = summarize_validation_results((r1, r2))
    assert summary.count == 2
    assert "gemm" in summary.by_workload_kind
    assert summary.by_workload_kind["gemm"]["count"] == 2
    assert "compute-bound" in summary.by_predicted_bottleneck
    assert summary.max_absolute_percentage_error == max(r1.absolute_percentage_error, r2.absolute_percentage_error)


def test_summarize_validation_results_rejects_empty():
    with pytest.raises(ValueError):
        summarize_validation_results(())
