"""Tests for tensorforge_ops.regression: pure comparison, policy
validation, and error/mismatch handling. No torch/mlflow dependency.
"""

import json

import pytest

from tensorforge_ops.benchmark import BenchmarkResult, LatencyStatistics
from tensorforge_ops.regression import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_NOT_COMPARABLE,
    STATUS_PASS,
    MetricPolicy,
    RegressionPolicy,
    compare_benchmark_results,
    compare_regression_suite,
    regression_policy_from_dict,
    render_markdown_report,
)


def make_benchmark(
    p50, mean=None, p95=None, throughput=None, peak_mem=None,
    preset="gemm_large_square", kind="gemm", backend="pytorch", device="cuda:0", dtype="fp16",
    warmup=10, iterations=50, runtime_metadata=None, core_fingerprint=None,
):
    mean = mean if mean is not None else p50
    throughput = throughput if throughput is not None else (1.0 / mean)
    stats = LatencyStatistics(
        count=iterations, mean_seconds=mean, p50_seconds=p50,
        p95_seconds=p95, p99_seconds=None, min_seconds=p50 * 0.9, max_seconds=p50 * 1.2,
        throughput_per_second=throughput,
    )
    return BenchmarkResult(
        core_result_fingerprint=core_fingerprint or ("sha256:" + "0" * 64),
        workload_preset=preset, workload_kind=kind, backend=backend, device=device, dtype=dtype,
        warmup_iterations=warmup, measured_iterations=iterations,
        latency_samples_seconds=(p50,), statistics=stats, peak_memory_allocated_bytes=peak_mem,
        runtime_metadata=runtime_metadata if runtime_metadata is not None else {
            "torch_version": "2.13.0+cu126", "device_index": 0, "device_name": "RTX 3050 Laptop GPU",
            "cuda_runtime_version": "12.6",
        },
    )


def latency_policy(relative=0.10, absolute_seconds=None):
    return RegressionPolicy(metrics={
        "p50_latency_seconds": MetricPolicy(max_relative_regression=relative, max_absolute_regression_seconds=absolute_seconds),
    })


# --- MetricPolicy / RegressionPolicy validation -------------------------------------

def test_metric_policy_requires_at_least_one_threshold():
    with pytest.raises(ValueError):
        MetricPolicy()


def test_metric_policy_rejects_negative_threshold():
    with pytest.raises(ValueError):
        MetricPolicy(max_relative_regression=-0.1)


def test_metric_policy_rejects_nan_and_inf():
    with pytest.raises(ValueError):
        MetricPolicy(max_relative_regression=float("nan"))
    with pytest.raises(ValueError):
        MetricPolicy(max_relative_regression=float("inf"))


def test_regression_policy_rejects_empty_metrics():
    with pytest.raises(ValueError):
        RegressionPolicy(metrics={})


def test_regression_policy_rejects_unknown_metric_name():
    with pytest.raises(ValueError):
        RegressionPolicy(metrics={"not_a_real_metric": MetricPolicy(max_relative_regression=0.1)})


def test_regression_policy_rejects_wrong_schema_version():
    with pytest.raises(ValueError):
        RegressionPolicy(metrics={"p50_latency_seconds": MetricPolicy(max_relative_regression=0.1)}, regression_policy_schema_version=99)


def test_regression_policy_from_dict_roundtrip():
    policy = latency_policy(relative=0.1, absolute_seconds=1e-5)
    restored = regression_policy_from_dict(json.loads(policy.to_json()))
    assert restored.to_json() == policy.to_json()


# --- known-value PASS/FAIL (hard-coded per spec) ------------------------------------

def test_known_pass_latency_within_relative_threshold():
    baseline = make_benchmark(p50=0.100)
    candidate = make_benchmark(p50=0.105)  # +5%
    result = compare_benchmark_results(baseline, candidate, latency_policy(relative=0.10))
    assert result.status == STATUS_PASS


def test_known_fail_latency_beyond_relative_threshold():
    baseline = make_benchmark(p50=0.100)
    candidate = make_benchmark(p50=0.115)  # +15%
    result = compare_benchmark_results(baseline, candidate, latency_policy(relative=0.10))
    assert result.status == STATUS_FAIL
    assert "p50_latency_seconds" in result.failed_metrics


def test_absolute_tolerance_noise_floor_pass():
    # baseline 10us, candidate 11us: relative +10% vs relative allowance 5%,
    # but absolute allowance 2us -> allowed = max(0.5us, 2us) = 2us; actual 1us -> PASS
    baseline = make_benchmark(p50=10e-6)
    candidate = make_benchmark(p50=11e-6)
    policy = latency_policy(relative=0.05, absolute_seconds=2e-6)
    result = compare_benchmark_results(baseline, candidate, policy)
    comparison = result.metric_comparisons[0]
    assert comparison.allowed_regression == pytest.approx(2e-6)
    assert result.status == STATUS_PASS


# --- throughput ----------------------------------------------------------------------

def test_throughput_pass_within_allowed_drop():
    baseline = make_benchmark(p50=0.001, throughput=1000.0)
    candidate = make_benchmark(p50=0.001, throughput=920.0)  # -8%
    policy = RegressionPolicy(metrics={"throughput_per_second": MetricPolicy(max_relative_regression=0.10)})
    result = compare_benchmark_results(baseline, candidate, policy)
    assert result.status == STATUS_PASS


def test_throughput_fail_beyond_allowed_drop():
    baseline = make_benchmark(p50=0.001, throughput=1000.0)
    candidate = make_benchmark(p50=0.001, throughput=880.0)  # -12%
    policy = RegressionPolicy(metrics={"throughput_per_second": MetricPolicy(max_relative_regression=0.10)})
    result = compare_benchmark_results(baseline, candidate, policy)
    assert result.status == STATUS_FAIL


def test_throughput_improvement_passes():
    baseline = make_benchmark(p50=0.001, throughput=1000.0)
    candidate = make_benchmark(p50=0.001, throughput=1200.0)  # improvement
    policy = RegressionPolicy(metrics={"throughput_per_second": MetricPolicy(max_relative_regression=0.10)})
    result = compare_benchmark_results(baseline, candidate, policy)
    assert result.status == STATUS_PASS


# --- memory ----------------------------------------------------------------------------

def test_memory_pass_within_threshold():
    baseline = make_benchmark(p50=0.001, peak_mem=100 * 1024 * 1024)
    candidate = make_benchmark(p50=0.001, peak_mem=108 * 1024 * 1024)  # +8%
    policy = RegressionPolicy(metrics={"peak_memory_allocated_bytes": MetricPolicy(max_relative_regression=0.10)})
    result = compare_benchmark_results(baseline, candidate, policy)
    assert result.status == STATUS_PASS


def test_memory_fail_beyond_threshold():
    baseline = make_benchmark(p50=0.001, peak_mem=100 * 1024 * 1024)
    candidate = make_benchmark(p50=0.001, peak_mem=120 * 1024 * 1024)  # +20%
    policy = RegressionPolicy(metrics={"peak_memory_allocated_bytes": MetricPolicy(max_relative_regression=0.10)})
    result = compare_benchmark_results(baseline, candidate, policy)
    assert result.status == STATUS_FAIL


def test_memory_none_none_is_not_comparable_but_can_still_pass_overall():
    baseline = make_benchmark(p50=0.001, peak_mem=None, device="cpu", runtime_metadata={"torch_version": "2.13.0+cu126"})
    candidate = make_benchmark(p50=0.00102, peak_mem=None, device="cpu", runtime_metadata={"torch_version": "2.13.0+cu126"})
    policy = RegressionPolicy(metrics={
        "p50_latency_seconds": MetricPolicy(max_relative_regression=0.10),
        "peak_memory_allocated_bytes": MetricPolicy(max_relative_regression=0.10),
    })
    result = compare_benchmark_results(baseline, candidate, policy)
    memory_comparison = next(c for c in result.metric_comparisons if c.metric == "peak_memory_allocated_bytes")
    assert memory_comparison.status == STATUS_NOT_COMPARABLE
    assert result.status == STATUS_PASS


def test_memory_present_on_baseline_missing_on_candidate_is_error():
    baseline = make_benchmark(p50=0.001, peak_mem=100 * 1024 * 1024)
    candidate = make_benchmark(p50=0.001, peak_mem=None)
    policy = RegressionPolicy(metrics={"peak_memory_allocated_bytes": MetricPolicy(max_relative_regression=0.10)})
    result = compare_benchmark_results(baseline, candidate, policy)
    assert result.status == STATUS_ERROR


# --- multi-metric failure --------------------------------------------------------------

def test_multi_metric_failure_lists_exactly_failed_metrics():
    baseline = make_benchmark(p50=0.100, p95=0.110, throughput=1000.0, peak_mem=100 * 1024 * 1024)
    candidate = make_benchmark(p50=0.103, p95=0.140, throughput=850.0, peak_mem=105 * 1024 * 1024)
    policy = RegressionPolicy(metrics={
        "p50_latency_seconds": MetricPolicy(max_relative_regression=0.10),   # +3% -> PASS
        "p95_latency_seconds": MetricPolicy(max_relative_regression=0.10),   # +27% -> FAIL
        "throughput_per_second": MetricPolicy(max_relative_regression=0.10),  # -15% -> FAIL
        "peak_memory_allocated_bytes": MetricPolicy(max_relative_regression=0.10),  # +5% -> PASS
    })
    result = compare_benchmark_results(baseline, candidate, policy)
    assert result.status == STATUS_FAIL
    assert set(result.failed_metrics) == {"p95_latency_seconds", "throughput_per_second"}


# --- identity mismatch -> ERROR ---------------------------------------------------------

def test_workload_mismatch_is_error():
    baseline = make_benchmark(p50=0.001, preset="gemm_tiny", kind="gemm")
    candidate = make_benchmark(p50=0.001, preset="conv_spatial", kind="conv2d")
    result = compare_benchmark_results(baseline, candidate, latency_policy())
    assert result.status == STATUS_ERROR
    assert result.metric_comparisons == ()


def test_device_mismatch_is_error():
    baseline = make_benchmark(p50=0.001, runtime_metadata={"torch_version": "2.13.0+cu126", "device_index": 0, "device_name": "RTX 3050 Laptop GPU"})
    candidate = make_benchmark(p50=0.001, runtime_metadata={"torch_version": "2.13.0+cu126", "device_index": 0, "device_name": "RTX 4090"})
    result = compare_benchmark_results(baseline, candidate, latency_policy())
    assert result.status == STATUS_ERROR


def test_backend_mismatch_is_error():
    baseline = make_benchmark(p50=0.001, backend="pytorch")
    candidate = make_benchmark(p50=0.001, backend="onnxruntime")
    result = compare_benchmark_results(baseline, candidate, latency_policy())
    assert result.status == STATUS_ERROR


def test_dtype_mismatch_is_error():
    baseline = make_benchmark(p50=0.001, dtype="fp16")
    candidate = make_benchmark(p50=0.001, dtype="fp32")
    result = compare_benchmark_results(baseline, candidate, latency_policy())
    assert result.status == STATUS_ERROR


def test_cpu_vs_cuda_device_type_mismatch_is_error():
    baseline = make_benchmark(p50=0.001, device="cuda:0")
    candidate = make_benchmark(p50=0.001, device="cpu", runtime_metadata={"torch_version": "2.13.0+cu126"})
    result = compare_benchmark_results(baseline, candidate, latency_policy())
    assert result.status == STATUS_ERROR


def test_different_pytorch_version_is_not_automatically_rejected():
    baseline = make_benchmark(p50=0.100, runtime_metadata={"torch_version": "2.13.0+cu126", "device_index": 0, "device_name": "RTX 3050 Laptop GPU"})
    candidate = make_benchmark(p50=0.103, runtime_metadata={"torch_version": "2.14.0+cu126", "device_index": 0, "device_name": "RTX 3050 Laptop GPU"})
    result = compare_benchmark_results(baseline, candidate, latency_policy(relative=0.10))
    assert result.status == STATUS_PASS  # runtime change is surfaced in the report, not rejected


def test_iteration_count_mismatch_does_not_invalidate_identity():
    baseline = make_benchmark(p50=0.100, warmup=5, iterations=20)
    candidate = make_benchmark(p50=0.103, warmup=10, iterations=50)
    result = compare_benchmark_results(baseline, candidate, latency_policy(relative=0.10))
    assert result.status == STATUS_PASS
    assert result.baseline_measured_iterations == 20
    assert result.candidate_measured_iterations == 50


# --- report content ----------------------------------------------------------------------

def test_markdown_report_contains_key_content():
    baseline = make_benchmark(p50=0.00082, peak_mem=82 * 1024 * 1024)
    candidate = make_benchmark(p50=0.00091, peak_mem=84 * 1024 * 1024)
    policy = RegressionPolicy(metrics={
        "p50_latency_seconds": MetricPolicy(max_relative_regression=0.08),
        "peak_memory_allocated_bytes": MetricPolicy(max_relative_regression=0.10),
    })
    result = compare_benchmark_results(baseline, candidate, policy)
    report = render_markdown_report(result)
    assert "gemm_large_square" in report
    assert "FAIL" in report
    assert "p50 latency" in report
    assert "Failed metrics" in report


def test_markdown_report_for_error_has_no_verdict_table():
    baseline = make_benchmark(p50=0.001, preset="gemm_tiny")
    candidate = make_benchmark(p50=0.001, preset="conv_spatial", kind="conv2d")
    result = compare_benchmark_results(baseline, candidate, latency_policy())
    report = render_markdown_report(result)
    assert "ERROR" in report
    assert "|" not in report  # no metric table rendered for an ERROR result


def test_regression_result_json_roundtrip_structure():
    baseline = make_benchmark(p50=0.100)
    candidate = make_benchmark(p50=0.103)
    result = compare_benchmark_results(baseline, candidate, latency_policy(relative=0.10))
    d = json.loads(result.to_json())
    assert d["status"] == "PASS"
    assert d["workload"]["preset"] == "gemm_large_square"
    assert isinstance(d["metrics"], list)


# --- suite ---------------------------------------------------------------------------

def test_suite_overall_pass():
    pairs = [
        (make_benchmark(p50=0.1, preset="gemm_tiny"), make_benchmark(p50=0.103, preset="gemm_tiny")),
        (make_benchmark(p50=0.2, preset="gemm_large_square"), make_benchmark(p50=0.205, preset="gemm_large_square")),
    ]
    result = compare_regression_suite(pairs, latency_policy(relative=0.10))
    assert result.status == STATUS_PASS
    assert result.failed_workloads == ()
    assert result.errored_workloads == ()


def test_suite_overall_fail_when_one_member_fails():
    pairs = [
        (make_benchmark(p50=0.1, preset="gemm_tiny"), make_benchmark(p50=0.103, preset="gemm_tiny")),
        (make_benchmark(p50=0.2, preset="gemm_large_square"), make_benchmark(p50=0.3, preset="gemm_large_square")),
    ]
    result = compare_regression_suite(pairs, latency_policy(relative=0.10))
    assert result.status == STATUS_FAIL
    assert result.failed_workloads == ("gemm_large_square",)


def test_suite_overall_error_precedence_over_fail():
    pairs = [
        (make_benchmark(p50=0.2, preset="gemm_large_square"), make_benchmark(p50=0.3, preset="gemm_large_square")),  # FAIL
        (make_benchmark(p50=0.1, preset="gemm_tiny"), make_benchmark(p50=0.1, preset="conv_spatial", kind="conv2d")),  # ERROR
    ]
    result = compare_regression_suite(pairs, latency_policy(relative=0.10))
    assert result.status == STATUS_ERROR
    assert result.errored_workloads == ("gemm_tiny",)


def test_suite_rejects_empty_pairs():
    with pytest.raises(ValueError):
        compare_regression_suite([], latency_policy())
