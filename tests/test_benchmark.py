import pytest

from tensorforge_ops.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkResult,
    LatencyStatistics,
    compute_latency_statistics,
    run_timed_iterations,
)


# --- BenchmarkConfig validation --------------------------------------------------

def test_valid_config():
    config = BenchmarkConfig(backend="pytorch", device="cpu", warmup_iterations=10, measured_iterations=50)
    assert config.backend == "pytorch"
    assert config.warmup_iterations == 10


def test_rejects_unknown_backend():
    with pytest.raises(ValueError):
        BenchmarkConfig(backend="tensorflow", device="cpu")


def test_rejects_empty_device():
    with pytest.raises(ValueError):
        BenchmarkConfig(backend="pytorch", device="")


def test_rejects_negative_warmup():
    with pytest.raises(ValueError):
        BenchmarkConfig(backend="pytorch", device="cpu", warmup_iterations=-1)


def test_allows_zero_warmup():
    config = BenchmarkConfig(backend="pytorch", device="cpu", warmup_iterations=0, measured_iterations=5)
    assert config.warmup_iterations == 0


def test_rejects_nonpositive_measured_iterations():
    with pytest.raises(ValueError):
        BenchmarkConfig(backend="pytorch", device="cpu", measured_iterations=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(backend="pytorch", device="cpu", measured_iterations=-5)


# --- latency statistics: known-value tests ---------------------------------------

def test_statistics_known_small_array():
    # [1,2,3,4,5] seconds: p50 = median = 3.0 exactly (linear interpolation,
    # k=(5-1)*0.5=2, an exact rank -> sorted[2]=3).
    stats = compute_latency_statistics([5.0, 1.0, 3.0, 2.0, 4.0])  # unsorted input, must sort internally
    assert stats.count == 5
    assert stats.mean_seconds == 3.0
    assert stats.p50_seconds == 3.0
    assert stats.min_seconds == 1.0
    assert stats.max_seconds == 5.0
    assert stats.throughput_per_second == 1.0 / 3.0
    # Below the 20-sample threshold: p95/p99 must be None, not fabricated.
    assert stats.p95_seconds is None
    assert stats.p99_seconds is None


def test_statistics_p95_known_value_small_array():
    # k = (5-1) * 0.95 = 3.8; f=3, c=4
    # d0 = sorted[3]*(4-3.8) = 4*0.2 = 0.8; d1 = sorted[4]*(3.8-3) = 5*0.8 = 4.0
    # p95 = 4.8 -- but only computed once count >= 20, so use a 20-sample array.
    samples = list(range(1, 21))  # 1..20, count=20
    stats = compute_latency_statistics([float(x) for x in samples])
    assert stats.count == 20
    # k = 19 * 0.95 = 18.05; f=18,c=19; sorted[18]=19, sorted[19]=20
    # p95 = 19*(19-18.05) + 20*(18.05-18) = 19*0.95 + 20*0.05 = 18.05 + 1.0 = 19.05
    assert stats.p95_seconds == pytest.approx(19.05)
    assert stats.p99_seconds is None  # still below the 100-sample threshold


def test_statistics_p99_requires_100_samples():
    samples = [float(x) for x in range(1, 101)]  # 1..100, count=100
    stats = compute_latency_statistics(samples)
    assert stats.count == 100
    # k = 99 * 0.99 = 98.01; f=98,c=99; sorted[98]=99, sorted[99]=100
    # p99 = 99*(99-98.01) + 100*(98.01-98) = 99*0.99 + 100*0.01 = 98.01 + 1.0 = 99.01
    assert stats.p99_seconds == pytest.approx(99.01)
    assert stats.p95_seconds is not None


def test_statistics_single_sample():
    stats = compute_latency_statistics([2.5])
    assert stats.count == 1
    assert stats.mean_seconds == 2.5
    assert stats.p50_seconds == 2.5
    assert stats.min_seconds == stats.max_seconds == 2.5


def test_statistics_rejects_empty_samples():
    with pytest.raises(ValueError):
        compute_latency_statistics([])


# --- warmup exclusion / timed iteration loop --------------------------------------

def test_run_timed_iterations_excludes_warmup_from_samples():
    calls = []

    def fn():
        calls.append("call")

    samples = run_timed_iterations(fn, warmup_iterations=3, measured_iterations=5)

    assert len(calls) == 3 + 5  # warmup + measured both actually executed
    assert len(samples) == 5  # but only measured iterations produce samples
    assert all(s >= 0 for s in samples)


def test_run_timed_iterations_zero_warmup():
    calls = []
    samples = run_timed_iterations(lambda: calls.append(1), warmup_iterations=0, measured_iterations=4)
    assert len(calls) == 4
    assert len(samples) == 4


def test_run_timed_iterations_calls_sync_fn():
    sync_calls = []
    fn_calls = []

    samples = run_timed_iterations(
        lambda: fn_calls.append(1),
        warmup_iterations=2,
        measured_iterations=3,
        sync_fn=lambda: sync_calls.append(1),
    )
    assert len(fn_calls) == 5
    assert len(samples) == 3
    # sync_fn called once per warmup iteration, plus twice per measured
    # iteration (before starting the clock and before stopping it).
    assert len(sync_calls) == 2 + 3 * 2


# --- BenchmarkResult serialization -------------------------------------------------

def test_benchmark_result_to_dict_and_json():
    stats = compute_latency_statistics([0.001, 0.002, 0.0015])
    result = BenchmarkResult(
        core_result_fingerprint="sha256:abc123",
        workload_preset="gemm_tiny",
        workload_kind="gemm",
        backend="pytorch",
        device="cpu",
        dtype="fp32",
        warmup_iterations=2,
        measured_iterations=3,
        latency_samples_seconds=(0.001, 0.002, 0.0015),
        statistics=stats,
        peak_memory_allocated_bytes=None,
        runtime_metadata={"python_torch_version": "2.13.0"},
    )
    data = result.to_dict()
    assert data["benchmark_schema_version"] == BENCHMARK_SCHEMA_VERSION == 1
    assert data["core_result_fingerprint"] == "sha256:abc123"
    assert data["workload"] == {"preset": "gemm_tiny", "kind": "gemm"}
    assert data["measured_peak_memory_allocated_bytes"] is None
    assert data["latency_samples_seconds"] == [0.001, 0.002, 0.0015]

    import json
    parsed = json.loads(result.to_json())
    assert parsed == data
