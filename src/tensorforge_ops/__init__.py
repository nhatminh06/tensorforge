from tensorforge_ops.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkResult,
    LatencyStatistics,
    compute_latency_statistics,
    run_timed_iterations,
)
from tensorforge_ops.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION,
    CalibratedPrediction,
    CalibrationConfig,
    ComputeProbeResult,
    DeviceCalibrationProfile,
    MemoryProbeResult,
    ValidationResult,
    ValidationSummary,
    calibration_profile_from_dict,
    compute_calibration_fingerprint,
    extract_workload_features,
    predict,
    summarize_validation_results,
    validate_prediction,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkConfig",
    "BenchmarkResult",
    "LatencyStatistics",
    "compute_latency_statistics",
    "run_timed_iterations",
    "CALIBRATION_SCHEMA_VERSION",
    "VALIDATION_SCHEMA_VERSION",
    "CalibratedPrediction",
    "CalibrationConfig",
    "ComputeProbeResult",
    "DeviceCalibrationProfile",
    "MemoryProbeResult",
    "ValidationResult",
    "ValidationSummary",
    "calibration_profile_from_dict",
    "compute_calibration_fingerprint",
    "extract_workload_features",
    "predict",
    "summarize_validation_results",
    "validate_prediction",
]

# tracking.py imports mlflow, an optional "ops" extra -- benchmark.py/
# calibration.py above have zero runtime dependencies and must stay
# importable without it.
try:
    from tensorforge_ops.tracking import (
        TrackedRun,
        TrackingConfig,
        compute_result_fingerprint,
        list_runs,
        log_benchmark_result,
        log_calibration_profile,
        log_validation_result,
        log_validation_summary,
        resolve_tracking_uri,
        track_experiment,
        track_result,
    )
except ImportError:
    pass
else:
    __all__ += [
        "TrackedRun",
        "TrackingConfig",
        "compute_result_fingerprint",
        "list_runs",
        "log_benchmark_result",
        "log_calibration_profile",
        "log_validation_result",
        "log_validation_summary",
        "resolve_tracking_uri",
        "track_experiment",
        "track_result",
    ]
