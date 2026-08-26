"""Physical-device calibration and prediction-vs-measurement validation.

Torch-free: this module contains only pure dataclasses, formulas, and
Core-feature extraction. Real probe execution lives in
`calibration_pytorch.py`. Safe to import with no benchmark runtime
installed, exactly like `benchmark.py`.

Modeling boundary (read before changing anything here):

TensorForge Core's PE array is an educational architecture abstraction.
An NVIDIA GPU is not "N TensorForge PEs" -- it has CUDA cores, Tensor
Cores, SMs, register files, shared memory, an L2 cache, DRAM, multiple
clock domains, and vendor-library kernels with their own internal
mapping. This module never invents a PE-count/clock mapping for a real
GPU. Instead it measures two empirical ceilings directly on the actual
runtime/device -- sustained compute throughput and sustained
memory-copy bandwidth -- and uses those, not any Core accelerator
preset's PE geometry, as the basis for a physical-device prediction.

    physical device
        -> compute probe (torch.mm)          -> effective_compute_flops_per_second
        -> memory probe (tensor copy)         -> effective_memory_bandwidth_bytes_per_second
    DeviceCalibrationProfile
        + TensorForge workload's modeled FLOPs / baseline DRAM bytes
    CalibratedPrediction (empirical roofline lower bound)
        vs. BenchmarkResult.statistics.p50_seconds (Milestone 12, real execution)
    ValidationResult (signed/absolute/relative error, APE, ratio)

This is an "empirical roofline prediction" / "device-calibrated
analytical lower bound" -- not a hardware-accurate, cycle-accurate, or
simulated prediction. It is still `max(compute_time, memory_time)` with
no additive overhead term and no fitted correction coefficient; the gap
between this lower bound and real measured latency is itself the
Milestone-13 evidence, not something to be fitted away.

Calibration vs. validation are deliberately separate sets: calibration
uses only the synthetic compute/memory probes below; validation uses
TensorForge workload presets (gemm_tiny, conv_spatial, ...). Effective
rates are never tuned against validation-workload error.
"""

import statistics
from dataclasses import dataclass, field

from tensorforge.convolution import lower_conv2d_to_gemm
from tensorforge.transformer import block_total_flops, derive_transformer_gemms

CALIBRATION_SCHEMA_VERSION = 1
VALIDATION_SCHEMA_VERSION = 1

# Relative tolerance for calling a predicted bottleneck "balanced" instead
# of compute-/memory-bound. Real floating-point compute/memory times are
# essentially never exactly equal; this tolerance exists so the "balanced"
# label is reachable and documented rather than dead code.
_BOTTLENECK_RELATIVE_TOLERANCE = 1e-9


# --- probe rate formulas (pure, shared with calibration_pytorch.py) -------------

def compute_probe_rate(flops: int, p50_seconds: float) -> float:
    """Effective sustained compute rate: FLOPs / p50 latency."""
    return flops / p50_seconds


def memory_probe_traffic_bytes(payload_bytes: int) -> int:
    """One copy of payload_bytes moves payload_bytes read + payload_bytes
    written -- modeled traffic is 2x the payload."""
    return 2 * payload_bytes


def memory_probe_bandwidth(payload_bytes: int, p50_seconds: float) -> float:
    """Effective sustained memory-copy bandwidth: modeled copy traffic / p50 latency."""
    return memory_probe_traffic_bytes(payload_bytes) / p50_seconds


@dataclass(frozen=True)
class CalibrationConfig:
    backend: str
    device: str
    dtype: str
    compute_m: int = 2048
    compute_n: int = 2048
    compute_k: int = 2048
    memory_probe_mib: int = 64
    warmup_iterations: int = 10
    measured_iterations: int = 50

    def __post_init__(self) -> None:
        if self.backend != "pytorch":
            raise ValueError(f"backend must be 'pytorch', got {self.backend!r}")
        if not self.device:
            raise ValueError("device must not be empty")
        if self.dtype not in ("fp16", "fp32"):
            raise ValueError(f"dtype must be 'fp16' or 'fp32', got {self.dtype!r}")
        for name, value in (
            ("compute_m", self.compute_m), ("compute_n", self.compute_n), ("compute_k", self.compute_k),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be an int > 0, got {value!r}")
        if not isinstance(self.memory_probe_mib, int) or isinstance(self.memory_probe_mib, bool) or self.memory_probe_mib <= 0:
            raise ValueError(f"memory_probe_mib must be an int > 0, got {self.memory_probe_mib!r}")
        if not isinstance(self.warmup_iterations, int) or self.warmup_iterations < 0:
            raise ValueError(f"warmup_iterations must be an int >= 0, got {self.warmup_iterations!r}")
        if not isinstance(self.measured_iterations, int) or self.measured_iterations <= 0:
            raise ValueError(f"measured_iterations must be an int > 0, got {self.measured_iterations!r}")


@dataclass(frozen=True)
class ComputeProbeResult:
    """Sustained compute rate from a dense M x N x K GEMM.

    Uses p50 latency (steady-state central measurement, less sensitive to
    outliers than min/max) to derive a "measured sustained compute rate" --
    never called GPU peak FLOPs, since this is one probe shape/dtype/
    runtime, not a vendor peak-throughput claim.
    """

    shape_m: int
    shape_n: int
    shape_k: int
    flops: int
    latency_samples_seconds: tuple[float, ...]
    p50_seconds: float
    effective_compute_flops_per_second: float

    def to_dict(self) -> dict:
        return {
            "shape_m": self.shape_m, "shape_n": self.shape_n, "shape_k": self.shape_k,
            "flops": self.flops,
            "latency_samples_seconds": list(self.latency_samples_seconds),
            "p50_seconds": self.p50_seconds,
            "effective_compute_flops_per_second": self.effective_compute_flops_per_second,
        }


@dataclass(frozen=True)
class MemoryProbeResult:
    """Sustained device-memory-copy bandwidth from `dst.copy_(src)`.

    One copy of an X-byte tensor logically moves X bytes read + X bytes
    written, so modeled traffic is 2 * payload_bytes. This is an empirical
    memory-throughput proxy (kernel/runtime/cache effects included), not a
    measurement of raw DRAM hardware peak bandwidth -- called "effective
    sustained memory-copy bandwidth" everywhere, never "peak bandwidth".
    """

    payload_bytes: int
    modeled_copy_traffic_bytes: int
    latency_samples_seconds: tuple[float, ...]
    p50_seconds: float
    effective_memory_bandwidth_bytes_per_second: float

    def to_dict(self) -> dict:
        return {
            "payload_bytes": self.payload_bytes,
            "modeled_copy_traffic_bytes": self.modeled_copy_traffic_bytes,
            "latency_samples_seconds": list(self.latency_samples_seconds),
            "p50_seconds": self.p50_seconds,
            "effective_memory_bandwidth_bytes_per_second": self.effective_memory_bandwidth_bytes_per_second,
        }


def compute_probe_from_dict(d: dict) -> ComputeProbeResult:
    return ComputeProbeResult(
        shape_m=d["shape_m"], shape_n=d["shape_n"], shape_k=d["shape_k"], flops=d["flops"],
        latency_samples_seconds=tuple(d["latency_samples_seconds"]),
        p50_seconds=d["p50_seconds"],
        effective_compute_flops_per_second=d["effective_compute_flops_per_second"],
    )


def memory_probe_from_dict(d: dict) -> MemoryProbeResult:
    return MemoryProbeResult(
        payload_bytes=d["payload_bytes"],
        modeled_copy_traffic_bytes=d["modeled_copy_traffic_bytes"],
        latency_samples_seconds=tuple(d["latency_samples_seconds"]),
        p50_seconds=d["p50_seconds"],
        effective_memory_bandwidth_bytes_per_second=d["effective_memory_bandwidth_bytes_per_second"],
    )


@dataclass(frozen=True)
class DeviceCalibrationProfile:
    """One empirical calibration result for one specific backend/device/
    dtype/runtime combination. Unlike Core's experiment fingerprint,
    re-running calibration on the same config can legitimately produce a
    different profile (and therefore a different fingerprint) because
    measured rates vary run to run -- this is expected, not a bug.
    """

    calibration_schema_version: int
    backend: str
    device_type: str
    device_index: int | None
    device_name: str | None
    dtype: str

    runtime_metadata: dict
    device_metadata: dict

    compute_probe: ComputeProbeResult
    memory_probe: MemoryProbeResult

    effective_compute_flops_per_second: float
    effective_memory_bandwidth_bytes_per_second: float

    def device_signature(self) -> dict:
        """Canonical, deterministic identity used to decide whether a
        benchmark's device/runtime is the one this profile was measured
        on. Deliberately excludes username/hostname/paths/env vars/Git
        state -- none of those are physical-device or runtime identity.
        """
        return {
            "backend": self.backend,
            "device_type": self.device_type,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "dtype": self.dtype,
            "torch_version": self.runtime_metadata.get("torch_version"),
            "cuda_runtime_version": self.runtime_metadata.get("cuda_runtime_version"),
            "compute_capability": self.device_metadata.get("compute_capability"),
            "total_device_memory_bytes": self.device_metadata.get("total_device_memory_bytes"),
        }

    def to_dict(self) -> dict:
        return {
            "calibration_schema_version": self.calibration_schema_version,
            "backend": self.backend,
            "device_type": self.device_type,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "dtype": self.dtype,
            "runtime_metadata": dict(self.runtime_metadata),
            "device_metadata": dict(self.device_metadata),
            "compute_probe": self.compute_probe.to_dict(),
            "memory_probe": self.memory_probe.to_dict(),
            "effective_compute_flops_per_second": self.effective_compute_flops_per_second,
            "effective_memory_bandwidth_bytes_per_second": self.effective_memory_bandwidth_bytes_per_second,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def calibration_profile_from_dict(d: dict) -> DeviceCalibrationProfile:
    return DeviceCalibrationProfile(
        calibration_schema_version=d["calibration_schema_version"],
        backend=d["backend"],
        device_type=d["device_type"],
        device_index=d["device_index"],
        device_name=d["device_name"],
        dtype=d["dtype"],
        runtime_metadata=dict(d["runtime_metadata"]),
        device_metadata=dict(d["device_metadata"]),
        compute_probe=compute_probe_from_dict(d["compute_probe"]),
        memory_probe=memory_probe_from_dict(d["memory_probe"]),
        effective_compute_flops_per_second=d["effective_compute_flops_per_second"],
        effective_memory_bandwidth_bytes_per_second=d["effective_memory_bandwidth_bytes_per_second"],
    )


def compute_calibration_fingerprint(profile: DeviceCalibrationProfile) -> str:
    import hashlib

    digest = hashlib.sha256(profile.to_json().encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# --- Core feature extraction: modeled FLOPs + baseline DRAM bytes ---------------
#
# Deliberately NOT the same bytes as ExperimentResult.primary_metrics
# ["total_dram_bytes"], which reflects one specific tile/schedule mapping.
# cuBLAS/cuDNN/PyTorch choose their own kernels and traffic pattern, so a
# physical-device prediction uses the simpler untiled baseline (read each
# operand once, write the output once) already exposed by Core's own
# Gemm/Conv2D/Transformer objects -- never re-derived here.

_SUPPORTED_FEATURE_KINDS = ("gemm", "conv2d", "transformer")


def extract_workload_features(preset) -> tuple[int, int]:
    """Return (modeled_flops, modeled_baseline_dram_bytes) for one workload
    preset, built only from existing Core properties/functions.
    """
    if preset.kind == "gemm":
        gemm = preset.gemm()
        return gemm.flops, gemm.dram_bytes
    if preset.kind == "conv2d":
        lowering = lower_conv2d_to_gemm(preset.spec())
        return lowering.gemm.flops, lowering.gemm.dram_bytes
    if preset.kind == "transformer":
        ops = derive_transformer_gemms(preset.spec())
        flops = block_total_flops(ops)
        baseline_bytes = sum(op.gemm.dram_bytes * op.repetitions for op in ops)
        return flops, baseline_bytes
    raise ValueError(
        f"calibrated prediction does not support workload kind {preset.kind!r} yet "
        f"(supported: {', '.join(_SUPPORTED_FEATURE_KINDS)})"
    )


@dataclass(frozen=True)
class CalibratedPrediction:
    """Empirical roofline prediction: max(compute_time, memory_time) using
    a real device's measured compute/memory ceilings -- not a hardware-
    accurate, cycle-accurate, or simulated result.
    """

    workload_preset: str
    workload_kind: str

    modeled_flops: int
    modeled_baseline_dram_bytes: int

    effective_compute_flops_per_second: float
    effective_memory_bandwidth_bytes_per_second: float

    predicted_compute_seconds: float
    predicted_memory_seconds: float
    predicted_latency_seconds: float
    predicted_bottleneck: str

    calibration_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "workload_preset": self.workload_preset,
            "workload_kind": self.workload_kind,
            "modeled_flops": self.modeled_flops,
            "modeled_baseline_dram_bytes": self.modeled_baseline_dram_bytes,
            "effective_compute_flops_per_second": self.effective_compute_flops_per_second,
            "effective_memory_bandwidth_bytes_per_second": self.effective_memory_bandwidth_bytes_per_second,
            "predicted_compute_seconds": self.predicted_compute_seconds,
            "predicted_memory_seconds": self.predicted_memory_seconds,
            "predicted_latency_seconds": self.predicted_latency_seconds,
            "predicted_bottleneck": self.predicted_bottleneck,
            "calibration_fingerprint": self.calibration_fingerprint,
        }


def predict(preset, profile: DeviceCalibrationProfile) -> CalibratedPrediction:
    """empirical_roofline_seconds = max(flops / compute_rate, bytes / bandwidth).

    No additive overhead term, no fitted correction coefficient. The
    "predicted_bottleneck" here is which analytical term is larger --
    it is never a claim about which resource the real, measured execution
    was actually bound by (see validate_prediction's docstring).
    """
    flops, baseline_bytes = extract_workload_features(preset)
    compute_seconds = flops / profile.effective_compute_flops_per_second
    memory_seconds = baseline_bytes / profile.effective_memory_bandwidth_bytes_per_second
    latency_seconds = max(compute_seconds, memory_seconds)

    larger = max(compute_seconds, memory_seconds)
    if larger > 0 and abs(compute_seconds - memory_seconds) / larger <= _BOTTLENECK_RELATIVE_TOLERANCE:
        bottleneck = "balanced"
    elif compute_seconds > memory_seconds:
        bottleneck = "compute-bound"
    else:
        bottleneck = "memory-bound"

    return CalibratedPrediction(
        workload_preset=preset.name,
        workload_kind=preset.kind,
        modeled_flops=flops,
        modeled_baseline_dram_bytes=baseline_bytes,
        effective_compute_flops_per_second=profile.effective_compute_flops_per_second,
        effective_memory_bandwidth_bytes_per_second=profile.effective_memory_bandwidth_bytes_per_second,
        predicted_compute_seconds=compute_seconds,
        predicted_memory_seconds=memory_seconds,
        predicted_latency_seconds=latency_seconds,
        predicted_bottleneck=bottleneck,
        calibration_fingerprint=compute_calibration_fingerprint(profile),
    )


# --- device/runtime match check --------------------------------------------------

def _device_mismatch_reason(profile: DeviceCalibrationProfile, benchmark_result) -> str | None:
    if profile.backend != benchmark_result.backend:
        return f"backend mismatch: calibration={profile.backend!r} benchmark={benchmark_result.backend!r}"

    benchmark_device_type = benchmark_result.device.split(":")[0]
    if profile.device_type != benchmark_device_type:
        return f"device type mismatch: calibration={profile.device_type!r} benchmark={benchmark_device_type!r}"

    if profile.dtype != benchmark_result.dtype:
        return f"dtype mismatch: calibration={profile.dtype!r} benchmark={benchmark_result.dtype!r}"

    profile_torch = profile.runtime_metadata.get("torch_version")
    benchmark_torch = benchmark_result.runtime_metadata.get("torch_version")
    if profile_torch != benchmark_torch:
        return f"PyTorch runtime mismatch: calibration={profile_torch!r} benchmark={benchmark_torch!r}"

    if profile.device_type == "cuda":
        for key in ("device_index", "device_name", "cuda_runtime_version"):
            profile_value = profile.runtime_metadata.get(key)
            benchmark_value = benchmark_result.runtime_metadata.get(key)
            if profile_value != benchmark_value:
                return (
                    f"CUDA device/runtime mismatch on {key}: "
                    f"calibration={profile_value!r} benchmark={benchmark_value!r}"
                )

    return None


@dataclass(frozen=True)
class ValidationResult:
    core_result_fingerprint: str
    calibration_fingerprint: str

    workload_preset: str
    workload_kind: str
    backend: str
    device: str
    dtype: str

    predicted_compute_seconds: float
    predicted_memory_seconds: float
    predicted_latency_seconds: float
    predicted_bottleneck: str

    measured_p50_latency_seconds: float
    measured_mean_latency_seconds: float
    measured_p95_latency_seconds: float | None

    signed_error_seconds: float
    absolute_error_seconds: float
    relative_error: float
    absolute_percentage_error: float
    measured_to_predicted_ratio: float

    def to_dict(self) -> dict:
        return {
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
            "core_result_fingerprint": self.core_result_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "workload": {"preset": self.workload_preset, "kind": self.workload_kind},
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "prediction": {
                "compute_seconds": self.predicted_compute_seconds,
                "memory_seconds": self.predicted_memory_seconds,
                "latency_seconds": self.predicted_latency_seconds,
                "bottleneck": self.predicted_bottleneck,
            },
            "measurement": {
                "p50_latency_seconds": self.measured_p50_latency_seconds,
                "mean_latency_seconds": self.measured_mean_latency_seconds,
                "p95_latency_seconds": self.measured_p95_latency_seconds,
            },
            "error": {
                "signed_error_seconds": self.signed_error_seconds,
                "absolute_error_seconds": self.absolute_error_seconds,
                "relative_error": self.relative_error,
                "absolute_percentage_error": self.absolute_percentage_error,
                "measured_to_predicted_ratio": self.measured_to_predicted_ratio,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def validation_result_from_dict(d: dict) -> ValidationResult:
    workload = d["workload"]
    prediction = d["prediction"]
    measurement = d["measurement"]
    error = d["error"]
    return ValidationResult(
        core_result_fingerprint=d["core_result_fingerprint"],
        calibration_fingerprint=d["calibration_fingerprint"],
        workload_preset=workload["preset"],
        workload_kind=workload["kind"],
        backend=d["backend"],
        device=d["device"],
        dtype=d["dtype"],
        predicted_compute_seconds=prediction["compute_seconds"],
        predicted_memory_seconds=prediction["memory_seconds"],
        predicted_latency_seconds=prediction["latency_seconds"],
        predicted_bottleneck=prediction["bottleneck"],
        measured_p50_latency_seconds=measurement["p50_latency_seconds"],
        measured_mean_latency_seconds=measurement["mean_latency_seconds"],
        measured_p95_latency_seconds=measurement["p95_latency_seconds"],
        signed_error_seconds=error["signed_error_seconds"],
        absolute_error_seconds=error["absolute_error_seconds"],
        relative_error=error["relative_error"],
        absolute_percentage_error=error["absolute_percentage_error"],
        measured_to_predicted_ratio=error["measured_to_predicted_ratio"],
    )


def load_validation_result(path: str) -> ValidationResult:
    import json

    with open(path, encoding="utf-8") as f:
        return validation_result_from_dict(json.load(f))


def validate_prediction(
    profile: DeviceCalibrationProfile,
    prediction: CalibratedPrediction,
    core_result_fingerprint: str,
    benchmark_result,
) -> ValidationResult:
    """Pure comparison: predicted (empirical roofline) vs. measured p50.

    error sign convention: signed_error = predicted - measured.
        negative -> prediction underestimates real runtime
        positive -> prediction overestimates real runtime

    measured_to_predicted_ratio = measured / predicted.
        1.0 -> exact match, >1 -> measured slower than predicted,
        <1 -> measured faster than predicted.

    predicted_bottleneck is copied from `prediction` unchanged -- it is an
    analytical classification of which term in the roofline formula is
    larger, NOT a claim about which resource real execution was bound by.
    Latency measurements alone cannot prove actual GPU compute/memory
    boundedness; that requires device telemetry (a later milestone).

    Never reruns a benchmark -- both `benchmark_result` and `prediction`
    must already be computed. Raises ValueError if the calibration
    profile's backend/device/dtype/runtime does not match the benchmark's.
    """
    mismatch = _device_mismatch_reason(profile, benchmark_result)
    if mismatch is not None:
        raise ValueError(
            "calibration profile does not match this benchmark's device/runtime, "
            f"cannot validate: {mismatch}"
        )

    stats = benchmark_result.statistics
    measured_p50 = stats.p50_seconds
    predicted = prediction.predicted_latency_seconds

    signed_error = predicted - measured_p50
    absolute_error = abs(signed_error)
    relative_error = signed_error / measured_p50
    ape = abs(relative_error) * 100.0
    ratio = measured_p50 / predicted

    return ValidationResult(
        core_result_fingerprint=core_result_fingerprint,
        calibration_fingerprint=prediction.calibration_fingerprint,
        workload_preset=prediction.workload_preset,
        workload_kind=prediction.workload_kind,
        backend=benchmark_result.backend,
        device=benchmark_result.device,
        dtype=benchmark_result.dtype,
        predicted_compute_seconds=prediction.predicted_compute_seconds,
        predicted_memory_seconds=prediction.predicted_memory_seconds,
        predicted_latency_seconds=prediction.predicted_latency_seconds,
        predicted_bottleneck=prediction.predicted_bottleneck,
        measured_p50_latency_seconds=measured_p50,
        measured_mean_latency_seconds=stats.mean_seconds,
        measured_p95_latency_seconds=stats.p95_seconds,
        signed_error_seconds=signed_error,
        absolute_error_seconds=absolute_error,
        relative_error=relative_error,
        absolute_percentage_error=ape,
        measured_to_predicted_ratio=ratio,
    )


# --- validation summary ------------------------------------------------------

def _group_stats(results: tuple[ValidationResult, ...]) -> dict:
    apes = [r.absolute_percentage_error for r in results]
    ratios = [r.measured_to_predicted_ratio for r in results]
    return {
        "count": len(results),
        "mean_absolute_percentage_error": statistics.fmean(apes),
        "median_absolute_percentage_error": statistics.median(apes),
        "median_measured_to_predicted_ratio": statistics.median(ratios),
    }


@dataclass(frozen=True)
class ValidationSummary:
    results: tuple[ValidationResult, ...]
    count: int
    mean_absolute_percentage_error: float
    median_absolute_percentage_error: float
    max_absolute_percentage_error: float
    median_measured_to_predicted_ratio: float
    by_workload_kind: dict = field(default_factory=dict)
    by_predicted_bottleneck: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
            "count": self.count,
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error,
            "median_absolute_percentage_error": self.median_absolute_percentage_error,
            "max_absolute_percentage_error": self.max_absolute_percentage_error,
            "median_measured_to_predicted_ratio": self.median_measured_to_predicted_ratio,
            "by_workload_kind": self.by_workload_kind,
            "by_predicted_bottleneck": self.by_predicted_bottleneck,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def summarize_validation_results(results: tuple[ValidationResult, ...]) -> ValidationSummary:
    """No pass/fail threshold -- this reports model error, it does not
    grade it. Grouping by workload kind and by predicted (not measured)
    bottleneck class is the first structured statement about where the
    simplified Core model tracks real hardware better or worse.
    """
    if len(results) == 0:
        raise ValueError("results must not be empty")

    apes = [r.absolute_percentage_error for r in results]
    ratios = [r.measured_to_predicted_ratio for r in results]

    by_kind: dict[str, list[ValidationResult]] = {}
    by_bottleneck: dict[str, list[ValidationResult]] = {}
    for r in results:
        by_kind.setdefault(r.workload_kind, []).append(r)
        by_bottleneck.setdefault(r.predicted_bottleneck, []).append(r)

    return ValidationSummary(
        results=results,
        count=len(results),
        mean_absolute_percentage_error=statistics.fmean(apes),
        median_absolute_percentage_error=statistics.median(apes),
        max_absolute_percentage_error=max(apes),
        median_measured_to_predicted_ratio=statistics.median(ratios),
        by_workload_kind={k: _group_stats(tuple(v)) for k, v in by_kind.items()},
        by_predicted_bottleneck={k: _group_stats(tuple(v)) for k, v in by_bottleneck.items()},
    )
