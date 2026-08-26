"""Provider-independent GPU telemetry data types, summaries, and
baseline-vs-candidate correlation.

Torch-free and MLflow-free: this module holds only pure dataclasses and
arithmetic. Live collection lives in `telemetry_nvml.py` (imports
`pynvml`, imported lazily by the CLI so telemetry stays optional).

Telemetry is diagnostic evidence, never a gate. It never changes a
RegressionResult's PASS/FAIL/ERROR status (Milestone 14), and it is
never merged into BenchmarkResult -- it is its own artifact/schema
(`telemetry-trace.json`, `telemetry-summary.json`,
`telemetry-correlation.json`).

Latency-vs-telemetry separation: TensorForge's latency measurement
(Milestone 12, `benchmark_pytorch.run_pytorch_benchmark`) is untouched by
this module and never runs a telemetry sampler concurrently with its
timed loop -- background sampling threads/NVML calls could perturb
timing. Telemetry is collected in a SEPARATE phase that repeatedly
re-executes the same prepared workload for several seconds while a
sampler thread polls the GPU concurrently (see `telemetry_nvml.py`).
This trades exact per-invocation timing for a telemetry window long
enough that NVML's own internal sampling period (which is itself several
hundred milliseconds to seconds, far longer than many TensorForge
microbenchmarks) can produce meaningful values.

Metric semantics (read before adding a new metric or renaming one):

- `gpu_utilization_percent`: NVML's coarse "percentage of the recent
  sampling period during which one or more kernels were executing."
  This is NOT SM occupancy, NOT SM saturation, NOT compute utilization
  in a profiler sense -- it only reports "busy at all" vs. "idle".
- `memory_activity_percent`: NVML's coarse device-memory read/write
  activity fraction over the same kind of sampling period. This is NOT
  DRAM bandwidth utilization (bytes/sec relative to peak bandwidth) --
  NVML's basic API does not report that. A metric may only be labeled
  "*_bandwidth_utilization_percent" if it actually comes from an API
  that reports bandwidth utilization (e.g. NVML GPM DRAM-activity, or a
  DCGM profiling field) -- and even then only when runtime capability
  discovery proves it is supported on this device.
- `memory_used_bytes`/`memory_free_bytes`/`memory_total_bytes`: DEVICE
  level, may include other processes sharing the GPU. Not the same
  thing as PyTorch's per-process peak allocator memory
  (`BenchmarkResult.peak_memory_allocated_bytes`).
- Advanced fields (`sm_activity_percent`, `sm_occupancy_percent`,
  `tensor_activity_percent`, `dram_bandwidth_utilization_percent`) come
  from NVML's newer GPM (GPU Performance Monitoring) API, which is
  proven at runtime to be supported or not (see `telemetry_nvml.py`) --
  never assumed available on a given GPU. Unsupported means `None`,
  never `0`.
- `clock_event_reasons`: decoded NVML clocks-event-reason bit names.
  Their presence is evidence a reason bit was set, not proof of the
  performance impact of that reason; see the diagnostic-language rules
  in `correlate_telemetry()`'s docstring.
"""

import statistics
from dataclasses import dataclass, field

TELEMETRY_SCHEMA_VERSION = 1
TELEMETRY_CORRELATION_SCHEMA_VERSION = 1

_KNOWN_BACKENDS = ("nvml", "dcgm-exporter")


@dataclass(frozen=True)
class TelemetryConfig:
    backend: str
    device_index: int
    sample_interval_seconds: float = 0.15
    telemetry_duration_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.backend not in _KNOWN_BACKENDS:
            raise ValueError(f"backend must be one of {_KNOWN_BACKENDS}, got {self.backend!r}")
        if not isinstance(self.device_index, int) or isinstance(self.device_index, bool) or self.device_index < 0:
            raise ValueError(f"device_index must be an int >= 0, got {self.device_index!r}")
        if not isinstance(self.sample_interval_seconds, (int, float)) or self.sample_interval_seconds <= 0:
            raise ValueError(f"sample_interval_seconds must be > 0, got {self.sample_interval_seconds!r}")
        if not isinstance(self.telemetry_duration_seconds, (int, float)) or self.telemetry_duration_seconds <= 0:
            raise ValueError(f"telemetry_duration_seconds must be > 0, got {self.telemetry_duration_seconds!r}")


@dataclass(frozen=True)
class TelemetrySample:
    """One point-in-time GPU telemetry reading. `timestamp_seconds` is
    monotonic time elapsed since sampling started (time.monotonic()-based
    in telemetry_nvml.py), not wall-clock time -- correlation never needs
    wall-clock alignment.

    Every field is `None`, never `0`, when the underlying provider does
    not support or could not read that value for this sample.
    """

    timestamp_seconds: float

    gpu_utilization_percent: float | None
    memory_activity_percent: float | None

    memory_used_bytes: int | None
    memory_free_bytes: int | None
    memory_total_bytes: int | None

    power_watts: float | None
    temperature_celsius: float | None

    sm_clock_mhz: int | None
    memory_clock_mhz: int | None

    performance_state: int | None
    clock_event_reasons_bitmask: int | None
    clock_event_reasons: tuple = ()

    sm_activity_percent: float | None = None
    sm_occupancy_percent: float | None = None
    tensor_activity_percent: float | None = None
    dram_bandwidth_utilization_percent: float | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_activity_percent": self.memory_activity_percent,
            "memory_used_bytes": self.memory_used_bytes,
            "memory_free_bytes": self.memory_free_bytes,
            "memory_total_bytes": self.memory_total_bytes,
            "power_watts": self.power_watts,
            "temperature_celsius": self.temperature_celsius,
            "sm_clock_mhz": self.sm_clock_mhz,
            "memory_clock_mhz": self.memory_clock_mhz,
            "performance_state": self.performance_state,
            "clock_event_reasons_bitmask": self.clock_event_reasons_bitmask,
            "clock_event_reasons": list(self.clock_event_reasons),
            "sm_activity_percent": self.sm_activity_percent,
            "sm_occupancy_percent": self.sm_occupancy_percent,
            "tensor_activity_percent": self.tensor_activity_percent,
            "dram_bandwidth_utilization_percent": self.dram_bandwidth_utilization_percent,
        }


def telemetry_sample_from_dict(d: dict) -> TelemetrySample:
    return TelemetrySample(
        timestamp_seconds=d["timestamp_seconds"],
        gpu_utilization_percent=d["gpu_utilization_percent"],
        memory_activity_percent=d["memory_activity_percent"],
        memory_used_bytes=d["memory_used_bytes"],
        memory_free_bytes=d["memory_free_bytes"],
        memory_total_bytes=d["memory_total_bytes"],
        power_watts=d["power_watts"],
        temperature_celsius=d["temperature_celsius"],
        sm_clock_mhz=d["sm_clock_mhz"],
        memory_clock_mhz=d["memory_clock_mhz"],
        performance_state=d["performance_state"],
        clock_event_reasons_bitmask=d["clock_event_reasons_bitmask"],
        clock_event_reasons=tuple(d.get("clock_event_reasons", ())),
        sm_activity_percent=d.get("sm_activity_percent"),
        sm_occupancy_percent=d.get("sm_occupancy_percent"),
        tensor_activity_percent=d.get("tensor_activity_percent"),
        dram_bandwidth_utilization_percent=d.get("dram_bandwidth_utilization_percent"),
    )


@dataclass(frozen=True)
class TelemetryTrace:
    telemetry_schema_version: int
    backend: str
    device_index: int
    device_name: str | None
    sample_interval_seconds: float
    requested_duration_seconds: float
    actual_duration_seconds: float
    samples: tuple

    workload_preset: str | None = None
    workload_kind: str | None = None
    benchmark_backend: str | None = None
    dtype: str | None = None
    core_result_fingerprint: str | None = None
    runtime_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "telemetry_schema_version": self.telemetry_schema_version,
            "backend": self.backend,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "sample_interval_seconds": self.sample_interval_seconds,
            "requested_duration_seconds": self.requested_duration_seconds,
            "actual_duration_seconds": self.actual_duration_seconds,
            "workload": {
                "preset": self.workload_preset, "kind": self.workload_kind,
                "backend": self.benchmark_backend, "dtype": self.dtype,
            },
            "core_result_fingerprint": self.core_result_fingerprint,
            "runtime_metadata": dict(self.runtime_metadata),
            "samples": [s.to_dict() for s in self.samples],
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def telemetry_trace_from_dict(d: dict) -> TelemetryTrace:
    schema_version = d.get("telemetry_schema_version")
    if schema_version != TELEMETRY_SCHEMA_VERSION:
        raise ValueError(
            f"telemetry schema version mismatch: expected {TELEMETRY_SCHEMA_VERSION}, got {schema_version!r}"
        )
    workload = d.get("workload", {})
    return TelemetryTrace(
        telemetry_schema_version=schema_version,
        backend=d["backend"], device_index=d["device_index"], device_name=d.get("device_name"),
        sample_interval_seconds=d["sample_interval_seconds"],
        requested_duration_seconds=d["requested_duration_seconds"],
        actual_duration_seconds=d["actual_duration_seconds"],
        samples=tuple(telemetry_sample_from_dict(s) for s in d["samples"]),
        workload_preset=workload.get("preset"), workload_kind=workload.get("kind"),
        benchmark_backend=workload.get("backend"), dtype=workload.get("dtype"),
        core_result_fingerprint=d.get("core_result_fingerprint"),
        runtime_metadata=dict(d.get("runtime_metadata", {})),
    )


def load_telemetry_trace(path: str) -> TelemetryTrace:
    import json

    with open(path, encoding="utf-8") as f:
        return telemetry_trace_from_dict(json.load(f))


# --- summary ------------------------------------------------------------------

@dataclass(frozen=True)
class MetricStat:
    """mean/min/max over whatever samples actually had a non-None value
    for this metric. All fields None (sample_count=0) when the metric
    was unavailable in every sample -- never coerced to 0.
    """

    sample_count: int
    mean: float | None
    min: float | None
    max: float | None

    def to_dict(self) -> dict:
        return {"sample_count": self.sample_count, "mean": self.mean, "min": self.min, "max": self.max}


def _metric_stat_from_dict(d: dict) -> MetricStat:
    return MetricStat(sample_count=d["sample_count"], mean=d["mean"], min=d["min"], max=d["max"])


def _stat(values: list) -> MetricStat:
    present = [v for v in values if v is not None]
    if not present:
        return MetricStat(sample_count=0, mean=None, min=None, max=None)
    return MetricStat(sample_count=len(present), mean=statistics.fmean(present), min=min(present), max=max(present))


_SUMMARY_METRICS = (
    "gpu_utilization_percent", "memory_activity_percent",
    "memory_used_bytes", "power_watts", "temperature_celsius",
    "sm_clock_mhz", "memory_clock_mhz",
    "sm_activity_percent", "sm_occupancy_percent",
    "tensor_activity_percent", "dram_bandwidth_utilization_percent",
)


@dataclass(frozen=True)
class TelemetrySummary:
    telemetry_schema_version: int
    sample_count: int
    duration_seconds: float

    gpu_utilization_percent: MetricStat
    memory_activity_percent: MetricStat
    memory_used_bytes: MetricStat
    memory_total_bytes: int | None
    power_watts: MetricStat
    temperature_celsius: MetricStat
    sm_clock_mhz: MetricStat
    memory_clock_mhz: MetricStat

    sm_activity_percent: MetricStat
    sm_occupancy_percent: MetricStat
    tensor_activity_percent: MetricStat
    dram_bandwidth_utilization_percent: MetricStat

    available_metrics: tuple
    unavailable_metrics: tuple
    throttle_reasons_observed: tuple

    def to_dict(self) -> dict:
        return {
            "telemetry_schema_version": self.telemetry_schema_version,
            "sample_count": self.sample_count,
            "duration_seconds": self.duration_seconds,
            "gpu_utilization_percent": self.gpu_utilization_percent.to_dict(),
            "memory_activity_percent": self.memory_activity_percent.to_dict(),
            "memory_used_bytes": self.memory_used_bytes.to_dict(),
            "memory_total_bytes": self.memory_total_bytes,
            "power_watts": self.power_watts.to_dict(),
            "temperature_celsius": self.temperature_celsius.to_dict(),
            "sm_clock_mhz": self.sm_clock_mhz.to_dict(),
            "memory_clock_mhz": self.memory_clock_mhz.to_dict(),
            "sm_activity_percent": self.sm_activity_percent.to_dict(),
            "sm_occupancy_percent": self.sm_occupancy_percent.to_dict(),
            "tensor_activity_percent": self.tensor_activity_percent.to_dict(),
            "dram_bandwidth_utilization_percent": self.dram_bandwidth_utilization_percent.to_dict(),
            "available_metrics": list(self.available_metrics),
            "unavailable_metrics": list(self.unavailable_metrics),
            "throttle_reasons_observed": list(self.throttle_reasons_observed),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def telemetry_summary_from_dict(d: dict) -> TelemetrySummary:
    schema_version = d.get("telemetry_schema_version")
    if schema_version != TELEMETRY_SCHEMA_VERSION:
        raise ValueError(
            f"telemetry schema version mismatch: expected {TELEMETRY_SCHEMA_VERSION}, got {schema_version!r}"
        )
    return TelemetrySummary(
        telemetry_schema_version=schema_version,
        sample_count=d["sample_count"], duration_seconds=d["duration_seconds"],
        gpu_utilization_percent=_metric_stat_from_dict(d["gpu_utilization_percent"]),
        memory_activity_percent=_metric_stat_from_dict(d["memory_activity_percent"]),
        memory_used_bytes=_metric_stat_from_dict(d["memory_used_bytes"]),
        memory_total_bytes=d["memory_total_bytes"],
        power_watts=_metric_stat_from_dict(d["power_watts"]),
        temperature_celsius=_metric_stat_from_dict(d["temperature_celsius"]),
        sm_clock_mhz=_metric_stat_from_dict(d["sm_clock_mhz"]),
        memory_clock_mhz=_metric_stat_from_dict(d["memory_clock_mhz"]),
        sm_activity_percent=_metric_stat_from_dict(d["sm_activity_percent"]),
        sm_occupancy_percent=_metric_stat_from_dict(d["sm_occupancy_percent"]),
        tensor_activity_percent=_metric_stat_from_dict(d["tensor_activity_percent"]),
        dram_bandwidth_utilization_percent=_metric_stat_from_dict(d["dram_bandwidth_utilization_percent"]),
        available_metrics=tuple(d["available_metrics"]),
        unavailable_metrics=tuple(d["unavailable_metrics"]),
        throttle_reasons_observed=tuple(d["throttle_reasons_observed"]),
    )


def load_telemetry_summary(path: str) -> TelemetrySummary:
    import json

    with open(path, encoding="utf-8") as f:
        return telemetry_summary_from_dict(json.load(f))


def summarize_telemetry_trace(trace: TelemetryTrace) -> TelemetrySummary:
    """Pure, deterministic summary over a fixed TelemetryTrace. A metric
    with zero non-None samples across the whole trace gets an
    all-None MetricStat (never 0) and is listed in `unavailable_metrics`
    rather than `available_metrics`.
    """
    samples = trace.samples
    if not samples:
        raise ValueError("trace must contain at least one sample")

    stats = {name: _stat([getattr(s, name) for s in samples]) for name in _SUMMARY_METRICS}

    memory_total_values = [s.memory_total_bytes for s in samples if s.memory_total_bytes is not None]
    memory_total_bytes = memory_total_values[0] if memory_total_values else None

    available = tuple(name for name in _SUMMARY_METRICS if stats[name].sample_count > 0)
    unavailable = tuple(name for name in _SUMMARY_METRICS if stats[name].sample_count == 0)

    throttle_reasons = set()
    for s in samples:
        throttle_reasons.update(s.clock_event_reasons)
    throttle_reasons_observed = tuple(sorted(throttle_reasons))

    return TelemetrySummary(
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        sample_count=len(samples),
        duration_seconds=trace.actual_duration_seconds,
        gpu_utilization_percent=stats["gpu_utilization_percent"],
        memory_activity_percent=stats["memory_activity_percent"],
        memory_used_bytes=stats["memory_used_bytes"],
        memory_total_bytes=memory_total_bytes,
        power_watts=stats["power_watts"],
        temperature_celsius=stats["temperature_celsius"],
        sm_clock_mhz=stats["sm_clock_mhz"],
        memory_clock_mhz=stats["memory_clock_mhz"],
        sm_activity_percent=stats["sm_activity_percent"],
        sm_occupancy_percent=stats["sm_occupancy_percent"],
        tensor_activity_percent=stats["tensor_activity_percent"],
        dram_bandwidth_utilization_percent=stats["dram_bandwidth_utilization_percent"],
        available_metrics=available,
        unavailable_metrics=unavailable,
        throttle_reasons_observed=throttle_reasons_observed,
    )


_LOW_UTILIZATION_THRESHOLD_PERCENT = 20.0


def low_utilization_note(summary: TelemetrySummary) -> str | None:
    """Standalone, single-trace observation (not a baseline/candidate
    comparison): a workload can show high latency together with low mean
    GPU activity, consistent with small kernels, launch/dispatch
    overhead, CPU-side gaps, or serialization -- basic telemetry alone
    cannot distinguish among those causes, so this is worded as an
    underutilization signal, never a specific root cause.
    """
    mean = summary.gpu_utilization_percent.mean
    if mean is None:
        return None
    if mean < _LOW_UTILIZATION_THRESHOLD_PERCENT:
        return (
            f"Mean GPU utilization during the telemetry window was low ({mean:.1f}%), "
            "consistent with underutilization (e.g. small kernels, launch/dispatch overhead, "
            "or serialization) -- basic telemetry cannot distinguish which."
        )
    return None


# --- correlation ------------------------------------------------------------

# Thresholds below decide only whether a diagnostic SENTENCE is worded as
# "materially" changed in a report -- they never gate PASS/FAIL (that
# remains Milestone 14's RegressionResult, entirely untouched by this
# module). Kept as named constants, not inline magic numbers, so the
# rationale is visible and the values are easy to review/adjust.
_MATERIAL_UTILIZATION_DELTA_PP = 10.0        # percentage points
_MATERIAL_CLOCK_RELATIVE_DELTA = 0.05        # 5%
_MATERIAL_TEMPERATURE_DELTA_C = 5.0          # degrees Celsius
_MATERIAL_VRAM_FRACTION_DELTA_PP = 10.0      # percentage points of total capacity

_THERMAL_POWER_REASON_NAMES = frozenset({
    "HwPowerBrakeSlowdown", "HwThermalSlowdown", "SwPowerCap", "SwThermalSlowdown",
})


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline_value: float | None
    candidate_value: float | None
    absolute_delta: float | None
    delta_kind: str  # "percentage_points" | "absolute_units" | "relative_fraction"
    unit: str

    def to_dict(self) -> dict:
        return {
            "metric": self.metric, "baseline_value": self.baseline_value, "candidate_value": self.candidate_value,
            "absolute_delta": self.absolute_delta, "delta_kind": self.delta_kind, "unit": self.unit,
        }


def _delta(metric: str, baseline: float | None, candidate: float | None, delta_kind: str, unit: str) -> MetricDelta:
    absolute_delta = (candidate - baseline) if (baseline is not None and candidate is not None) else None
    return MetricDelta(
        metric=metric, baseline_value=baseline, candidate_value=candidate,
        absolute_delta=absolute_delta, delta_kind=delta_kind, unit=unit,
    )


@dataclass(frozen=True)
class TelemetryCorrelationResult:
    telemetry_correlation_schema_version: int
    workload_preset: str | None
    workload_kind: str | None
    regression_status: str | None

    baseline_summary: TelemetrySummary
    candidate_summary: TelemetrySummary

    metric_deltas: tuple
    signals: tuple
    unavailable_metrics: tuple

    def to_dict(self) -> dict:
        return {
            "telemetry_correlation_schema_version": self.telemetry_correlation_schema_version,
            "workload": {"preset": self.workload_preset, "kind": self.workload_kind},
            "regression_status": self.regression_status,
            "baseline_summary": self.baseline_summary.to_dict(),
            "candidate_summary": self.candidate_summary.to_dict(),
            "metric_deltas": [d.to_dict() for d in self.metric_deltas],
            "signals": list(self.signals),
            "unavailable_metrics": list(self.unavailable_metrics),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def correlate_telemetry(
    baseline_summary: TelemetrySummary,
    candidate_summary: TelemetrySummary,
    regression_status: str | None = None,
    workload_preset: str | None = None,
    workload_kind: str | None = None,
) -> TelemetryCorrelationResult:
    """Pure comparison of two TelemetrySummarys. Never re-collects
    telemetry, never touches a RegressionResult's status -- `regression_status`
    is carried through purely as context for the report.

    Signal language is deliberately cautious: "evidence suggests",
    "coincided with", "observed alongside", "consistent with" -- never
    "proves", "root cause", "definitely caused by", or a specific
    bottleneck classification. Basic NVML utilization/activity telemetry
    cannot establish those claims on its own.
    """
    b, c = baseline_summary, candidate_summary

    deltas = [
        _delta("gpu_utilization_percent_mean", b.gpu_utilization_percent.mean, c.gpu_utilization_percent.mean, "percentage_points", "%"),
        _delta("gpu_utilization_percent_max", b.gpu_utilization_percent.max, c.gpu_utilization_percent.max, "percentage_points", "%"),
        _delta("memory_activity_percent_mean", b.memory_activity_percent.mean, c.memory_activity_percent.mean, "percentage_points", "%"),
        _delta("memory_activity_percent_max", b.memory_activity_percent.max, c.memory_activity_percent.max, "percentage_points", "%"),
        _delta("memory_used_bytes_max", b.memory_used_bytes.max, c.memory_used_bytes.max, "absolute_units", "bytes"),
        _delta("power_watts_mean", b.power_watts.mean, c.power_watts.mean, "absolute_units", "W"),
        _delta("power_watts_max", b.power_watts.max, c.power_watts.max, "absolute_units", "W"),
        _delta("temperature_celsius_max", b.temperature_celsius.max, c.temperature_celsius.max, "absolute_units", "C"),
        _delta("sm_clock_mhz_mean", b.sm_clock_mhz.mean, c.sm_clock_mhz.mean, "absolute_units", "MHz"),
        _delta("sm_clock_mhz_min", b.sm_clock_mhz.min, c.sm_clock_mhz.min, "absolute_units", "MHz"),
        _delta("memory_clock_mhz_mean", b.memory_clock_mhz.mean, c.memory_clock_mhz.mean, "absolute_units", "MHz"),
    ]
    for advanced in ("sm_activity_percent", "sm_occupancy_percent", "tensor_activity_percent", "dram_bandwidth_utilization_percent"):
        b_stat, c_stat = getattr(b, advanced), getattr(c, advanced)
        if b_stat.sample_count > 0 or c_stat.sample_count > 0:
            deltas.append(_delta(f"{advanced}_mean", b_stat.mean, c_stat.mean, "percentage_points", "%"))

    signals = []

    gpu_util_delta = c.gpu_utilization_percent.mean - b.gpu_utilization_percent.mean if (
        b.gpu_utilization_percent.mean is not None and c.gpu_utilization_percent.mean is not None
    ) else None
    if gpu_util_delta is not None and gpu_util_delta <= -_MATERIAL_UTILIZATION_DELTA_PP:
        signals.append(
            f"Candidate showed lower GPU activity during the telemetry window "
            f"(mean {b.gpu_utilization_percent.mean:.1f}% -> {c.gpu_utilization_percent.mean:.1f}%), "
            "evidence suggests reduced GPU-side work relative to baseline."
        )

    mem_act_delta = c.memory_activity_percent.mean - b.memory_activity_percent.mean if (
        b.memory_activity_percent.mean is not None and c.memory_activity_percent.mean is not None
    ) else None
    if mem_act_delta is not None and mem_act_delta >= _MATERIAL_UTILIZATION_DELTA_PP:
        signals.append(
            f"Candidate showed greater device-memory activity during the telemetry window "
            f"(mean {b.memory_activity_percent.mean:.1f}% -> {c.memory_activity_percent.mean:.1f}%)."
        )

    if b.sm_clock_mhz.mean and c.sm_clock_mhz.mean is not None:
        relative = (c.sm_clock_mhz.mean - b.sm_clock_mhz.mean) / b.sm_clock_mhz.mean
        if relative <= -_MATERIAL_CLOCK_RELATIVE_DELTA:
            signals.append(
                f"Candidate observed lower SM clocks during the telemetry window "
                f"(mean {b.sm_clock_mhz.mean:.0f} MHz -> {c.sm_clock_mhz.mean:.0f} MHz), "
                "consistent with (but not proof of) power/thermal management activity."
            )

    candidate_thermal_power_reasons = tuple(
        r for r in c.throttle_reasons_observed if r in _THERMAL_POWER_REASON_NAMES
    )
    if candidate_thermal_power_reasons:
        signals.append(
            "A thermal or power throttle event was observed during the candidate telemetry "
            f"window (reasons: {', '.join(candidate_thermal_power_reasons)})."
        )

    if c.memory_total_bytes and b.memory_total_bytes and c.memory_used_bytes.max is not None and b.memory_used_bytes.max is not None:
        b_fraction = b.memory_used_bytes.max / b.memory_total_bytes * 100
        c_fraction = c.memory_used_bytes.max / c.memory_total_bytes * 100
        if c_fraction - b_fraction >= _MATERIAL_VRAM_FRACTION_DELTA_PP:
            signals.append(
                f"Candidate showed higher peak VRAM usage during the telemetry window "
                f"({b_fraction:.1f}% -> {c_fraction:.1f}% of total device memory)."
            )

    unavailable = tuple(sorted(set(b.unavailable_metrics) | set(c.unavailable_metrics)))

    return TelemetryCorrelationResult(
        telemetry_correlation_schema_version=TELEMETRY_CORRELATION_SCHEMA_VERSION,
        workload_preset=workload_preset, workload_kind=workload_kind, regression_status=regression_status,
        baseline_summary=baseline_summary, candidate_summary=candidate_summary,
        metric_deltas=tuple(deltas), signals=tuple(signals), unavailable_metrics=unavailable,
    )


def render_telemetry_markdown_section(correlation: TelemetryCorrelationResult) -> str:
    """Markdown block meant to be appended to a Milestone-14 regression
    report. Always ends with the explicit "does not affect the gate"
    statement so the distinction is never lost in a rendered report.
    """
    lines = ["## GPU Telemetry Context", ""]
    lines.append("| Signal | Base | Candidate | Delta |")
    lines.append("| --- | ---: | ---: | ---: |")
    for d in correlation.metric_deltas:
        base = f"{d.baseline_value:.2f}" if d.baseline_value is not None else "n/a"
        candidate = f"{d.candidate_value:.2f}" if d.candidate_value is not None else "n/a"
        if d.absolute_delta is None:
            delta_str = "n/a"
        elif d.delta_kind == "percentage_points":
            delta_str = f"{d.absolute_delta:+.1f} pp"
        else:
            delta_str = f"{d.absolute_delta:+.2f} {d.unit}"
        lines.append(f"| {d.metric} | {base} {d.unit} | {candidate} {d.unit} | {delta_str} |")
    lines.append("")

    if correlation.signals:
        lines.append("Observed signals:")
        lines.append("")
        for s in correlation.signals:
            lines.append(f"- {s}")
    else:
        lines.append("No material telemetry signals observed between baseline and candidate.")
    lines.append("")

    if correlation.unavailable_metrics:
        lines.append(f"Unavailable metrics: {', '.join(correlation.unavailable_metrics)}")
        lines.append("")

    lines.append(
        "Telemetry is diagnostic context and does not affect the regression gate result "
        f"(regression status: {correlation.regression_status or 'n/a'})."
    )
    return "\n".join(lines) + "\n"
