"""Live NVML GPU telemetry collection.

Imports `pynvml` (the `nvidia-ml-py` package -- NVIDIA's own maintained
bindings) -- only import this module when the "telemetry" extra is
installed. `tensorforge_ops.cli` imports it lazily, inside telemetry
command handlers, so the rest of Ops keeps working without it installed.

NVML is the primary/mandatory local telemetry backend for this project
(see telemetry_dcgm.py for the optional DCGM Exporter backend). NVML
supports basic monitoring broadly across Maxwell-and-newer NVIDIA GPUs,
including consumer/laptop GeForce parts -- unlike DCGM's advanced
profiling metrics, which NVIDIA documents as limited to supported
datacenter GPUs and is not assumed available here.

Every metric this module reads is wrapped so an unsupported/failed NVML
call for one field becomes `None` on the sample, never `0` and never a
fatal error for the rest of the sample. Only a failure to initialize
NVML itself, or to get a device handle, raises -- see
`NvmlUnavailableError`.
"""

import threading
import time

import pynvml

from tensorforge_ops.telemetry import TELEMETRY_SCHEMA_VERSION, TelemetryConfig, TelemetrySample, TelemetryTrace

_CLOCK_EVENT_REASON_NAMES = {
    pynvml.nvmlClocksEventReasonGpuIdle: "GpuIdle",
    pynvml.nvmlClocksEventReasonApplicationsClocksSetting: "ApplicationsClocksSetting",
    pynvml.nvmlClocksEventReasonSwPowerCap: "SwPowerCap",
    pynvml.nvmlClocksEventReasonHwSlowdown: "HwSlowdown",
    pynvml.nvmlClocksEventReasonSyncBoost: "SyncBoost",
    pynvml.nvmlClocksEventReasonSwThermalSlowdown: "SwThermalSlowdown",
    pynvml.nvmlClocksEventReasonHwThermalSlowdown: "HwThermalSlowdown",
    pynvml.nvmlClocksEventReasonHwPowerBrakeSlowdown: "HwPowerBrakeSlowdown",
    pynvml.nvmlClocksEventReasonDisplayClockSetting: "DisplayClockSetting",
}


def decode_clock_event_reasons(bitmask: int | None) -> tuple:
    """Decode an NVML clocks-event-reason bitmask into known reason
    names. A decoded reason name is evidence that bit was set at sample
    time -- it is not, by itself, proof of that reason's performance
    impact (see telemetry.py's correlation docstring).
    """
    if bitmask is None:
        return ()
    return tuple(name for bit, name in _CLOCK_EVENT_REASON_NAMES.items() if bitmask & bit)


class NvmlUnavailableError(RuntimeError):
    """NVML itself could not be initialized or the device handle could
    not be obtained -- distinct from a single unsupported metric field,
    which becomes None on the sample rather than raising.
    """


def _safe(fn):
    try:
        return fn()
    except pynvml.NVMLError:
        return None


def _nvidia_ml_py_version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version("nvidia-ml-py")
    except importlib.metadata.PackageNotFoundError:
        return None


class NvmlProvider:
    """NVML lifecycle: nvmlInit() once, get the device handle once,
    sample repeatedly, nvmlShutdown() once. Never re-initializes NVML
    per sample -- that would be both slow and unnecessary.
    """

    def __init__(self, device_index: int):
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            raise NvmlUnavailableError(f"failed to initialize NVML: {exc}") from exc
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except pynvml.NVMLError as exc:
            pynvml.nvmlShutdown()
            raise NvmlUnavailableError(f"failed to get NVML device handle for index {device_index}: {exc}") from exc

        self.device_index = device_index
        self.device_name = _safe(lambda: pynvml.nvmlDeviceGetName(self._handle))
        # GPM (GPU Performance Monitoring) advanced metrics (SM/tensor/DRAM
        # activity) are proven supported or not at construction time, once --
        # never assumed. Proven UNSUPPORTED on the RTX 3050 Laptop GPU this
        # project was developed/verified against; actual GPM sample
        # collection is intentionally not implemented here because it could
        # not be live-verified on that hardware (see docs/ops/gpu-telemetry.md).
        # The four advanced fields remain None on every sample until this is
        # implemented and verified against a device that actually supports it.
        self.gpm_supported = self._probe_gpm_support()

    def _probe_gpm_support(self) -> bool:
        try:
            support = pynvml.nvmlGpmQueryDeviceSupport(self._handle)
            return bool(support.isSupportedDevice)
        except (pynvml.NVMLError, AttributeError):
            return False

    def runtime_metadata(self) -> dict:
        metadata = {
            "nvidia_ml_py_version": _nvidia_ml_py_version(),
            "nvml_driver_version": _safe(pynvml.nvmlSystemGetDriverVersion),
            "gpm_supported": self.gpm_supported,
        }
        uuid = _safe(lambda: pynvml.nvmlDeviceGetUUID(self._handle))
        if uuid is not None:
            metadata["device_uuid"] = uuid
        compute_capability = _safe(lambda: pynvml.nvmlDeviceGetCudaComputeCapability(self._handle))
        if compute_capability is not None:
            metadata["compute_capability"] = f"{compute_capability[0]}.{compute_capability[1]}"
        return metadata

    def sample(self, timestamp_seconds: float) -> TelemetrySample:
        utilization = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(self._handle))
        gpu_utilization_percent = float(utilization.gpu) if utilization is not None else None
        memory_activity_percent = float(utilization.memory) if utilization is not None else None

        memory_info = _safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(self._handle))
        memory_used_bytes = int(memory_info.used) if memory_info is not None else None
        memory_free_bytes = int(memory_info.free) if memory_info is not None else None
        memory_total_bytes = int(memory_info.total) if memory_info is not None else None

        power_mw = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(self._handle))
        power_watts = power_mw / 1000.0 if power_mw is not None else None

        temperature = _safe(lambda: pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU))
        temperature_celsius = float(temperature) if temperature is not None else None

        sm_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM))
        memory_clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_MEM))

        performance_state = _safe(lambda: pynvml.nvmlDeviceGetPerformanceState(self._handle))
        performance_state = int(performance_state) if performance_state is not None else None

        reasons_bitmask = _safe(lambda: pynvml.nvmlDeviceGetCurrentClocksEventReasons(self._handle))
        reasons_bitmask = int(reasons_bitmask) if reasons_bitmask is not None else None

        return TelemetrySample(
            timestamp_seconds=timestamp_seconds,
            gpu_utilization_percent=gpu_utilization_percent,
            memory_activity_percent=memory_activity_percent,
            memory_used_bytes=memory_used_bytes,
            memory_free_bytes=memory_free_bytes,
            memory_total_bytes=memory_total_bytes,
            power_watts=power_watts,
            temperature_celsius=temperature_celsius,
            sm_clock_mhz=int(sm_clock) if sm_clock is not None else None,
            memory_clock_mhz=int(memory_clock) if memory_clock is not None else None,
            performance_state=performance_state,
            clock_event_reasons_bitmask=reasons_bitmask,
            clock_event_reasons=decode_clock_event_reasons(reasons_bitmask),
            # GPM sample collection not implemented -- see class docstring.
            sm_activity_percent=None,
            sm_occupancy_percent=None,
            tensor_activity_percent=None,
            dram_bandwidth_utilization_percent=None,
        )

    def shutdown(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def probe_capabilities(device_index: int) -> dict:
    """One-shot capability report: which telemetry fields this device/
    driver/NVML combination actually returns a value for right now.
    Used by `python -m tensorforge_ops telemetry probe`.
    """
    provider = NvmlProvider(device_index)
    try:
        sample = provider.sample(0.0)
        capabilities = {
            "gpu_utilization_percent": sample.gpu_utilization_percent is not None,
            "memory_activity_percent": sample.memory_activity_percent is not None,
            "memory_used_bytes": sample.memory_used_bytes is not None,
            "power_watts": sample.power_watts is not None,
            "temperature_celsius": sample.temperature_celsius is not None,
            "sm_clock_mhz": sample.sm_clock_mhz is not None,
            "memory_clock_mhz": sample.memory_clock_mhz is not None,
            "performance_state": sample.performance_state is not None,
            "clock_event_reasons": sample.clock_event_reasons_bitmask is not None,
            "sm_activity_percent": sample.sm_activity_percent is not None,
            "sm_occupancy_percent": sample.sm_occupancy_percent is not None,
            "tensor_activity_percent": sample.tensor_activity_percent is not None,
            "dram_bandwidth_utilization_percent": sample.dram_bandwidth_utilization_percent is not None,
        }
        return {
            "device_index": device_index,
            "device_name": provider.device_name,
            "runtime_metadata": provider.runtime_metadata(),
            "capabilities": capabilities,
        }
    finally:
        provider.shutdown()


def run_telemetry_window(
    workload_fn,
    config: TelemetryConfig,
    sync_fn=None,
    workload_preset: str | None = None,
    workload_kind: str | None = None,
    benchmark_backend: str | None = None,
    dtype: str | None = None,
    core_result_fingerprint: str | None = None,
) -> TelemetryTrace:
    """PHASE B (see telemetry.py's module docstring): repeatedly execute
    `workload_fn` (the SAME prepared workload the latency benchmark used
    -- same shapes/device/dtype) for `config.telemetry_duration_seconds`
    while a background sampler thread polls NVML at
    `config.sample_interval_seconds`. Never measures or returns latency.

    If `workload_fn` raises, the sampler is still stopped/joined cleanly
    and NVML is still shut down before the original exception propagates
    -- no background thread and no NVML handle are ever leaked.
    """
    if config.backend != "nvml":
        raise ValueError(f"telemetry_nvml.run_telemetry_window only supports backend='nvml', got {config.backend!r}")

    provider = NvmlProvider(config.device_index)
    samples = []
    stop_event = threading.Event()
    start = time.monotonic()

    def _sampler() -> None:
        while not stop_event.is_set():
            samples.append(provider.sample(time.monotonic() - start))
            stop_event.wait(config.sample_interval_seconds)

    sampler_thread = threading.Thread(target=_sampler, daemon=True)
    sampler_thread.start()

    workload_error = None
    actual_duration_seconds = 0.0
    try:
        while (time.monotonic() - start) < config.telemetry_duration_seconds:
            workload_fn()
        if sync_fn is not None:
            sync_fn()
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: must still clean up below
        workload_error = exc
    finally:
        stop_event.set()
        actual_duration_seconds = time.monotonic() - start
        sampler_thread.join(timeout=config.sample_interval_seconds * 10 + 5)
        provider.shutdown()

    if sampler_thread.is_alive():
        raise RuntimeError("telemetry sampler thread did not exit cleanly")

    if workload_error is not None:
        raise workload_error

    if not samples:
        raise RuntimeError("telemetry sampler produced no samples")

    return TelemetryTrace(
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        backend="nvml",
        device_index=config.device_index,
        device_name=provider.device_name,
        sample_interval_seconds=config.sample_interval_seconds,
        requested_duration_seconds=config.telemetry_duration_seconds,
        actual_duration_seconds=actual_duration_seconds,
        samples=tuple(samples),
        workload_preset=workload_preset,
        workload_kind=workload_kind,
        benchmark_backend=benchmark_backend,
        dtype=dtype,
        core_result_fingerprint=core_result_fingerprint,
        runtime_metadata=provider.runtime_metadata(),
    )
