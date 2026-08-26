"""PyTorch sustained compute / memory-copy calibration probes.

Imports torch -- only import this module when the "benchmark" extra is
installed. `tensorforge_ops.cli` imports it lazily, inside the
`calibrate` command handler.

Reuses `tensorforge_ops.benchmark`'s backend-agnostic timing loop and
statistics (no second, possibly-incorrect CUDA timer) and
`tensorforge_ops.benchmark_pytorch`'s device resolution / CUDA
synchronization / runtime-metadata helpers, so calibration and benchmark
measurements share identical timing semantics.
"""

import torch

from tensorforge_ops.benchmark import compute_latency_statistics, run_timed_iterations
from tensorforge_ops.benchmark_pytorch import _resolve_device, _runtime_metadata, _sync_fn_for
from tensorforge_ops.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationConfig,
    ComputeProbeResult,
    DeviceCalibrationProfile,
    MemoryProbeResult,
    compute_probe_rate,
    memory_probe_bandwidth,
    memory_probe_traffic_bytes,
)

_DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32}
_DTYPE_BYTES = {torch.float16: 2, torch.float32: 4}


def _resolve_calibration_dtype(dtype: str) -> torch.dtype:
    torch_dtype = _DTYPE_MAP.get(dtype)
    if torch_dtype is None:
        raise ValueError(
            f"dtype {dtype!r} is not supported by the PyTorch calibration backend "
            f"(supported: {', '.join(_DTYPE_MAP)})"
        )
    return torch_dtype


def _device_metadata(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "compute_capability": f"{major}.{minor}",
        "total_device_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
    }


def _run_compute_probe(
    device: torch.device, torch_dtype: torch.dtype, m: int, n: int, k: int,
    warmup_iterations: int, measured_iterations: int,
) -> ComputeProbeResult:
    a = torch.randn(m, k, device=device, dtype=torch_dtype)
    b = torch.randn(k, n, device=device, dtype=torch_dtype)
    c = torch.empty(m, n, device=device, dtype=torch_dtype)

    def _run() -> None:
        torch.mm(a, b, out=c)

    sync_fn = _sync_fn_for(device)
    with torch.inference_mode():
        samples = run_timed_iterations(_run, warmup_iterations, measured_iterations, sync_fn=sync_fn)

    stats = compute_latency_statistics(samples)
    flops = 2 * m * n * k
    return ComputeProbeResult(
        shape_m=m, shape_n=n, shape_k=k, flops=flops,
        latency_samples_seconds=samples,
        p50_seconds=stats.p50_seconds,
        effective_compute_flops_per_second=compute_probe_rate(flops, stats.p50_seconds),
    )


def _run_memory_probe(
    device: torch.device, torch_dtype: torch.dtype, payload_bytes: int,
    warmup_iterations: int, measured_iterations: int,
) -> MemoryProbeResult:
    bytes_per_element = _DTYPE_BYTES[torch_dtype]
    num_elements = max(1, payload_bytes // bytes_per_element)
    actual_payload_bytes = num_elements * bytes_per_element

    src = torch.randn(num_elements, device=device, dtype=torch_dtype)
    dst = torch.empty(num_elements, device=device, dtype=torch_dtype)

    def _run() -> None:
        dst.copy_(src)

    sync_fn = _sync_fn_for(device)
    with torch.inference_mode():
        samples = run_timed_iterations(_run, warmup_iterations, measured_iterations, sync_fn=sync_fn)

    stats = compute_latency_statistics(samples)
    traffic = memory_probe_traffic_bytes(actual_payload_bytes)
    return MemoryProbeResult(
        payload_bytes=actual_payload_bytes,
        modeled_copy_traffic_bytes=traffic,
        latency_samples_seconds=samples,
        p50_seconds=stats.p50_seconds,
        effective_memory_bandwidth_bytes_per_second=memory_probe_bandwidth(actual_payload_bytes, stats.p50_seconds),
    )


def run_pytorch_calibration(config: CalibrationConfig) -> DeviceCalibrationProfile:
    device = _resolve_device(config.device)
    torch_dtype = _resolve_calibration_dtype(config.dtype)

    compute_probe = _run_compute_probe(
        device, torch_dtype, config.compute_m, config.compute_n, config.compute_k,
        config.warmup_iterations, config.measured_iterations,
    )
    memory_probe = _run_memory_probe(
        device, torch_dtype, config.memory_probe_mib * 1024 * 1024,
        config.warmup_iterations, config.measured_iterations,
    )

    runtime_metadata = _runtime_metadata(device)
    device_index = runtime_metadata.get("device_index") if device.type == "cuda" else None
    device_name = runtime_metadata.get("device_name") if device.type == "cuda" else None

    return DeviceCalibrationProfile(
        calibration_schema_version=CALIBRATION_SCHEMA_VERSION,
        backend="pytorch",
        device_type=device.type,
        device_index=device_index,
        device_name=device_name,
        dtype=config.dtype,
        runtime_metadata=runtime_metadata,
        device_metadata=_device_metadata(device),
        compute_probe=compute_probe,
        memory_probe=memory_probe,
        effective_compute_flops_per_second=compute_probe.effective_compute_flops_per_second,
        effective_memory_bandwidth_bytes_per_second=memory_probe.effective_memory_bandwidth_bytes_per_second,
    )
