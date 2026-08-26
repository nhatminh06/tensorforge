"""PyTorch steady-state inference benchmark backend.

Imports torch -- only import this module when the "benchmark" extra is
installed. `tensorforge_ops.cli` imports it lazily, inside the
`benchmark` command handler, so `--help`/`track`/`list-runs` keep working
without PyTorch installed.

Benchmark scope (documented once, applies to every workload builder
below): inputs and weights are allocated on the target device *before*
`run_timed_iterations()` is called, so measured latency excludes model/
tensor construction and any initial host->device upload -- this is
steady-state, device-resident execution latency, not end-to-end request
latency. `torch.inference_mode()` wraps the whole timed region: no
autograd tracking, no gradients, no training-mode behavior (dropout/
batchnorm running-stat updates) anywhere.

Transformer workload: only the GEMM operations TensorForge Core models
are executed (Q/K/V/output projections, per-head attention-score and
attention-value matmuls, MLP up/down) -- no softmax, no normalization, no
residual connections, no dropout. Attention heads are executed in an
explicit Python loop (one iteration per (batch, head) pair) rather than a
single batched/vectorized matmul, to match Core's sequential-repetition
model as closely as practical; this is a modeling-alignment benchmark,
not an optimized Transformer kernel. This is why the workload is called
"transformer_gemm_only" everywhere, never "Transformer inference."
"""

import torch

from tensorforge.gemm import DType
from tensorforge.presets import ConvPreset, GemmPreset, TransformerPreset, WorkloadPreset

from tensorforge_ops.benchmark import BenchmarkConfig, BenchmarkResult, compute_latency_statistics, run_timed_iterations

_DTYPE_TO_TORCH = {
    DType.FP32: torch.float32,
    DType.FP16: torch.float16,
}


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"device {device_str!r} was explicitly requested but torch.cuda.is_available() "
            "is False -- refusing to silently fall back to CPU"
        )
    return device


def _resolve_dtype(dtype: DType) -> torch.dtype:
    torch_dtype = _DTYPE_TO_TORCH.get(dtype)
    if torch_dtype is None:
        raise ValueError(
            f"dtype {dtype.name} is not supported by the PyTorch benchmark backend "
            f"(supported: {', '.join(d.name for d in _DTYPE_TO_TORCH)})"
        )
    return torch_dtype


def _sync_fn_for(device: torch.device):
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    return None


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_allocated(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None  # CPU peak accelerator memory is intentionally unavailable this milestone.


def _runtime_metadata(device: torch.device) -> dict:
    metadata = {"torch_version": torch.__version__, "device_type": device.type}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata["device_index"] = index
        metadata["device_name"] = torch.cuda.get_device_name(index)
        if torch.version.cuda is not None:
            metadata["cuda_runtime_version"] = torch.version.cuda
    return metadata


def _build_gemm_callable(preset: GemmPreset, device: torch.device, dtype: torch.dtype):
    gemm = preset.gemm()
    a = torch.randn(gemm.m, gemm.k, device=device, dtype=dtype)
    b = torch.randn(gemm.k, gemm.n, device=device, dtype=dtype)

    def _run() -> None:
        torch.matmul(a, b)

    return _run


def _build_conv_callable(preset: ConvPreset, device: torch.device, dtype: torch.dtype):
    spec = preset.spec()
    x = torch.randn(
        spec.batch_size, spec.in_channels, spec.input_height, spec.input_width,
        device=device, dtype=dtype,
    )
    weight = torch.randn(
        spec.out_channels, spec.in_channels, spec.kernel_height, spec.kernel_width,
        device=device, dtype=dtype,
    )
    stride = (spec.stride_height, spec.stride_width)
    padding = (spec.padding_height, spec.padding_width)

    def _run() -> None:
        # bias=None: Core's Conv2D MAC model does not include a bias term.
        torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)

    return _run


def _build_transformer_gemm_callable(preset: TransformerPreset, device: torch.device, dtype: torch.dtype):
    spec = preset.spec()
    batch, seq, d_model, num_heads, head_dim, d_ff = (
        spec.batch_size, spec.sequence_length, spec.d_model, spec.num_heads, spec.head_dim, spec.d_ff,
    )
    rows = batch * seq

    x = torch.randn(rows, d_model, device=device, dtype=dtype)
    w_q = torch.randn(d_model, d_model, device=device, dtype=dtype)
    w_k = torch.randn(d_model, d_model, device=device, dtype=dtype)
    w_v = torch.randn(d_model, d_model, device=device, dtype=dtype)
    w_o = torch.randn(d_model, d_model, device=device, dtype=dtype)
    w_up = torch.randn(d_model, d_ff, device=device, dtype=dtype)
    w_down = torch.randn(d_ff, d_model, device=device, dtype=dtype)

    def _run() -> None:
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        qh = q.view(batch, seq, num_heads, head_dim)
        kh = k.view(batch, seq, num_heads, head_dim)
        vh = v.view(batch, seq, num_heads, head_dim)

        batch_outputs = []
        for b in range(batch):
            head_outputs = []
            for h in range(num_heads):  # explicit sequential loop, matching Core's B*H repetition model
                qb = qh[b, :, h, :]
                kb = kh[b, :, h, :]
                vb = vh[b, :, h, :]
                scores = qb @ kb.transpose(0, 1)  # (S, S) -- no softmax
                head_outputs.append(scores @ vb)   # (S, head_dim)
            batch_outputs.append(torch.cat(head_outputs, dim=-1))  # (S, D)
        attn = torch.stack(batch_outputs, dim=0).view(rows, d_model)

        projected = attn @ w_o
        hidden = projected @ w_up
        hidden @ w_down

    return _run


_UNSUPPORTED_KIND_MESSAGE = (
    "the PyTorch benchmark backend does not support workload kind {kind!r} yet "
    "(supported: gemm, conv2d, transformer)"
)


def run_pytorch_benchmark(
    preset: WorkloadPreset,
    core_result_fingerprint: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    device = _resolve_device(config.device)
    torch_dtype = _resolve_dtype(preset.dtype)

    if preset.kind == "gemm":
        fn = _build_gemm_callable(preset, device, torch_dtype)
    elif preset.kind == "conv2d":
        fn = _build_conv_callable(preset, device, torch_dtype)
    elif preset.kind == "transformer":
        fn = _build_transformer_gemm_callable(preset, device, torch_dtype)
    else:
        raise ValueError(_UNSUPPORTED_KIND_MESSAGE.format(kind=preset.kind))

    sync_fn = _sync_fn_for(device)
    _reset_peak_memory(device)

    with torch.inference_mode():
        samples = run_timed_iterations(
            fn, config.warmup_iterations, config.measured_iterations, sync_fn=sync_fn
        )

    return BenchmarkResult(
        core_result_fingerprint=core_result_fingerprint,
        workload_preset=preset.name,
        workload_kind=preset.kind,
        backend="pytorch",
        device=str(device),
        dtype=preset.dtype.name.lower(),
        warmup_iterations=config.warmup_iterations,
        measured_iterations=config.measured_iterations,
        latency_samples_seconds=samples,
        statistics=compute_latency_statistics(samples),
        peak_memory_allocated_bytes=_peak_memory_allocated(device),
        runtime_metadata=_runtime_metadata(device),
    )
