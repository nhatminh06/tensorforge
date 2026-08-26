"""Reproducible workload and accelerator presets.

These exist for reproducibility, demos, regression experiments, and
documentation -- they are NOT claims about any real commercial hardware
or any real neural-network model. Accelerator presets use generic names
(`small`, `balanced`, `compute-heavy`, `bandwidth-heavy`) and workload
presets use generic names (`gemm_tiny`, `transformer_medium`,
`conv_spatial`, `cnn_like_small`). None of them are named after or
claimed to model any specific real GPU/TPU/model architecture.

Peak-compute derivation for accelerator presets: since this project's
model is `1 PE = 1 MAC/cycle` and `1 MAC = 2 FLOPs` (see gemm.py,
pe_array.py), an internally-coherent preset derives its roofline peak
compute directly from PE geometry and clock rather than picking an
unrelated number:

    peak_compute_flops_per_second = 2 * pe_rows * pe_cols * clock_hz

This derivation applies only to preset construction -- it does not change
how `--peak-tflops` behaves for explicit, user-supplied CLI configurations
elsewhere in this project.
"""

from dataclasses import dataclass

from tensorforge.convolution import CnnLayerSpec, CnnWorkload, Conv2DSpec
from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy
from tensorforge.pe_array import PeArray
from tensorforge.timing import TimingConfig
from tensorforge.transformer import TransformerBlockSpec


@dataclass(frozen=True)
class GemmPreset:
    name: str
    m: int
    n: int
    k: int
    dtype: DType
    description: str

    kind: str = "gemm"

    def gemm(self) -> Gemm:
        return Gemm(m=self.m, n=self.n, k=self.k, dtype=self.dtype)


@dataclass(frozen=True)
class TransformerPreset:
    name: str
    batch_size: int
    sequence_length: int
    d_model: int
    num_heads: int
    d_ff: int
    dtype: DType
    description: str

    kind: str = "transformer"

    def spec(self) -> TransformerBlockSpec:
        return TransformerBlockSpec(
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class ConvPreset:
    name: str
    batch_size: int
    in_channels: int
    input_height: int
    input_width: int
    out_channels: int
    kernel_height: int
    kernel_width: int
    stride_height: int
    stride_width: int
    padding_height: int
    padding_width: int
    dtype: DType
    description: str

    kind: str = "conv2d"

    def spec(self) -> Conv2DSpec:
        return Conv2DSpec(
            batch_size=self.batch_size,
            in_channels=self.in_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            out_channels=self.out_channels,
            kernel_height=self.kernel_height,
            kernel_width=self.kernel_width,
            stride_height=self.stride_height,
            stride_width=self.stride_width,
            padding_height=self.padding_height,
            padding_width=self.padding_width,
            dtype=self.dtype,
        )


@dataclass(frozen=True)
class CnnPreset:
    name: str
    batch_size: int
    in_channels: int
    input_height: int
    input_width: int
    dtype: DType
    layers: tuple[CnnLayerSpec, ...]
    description: str

    kind: str = "cnn"

    def workload(self) -> CnnWorkload:
        return CnnWorkload(
            batch_size=self.batch_size,
            in_channels=self.in_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            dtype=self.dtype,
            layers=self.layers,
        )


WorkloadPreset = GemmPreset | TransformerPreset | ConvPreset | CnnPreset


@dataclass(frozen=True)
class AcceleratorPreset:
    """A generic experiment accelerator configuration -- not a real chip."""

    name: str
    pe_rows: int
    pe_cols: int
    sram_bytes: int
    clock_hz: float
    bandwidth_bytes_per_second: float
    description: str

    def __post_init__(self) -> None:
        for field_name in ("pe_rows", "pe_cols"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive int, got {value!r}")
        if not isinstance(self.sram_bytes, int) or isinstance(self.sram_bytes, bool) or self.sram_bytes <= 0:
            raise ValueError(f"sram_bytes must be a positive int, got {self.sram_bytes!r}")
        if self.clock_hz <= 0:
            raise ValueError(f"clock_hz must be > 0, got {self.clock_hz}")
        if self.bandwidth_bytes_per_second <= 0:
            raise ValueError(
                f"bandwidth_bytes_per_second must be > 0, got {self.bandwidth_bytes_per_second}"
            )

    @property
    def peak_compute_flops_per_second(self) -> float:
        """2 FLOPs/MAC * PE count * clock_hz (1 MAC/PE/cycle)."""
        return 2 * self.pe_rows * self.pe_cols * self.clock_hz

    def pe_array(self) -> PeArray:
        return PeArray(rows=self.pe_rows, columns=self.pe_cols)

    def memory_hierarchy(self) -> MemoryHierarchy:
        return MemoryHierarchy(sram_bytes=self.sram_bytes)

    def hardware_config(self) -> HardwareConfig:
        return HardwareConfig(
            peak_compute_flops_per_second=self.peak_compute_flops_per_second,
            memory_bandwidth_bytes_per_second=self.bandwidth_bytes_per_second,
            name=self.name,
        )

    def timing_config(self) -> TimingConfig:
        return TimingConfig(clock_hz=self.clock_hz)


_GEMM_PRESETS: tuple[GemmPreset, ...] = (
    GemmPreset("gemm_tiny", m=128, n=128, k=128, dtype=DType.FP16,
               description="Small square GEMM for fast experiments."),
    GemmPreset("gemm_tall", m=2048, n=256, k=512, dtype=DType.FP16,
               description="Tall/skinny GEMM (M >> N)."),
    GemmPreset("gemm_wide", m=256, n=2048, k=512, dtype=DType.FP16,
               description="Wide/short GEMM (N >> M)."),
    GemmPreset("gemm_large_square", m=1024, n=1024, k=1024, dtype=DType.FP16,
               description="Large square GEMM."),
)

_TRANSFORMER_PRESETS: tuple[TransformerPreset, ...] = (
    TransformerPreset("transformer_small", batch_size=1, sequence_length=128, d_model=512,
                       num_heads=8, d_ff=2048, dtype=DType.FP16,
                       description="Generic Transformer-like block, short sequence."),
    TransformerPreset("transformer_medium", batch_size=1, sequence_length=256, d_model=768,
                       num_heads=12, d_ff=3072, dtype=DType.FP16,
                       description="Generic Transformer-like block, medium sequence."),
    TransformerPreset("transformer_long_sequence", batch_size=1, sequence_length=1024, d_model=768,
                       num_heads=12, d_ff=3072, dtype=DType.FP16,
                       description="Generic Transformer-like block, long sequence (attention-heavy)."),
)

_CONV_PRESETS: tuple[ConvPreset, ...] = (
    ConvPreset("conv_pointwise", batch_size=1, in_channels=64, input_height=56, input_width=56,
               out_channels=128, kernel_height=1, kernel_width=1,
               stride_height=1, stride_width=1, padding_height=0, padding_width=0, dtype=DType.FP16,
               description="1x1 (pointwise) convolution -- minimal im2col expansion."),
    ConvPreset("conv_spatial", batch_size=1, in_channels=64, input_height=56, input_width=56,
               out_channels=128, kernel_height=3, kernel_width=3,
               stride_height=1, stride_width=1, padding_height=1, padding_width=1, dtype=DType.FP16,
               description="3x3 convolution, same spatial resolution -- notable im2col expansion."),
    ConvPreset("conv_downsample", batch_size=1, in_channels=64, input_height=56, input_width=56,
               out_channels=128, kernel_height=3, kernel_width=3,
               stride_height=2, stride_width=2, padding_height=1, padding_width=1, dtype=DType.FP16,
               description="3x3 stride-2 convolution -- halves spatial resolution."),
)

_CNN_PRESETS: tuple[CnnPreset, ...] = (
    CnnPreset(
        "cnn_like_small",
        batch_size=1, in_channels=3, input_height=32, input_width=32, dtype=DType.FP16,
        layers=(
            CnnLayerSpec("stage1", out_channels=8, kernel_height=3, kernel_width=3,
                         stride_height=1, stride_width=1, padding_height=1, padding_width=1),
            CnnLayerSpec("stage2", out_channels=16, kernel_height=3, kernel_width=3,
                         stride_height=2, stride_width=2, padding_height=1, padding_width=1),
            CnnLayerSpec("stage3", out_channels=32, kernel_height=3, kernel_width=3,
                         stride_height=2, stride_width=2, padding_height=1, padding_width=1),
            CnnLayerSpec("stage4_pointwise", out_channels=32, kernel_height=1, kernel_width=1,
                         stride_height=1, stride_width=1, padding_height=0, padding_width=0),
        ),
        description=(
            "Small generic Conv-only CNN-like workload (4 stages, changing channel width, "
            "two stride-2 downsamples, one pointwise stage). Intentionally small (32x32 input) "
            "for fast, deterministic exploration -- not a reproduction of any real CNN architecture."
        ),
    ),
)

_WORKLOAD_PRESETS: dict[str, WorkloadPreset] = {
    p.name: p for p in (*_GEMM_PRESETS, *_TRANSFORMER_PRESETS, *_CONV_PRESETS, *_CNN_PRESETS)
}

_ACCELERATOR_PRESETS: tuple[AcceleratorPreset, ...] = (
    AcceleratorPreset("small", pe_rows=16, pe_cols=16, sram_bytes=256 * 1024,
                       clock_hz=1e9, bandwidth_bytes_per_second=50e9,
                       description="Small generic accelerator: modest compute and bandwidth."),
    AcceleratorPreset("balanced", pe_rows=32, pe_cols=32, sram_bytes=512 * 1024,
                       clock_hz=1e9, bandwidth_bytes_per_second=100e9,
                       description="Balanced generic accelerator (baseline for comparisons)."),
    AcceleratorPreset("compute_heavy", pe_rows=64, pe_cols=64, sram_bytes=512 * 1024,
                       clock_hz=1e9, bandwidth_bytes_per_second=100e9,
                       description="4x the balanced preset's PE count, same bandwidth/SRAM."),
    AcceleratorPreset("bandwidth_heavy", pe_rows=32, pe_cols=32, sram_bytes=512 * 1024,
                       clock_hz=1e9, bandwidth_bytes_per_second=300e9,
                       description="Same PE/SRAM as balanced, 3x the DRAM bandwidth."),
)

_ACCELERATOR_PRESETS_BY_NAME: dict[str, AcceleratorPreset] = {p.name: p for p in _ACCELERATOR_PRESETS}


def list_workload_presets() -> tuple[str, ...]:
    return tuple(_WORKLOAD_PRESETS.keys())


def list_accelerator_presets() -> tuple[str, ...]:
    return tuple(_ACCELERATOR_PRESETS_BY_NAME.keys())


def get_workload_preset(name: str) -> WorkloadPreset:
    try:
        return _WORKLOAD_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown workload preset {name!r}; available: {', '.join(sorted(_WORKLOAD_PRESETS))}"
        ) from None


def get_accelerator_preset(name: str) -> AcceleratorPreset:
    try:
        return _ACCELERATOR_PRESETS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown accelerator preset {name!r}; available: "
            f"{', '.join(sorted(_ACCELERATOR_PRESETS_BY_NAME))}"
        ) from None
