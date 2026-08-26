"""Deterministic, machine-readable experiment representation.

An experiment names one workload preset, one accelerator preset, and a
bounded tile/schedule search. Running it reuses the existing analytical
models unchanged (explore.explore(), evaluate_transformer_block(),
evaluate_conv2d(), evaluate_cnn_workload()) -- this module only wires
presets together and shapes the result into deterministic JSON.

    TensorForge experiment
          |
    deterministic result JSON (schema_version)
          |
    (future Phase 2: tracking adapter / MLflow / database / CI regression)

Core does not depend on any future tracking layer; the direction of
dependency is presets/experiments -> workload models -> core analytical
models, never the reverse.

The result JSON deliberately excludes wall-clock timestamps or any
non-deterministic identifier: the same ExperimentSpec run twice produces
identical `to_dict()` output, because every underlying model is a pure
function of its inputs.
"""

import json
import os
from dataclasses import dataclass

from tensorforge.convolution import evaluate_cnn_workload, evaluate_conv2d
from tensorforge.explore import explore
from tensorforge.presets import get_accelerator_preset, get_workload_preset
from tensorforge.tiling import GemmSchedule
from tensorforge.transformer import evaluate_transformer_block

SCHEMA_VERSION = 1

_SCHEDULE_NAME_TO_ENUM = {
    "c-resident": GemmSchedule.C_RESIDENT,
    "a-resident": GemmSchedule.A_RESIDENT,
    "b-resident": GemmSchedule.B_RESIDENT,
}
_ALL_SCHEDULE_NAMES = ("c-resident", "a-resident", "b-resident")

LIMITATIONS = (
    "analytical model, not cycle-accurate",
    "bandwidth-only DRAM timing (no DRAM/SRAM latency, no bank/controller/NoC model)",
    "bounded discrete tile/schedule search over user-supplied candidates only, "
    "not a proof of optimality",
    "Transformer/CNN results are GEMM-only: softmax, normalization, activation, pooling, "
    "and residual operations are excluded, not free",
    "convolution uses a materialized-im2col GEMM lowering, not direct-convolution traffic",
    "no cross-operation/cross-layer SRAM residency (each GEMM evaluated independently "
    "at the DRAM boundary)",
    "no real-hardware validation; accelerator presets are generic experiment "
    "configurations, not real hardware specifications",
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    workload_preset: str
    accelerator_preset: str
    tile_m_values: tuple[int, ...]
    tile_n_values: tuple[int, ...]
    tile_k_values: tuple[int, ...]
    schedule_names: tuple[str, ...] | None = None  # None -> all three schedules

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        for values, label in (
            (self.tile_m_values, "tile_m_values"),
            (self.tile_n_values, "tile_n_values"),
            (self.tile_k_values, "tile_k_values"),
        ):
            if len(values) == 0:
                raise ValueError(f"{label} must not be empty")
        if self.schedule_names is not None:
            if len(self.schedule_names) == 0:
                raise ValueError("schedule_names must not be empty when provided")
            unknown = set(self.schedule_names) - set(_ALL_SCHEDULE_NAMES)
            if unknown:
                raise ValueError(f"unknown schedule name(s): {sorted(unknown)}")


@dataclass(frozen=True)
class ExperimentResult:
    spec: ExperimentSpec
    workload_kind: str
    workload_summary: dict
    accelerator_summary: dict
    primary_metrics: dict
    selected_mappings: tuple[dict, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.spec.name,
            "workload": {
                "preset": self.spec.workload_preset,
                "kind": self.workload_kind,
                **self.workload_summary,
            },
            "accelerator": {
                "preset": self.spec.accelerator_preset,
                **self.accelerator_summary,
            },
            "search": {
                "tile_m_values": list(self.spec.tile_m_values),
                "tile_n_values": list(self.spec.tile_n_values),
                "tile_k_values": list(self.spec.tile_k_values),
                "schedules": list(self.spec.schedule_names)
                if self.spec.schedule_names is not None
                else list(_ALL_SCHEDULE_NAMES),
            },
            "primary_metrics": self.primary_metrics,
            "selected_mappings": [dict(m) for m in self.selected_mappings],
            "limitations": list(self.limitations),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def save(self, path: str, force: bool = False) -> None:
        if not force and os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists; pass force=True (CLI: --force) to overwrite"
            )
        with open(path, "w") as f:
            f.write(self.to_json())
            f.write("\n")


def run_experiment(spec: ExperimentSpec) -> ExperimentResult:
    """Run one experiment: reuse the existing workload evaluators unchanged."""
    workload = get_workload_preset(spec.workload_preset)
    accelerator = get_accelerator_preset(spec.accelerator_preset)

    hierarchy = accelerator.memory_hierarchy()
    hardware = accelerator.hardware_config()
    timing_config = accelerator.timing_config()
    pe_array = accelerator.pe_array()

    schedules = (
        [_SCHEDULE_NAME_TO_ENUM[s] for s in spec.schedule_names]
        if spec.schedule_names is not None
        else None
    )
    tile_kwargs = dict(
        tile_m_values=list(spec.tile_m_values),
        tile_n_values=list(spec.tile_n_values),
        tile_k_values=list(spec.tile_k_values),
    )

    if workload.kind == "gemm":
        gemm = workload.gemm()
        exploration = explore(
            gemm, hierarchy, hardware, timing_config,
            pe_rows_values=[pe_array.rows], pe_cols_values=[pe_array.columns],
            schedules=schedules, top_k=1, **tile_kwargs,
        )
        if exploration.feasible_candidates == 0:
            raise ValueError(
                f"no feasible tile/schedule mapping for workload preset "
                f"{spec.workload_preset!r} on accelerator preset {spec.accelerator_preset!r}"
            )
        best = exploration.ranked[0]
        t = best.timing_result
        workload_summary = {"m": gemm.m, "n": gemm.n, "k": gemm.k, "dtype": gemm.dtype.name}
        primary_metrics = {
            "total_macs": gemm.macs,
            "total_flops": gemm.flops,
            "total_dram_bytes": t.dram_bytes,
            "serialized_time_seconds": t.serialized_time_seconds,
            "perfect_overlap_time_seconds": t.perfect_overlap_time_seconds,
            "bottleneck": t.bottleneck,
            "effective_arithmetic_intensity": best.tiling_result.effective_arithmetic_intensity,
        }
        selected_mappings = (
            {
                "operation": "gemm",
                "tile_m": best.candidate.tile_m,
                "tile_n": best.candidate.tile_n,
                "tile_k": best.candidate.tile_k,
                "schedule": best.candidate.schedule.value,
            },
        )

    elif workload.kind == "transformer":
        tspec = workload.spec()
        block = evaluate_transformer_block(
            tspec, pe_array, hierarchy, hardware, timing_config, schedules=schedules, **tile_kwargs
        )
        workload_summary = {
            "batch_size": tspec.batch_size, "sequence_length": tspec.sequence_length,
            "d_model": tspec.d_model, "num_heads": tspec.num_heads, "d_ff": tspec.d_ff,
            "dtype": tspec.dtype.name,
        }
        primary_metrics = {
            "total_macs": block.total_macs,
            "total_flops": block.total_flops,
            "total_dram_bytes": block.total_dram_bytes,
            "serialized_time_seconds": block.serialized_time_seconds,
            "perfect_overlap_time_seconds": block.perfect_overlap_time_seconds,
            "largest_time_contributor": block.largest_time_contributor,
            "largest_dram_contributor": block.largest_dram_contributor,
        }
        selected_mappings = tuple(
            {
                "operation": r.name,
                "tile_m": r.mapping.candidate.tile_m,
                "tile_n": r.mapping.candidate.tile_n,
                "tile_k": r.mapping.candidate.tile_k,
                "schedule": r.mapping.candidate.schedule.value,
                "repetitions": r.repetitions,
            }
            for r in block.operation_results
        )

    elif workload.kind == "conv2d":
        cspec = workload.spec()
        evaluation = evaluate_conv2d(
            cspec, pe_array, hierarchy, hardware, timing_config, schedules=schedules, **tile_kwargs
        )
        t = evaluation.mapping.timing_result
        workload_summary = {
            "batch_size": cspec.batch_size, "in_channels": cspec.in_channels,
            "input_height": cspec.input_height, "input_width": cspec.input_width,
            "out_channels": cspec.out_channels,
            "kernel_height": cspec.kernel_height, "kernel_width": cspec.kernel_width,
            "stride_height": cspec.stride_height, "stride_width": cspec.stride_width,
            "padding_height": cspec.padding_height, "padding_width": cspec.padding_width,
            "dtype": cspec.dtype.name,
            "output_height": evaluation.lowering.output_height,
            "output_width": evaluation.lowering.output_width,
            "im2col_expansion_ratio": evaluation.lowering.im2col_expansion_ratio,
        }
        primary_metrics = {
            "total_macs": cspec.macs,
            "total_flops": cspec.flops,
            "total_dram_bytes": t.dram_bytes,
            "serialized_time_seconds": t.serialized_time_seconds,
            "perfect_overlap_time_seconds": t.perfect_overlap_time_seconds,
            "bottleneck": t.bottleneck,
            "effective_arithmetic_intensity": evaluation.mapping.tiling_result.effective_arithmetic_intensity,
        }
        c = evaluation.mapping.candidate
        selected_mappings = (
            {"operation": "conv2d", "tile_m": c.tile_m, "tile_n": c.tile_n, "tile_k": c.tile_k,
             "schedule": c.schedule.value},
        )

    elif workload.kind == "cnn":
        cnn_workload = workload.workload()
        result = evaluate_cnn_workload(
            cnn_workload, pe_array, hierarchy, hardware, timing_config, schedules=schedules, **tile_kwargs
        )
        workload_summary = {
            "batch_size": cnn_workload.batch_size, "in_channels": cnn_workload.in_channels,
            "input_height": cnn_workload.input_height, "input_width": cnn_workload.input_width,
            "dtype": cnn_workload.dtype.name, "num_layers": len(cnn_workload.layers),
        }
        primary_metrics = {
            "total_macs": result.total_macs,
            "total_flops": result.total_flops,
            "total_dram_bytes": result.total_dram_bytes,
            "serialized_time_seconds": result.serialized_time_seconds,
            "perfect_overlap_time_seconds": result.perfect_overlap_time_seconds,
            "largest_time_contributor": result.largest_time_contributor,
            "largest_dram_contributor": result.largest_dram_contributor,
        }
        selected_mappings = tuple(
            {
                "operation": lr.name,
                "tile_m": lr.evaluation.mapping.candidate.tile_m,
                "tile_n": lr.evaluation.mapping.candidate.tile_n,
                "tile_k": lr.evaluation.mapping.candidate.tile_k,
                "schedule": lr.evaluation.mapping.candidate.schedule.value,
            }
            for lr in result.layer_results
        )

    else:
        raise ValueError(f"unknown workload kind: {workload.kind}")

    accelerator_summary = {
        "pe_rows": accelerator.pe_rows,
        "pe_cols": accelerator.pe_cols,
        "sram_bytes": accelerator.sram_bytes,
        "clock_hz": accelerator.clock_hz,
        "bandwidth_bytes_per_second": accelerator.bandwidth_bytes_per_second,
        "peak_compute_flops_per_second": accelerator.peak_compute_flops_per_second,
    }

    return ExperimentResult(
        spec=spec,
        workload_kind=workload.kind,
        workload_summary=workload_summary,
        accelerator_summary=accelerator_summary,
        primary_metrics=primary_metrics,
        selected_mappings=selected_mappings,
        limitations=LIMITATIONS,
    )
