"""Command-line interface for analytical GEMM + roofline modeling.

Example:
    python -m tensorforge --m 1024 --n 1024 --k 1024 --dtype fp16 \\
        --peak-tflops 10 --bandwidth-gbps 200
"""

import argparse

from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.memory import MemoryHierarchy, analyze_memory
from tensorforge.pe_array import PeArray, map_gemm
from tensorforge.roofline import compute_roofline
from tensorforge.tiling import GemmSchedule, GemmTile, analyze_tiling, compare_schedules
from tensorforge.timing import TimingConfig, estimate_execution_time
from tensorforge.explore import explore
from tensorforge.transformer import (
    TransformerBlockSpec,
    evaluate_transformer_block,
    explore_transformer_architectures,
)

_DTYPE_CHOICES = {"fp32": DType.FP32, "fp16": DType.FP16, "int8": DType.INT8}
_OP_DISPLAY_NAMES = {
    "q_projection": "Q projection",
    "k_projection": "K projection",
    "v_projection": "V projection",
    "attention_scores": "Attention scores",
    "attention_value": "Attention value",
    "output_projection": "Output projection",
    "mlp_up": "MLP up",
    "mlp_down": "MLP down",
}
_SCHEDULE_CHOICES = {
    "c-resident": GemmSchedule.C_RESIDENT,
    "a-resident": GemmSchedule.A_RESIDENT,
    "b-resident": GemmSchedule.B_RESIDENT,
}
_KIB = 1024


def _format_bytes(num_bytes: float) -> str:
    if num_bytes < _KIB:
        return f"{num_bytes:,.0f} B"
    if num_bytes < _KIB**2:
        return f"{num_bytes / _KIB:.2f} KiB"
    return f"{num_bytes / _KIB**2:.2f} MiB"


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _parse_int_list(parser: argparse.ArgumentParser, flag: str, raw: str) -> list[int]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        try:
            values.append(int(piece))
        except ValueError:
            parser.error(f"{flag}: could not parse {piece!r} as an int")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorforge",
        description="Analytical GEMM + roofline modeling.",
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--dtype", choices=sorted(_DTYPE_CHOICES), default="fp32")
    parser.add_argument(
        "--peak-tflops",
        type=float,
        required=True,
        help="Peak compute throughput in TFLOP/s (10^12 FLOP/s).",
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        required=True,
        help="Memory bandwidth in GB/s (10^9 bytes/s).",
    )
    parser.add_argument("--name", default="Example Accelerator")
    parser.add_argument("--pe-rows", type=int, default=None)
    parser.add_argument("--pe-cols", type=int, default=None)
    parser.add_argument(
        "--sram-kib",
        type=float,
        default=None,
        help="Modeled global SRAM capacity in KiB (1 KiB = 1024 bytes).",
    )
    parser.add_argument("--tile-m", type=int, default=None)
    parser.add_argument("--tile-n", type=int, default=None)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--schedule", choices=sorted(_SCHEDULE_CHOICES), default=None)
    parser.add_argument("--compare-schedules", action="store_true")
    parser.add_argument(
        "--clock-ghz",
        type=float,
        default=None,
        help="Modeled PE clock frequency in GHz (1 GHz = 1e9 Hz), decimal.",
    )
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--tile-m-values", default=None, help="Comma-separated ints, e.g. 16,32,64")
    parser.add_argument("--tile-n-values", default=None, help="Comma-separated ints, e.g. 16,32,64")
    parser.add_argument("--tile-k-values", default=None, help="Comma-separated ints, e.g. 16,32")
    parser.add_argument("--pe-row-values", default=None, help="Comma-separated ints, e.g. 8,16,32")
    parser.add_argument("--pe-col-values", default=None, help="Comma-separated ints, e.g. 8,16,32")
    parser.add_argument(
        "--schedule-values",
        default=None,
        help="Comma-separated schedule names (default: all three, c-resident,a-resident,b-resident)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--transformer-block", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--explore-pe", action="store_true")
    return parser


def _explain_winner(ranked, best) -> list[str]:
    """Deterministic, metric-based bullet points comparing the winner to a
    few directly relevant candidates already present in the ranked list.
    No free-text generation — only comparisons of actual field values.
    """
    lines = []
    bc, bt = best.candidate, best.timing_result

    if len(ranked) > 1:
        runner_up = ranked[1]
        if runner_up.timing_result.perfect_overlap_time_seconds == bt.perfect_overlap_time_seconds:
            lines.append("ties with the next-ranked candidate on perfect-overlap time")
        else:
            lines.append(
                f"beats the next-ranked candidate ({runner_up.candidate.tile_m}x"
                f"{runner_up.candidate.tile_n}x{runner_up.candidate.tile_k}, "
                f"{runner_up.candidate.schedule.value}, "
                f"{runner_up.candidate.pe_rows}x{runner_up.candidate.pe_cols}) by "
                f"{_format_seconds(runner_up.timing_result.perfect_overlap_time_seconds - bt.perfect_overlap_time_seconds)}"
            )

    smaller_pe = [
        r for r in ranked
        if r.candidate.tile_m == bc.tile_m and r.candidate.tile_n == bc.tile_n
        and r.candidate.tile_k == bc.tile_k and r.candidate.schedule == bc.schedule
        and r.candidate.pe_rows * r.candidate.pe_cols < bc.pe_rows * bc.pe_cols
    ]
    if smaller_pe:
        smaller_pe.sort(key=lambda r: r.candidate.pe_rows * r.candidate.pe_cols, reverse=True)
        alt = smaller_pe[0]
        if alt.timing_result.perfect_overlap_time_seconds == bt.perfect_overlap_time_seconds:
            lines.append(
                f"a smaller PE array ({alt.candidate.pe_rows}x{alt.candidate.pe_cols}) with the "
                "same tile/schedule ties this perfect-overlap time (already memory-bound here)"
            )
        else:
            lines.append(
                f"a smaller PE array ({alt.candidate.pe_rows}x{alt.candidate.pe_cols}) with the "
                f"same tile/schedule would raise perfect-overlap time to "
                f"{_format_seconds(alt.timing_result.perfect_overlap_time_seconds)}"
            )

    other_schedule = [
        r for r in ranked
        if r.candidate.tile_m == bc.tile_m and r.candidate.tile_n == bc.tile_n
        and r.candidate.tile_k == bc.tile_k and r.candidate.pe_rows == bc.pe_rows
        and r.candidate.pe_cols == bc.pe_cols and r.candidate.schedule != bc.schedule
    ]
    if other_schedule:
        other_schedule.sort(key=lambda r: r.timing_result.dram_bytes)
        alt = other_schedule[0]
        if alt.timing_result.dram_bytes != bt.dram_bytes:
            lines.append(
                f"same tile/PE with {alt.candidate.schedule.value} moves "
                f"{_format_bytes(alt.timing_result.dram_bytes)} of DRAM traffic "
                f"({_format_bytes(alt.timing_result.dram_bytes - bt.dram_bytes)} more than the winner)"
            )

    return lines


def _run_explore(parser: argparse.ArgumentParser, args, gemm: Gemm, hardware: HardwareConfig) -> None:
    tile_m_values = _parse_int_list(parser, "--tile-m-values", args.tile_m_values)
    tile_n_values = _parse_int_list(parser, "--tile-n-values", args.tile_n_values)
    tile_k_values = _parse_int_list(parser, "--tile-k-values", args.tile_k_values)
    pe_row_values = _parse_int_list(parser, "--pe-row-values", args.pe_row_values)
    pe_col_values = _parse_int_list(parser, "--pe-col-values", args.pe_col_values)

    schedules = None
    if args.schedule_values is not None:
        schedules = []
        for name in args.schedule_values.split(","):
            name = name.strip()
            if name not in _SCHEDULE_CHOICES:
                parser.error(f"--schedule-values: unknown schedule {name!r}")
            schedules.append(_SCHEDULE_CHOICES[name])

    try:
        sram_bytes = round(args.sram_kib * _KIB)
        hierarchy = MemoryHierarchy(sram_bytes=sram_bytes)
        timing_config = TimingConfig(clock_hz=args.clock_ghz * 1e9)
        exploration = explore(
            gemm, hierarchy, hardware, timing_config,
            tile_m_values=tile_m_values, tile_n_values=tile_n_values, tile_k_values=tile_k_values,
            pe_rows_values=pe_row_values, pe_cols_values=pe_col_values,
            schedules=schedules, top_k=args.top_k,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Workload: GEMM {gemm.m} x {gemm.n} x {gemm.k} ({args.dtype})")
    print()
    print("Exploration (bounded, deterministic search over user-supplied candidates)")
    print(f"  total candidates        {exploration.total_candidates:,}")
    print(f"  feasible candidates     {exploration.feasible_candidates:,}")
    print(f"  infeasible candidates   {exploration.infeasible_candidates:,}")
    print("  objective               perfect-overlap time (ascending)")

    if exploration.feasible_candidates == 0:
        print()
        print("No feasible configurations: every candidate tile exceeded the modeled SRAM capacity.")
        return

    top = exploration.ranked[: exploration.top_k]
    print()
    print(f"Top {len(top)} configurations (of {exploration.feasible_candidates} feasible)")
    for i, r in enumerate(top, start=1):
        c, t = r.candidate, r.timing_result
        print()
        print(f"#{i}")
        print(f"  tile                    {c.tile_m} x {c.tile_n} x {c.tile_k}")
        print(f"  schedule                {c.schedule.value}")
        print(f"  PE array                {c.pe_rows} x {c.pe_cols}")
        print(f"  tile working set        {_format_bytes(r.tiling_result.max_tile_working_set_bytes)}")
        print(f"  compute cycles          {t.compute_cycles:,}")
        print(f"  compute time            {_format_seconds(t.compute_time_seconds)}")
        print(f"  DRAM traffic            {_format_bytes(t.dram_bytes)}")
        print(f"  memory time             {_format_seconds(t.memory_time_seconds)}")
        print(f"  perfect-overlap         {_format_seconds(t.perfect_overlap_time_seconds)}")
        print(f"  serialized              {_format_seconds(t.serialized_time_seconds)}")
        print(f"  bottleneck              {t.bottleneck}")
        print(
            f"  effective arithmetic intensity {r.tiling_result.effective_arithmetic_intensity:.4f} FLOP/byte"
        )

    if exploration.tied_for_best > 1:
        print()
        print(f"Note: {exploration.tied_for_best} feasible candidates tie on perfect-overlap time.")

    best = exploration.ranked[0]
    bc = best.candidate
    print()
    print(
        f"Best among searched candidates: tile {bc.tile_m}x{bc.tile_n}x{bc.tile_k}, "
        f"{bc.schedule.value}, PE {bc.pe_rows}x{bc.pe_cols}"
    )
    explanation = _explain_winner(list(exploration.ranked), best)
    if explanation:
        print()
        print("Why:")
        for line in explanation:
            print(f"  - {line}")


def _print_block_header(spec: TransformerBlockSpec, hierarchy, hardware, args, pe_array) -> None:
    print("Transformer GEMM block (GEMM-only estimate, not complete Transformer latency)")
    print()
    print("Block")
    print(f"  batch                    {spec.batch_size}")
    print(f"  sequence length          {spec.sequence_length}")
    print(f"  d_model                  {spec.d_model}")
    print(f"  heads                    {spec.num_heads}")
    print(f"  head_dim                 {spec.head_dim}")
    print(f"  d_ff                     {spec.d_ff}")
    print()
    print("Hardware")
    print(f"  PE array                 {pe_array.rows} x {pe_array.columns}")
    print(f"  SRAM                     {_format_bytes(hierarchy.sram_bytes)}")
    print(f"  clock                    {args.clock_ghz:.2f} GHz")
    print(f"  DRAM bandwidth           {args.bandwidth_gbps:.2f} GB/s")


def _print_block_operations(block_result) -> None:
    print()
    print("Modeled GEMMs")
    for r in block_result.operation_results:
        c, t = r.mapping.candidate, r.mapping.timing_result
        print()
        print(_OP_DISPLAY_NAMES.get(r.name, r.name))
        print(f"  shape                    {r.gemm.m} x {r.gemm.n} x {r.gemm.k}")
        print(f"  repetitions              {r.repetitions}")
        print(f"  selected tile            {c.tile_m} x {c.tile_n} x {c.tile_k}")
        print(f"  schedule                 {c.schedule.value}")
        print(f"  per-execution time       {_format_seconds(t.perfect_overlap_time_seconds)}")
        if r.repetitions > 1:
            print(
                f"  aggregate time           {_format_seconds(r.aggregate_perfect_overlap_time_seconds)}"
            )


def _print_block_totals(block_result) -> None:
    print()
    print("Block totals (GEMM-only)")
    print(f"  modeled GEMM MACs        {block_result.total_macs:,}")
    print(f"  modeled GEMM FLOPs       {block_result.total_flops:,}")
    print(f"  modeled DRAM traffic     {_format_bytes(block_result.total_dram_bytes)}")
    print(f"  perfect-overlap time     {_format_seconds(block_result.perfect_overlap_time_seconds)}")
    print(f"  serialized time          {_format_seconds(block_result.serialized_time_seconds)}")
    print()
    print(f"Largest modeled GEMM time contributor: {_OP_DISPLAY_NAMES.get(block_result.largest_time_contributor, block_result.largest_time_contributor)}")
    print(f"Largest modeled DRAM contributor: {_OP_DISPLAY_NAMES.get(block_result.largest_dram_contributor, block_result.largest_dram_contributor)}")
    print()
    print("Unmodeled (excluded, not zero-cost):")
    for name in block_result.unmodeled_operations:
        print(f"  - {name}")


def _run_transformer_block(parser: argparse.ArgumentParser, args, hardware: HardwareConfig) -> None:
    tile_m_values = _parse_int_list(parser, "--tile-m-values", args.tile_m_values)
    tile_n_values = _parse_int_list(parser, "--tile-n-values", args.tile_n_values)
    tile_k_values = _parse_int_list(parser, "--tile-k-values", args.tile_k_values)

    schedules = None
    if args.schedule_values is not None:
        schedules = []
        for name in args.schedule_values.split(","):
            name = name.strip()
            if name not in _SCHEDULE_CHOICES:
                parser.error(f"--schedule-values: unknown schedule {name!r}")
            schedules.append(_SCHEDULE_CHOICES[name])

    try:
        spec = TransformerBlockSpec(
            batch_size=args.batch_size,
            sequence_length=args.seq_len,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            dtype=_DTYPE_CHOICES[args.dtype],
        )
        sram_bytes = round(args.sram_kib * _KIB)
        hierarchy = MemoryHierarchy(sram_bytes=sram_bytes)
        timing_config = TimingConfig(clock_hz=args.clock_ghz * 1e9)
    except ValueError as exc:
        parser.error(str(exc))

    if args.explore_pe:
        pe_row_values = _parse_int_list(parser, "--pe-row-values", args.pe_row_values)
        pe_col_values = _parse_int_list(parser, "--pe-col-values", args.pe_col_values)

        try:
            arch_result = explore_transformer_architectures(
                spec, hierarchy, hardware, timing_config,
                tile_m_values=tile_m_values, tile_n_values=tile_n_values, tile_k_values=tile_k_values,
                pe_rows_values=pe_row_values, pe_cols_values=pe_col_values,
                schedules=schedules, top_k=args.top_k,
            )
        except ValueError as exc:
            parser.error(str(exc))

        print("Transformer block architecture exploration (GEMM-only estimate)")
        print()
        print(f"Architectures evaluated: {arch_result.total_architectures}")
        print(f"Feasible: {arch_result.feasible_architectures}")

        if arch_result.feasible_architectures == 0:
            print()
            print("No feasible PE architectures: some operation had no feasible tile mapping.")
            return

        top = arch_result.ranked[: arch_result.top_k]
        print()
        print(f"Top {len(top)} configurations (of {arch_result.feasible_architectures} feasible)")
        for i, a in enumerate(top, start=1):
            br = a.block_result
            print()
            print(f"#{i}")
            print(f"  PE array                 {a.pe_rows} x {a.pe_cols}")
            print(f"  PE count                 {a.pe_count:,}")
            print(f"  block perfect-overlap    {_format_seconds(br.perfect_overlap_time_seconds)}")
            print(f"  block serialized         {_format_seconds(br.serialized_time_seconds)}")
            print(f"  block DRAM traffic       {_format_bytes(br.total_dram_bytes)}")
            print(
                f"  largest contributor      {_OP_DISPLAY_NAMES.get(br.largest_time_contributor, br.largest_time_contributor)}"
            )

        best = arch_result.ranked[0]
        print()
        print(f"Best among searched PE architectures: {best.pe_rows} x {best.pe_cols}")
        print()
        _print_block_operations(best.block_result)
        _print_block_totals(best.block_result)
    else:
        pe_array = PeArray(rows=args.pe_rows, columns=args.pe_cols)
        try:
            block_result = evaluate_transformer_block(
                spec, pe_array, hierarchy, hardware, timing_config,
                tile_m_values=tile_m_values, tile_n_values=tile_n_values, tile_k_values=tile_k_values,
                schedules=schedules,
            )
        except ValueError as exc:
            parser.error(str(exc))

        _print_block_header(spec, hierarchy, hardware, args, pe_array)
        _print_block_operations(block_result)
        _print_block_totals(block_result)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if (args.pe_rows is None) != (args.pe_cols is None):
        parser.error("--pe-rows and --pe-cols must be provided together")

    tile_args = (args.tile_m, args.tile_n, args.tile_k)
    if any(t is not None for t in tile_args):
        if any(t is None for t in tile_args):
            parser.error("--tile-m, --tile-n, and --tile-k must all be provided together")
        if args.sram_kib is None:
            parser.error("--tile-m/--tile-n/--tile-k require --sram-kib")

    if args.schedule is not None or args.compare_schedules:
        if any(t is None for t in tile_args) or args.sram_kib is None:
            parser.error(
                "--schedule/--compare-schedules require --tile-m, --tile-n, "
                "--tile-k, and --sram-kib"
            )

    if args.clock_ghz is not None and not args.explore and not args.transformer_block:
        if args.pe_rows is None or any(t is None for t in tile_args) or args.sram_kib is None:
            parser.error(
                "--clock-ghz requires --pe-rows/--pe-cols, --tile-m/--tile-n/--tile-k, "
                "and --sram-kib"
            )
        if args.compare_schedules:
            parser.error("--clock-ghz cannot be combined with --compare-schedules")
        if args.clock_ghz <= 0:
            parser.error("--clock-ghz must be > 0")

    if args.explore and args.clock_ghz is not None and args.clock_ghz <= 0:
        parser.error("--clock-ghz must be > 0")

    if args.explore:
        if any(t is not None for t in tile_args):
            parser.error("--explore cannot be combined with --tile-m/--tile-n/--tile-k "
                         "(use --tile-m-values/--tile-n-values/--tile-k-values instead)")
        if args.pe_rows is not None or args.pe_cols is not None:
            parser.error("--explore cannot be combined with --pe-rows/--pe-cols "
                         "(use --pe-row-values/--pe-col-values instead)")
        if args.schedule is not None or args.compare_schedules:
            parser.error("--explore cannot be combined with --schedule/--compare-schedules "
                         "(use --schedule-values instead)")
        if args.sram_kib is None or args.clock_ghz is None:
            parser.error("--explore requires --sram-kib and --clock-ghz")
        required_value_flags = (
            ("--tile-m-values", args.tile_m_values),
            ("--tile-n-values", args.tile_n_values),
            ("--tile-k-values", args.tile_k_values),
            ("--pe-row-values", args.pe_row_values),
            ("--pe-col-values", args.pe_col_values),
        )
        missing = [flag for flag, value in required_value_flags if value is None]
        if missing:
            parser.error(f"--explore requires {', '.join(missing)}")

    if args.transformer_block:
        conflicting = (
            ("--m/--n/--k", any(v is not None for v in (args.m, args.n, args.k))),
            ("--tile-m/--tile-n/--tile-k", any(t is not None for t in tile_args)),
            ("--schedule", args.schedule is not None),
            ("--compare-schedules", args.compare_schedules),
            ("--explore", args.explore),
        )
        conflicts = [name for name, present in conflicting if present]
        if conflicts:
            parser.error(f"--transformer-block cannot be combined with {', '.join(conflicts)}")

        block_dims = (
            ("--batch-size", args.batch_size),
            ("--seq-len", args.seq_len),
            ("--d-model", args.d_model),
            ("--num-heads", args.num_heads),
            ("--d-ff", args.d_ff),
        )
        missing_dims = [flag for flag, value in block_dims if value is None]
        if missing_dims:
            parser.error(f"--transformer-block requires {', '.join(missing_dims)}")

        if args.sram_kib is None or args.clock_ghz is None:
            parser.error("--transformer-block requires --sram-kib and --clock-ghz")
        if any(v is None for v in (args.tile_m_values, args.tile_n_values, args.tile_k_values)):
            parser.error(
                "--transformer-block requires --tile-m-values, --tile-n-values, "
                "and --tile-k-values"
            )

        if args.explore_pe:
            if args.pe_rows is not None or args.pe_cols is not None:
                parser.error("--explore-pe cannot be combined with --pe-rows/--pe-cols "
                             "(use --pe-row-values/--pe-col-values instead)")
            if args.pe_row_values is None or args.pe_col_values is None:
                parser.error("--explore-pe requires --pe-row-values and --pe-col-values")
        else:
            if args.pe_rows is None or args.pe_cols is None:
                parser.error(
                    "--transformer-block requires --pe-rows/--pe-cols (fixed architecture), "
                    "or --explore-pe with --pe-row-values/--pe-col-values"
                )
            if args.pe_row_values is not None or args.pe_col_values is not None:
                parser.error("--pe-row-values/--pe-col-values require --explore-pe")

    elif args.explore_pe:
        parser.error("--explore-pe requires --transformer-block")

    if args.transformer_block:
        hardware = HardwareConfig(
            peak_compute_flops_per_second=args.peak_tflops * 1e12,
            memory_bandwidth_bytes_per_second=args.bandwidth_gbps * 1e9,
            name=args.name,
        )
        _run_transformer_block(parser, args, hardware)
        return

    if args.m is None or args.n is None or args.k is None:
        parser.error("--m, --n, and --k are required (unless using --transformer-block)")

    gemm = Gemm(m=args.m, n=args.n, k=args.k, dtype=_DTYPE_CHOICES[args.dtype])
    hardware = HardwareConfig(
        peak_compute_flops_per_second=args.peak_tflops * 1e12,
        memory_bandwidth_bytes_per_second=args.bandwidth_gbps * 1e9,
        name=args.name,
    )

    if args.explore:
        _run_explore(parser, args, gemm, hardware)
        return

    result = compute_roofline(gemm, hardware)

    print(f"Workload: GEMM {gemm.m} x {gemm.n} x {gemm.k} ({args.dtype})")
    print(f"Hardware: {hardware.name}")
    print()
    print("Operations")
    print(f"  MACs                 {gemm.macs:,}")
    print(f"  FLOPs                {gemm.flops:,}")
    print()
    print("Tensor bytes")
    print(f"  A bytes              {gemm.a_bytes:,}")
    print(f"  B bytes              {gemm.b_bytes:,}")
    print(f"  C bytes              {gemm.c_bytes:,}")
    print(f"  modeled DRAM bytes   {gemm.dram_bytes:,}")
    print()
    print("Roofline")
    print(f"  arithmetic intensity   {result.arithmetic_intensity:.6f} FLOP/byte")
    print(f"  ridge point            {result.ridge_point:.6f} FLOP/byte")
    print(f"  compute ceiling        {result.compute_ceiling_flops_per_second:.6e} FLOP/s")
    print(f"  memory ceiling         {result.memory_ceiling_flops_per_second:.6e} FLOP/s")
    print(f"  attainable performance {result.attainable_flops_per_second:.6e} FLOP/s")
    print()
    print("Estimated time (analytical lower bound, perfect overlap assumed)")
    print(f"  compute time         {result.compute_time_seconds:.6e} s")
    print(f"  memory time          {result.memory_time_seconds:.6e} s")
    print(f"  estimated time       {result.estimated_time_seconds:.6e} s")
    print()
    print(f"Primary bound: {result.classification}")

    if args.pe_rows is not None:
        pe_array = PeArray(rows=args.pe_rows, columns=args.pe_cols)
        mapping = map_gemm(gemm, pe_array)

        print()
        print("PE array")
        print(f"  shape                   {pe_array.rows} x {pe_array.columns}")
        print(f"  processing elements     {mapping.pe_count:,}")
        print(f"  peak throughput         {mapping.pe_count:,} MAC/cycle")
        print()
        print("Mapping (idealized spatial mapping, M->rows, N->columns)")
        print(f"  row waves               {mapping.row_waves:,}")
        print(f"  column waves            {mapping.column_waves:,}")
        print(f"  total waves             {mapping.waves:,}")
        print(f"  useful PE slots         {mapping.useful_pe_slots:,}")
        print(f"  available PE slots      {mapping.available_pe_slots:,}")
        print(f"  unused PE slots         {mapping.unused_pe_slots:,}")
        print(f"  spatial utilization     {mapping.spatial_utilization * 100:.2f}%")
        print()
        print("Compute (analytical cycles, no memory stalls or pipeline modeled)")
        print(f"  cycles per wave         {mapping.cycles_per_wave:,}")
        print(f"  analytical cycles       {mapping.compute_cycles:,}")

    if args.sram_kib is not None:
        sram_bytes = round(args.sram_kib * _KIB)
        hierarchy = MemoryHierarchy(sram_bytes=sram_bytes)
        mem = analyze_memory(gemm, hierarchy)

        print()
        print("Memory hierarchy")
        print(f"  SRAM capacity           {_format_bytes(mem.sram_bytes)}")
        print()
        print("Tensor footprint")
        print(f"  A                       {_format_bytes(mem.a_bytes)}")
        print(f"  B                       {_format_bytes(mem.b_bytes)}")
        print(f"  C                       {_format_bytes(mem.c_bytes)}")
        print(f"  full working set        {_format_bytes(mem.working_set_bytes)}")
        print()
        print("Capacity (fit is a capacity fact, not a reuse guarantee)")
        print(f"  A fits                  {'yes' if mem.a_fits else 'no'}")
        print(f"  B fits                  {'yes' if mem.b_fits else 'no'}")
        print(f"  C fits                  {'yes' if mem.c_fits else 'no'}")
        print(f"  full working set fits   {'yes' if mem.full_working_set_fits else 'no'}")
        print(f"  capacity headroom       {_format_bytes(mem.capacity_headroom_bytes)}")
        print(f"  capacity deficit        {_format_bytes(mem.capacity_deficit_bytes)}")
        print(f"  working-set ratio       {mem.working_set_to_capacity_ratio:.2f}x")
        print(f"  tiling required         {'yes' if mem.tiling_required else 'no'}")
        print()
        print("Baseline modeled DRAM <-> SRAM traffic (idealized, unchanged by fit)")
        print(f"  reads                   {_format_bytes(mem.dram_read_bytes)}")
        print(f"  writes                  {_format_bytes(mem.dram_write_bytes)}")
        print(f"  total                   {_format_bytes(mem.total_dram_bytes)}")

        if args.tile_m is not None:
            tile = GemmTile(tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k)
            schedule = _SCHEDULE_CHOICES[args.schedule] if args.schedule else GemmSchedule.C_RESIDENT

            if args.compare_schedules:
                try:
                    comparison = compare_schedules(gemm, tile, hierarchy)
                except ValueError as exc:
                    parser.error(str(exc))

                print()
                print("Schedule comparison (exact under each modeled schedule only)")
                print(f"  tile shape              {tile.tile_m} x {tile.tile_n} x {tile.tile_k}")
                print()
                header = f"  {'schedule':<12}{'A read':>12}{'B read':>12}{'C read':>12}{'C write':>12}{'total':>12}{'amp':>9}{'AI':>10}"
                print(header)
                for result in (comparison.c_resident, comparison.a_resident, comparison.b_resident):
                    print(
                        f"  {result.schedule.value:<12}"
                        f"{_format_bytes(result.a_dram_read_bytes):>12}"
                        f"{_format_bytes(result.b_dram_read_bytes):>12}"
                        f"{_format_bytes(result.c_dram_read_bytes):>12}"
                        f"{_format_bytes(result.c_dram_write_bytes):>12}"
                        f"{_format_bytes(result.total_dram_bytes):>12}"
                        f"{result.traffic_amplification:>8.2f}x"
                        f"{result.effective_arithmetic_intensity:>9.3f}"
                    )
                print()
                best = comparison.best_among_modeled_schedules
                if len(best) == 1:
                    print(f"Best among modeled schedules: {best[0].value}")
                else:
                    tied = ", ".join(s.value for s in best)
                    print(f"Best among modeled schedules: tie ({tied})")
            else:
                try:
                    tiling = analyze_tiling(gemm, tile, hierarchy, schedule=schedule)
                except ValueError as exc:
                    parser.error(str(exc))

                print()
                print(f"Tiling ({tiling.schedule.value} schedule, exact under this schedule only)")
                print(f"  tile shape              {tile.tile_m} x {tile.tile_n} x {tile.tile_k}")
                print(f"  tile counts             {tiling.m_tiles} x {tiling.n_tiles} x {tiling.k_tiles}")
                print(f"  tile steps              {tiling.tile_steps:,}")
                print(f"  max tile working set    {_format_bytes(tiling.max_tile_working_set_bytes)}")
                print(f"  tile fits SRAM          {'yes' if tiling.tile_fits else 'no'}")
                print()
                print("DRAM traffic (tiled)")
                print(f"  A reads                 {_format_bytes(tiling.a_dram_read_bytes)}")
                print(f"  B reads                 {_format_bytes(tiling.b_dram_read_bytes)}")
                print(f"  C reads                 {_format_bytes(tiling.c_dram_read_bytes)}")
                print(f"  C writes                {_format_bytes(tiling.c_dram_write_bytes)}")
                print(f"  total                   {_format_bytes(tiling.total_dram_bytes)}")
                print()
                print(f"  ideal baseline          {_format_bytes(tiling.ideal_baseline_dram_bytes)}")
                print(f"  amplification           {tiling.traffic_amplification:.2f}x")
                print(f"  tiled arithmetic intensity {tiling.effective_arithmetic_intensity:.4f} FLOP/byte")

                if args.clock_ghz is not None:
                    clock_hz = args.clock_ghz * 1e9
                    timing_config = TimingConfig(clock_hz=clock_hz)
                    timing_result = estimate_execution_time(mapping, tiling, hardware, timing_config)

                    print()
                    print("Execution timing (analytical; no pipeline/memory-stall/NoC modeling)")
                    print(f"  PE clock                {args.clock_ghz:.2f} GHz")
                    print(f"  PE compute cycles       {timing_result.compute_cycles:,}")
                    print(f"  compute time            {_format_seconds(timing_result.compute_time_seconds)}")
                    print()
                    print(f"  schedule                {tiling.schedule.value}")
                    print(f"  modeled DRAM traffic    {_format_bytes(timing_result.dram_bytes)}")
                    print(
                        f"  DRAM bandwidth          {timing_result.memory_bandwidth_bytes_per_second / 1e9:.2f} GB/s"
                    )
                    print(f"  memory time             {_format_seconds(timing_result.memory_time_seconds)}")
                    print()
                    print("Timing bounds")
                    print(
                        f"  serialized estimate     {_format_seconds(timing_result.serialized_time_seconds)}"
                    )
                    print(
                        f"  perfect-overlap estimate {_format_seconds(timing_result.perfect_overlap_time_seconds)}"
                    )
                    print()
                    print(f"Primary modeled bound: {timing_result.bottleneck}")


if __name__ == "__main__":
    main()
