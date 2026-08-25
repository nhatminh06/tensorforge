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
from tensorforge.tiling import GemmTile, analyze_tiling

_DTYPE_CHOICES = {"fp32": DType.FP32, "fp16": DType.FP16, "int8": DType.INT8}
_KIB = 1024


def _format_bytes(num_bytes: float) -> str:
    if num_bytes < _KIB:
        return f"{num_bytes:,.0f} B"
    if num_bytes < _KIB**2:
        return f"{num_bytes / _KIB:.2f} KiB"
    return f"{num_bytes / _KIB**2:.2f} MiB"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorforge",
        description="Analytical GEMM + roofline modeling.",
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
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
    return parser


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

    gemm = Gemm(m=args.m, n=args.n, k=args.k, dtype=_DTYPE_CHOICES[args.dtype])
    hardware = HardwareConfig(
        peak_compute_flops_per_second=args.peak_tflops * 1e12,
        memory_bandwidth_bytes_per_second=args.bandwidth_gbps * 1e9,
        name=args.name,
    )
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
            try:
                tiling = analyze_tiling(gemm, tile, hierarchy)
            except ValueError as exc:
                parser.error(str(exc))

            print()
            print("Tiling (fixed C-output-tile-resident schedule, exact under this schedule only)")
            print(f"  tile shape              {tile.tile_m} x {tile.tile_n} x {tile.tile_k}")
            print(f"  tile counts             {tiling.m_tiles} x {tiling.n_tiles} x {tiling.k_tiles}")
            print(f"  tile steps              {tiling.tile_steps:,}")
            print(f"  max tile working set    {_format_bytes(tiling.max_tile_working_set_bytes)}")
            print(f"  tile fits SRAM          {'yes' if tiling.tile_fits else 'no'}")
            print()
            print("DRAM traffic (tiled)")
            print(f"  A reads                 {_format_bytes(tiling.a_dram_read_bytes)}")
            print(f"  B reads                 {_format_bytes(tiling.b_dram_read_bytes)}")
            print(f"  C writes                {_format_bytes(tiling.c_dram_write_bytes)}")
            print(f"  total                   {_format_bytes(tiling.total_dram_bytes)}")
            print()
            print(f"  ideal baseline          {_format_bytes(tiling.ideal_baseline_dram_bytes)}")
            print(f"  amplification           {tiling.traffic_amplification:.2f}x")
            print(f"  tiled arithmetic intensity {tiling.effective_arithmetic_intensity:.4f} FLOP/byte")


if __name__ == "__main__":
    main()
