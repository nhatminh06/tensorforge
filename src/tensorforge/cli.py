"""Command-line interface for analytical GEMM + roofline modeling.

Example:
    python -m tensorforge --m 1024 --n 1024 --k 1024 --dtype fp16 \\
        --peak-tflops 10 --bandwidth-gbps 200
"""

import argparse

from tensorforge.gemm import DType, Gemm
from tensorforge.hardware import HardwareConfig
from tensorforge.roofline import compute_roofline

_DTYPE_CHOICES = {"fp32": DType.FP32, "fp16": DType.FP16, "int8": DType.INT8}


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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

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


if __name__ == "__main__":
    main()
