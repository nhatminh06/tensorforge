"""TensorForge Ops CLI: track a Core experiment preset run in MLflow.

This is a deliberately separate entry point from `python -m tensorforge`.
The Core CLI remains entirely about running/inspecting analytical
experiments; this CLI owns MLflow tracking and only accepts the compact
preset-based configuration (workload preset, accelerator preset, tile
candidates) rather than exposing every raw Core CLI flag.
"""

import argparse

from tensorforge.experiments import ExperimentSpec, run_experiment
from tensorforge.presets import get_workload_preset

from tensorforge_ops.benchmark import BenchmarkConfig
from tensorforge_ops.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    TrackingConfig,
    compute_result_fingerprint,
    list_runs,
    log_benchmark_result,
    resolve_tracking_uri,
    track_experiment,
    track_result,
)


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
        prog="tensorforge_ops",
        description="Track TensorForge Core experiment results in MLflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    track = sub.add_parser("track", help="Run a Core preset experiment and log it to MLflow.")
    track.add_argument("--workload-preset", required=True)
    track.add_argument("--accelerator-preset", required=True)
    track.add_argument("--tile-m-values", required=True)
    track.add_argument("--tile-n-values", required=True)
    track.add_argument("--tile-k-values", required=True)
    track.add_argument(
        "--schedule-values", default=None,
        help="Comma-separated schedule names (default: all three).",
    )
    track.add_argument(
        "--tracking-uri", default=None,
        help="Defaults to $MLFLOW_TRACKING_URI, else MLflow's own local default (./mlruns).",
    )
    track.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME)
    track.add_argument("--run-name", default=None)

    list_cmd = sub.add_parser("list-runs", help="List recent tracked runs in one MLflow experiment.")
    list_cmd.add_argument("--tracking-uri", default=None)
    list_cmd.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME)
    list_cmd.add_argument("--max-results", type=int, default=20)

    bench = sub.add_parser(
        "benchmark",
        help="Run a real PyTorch/ONNX Runtime benchmark for a Core preset experiment.",
    )
    bench.add_argument("--workload-preset", required=True)
    bench.add_argument("--accelerator-preset", required=True)
    bench.add_argument("--tile-m-values", required=True)
    bench.add_argument("--tile-n-values", required=True)
    bench.add_argument("--tile-k-values", required=True)
    bench.add_argument("--schedule-values", default=None)
    bench.add_argument("--backend", choices=["pytorch", "onnxruntime"], default="pytorch")
    bench.add_argument("--device", default="cpu")
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iterations", type=int, default=50)
    bench.add_argument("--tracking-uri", default=None)
    bench.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME)
    bench.add_argument("--run-name", default=None)
    bench.add_argument(
        "--no-track", action="store_true",
        help="Run the benchmark without logging to MLflow (prints the summary only).",
    )

    return parser


def _build_spec_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> ExperimentSpec:
    tile_m_values = _parse_int_list(parser, "--tile-m-values", args.tile_m_values)
    tile_n_values = _parse_int_list(parser, "--tile-n-values", args.tile_n_values)
    tile_k_values = _parse_int_list(parser, "--tile-k-values", args.tile_k_values)
    schedule_names = (
        tuple(s.strip() for s in args.schedule_values.split(",")) if args.schedule_values else None
    )

    name = f"{args.workload_preset}-on-{args.accelerator_preset}"
    try:
        return ExperimentSpec(
            name=name,
            workload_preset=args.workload_preset,
            accelerator_preset=args.accelerator_preset,
            tile_m_values=tuple(tile_m_values),
            tile_n_values=tuple(tile_n_values),
            tile_k_values=tuple(tile_k_values),
            schedule_names=schedule_names,
        )
    except ValueError as exc:
        parser.error(str(exc))


def _run_track(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    spec = _build_spec_from_args(parser, args)

    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    tracking = TrackingConfig(tracking_uri=tracking_uri, experiment_name=args.experiment)

    # A ValueError here is a Core failure (e.g. no feasible mapping); any
    # other exception is a tracking/MLflow failure and is left uncaught so
    # it is never confused with, or silently converted into, a Core error.
    try:
        tracked = track_experiment(spec, tracking, run_name=args.run_name)
    except ValueError as exc:
        parser.error(str(exc))

    print("TensorForge Ops")
    print()
    print("Core experiment")
    print(f"  workload          {args.workload_preset}")
    print(f"  accelerator       {args.accelerator_preset}")
    print(f"  fingerprint       {tracked.result_fingerprint}")
    print()
    print("MLflow")
    print(f"  experiment        {tracked.experiment_name}")
    print(f"  run id            {tracked.run_id}")
    print(f"  tracking URI      {tracked.tracking_uri or '(MLflow default: ./mlruns)'}")
    print()
    print("Tracked successfully.")


def _run_list(args: argparse.Namespace) -> None:
    tracking_uri = resolve_tracking_uri(args.tracking_uri)
    tracking = TrackingConfig(tracking_uri=tracking_uri, experiment_name=args.experiment)
    runs = list_runs(tracking, max_results=args.max_results)

    if not runs:
        print(f"No runs found in experiment {args.experiment!r}.")
        return

    for run in runs:
        print(
            f"{run['run_id']}  {run['workload_preset']:<20} {run['accelerator_preset']:<16} "
            f"perfect_overlap={run['perfect_overlap_time_seconds']}  "
            f"fingerprint={run['result_fingerprint']}"
        )


def _format_seconds(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def _run_benchmark(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    spec = _build_spec_from_args(parser, args)

    try:
        result = run_experiment(spec)
    except ValueError as exc:
        parser.error(str(exc))
    fingerprint = compute_result_fingerprint(result)
    preset = get_workload_preset(args.workload_preset)

    tracking = None
    tracked = None
    if not args.no_track:
        tracking_uri = resolve_tracking_uri(args.tracking_uri)
        tracking = TrackingConfig(tracking_uri=tracking_uri, experiment_name=args.experiment)
        try:
            tracked = track_result(result, tracking, run_name=args.run_name)
        except ValueError as exc:
            parser.error(str(exc))
        assert tracked.result_fingerprint == fingerprint  # same result, same fingerprint, by construction

    config = BenchmarkConfig(
        backend=args.backend, device=args.device,
        warmup_iterations=args.warmup, measured_iterations=args.iterations,
    )

    if args.backend == "pytorch":
        from tensorforge_ops.benchmark_pytorch import run_pytorch_benchmark
        try:
            benchmark_result = run_pytorch_benchmark(preset, fingerprint, config)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        parser.error(
            "--backend onnxruntime is DEFERRED: this milestone implements the PyTorch "
            "benchmark backend only. Use --backend pytorch."
        )

    if tracking is not None and tracked is not None:
        log_benchmark_result(tracking, tracked.run_id, benchmark_result)

    stats = benchmark_result.statistics
    print("TensorForge Benchmark")
    print()
    print("Core")
    print(f"  workload                   {args.workload_preset}")
    print(f"  accelerator                {args.accelerator_preset}")
    print(f"  fingerprint                {fingerprint}")
    print()
    print("Measured")
    print(f"  backend                    {benchmark_result.backend}")
    print(f"  device                     {benchmark_result.device}")
    print(f"  dtype                      {benchmark_result.dtype}")
    print(f"  warmup                     {benchmark_result.warmup_iterations}")
    print(f"  iterations                 {benchmark_result.measured_iterations}")
    print()
    print("Latency")
    print(f"  mean                       {_format_seconds(stats.mean_seconds)}")
    print(f"  p50                        {_format_seconds(stats.p50_seconds)}")
    print(f"  p95                        {_format_seconds(stats.p95_seconds) if stats.p95_seconds is not None else 'n/a (< 20 samples)'}")
    print(f"  p99                        {_format_seconds(stats.p99_seconds) if stats.p99_seconds is not None else 'n/a (< 100 samples)'}")
    print(f"  min                        {_format_seconds(stats.min_seconds)}")
    print(f"  max                        {_format_seconds(stats.max_seconds)}")
    print()
    print("Throughput")
    print(f"  invocations/sec            {stats.throughput_per_second:.2f}")
    print()
    print("Memory")
    if benchmark_result.peak_memory_allocated_bytes is not None:
        print(f"  peak allocated             {benchmark_result.peak_memory_allocated_bytes:,} bytes")
    else:
        print("  peak allocated             n/a (not available for this device)")

    if tracked is not None:
        print()
        print("MLflow")
        print(f"  experiment                 {tracked.experiment_name}")
        print(f"  run id                     {tracked.run_id}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "track":
        _run_track(parser, args)
    elif args.command == "list-runs":
        _run_list(args)
    elif args.command == "benchmark":
        _run_benchmark(parser, args)


if __name__ == "__main__":
    main()
