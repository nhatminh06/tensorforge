"""TensorForge Ops CLI: track a Core experiment preset run in MLflow.

This is a deliberately separate entry point from `python -m tensorforge`.
The Core CLI remains entirely about running/inspecting analytical
experiments; this CLI owns MLflow tracking and only accepts the compact
preset-based configuration (workload preset, accelerator preset, tile
candidates) rather than exposing every raw Core CLI flag.
"""

import argparse

from tensorforge.experiments import ExperimentSpec

from tensorforge_ops.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    TrackingConfig,
    list_runs,
    resolve_tracking_uri,
    track_experiment,
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

    return parser


def _run_track(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    tile_m_values = _parse_int_list(parser, "--tile-m-values", args.tile_m_values)
    tile_n_values = _parse_int_list(parser, "--tile-n-values", args.tile_n_values)
    tile_k_values = _parse_int_list(parser, "--tile-k-values", args.tile_k_values)
    schedule_names = (
        tuple(s.strip() for s in args.schedule_values.split(",")) if args.schedule_values else None
    )

    name = f"{args.workload_preset}-on-{args.accelerator_preset}"
    try:
        spec = ExperimentSpec(
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


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "track":
        _run_track(parser, args)
    elif args.command == "list-runs":
        _run_list(args)


if __name__ == "__main__":
    main()
