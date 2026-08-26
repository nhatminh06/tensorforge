"""TensorForge Ops CLI: track a Core experiment preset run in MLflow.

This is a deliberately separate entry point from `python -m tensorforge`.
The Core CLI remains entirely about running/inspecting analytical
experiments; this CLI owns MLflow tracking and only accepts the compact
preset-based configuration (workload preset, accelerator preset, tile
candidates) rather than exposing every raw Core CLI flag.
"""

import argparse
import json
import os
import sys

from tensorforge.experiments import ExperimentSpec, run_experiment
from tensorforge.presets import get_workload_preset

from tensorforge_ops.benchmark import BenchmarkConfig, load_benchmark_result
from tensorforge_ops.calibration import (
    CalibrationConfig,
    calibration_profile_from_dict,
    compute_calibration_fingerprint,
    predict,
    summarize_validation_results,
    validate_prediction,
)
from tensorforge_ops.regression import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_NOT_COMPARABLE,
    compare_benchmark_results,
    load_regression_policy,
    render_markdown_report,
)
from tensorforge_ops.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    TrackingConfig,
    compute_result_fingerprint,
    list_runs,
    log_benchmark_result,
    log_calibration_profile,
    log_validation_result,
    log_validation_summary,
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
    bench.add_argument("--output-json", default=None, help="Path to save the BenchmarkResult JSON.")
    bench.add_argument("--force", action="store_true", help="Overwrite --output-json if it already exists.")

    calibrate = sub.add_parser(
        "calibrate",
        help="Measure this device's sustained compute/memory-copy rates and save a calibration profile.",
    )
    calibrate.add_argument("--backend", choices=["pytorch"], default="pytorch")
    calibrate.add_argument("--device", default="cpu")
    calibrate.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    calibrate.add_argument("--compute-m", type=int, default=2048)
    calibrate.add_argument("--compute-n", type=int, default=2048)
    calibrate.add_argument("--compute-k", type=int, default=2048)
    calibrate.add_argument("--memory-probe-mib", type=int, default=64)
    calibrate.add_argument("--warmup", type=int, default=10)
    calibrate.add_argument("--iterations", type=int, default=50)
    calibrate.add_argument("--output", default=None, help="Path to save the calibration profile JSON.")
    calibrate.add_argument(
        "--force", action="store_true", help="Overwrite --output if it already exists.",
    )

    validate = sub.add_parser(
        "validate",
        help="Validate a calibrated analytical prediction against one real benchmark measurement.",
    )
    validate.add_argument("--calibration", required=True, help="Path to a saved calibration profile JSON.")
    validate.add_argument("--workload-preset", required=True)
    validate.add_argument("--accelerator-preset", required=True)
    validate.add_argument("--tile-m-values", required=True)
    validate.add_argument("--tile-n-values", required=True)
    validate.add_argument("--tile-k-values", required=True)
    validate.add_argument("--schedule-values", default=None)
    validate.add_argument("--backend", choices=["pytorch", "onnxruntime"], default="pytorch")
    validate.add_argument("--device", default="cpu")
    validate.add_argument("--warmup", type=int, default=10)
    validate.add_argument("--iterations", type=int, default=50)
    validate.add_argument("--tracking-uri", default=None)
    validate.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME)
    validate.add_argument("--run-name", default=None)
    validate.add_argument(
        "--no-track", action="store_true",
        help="Run validation without logging to MLflow (prints the summary only).",
    )

    validate_suite = sub.add_parser(
        "validate-suite",
        help="Validate several workload presets against one calibration profile and summarize error.",
    )
    validate_suite.add_argument("--calibration", required=True)
    validate_suite.add_argument(
        "--workload-presets", required=True,
        help="Comma-separated workload preset names, e.g. gemm_tiny,conv_spatial,transformer_small.",
    )
    validate_suite.add_argument("--accelerator-preset", required=True)
    validate_suite.add_argument("--tile-m-values", required=True)
    validate_suite.add_argument("--tile-n-values", required=True)
    validate_suite.add_argument("--tile-k-values", required=True)
    validate_suite.add_argument("--schedule-values", default=None)
    validate_suite.add_argument("--backend", choices=["pytorch", "onnxruntime"], default="pytorch")
    validate_suite.add_argument("--device", default="cpu")
    validate_suite.add_argument("--warmup", type=int, default=10)
    validate_suite.add_argument("--iterations", type=int, default=50)
    validate_suite.add_argument("--tracking-uri", default=None)
    validate_suite.add_argument("--experiment", default=DEFAULT_EXPERIMENT_NAME)
    validate_suite.add_argument(
        "--no-track", action="store_true",
        help="Run the suite without logging to MLflow (prints the summary only).",
    )
    validate_suite.add_argument("--output", default=None, help="Path to save the validation-summary JSON.")
    validate_suite.add_argument(
        "--force", action="store_true", help="Overwrite --output if it already exists.",
    )

    regression = sub.add_parser(
        "regression",
        help="Compare a baseline and candidate BenchmarkResult against an explicit regression policy.",
    )
    regression.add_argument("--baseline", required=True, help="Path to the baseline BenchmarkResult JSON.")
    regression.add_argument("--candidate", required=True, help="Path to the candidate BenchmarkResult JSON.")
    regression.add_argument("--policy", required=True, help="Path to a RegressionPolicy JSON.")
    regression.add_argument("--output-json", default=None, help="Path to save the RegressionResult JSON.")
    regression.add_argument("--output-markdown", default=None, help="Path to save the Markdown report.")
    regression.add_argument(
        "--force", action="store_true", help="Overwrite --output-json/--output-markdown if they already exist.",
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

    if args.output_json is not None:
        _write_output_file(parser, args.output_json, benchmark_result.to_json(), args.force)

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

    if args.output_json is not None:
        print()
        print(f"Saved to {args.output_json}")

    if tracked is not None:
        print()
        print("MLflow")
        print(f"  experiment                 {tracked.experiment_name}")
        print(f"  run id                     {tracked.run_id}")


def _format_signed_seconds(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    return sign + _format_seconds(abs(seconds))


def _write_output_file(parser: argparse.ArgumentParser, path: str, text: str, force: bool) -> None:
    if os.path.exists(path) and not force:
        parser.error(f"{path} already exists; pass --force to overwrite")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _load_calibration_profile(parser: argparse.ArgumentParser, path: str):
    if not os.path.exists(path):
        parser.error(f"calibration profile not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return calibration_profile_from_dict(data)


def _run_calibrate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    config = CalibrationConfig(
        backend=args.backend, device=args.device, dtype=args.dtype,
        compute_m=args.compute_m, compute_n=args.compute_n, compute_k=args.compute_k,
        memory_probe_mib=args.memory_probe_mib,
        warmup_iterations=args.warmup, measured_iterations=args.iterations,
    )
    from tensorforge_ops.calibration_pytorch import run_pytorch_calibration
    try:
        profile = run_pytorch_calibration(config)
    except ValueError as exc:
        parser.error(str(exc))
    fingerprint = compute_calibration_fingerprint(profile)

    if args.output is not None:
        _write_output_file(parser, args.output, profile.to_json(), args.force)

    cprobe = profile.compute_probe
    mprobe = profile.memory_probe
    device_label = profile.device_type + (f":{profile.device_index}" if profile.device_index is not None else "")
    print("TensorForge Device Calibration")
    print()
    print("Device")
    print(f"  backend                    {profile.backend}")
    print(f"  device                     {device_label}")
    print(f"  name                       {profile.device_name or 'n/a'}")
    print(f"  dtype                      {profile.dtype}")
    print()
    print("Compute probe")
    print(f"  shape                      {cprobe.shape_m}x{cprobe.shape_n}x{cprobe.shape_k}")
    print(f"  p50                        {_format_seconds(cprobe.p50_seconds)}")
    print(f"  sustained compute          {cprobe.effective_compute_flops_per_second / 1e12:.3f} TFLOP/s")
    print()
    print("Memory probe")
    print(f"  payload                    {mprobe.payload_bytes / (1024 * 1024):.1f} MiB")
    print(f"  modeled copy traffic       {mprobe.modeled_copy_traffic_bytes / (1024 * 1024):.1f} MiB")
    print(f"  p50                        {_format_seconds(mprobe.p50_seconds)}")
    print(f"  sustained bandwidth        {mprobe.effective_memory_bandwidth_bytes_per_second / 1e9:.3f} GB/s")
    print()
    print("Profile")
    print(f"  fingerprint                {fingerprint}")
    print(f"  output                     {args.output or '(not saved)'}")


def _print_validation_result(workload_preset: str, workload_kind: str, profile, validation) -> None:
    print("TensorForge Validation")
    print()
    print("Workload")
    print(f"  preset                     {workload_preset}")
    print(f"  kind                       {workload_kind}")
    print()
    print("Physical device")
    print(f"  {profile.device_name or profile.device_type}")
    print(f"  PyTorch {profile.runtime_metadata.get('torch_version', 'unknown')}")
    print(f"  dtype                      {profile.dtype}")
    print()
    print("Calibration")
    print(f"  compute ceiling            {profile.effective_compute_flops_per_second / 1e12:.3f} TFLOP/s")
    print(f"  memory bandwidth           {profile.effective_memory_bandwidth_bytes_per_second / 1e9:.3f} GB/s")
    print()
    print("Prediction")
    print(f"  compute                    {_format_seconds(validation.predicted_compute_seconds)}")
    print(f"  memory                     {_format_seconds(validation.predicted_memory_seconds)}")
    print(f"  predicted                  {_format_seconds(validation.predicted_latency_seconds)}")
    print(f"  predicted bound            {validation.predicted_bottleneck}")
    print()
    print("Measurement")
    print(f"  p50                        {_format_seconds(validation.measured_p50_latency_seconds)}")
    print(f"  mean                       {_format_seconds(validation.measured_mean_latency_seconds)}")
    p95 = validation.measured_p95_latency_seconds
    print(f"  p95                        {_format_seconds(p95) if p95 is not None else 'n/a (< 20 samples)'}")
    print()
    print("Error vs p50")
    print(f"  signed                     {_format_signed_seconds(validation.signed_error_seconds)}")
    print(f"  absolute                   {_format_seconds(validation.absolute_error_seconds)}")
    print(f"  APE                        {validation.absolute_percentage_error:.1f}%")
    print(f"  measured/predicted         {validation.measured_to_predicted_ratio:.2f}x")


def _run_validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    spec = _build_spec_from_args(parser, args)

    try:
        core_result = run_experiment(spec)
    except ValueError as exc:
        parser.error(str(exc))
    core_fingerprint = compute_result_fingerprint(core_result)
    preset = get_workload_preset(args.workload_preset)
    profile = _load_calibration_profile(parser, args.calibration)

    tracking = None
    tracked = None
    if not args.no_track:
        tracking_uri = resolve_tracking_uri(args.tracking_uri)
        tracking = TrackingConfig(tracking_uri=tracking_uri, experiment_name=args.experiment)
        try:
            tracked = track_result(core_result, tracking, run_name=args.run_name)
        except ValueError as exc:
            parser.error(str(exc))

    bconfig = BenchmarkConfig(
        backend=args.backend, device=args.device,
        warmup_iterations=args.warmup, measured_iterations=args.iterations,
    )
    if args.backend == "pytorch":
        from tensorforge_ops.benchmark_pytorch import run_pytorch_benchmark
        try:
            benchmark_result = run_pytorch_benchmark(preset, core_fingerprint, bconfig)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        parser.error(
            "--backend onnxruntime is DEFERRED: this milestone implements the PyTorch "
            "benchmark backend only. Use --backend pytorch."
        )

    try:
        prediction = predict(preset, profile)
        validation = validate_prediction(profile, prediction, core_fingerprint, benchmark_result)
    except ValueError as exc:
        parser.error(str(exc))

    if tracking is not None and tracked is not None:
        log_benchmark_result(tracking, tracked.run_id, benchmark_result)
        log_calibration_profile(tracking, tracked.run_id, profile)
        log_validation_result(tracking, tracked.run_id, validation)

    _print_validation_result(args.workload_preset, preset.kind, profile, validation)

    if tracked is not None:
        print()
        print("MLflow")
        print(f"  experiment                 {tracked.experiment_name}")
        print(f"  run id                     {tracked.run_id}")


def _run_validate_suite(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    profile = _load_calibration_profile(parser, args.calibration)
    workload_presets = [p.strip() for p in args.workload_presets.split(",") if p.strip()]
    if not workload_presets:
        parser.error("--workload-presets must contain at least one preset name")

    tile_m_values = _parse_int_list(parser, "--tile-m-values", args.tile_m_values)
    tile_n_values = _parse_int_list(parser, "--tile-n-values", args.tile_n_values)
    tile_k_values = _parse_int_list(parser, "--tile-k-values", args.tile_k_values)
    schedule_names = (
        tuple(s.strip() for s in args.schedule_values.split(",")) if args.schedule_values else None
    )

    tracking = None
    if not args.no_track:
        tracking_uri = resolve_tracking_uri(args.tracking_uri)
        tracking = TrackingConfig(tracking_uri=tracking_uri, experiment_name=args.experiment)

    bconfig = BenchmarkConfig(
        backend=args.backend, device=args.device,
        warmup_iterations=args.warmup, measured_iterations=args.iterations,
    )

    rows = []
    for preset_name in workload_presets:
        preset = get_workload_preset(preset_name)
        preset_dtype = preset.dtype.name.lower()
        if preset_dtype != profile.dtype:
            print(f"skipping {preset_name!r}: preset dtype {preset_dtype!r} != calibration dtype {profile.dtype!r}")
            continue

        try:
            core_spec = ExperimentSpec(
                name=f"{preset_name}-on-{args.accelerator_preset}",
                workload_preset=preset_name, accelerator_preset=args.accelerator_preset,
                tile_m_values=tuple(tile_m_values), tile_n_values=tuple(tile_n_values),
                tile_k_values=tuple(tile_k_values), schedule_names=schedule_names,
            )
            core_result = run_experiment(core_spec)
        except ValueError as exc:
            parser.error(str(exc))
        core_fingerprint = compute_result_fingerprint(core_result)

        tracked = None
        if tracking is not None:
            try:
                tracked = track_result(core_result, tracking, run_name=preset_name)
            except ValueError as exc:
                parser.error(str(exc))

        if args.backend == "pytorch":
            from tensorforge_ops.benchmark_pytorch import run_pytorch_benchmark
            try:
                benchmark_result = run_pytorch_benchmark(preset, core_fingerprint, bconfig)
            except ValueError as exc:
                parser.error(str(exc))
        else:
            parser.error(
                "--backend onnxruntime is DEFERRED: this milestone implements the PyTorch "
                "benchmark backend only. Use --backend pytorch."
            )

        try:
            prediction = predict(preset, profile)
            validation = validate_prediction(profile, prediction, core_fingerprint, benchmark_result)
        except ValueError as exc:
            parser.error(str(exc))

        if tracking is not None and tracked is not None:
            log_benchmark_result(tracking, tracked.run_id, benchmark_result)
            log_calibration_profile(tracking, tracked.run_id, profile)
            log_validation_result(tracking, tracked.run_id, validation)

        rows.append((preset_name, validation))

    if not rows:
        parser.error("no compatible workload presets to validate (dtype mismatch for all requested presets)")

    summary = summarize_validation_results(tuple(v for _, v in rows))

    print("Validation summary")
    print()
    print(f"{'Workload':<22}{'Predicted':>12}{'Measured p50':>15}{'APE':>9}   {'Bound':<15}")
    for preset_name, v in rows:
        print(
            f"{preset_name:<22}{_format_seconds(v.predicted_latency_seconds):>12}"
            f"{_format_seconds(v.measured_p50_latency_seconds):>15}{v.absolute_percentage_error:>8.1f}%   "
            f"{v.predicted_bottleneck:<15}"
        )
    print()
    print("Overall")
    print(f"  median APE                 {summary.median_absolute_percentage_error:.1f}%")
    print(f"  mean APE                   {summary.mean_absolute_percentage_error:.1f}%")
    print(f"  max APE                    {summary.max_absolute_percentage_error:.1f}%")
    print()
    print("By workload kind")
    for kind, kind_stats in sorted(summary.by_workload_kind.items()):
        print(f"  {kind:<15}median APE {kind_stats['median_absolute_percentage_error']:.1f}%  (n={kind_stats['count']})")
    print()
    print("By predicted bound")
    for bound, bound_stats in sorted(summary.by_predicted_bottleneck.items()):
        print(f"  {bound:<15}median APE {bound_stats['median_absolute_percentage_error']:.1f}%  (n={bound_stats['count']})")

    if args.output is not None:
        _write_output_file(parser, args.output, summary.to_json(), args.force)
        print()
        print(f"Summary saved to {args.output}")

    if tracking is not None:
        run_id = log_validation_summary(tracking, summary)
        print()
        print("MLflow")
        print(f"  experiment                 {tracking.experiment_name}")
        print(f"  summary run id             {run_id}")


def _format_metric_for_cli(metric: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    if metric.endswith("_latency_seconds"):
        return _format_seconds(value)
    if metric == "peak_memory_allocated_bytes":
        return f"{value / (1024 * 1024):.1f} MiB"
    if metric == "throughput_per_second":
        return f"{value:.1f}/s"
    return str(value)


def _run_regression(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Exit-code contract (checked by tests, relied on by CI):
        0 -> guard passed (RegressionResult.status == PASS)
        1 -> operational/configuration/comparison error (bad input files,
             incompatible schema, mismatched workload/device/backend/dtype)
        2 -> valid comparison, but the regression policy failed
    This lets CI distinguish "the code got slower" from "the benchmark
    infrastructure is broken" -- the two must never be conflated.
    """
    try:
        baseline = load_benchmark_result(args.baseline)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: could not load --baseline {args.baseline!r}: {exc}", file=sys.stderr)
        return 1
    try:
        candidate = load_benchmark_result(args.candidate)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: could not load --candidate {args.candidate!r}: {exc}", file=sys.stderr)
        return 1
    try:
        policy = load_regression_policy(args.policy)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: could not load --policy {args.policy!r}: {exc}", file=sys.stderr)
        return 1

    result = compare_benchmark_results(baseline, candidate, policy)
    report = render_markdown_report(result)

    # Deliberately NOT _write_output_file()/parser.error() here: argparse
    # always exits 2, which would collide with this command's own "2 ==
    # regression FAIL" convention. An overwrite-protection violation is an
    # operational error (exit 1), never a regression failure (exit 2).
    for path, text in ((args.output_json, result.to_json()), (args.output_markdown, report)):
        if path is None:
            continue
        if os.path.exists(path) and not args.force:
            print(f"error: {path} already exists; pass --force to overwrite", file=sys.stderr)
            return 1
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    print("TensorForge Performance Regression Guard")
    print()

    if result.status == STATUS_ERROR:
        print("Result: ERROR")
        print()
        print(result.error_message)
        return 1

    print(f"Workload: {result.workload_preset}")
    print()
    print(f"Result: {result.status}")
    print()
    for c in result.metric_comparisons:
        if c.status == STATUS_NOT_COMPARABLE:
            continue
        base = _format_metric_for_cli(c.metric, c.baseline_value)
        candidate_str = _format_metric_for_cli(c.metric, c.candidate_value)
        change = f"{c.relative_delta * 100:+.1f}%" if c.relative_delta is not None else "n/a"
        print(f"{c.metric}:")
        print(f"  {base} -> {candidate_str} ({change})  [{c.status}]")
    print()

    if args.output_markdown is not None:
        print(f"Report: {args.output_markdown}")

    return 2 if result.status == STATUS_FAIL else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "track":
        _run_track(parser, args)
    elif args.command == "list-runs":
        _run_list(args)
    elif args.command == "benchmark":
        _run_benchmark(parser, args)
    elif args.command == "calibrate":
        _run_calibrate(parser, args)
    elif args.command == "validate":
        _run_validate(parser, args)
    elif args.command == "validate-suite":
        _run_validate_suite(parser, args)
    elif args.command == "regression":
        return _run_regression(parser, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
