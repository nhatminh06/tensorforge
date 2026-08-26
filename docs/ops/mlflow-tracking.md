# MLflow tracking reference

## Core / Ops boundary

`tensorforge.experiments.run_experiment(spec)` produces an immutable
`ExperimentResult`. `tensorforge_ops.tracking` consumes that result and
logs it to MLflow — it never recomputes a metric, never adjusts a value
to "look better" in MLflow, and never mutates the `ExperimentResult` it
was given (`test_tracking.py::test_tracking_does_not_mutate_core_result`
verifies this directly).

```python
result = run_experiment(spec)          # tensorforge (Core)
tracked = track_result(result, tracking)  # tensorforge_ops (Ops)
```

`track_experiment(spec, tracking)` is a convenience wrapper that calls
both in sequence — but `track_result()` alone accepts any already-computed
`ExperimentResult`, so a result produced elsewhere (a saved JSON file, a
CI job, a future benchmark worker) can be tracked without re-running the
Core model.

## Local MLflow server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then in another shell:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m tensorforge_ops track --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128
```

`--host 127.0.0.1` binds to localhost only — never `0.0.0.0` by default.
No authentication is configured (out of scope for this milestone).

## Tracking URI resolution

Precedence, implemented by `resolve_tracking_uri()`:

1. explicit `--tracking-uri` (CLI) / `tracking_uri` argument (API)
2. `MLFLOW_TRACKING_URI` environment variable
3. MLflow's own default (a local `./mlruns` directory) — TensorForge Ops
   does not invent or hard-code a different default path.

If an explicit tracking URI (e.g. `http://127.0.0.1:5000`) is unreachable,
tracking **fails with an MLflow/connection error** — there is no silent
fallback to a different backend. An experiment appearing to succeed in
the wrong place would be worse than a clear failure.

For local testing without a running server, an MLflow-supported local
backend URI works directly, e.g. `sqlite:///$(pwd)/mlflow.db` — this is
what the automated test suite uses (in a pytest `tmp_path`, never
`mlflow.db`/`mlruns/` inside the repository; both are also in
`.gitignore` as a safety net for manual testing).

## Tracking configuration

```python
TrackingConfig(tracking_uri, experiment_name="tensorforge")
```

Experiment name is created/selected via `mlflow.set_experiment()` — no
custom experiment database.

## Result fingerprint

```python
compute_result_fingerprint(result) -> "sha256:<hex>"
```

SHA-256 of `result.to_json()` (the canonical deterministic Core JSON).
This identifies **a deterministic Core analytical result** — not a
machine environment, not an MLflow run, and not (later) a real hardware
benchmark. Running the exact same `ExperimentSpec` twice produces two
different MLflow run IDs but the identical fingerprint; changing any
analytical input (a different accelerator preset, a different tile
candidate set) changes the fingerprint.

## What gets logged

### Parameters (`mlflow.log_params`)

`workload_preset`, `workload_kind`, `accelerator_preset`,
`tile_m_values`, `tile_n_values`, `tile_k_values` (comma-separated,
sorted-deduplicated, e.g. `"32,64,128"`), `schedules` (comma-separated),
`pe_rows`, `pe_cols`, `sram_bytes`, `clock_hz`,
`bandwidth_bytes_per_second`, `schema_version`.

### Metrics (`mlflow.log_metrics`) — only when present in the Core result

`modeled_macs`, `modeled_flops`, `modeled_dram_bytes`,
`serialized_time_seconds`, `perfect_overlap_time_seconds`,
`effective_arithmetic_intensity` (GEMM/Conv2D only),
`im2col_expansion_ratio` (Conv2D only). A metric that Core did not
produce for a given workload kind (e.g. `effective_arithmetic_intensity`
for Transformer/CNN, which report `largest_time_contributor` instead) is
simply omitted — never logged as `0.0`.

### Tags (`mlflow.set_tags`)

`tensorforge.phase="ops"`, `tensorforge.core_schema_version`,
`tensorforge.workload_kind`, `tensorforge.result_fingerprint`,
`git.commit`, `git.branch`, `git.dirty` (`"true"`/`"false"`/`"unknown"`),
`source="tensorforge-core"`, and — only when present — categorical
per-workload-kind results as tags (MLflow metrics must be numeric):
`tensorforge.bottleneck` (GEMM/Conv2D), `tensorforge.largest_time_contributor`
and `tensorforge.largest_dram_contributor` (Transformer/CNN).

### Artifacts (`mlflow.log_artifact`)

- `core-result.json` — byte-identical to `ExperimentResult.to_json()`.
  No MLflow metadata is inserted into it
  (`test_tracking.py::test_logged_artifact_is_byte_identical_to_core_json`).
- `tracking-metadata.json` — MLflow experiment name/ID, Git metadata,
  tracking adapter schema version. Kept entirely separate from the Core
  result; no wall-clock timestamp is injected into the *Core* JSON either
  way (Core's own determinism guarantee is untouched).

## Git metadata

Read via `git rev-parse HEAD` / `--abbrev-ref HEAD` / `status --porcelain`,
run against a fixed, non-user-supplied argv list (no shell, no injection
surface) with a 5-second timeout. If Git is unavailable or this isn't a
Git working tree, fields fall back to `"unknown"` rather than failing the
whole tracked run. A dirty working tree is recorded as `git.dirty=true`
and tracking still proceeds — never silently pretending the run came from
a clean revision. No environment variables, credentials, remote URLs, or
machine details are logged.

## Source-of-truth rule

**`core-result.json` is the complete, authoritative analytical result.**
MLflow parameters/metrics/tags are an index built from it, useful for
searching and comparing runs in the UI — never a second, independently
computed result. If you need the full per-operation mapping detail (e.g.
every Transformer operation's selected tile), read the artifact; the
tracked metrics intentionally summarize, not replace, it.

## Comparison workflow

In the MLflow UI: select an experiment, check multiple runs, use
"Compare" to see parameters/metrics side by side, sorted/filtered by any
logged param or metric (e.g. filter by `accelerator_preset`, sort by
`perfect_overlap_time_seconds`). `python -m tensorforge_ops list-runs
--experiment <name>` gives the same comparison from the terminal without
opening a browser.

## Future extension (not implemented in this milestone)

```
TensorForge Core result
      ↓
MLflow tracking            (this milestone)
      ↓
future real benchmark       (Milestone 12+)
      ↓
predicted vs. measured      (Milestone 13)
      ↓
future regression gate      (Milestone 14)
```

None of the arrows below "MLflow tracking" exist yet. When a real
benchmark result is added later, it must be recorded as a clearly
separate, explicitly labeled measured value — never merged into a field
that currently means "modeled" or "analytical."

## Limitations

- Local-first only: no remote artifact storage (S3/MinIO/GCS/Azure), no
  authentication, no containerization.
- No model registry — there is no trained model here to register.
- No system/GPU telemetry (`mlflow.autolog()` and system metrics
  collection are intentionally not used).
- No nested runs — one TensorForge experiment maps to exactly one MLflow
  run.
- Everything logged is an **analytical prediction**, never a
  "benchmark" or "measurement" — see `docs/limitations.md` for the full
  Core limitations, all of which still apply.
