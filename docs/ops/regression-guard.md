# PR performance regression guard reference

## What this answers, and what it does not

The guard answers exactly one question: **did measured inference
performance get meaningfully worse from a PR's base commit to its head
commit, on this benchmark environment, under this explicit policy?**

It does **not** answer "is this implementation fast" -- a candidate can
PASS the regression guard while being objectively slow, and a candidate
can FAIL while still being faster than some other, unrelated
implementation. It compares one commit to another, nothing more.

## Why measured, not predicted

The guard gates on **real, measured** `BenchmarkResult` data (Milestone
12), never on TensorForge's predicted/calibrated analytical latency
(Milestone 13). Milestone 13's own holdout validation showed Core
prediction error can be large -- tiny GEMMs, materialized-im2col Conv2D,
and multi-launch Transformer workloads all showed APE well above 50% on
the real RTX 3050 -- so an analytical prediction is not trustworthy
enough to gate a merge. Any Core/calibration data that does appear in a
regression report is explicitly labeled "Analytical context": it may
help explain a measured change, and it never decides PASS/FAIL.

## Comparison identity

Before comparing any metric, `compare_benchmark_results()` requires the
two `BenchmarkResult`s to describe the same benchmark semantics:
`workload_preset`, `workload_kind`, `backend`, device type (`cpu`/
`cuda`), `dtype`, and (for CUDA) `device_index`/`device_name`. Any
mismatch returns a `RegressionResult` with `status = "ERROR"` and a
specific reason -- never a performance verdict. `gemm_tiny` is never
compared against `conv_spatial`; an RTX 3050 result is never compared
against a different physical GPU.

Iteration counts (`warmup_iterations`/`measured_iterations`) are
deliberately **not** part of identity -- a different `--iterations`
value between two runs doesn't change what workload was measured, and
both counts are recorded in the result for transparency.

A **different PyTorch version** between baseline and candidate is
**not** automatically rejected -- a PR may intentionally bump a
dependency, and that could be exactly the source of a real, legitimate
performance change. The comparison proceeds, and the runtime-version
change is surfaced prominently in the report ("Runtime change" section)
so a reviewer sees it plainly.

A **benchmark_schema_version** mismatch, however, always fails to load
(`benchmark_result_from_dict()` raises before comparison even starts) --
a schema change means measurement semantics themselves changed, so
silent conversion is never attempted.

## Metric direction

| Metric | Direction | Note |
|---|---|---|
| `p50_latency_seconds` | higher = worse | primary latency metric |
| `p95_latency_seconds` | higher = worse | tail latency |
| `mean_latency_seconds`, `p99_latency_seconds` | higher = worse | optional, not required by default |
| `throughput_per_second` | **lower = worse** | opposite sign convention from latency |
| `peak_memory_allocated_bytes` | higher = worse | CUDA only |

This asymmetry is implemented explicitly per metric
(`_HIGHER_IS_WORSE`/`_LOWER_IS_WORSE` in `regression.py`) -- there is no
single shared sign rule applied to every metric.

## Regression policy and thresholds

A `RegressionPolicy` (`regression_policy_schema_version = 1`, a schema
distinct from `BenchmarkResult`'s or Core's) maps metric names to a
`MetricPolicy`:

```json
{
  "regression_policy_schema_version": 1,
  "metrics": {
    "p50_latency_seconds": {
      "max_relative_regression": 0.10,
      "max_absolute_regression_seconds": 0.00001
    },
    "throughput_per_second": {
      "max_relative_regression": 0.10
    }
  }
}
```

Every threshold is explicit, loaded from this file -- there is no hidden
constant anywhere in the comparison code, and no threshold is ever
learned automatically from the PR being checked (that would let a
regression redefine its own passing bar).

**Gate semantics** (latency/memory, higher-is-worse):

```
allowed_regression = max(
    configured_absolute_allowance,      # if set
    |baseline| * configured_relative_allowance,   # if set
)
FAIL if (candidate - baseline) > allowed_regression
```

**Gate semantics** (throughput, lower-is-worse):

```
allowed_drop = baseline * configured_relative_regression
FAIL if (baseline - candidate) > allowed_drop
```

Combining a relative and an absolute allowance avoids noisy failures on
tiny workloads: baseline=10us, candidate=11us is a "huge" +10% but a
tiny +1us -- with `max_relative_regression=0.05` and
`max_absolute_regression_seconds=2e-6`, the allowed regression is
`max(0.5us, 2us) = 2us`, so this passes.

## Metric status values

Each `MetricComparison` carries its own status, never hidden behind one
overall boolean:

- **PASS** / **FAIL** -- both values present, gated by policy, compared.
- **NOT_COMPARABLE** -- both baseline and candidate are `None` for this
  metric (e.g. `peak_memory_allocated_bytes` on CPU), or no policy entry
  exists for it. Never converted to 0.
- Metric-level **ERROR** (surfaces as overall `RegressionResult.status =
  "ERROR"`) -- present on one side but missing on the other (measurement
  coverage changed between baseline and candidate).

## Overall status and exit codes

`RegressionResult.status`: **PASS** only if every gated metric passes;
**FAIL** if at least one gated metric fails; **ERROR** if the two runs
cannot be validly compared (workload/backend/device/dtype mismatch, or
inconsistent metric coverage). A performance regression (FAIL) is never
conflated with an infrastructure problem (ERROR).

`python -m tensorforge_ops regression` exit codes:

| Code | Meaning |
|---|---|
| 0 | guard passed |
| 1 | operational/configuration/comparison error |
| 2 | valid comparison, but the regression policy failed |

This lets CI distinguish "the code got slower" from "the benchmark
infrastructure is broken."

## `RegressionSuiteResult` (multiple workloads)

`compare_regression_suite(pairs, policy)` compares several
baseline/candidate pairs under one policy. Overall status precedence is
**ERROR > FAIL > PASS** -- any infrastructure error anywhere in the
suite takes priority over reporting a plain pass/fail, since an error
means at least one comparison's result cannot be trusted.

## CLI

```bash
python -m tensorforge_ops regression \
    --baseline base-benchmark.json \
    --candidate candidate-benchmark.json \
    --policy perf/cpu-ci-policy.json \
    --output-json regression-result.json \
    --output-markdown regression-report.md
```

`--output-json`/`--output-markdown` refuse to silently overwrite an
existing file; pass `--force` to intentionally replace one.

`python -m tensorforge_ops benchmark` gained `--output-json`/`--force`
in this milestone so a `BenchmarkResult` can be exported to a file
without a running MLflow server -- useful for local before/after
comparisons. The GitHub workflow itself does **not** rely on this flag
(see below), specifically so it keeps working even when the *base*
checkout predates this flag's existence.

## Telemetry context

`python -m tensorforge_ops regression` optionally accepts
`--baseline-telemetry-summary`/`--candidate-telemetry-summary` (both
`TelemetrySummary` JSON files -- see
[gpu-telemetry.md](gpu-telemetry.md)) and `--telemetry-output-json`. When
both summaries are provided, a `## GPU Telemetry Context` section (GPU
activity/memory activity/VRAM/power/temperature/clock deltas, plus any
evidence-backed signals) is appended to the Markdown report, and the CLI
prints the observed signals. This runs strictly **after**
`compare_benchmark_results()` has already produced the final PASS/FAIL/
ERROR status -- telemetry is never read by, and never changes,
that decision. Every rendered telemetry section ends with an explicit
statement: *"Telemetry is diagnostic context and does not affect the
regression gate result."*

The standard GitHub-hosted CPU PR workflow (below) does not collect or
attach telemetry -- there is no GPU on those runners. Telemetry
correlation is for local/self-hosted-GPU investigation of a regression,
not part of the required PR check.

## GitHub Actions workflow

`.github/workflows/performance-regression.yml` triggers on `pull_request`
(never `pull_request_target` -- that would run PR code with elevated,
non-fork-safe permissions) for changes under `src/**`, `pyproject.toml`,
or `perf/**`.

**Same-runner, same-job requirement:** the workflow checks out the PR's
*base* SHA (`github.event.pull_request.base.sha`, not just `origin/main`
-- correct even when the PR targets a non-`main` branch) into `base/` and
the PR's *head* SHA into `candidate/`, both inside one job. Each gets its
own isolated virtualenv (`base/.venv`, `candidate/.venv`) installed with
`pip install -e ".[ops,benchmark]"` from its *own* checkout, so a PR that
intentionally changes a dependency is represented honestly on both sides
-- and both benchmarks execute back-to-back in the same job, on the same
runner instance, which is what makes the comparison meaningful (it
controls for CPU type, OS image, and runner class; it does not control
for e.g. transient host load, which is why policy thresholds below are
deliberately conservative).

**Base-commit compatibility:** the workflow does not use the `benchmark`
CLI's `--output-json` flag to produce the baseline JSON -- the *base*
commit may predate this milestone entirely, including that flag. Instead
it writes a small helper script at run time (via heredoc, not committed
to either checkout) that calls only the stable
`tensorforge`/`tensorforge_ops` package APIs
(`run_experiment`/`run_pytorch_benchmark`/`BenchmarkResult.to_json()`),
present since Milestone 11/12, and runs it under each side's own venv.
The regression *comparison* itself always uses the **candidate**
checkout's `tensorforge_ops` (the tool this milestone introduces) since a
base commit may not have `regression.py` at all.

**No MLflow server required:** everything is local JSON files; the
workflow never talks to a tracking server. (`mlflow` the *package* is
still installed, because computing a Core fingerprint reuses
`tracking.py`'s `compute_result_fingerprint()` rather than duplicating
that formula -- but no server connection is attempted.)

**Reporting, always:** the comparison step never aborts the job on
failure (`set +e` around the CLI call, capturing its exit code
explicitly) so the Markdown report is always generated and appended to
`$GITHUB_STEP_SUMMARY`, and `actions/upload-artifact` runs with
`if: always()` to preserve `base/`+`candidate/` JSON, per-workload
`*-regression-result.json`, and `*-regression-report.md`. The job's exit
status is set explicitly in a final "Gate on regression result" step, so
a pretty report is never mistaken for a passing check.

**PR comment (optional, same-repo only):** if the head repository is the
same as the base repository (`github.event.pull_request.head.repo.full_name
== github.repository`), the workflow posts or updates a single PR
comment marked with `<!-- tensorforge-performance-guard -->`, using only
the built-in `GITHUB_TOKEN` with `pull-requests: write` (no other
secret). Fork PRs are skipped gracefully -- the check and step summary
still work regardless; a fork PR never gets write-scoped credentials.

**Permissions:** `contents: read`, `pull-requests: write`. No project
secrets (no MLflow credentials, no cloud tokens, no PAT). The workflow
never commits, pushes, or otherwise writes to the base branch.

## GitHub-hosted CPU vs. self-hosted CUDA

GitHub-hosted runners do not provide the RTX 3050 Laptop GPU this
project was calibrated/validated on (Milestones 12-13). The default,
always-on `performance-regression.yml` therefore benchmarks on
**GitHub-hosted CPU** for both base and candidate. This provides real
regression detection for shared PyTorch/operator code paths, but it does
**not** validate CUDA-specific performance -- a change that only affects
CUDA kernels would not be caught by this workflow.

A real CUDA regression guard needs a GPU-enabled self-hosted runner (or
a future GPU-enabled hosted CI service); this repository does not
attempt to provision or auto-register one. `perf/gpu-regression-policy.json`
exists as a starting point for anyone who configures such a runner
(labeled e.g. `[self-hosted, linux, tensorforge-gpu]`) and wires a
`workflow_dispatch`-triggered workflow to it -- this milestone verifies
the comparator itself works correctly against real CUDA `BenchmarkResult`
data (see "Live verification" below) without shipping an
always-attempting-to-run GPU workflow that would create a permanently
pending check on a repository with no such runner configured.

## CPU CI policy: `perf/cpu-ci-policy.json`

These thresholds are **this repository's CI policy**, not a universal
TensorForge recommendation, and were **not** chosen arbitrarily: 5
repeated, unchanged-code runs of `gemm_tiny` and `conv_pointwise` were
measured locally before picking them (see "Observed repeatability"
below), and the thresholds were set with margin above the largest
observed run-to-run spread, not tuned to the minimum that happened to
pass.

| Metric | Relative | Absolute |
|---|---|---|
| p50 latency | 35% | 3 ms |
| p95 latency | 45% | -- |
| throughput | 35% drop | -- |

`peak_memory_allocated_bytes` is intentionally absent from the CPU
policy -- it is always `None` on CPU (both sides `NOT_COMPARABLE`),
so gating it there would be meaningless.

## Observed repeatability (evidence for the thresholds above)

**CPU** (this development machine, 5 back-to-back runs, unchanged code,
`warmup=5`/`iterations=30`):

| Workload | p50 range | max spread |
|---|---|---|
| `gemm_tiny` | 3.61 ms - 4.51 ms | ~24.8% |
| `conv_pointwise` | 44.48 ms - 54.89 ms | ~23.4% |

This is a noisier result than one might expect from a dedicated
development machine, and GitHub-hosted shared runners are generally
*at least* as noisy -- hence the 35%/45% thresholds above, well above
the largest spread actually observed, rather than a value picked to
"look reasonable."

**CUDA** (RTX 3050 Laptop GPU, 5 back-to-back runs of
`gemm_large_square`, unchanged code, `warmup=10`/`iterations=50`):

| | range |
|---|---|
| p50 | 245.1 us - 300.9 us (~22.7% spread) |
| p95 | 253.2 us - **1186.0 us** (one run showed a large tail spike) |
| peak memory | identical (14,811,136 bytes) across all 5 runs |

The single large p95 outlier is why `perf/gpu-regression-policy.json`
gives p95 a much wider allowance (50%) than p50 (20%) -- tail latency on
a shared/laptop GPU is visibly less stable than steady-state p50, and
peak memory (fully deterministic here) can reasonably use a tight
threshold.

This is **observed repeatability in this environment**, from 5 runs --
not a statistically complete noise model, and not a claim that these
numbers generalize to other machines, other RTX 3050 units, or different
driver/power states.

## Limitations

- Relative performance guard only -- it never claims an implementation
  is fast in absolute terms.
- Same-runner comparison controls for hardware/OS class, not for
  transient host load; thresholds are set conservatively to absorb this.
- GitHub-hosted CPU variability is real (see above); do not lower these
  thresholds without re-running the repeatability study.
- No CUDA gate runs automatically -- GPU regression checking requires a
  self-hosted GPU runner that this repository does not provision.
- No statistical hypothesis testing (no confidence intervals, no
  significance test) -- comparisons are single-run point comparisons
  against an explicit tolerance, nothing more.
- No measured-bottleneck diagnosis, no GPU telemetry (DCGM/Nsight),
  no automated root-cause attribution -- a regression report may surface
  analytical context (Core fingerprint/FLOPs changes) as a *possible*
  explanation, never a proven cause.
- No cost/SLO policy, no hardware right-sizing (future milestones).
- Synthetic TensorForge workload presets only -- not real production
  model workloads.
