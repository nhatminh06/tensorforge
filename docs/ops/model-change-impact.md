# Model-change impact report reference

## What this answers

> Here is what changed, what we measured, what it costs, what evidence
> supports the conclusion, and what we still do NOT know.

`tensorforge_ops.impact` composes already-computed evidence from every
other Ops domain -- Core, real benchmark, regression, telemetry,
calibration, right-sizing -- into one deterministic report. It never
re-runs a benchmark, telemetry window, calibration probe, or sizing
plan; it is a fast, offline composition layer over artifacts that
already exist.

## Scope: performance and infrastructure only

**The recommendation covers measured performance and deployment/
infrastructure readiness only.** It never evaluates model accuracy, F1,
mAP, perplexity, BLEU, correctness, business value, safety, or user
experience -- TensorForge Ops has no evidence about any of those. Every
report -- JSON and Markdown -- carries this sentence verbatim:

> TensorForge Ops does not assess model quality or correctness. This
> recommendation covers measured performance and deployment/
> infrastructure evidence only.

Readiness states are deliberately not phrased as "safe to promote":

| State | Meaning |
|---|---|
| `PERFORMANCE_READY` | all policy-required measured performance/deployment constraints pass |
| `PERFORMANCE_BLOCKED` | valid evidence proves one or more explicit policy requirements fail |
| `REVIEW_REQUIRED` | required evidence is missing, invalid, or internally inconsistent |

## No composite score

There is no `impact score = 87/100`, no `deployment confidence = 92%`,
no weighted composite of any kind. There is no defensible universal
weighting between "12% faster" and "20% more expensive" -- that tradeoff
belongs to the team, not to this tool. Evidence stays separate; the only
thing that turns evidence into a decision is the explicit `ImpactPolicy`
supplied on the command line.

## Decision hierarchy: measured evidence is primary

```
RegressionResult (measured base-vs-candidate performance, Milestone 14)
SizingPlanResult (measured deployment feasibility/cost, Milestone 16)
    --> together with an explicit ImpactPolicy, decide readiness

Core result / ValidationResult / TelemetrySummary
    --> context only; NEVER override a measured decision
```

A cheaper deployment never overrides a failed regression. A better (or
worse) analytical prediction never overrides a failed regression.
Telemetry never gates anything in this milestone. `RegressionResult` and
`SizingPlanResult` are re-used directly (their own metric comparisons,
their own feasibility/cost fields) -- this module never recomputes the
Milestone-14 or Milestone-16 verdicts, it only reads them.

## ImpactManifest

```json
{
  "impact_manifest_schema_version": 1,
  "regression_result": "regression-result.json",
  "baseline": {
    "core_result": "baseline/core-result.json",
    "benchmark_result": "baseline/benchmark-result.json",
    "validation_result": "baseline/validation-result.json",
    "telemetry_summary": "baseline/telemetry-summary.json",
    "sizing_result": "baseline/sizing-result.json"
  },
  "candidate": {
    "core_result": "candidate/core-result.json",
    "benchmark_result": "candidate/benchmark-result.json",
    "validation_result": "candidate/validation-result.json",
    "telemetry_summary": "candidate/telemetry-summary.json",
    "sizing_result": "candidate/sizing-result.json"
  }
}
```

All paths resolve **relative to the manifest file's own directory**
(unless already absolute) -- no URL, no S3, no cloud storage is ever
fetched.

`regression_result` is the only strictly required top-level field --
without it there is no measured performance evidence at all, and
loading fails clearly (`ValueError`) before any composition is
attempted. Every `baseline`/`candidate` field is optional; missing
optional evidence is reported as `"<side> <field>: NOT PROVIDED"` in the
result, **never** silently treated as zero, pass, or unchanged.

`baseline`/`candidate` `benchmark_result` and `core_result` entries, when
supplied, are used only for a light identity cross-check against the
regression result's own workload/kind (`artifact_identity_mismatch` ->
`REVIEW_REQUIRED` on mismatch) -- the actual measured metrics displayed
always come from `RegressionResult.metric_comparisons`, never re-derived
from a separately loaded `BenchmarkResult`.

Telemetry correlation is **always freshly computed** from
`baseline.telemetry_summary`/`candidate.telemetry_summary` via the
existing `correlate_telemetry()` (Milestone 15) when both are present --
there is no separate "precomputed correlation" input, so there is only
one telemetry-comparison code path in the whole project.

## ImpactPolicy

```json
{
  "impact_policy_schema_version": 1,
  "require_regression_pass": true,
  "require_feasible_candidate_deployment": true,
  "max_hourly_cost_increase_fraction": 0.20,
  "max_monthly_cost_increase_fraction": null,
  "max_replica_increase": null
}
```

**Design choice, stated explicitly:** the two boolean requirements
default to `true` -- these are explicit boolean semantics (a team either
requires a measured-passing regression and a feasible deployment, or it
doesn't), not hidden numeric thresholds. The three cost/replica ceilings
default to `None` (unenforced) because there is no defensible universal
number for "how much cost increase is acceptable" -- that is a genuine
business tradeoff each team must state. `--policy` is **always required**
on the CLI; there is no bundled default policy file and no silent
fallback, so the decision criteria for any given report are always
visible in the invocation that produced it, not buried in this module's
source.

## Readiness decision, in order

1. **Artifact identity mismatch** (any supplied `core_result`/
   `benchmark_result` disagreeing with the regression result's own
   workload/kind) -> `REVIEW_REQUIRED` immediately.
2. **Regression status**: `ERROR` -> `REVIEW_REQUIRED`
   (`regression_error`); `FAIL` with `require_regression_pass=true` ->
   `PERFORMANCE_BLOCKED` (`measured_regression_failed`). This module
   never recomputes the Milestone-14 gate.
3. **Deployment feasibility** (only if `require_feasible_candidate_deployment=true`):
   candidate sizing missing -> `REVIEW_REQUIRED`
   (`missing_candidate_sizing`); candidate sizing itself invalid
   (`SizingPlanResult.status == ERROR`, e.g. a malformed catalog) ->
   `REVIEW_REQUIRED` (`candidate_sizing_invalid`); valid sizing evidence
   but no feasible candidate -> `PERFORMANCE_BLOCKED`
   (`candidate_deployment_infeasible`).
4. **Cost/replica policy** (only for whichever limits are explicitly
   configured): comparison unavailable (e.g. one side's sizing missing
   monthly cost, or a `0`-cost baseline making a relative comparison
   undefined) -> `REVIEW_REQUIRED` (`missing_cost_evidence`/
   `missing_replica_evidence`); comparison available and over the
   configured limit -> `PERFORMANCE_BLOCKED`
   (`hourly_cost_increase_exceeded`/`monthly_cost_increase_exceeded`/
   `replica_increase_exceeded`).
5. If any step produced a `REVIEW_REQUIRED` reason, the overall
   readiness is `REVIEW_REQUIRED` (evidence problems take priority --
   an infrastructure error is never silently reported as a performance
   block). Otherwise `PERFORMANCE_BLOCKED` if any block reason fired,
   else `PERFORMANCE_READY`.

## Cost/replica delta semantics

- **Absolute delta**: `candidate - baseline`, always computed when both
  sides have the value.
- **Relative delta**: `absolute_delta / baseline`, computed **only when
  baseline is nonzero** -- a `0`-cost baseline (e.g. owned/local
  hardware with an intentional `$0`/hour entry) yields `None`, never an
  invented "infinite percent increase."
- Both hourly and monthly cost deltas are computed independently; if
  only one side of a pair has a monthly figure (e.g. one `SloPolicy` set
  `monthly_hours`, the other didn't), the monthly comparison is `None`
  while the hourly comparison may still be available.
- Replica delta is a plain integer difference (`candidate_replicas -
  baseline_replicas`); negative means fewer replicas.

## Analytical / validation / telemetry context (never gating)

**Analytical**: a small whitelist of numeric metrics already present in
Core's own `ExperimentResult.primary_metrics` (`total_flops`,
`total_macs`, `total_dram_bytes`, `serialized_time_seconds`,
`perfect_overlap_time_seconds`, `effective_arithmetic_intensity`) --
never re-derived, never a mechanical diff of every JSON field. The Core
fingerprint shown always comes directly from `RegressionResult`'s own
`baseline_core_fingerprint`/`candidate_core_fingerprint` fields, never
recomputed by re-hashing a saved `core-result.json` file (which is
fragile: `ExperimentResult.save()` appends a trailing newline that the
canonical fingerprint hash does not include, so re-hashing the file
would silently produce a different, wrong fingerprint).

**Validation**: if both sides have a `ValidationResult`, APE and
measured/predicted ratio are compared (e.g. *"Candidate calibration
error increased from 25% to 40%"*) -- this describes analytical-model-
vs-hardware agreement, never deployment reliability, and never changes
readiness.

**Telemetry**: reuses Milestone 15's `correlate_telemetry()` signals
verbatim (cautious, evidence-worded language -- "evidence suggests",
never "root cause" or "proves"). A candidate can show much lower GPU
activity, higher VRAM usage, or an observed power/thermal throttle event
and still be `PERFORMANCE_READY` if the measured SLO and cost policy
both pass -- telemetry is diagnostic context only.

## ImpactResult

Schema-versioned (`impact_result_schema_version = 1`, distinct from
every other schema in this project). Deterministic for fixed input
artifacts and policy -- no timestamps, random IDs, or current-environment
state inside the result itself (any such runtime metadata stays
separate, e.g. in MLflow's own run metadata). Fields include: readiness,
decision reasons, workload identity, Core fingerprints (+ whether they
changed), the full set of `RegressionResult.metric_comparisons`,
analytical deltas, validation context, telemetry correlation, deployment
impact, missing-evidence list, limitations, and the model-quality
disclaimer.

## CLI

```bash
python -m tensorforge_ops impact \
    --manifest impact-manifest.json --policy impact-policy.json \
    --output-json impact-result.json --output-markdown impact-report.md
```

Exit codes (deliberately parallel to the Milestone-14 `regression`
command's 0/1/2 convention, but a distinct contract for a distinct
concern):

| Code | Meaning |
|---|---|
| 0 | `PERFORMANCE_READY` |
| 1 | `REVIEW_REQUIRED`, or a configuration/operational error |
| 2 | `PERFORMANCE_BLOCKED` |

Output-file overwrite errors are handled without argparse's
`parser.error()` (which always exits 2) specifically so they cannot
collide with the "2 == BLOCKED" convention -- the same fix already
applied to `regression` and `right-size` in earlier milestones.
`--track` (off by default) logs the report to MLflow as its own new run
(a composition artifact spanning multiple domains, so it does not belong
to any single existing run); it never duplicates or mutates
`core-result.json`/`benchmark-result.json`/`regression-result.json`/
`sizing-result.json`/`telemetry-summary.json`/`validation-result.json`
on any other run.

## GitHub integration

Impact reporting is **not** part of the Milestone-14 PR performance-
regression guard workflow (`.github/workflows/performance-regression.yml`),
which remains a valid, independent regression gate whether or not
sizing/telemetry/impact evidence exists for a given PR. Deployment
planning and PR performance gating are different concerns with
different required inputs (pricing, an SLO) that should never be forced
onto every PR check.

## Limitations

- Performance-and-infrastructure readiness only -- no model quality/
  correctness/business-value assessment of any kind.
- No composite/weighted score.
- Composes existing artifacts only; never re-runs anything.
- Cost/replica policy thresholds, when configured, are explicit user
  policy with no default numeric value anywhere in this module.
- Analytical/validation/telemetry evidence is context, never a gate.
- Not integrated into the standard GitHub PR performance-regression
  workflow.
