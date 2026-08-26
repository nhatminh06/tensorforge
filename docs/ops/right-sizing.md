# Hardware right-sizing + inference SLO/cost planner reference

## The question this module answers

> Given the performance we actually measured, what is the cheapest
> deployment configuration among the supplied candidates that satisfies
> the explicit SLO assumptions?

`tensorforge_ops.sizing` is pure arithmetic over already-measured
`BenchmarkResult` evidence (Milestone 12), with optional
`TelemetrySummary` (Milestone 15) and `ValidationResult` (Milestone 13)
attached purely as diagnostic context. It never re-runs a benchmark,
never fetches a cloud price, and never claims a global optimum -- only
"cheapest feasible candidate among the ones you supplied."

## Two capacity-language rules that matter more than any formula

**"Measured steady-state benchmark throughput is treated as a
per-replica capacity proxy."** `BenchmarkResult.statistics.throughput_per_second`
comes from Milestone 12 repeating one prepared workload steady-state,
not from a load-tested HTTP endpoint. It is **not** guaranteed
production requests/sec, **not** load-tested throughput, and **not** a
queueing-theory result. No M/M/1, M/M/c, Little's Law, request
concurrency, dynamic batching, or autoscaler dynamics are modeled in
this milestone.

**"Benchmark p95 is the measured workload's own execution latency, not
end-to-end service latency."** It excludes network time, request
queueing, preprocessing, and response serialization.

Every generated report repeats both statements verbatim in its
"Assumptions" section -- this is not decoration, it is the load-bearing
caveat that keeps this module honest.

## Deployment-unit semantics

One `DeploymentCandidate` = one serving unit corresponding to whatever
environment its `BenchmarkResult` was actually measured on (typically
one GPU, one VM, or one accelerator-backed worker). The planner assumes
**one replica = one such serving unit**, and that identical replicas
scale capacity **perfectly linearly**:

```
total_capacity = replicas * per_replica_capacity
```

This ignores shared databases, network bottlenecks, load-balancer
overhead, storage bottlenecks, inter-replica contention, and cluster
scheduling effects -- stated, not hidden. No MIG, no multi-GPU tensor/
model parallelism, no replicas sharing one GPU are modeled.

## Deployment candidate / catalog

```json
{
  "deployment_catalog_schema_version": 1,
  "currency": "USD",
  "candidates": [
    {
      "id": "gpu-a",
      "benchmark_result": "results/gpu-a-benchmark.json",
      "hourly_cost": 0.45,
      "provider": "example-provider",
      "region": "example-region",
      "hardware_label": "Example GPU",
      "price_source": "manual",
      "price_as_of": "2026-08-26",
      "telemetry_summary": "results/gpu-a-telemetry-summary.json",
      "validation_result": "results/gpu-a-validation-result.json"
    }
  ]
}
```

`benchmark_result`/`telemetry_summary`/`validation_result` paths resolve
**relative to the catalog file's own directory** (unless already
absolute) -- so a catalog can be moved together with its `results/`
directory. `telemetry_summary`/`validation_result` are optional; only
`benchmark_result` is mandatory. If a candidate has no `BenchmarkResult`,
it cannot be represented at all -- this module never fabricates
benchmark values from a Core/calibration prediction when a measurement
is absent.

`hourly_cost` accepts `0` (an intentional marginal-cost-zero value for
owned/local hardware) but rejects negative, `NaN`, or `Infinity` values.
All candidates in one catalog must share one `currency` -- mixed
currencies are rejected outright (`ValueError`); **no FX conversion is
performed, ever**. Pricing provenance fields (`provider`/`region`/
`price_source`/`price_as_of`) are metadata only, carried through to the
report unmodified -- nothing here fetches or verifies a price over the
network.

## SLO policy

```json
{
  "slo_policy_schema_version": 1,
  "required_invocations_per_second": 120.0,
  "max_p95_latency_seconds": 0.025,
  "capacity_headroom_fraction": 0.20,
  "monthly_hours": 730,
  "max_monthly_cost": 900.0,
  "max_replicas": 10,
  "max_peak_memory_allocated_bytes": 16000000000
}
```

- `required_invocations_per_second` (required, > 0): named this way, not
  "requests/sec", because it is only guaranteed to correspond one-to-one
  with real requests when the benchmarked workload genuinely does.
- `max_p95_latency_seconds` (required, > 0).
- `capacity_headroom_fraction` (`0 <= x < 1`, **default `0.0`**): the
  planner never invents an operational safety margin. `0.20` reserves
  20% of measured capacity; an explicit choice belongs to the caller.
- `monthly_hours`: required before `max_monthly_cost` can be used (a
  monthly budget without billing hours is undefined). Documentation
  examples use 730 hours/month, but that is never assumed silently.
- `max_monthly_cost`, `max_replicas`, `max_peak_memory_allocated_bytes`:
  all optional constraints.

## Feasibility evaluation

For each candidate:

1. **Latency**: if `p95_seconds` is `None` (insufficient benchmark
   samples), the candidate gets `missing_p95` and is
   `INSUFFICIENT_EVIDENCE` -- **never silently falls back to p50 or
   mean**. If `p95 > max_p95_latency_seconds`, `latency_slo_exceeded`
   (`SLO_VIOLATION`). Equality (`p95 == limit`) passes.
2. **Usable capacity** (always computed, independent of the latency
   outcome, since it is a pure per-replica ratio):
   `usable_capacity_per_replica = throughput_per_second * (1 - headroom)`.
3. **Only if latency passed**, replicas/cost are computed:
   `required_replicas = max(1, ceil(demand / usable_capacity_per_replica))`,
   `total_hourly_cost = replicas * candidate.hourly_cost`,
   `monthly_cost = total_hourly_cost * monthly_hours` (if configured),
   `cost_per_million_required_invocations = total_hourly_cost / (demand * 3600) * 1_000_000`
   (an optional, sustained-demand-conditional metric -- never a
   universal per-inference cost claim). `max_replicas`/`max_monthly_cost`
   are then checked against these already-computed numbers -- a budget
   or replica-count violation does **not** blank out the computed cost
   (it is still shown, just flagged), unlike a latency failure or
   missing evidence, which does leave replicas/cost as `--`/`None`
   entirely (the deployment is not viable at all in that case,
   independent of any budget).
4. **Memory** (only if `max_peak_memory_allocated_bytes` is configured):
   `missing_peak_memory` (`INSUFFICIENT_EVIDENCE`) if the benchmark
   never recorded `peak_memory_allocated_bytes` (true for every current
   CPU benchmark), else `peak_memory_exceeded` (`SLO_VIOLATION`) if over
   the limit.
5. **Status**: `INSUFFICIENT_EVIDENCE` if any reason starts with
   `missing_`, else `SLO_VIOLATION` if any reason is present, else
   `FEASIBLE`. `reasons` always lists every applicable reason, never
   collapsed into a single boolean.

## Money arithmetic

All cost math uses Python `Decimal` (parsed via `Decimal(str(value))`,
which avoids the classic `Decimal(0.1) != Decimal("0.1")` binary-float
artifact) -- this keeps replica-count-times-unit-price and monthly
multiplication exact, and avoids surprising floating-point rounding in
budget comparisons. JSON output converts `Decimal -> float` for
readability; the computation itself never rounds arbitrarily.

## Ranking

Only `FEASIBLE` candidates are ranked. Primary key: lowest
`total_hourly_cost` (equivalent to lowest `monthly_cost` when
`monthly_hours` is shared across the whole catalog, since it is a
constant linear multiplier). Deterministic tie-breakers, in order:

1. lower total cost
2. fewer replicas
3. lower measured p95
4. greater spare usable capacity
5. lower peak memory (only when both sides report it; otherwise this
   step is skipped)
6. `candidate_id` (final deterministic fallback)

The recommended candidate is always phrased as **"cheapest feasible
candidate among the supplied candidates"** -- never "optimal" or
"globally best." If no candidate is `FEASIBLE`, the result explicitly
says `NO FEASIBLE CANDIDATE` with each candidate's specific reasons;
the planner never silently degrades to "least bad."

## Workload identity across candidates

Every candidate in one plan must share: `workload_preset`,
`workload_kind`, `backend`, `dtype`. A mismatch on any of these makes
the **whole plan invalid** (`SizingPlanResult.status = "ERROR"`, no
candidates ranked) -- comparing `gemm_large_square` against
`conv_spatial`, or FP16 against FP32, is not a hardware decision, it is
a different question. **Device identity is deliberately allowed to
differ** -- an RTX 3050 candidate and an entirely different GPU
candidate in one catalog is exactly the point of right-sizing across
hardware. Runtime/PyTorch versions may also differ across candidates
(a deployment candidate is hardware *and* software environment
together) and are surfaced in the report, never silently rejected --
though comparing candidates benchmarked with materially different
software stacks is a weaker hardware-only signal, and the documentation
recommends comparable stacks where practical.

## Telemetry / calibration context (never gating)

If a candidate carries a `TelemetrySummary`, its mean/max GPU activity,
memory activity, peak VRAM, power, and any throttle reasons observed
appear in the report as diagnostic context -- e.g. *"Candidate satisfied
the measured SLO while showing 73% mean GPU activity during the
separate telemetry window."* A `ValidationResult`'s prediction error
(APE, measured/predicted ratio, predicted bottleneck) appears the same
way. **Neither ever changes a candidate's FEASIBLE/SLO_VIOLATION/
INSUFFICIENT_EVIDENCE status or its rank** -- a candidate is never
rejected because TensorForge predicts it is memory-bound, and never
preferred because its GPU utilization looked high. Cost and measured
SLO evidence alone drive the recommendation. No uncertainty/confidence
score is invented anywhere in this module.

## Demand scenarios

```python
scenarios = (DemandScenario("1x", 100.0), DemandScenario("2x", 200.0), DemandScenario("5x", 500.0))
results = build_sizing_plans_for_scenarios(catalog, base_slo, scenarios)
```

Only `required_invocations_per_second` changes per scenario; latency
limit, headroom, and pricing stay fixed (unless the caller explicitly
builds a different base `SloPolicy`). These are **static snapshots** --
no time-varying traffic, scale-up/down delay, or autoscaler behavior is
simulated. The live study below shows a real recommendation flip driven
purely by demand.

## CLI

```bash
python -m tensorforge_ops right-size \
    --catalog deployment-catalog.json --slo inference-slo.json \
    --output-json sizing-result.json --output-markdown sizing-report.md
```

Exit codes:

| Code | Meaning |
|---|---|
| 0 | plan calculated, at least one feasible candidate exists |
| 1 | configuration/operational error (bad catalog/SLO, mismatched workload/backend/dtype) |
| 2 | valid plan, but no supplied candidate satisfies the SLO |

`--track` (off by default -- the planner works fully offline) logs the
plan to MLflow as its own new run (a sizing plan spans multiple
candidates/benchmark artifacts, so it does not belong to any single
existing run); it never modifies any other run's
`core-result.json`/`benchmark-result.json`/`validation-result.json`/
telemetry artifacts. This is **not** part of the Milestone-14 GitHub PR
performance-regression guard -- deployment planning and PR performance
gating are different concerns and stay in different workflows.

## Live demonstration: real RTX 3050 + synthetic peers

**Catalog**: one real, live-measured `gemm_large_square` result from the
RTX 3050 Laptop GPU (`price_source: "user-supplied-example-cost"`,
**not** a market price) alongside four fully synthetic peer candidates
at comparable latency/throughput scale
(`price_source: "synthetic-demonstration-data"` on each). SLO: demand
2000 invocations/sec, p95 <= 500 us, 10% headroom, 730 monthly hours.

| Candidate | p95 | Measured cap. | Status | Hourly |
|---|---|---|---|---|
| **rtx3050-local** (real) | 282.60 us | 3930.9/s | FEASIBLE | $0.35 |
| budget (synthetic) | 900.00 us | 1200.0/s | SLO_VIOLATION (latency) | -- |
| balanced (synthetic) | 400.00 us | 2800.0/s | FEASIBLE | $0.40 |
| fast (synthetic) | 150.00 us | 7000.0/s | FEASIBLE | $1.20 |
| premium (synthetic) | 80.00 us | 15000.0/s | FEASIBLE | $3.50 |

**Recommended: `rtx3050-local`** -- the real, live-measured candidate
wins on cost even though `fast` and `premium` are objectively faster
(the "fastest hardware may lose" lesson), and `budget` fails the
latency SLO outright despite being the cheapest per-unit price (the
"cheapest unit may lose/fail" lesson).

### Headroom sensitivity (same catalog, demand raised to 3600/s)

| Headroom | rtx3050 replicas | Hourly |
|---|---|---|
| 0% | 1 | $0.35 |
| 10% | 2 | $0.70 |
| 20% | 2 | $0.70 |
| 30% | 2 | $0.70 |

Headroom is a **user policy lever**, not a measured property -- moving
from 0% to 10% headroom alone pushed this candidate from 1 to 2
replicas at this demand level.

### Budget sensitivity (demand 2000/s, monthly_hours=730)

- `max_monthly_cost = $300`: `rtx3050-local` ($255.50/mo) still wins;
  `fast`/`premium` now additionally show `monthly_budget_exceeded`.
- `max_monthly_cost = $200`: **every** candidate now violates the
  budget (even the previously-recommended one) -> `NO FEASIBLE
  CANDIDATE`, CLI exit code 2. The planner never silently violates a
  stated budget to still produce a recommendation.

### SLO (latency limit) sensitivity, same catalog

| max p95 | Recommended |
|---|---|
| 200 us (strict) | `fast` ($1.20/hr) -- `rtx3050-local`/`balanced`/`budget` all fail latency |
| 500 us (moderate) | `rtx3050-local` ($0.35/hr) |
| 1 ms (relaxed) | `budget` ($0.30/hr, 2 replicas) -- now cheapest even at 2 replicas |

The recommended candidate changes at every latency threshold -- there is
no single "best" hardware independent of the SLO.

### Demand-driven recommendation flip (dedicated 2-candidate synthetic catalog)

`cheap-low-throughput` ($0.50/hr, 60/s) vs. `pricier-high-throughput`
($1.50/hr, 200/s), 10% headroom, p95 limit 2ms (both pass latency):

| Demand | cheap-low-throughput | pricier-high-throughput | Recommended |
|---|---|---|---|
| 50/s (1x) | 1 replica, $0.50 | 1 replica, $1.50 | `cheap-low-throughput` |
| 150/s (3x) | 3 replicas, $1.50 | 1 replica, $1.50 | `pricier-high-throughput` (cost tie -> fewer replicas wins) |
| 500/s (10x) | 10 replicas, $5.00 | 3 replicas, $4.50 | `pricier-high-throughput` |

This is a genuine, deterministic crossover: the cheap-per-unit candidate
wins at low demand and loses once replica-count overhead outweighs its
lower unit price -- exactly the reason right-sizing is demand-dependent,
not a fixed hardware choice.

## Multi-hardware evidence status

**ENGINE PROVEN WITH SYNTHETIC MULTI-CANDIDATE DATA; LIVE RTX 3050
CANDIDATE PROVEN; MULTI-HARDWARE LIVE COMPARISON REQUIRES ADDITIONAL
BENCHMARK ARTIFACTS.** This development environment has exactly one
physical GPU (RTX 3050 Laptop GPU). All ranking/trap/sensitivity studies
above that compare *multiple hardware tiers* use clearly labeled
synthetic `BenchmarkResult` fixtures at a plausible scale, not real
measurements from other physical devices. The one real candidate in
every demo (`rtx3050-local`) is genuine, live-measured evidence with an
explicitly labeled example price, never a claimed market price.

## Limitations

- Steady-state benchmark throughput/p95 are capacity/latency proxies,
  not measured production traffic (see the two capacity-language rules
  above).
- Replica scaling is assumed perfectly linear; no shared-resource
  contention, network, or scheduling effects are modeled.
- No queueing-theory model (M/M/1, M/M/c, Little's Law), no dynamic
  batching, no autoscaler simulation.
- Pricing is entirely user-supplied and unverified; no currency
  conversion.
- Demand scenarios are static snapshots, not a time-varying traffic
  simulation.
- Telemetry/validation context never changes feasibility or ranking.
- No uncertainty/confidence score is computed anywhere.
- Multi-hardware live comparison has not been proven beyond one real
  device in this environment (see above).
