# Limitations

Organized by layer. This is a complete, honest accounting -- not a
marketing summary.

### Compute

- No PE pipeline fill/drain, no operand propagation latency.
- No SIMD/vector width -- 1 PE = 1 MAC/cycle, uniformly.
- No variable dtype throughput (FP16 is not modeled as faster compute
  than FP32; only tensor *bytes* change with dtype).
- No cycle accuracy of any kind.

### Memory

- One global SRAM level -- no register files, no PE-local buffers, no
  multi-level cache hierarchy.
- No SRAM bandwidth or latency model (SRAM capacity is checked; SRAM
  *access time* is not modeled).
- No DRAM latency, no bank/rank/channel model, no memory controller
  model, no queueing.
- No network-on-chip (NoC) model.

### Mapping

- Exactly three explicit residency schedules (`c-resident`, `a-resident`,
  `b-resident`) -- not a general dataflow description language, and not
  a claim that these correspond precisely to "output/input/weight
  stationary" hardware dataflow definitions.
- Tile and schedule candidates are user-supplied and bounded; there is no
  automatic tile/schedule generation and no proof that the true optimum
  is within the searched set.
- PE-array candidates are likewise user-supplied and bounded.

### Transformer

- GEMM-only: softmax, normalization (LayerNorm/RMSNorm), activation
  functions, and residual additions are excluded from every aggregate --
  not modeled as free, simply absent.
- Attention heads (`batch*heads` repetitions) are modeled executing
  sequentially -- no head parallelism.
- No QKV fusion, no attention fusion (no FlashAttention-style semantics).
- No KV cache, no autoregressive decoding loop.
- No cross-operation SRAM residency (e.g. Q projection's output being
  kept resident for the attention-score GEMM is not modeled).

### Convolution

- Materialized-im2col lowering only -- no direct/implicit convolution
  dataflow, no Winograd, no FFT convolution. This is one deliberate
  choice among several real options, and can substantially overstate
  DRAM traffic for kernels larger than 1x1 (see `docs/convolution.md`).
- No depthwise, grouped, dilated, or transposed convolution.
- No activation, normalization, pooling, or residual-addition timing in
  `CnnWorkload` evaluation.
- No cross-layer SRAM residency.

### Presets and experiments (this milestone)

- Accelerator presets are generic experiment configurations
  (`small`/`balanced`/`compute_heavy`/`bandwidth_heavy`), not real
  hardware specifications, and not validated against any real chip.
- Workload presets are generic representative shapes, not reproductions
  of any specific published model or network.
- The bounded-search size limit (`explore.MAX_CANDIDATES`) applies to
  every search launched through presets/experiments exactly as it does
  to the raw CLI -- large candidate sets are rejected up front, not
  silently truncated.

### Validation

- All validation in this project is analytical: independently-derived
  formula cross-checks and known-value hand calculations (see
  `docs/validation.md`). There is no silicon validation, no comparison
  against measured hardware runs, and no claim of predictive accuracy for
  real accelerators.

### General

- No power or energy model of any kind. A tie-break that prefers fewer
  PEs is a deterministic tie-break rule, not an energy or cost claim.
- No area model.
- Not validated against real hardware; not cycle-accurate; results are
  "modeled" or "analytical," never "measured" or "benchmarked."

### TensorForge Ops (Phase 2)

This project's Core limitations above apply unchanged inside Ops
(a `BenchmarkResult`/`SizingPlanResult`/etc. still wraps a Core
experiment with all the modeling limits listed above). Ops adds its own,
documented per-milestone in `docs/ops/`:

- Real benchmarking measures steady-state single-stream latency on one
  process/device -- no production request queue, no dynamic batching, no
  autoscaler dynamics ([benchmarking.md](ops/benchmarking.md)).
- Calibration/validation is empirical against one live device's own
  measured compute/memory probes -- never a mapping of physical GPU
  cores onto TensorForge PEs, and not accurate for every workload shape
  ([calibration.md](ops/calibration.md), [validation.md](ops/validation.md)).
- GPU telemetry is coarse NVML sampling, evidence-worded, and never
  proves a root cause on its own ([gpu-telemetry.md](ops/gpu-telemetry.md)).
- Right-sizing and cost planning use entirely user-supplied prices --
  no live cloud pricing/FX, no deployment automation, no queueing model
  ([right-sizing.md](ops/right-sizing.md)).
- The model-change impact report composes the evidence above into a
  performance-and-infrastructure readiness recommendation only -- no
  composite score, no model-quality/correctness/business-value claim,
  and only one physical GPU has ever produced real evidence for this
  project ([model-change-impact.md](ops/model-change-impact.md)).
