# Design decisions

Concise record of the important, deliberate modeling choices made across
TensorForge Core, and why. Full derivations live in the per-topic docs;
this is the "why we chose this" index.

- **Analytical, not cycle-accurate.** Every number is a closed-form
  formula over explicit inputs (dimensions, dtype, PE geometry, SRAM
  capacity, clock, bandwidth) -- there is no cycle-by-cycle simulation
  loop. This keeps every result traceable to a formula a reader can
  re-derive by hand (see `docs/validation.md`).

- **1 PE = 1 MAC/cycle, 1 MAC = 2 FLOPs.** The simplest possible compute
  primitive. Chosen so PE throughput, cycle counts, and FLOP conversions
  stay hand-checkable; wider/vectorized PEs are a natural but unimplemented
  extension.

- **Materialized im2col for convolution**, not direct/implicit convolution,
  Winograd, or FFT convolution. Chosen because it reduces Conv2D to the
  exact same `Gemm` abstraction everything else already uses -- no new
  compute/memory/timing model was needed. The tradeoff is explicit and
  documented: materialized im2col can substantially overstate real DRAM
  traffic for kernels larger than 1x1, because overlapping receptive
  fields are counted at full duplicated size.

- **Three fixed, explicit residency schedules** (`c-resident`,
  `a-resident`, `b-resident`) rather than a general dataflow description
  language. Each is a literal, fully-specified loop nest with a
  closed-form traffic formula -- chosen so traffic differences between
  schedules are provably exact, not simulated or approximated.

- **Bandwidth-only DRAM timing.** `memory_time = bytes / bandwidth`, with
  no DRAM/SRAM latency, controller overhead, or queueing. Chosen as the
  simplest model that still produces a meaningful compute-vs-memory
  comparison; adding latency would require a request-level memory
  transaction model, a substantially larger scope.

- **Two timing bounds, not one estimate.** `serialized_time` (no overlap)
  and `perfect_overlap_time` (full overlap) bracket the real answer
  rather than guessing at a specific overlap percentage. No partial-overlap
  coefficient is invented.

- **Bounded discrete exploration**, not continuous optimization. The user
  supplies finite tile/schedule/PE candidate lists; `explore()` evaluates
  every combination (up to a hard size limit) and ranks them
  deterministically. This was chosen over `scipy.optimize`/genetic/
  Bayesian search because every result stays exactly attributable to an
  explicit, inspectable candidate -- "best among searched candidates,"
  never "the optimum."

- **No cross-operation/cross-layer SRAM residency.** Each GEMM (a
  Transformer operation, a CNN layer) is evaluated independently at its
  own DRAM boundary. This is a known pessimism: a real fused kernel
  keeping one layer's output resident for the next would move less data.
  Modeling that residency is future work, not silently assumed here.

- **Deterministic experiment JSON with no wall-clock timestamp.** The
  same `ExperimentSpec` must produce byte-identical `to_json()` output on
  every run, because every function in the data path is pure. This makes
  the JSON usable as a stable artifact for future tooling (diffing,
  caching, regression comparison) without needing to special-case time.

- **No real-hardware claim anywhere.** Accelerator presets are named
  `small`/`balanced`/`compute_heavy`/`bandwidth_heavy` -- generic
  experiment configurations, never real GPU/TPU model names, and never
  presented as validated against silicon.
