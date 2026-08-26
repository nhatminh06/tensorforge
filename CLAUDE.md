# CLAUDE.md

## Project

TensorForge is an educational AI-accelerator architecture simulator.

Its purpose is to understand how neural-network workloads interact with:

```text
compute
memory bandwidth
memory capacity
processing-element arrays
tiling
data reuse
dataflow
```

The project begins with analytical models and becomes more detailed incrementally.

Long-term progression:

```text
GEMM + Roofline
      ↓
PE Array
      ↓
Memory Hierarchy
      ↓
Tiling
      ↓
Dataflows
      ↓
Convolution
      ↓
Transformer Operations
      ↓
Architecture Sweeps
      ↓
Validation
```

Do not skip foundational models simply to make the simulator appear more sophisticated.

---

## Primary language

Use:

```text
Python 3.12+
```

initially.

Python is appropriate because the early project is about architecture modeling and mathematical reasoning rather than simulator execution speed.

Do not introduce C++, CUDA, Verilog, or SystemVerilog until a concrete later milestone requires them.

---

## Primary goals

Prioritize:

1. mathematical correctness
2. explicit architectural assumptions
3. correct units
4. reproducible calculations
5. understandable models
6. meaningful experiments
7. incremental complexity

Every result should be explainable from its inputs.

---

## Git policy

Do not commit unless explicitly instructed.

Do not push unless explicitly instructed.

Do not:

* force push
* rewrite history
* rebase without permission
* stage unrelated files
* perform destructive Git operations without approval

Before a requested commit report:

```text
files changed
diff summary
tests
documentation
proposed commit message
```

---

## No AI attribution

Never add references to:

* Claude
* Anthropic
* ChatGPT
* OpenAI
* Copilot
* AI-generated
* generated-by
* assisted-by

Do not add AI systems as:

* authors
* contributors
* co-authors

Never add AI `Co-Authored-By` trailers.

---

## Avoid AI slop

Do not add unnecessary:

* managers
* factories
* frameworks
* plugin systems
* interfaces
* abstract base classes
* configuration systems
* logging systems
* helper modules
* TODOs
* comments

Avoid generic names such as:

```text
SimulationManager
AcceleratorManager
ArchitectureFramework
PerformanceEngine
HardwareComponentFactory
```

Prefer concrete architectural concepts:

```text
Gemm
HardwareConfig
RooflineResult
ComputeArray
MemoryHierarchy
Tile
Mapping
```

only when they are actually needed.

---

## Every number must be explainable

Every reported metric must have:

```text
definition
formula
units
inputs
assumptions
```

For example:

```text
MACs = M × N × K
```

If:

```text
1 MAC = 2 FLOPs
```

the convention must be documented.

Do not mix:

```text
MAC
operation
FLOP
cycle
byte
transfer
```

without defining what each means.

---

## Units

Use units carefully.

Prefer internal base units such as:

```text
bytes
seconds
cycles
Hz
FLOP/s
bytes/s
```

Convert only for presentation.

Remember:

```text
GB  = 10^9 bytes
GiB = 2^30 bytes
```

Do not silently mix decimal and binary units.

---

## Analytical vs measured values

Never describe an analytical estimate as a measurement.

Use terms such as:

```text
estimated
modeled
analytical
predicted
```

when appropriate.

Reserve:

```text
measured
observed
```

for actual experiments.

---

## Cycle accuracy

Do not describe TensorForge as:

```text
cycle accurate
```

unless its behavior has explicitly reached and validated that standard.

Early models are analytical.

State that clearly.

---

## GEMM convention

For:

```text
C = A × B

A = M × K
B = K × N
C = M × N
```

initial operation convention:

```text
MACs = M × N × K

FLOPs = 2 × MACs
```

unless a later explicit project-wide decision changes it.

Tests should independently verify this.

---

## Arithmetic intensity

Always specify which memory boundary is being modeled.

Initial convention may be:

```text
Arithmetic intensity =
FLOPs / modeled DRAM bytes
```

with units:

```text
FLOP/byte
```

Do not use the phrase arithmetic intensity without identifying the denominator being modeled when multiple memory levels exist later.

---

## Memory traffic

Do not claim realistic traffic unless the simulator actually models reuse and capacity.

Early model:

```text
read A once
read B once
write C once
```

is an analytical simplifying assumption.

Once tiling and memory hierarchy exist, explicitly distinguish:

```text
DRAM traffic
SRAM traffic
PE-local traffic
```

---

## Roofline

For the analytical roofline model:

```text
memory ceiling =
arithmetic intensity × memory bandwidth
```

and:

```text
attainable performance =
min(peak compute, memory ceiling)
```

Ridge point:

```text
peak compute / memory bandwidth
```

Keep units consistent.

---

## Bottleneck terminology

Do not call something memory-bound or compute-bound without the model supporting that result.

For the basic roofline model:

```text
AI < ridge point
→ memory-bound

AI > ridge point
→ compute-bound
```

Handle equality consistently.

Later models may require more nuanced terminology.

---

## Hardware configuration

Only model hardware parameters that the active milestone uses.

Do not add speculative fields such as:

```text
NoC bandwidth
register-file banks
PE pipeline depth
SRAM ports
cache associativity
```

before those concepts actually participate in calculations.

Unused architecture parameters create fake sophistication.

---

## Processing elements

When PE arrays are introduced, define precisely:

```text
what one PE computes
MACs per PE per cycle
mapping of workload dimensions
tail behavior
utilization
```

Do not assume 100% utilization.

Edge effects must be accounted for.

---

## Memory hierarchy

When memory hierarchy is introduced, make each boundary explicit.

Example:

```text
DRAM
  ↓
Global SRAM
  ↓
PE-local storage
  ↓
MAC
```

Track movement by level.

Do not collapse all memory costs into one value once multiple levels exist.

---

## Tiling

When tiling is introduced, a tile must fit the memory capacity it claims to use.

Validate capacity mathematically.

Do not report reuse that is impossible given the modeled storage.

---

## Dataflows

Do not describe:

```text
weight stationary
output stationary
row stationary
```

as labels only.

For each implemented dataflow, identify what remains stationary and how that changes movement of:

```text
weights
activations
partial sums
```

Compare dataflows with actual modeled traffic.

---

## Transformer modeling

When Transformer support is introduced, decompose it explicitly.

Examples:

```text
Q projection
K projection
V projection
QK^T
softmax
attention × V
output projection
MLP
```

Do not hide the Transformer behind one generic FLOP count.

Intermediate tensors and memory behavior matter.

---

## Real hardware

Do not claim a configuration models:

```text
NVIDIA H100
A100
TPU
Apple Neural Engine
```

without verified specifications.

Generic hardware configurations are preferred initially.

If real hardware is modeled later:

* document sources
* state which details are modeled
* state which details are omitted

---

## Validation

Whenever possible, validate formulas independently.

Do not test only:

```text
implementation → implementation
```

because shared bugs can pass.

Use:

* hand calculations
* known-value tests
* independent scripts
* published formulas
* later, real hardware experiments

where appropriate.

---

## Tests

Tests should focus on mathematical and architectural invariants.

Examples:

```text
known GEMM operation counts
tensor sizes
dtype scaling
arithmetic intensity
ridge point
memory-bound classification
compute-bound classification
tail utilization
tile capacity
traffic accounting
```

Avoid tests that merely mirror implementation code.

---

## Edge cases

Validate inputs.

Reject:

```text
zero dimensions
negative dimensions
zero bandwidth
negative bandwidth
zero peak compute
negative peak compute
NaN
infinity
invalid dtype width
```

Do not allow nonsensical configurations to silently produce results.

---

## Dependencies

Prefer the standard library initially.

Do not add NumPy merely for scalar equations.

Do not add:

```text
PyTorch
TensorFlow
JAX
Timeloop
Accelergy
MAESTRO
```

as the underlying simulator.

Independent tools may later be used for comparison.

---

## Visualization

Do not add charts merely because this is a simulator.

Numerical correctness comes first.

A roofline plot or architecture-sweep plot is appropriate only after the corresponding numerical model is trustworthy.

---

## Optimization

Do not optimize simulator runtime prematurely.

A correct architecture model taking milliseconds longer is preferable to an opaque optimized implementation.

Vectorization and native acceleration should come only when simulation scale actually requires them.

---

## Documentation

README should state:

```text
what TensorForge is
what currently works
how the model works
how to run it
what assumptions exist
what is unsupported
```

Do not turn README into a computer-architecture textbook.

Detailed mathematical assumptions may live in focused documents such as:

```text
docs/model.md
docs/pe-array.md
docs/memory.md
docs/dataflow.md
```

only after those implementations exist.

Do not create empty documentation shells.

---

## No unsupported claims

Do not call TensorForge:

```text
cycle-accurate
hardware-validated
silicon-accurate
production-grade
high-performance
```

unless those claims have actually been demonstrated.

Prefer:

```text
analytical
modeled
estimated
implemented
tested
validated against ...
```

---

## Scope discipline

Follow the current milestone.

Do not implement future features merely because their eventual need is obvious.

If an unrelated architectural defect is discovered:

1. explain the defect
2. explain why it matters
3. propose the smallest fix
4. do not silently redesign unrelated components

---

## Refactoring

Before a meaningful refactor:

1. identify the concrete problem
2. identify affected formulas/invariants
3. inspect tests
4. make the smallest necessary change
5. verify results remain correct

Do not redesign working code for aesthetic preference.

---

## Python style

Use clear Python.

Prefer:

* type hints
* simple dataclasses where they genuinely represent domain values
* explicit calculations
* small pure functions

Avoid:

* deep inheritance
* metaprogramming
* clever operator overloading
* excessive decorators
* generic frameworks

Architecture formulas should be easy to read in source.

---

## Formatting and tooling

Keep tooling light.

One formatter/linter such as:

```text
ruff
```

is enough if needed.

Do not stack:

```text
black
isort
flake8
pylint
ruff
mypy
```

without a concrete reason.

Run:

```bash
pytest
```

for normal verification.

---

## Learning requirement

This repository is not only being built; it is being used to learn computer architecture.

After every meaningful milestone, explain what was built so the user can understand it.

The final report must include a concise learning section.

It should answer:

```text
What did we just add?
Why does it matter?
How do the major pieces interact?
What should I understand before continuing?
```

Do not provide a long lecture.

---

## Code walkthrough requirement

After each milestone, identify only the most important:

```text
2–5 files/functions/classes
```

for the user to read.

For each one explain briefly:

```text
what it does
why it matters
what concept to notice
```

Do not tell the user to read the entire repository.

---

## Hands-on commands

After every milestone, provide exact copyable commands that let the user reproduce important behavior.

Examples may include:

```bash
pytest -q
```

and one or more actual TensorForge runs.

Commands must match the implementation that currently exists.

Never invent CLI flags.

---

## Parameter experiment

Whenever possible, provide one small experiment where the user changes exactly one architectural/workload parameter and observes the consequence.

Examples:

```text
memory bandwidth
peak compute
dtype size
PE dimensions
SRAM capacity
tile size
```

Explain what they should observe, but do not replace their experiment by doing everything for them.

---

## Manual calculation

For mathematically important milestones, provide one tiny hand-checkable example.

Example:

```text
M=2
N=3
K=4
```

The user should be able to derive the important numbers manually and compare them against TensorForge.

This ensures the simulator does not become a black box.

---

## Check-your-understanding section

After each milestone, provide 2–4 questions the user should be able to answer.

Also provide short answers so they can self-check.

Questions should target the central concept, not trivia.

For example:

```text
Why does arithmetic intensity stay constant if bandwidth changes?

What determines the roofline ridge point?

Why is the initial DRAM traffic model optimistic?
```

---

## Learning report format

After the normal engineering report append:

```text
## What you should understand

## Code to read

## Commands to run yourself

## Manual calculation

## Check your understanding

## Hands-on exercise
```

Keep this section concise.

Do not repeat the engineering report.

---

## Before implementation

For non-trivial work briefly state:

1. what currently exists
2. what will change
3. the mathematical/architectural assumption being introduced

Then implement.

Do not produce a long speculative essay.

---

## After implementation

Report:

### Changed

What actually changed.

### Model

What mathematical or architectural model now exists.

### Assumptions

What simplifying assumptions remain.

### Verification

Exact tests/commands actually run.

### Limitations

What the model still cannot represent.

### Learning

Give the short walkthrough required above.

---

## Definition of done

A feature is not complete merely because code exists.

It should normally have:

```text
clear formula
defined units
explicit assumptions
input validation
known-value test
boundary tests
documentation
pytest passing
narrow diff
no unsupported claims
learning walkthrough
copyable commands
```

---

## Project mindset

When choosing between:

```text
more hardware features
```

and:

```text
understanding one performance model
```

understand the model.

When choosing between:

```text
more realistic-looking output
```

and:

```text
numbers that can be manually verified
```

choose verifiable numbers.

When choosing between:

```text
complex simulator architecture
```

and:

```text
explicit equations
```

prefer explicit equations.

When choosing between:

```text
simulator as a black box
```

and:

```text
the user understanding why every result exists
```

make the result understandable.

---

## TensorForge Ops (Phase 2)

TensorForge Core (`src/tensorforge/`) is frozen as a modeling boundary.

- Core must never import `mlflow` or `tensorforge_ops`. Dependency
  direction is strictly `tensorforge_ops -> tensorforge`, never the
  reverse.
- `tensorforge_ops` consumes an already-computed, immutable
  `ExperimentResult` -- it never re-derives or adjusts an analytical
  value to make it "fit" MLflow.
- Tracking is a side effect layered around a deterministic Core result,
  not a source of analytical truth. `core-result.json` (the exact
  `ExperimentResult.to_json()` bytes) is authoritative; MLflow
  params/metrics/tags are an index for filtering and comparison only.
- No silent tracking fallback: if a user explicitly configures a
  tracking URI and it is unreachable, fail clearly. Never substitute a
  different backend than the one requested.
- No AI attribution, ever, in Ops code either.
- Do not commit changes until pytest passes, a real local MLflow
  verification has been run, and the Core/Ops import boundary has been
  checked.
- Real hardware benchmarks (a later milestone) must always be recorded
  and labeled separately from analytical predictions -- never merge a
  measured value into a field that currently means "modeled" or
  "analytical," and never call an analytical output "benchmark data."
- Performance claims require actual measurements; an analytical
  TensorForge result is a prediction, not a benchmark, until compared
  against one.
- The learning-report requirement from Phase 1 continues to apply to
  every Phase-2 milestone.
- Real benchmark fields always use an explicit `measured_*` prefix
  (e.g. `measured_mean_latency_seconds`); Core's analytical field names
  (e.g. `perfect_overlap_time_seconds`) are never overwritten or reused
  for a measured value.
- A benchmark backend (PyTorch, later ONNX Runtime) is a runtime
  dependency of `tensorforge_ops` only, imported lazily so the rest of
  Ops (and all of Core) keeps working with that backend not installed.
- CUDA timing must be explicitly synchronized (`torch.cuda.synchronize()`
  around the measured region); never time only kernel-launch/enqueue.
  Warmup iterations are always excluded from statistics. Never silently
  fall back from a requested device/backend to a different one -- fail
  clearly instead.
- Do not describe a benchmark as measuring a specific real accelerator
  (e.g. "RTX 4090") unless it is the literal physical machine the
  benchmark ran on; Core's generic accelerator presets are not the
  benchmarking machine.
- Never map a physical GPU's cores/SMs/Tensor Cores directly onto
  TensorForge PE rows/columns, and never add a real-GPU accelerator
  preset built from marketing specifications; physical-device
  predictions must be built from empirically measured compute/memory
  ceilings, not an invented PE mapping.
- Calibration probes (synthetic compute/memory-copy probes) and
  validation workloads (Core workload presets) are separate sets;
  never tune an effective calibrated rate to reduce error on a specific
  validation workload.
- Measured p50 latency is the primary target for prediction-vs-
  measurement validation, not p95/mean/min/max.
- "Predicted bottleneck" (which term of an analytical formula is larger)
  and "measured bottleneck" (an actual hardware-execution claim) are
  distinct terms; never claim a measured bottleneck without device
  telemetry to support it.
- No fitted correction coefficient, additive overhead term, or
  workload-specific fudge factor in a calibrated/empirical prediction
  without an explicit, documented model or validation rationale.
- A device/runtime/dtype mismatch between a calibration profile and a
  benchmark result invalidates the comparison -- reject it clearly
  rather than computing a meaningless error.
- Large prediction error is a legitimate validation finding to report,
  not a defect to hide or fit away.
- Calibration artifacts (`calibration-profile.json`), benchmark
  artifacts, Core artifacts, and validation artifacts stay separate
  files/schemas -- never merged into one combined schema.
- PR performance regression gates use measured base-vs-candidate
  BenchmarkResult metrics, never Core's predicted/calibrated latency;
  analytical/calibration data may appear only as explanatory context.
- Baseline and candidate benchmarks must run on the same physical
  runner/job where possible; do not compare results from separate,
  unrelated runner instances as if they were controlled.
- No hidden regression thresholds -- every gate value comes from an
  explicit, committed policy file.
- A workload/backend/device/dtype/schema mismatch between baseline and
  candidate benchmarks invalidates the comparison (ERROR), never a
  computed pass/fail.
- A metric that is `None` on both sides is NOT_COMPARABLE, never
  converted to zero; a metric missing on only one side is a comparison
  ERROR (measurement coverage changed).
- Regression FAIL (valid comparison, policy violated) is a distinct
  status from ERROR (comparison could not be validly performed) -- never
  conflate the two, including in CLI exit codes.
- Never use `pull_request_target` to execute untrusted PR code.
- Never weaken a regression policy solely to make a specific PR pass;
  changing a threshold requires the same evidence-based justification as
  setting it the first time.
- No AI attribution.
