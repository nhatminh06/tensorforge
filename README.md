# TensorForge

An analytical AI-accelerator performance modeling toolkit for studying how
GEMM, Transformer, and Conv2D workloads interact with PE geometry, SRAM
capacity, tiling, data residency, and DRAM bandwidth.

## Why TensorForge

Understanding accelerator performance requires reasoning across several
layers at once: how much arithmetic a workload needs, how it maps onto a
finite PE array, whether its working set fits in SRAM, how tiling and
loop order change data movement, and whether the result is bound by
compute or by memory bandwidth. TensorForge makes every one of those
steps explicit and traceable to a formula, rather than hiding them behind
a single opaque "runtime" number.

## What it models

- **GEMM**: MAC/FLOP counts, tensor byte footprints, an ideal roofline
  model, rectangular PE-array mapping with spatial utilization, SRAM
  capacity checks, explicit M/N/K tiling under three residency schedules
  (`c-resident`/`a-resident`/`b-resident`) with exact DRAM traffic, and
  analytical compute/memory timing.
- **Transformer**: a GEMM-heavy block (Q/K/V/output projections,
  attention scores/value, MLP up/down) decomposed from `batch`,
  `sequence_length`, `d_model`, `num_heads`, `d_ff`, with per-operation
  tile/schedule mapping on one shared, fixed PE array.
- **Conv2D / CNN**: convolution lowered through an explicit
  **materialized-im2col** GEMM mapping, plus a chained Conv-only CNN
  workload representation.
- **Bounded design-space exploration**: a deterministic search over
  user-supplied tile/schedule/PE-array candidates, ranked by modeled
  execution time.
- **Reproducible presets and experiments**: named workload and
  accelerator configurations that produce deterministic, machine-readable
  JSON results.

TensorForge is a design-space explorer and analytical modeling toolkit,
not a cycle-accurate simulator, not an architecture optimizer, and not a
complete Transformer/CNN inference simulator (see [Limitations](#limitations)).

## Architecture

```
Workload (GEMM / Transformer / Conv2D / CNN)
        |
Gemm normalization  ->  PE mapping  ->  SRAM feasibility
        |                                     |
        +-------------- Tiling + schedule ----+
                            |
                          Timing
                            |
                  Bounded exploration
                            |
              Presets + deterministic experiments
```

See [docs/architecture.md](docs/architecture.md) for the full module map.

## Example

```bash
python -m tensorforge --m 1024 --n 1024 --k 1024 --dtype fp16 \
    --peak-tflops 10 --bandwidth-gbps 200 --pe-rows 32 --pe-cols 32
```

prints GEMM operation counts, tensor byte sizes, roofline ceilings,
PE-array mapping and utilization, and a compute-bound/memory-bound
classification.

## Workloads

- **GEMM** — `python -m tensorforge --m M --n N --k K ...`
- **Transformer** — `python -m tensorforge --transformer-block --batch-size ... --seq-len ... --d-model ... --num-heads ... --d-ff ...`
- **Conv2D** — `python -m tensorforge --conv2d --batch-size ... --in-channels ... --input-height ... --input-width ... --out-channels ... --kernel-h ... --kernel-w ...`

Each mode accepts `--dtype`, `--peak-tflops`, `--bandwidth-gbps`, and
optionally `--sram-kib`, `--pe-rows`/`--pe-cols`, tile candidate lists,
and `--schedule-values`. See `python -m tensorforge --help`.

## Accelerator model

One global SRAM, one rectangular PE array (`1 PE = 1 MAC/cycle`, `1 MAC =
2 FLOPs`), one DRAM bandwidth figure, one PE clock. Compute time is
`compute_cycles / clock_hz`; memory time is `modeled_dram_bytes /
bandwidth`; both a no-overlap (`serialized`) and full-overlap
(`perfect_overlap`) bound are reported. See
[docs/timing.md](docs/timing.md).

## Mapping model

A GEMM is split into `Tm x Tn x Tk` tiles that must fit the modeled SRAM.
Three explicit, fully-specified loop schedules (`c-resident`,
`a-resident`, `b-resident`) each produce a different, exactly-derived
DRAM traffic total from the same arithmetic work. See
[docs/tiling.md](docs/tiling.md) and [docs/schedules.md](docs/schedules.md).

## Design-space exploration

`explore()` evaluates the full Cartesian product of user-supplied tile,
schedule, and PE-array candidates (bounded, with a hard size limit),
ranks them deterministically by modeled execution time, and reports the
**best among searched candidates** — never a claim of global optimality.
See [docs/exploration.md](docs/exploration.md).

## Reproducible experiments

```bash
python -m tensorforge --list-presets

python -m tensorforge --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128

python -m tensorforge --workload-preset transformer_medium --accelerator-preset compute_heavy \
    --tile-m-values 32,64 --tile-n-values 32,64 --tile-k-values 32,64 \
    --output-json result.json
```

Presets are generic, reproducible configurations (`gemm_tiny`,
`transformer_medium`, `conv_spatial`, `cnn_like_small`, `small`,
`balanced`, `compute_heavy`, `bandwidth_heavy`, ...) — not claims about
any real model or chip. `--output-json` writes a deterministic result
(`schema_version: 1`, sorted keys, no timestamps): the same experiment
run twice produces byte-identical JSON.

## Validation

All validation is analytical: independently hand-derived known-value
checks, closed-form invariant cross-checks, and scaling-law tests — no
silicon validation. 223 tests currently pass (`pytest -q`). See
[docs/validation.md](docs/validation.md) and
[docs/experiments.md](docs/experiments.md) for the full studies,
including a 4-accelerator x 4-workload tradeoff study, compute/memory
crossover demonstrations, and SRAM/bandwidth/PE/sequence-length/kernel-size
sensitivity sweeps.

## What TensorForge demonstrates

Roofline reasoning; PE-array spatial utilization and its sensitivity to
array shape (not just PE count); SRAM capacity as a hard constraint on
feasible tiling; how loop order/residency changes DRAM traffic for
identical arithmetic; compute-bound vs. memory-bound classification and
the crossover between them; bounded, deterministic architecture search;
Transformer workload decomposition and quadratic-vs-linear scaling;
convolution-to-GEMM lowering and its memory-footprint tradeoffs;
reproducible, machine-readable performance experiments.

## Limitations

Every layer's limitations are documented in full in
[docs/limitations.md](docs/limitations.md). In short: analytical only
(not cycle-accurate); one global SRAM with no latency model;
bandwidth-only DRAM timing; three fixed residency schedules over
user-supplied, bounded candidates; Transformer/CNN modeling is GEMM-only
(no softmax/normalization/activation/pooling/residual, no head
parallelism, no fusion, no cross-operation SRAM residency); convolution
uses one materialized-im2col lowering (no direct convolution, no
grouped/depthwise/dilated convolution); no power/energy or area model;
no real-hardware validation.

## Repository structure

```
src/tensorforge/
  gemm.py roofline.py pe_array.py memory.py     core analytical models
  tiling.py timing.py explore.py                mapping, timing, search
  transformer.py convolution.py                 workload decomposition
  presets.py experiments.py                     reproducible experiments
  cli.py                                         command-line interface
tests/            one test file per module, 223 tests total
docs/             per-topic model docs + architecture/design/validation/
                  limitations/experiments
scripts/          demo.sh, validate.sh
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q

python -m tensorforge --list-presets
python -m tensorforge --workload-preset gemm_tiny --accelerator-preset balanced \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128

bash scripts/demo.sh
```

## Next phase

TensorForge Core is now frozen as a standalone modeling engine.
`experiments.py` already produces deterministic JSON designed to be
consumed by an external tracking layer without TensorForge Core ever
depending on it. A future ML Systems / MLOps phase (experiment tracking,
CI performance regression, benchmark validation) would build on top of
this interface, not inside it.
