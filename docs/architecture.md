# TensorForge Core architecture

Concise implementation map. For per-layer formulas see the milestone docs
(`model.md`, `pe-array.md`, `memory.md`, `tiling.md`, `schedules.md`,
`timing.md`, `exploration.md`, `transformer.md`, `convolution.md`).

## Data path

```
Workload (GEMM / Transformer / Conv2D / CNN)
        |
Gemm normalization                    gemm.py
   M, N, K, dtype -> MACs, FLOPs, tensor bytes
        |
PE mapping                            pe_array.py
   PeArray(rows, cols) -> waves, spatial utilization, compute cycles
        |
Memory capacity                       memory.py
   MemoryHierarchy(sram_bytes) -> full-tensor fit/headroom/deficit
        |
Tiling + schedule                     tiling.py
   GemmTile + GemmSchedule -> exact DRAM traffic under one loop schedule
        |
Timing                                timing.py
   TimingConfig(clock_hz) + HardwareConfig -> compute/memory/serialized/
   perfect-overlap time, bottleneck classification
        |
Bounded exploration                   explore.py
   Cartesian product of tile/schedule/PE candidates -> ranked, deterministic
        |
Workload-specific composition         transformer.py / convolution.py
   Derive N GEMMs from workload dimensions, evaluate each via explore(),
   aggregate sequentially
        |
Presets + experiments                 presets.py / experiments.py
   Named reproducible configurations -> deterministic JSON result
```

## Module responsibilities (unchanged since Milestone 2-9; preserved this milestone)

| Module | Responsibility |
|---|---|
| `gemm.py` | Mathematical workload: `Gemm(M,N,K,dtype)`, MAC/FLOP/byte formulas. |
| `hardware.py` | Roofline-facing hardware limits: peak FLOP/s, DRAM bandwidth. |
| `roofline.py` | Milestone-1 ideal roofline model (unchanged, kept separate). |
| `pe_array.py` | Compute geometry: `PeArray(rows, cols)`, spatial mapping, cycles. |
| `memory.py` | SRAM capacity: `MemoryHierarchy(sram_bytes)`, full-tensor fit. |
| `tiling.py` | Tile geometry (`GemmTile`) + residency semantics (`GemmSchedule`) -> exact DRAM traffic under one schedule. |
| `timing.py` | Converts cycles/bytes into time: `TimingConfig`, `estimate_execution_time()`. |
| `explore.py` | Bounded, deterministic candidate evaluation and ranking (`explore()`). |
| `transformer.py` | Transformer block -> ordered list of GEMMs, per-op mapping, aggregation. |
| `convolution.py` | Conv2D -> materialized-im2col GEMM, CNN layer chaining, aggregation. |
| `presets.py` | Named, reproducible workload/accelerator configurations (this milestone). |
| `experiments.py` | Wires a workload preset + accelerator preset + search config into a deterministic JSON result (this milestone). |
| `cli.py` | User interface only -- argument parsing and text/JSON output; contains no analytical formulas. |

This milestone did not change these boundaries: `presets.py`/`experiments.py`
only call the existing public functions of the modules above (`explore()`,
`evaluate_transformer_block()`, `evaluate_conv2d()`,
`evaluate_cnn_workload()`) -- no formula was duplicated or moved.

## Workload-specific fronts

**Transformer**: `derive_transformer_gemms(spec)` turns
`(batch, seq_len, d_model, num_heads, d_ff)` into 8 named GEMMs (Q/K/V/
output projections, attention scores/value with `batch*heads`
repetitions, MLP up/down). Each is evaluated independently via
`explore()` on one fixed, shared PE array; results are summed
sequentially (no cross-operation SRAM residency).

**Conv2D**: `lower_conv2d_to_gemm(spec)` turns Conv2D dimensions into one
materialized-im2col `Gemm(M=batch*Hout*Wout, N=Cout, K=Cin*Kh*Kw)`. A
`CnnWorkload` chains multiple `Conv2DSpec`s (each layer's input shape
derived from the previous layer's output) and aggregates them the same
way as a Transformer block: fixed PE array, per-layer mapping, sequential
sum, no cross-layer SRAM residency.

## Presets and experiments (this milestone)

```
presets.py          -- named GemmPreset/TransformerPreset/ConvPreset/CnnPreset
                        and AcceleratorPreset, purely declarative
        |
experiments.py       -- ExperimentSpec (name + preset names + search config)
                         run_experiment() calls the existing evaluators
                         ExperimentResult.to_dict()/to_json() -- deterministic,
                         schema_version=1, sort_keys=True, no timestamps
        |
cli.py               -- --list-presets, --workload-preset/--accelerator-preset,
                         --output-json/--force
```

Dependency direction is one-way: `presets`/`experiments` depend on the
workload models, which depend on the core analytical models. Nothing in
`gemm.py`...`explore.py` imports from `presets.py` or `experiments.py`.
A future Phase-2 tracking layer would sit *outside* this diagram,
consuming the deterministic JSON `experiments.py` already produces --
TensorForge Core has no dependency on it.
