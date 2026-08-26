# Milestone 9 model: Conv2D via materialized-im2col GEMM lowering

## Scope and core decision

Conv2D is modeled by deriving its output geometry, then lowering it to a
GEMM via **materialized im2col**, and delegating everything else (PE
mapping, SRAM capacity, tiling, residency schedules, timing, bounded
exploration) to the existing GEMM accelerator model unchanged. This is
one explicit, deliberately simple lowering choice — not a claim that
real accelerators execute convolution this way, and not a claim that
materialized im2col is optimal. Direct/implicit convolution dataflows,
Winograd, and FFT-based convolution are different, unmodeled approaches.

Only `dilation=1, groups=1` is supported: no depthwise, grouped, dilated,
or transposed convolution.

## Conv2DSpec

```
Conv2DSpec(batch_size, in_channels, input_height, input_width,
           out_channels, kernel_height, kernel_width,
           stride_height, stride_width,
           padding_height, padding_width, dtype)
```

All dims positive ints except padding (`>= 0`). Output dimensions are
validated at construction time (accessing them immediately, so an
invalid — non-positive — output geometry fails at construction, not
silently later).

## Output dimensions

```
output_height = (input_height + 2*padding_height - kernel_height) // stride_height + 1
output_width  = (input_width  + 2*padding_width  - kernel_width)  // stride_width  + 1
```

Rejected if either is `<= 0`.

## Direct-convolution compute (independently derived)

```
MACs  = batch_size * output_height * output_width
        * out_channels * in_channels * kernel_height * kernel_width
FLOPs = 2 * MACs
```

Computed via `Conv2DSpec.macs`/`.flops` using this direct formula —
independently of the lowered GEMM's `M*N*K`, so tests can verify the two
formulas agree (`spec.macs == lowering.gemm.macs`) as a genuine
mathematical cross-check, not a self-referential one.

## Im2col lowering

```
M = batch_size * output_height * output_width   (one row per output position)
N = out_channels
K = in_channels * kernel_height * kernel_width   (flattened receptive field)

Gemm(M, N, K, dtype)
```

Note: convolution literature often uses `N` for batch size — this
project uses `batch_size` for the convolution's batch dimension and
reserves `N` strictly for `Gemm`'s output-column dimension, to avoid
collision.

`lower_conv2d_to_gemm(spec)` is pure and deterministic: it derives shapes
and byte counts only — no tensor is ever allocated, no NumPy, no actual
convolution numerics.

## Logical tensor footprints

```
input_bytes  = batch_size * in_channels * input_height * input_width * bytes_per_element
weight_bytes = out_channels * in_channels * kernel_height * kernel_width * bytes_per_element
output_bytes = batch_size * out_channels * output_height * output_width * bytes_per_element
```

These correspond exactly to the lowered GEMM's operand bytes:
`weight_bytes == gemm.b_bytes`, `output_bytes == gemm.c_bytes`.

## Materialized im2col footprint

```
im2col_bytes = M * K * bytes_per_element
             = (batch_size * output_height * output_width * in_channels
                * kernel_height * kernel_width) * bytes_per_element
             == gemm.a_bytes
```

Zero-padded positions are counted at full size — no sparse compression
of padding is modeled.

## Expansion ratio

```
im2col_expansion_ratio = im2col_bytes / input_bytes
```

Not assumed to always exceed 1 (a 1x1, stride-1, no-padding convolution
gives exactly 1.0, since each input pixel maps to exactly one im2col
row-column position with no duplication). Larger kernels increase both
`K` and the materialized footprint by roughly `kernel_height *
kernel_width`, because each input pixel now appears in every overlapping
receptive field it belongs to.

## Central memory-model semantics

Because the existing tiling/schedule model reads/writes `Gemm.a_bytes`
(and its tiled variants) to compute DRAM traffic, applying that model to
this lowering means **the "A" operand traffic is materialized im2col
traffic — a fully-expanded activation-patch matrix — not direct-
convolution input traffic.** Overlapping receptive fields cause the same
input pixel to appear in multiple im2col rows; the schedule model sees
and accounts for every one of those duplicated appearances. A real
direct-convolution accelerator reading straight from the input tensor
would generally move far less data. This is never described as "direct
convolution traffic" or "actual CNN accelerator traffic" anywhere in this
codebase's output.

## Conv2D evaluation

`evaluate_conv2d(spec, pe_array, hierarchy, hardware, timing_config,
tile_m_values, tile_n_values, tile_k_values, schedules=None)` lowers the
spec to a `Gemm`, then calls the existing `explore.explore()` with the PE
candidate set restricted to exactly the one PE shape given, taking the
top-ranked (`top_k=1`) result — the same ranking objective
(`perfect_overlap_time_seconds`) as every other GEMM in this project. No
new ranking or evaluation logic is introduced.

`explore_conv2d_architectures(...)` compares candidate PE-array shapes
for one Conv2D the same way Milestone 8 compares them for a Transformer
block: same deterministic tie-break order (perfect-overlap time →
serialized time → DRAM traffic → PE count → rows → cols), same
`MAX_CANDIDATES`-based search-size bound checked before evaluating
anything.

## Conv-only CNN workload

```
CnnWorkload(batch_size, in_channels, input_height, input_width, dtype,
            layers=(CnnLayerSpec(name, out_channels, kernel_h, kernel_w,
                                  stride_h, stride_w, padding_h, padding_w), ...))
```

**Chained geometry** (the simpler of the two designs considered): each
layer specifies only its own `out_channels`/kernel/stride/padding: its
input channels and spatial size are derived automatically from the
previous layer's output (or the workload's input shape, for the first
layer). No independent per-layer input-shape override, no pooling, no
hidden resizing.

`evaluate_cnn_workload(...)` evaluates every layer on one **fixed** PE
array (shared hardware), letting each layer choose its own tile/schedule
mapping (different layer shapes may prefer different mappings). Layers
execute **sequentially with no cross-layer SRAM residency** — each layer
is evaluated independently at its own DRAM boundary, so total CNN DRAM
traffic is the sum of independently-evaluated layers (a layer's output
being kept resident for the next layer's input is not modeled — this can
overcount realistic fused/resident CNN traffic). Aggregation:

```
total_macs   = sum(layer.spec.macs)
total_flops  = sum(layer.spec.flops)
total_dram_bytes            = sum(layer.dram_bytes)
serialized_time_seconds     = sum(layer.serialized_time)
perfect_overlap_time_seconds = sum(layer.perfect_overlap_time)
```

`explore_cnn_architectures(...)` mirrors the single-Conv2D and
Transformer architecture search: PE array fixed across all layers of one
candidate, per-layer mapping free, same deterministic ranking.

## Modeled vs. unmodeled

Modeled: Conv2D layers only (materialized-im2col GEMM lowering).

Unmodeled (excluded, not free): activation, normalization, pooling,
residual additions, and fully connected/classifier layers (unless
manually represented as a separate `Gemm`).

## Limitations

- Materialized-im2col convolution only — no direct/implicit convolution
  dataflow, no Winograd, no FFT convolution.
- No depthwise, grouped, dilated, or transposed convolution.
- No activation, normalization, pooling, or residual-addition timing.
- No cross-layer SRAM residency in `CnnWorkload` evaluation.
- Not a claim of real CNN inference latency, real-hardware accuracy, or
  cycle accuracy.
- No power/energy model.
