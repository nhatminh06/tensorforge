# Milestone 8 model: Transformer GEMM-block workload

## Scope

This models the **GEMM-heavy portion** of one Transformer block: it
decomposes B/S/D/H/F dimensions into 8 GEMM operation groups, evaluates
each on one shared, fixed accelerator using the existing tiling/PE/timing
models, and aggregates the result. The output is a **GEMM-only
Transformer block estimate**, never "complete Transformer block latency"
— several real, non-trivial costs are explicitly excluded (see below).

## TransformerBlockSpec

```
TransformerBlockSpec(batch_size=B, sequence_length=S, d_model=D,
                      num_heads=H, d_ff=F, dtype=...)
```

All of B, S, D, H, F must be positive ints. `D % H == 0` is enforced
(head_dim must be integral):

```
head_dim = D // H
```

## Derived GEMMs (in block execution order)

| Operation | M | N | K | Repetitions |
|---|---|---|---|---|
| `q_projection` | B*S | D | D | 1 |
| `k_projection` | B*S | D | D | 1 |
| `v_projection` | B*S | D | D | 1 |
| `attention_scores` | S | S | head_dim | B*H |
| `attention_value` | S | head_dim | S | B*H |
| `output_projection` | B*S | D | D | 1 |
| `mlp_up` | B*S | F | D | 1 |
| `mlp_down` | B*S | D | F | 1 |

- **Q/K/V/output projections**: `X: (B*S) x D` times a `D x D` weight.
- **Attention scores**: per head, `Q_head: S x head_dim` times
  `K_head^T: head_dim x S`, repeated once per (batch, head) pair — **B*H
  repetitions execute sequentially** on the modeled accelerator (no head
  parallelism). Aggregate contribution = single-execution value * B*H.
- **Attention value**: per head, `probs: S x S` times `V_head: S x head_dim`,
  same B*H repetition and sequential-execution assumption.
- **MLP up/down**: `(B*S) x D` times `D x F`, then `(B*S) x F` times `F x D`.

## Modeled FLOP formula

Since `H * head_dim = D`, the modeled GEMM FLOPs simplify to:

```
8*B*S*D^2      (Q + K + V + output projections: 4 * 2*B*S*D^2)
+ 4*B*S*D*F    (MLP up + down: 2*B*S*D*F + 2*B*S*F*D)
+ 4*B*S^2*D    (attention scores + value: each 2*B*S^2*D)
```

For `B=1, S=2, D=4, H=2, F=8` (head_dim=2): `8*1*2*16 + 4*1*2*4*8 + 4*1*4*4
= 256 + 256 + 64 = 576` FLOPs — independently hard-coded and verified in
`tests/test_transformer.py`.

## Block evaluation: fixed hardware, per-operation mapping

Within one block evaluation, the **PE array is identical for every
operation** — it represents the physical accelerator, and letting
different operations use different PE shapes would represent different
hardware. **Tile shape and residency schedule may differ per operation**,
chosen independently from the same user-supplied candidate sets, because
Transformer GEMM shapes differ substantially (e.g. attention's small
`S x head_dim x S` vs. MLP's much larger `B*S x D x F`) — forcing one
tile shape across all of them would be needlessly restrictive.

For each operation, the existing bounded explorer (`explore.explore()`)
is called with the operation's `Gemm`, the shared SRAM/clock/bandwidth,
the shared tile/schedule candidates, and the PE candidate set restricted
to exactly the one PE shape being evaluated — then the top-ranked
(`top_k=1`) result is taken as that operation's mapping. No new ranking
logic is introduced; the existing `perfect_overlap_time_seconds`
objective and tie-break order from Milestone 7 are reused unchanged.

Operations that share an identical GEMM shape (Q/K/V/output projections
all have `M=B*S, N=D, K=D`) are explored **once** and the result reused —
a simple dict cache keyed on `(M, N, K, dtype)`, not a generic caching
framework.

**Repetition does not change the best mapping**: every candidate's time
for a repeated operation is multiplied by the same positive repetition
count, so the ranking order among candidates is unaffected — the
exploration therefore evaluates each GEMM *shape* once regardless of how
many times it repeats, and only the final aggregation multiplies by
repetitions.

## Block timing aggregation (operations execute sequentially)

Operations are modeled as executing **sequentially** — perfect overlap
is allowed *inside* each GEMM between its own compute and DRAM transfer
(as in Milestone 6), but operations do **not** overlap with each other:

```
block_perfect_overlap_time = sum(op.perfect_overlap_time * op.repetitions)
block_serialized_time      = sum(op.serialized_time * op.repetitions)
block_total_dram_bytes     = sum(op.total_dram_bytes * op.repetitions)
```

This DRAM total only includes traffic from the current GEMM/schedule
model — it does **not** include softmax, normalization, activation, or
residual traffic (all unmodeled).

## Largest contributors

`largest_time_contributor` / `largest_dram_contributor` identify the
operation with the largest aggregate (`perfect_overlap_time *
repetitions`) / (`dram_bytes * repetitions`) respectively, using stable
`max()` over operations in block execution order (ties resolve to
whichever operation appears first in that fixed order). Called "largest
modeled GEMM time/DRAM contributor" — never "the bottleneck," since
non-GEMM operations aren't in the comparison at all.

## Architecture exploration across the block

`explore_transformer_architectures()` compares candidate **PE-array
shapes** (rows x columns) across the whole block, with the Transformer
spec, dtype, SRAM, clock, bandwidth, and tile/schedule candidate sets
held fixed. For each PE candidate, every block GEMM independently
re-selects its own best tile/schedule mapping (since a different PE
shape can change which mapping is best). Ranked by
`block_perfect_overlap_time_seconds` ascending; ties broken by
`block_serialized_time_seconds`, then `block_total_dram_bytes`, then PE
count, then PE rows, then PE columns — the same style of deterministic
tie order as Milestone 7's single-GEMM ranking.

A search-size bound (reusing `explore.MAX_CANDIDATES`) is checked before
evaluating anything: `architectures * unique_GEMM_shapes *
per_shape_candidates` must not exceed the limit, or the search is
rejected with a clear error.

Terminology: results are reported as **"best PE architecture among
searched candidates"** — never "optimal Transformer accelerator" or
"best Transformer hardware." The PE candidate list is finite, the memory
model is idealized, non-GEMM operations are entirely absent, power/area
are unmodeled, and there is no real-hardware validation.

## Explicitly modeled operations

Q projection, K projection, V projection, attention-score GEMM,
attention-value GEMM, output projection, MLP up projection, MLP down
projection.

## Explicitly unmodeled operations

Softmax, normalization (LayerNorm/RMSNorm), activation (GELU/SwiGLU/etc.),
residual additions. These are **excluded from the aggregated numbers**,
not treated as zero-cost — `TransformerBlockResult.unmodeled_operations`
lists them explicitly so the coverage gap is never hidden.

## Limitations

- GEMM-only Transformer block estimate — not complete Transformer block
  latency, not LLM inference latency.
- Softmax, normalization, activation, and residual operations are
  entirely unmodeled (excluded, not free).
- Attention heads (B*H repetitions) are modeled executing sequentially —
  no head parallelism.
- No QKV fusion, no attention fusion (no FlashAttention-style semantics):
  Q/K/V remain three separate GEMMs, attention remains two separate
  GEMMs with softmax excluded between them.
- No cross-operation SRAM residency: each GEMM is evaluated independently
  at its own DRAM boundary (e.g. Q projection's output being kept
  resident in SRAM for the attention-score GEMM is not modeled) — block
  DRAM traffic may overcount opportunities a fused/resident implementation
  would avoid.
- No KV cache, no autoregressive decoding loop, no cross-layer pipeline
  or tensor/data parallelism, no multi-chip modeling.
- No power/energy model — a tie-break preferring fewer PEs is not a
  claim of lower energy or cost.
- Inherits every limitation of the underlying GEMM/tiling/PE/timing
  models (idealized bandwidth-only DRAM timing, no cycle accuracy, no
  real-hardware validation).
