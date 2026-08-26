# Experiment results

All results below are **modeled/analytical results** from TensorForge
Core, produced by the commands shown. They are not measurements of real
hardware. "Under this modeled workload/configuration..." applies to every
claim in this document.

## Architecture tradeoff study

4 accelerator presets x 4 workload presets. Tile candidates
`(32, 64, 128)` for M/N/K, all three schedules searched, `gemm_large_square`
uses FP16 (see `presets.py` for exact shapes).

Command (repeat per workload/accelerator pair):
```bash
python -m tensorforge --workload-preset <workload> --accelerator-preset <accelerator> \
    --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128
```

| Workload | small | balanced | compute_heavy | bandwidth_heavy | Winner |
|---|---|---|---|---|---|
| gemm_large_square | 4.194ms (compute-bound) | 1.049ms (compute-bound) | **0.357ms** (memory-bound) | 1.049ms (compute-bound) | compute_heavy |
| transformer_medium | 7.471ms | 1.868ms | **0.657ms** | 1.868ms | compute_heavy |
| conv_spatial | 0.903ms | 0.226ms | **0.081ms** | 0.226ms | compute_heavy |
| cnn_like_small | 4.29us | 1.86us | **1.61us** | 1.79us | compute_heavy |

(times are modeled `perfect_overlap_time_seconds`; DRAM traffic and FLOPs
are identical across accelerators for a given workload -- only PE/clock
change compute time.)

**Why `compute_heavy` wins every workload here**: none of the four
workloads reach the memory-bandwidth ceiling at 100 GB/s under these
tile candidates until PE count reaches 64x64 (`gemm_large_square` flips
to memory-bound exactly at `compute_heavy`). Consequently `bandwidth_heavy`
(3x the bandwidth of `balanced`, same PE count) produces **exactly the
same modeled time as `balanced`** for every workload in this table --
extra bandwidth is wasted when the workload is already compute-bound.
This is not a claim that `compute_heavy` is universally best: for a
workload that is already memory-bound at `balanced`'s PE count (see the
compute-vs-memory section below), more bandwidth -- not more PEs -- is
what actually helps.

## Compute-bound vs. memory-bound evidence

Workload: `Gemm(1024, 1024, 1024, FP16)` (`gemm_large_square` shape),
SRAM=512 KiB, clock=1 GHz, tiles `(32,64,128)`.

```bash
python3 -c "... see docs/architecture.md data path; PE array swept 8x8..64x64, bandwidth fixed at 100 GB/s ..."
```

| PE array | compute time | memory time | perfect-overlap | classification |
|---|---|---|---|---|
| 8x8 | 16.777ms | 0.357ms | 16.777ms | compute-bound |
| 16x16 | 4.194ms | 0.357ms | 4.194ms | compute-bound |
| 32x32 | 1.049ms | 0.357ms | 1.049ms | compute-bound |
| 64x64 | 0.262ms | 0.357ms | **0.357ms** | **memory-bound** |

At 64x64, compute time (0.262ms) drops below memory time (0.357ms,
constant across all PE sizes since DRAM traffic doesn't depend on PE
shape) -- the workload crosses from compute-bound to memory-bound, and
`perfect_overlap_time` stops improving even though compute keeps getting
faster.

## PE sensitivity study (diminishing returns)

Same GEMM/SRAM/bandwidth as above; PE swept 8x8/16x16/32x32/64x64
(matches the table above exactly): compute time halves-then-quarters
predictably (16.777 -> 4.194 -> 1.049 -> 0.262 ms) while memory time
stays fixed at 0.357ms -- once PE count crosses the point where compute
time < memory time (between 32x32 and 64x64 here), further PE growth
stops reducing `perfect_overlap_time`.

## Mapping matters (same hardware, different schedule)

Workload: `Gemm(2048, 256, 512, FP16)` (`gemm_tall` shape), PE=64x64,
SRAM=512 KiB, clock=1 GHz, bandwidth=20 GB/s (deliberately memory-bound),
tiles `tile_m in {128,256}, tile_n in {64,128}, tile_k in {128,256}`.

| Schedule | tile | DRAM bytes | effective AI | perfect-overlap |
|---|---|---|---|---|
| c-resident | 256x128x128 | 7,340,032 | 73.14 FLOP/B | 367.00us |
| a-resident | 256x64x256 | 7,340,032 | 73.14 FLOP/B | 367.00us |
| b-resident | 128x128x256 | 7,602,176 | 70.62 FLOP/B | **380.11us** |

Under this memory-bound configuration, b-resident's extra 262,144 bytes
of DRAM traffic (from repeated partial-sum reads/writes, see
`docs/schedules.md`) directly costs 13.11us of additional modeled
execution time versus c-resident/a-resident, on **identical hardware**.

## PE-shape matters (same PE count, different shape)

Workload: `Gemm(17, 64, 128, FP16)` (deliberately non-divisible
dimensions, as in Milestone 2's original example):

| PE array | PE count | waves | spatial utilization | compute cycles |
|---|---|---|---|---|
| 4x16 | 64 | 20 | 85.00% | 2,560 |
| 8x8 | 64 | 24 | 70.83% | 3,072 |
| 16x4 | 64 | 32 | 53.12% | 4,096 |

All three arrays have identical PE count (64), but spatial utilization
(and therefore compute cycles) differ by more than 1.5x between the best
(4x16) and worst (16x4) shape for this workload's M/N dimensions.

## SRAM sensitivity study

Workload: `Gemm(1024, 1024, 1024, FP16)`, PE=32x32, clock=1 GHz,
bandwidth=100 GB/s, tile candidates `tile_m/n in {128, 256}, tile_k in
{16, 64, 256}` (chosen so a 256x256 tile only becomes feasible with more
SRAM).

| SRAM | feasible candidates | selected tile | DRAM bytes | serialized time |
|---|---|---|---|---|
| 128 KiB | 18/36 | 128x256x16 | 27,262,976 | 1.321ms |
| 256 KiB | 33/36 | 256x256x16 | **18,874,368** | **1.237ms** |
| 512 KiB | 36/36 | 256x256x16 | 18,874,368 | 1.237ms |
| 1024 KiB | 36/36 | 256x256x16 | 18,874,368 | 1.237ms |

Growing SRAM from 128 KiB to 256 KiB enables a larger feasible tile,
cutting modeled DRAM traffic by ~31% and serialized time by ~6.3%.
Beyond 256 KiB, more SRAM changes nothing further **for this candidate
set** -- confirming that more SRAM only helps when it unlocks a better
mapping among the *supplied* candidates, not automatically or
monotonically.

## Bandwidth sensitivity study

Workload: `Gemm(1024, 1024, 1024, FP16)`, PE=64x64 (deliberately
memory-leaning), SRAM=512 KiB, clock=1 GHz, tiles `(32,64,128)`.

| Bandwidth | compute time | memory time | perfect-overlap | classification |
|---|---|---|---|---|
| 50 GB/s | 0.262ms | 0.713ms | 0.713ms | memory-bound |
| 100 GB/s | 0.262ms | 0.357ms | 0.357ms | memory-bound |
| 200 GB/s | 0.262ms | 0.178ms | **0.262ms** | **compute-bound** |
| 400 GB/s | 0.262ms | 0.089ms | 0.262ms | compute-bound |

Memory time scales exactly inversely with bandwidth (halves each
doubling); compute time is unaffected. The crossover from memory-bound to
compute-bound happens between 100 and 200 GB/s for this workload/PE
combination -- beyond that crossover, more bandwidth no longer improves
`perfect_overlap_time`.

## Transformer sequence-length study

`batch=1, d_model=768, num_heads=12, d_ff=3072` (matches
`transformer_medium`/`transformer_long_sequence` dimensions), PE=32x32,
SRAM=512 KiB, clock=1 GHz, bandwidth=100 GB/s, tiles `(32,64,128)`.

```bash
python -m tensorforge --transformer-block --batch-size 1 --seq-len <S> \
    --d-model 768 --num-heads 12 --d-ff 3072 --dtype fp16 \
    --peak-tflops 2.048 --bandwidth-gbps 100 --sram-kib 512 --clock-ghz 1 \
    --pe-rows 32 --pe-cols 32 --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128
```

| S | total GEMM FLOPs | attention fraction | block time | largest contributor |
|---|---|---|---|---|
| 128 | 1,862,270,976 | 2.70% | 0.909ms | mlp_up |
| 256 | 3,825,205,248 | 5.26% | 1.868ms | mlp_up |
| 512 | 8,053,063,680 | 10.00% | 3.932ms | mlp_up |
| 1024 | 17,716,740,096 | 18.18% | 8.651ms | mlp_up |

Attention's share of total FLOPs grows from 2.70% to 18.18% as S grows
8x (128->1024), because attention-score/value work scales with S^2 while
projection/MLP work scales linearly with S -- even though `mlp_up`
remains the single largest contributor throughout this range (D=768,
F=3072 keeps MLP work large relative to attention at these lengths).

## Conv2D kernel-size study

`batch=1, Cin=64, H=W=56, Cout=128, FP16`, padding chosen so output stays
56x56 for every kernel size, PE=32x32, SRAM=512 KiB, clock=1 GHz,
bandwidth=100 GB/s.

```bash
python -m tensorforge --conv2d --batch-size 1 --in-channels 64 \
    --input-height 56 --input-width 56 --out-channels 128 \
    --kernel-h <K> --kernel-w <K> --padding-h <(K-1)/2> --padding-w <(K-1)/2> \
    --dtype fp16 --peak-tflops 2.048 --bandwidth-gbps 100 --sram-kib 512 --clock-ghz 1 \
    --pe-rows 32 --pe-cols 32 --tile-m-values 32,64,128 --tile-n-values 32,64,128 --tile-k-values 32,64,128,256,512,576,1600
```

| Kernel | K (Cin*Kh*Kw) | MACs | im2col bytes | expansion | DRAM traffic | perfect-overlap |
|---|---|---|---|---|---|---|
| 1x1 | 64 | 25,690,112 | 401,408 | 1.00x | 1,220,608 | 25.09us |
| 3x3 | 576 | 231,211,008 | 3,612,672 | 9.00x | 4,562,944 | 225.79us |
| 5x5 | 1600 | 642,252,800 | 10,035,200 | 25.00x | 11,247,616 | 627.20us |

Expansion ratio grows exactly as kernel area (`k^2`: 1, 9, 25) since
output resolution -- and therefore the logical input tensor size -- is
held fixed while K and the materialized im2col footprint both scale with
`Cin*Kh*Kw`.

## Cross-workload study (same accelerator, different behavior)

Accelerator: `balanced` (PE=32x32, SRAM=512 KiB, 1 GHz, 100 GB/s).

| Workload | FLOPs | DRAM bytes | perfect-overlap | bottleneck/largest contributor |
|---|---|---|---|---|
| gemm_large_square | 2,147,483,648 | 35,651,584 | 1.049ms | compute-bound |
| transformer_medium | 3,825,205,248 | 65,667,072 | 1.868ms | mlp_up |
| conv_spatial | 462,422,016 | 8,101,888 | 0.226ms | compute-bound |
| cnn_like_small | 1,753,088 | 161,456 | 1.86us | stage1 |

(from the architecture-tradeoff table's `balanced` column.) Each
workload stresses the accelerator differently: the raw GEMM is a single
large dense operation; the Transformer block's cost concentrates in its
widest MLP GEMM even at moderate sequence length; the convolution's cost
is dominated by its materialized-im2col K dimension; the small CNN
workload's cost concentrates in its first (largest spatial resolution)
stage. Raw execution times are **not comparable as a ranking** -- each
workload performs a different amount of total work.
