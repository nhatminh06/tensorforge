# Milestone 6 model: analytical compute/memory execution timing

## What this models

This combines two already-modeled quantities that previously had no time
units:

```
PeMappingResult.compute_cycles   (Milestone 2 — PE-array mapping)
TilingResult.total_dram_bytes    (Milestone 4/5 — schedule-specific tiling traffic)
```

into an analytical estimate of how long idealized compute and idealized
DRAM transfer take, and which one bounds the other.

## Clock frequency

`TimingConfig(clock_hz)` — a new, separate timing parameter, validated
finite and `> 0`. It is **not** folded into `PeArray`, which continues to
model only rows/columns geometry; clock frequency is a distinct
hardware-execution parameter. CLI input is `--clock-ghz` (decimal GHz,
`1 GHz = 1e9 Hz`).

## Compute time

```
compute_time_seconds = compute_cycles / clock_hz
```

Units: `cycles / (cycles/second) = seconds`. This is an idealized PE
compute time. It ignores pipeline fill/drain, memory stalls, NoC,
synchronization, and control overhead — it is not measured latency.

## Memory time

```
memory_time_seconds = total_dram_bytes / memory_bandwidth_bytes_per_second
```

**`total_dram_bytes` is the schedule-specific traffic from
`TilingResult`** (already reflecting tile shape, edge tiles, and the
chosen residency schedule) — **not** the Milestone-1 idealized
`Gemm.dram_bytes`. This is the first milestone where schedule choice
directly changes an execution-time estimate. Memory time here is a
bandwidth-only transfer time: it ignores DRAM latency, bank conflicts,
burst inefficiency, controller overhead, and queueing.

## Two analytical bounds

```
serialized_time_seconds      = compute_time_seconds + memory_time_seconds
perfect_overlap_time_seconds = max(compute_time_seconds, memory_time_seconds)
```

Both are reported — neither is hidden in favor of the other:

- **Serialized**: models zero overlap between compute and DRAM transfer.
  A conservative bound, not a universal claim about worst-case hardware.
- **Perfect-overlap**: models complete overlap, analogous in spirit to
  the Milestone-1 roofline bottleneck model. Not a claim that real
  hardware achieves it.

No partial-overlap model (no "50% overlap," no DMA-overlap coefficient)
is introduced — these two bounds bracket the range; real behavior would
fall somewhere between them depending on an eventual scheduling model.

## Bottleneck classification

```
compute_time_seconds > memory_time_seconds  -> compute-bound
memory_time_seconds  > compute_time_seconds -> memory-bound
approximately equal (relative tolerance)    -> balanced
```

## PE-implied throughput

```
pe_implied_peak_flops_per_second = 2 * pe_count * clock_hz
```

(1 MAC/PE/cycle, 1 MAC = 2 FLOPs — same convention as `gemm.py`/`pe_array.py`.)
This is reported alongside `HardwareConfig.peak_compute_flops_per_second`
(exposed as `configured_roofline_peak_flops_per_second`) purely for
visibility. **The two are not required to match** — they are separate
analytical parameterizations (one derived from PE geometry + clock, the
other a user-configured roofline ceiling). A mismatch is not an error;
reconciling them is a possible future milestone.

## Relationship to existing models

- `compute_roofline()` (Milestone 1) is unchanged and uses the ideal
  baseline `Gemm.dram_bytes` and a user-configured peak FLOP/s — it
  remains a separate, simpler analytical view.
- `map_gemm()` / `PeArray` (Milestone 2) are unchanged; timing only reads
  `PeMappingResult.compute_cycles` and `.pe_count`.
- `analyze_tiling()` / schedule traffic (Milestone 4/5) are unchanged;
  timing only reads `TilingResult.total_dram_bytes`.

## Limitations

- Analytical estimate only — not measured runtime.
- Bandwidth-only DRAM timing: no DRAM latency, no burst/queueing effects.
- No SRAM timing (no SRAM bandwidth or latency).
- Only two bounds modeled: no-overlap (serialized) and perfect-overlap.
  No partial-overlap model.
- No PE pipeline timing (fill/drain), no NoC, no power/energy.
- Not cycle-accurate.
