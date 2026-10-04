# TensorForge portfolio recording

The final recording is a factual walkthrough of the committed canonical
evidence, not a live benchmark. Target **3:30** and keep the finished video
between three and four minutes.

## Preparation

- Record at 1920×1080 with browser zoom at 90–100%.
- Use a terminal font of at least 18 px if the optional terminal moment is included.
- Close unrelated tabs and GPU-heavy applications, hide notifications, connect
  laptop power, and verify that no secrets or avoidable personal paths appear.
- Open the deployed [project showcase](https://nhatminh06.github.io/tensorforge/)
  and allow all evidence JSON to load before recording.
- Do not recapture the canonical benchmark. Cuts may remove waiting, but do not
  use fake typing, fake execution, generated narration, or artificial numbers.

## Shot list and narration

| Time | View | Spoken point / caption |
|---|---|---|
| 0:00–0:20 | Hero and primary metrics | “TensorForge is analytical accelerator modeling validated against real CUDA execution.” Show 183.74 µs predicted, 195.16 µs measured, and 5.86% APE. |
| 0:20–0:45 | Architecture | “Core performs analytical modeling. Ops adds measurement and evidence. The dependency is `tensorforge_ops → tensorforge`; Core remains independent of PyTorch, MLflow, and NVML.” |
| 0:45–1:10 | Prediction anatomy | Follow workload → FLOPs and traffic → mapping and tiling → calibration → max(compute, memory) → predicted latency. Call `compute-bound` a predicted bottleneck only. |
| 1:10–1:40 | Calibration | Show 11.69 TFLOP/s and 180.75 GB/s. “These are measured empirical ceilings from this RTX 3050, not theoretical peaks.” |
| 1:40–2:10 | `gemm_large_square` | Show 183.74 µs predicted, 195.16 µs p50, 225.12 µs p95, and 5.86% APE. “On this capture, the calibrated lower bound tracks the large GEMM closely.” |
| 2:10–2:35 | `gemm_tiny` | Show 0.544 µs predicted, 12.28 µs measured, 22.58× ratio, and 95.57% APE. “Small workloads expose what the analytical model does not capture.” Do not claim a proven cause. |
| 2:35–2:55 | Telemetry | Show GPU utilization, switch briefly to power, and point out `SwPowerCap`. “NVML telemetry is diagnostic context from a separate execution phase.” Do not infer power limitation. |
| 2:55–3:15 | Evidence boundaries | Show regression `NOT EVALUATED`, right-sizing `OMITTED`, and impact `NOT EVALUATED`. “TensorForge refuses to produce conclusions when the required evidence does not exist.” |
| 3:15–3:30 | Provenance | Show RTX 3050 Laptop GPU, 13 SHA-256 verified artifacts, capture source, evidence commit, and repository links. End without an animated outro. |

An optional terminal shot may run:

```bash
python tools/demo/summarize.py docs/evidence/canonical
```

This displays the real committed evidence. Do not rerun the GPU benchmark for
spectacle; live timing variability adds no evidence.

## Review before publishing

Watch the finished video from beginning to end and verify that:

- every number is readable and agrees with the canonical evidence;
- no secret, notification, or stale wording is visible;
- the predicted bottleneck is never presented as measured;
- `SwPowerCap` is not presented as proof of power limitation;
- regression is not called pass or fail;
- right-sizing, production capacity, and model quality are not claimed;
- the final duration is between three and four minutes.

Host the video on a stable platform such as YouTube or a GitHub release asset.
Do not commit the large video file to Git history. Add the final `Demo video`
URL to the README and showcase only after the hosted asset exists.
