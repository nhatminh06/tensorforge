#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def duration(value):
    if value < 1e-6:
        return f"{value * 1e9:.2f} ns"
    if value < 1e-3:
        return f"{value * 1e6:.2f} us"
    return f"{value * 1e3:.2f} ms"


def main():
    parser = argparse.ArgumentParser(description="Summarize a TensorForge canonical evidence bundle.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    m = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    contrast = json.loads((args.bundle / "contrast-validation.json").read_text(encoding="utf-8"))
    v = m["validation"]
    print("TensorForge canonical evidence\n")
    print("DEVICE")
    print(m["device"]["name"])
    print(f"PyTorch {m['runtime']['pytorch_version']} / CUDA {m['runtime']['cuda_runtime_version']} / {m['device']['dtype'].upper()}\n")
    print("CALIBRATION")
    print(f"compute          {m['calibration']['effective_compute_tflops']:.3f} TFLOP/s")
    print(f"memory           {m['calibration']['effective_memory_gbps']:.3f} GB/s\n")
    print(f"PRIMARY WORKLOAD\n{m['primary_workload']}")
    print(f"prediction       {duration(v['predicted_seconds'])}")
    print(f"measured p50     {duration(v['measured_p50_seconds'])}")
    print(f"measured p95     {duration(v['measured_p95_seconds']) if v['measured_p95_seconds'] else 'n/a'}")
    print(f"APE              {v['ape_percent']:.1f}%\n")
    print(f"CONTRAST\n{m['contrast_workload']}")
    print(f"prediction       {duration(contrast['prediction']['latency_seconds'])}")
    print(f"measured p50     {duration(contrast['measurement']['p50_latency_seconds'])}")
    print(f"APE              {contrast['error']['absolute_percentage_error']:.1f}%\n")
    print(f"REGRESSION\n{m['regression']['status']}\n")
    print(f"TELEMETRY\n{m['telemetry']['status']} ({m['telemetry']['sample_count']} samples)\n")
    print(f"IMPACT\n{m['impact']['status']}")


if __name__ == "__main__":
    main()
