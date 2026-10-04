#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


REQUIRED = (
    "manifest.json", "calibration.json", "primary-core.json", "primary-benchmark.json",
    "primary-validation.json", "contrast-core.json", "contrast-benchmark.json",
    "contrast-validation.json", "baseline-benchmark.json", "candidate-benchmark.json",
    "regression-policy.json", "regression-result.json", "regression-report.md",
    "baseline-telemetry-summary.json", "telemetry-summary.json", "impact-manifest.json",
    "impact-policy.json", "impact-result.json", "impact-report.md",
)


def main():
    parser = argparse.ArgumentParser(description="Validate canonical artifacts and write SHA256SUMS.")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.bundle / name).is_file()]
    if missing:
        raise SystemExit("missing artifacts: " + ", ".join(missing))
    for path in args.bundle.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    forbidden = (str(Path.home()), "hostname", "username", "token")
    for path in args.bundle.iterdir():
        if path.is_file() and path.name != "SHA256SUMS":
            text = path.read_text(encoding="utf-8")
            if any(value and value in text for value in forbidden):
                raise SystemExit(f"unsafe environment metadata found in {path.name}")
    lines = []
    for path in sorted(p for p in args.bundle.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (args.bundle / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validated {len(lines)} artifacts in {args.bundle}")


if __name__ == "__main__":
    main()
