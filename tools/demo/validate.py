#!/usr/bin/env python3
"""Validate a canonical bundle and create/verify its SHA-256 index."""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


MEASUREMENT_REQUIRED = {
    "manifest.json", "README.md", "calibration.json", "primary-core.json",
    "primary-benchmark.json", "candidate-benchmark.json", "primary-validation.json",
    "contrast-core.json", "contrast-benchmark.json", "contrast-validation.json",
    "telemetry-summary.json", "telemetry-trace.json", "telemetry-capabilities.json",
}
REGRESSION_REQUIRED = {
    "baseline-benchmark.json", "baseline-telemetry-summary.json",
    "regression-policy.json", "regression-result.json", "regression-report.md",
    "impact-manifest.json", "impact-policy.json", "impact-result.json",
    "impact-report.md",
}


def fail(message: str) -> None:
    raise SystemExit(f"validate: {message}")


def load_json(path: Path):
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: fail(f"non-finite JSON value {value} in {path.name}"),
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON in {path.name}: {exc}")


def assert_finite(value, location="root"):
    if isinstance(value, float) and not math.isfinite(value):
        fail(f"non-finite number at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, f"{location}[{index}]")


def check_sha256(bundle: Path, files: list[Path]) -> None:
    expected = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    checksum_path = bundle / "SHA256SUMS"
    if checksum_path.exists():
        actual = {}
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
                fail("malformed SHA256SUMS")
            actual[name] = digest
        if actual != expected:
            fail("SHA256SUMS does not match bundle contents")
    checksum_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(expected.items())),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle
    if not bundle.is_dir():
        fail(f"bundle directory not found: {bundle}")
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        fail("missing artifacts: manifest.json")
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 2:
        fail(f"manifest schema version must be 2, got {manifest.get('schema_version')!r}")

    regression_status = manifest.get("regression", {}).get("status")
    impact_status = manifest.get("impact", {}).get("status")
    measurement_only = regression_status == "not_evaluated"
    if measurement_only:
        if not manifest["regression"].get("reason"):
            fail("not_evaluated regression requires a reason")
        if impact_status != "not_evaluated" or not manifest["impact"].get("reason"):
            fail("measurement-only bundle requires impact status not_evaluated with a reason")
        forbidden = REGRESSION_REQUIRED & {path.name for path in bundle.iterdir() if path.is_file()}
        if forbidden:
            fail("measurement-only bundle contains regression/impact artifacts: " + ", ".join(sorted(forbidden)))
        required = MEASUREMENT_REQUIRED
    elif regression_status in {"PASS", "FAIL", "ERROR"}:
        if impact_status == "not_evaluated":
            fail("evaluated regression cannot have impact status not_evaluated")
        required = MEASUREMENT_REQUIRED | REGRESSION_REQUIRED
    else:
        fail(f"unsupported regression status: {regression_status!r}")

    present = {path.name for path in bundle.iterdir() if path.is_file()}
    missing = sorted(required - present)
    if missing:
        fail("missing artifacts: " + ", ".join(missing))
    if manifest.get("right_sizing", {}).get("status") != "omitted":
        fail("canonical single-device bundle must mark right_sizing omitted")

    documents = {}
    for path in bundle.glob("*.json"):
        documents[path.name] = load_json(path)
        assert_finite(documents[path.name], path.name)
    expected_schemas = {
        "calibration.json": ("calibration_schema_version", 1),
        "primary-core.json": ("schema_version", 1),
        "contrast-core.json": ("schema_version", 1),
        "primary-benchmark.json": ("benchmark_schema_version", 1),
        "candidate-benchmark.json": ("benchmark_schema_version", 1),
        "contrast-benchmark.json": ("benchmark_schema_version", 1),
        "primary-validation.json": ("validation_schema_version", 1),
        "contrast-validation.json": ("validation_schema_version", 1),
        "telemetry-summary.json": ("telemetry_schema_version", 1),
        "telemetry-trace.json": ("telemetry_schema_version", 1),
    }
    for name, (field, version) in expected_schemas.items():
        if documents[name].get(field) != version:
            fail(f"{name} has unexpected {field}")

    if documents["primary-benchmark.json"] != documents["candidate-benchmark.json"]:
        fail("primary and candidate benchmark artifacts must describe the same measurement")
    for name in ("primary-validation.json", "contrast-validation.json"):
        validation = documents[name]
        if "prediction" not in validation or "measurement" not in validation or "error" not in validation:
            fail(f"{name} does not keep prediction, measurement, and error fields separate")
        if validation["prediction"].get("bottleneck") not in {"compute-bound", "memory-bound", "balanced"}:
            fail(f"{name} has an invalid predicted bottleneck")

    telemetry = documents["telemetry-summary.json"]
    for field in (
        "sm_activity_percent", "sm_occupancy_percent", "tensor_activity_percent",
        "dram_bandwidth_utilization_percent",
    ):
        metric = telemetry[field]
        if metric["sample_count"] == 0 and any(metric[key] is not None for key in ("mean", "min", "max")):
            fail(f"unsupported telemetry metric {field} must remain null")

    sha = manifest.get("tensorforge_commit")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        fail("manifest tensorforge_commit must be a full Git SHA")
    forbidden_keys = re.compile(r'(?i)"(?:hostname|username|password|secret|access_token|auth_token)"\s*:')
    for path in bundle.iterdir():
        if path.is_file() and path.name != "SHA256SUMS":
            text = path.read_text(encoding="utf-8")
            if str(Path.home()) in text or forbidden_keys.search(text):
                fail(f"unsafe environment metadata found in {path.name}")

    files = sorted(path for path in bundle.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    check_sha256(bundle, files)
    print(f"Validated {len(files)} artifacts in {bundle}")


if __name__ == "__main__":
    main()
