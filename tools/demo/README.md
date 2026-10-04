# Canonical evidence capture

This directory composes the existing TensorForge Core and Ops APIs into one
file-based, reproducible GPU evidence story. It adds no modeling, benchmark,
calibration, telemetry, regression, sizing, or impact logic.

The regression comparison requires a real `gemm_large_square` baseline from
the revision being compared and its separately captured telemetry summary.
That requirement prevents the demo from presenting two unchanged-code repeats
as a model/configuration change.

```bash
python -m pip install -e '.[ops,benchmark,telemetry]'
./tools/demo/capture.sh docs/evidence/canonical \
  --baseline-benchmark /path/to/baseline-benchmark.json \
  --baseline-telemetry /path/to/baseline-telemetry-summary.json
```

The canonical capture does not run an MLflow service. The `ops` extra is
currently required because the CLI/package dependency graph imports the
tracking module even though this file-based workflow does not contact MLflow.

When no defensible performance-changing historical baseline exists, omit both
baseline options. The bundle will record regression and impact as
`not_evaluated`. Supplying exactly one baseline option is an error; supplying
both preserves the full measured-regression workflow.

The capture requires CUDA and NVML and fails rather than falling back to CPU.
Latency measurement and telemetry collection remain separate phases. MLflow is
not required. The output directory must not already exist, so a failed or
partial run cannot silently mix with prior evidence.

Right-sizing is intentionally omitted: one measured physical GPU does not form
an honest multi-hardware catalog. Run `summarize.py` for screen-readable output
and `validate.py` to revalidate JSON/sanitization and refresh `SHA256SUMS`.

The performance workflow's official GitHub Actions remain consistently pinned
to major version tags (`actions/checkout@v4`, `actions/setup-python@v5`, and
`actions/upload-artifact@v4`). This phase does not mix that established style
with isolated SHA pins; a repository-wide supply-chain policy change should be
made separately and consistently.
