"""TensorForge Core must stay a frozen, runtime-dependency-free package.

These tests are a textual audit (not an import-time check) so they work
identically whether or not torch/mlflow/onnxruntime happen to be
installed in the current environment.
"""

import pathlib
import re

_CORE_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "tensorforge"
_FORBIDDEN = ("torch", "onnxruntime", "mlflow")


def _core_python_files():
    return sorted(_CORE_SRC.rglob("*.py"))


def test_core_source_files_exist():
    assert _core_python_files(), "expected tensorforge Core source files to exist"


def test_core_never_imports_torch_onnxruntime_or_mlflow():
    pattern = re.compile(r"^\s*(import|from)\s+(" + "|".join(_FORBIDDEN) + r")\b")
    offenders = []
    for path in _core_python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.match(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")
    assert not offenders, "Core must not depend on Ops runtimes:\n" + "\n".join(offenders)
