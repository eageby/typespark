"""Type check the examples in tests/typecheck with pyright and mypy.

Each example marks its expected errors with ``# type: ignore`` (both checkers)
or ``# pyright: ignore[rule]`` (pyright only). Both checkers report unnecessary
ignores in those files, so a case that stops erroring fails here.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TYPECHECK_DIR = Path(__file__).parent / "typecheck"
EXAMPLES = sorted(str(p) for p in TYPECHECK_DIR.glob("*.py"))


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_typecheck_examples_pyright() -> None:
    result = subprocess.run(
        ["pyright", "--pythonpath", sys.executable, *EXAMPLES],
        capture_output=True,
        check=False,  # the assert below reports pyright's output
        text=True,
    )
    assert result.returncode == 0, result.stdout


def test_typecheck_examples_mypy() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", *EXAMPLES],
        capture_output=True,
        check=False,  # the assert below reports mypy's output
        text=True,
    )
    assert result.returncode == 0, result.stdout
