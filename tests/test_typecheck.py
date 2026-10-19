"""Run pyright over the README examples so they can't drift from the API."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TYPECHECK_DIR = Path(__file__).parent / "typecheck"


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_readme_examples_typecheck() -> None:
    result = subprocess.run(
        [
            "pyright",
            "--pythonpath",
            sys.executable,
            *(str(p) for p in TYPECHECK_DIR.glob("*.py")),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
