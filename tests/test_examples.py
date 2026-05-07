"""Verify all example scripts run without errors."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _get_example_scripts() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", _get_example_scripts(), ids=lambda p: p.name)
def test_example_runs(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"{script.name} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
