"""Runs each script under examples/ as __main__ so their own asserts double
as regression coverage -- catches the examples silently drifting out of
sync with the real API (renamed field, changed callback signature, etc.)."""
import runpy
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_runs(script):
    runpy.run_path(str(script), run_name="__main__")
