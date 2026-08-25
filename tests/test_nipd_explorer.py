"""Browser-independent behavior contract for the nIPD evidence explorer."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for docs JS tests")
def test_nipd_explorer_interactions() -> None:
    subprocess.run(
        ["node", "--test", "tests/js/test_nipd_explorer.cjs"],
        cwd=ROOT,
        check=True,
    )
