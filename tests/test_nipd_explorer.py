"""Browser-independent behavior contract for the nIPD evidence explorer."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts" / "bench") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "bench"))

from plotting import style  # noqa: E402


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for docs JS tests")
def test_nipd_explorer_interactions() -> None:
    subprocess.run(
        ["node", "--test", "tests/js/test_nipd_explorer.cjs"],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for docs JS tests")
def test_the_explorer_s_in_plot_labels_agree_with_the_metadata_short_names() -> None:
    # nipd.json carries only the full published name, so the explorer derives the short
    # form by rule. The rule must give every model the short name the figures print.
    import pandas as pd

    metadata = pd.read_csv(ROOT / "scripts" / "bench" / "model_metadata.csv")
    names = [style.published_model_name(m) for m in metadata["model"]]
    script = (
        "const e = require(process.argv[1]);"
        "console.log(JSON.stringify(JSON.parse(process.argv[2]).map(e.shortName)));"
    )
    shown = subprocess.run(
        ["node", "-e", script, str(ROOT / "docs/_static/nipd-explorer.js"), json.dumps(names)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    expected = [style.published_model_name(name, short=True) for name in names]
    assert json.loads(shown) == expected
    assert "Mettle-1" in expected
