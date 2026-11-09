"""The paper's F(0) macros find each sample distribution through the layout.

``generate_paper_values`` used to open the absolute ``croma_samples_path`` a run recorded,
and crashed once the checkout that ran it was gone. The sdist leaves this module out
together with ``scripts/repro`` (ADR-0012); the layout and benchmark side is tested in
``test_sample_distribution_paths.py``, which it ships.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "bench", ROOT / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import generate_paper_values as gpv  # noqa: E402
from plotting.style import published_models  # noqa: E402
from sample_distribution_runs import directory, stale_run  # noqa: E402


def test_the_paper_f0_macros_read_each_distribution_from_the_run_not_the_recorded_path(
    tmp_path,
):
    results = directory(tmp_path / "run" / "results")
    stale_run(results)
    df = published_models(pd.read_csv(results / "metrics.csv"))
    assert "Mettle-1 (Pre-release)" in set(df["model"])

    lines, _ = gpv._probe_macros("Tolkach", df, results)

    # F(0) climbs with the probe (0, 1/4, 1/2, 3/4) while pooled CRoMa falls, so the two
    # rank correlations have opposite signs: only the distributions can give F(0) its +1.
    assert r"\newcommand{\TolkachProbeFZeroRho}{$1.00$}" in lines
    assert r"\newcommand{\TolkachProbeCromaRho}{$-1.00$}" in lines
