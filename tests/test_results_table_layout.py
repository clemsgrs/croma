r"""The main-results tables share one three-group, two-level header.

Nine score columns under one flat header did not say which numbers belong together. Every
main-results table -- the four tile tables from ``generate_results_table.py`` and both slide
panels from ``generate_panda_table.py`` -- now groups them as k-NN decodability, fixed-k
pooled scores and per-sample CRoMa, built with ``\multicolumn``/``\cline`` (booktabs is not
loaded) and with a direction arrow on every score column.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts/repro", ROOT / "scripts/bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_panda_table as panda_table  # noqa: E402
import generate_results_table as results_table  # noqa: E402
from paper_manifest import by_prefix  # noqa: E402

LEAVES = (
    r"Model & Biology$\uparrow$ & Centre & & \code{RI}$\uparrow$ & \code{MaRI}$\uparrow$ & "
    r"$\Delta$ & Support$\uparrow$ & & Median$\uparrow$ & $F(0)\downarrow$ & "
    r"$\mcode{LTM}_{10}\uparrow$ \\"
)


def _frame(k: int = 11) -> pd.DataFrame:
    """Two encoders and the control, already in ``load_frame``'s derived shape."""
    return pd.DataFrame(
        {
            "model": ["Good", "Bad", "DINOv2-B"],
            "bio_knn_bacc": [0.98, 0.96, 0.90],
            "confounder_knn_bacc": [0.95, 1.00, 0.91],
            "ri": [0.80, 0.02, 0.50],
            "mari": [0.82, 0.01, 0.52],
            "delta": [0.02, -0.01, 0.02],
            "croma": [0.20, -0.30, 0.05],
            "croma_frac_neg": [0.10, 0.80, 0.40],
            "croma_ltm_alpha": [-0.10, -0.60, -0.30],
            "support": [40.0, 20.0, 68.0],
            "k": [k] * 3,
            "confounder_display_name": ["Medical Center"] * 3,
        }
    )


@pytest.fixture
def table(monkeypatch):
    def render(prefix: str = "Camelyon", k: int = 11) -> str:
        monkeypatch.setattr(results_table, "load_frame", lambda _: _frame(k))
        return results_table.build_table(by_prefix(prefix), Path("unused.csv"))

    return render


def test_tile_tables_group_their_columns_under_a_two_level_header(table):
    tex = table(k=71)

    assert (
        r" & \multicolumn{2}{c}{$k$-NN decodability} & & "
        r"\multicolumn{4}{c}{Fixed-$k$ pooled ($k{=}71$)} & & "
        r"\multicolumn{3}{c}{\code{CRoMa} (per-sample, $m{=}5$)} \\"
    ) in tex
    assert r"\cline{2-3}\cline{5-8}\cline{10-12}" in tex
    assert LEAVES in tex
    assert (
        r"\begin{tabular}{lcc@{\hspace{6pt}}c@{\hspace{6pt}}cccc@{\hspace{6pt}}c@{\hspace{6pt}}ccc}"
        in tex
    )
    assert "bio bacc" not in tex and "conf bacc" not in tex


def test_rows_follow_the_grouped_order_and_keep_the_bold_best_rule(table):
    tex = table()

    # Support now closes the fixed-k group; Centre and Delta are diagnostics, never bold, and
    # the lowest F(0) is the best one. The control's support (68%) cannot win a bold.
    good = next(line for line in tex.splitlines() if line.startswith("Good &"))
    assert good == (
        r"Good & \textbf{0.980} & 0.950 & & \textbf{0.800} & \textbf{0.820} & $+0.020$ & "
        r"\textbf{40.0\%} & & \textbf{0.20} & \textbf{0.100} & \textbf{-0.10} \\"
    )
    bad = next(line for line in tex.splitlines() if line.startswith("Bad &"))
    assert r"& 1.000 &" in bad  # the largest Centre accuracy is not a "best"


def test_every_tile_table_renders_the_grouped_header(table):
    for prefix in ("Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach"):
        assert LEAVES in table(prefix)


def test_the_primary_caption_walks_the_three_groups_in_order(table):
    caption = next(line for line in table("Camelyon").splitlines() if line.startswith(r"\caption"))

    positions = [
        caption.index(r"\emph{$k$-NN decodability}"),
        caption.index(r"\emph{Fixed-$k$ pooled}"),
        caption.index(r"\emph{Per-sample \code{CRoMa}}"),
    ]
    assert positions == sorted(positions)
    assert r"$\uparrow$" in caption and r"$\downarrow$" in caption


def test_secondary_captions_still_refer_back_to_the_primary(table):
    caption = next(
        line for line in table("TcgaFourByFour").splitlines() if line.startswith(r"\caption")
    )

    assert r"Table~\ref{tab:main-results}" in caption
    assert r"$k$-NN decodability" not in caption


def test_both_slide_panels_render_the_grouped_header(monkeypatch):
    frame = _frame().iloc[:2].assign(k=[3, 9])
    monkeypatch.setattr(panda_table, "load_frame", lambda _: frame.copy())

    tex = panda_table.build_float(ROOT)

    headers = re.findall(r"^ & \\multicolumn\{2\}\{c\}\{\$k\$-NN decodability\}.*$", tex, re.M)
    assert len(headers) == 2
    assert all(r"\multicolumn{5}{c}{Fixed-$k$ pooled ($k{=}k^\star$)}" in h for h in headers)
    assert tex.count(r"Model & Biology$\uparrow$ & Centre & & $k^\star$ & \code{RI}$\uparrow$") == 2
