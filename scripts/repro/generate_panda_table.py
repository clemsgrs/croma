r"""Render the two-panel PCaBiop slide-level table (``supp/table_panda_isup.tex``) from both runs.

The tile tables are one-run-per-float and are handled by ``generate_results_table.py``. This
table is different: it puts cancer detection and ISUP grading side by side under one caption.
It therefore needs both frames at once and gets its own generator rather than a column-subset
flag on the other one.

The surrounding section (``paper/sections/supp/panda.tex``) is authorial prose that cites
macros; only this numeric table is derived, so the prose ``\input``s the file this generator
owns -- the same body/prose seam ``generate_model_tables.py`` uses. It is here because the
bodies had rotted past the point of being describable: panel (a) once listed every model at
``k=9`` and panel (b) put MOOZY at ``k=13``, neither matching any run still on disk (the live
k* runs give k in {1, 3, 9, 11}), and the caption announced a "shared operating point $k{=}9$"
for a protocol that has none. See ADR-0010.

Usage:
  python scripts/repro/generate_panda_table.py
  python scripts/repro/generate_panda_table.py --check
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M
from generate_results_table import (
    SIGNED,
    Column,
    _fmt,
    _join,
    column_spec,
    groups,
    header_lines,
    load_frame,
    primary,
    row_cells,
    total_columns,
)
from paper_manifest import by_prefix

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper/sections/supp/table_panda_isup.tex"

#: The tile tables' three column groups, with the per-model k* leading the pooled group: the
#: slide panel runs at k-star, so there is no one k to name in the group title.
KSTAR = Column("k", r"$k^\star$", 0)
GROUPS = groups(r"k^\star", pooled_lead=(KSTAR,))
PANEL_A = [(c.key, c.header, c.decimals, c.percent) for g in GROUPS for c in g.columns]
PANEL_B = list(PANEL_A)

NO_BOLD = ("k", "confounder_knn_bacc", "delta")
MINIMIZE = ("croma_frac_neg",)


def _panel(df: pd.DataFrame) -> list[str]:
    df = df.sort_values("croma", ascending=False)
    best = {
        c: (df[c].min() if c in MINIMIZE else df[c].max())
        for c, _, _, _ in PANEL_A
        if c not in NO_BOLD
    }
    lines = []
    for _, r in df.iterrows():

        def cell(column: Column) -> str:
            col = column.key
            is_best = col not in NO_BOLD and abs(r[col] - best[col]) < 1e-9
            return _fmt(r[col], column.decimals, column.percent, is_best, col in SIGNED)

        lines.append(_join([r["model"], *row_cells(GROUPS, cell)]) + r" \\")
    return lines


_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _spell(n: int) -> str:
    """Small counts read as words in caption prose, as they do everywhere else in the paper."""
    return _WORDS[n] if n < len(_WORDS) else str(n)


def _caption(a: pd.DataFrame, b: pd.DataFrame) -> str:
    """Describe the two panels without narrating observed model differences."""
    m = int(CROMA_HEADLINE_M)
    n = len(set(a["model"]) & set(b["model"]))
    return (
        rf"\textbf{{Robustness of slide-level representations on PCaBiop and PCaBiop-ISUP.}} "
        rf"The {_spell(n)} slide-level encoders are ordered within each panel by median "
        rf"\code{{CRoMa}} ($m{{=}}{m}$). Column groups, arrows and bold follow "
        rf"Table~\ref{{{primary().label}}}; the confounder is the data provider. Unlike the "
        r"tile tables, each encoder is evaluated at a model-specific, biologically selected "
        r"$k^\star$, listed in the pooled group: Biology and the fixed-$k$ pooled scores are "
        r"reported at $k^\star$, and Centre is the maximum over the evaluated $k$ grid. "
        r"\textbf{a,} Binary "
        r"cancer detection on $1{,}000$ slides. \textbf{b,} Six-class ISUP grading over "
        r"$3{,}000$ slides."
    )


def _tabular(title: str, df: pd.DataFrame) -> list[str]:
    return [
        r"\begin{tabular}{" + column_spec(GROUPS) + "}",
        r"\hline",
        rf"\multicolumn{{{total_columns(GROUPS)}}}{{l}}{{\emph{{{title}}}}} \\",
        r"\hline",
        *header_lines(GROUPS),
        r"\hline",
        *_panel(df),
        r"\hline",
        r"\end{tabular}",
    ]


def build_float(root: Path = REPO) -> str:
    a = load_frame(root / by_prefix("Panda").metrics_rel)
    b = load_frame(root / by_prefix("PandaIsup").metrics_rel)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        *_tabular("(a) Cancer detection", a),
        "",
        r"\vspace{5pt}",
        *_tabular("(b) ISUP grading", b),
        rf"\caption{{{_caption(a, b)}}}",
        # Both labels attach to this one float on purpose: the results section cites the
        # detection panel and the supplement cites the grading panel.
        r"\label{tab:main-results-panda}",
        r"\label{tab:main-results-panda-isup}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build(root: Path = REPO) -> str:
    return build_float(root) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    tex = build()
    if args.check:
        if not OUT.exists() or OUT.read_text() != tex:
            print(f"STALE  {OUT.relative_to(REPO)}", file=sys.stderr)
            return 1
        print(f"ok     {OUT.relative_to(REPO)}", file=sys.stderr)
        return 0
    if not OUT.parent.exists():
        print("skip   paper/ absent", file=sys.stderr)
        return 0
    OUT.write_text(tex)
    print(f"wrote  {OUT.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
