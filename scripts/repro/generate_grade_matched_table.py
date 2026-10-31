"""Supplementary table ``tab:grade-matched``: do grade-matched SO pools lift PCaBiop margins?

One row per slide encoder, ordered by its full-pool median CRoMa: median CRoMa and F(0) of
the 500 cancer anchors under the full, size-matched random and grade-matched SO pools, the
grade effect (matched minus random median), its one-sided empirical p, and the share of
non-positive anchors that grade matching lifts above zero. The numbers come from
``_grade_matched.load``, which fails when the study is missing or stale.

Run: python scripts/repro/generate_grade_matched_table.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from _grade_matched import REPO, GradeMatched, load
from _paper_tables import CROMA_HEADLINE_M, num_math, pct_round

OUT = Path("paper/sections/supp/grade_matched.tex")
DECIMALS = 3


def _random_cell(value: float, lo: float, hi: float) -> str:
    bracket = f"[{num_math(lo, DECIMALS)[1:-1]}, {num_math(hi, DECIMALS)[1:-1]}]"
    return rf"{num_math(value, DECIMALS)}\,{{\scriptsize${bracket}$}}"


def _p_cell(p: float, n_replicates: int) -> str:
    """Empirical p over ``n_replicates``; zero is reported as below one replicate's share."""
    floor = 1.0 / n_replicates
    return rf"$<{floor:g}$" if p < floor else num_math(p, DECIMALS)


def _row(name: str, x: pd.Series, n_replicates: int) -> str:
    lifted = x["sign_change_share"]
    cells = [
        name,
        num_math(x["full_median"], DECIMALS),
        _random_cell(x["random_median"], x["random_median_lo"], x["random_median_hi"]),
        num_math(x["matched_median"], DECIMALS),
        num_math(x["full_f0"], DECIMALS),
        _random_cell(x["random_f0"], x["random_f0_lo"], x["random_f0_hi"]),
        num_math(x["matched_f0"], DECIMALS),
        num_math(x["delta_grade"], DECIMALS),
        _p_cell(x["p_one_sided"], n_replicates),
        "--" if math.isnan(lifted) else pct_round(lifted),
    ]
    return " & ".join(cells) + r" \\"


def build(study: GradeMatched) -> str:
    meta = study.meta
    pooled = study.pooled.sort_values("full_median", ascending=False)
    n_replicates = int(meta["n_replicates"])
    per_grade = int(meta["pool_size"])
    full_pool = {p: sum(by_grade.values()) for p, by_grade in meta["cancer_slides_per_grade"].items()}
    (full_size,) = set(full_pool.values())
    m = int(meta["headline_m"])
    if m != int(CROMA_HEADLINE_M):
        raise ValueError(f"the study read CRoMa at m={m}, the paper reports m={CROMA_HEADLINE_M}")

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccccccccc}",
        r"\hline",
        r" & \multicolumn{3}{c}{median \code{CRoMa}} & \multicolumn{3}{c}{$F(0)$} & & & \\",
        r"\cline{2-4}\cline{5-7}",
        r"Model & full & random & matched & full & random & matched"
        r" & $\Delta_{\mathrm{grade}}$ & $p$ & lifted \\",
        r"\hline",
    ]
    lines += [_row(name, x, n_replicates) for name, x in pooled.iterrows()]
    caption = (
        r"\caption{\textbf{Grade-matched SO pools on PCaBiop.} "
        rf"Median per-sample $\mcode{{CRoMa}}(m{{=}}{m})$ and $F(0)$ over the "
        rf"${int(meta['n_cancer_anchors'])}$ cancer anchors of both centres, with each anchor's "
        r"same-label, other-centre (\textbf{SO}) candidates restricted to one of three pools: "
        rf"\emph{{full}}, all ${full_size}$ cancer slides at the other centre (the benchmark "
        r"run's own pool, reproduced exactly); \emph{matched}, those of the anchor's ISUP grade; "
        r"and \emph{random}, the same number drawn uniformly from the full pool, one draw "
        r"shared by the anchors of each centre and grade. The cohort is grade-balanced, with "
        rf"${per_grade}$ cancer slides per ISUP grade per centre, so matched and random pools both "
        rf"hold ${per_grade}$ slides. Only the SO side is matched: every anchor keeps its "
        r"opposite-label, same-centre (\textbf{OS}) candidates, the benign slides of its own "
        r"centre. The random columns give the median over "
        rf"${n_replicates}$ random replicates (seed ${int(meta['seed'])}$) of each replicate's "
        r"median or $F(0)$, with the $2.5$--$97.5$ percentile range in brackets. "
        r"$\Delta_{\mathrm{grade}}$ is the matched median minus the random one; $p$ is the share "
        r"of random replicates whose median is at least the matched median (one-sided); "
        r"\emph{lifted} is the share of anchors with $\mcode{CRoMa}\leq0$ under the "
        r"replicate-averaged random pool that the matched pool moves above $0$. "
        r"Rows are ordered by the full-pool median.}"
    )
    lines += [
        r"\hline",
        r"\end{tabular}",
        caption,
        r"\label{tab:grade-matched}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="Repo root holding output/.")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.write_text(build(load(args.root)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
