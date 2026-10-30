"""Ranking stability under cohort subsampling (fig:subsampling-stability).

One column per tile cohort. The top row draws Kendall's tau_K between each replicate's
ranking of the pathology encoders and the same metric's full-cohort ranking against the
share of units kept per class x centre cell (median line, interquartile band; the full
cohort sits at tau_K = 1). The bottom row draws the median over encoders of each encoder's
2.5-97.5 percentile rank range across replicates. RI and MaRI appear twice: solid at the
re-selected shared k, dashed at the full-cohort shared k.

  output/studies/subsampling-stability/plots/pdf/subsampling_stability.pdf

It renders beside the study it is drawn from, never into ``paper/``: copy the PDF to
``paper/figures/subsampling_stability.pdf`` by hand (``check_paper_figures.py`` reports
drift). The numbers come from ``_subsampling`` -- the same summaries the table generator
reads -- so the figure and table cannot disagree.

Run: python scripts/repro/figures/subsampling_stability_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt  # noqa: E402

import plotting as P  # noqa: E402
from plotting import style as plotstyle  # noqa: E402
import _subsampling as S  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

#: Beside the study, under ``plots/{pdf,png}/``.
OUT = S.study.study_root() / "plots" / "subsampling_stability.png"

#: metric -> (colour key, line style, legend label)
SERIES = {
    "ri": ("ri", "-", "RI"),
    "ri_fixed_k": ("ri", "--", "RI, fixed k"),
    "mari": ("mari", "-", "MaRI"),
    "mari_fixed_k": ("mari", "--", "MaRI, fixed k"),
    "croma": ("croma", "-", "median CRoMa"),
    "croma_ltm_alpha": ("ltm", "-", "LTM$_{10}$"),
}


def render(data: S.Study, out: Path = OUT) -> Path:
    """Draw the two-row figure; returns the PDF path."""
    plotstyle.apply_style()
    ranking = S.ranking_summary(data)
    cohorts = list(data.design["cohorts"])
    fractions = sorted({int(f) for f in ranking["fraction"]})
    fig, axes = plt.subplots(
        2,
        len(cohorts),
        figsize=(plotstyle.COL_DOUBLE, 4.4),
        sharex=True,
        squeeze=False,
    )
    for col, cohort in enumerate(cohorts):
        top, bottom = axes[0][col], axes[1][col]
        for metric, (colour_key, style, label) in SERIES.items():
            rows = ranking[(ranking["cohort"] == cohort) & (ranking["metric"] == metric)]
            rows = rows.set_index("fraction").loc[fractions]
            x = rows.index.to_numpy(dtype=float)
            colour = plotstyle.METRIC_COLOR[colour_key]
            top.plot(
                x,
                rows["tau_k_median"],
                style,
                color=colour,
                lw=plotstyle.LW_SERIES,
                label=label,
                marker="o",
                ms=2.5,
            )
            top.fill_between(
                x, rows["tau_k_q25"], rows["tau_k_q75"], color=colour, alpha=0.12, lw=0
            )
            bottom.plot(
                x,
                rows["rank_width_median"],
                style,
                color=colour,
                lw=plotstyle.LW_SERIES,
                label=label,
                marker="o",
                ms=2.5,
            )
        for ax in (top, bottom):
            plotstyle.style_axes(ax)
            ax.set_xticks(fractions)
            ax.set_xticklabels([f"{f}%" for f in fractions])
        plotstyle.set_panel_title(top, by_benchmark(cohort).short_name)
        top.axhline(1.0, color=plotstyle.REFERENCE_LINE_COLOR, lw=plotstyle.LW_REFERENCE, ls=":")
        top.set_ylim(None, 1.03)
        bottom.set_ylim(0, None)
        bottom.set_xlabel("Units kept per class × centre cell")
    axes[0][0].set_ylabel(r"Kendall $\tau_K$ vs full cohort")
    axes[1][0].set_ylabel("Median rank range (2.5-97.5%)")
    P._finalize_figure(
        fig,
        out_path=out,
        legend_axes=[axes[0][0]],
        left=0.085,
        right=0.985,
        top=0.93,
        bottom=0.20,
        legend_y=0.01,
        legend_ncol=len(SERIES),
        legend_fontsize=plotstyle.FS_ANNOT,
        hspace=0.12,
    )
    return P._pdf_export_path(out)


def main() -> None:
    print(f"wrote {render(S.load())}")


if __name__ == "__main__":
    main()
