"""Per-dataset CRoMa/APD composite figures (Figs. croma-apd-<benchmark>).

One 2x2 figure per faithful PathoROB tile benchmark (Camelyon, TCGA in its four-class
form, Tolkach-ESCA). Columns are the two APD regimes -- in-domain (left, test centres
held fixed, isolating shortcut reliance) and out-of-domain (right, unseen centres):

    top row     linear-probe accuracy as the training set's centre<->biology correlation
                (Cramer's V) is walked from balanced (V=0) to fully confounded (V=1);
                one errorbar curve per encoder, 95% t-interval over the seeds. This is the
                raw material the APD scalar reduces -- the absolute accuracy each curve
                departs from, which the ratio-based APD deliberately discards.
    bottom row  pooled CRoMa(m=5) against the APD scalar for that regime; one point per
                encoder, with the project scatter identity (family palette, square panels,
                per-panel Spearman rho, faint least-squares guide line, CRoMa=0 / APD=0
                reference lines).

The two rows share their columns, so within a regime the reader sees both what APD
measures (top) and how CRoMa predicts the number it collapses to (bottom). Both rows share
a y-axis across the two regimes of a benchmark, so ID and OOD are directly comparable;
scales differ between benchmarks, which span different accuracy and drop ranges. PANDA is
excluded (slide-level, only 4 points); prostate is a caveated second-organ extension
reported separately. The natural-image control holds no rank and is dropped throughout.

This replaces the single tall 3x2 ``croma_vs_apd_scatter`` grid: three self-contained
per-dataset floats read better on the page than one near-full-page figure, and they fold
in the per-V curves (previously rendered but never published) as each figure's top row.

The per-V panel primitives (roster, per-split mean/CI, the styled errorbar panel) are the
single source shared with ``apd_per_v_figure.py`` and imported from it here; the scatter
primitives come from the benchmark plotting library.
"""
import sys
from pathlib import Path

from scipy.stats import spearmanr

# The APD loaders live with the rest of the apd study package; reach them across dirs.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "studies" / "apd"))
from loaders import REPO, STUDY_DIR, ranked, read_joined, training_correlations  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plotting as P  # noqa: E402
from croma import plotstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# The per-V top-row panel and its data helpers live in apd_per_v_figure.py, which owns the
# 2x3 standalone rendition of the same curves. Importing them keeps one drawer for the
# accuracy-vs-V curve; a copy here would be a second thing to keep in step with the seeds.
from apd_per_v_figure import _draw_panel, _mean_and_ci, _roster  # noqa: E402

# (dataset key in the joined table, display label). TCGA APD is the four-class form.
BENCHMARKS = [
    ("camelyon", "Camelyon"),
    ("tcga_4x4", r"TCGA (4$\times$4)"),
    ("tolkach", "Tolkach-ESCA"),
]
# One column per regime: the column header (top-panel title), the per-V accuracy key that
# apd_per_v_figure names its regimes by, and the APD-scalar column the scatter plots.
REGIMES = [
    ("In-domain", "id", "apd_id"),
    ("Out-of-domain", "ood", "apd_ood"),
]
#: Where each per-dataset composite is rendered. ``_finalize_figure`` writes
#: ``plots/{pdf,png}/`` beneath this stem; one stem per benchmark, filled in below.
#:
#: Beside the study data it is drawn from, never into ``paper/``: a figure rendered
#: straight into the manuscript tree is a second source of truth that goes stale silently
#: (see ADR-0010 and the note this module used to carry about staging into
#: ``paper/figures/results/``). Copy the PDF across by hand if it earns a float; nothing
#: here stages it, and ``scripts/repro/check_paper_figures.py`` reports any copy that has
#: fallen behind its render.
def _out_stem(dataset: str) -> Path:
    return STUDY_DIR / "plots" / f"croma_apd_{dataset}.png"


def _perv_ylim(dataset: str) -> tuple[float, float]:
    """Shared accuracy limits over both regimes of one benchmark, spanning every curve's
    error caps. Both top panels use it so "OOD sits lower than ID at the same V" is legible
    off the panels rather than hidden by a per-panel autoscale (cf. apd_per_v_figure)."""
    curves = [
        _mean_and_ci(dataset, model, regime_key)
        for _, regime_key, _ in REGIMES
        for model in _roster(dataset)
    ]
    lo = min((mean - ci).min() for mean, ci in curves)
    hi = max((mean + ci).max() for mean, ci in curves)
    pad = max(0.8, (hi - lo) * 0.06)
    return (lo - pad, hi + pad)


def _render_dataset(dset: str, dlabel: str, joined, correlations) -> None:
    df = joined[joined["dataset"] == dset].copy()
    df["apd_id"] = df["apd_id"] * 100.0
    df["apd_ood"] = df["apd_ood"] * 100.0
    rows = df.to_dict("records")

    # The bottom row's legend is collected from its scatter, so a model that draws a per-V
    # curve on top but has no scatter point (or vice versa) would be silently unlabelled or
    # unplotted. The two rosters are the same benchmark's encoders and must agree; fail loud
    # if a data refresh ever desynchronises them.
    top_roster, bottom_roster = set(_roster(dset)), set(df["model"])
    if top_roster != bottom_roster:
        raise ValueError(
            f"{dset}: the per-V curves and the CRoMa-vs-APD scatter disagree on the roster "
            f"(only in per-V: {sorted(top_roster - bottom_roster)}; only in scatter: "
            f"{sorted(bottom_roster - top_roster)}). One artifact is stale."
        )

    # sharex/sharey per row: the top row's two panels share the accuracy axis and the V
    # axis, the bottom row's share the APD axis and the CRoMa axis. Rows stay independent
    # (accuracy vs V above, APD vs CRoMa below), so nothing couples the two scales.
    fig, axes = plt.subplots(
        2, len(REGIMES), figsize=(plotstyle.COL_DOUBLE, 8.0),
        sharex="row", sharey="row",
    )

    top_ylim = _perv_ylim(dset)
    # One APD limit for the bottom row, spanning both regimes; the extra top padding is
    # headroom for the rho annotation (mirrors the retired 3x2 grid's per-row limit).
    row_ys = df[["apd_id", "apd_ood"]].to_numpy()
    ypad = max(0.5, (row_ys.max() - row_ys.min()) * 0.10)
    bot_ylim = (row_ys.min() - ypad, row_ys.max() + ypad * 1.6)

    for j, (col_label, vkey, ykey) in enumerate(REGIMES):
        # Top: accuracy vs Cramer's V. _draw_panel styles the axes, draws every encoder's
        # errorbar curve, and sets the V ticks; we square the box and add the labels.
        ax_top = axes[0][j]
        _draw_panel(ax_top, dset, vkey, ylim=top_ylim, correlations=correlations)
        plotstyle.set_panel_title(ax_top, col_label)
        ax_top.set_xlabel(r"Cramér's $V$")
        # sharey blanks the right column's y ticks; only the left column keeps the label.
        if j == 0:
            ax_top.set_ylabel("LP accuracy [%]")

        # Bottom: CRoMa vs APD. Same scatter drawer, guide line and sign guard as the grid
        # this figure replaces -- a dotted fit that slopes against the annotated rho would
        # be a panel contradicting itself.
        ax_bot = axes[1][j]
        P._draw_model_scatter(
            ax_bot, rows, x_key="croma", y_key=ykey,
            xlabel="CRoMa", ylabel="APD (%)" if j == 0 else "",
            title="", ylim=bot_ylim, hline=0.0, vline=0.0,
        )
        # The scatter drawer forces a square data box (its own identity). In this composite
        # the panel must instead fill its grid cell so the four panels align and the figure
        # stays compact rather than portrait; clear the forced aspect.
        ax_bot.set_box_aspect(None)
        slope = P._draw_trend_line(ax_bot, df["croma"], df[ykey])
        rho, _ = spearmanr(df["croma"], df[ykey])
        if slope is not None and slope * rho < 0:
            raise ValueError(
                f"{dset}/{ykey}: the least-squares trend line slopes {slope:+.2f} but "
                f"Spearman rho is {rho:+.2f}. A guide-to-the-eye line that contradicts the "
                "annotated correlation cannot be drawn -- plot a monotone fit, or drop it."
            )
        ax_bot.text(0.04, 0.96, f"Spearman $\\rho = {rho:.2f}$",
                    transform=ax_bot.transAxes, ha="left", va="top",
                    fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)

    fig.suptitle(dlabel, y=0.985, fontsize=plotstyle.FS_TITLE + 1,
                 weight="bold", color=plotstyle.TEXT_COLOR)

    out = _out_stem(dset)
    P._finalize_figure(fig, out_path=out, legend_axes=[axes[1][0]],
                       top=0.915, bottom=0.20, left=0.095, right=0.985,
                       wspace=0.14, hspace=0.30,
                       legend_y=0.012, legend_ncol=6,
                       legend_fontsize=plotstyle.FS_ANNOT,
                       legend_columnspacing=1.1, legend_handlelength=1.7)
    print(f"wrote {P._pdf_export_path(out)} and {P._png_export_path(out)}")


def main():
    plotstyle.apply_style()
    # The joined table keeps the natural-image control; ranked() drops it, so a panel never
    # annotates a rho no table reports. Same filter, one helper.
    joined = ranked(read_joined())
    correlations = {dset: training_correlations(dset) for dset, _ in BENCHMARKS}
    for dset, dlabel in BENCHMARKS:
        _render_dataset(dset, dlabel, joined, correlations[dset])


if __name__ == "__main__":
    main()
