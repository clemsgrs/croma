"""Per-dataset CRoMa/nIPD composite figures (Figs. croma-nipd-<benchmark>).

One 2x2 figure per faithful PathoROB tile benchmark (Camelyon, TCGA in its four-class
form, Tolkach-ESCA). Columns are the two probe regimes -- in-domain (left, test centres
held fixed, isolating shortcut reliance) and out-of-domain (right, unseen centres):

    top row     normalized performance change as the training set's centre<->biology
                correlation (Cramer's V) is walked from balanced (V=0) to fully confounded
                (V=1); one errorbar curve per encoder, 95% t-interval over the seeds. This
                is the curve the nIPD scalar integrates.
    bottom row  pooled CRoMa(m=5) against nIPD for that regime; one point per
                encoder, with the project scatter identity (family palette, square panels,
                per-panel Spearman rho, faint least-squares guide line, CRoMa=0 / nIPD=0
                reference lines).

The two rows share their columns, so within a regime the reader sees both what nIPD
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

Normalized trajectories are the canonical publication mode. ``--raw-accuracy`` retains
the absolute probe-accuracy view under an explicit non-canonical filename. PCaBiop is
rendered separately with ``--pcabiop``; with only four whole-slide encoders, its scatter
panels report descriptive rank associations and omit the least-squares guide line.
"""
import argparse
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
from plotting import style as plotstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# The per-V top-row panel and its data helpers live in apd_per_v_figure.py, which owns the
# 2x3 standalone rendition of the same curves. Importing them keeps one drawer for the
# accuracy-vs-V curve; a copy here would be a second thing to keep in step with the seeds.
from apd_per_v_figure import (  # noqa: E402
    SERIES_STYLE,
    _draw_panel,
    _mean_and_ci,
    _normalized_mean_and_ci,
    _roster,
)

# (dataset key in the joined table, display label). TCGA uses the four-class form.
BENCHMARKS = [
    ("camelyon", "Camelyon"),
    ("tcga_4x4", r"TCGA (4$\times$4)"),
    ("tolkach", "Tolkach-ESCA"),
]
PCABIOP_BENCHMARK = ("pcabiop", "PCaBiop")
# One column per regime: the column header (top-panel title), the per-V accuracy key that
# apd_per_v_figure names its regimes by, and the nIPD column the scatter plots.
REGIMES = [
    ("In-domain", "id", "nipd_id"),
    ("Out-of-domain", "ood", "nipd_ood"),
]
FOCUSED_MODEL_COUNT = 6
#: Where each per-dataset composite is rendered. ``_finalize_figure`` writes
#: ``plots/{pdf,png}/`` beneath this stem; one stem per benchmark, filled in below.
#:
#: Beside the study data it is drawn from, never into ``paper/``: a figure rendered
#: straight into the manuscript tree is a second source of truth that goes stale silently
#: (see ADR-0010 and the note this module used to carry about staging into
#: ``paper/figures/results/``). Copy the PDF across by hand if it earns a float; nothing
#: here stages it. The raw-accuracy comparison uses a parallel name and never overwrites
#: the canonical normalized figure.
def _out_stem(dataset: str, *, normalized: bool = True,
              focused: bool = False) -> Path:
    if focused:
        return STUDY_DIR / "plots" / f"croma_nipd_focused_{dataset}.png"
    # Normalized trajectories are what nIPD integrates and therefore own the canonical
    # manuscript filename. Absolute probe accuracy remains available only as an explicitly
    # named exploratory output.
    infix = "" if normalized else "_raw_accuracy"
    return STUDY_DIR / "plots" / f"croma_nipd{infix}_{dataset}.png"


def _legend_bottom_margin(dataset: str) -> float | None:
    """Reserve only as much space as the dataset's model legend needs."""
    return None if dataset == "pcabiop" else 0.20


def focused_model_names(rows: list[dict], *, count: int = FOCUSED_MODEL_COUNT) -> list[str]:
    """Sample model names at equal intervals of the ascending CRoMa ranking."""
    ranked_rows = sorted(rows, key=lambda row: (float(row["croma"]), str(row["model"])))
    if count < 2 or count > len(ranked_rows):
        raise ValueError(f"count must lie in [2, {len(ranked_rows)}], got {count}")
    indices = [
        round(rank * (len(ranked_rows) - 1) / (count - 1))
        for rank in range(count)
    ]
    return [str(ranked_rows[index]["model"]) for index in indices]


def _perv_ylim(dataset: str, *, normalized: bool = True,
               models=None) -> tuple[float, float]:
    """Shared top-row limits over both regimes of one benchmark."""
    summary = _normalized_mean_and_ci if normalized else _mean_and_ci
    roster = _roster(dataset) if models is None else models
    curves = [
        summary(dataset, model, regime_key)
        for _, regime_key, _ in REGIMES
        for model in roster
    ]
    lo = min((mean - ci).min() for mean, ci in curves)
    hi = max((mean + ci).max() for mean, ci in curves)
    pad = max(0.8, (hi - lo) * 0.06)
    if normalized:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    return (lo - pad, hi + pad)


def _render_dataset(dset: str, dlabel: str, joined, correlations, *,
                    normalized: bool = True, focused: bool = False,
                    show_trend: bool = True, annotate_n: bool = False) -> None:
    df = joined[joined["dataset"] == dset].copy()
    df["nipd_id"] = df["nipd_id"] * 100.0
    df["nipd_ood"] = df["nipd_ood"] * 100.0
    rows = df.to_dict("records")
    top_models = (
        focused_model_names(rows) if focused else _roster(dset)
    )

    # The complete trajectory and scatter sources must describe the same model panel,
    # including when the focused view draws only a deterministic subset of the trajectories.
    # Fail loudly if a data refresh ever desynchronises them.
    full_roster, bottom_roster = set(_roster(dset)), set(df["model"])
    if full_roster != bottom_roster:
        raise ValueError(
            f"{dset}: the per-V curves and the CRoMa-vs-nIPD scatter disagree on the roster "
            f"(only in per-V: {sorted(full_roster - bottom_roster)}; only in scatter: "
            f"{sorted(bottom_roster - full_roster)}). One artifact is stale."
        )

    # sharex/sharey per row: the top row's two panels share the accuracy axis and the V
    # axis, the bottom row's share the nIPD axis and the CRoMa axis. Rows stay independent
    # (accuracy vs V above, nIPD vs CRoMa below), so nothing couples the two scales.
    fig, axes = plt.subplots(
        2, len(REGIMES), figsize=(plotstyle.COL_DOUBLE, 8.0),
        sharex="row", sharey="row",
    )

    top_ylim = _perv_ylim(dset, normalized=normalized, models=top_models)
    # One nIPD limit for the bottom row, spanning both regimes; the extra top padding is
    # headroom for the rho annotation (mirrors the retired 3x2 grid's per-row limit).
    row_ys = df[["nipd_id", "nipd_ood"]].to_numpy()
    ypad = max(0.5, (row_ys.max() - row_ys.min()) * 0.10)
    bot_ylim = (row_ys.min() - ypad, row_ys.max() + ypad * 1.6)

    for j, (col_label, vkey, ykey) in enumerate(REGIMES):
        # Top: accuracy vs Cramer's V. _draw_panel styles the axes, draws every encoder's
        # errorbar curve, and sets the V ticks; we square the box and add the labels.
        ax_top = axes[0][j]
        _draw_panel(ax_top, dset, vkey, ylim=top_ylim, correlations=correlations,
                    normalized=normalized, models=top_models)
        if focused and j == 0:
            for model in top_models:
                ax_top.plot(
                    [], [], color=plotstyle.color_for_model(model), label=model,
                    markeredgecolor="white", markeredgewidth=0.35, **SERIES_STYLE,
                )
        plotstyle.set_panel_title(ax_top, col_label)
        ax_top.set_xlabel(r"Cramér's $V$")
        # sharey blanks the right column's y ticks; only the left column keeps the label.
        if j == 0:
            ax_top.set_ylabel("Normalized performance change [%]" if normalized
                              else "LP accuracy [%]")

        # Bottom: CRoMa vs nIPD. Same scatter drawer, guide line and sign guard as the grid
        # this figure replaces -- a dotted fit that slopes against the annotated rho would
        # be a panel contradicting itself.
        ax_bot = axes[1][j]
        P._draw_model_scatter(
            ax_bot, rows, x_key="croma", y_key=ykey,
            xlabel="CRoMa", ylabel="nIPD (%)" if j == 0 else "",
            title="", ylim=bot_ylim, hline=0.0, vline=0.0,
        )
        if focused:
            selected = df[df["model"].isin(top_models)].to_dict("records")
            for row in selected:
                ax_bot.scatter(
                    [row["croma"]], [row[ykey]],
                    s=66, color=plotstyle.color_for_model(row["model"]),
                    edgecolors=plotstyle.SPINE_COLOR, linewidths=0.85,
                    zorder=4,
                )
        # The scatter drawer forces a square data box (its own identity). In this composite
        # the panel must instead fill its grid cell so the four panels align and the figure
        # stays compact rather than portrait; clear the forced aspect.
        ax_bot.set_box_aspect(None)
        slope = (
            P._draw_trend_line(ax_bot, df["croma"], df[ykey])
            if show_trend
            else None
        )
        rho, _ = spearmanr(df["croma"], df[ykey])
        if slope is not None and slope * rho < 0:
            raise ValueError(
                f"{dset}/{ykey}: the least-squares trend line slopes {slope:+.2f} but "
                f"Spearman rho is {rho:+.2f}. A guide-to-the-eye line that contradicts the "
                "annotated correlation cannot be drawn -- plot a monotone fit, or drop it."
            )
        annotation = f"Spearman $\\rho = {rho:.2f}$"
        if annotate_n:
            annotation += f"\n$n = {len(df)}$"
        ax_bot.text(0.04, 0.96, annotation,
                    transform=ax_bot.transAxes, ha="left", va="top",
                    fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)

    fig.suptitle(dlabel, y=0.985, fontsize=plotstyle.FS_TITLE + 1,
                 weight="bold", color=plotstyle.TEXT_COLOR)

    out = _out_stem(dset, normalized=normalized, focused=focused)
    legend_ax = axes[0][0] if focused else axes[1][0]
    P._finalize_figure(fig, out_path=out, legend_axes=[legend_ax],
                       top=0.915, bottom=_legend_bottom_margin(dset),
                       left=0.095, right=0.985,
                       wspace=0.14, hspace=0.30,
                       legend_y=0.012, legend_ncol=6,
                       legend_fontsize=plotstyle.FS_ANNOT,
                       legend_columnspacing=1.1, legend_handlelength=1.7)
    print(f"wrote {P._pdf_export_path(out)} and {P._png_export_path(out)}")


def main(*, normalized: bool = True, focused: bool = False,
         pcabiop: bool = False):
    plotstyle.apply_style()
    if pcabiop:
        normalized = True
        benchmarks = [PCABIOP_BENCHMARK]
    elif focused:
        normalized = True
        benchmarks = BENCHMARKS[:1]
    else:
        benchmarks = BENCHMARKS
    # The joined table keeps the natural-image control; ranked() drops it, so a panel never
    # annotates a rho no table reports. Same filter, one helper.
    # The joined study table stores registry identities; the labels are published.
    joined = plotstyle.published_models(ranked(read_joined()))
    correlations = {dset: training_correlations(dset) for dset, _ in benchmarks}
    for dset, dlabel in benchmarks:
        _render_dataset(dset, dlabel, joined, correlations[dset],
                        normalized=normalized, focused=focused,
                        show_trend=not pcabiop, annotate_n=pcabiop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--normalized",
        action="store_true",
        help="plot the canonical normalized nIPD integrand (the default)",
    )
    mode.add_argument(
        "--raw-accuracy",
        action="store_true",
        help="render exploratory absolute linear-probe accuracy under a non-canonical name",
    )
    mode.add_argument(
        "--focused",
        action="store_true",
        help="render the focused six-trajectory normalized Camelyon candidate",
    )
    mode.add_argument(
        "--pcabiop",
        action="store_true",
        help="render the normalized four-encoder PCaBiop composite",
    )
    args = parser.parse_args()
    main(
        normalized=args.normalized or not args.raw_accuracy,
        focused=args.focused,
        pcabiop=args.pcabiop,
    )
