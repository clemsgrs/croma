"""Probe accuracy against injected training-data correlation (Fig. apd-per-v).

2x3 grid: rows are the two APD test regimes (in-domain, out-of-domain), columns the three
faithful PathoROB tile benchmarks. Each curve is one encoder's linear-probe accuracy as the
centre<->biology correlation of the probe's *training* set is walked from balanced
(Cramer's V = 0) to fully confounded (V = 1); error bars are 95% t-intervals over the 20
seeds. Every panel shares the project plotting identity (family palette, shared model
legend). PANDA is excluded (slide-level); prostate is excluded (its single OOD centre).

This is the raw material the nIPD scalar integrates:

    nIPD = integral_0^1 (mean_acc(V) - mean_acc(0)) / (mean_acc(0) - chance) dV

The reduction is a *ratio*, so it discards precisely what this figure keeps -- the absolute
accuracy level each curve departs from. Two encoders with the same APD can sit ten accuracy
points apart, and one whose acc(0) is already high is penalised for having further to fall.
Figure ``fig:croma-vs-apd`` plots the scalar; this one plots what it was computed from.
"""
import json
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# The APD loaders live with the rest of the apd study package; reach them across dirs.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "studies" / "apd"))
from loaders import DATASETS, REPO, ranked, training_correlations  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "bench"))
import plotting as P  # noqa: E402
from plotting import style as plotstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# (dataset key in apd.csv, display label). TCGA APD is the four-class form. Same three
# benchmarks, in the same order, as the rows of the CRoMa-vs-APD scatter.
BENCHMARKS = [
    ("camelyon", "Camelyon"),
    ("tcga_4x4", r"TCGA (4$\times$4)"),
    ("tolkach", "Tolkach-ESCA"),
]
# (JSON key prefix, row label). Mirrors _apd.TARGETS, whose order the paper's captions follow.
REGIMES = [
    ("id", "In-domain (ID)"),
    ("ood", "Out-of-domain (OOD)"),
]
RAW = REPO / "output/studies/apd"
#: Beside the study data it is drawn from, not under ``paper/``. Nothing in this repo writes
#: into the manuscript tree: a figure that lands there by itself is a copy nobody owns, and
#: it drifts from the run behind it in silence. Copy the PDF by hand if it earns a float, and
#: run ``scripts/repro/check_paper_figures.py`` to see which copies have fallen behind.
#: ``_finalize_figure`` writes ``plots/{pdf,png}/`` beneath this stem (cf. layout.plots_dir).
OUT = REPO / "output/studies/apd/plots/apd_per_correlation.png"

CI_LEVEL = 0.95


@cache
def _roster(dataset):
    """The encoders this benchmark's APD was computed over, in canonical order.

    Read from ``apd.csv`` rather than from the raw JSONs on disk: the summary CSV is what
    every other APD artifact joins against, so a model with a stale JSON but no CSV row must
    not appear here and contradict the scatter's roster. The natural-image control is a
    floor, not a competitor, and ``ranked`` drops it here for the same reason it does there.
    """
    # The study CSV stores registry identities; the figure labels are a published surface.
    apd = plotstyle.published_models(ranked(pd.read_csv(RAW / "apd.csv")))
    models = apd.loc[apd["dataset"] == dataset, "model"]
    return sorted(models, key=lambda m: (plotstyle.model_sort_key(m), m))


@cache
def _accuracy_matrix(dataset, model, regime):
    # The rosters carry published names; the per-model JSONs keep registry filenames.
    filename = f"{plotstyle.registry_model_name(model)}.json"
    raw = json.loads((RAW / dataset / filename).read_text())
    return np.asarray(raw[f"{regime}_test_accuracies"], dtype=float)


@cache
def _mean_and_ci(dataset, model, regime):
    """Per-split mean accuracy and 95% t-interval half-width, in percent.

    ``<regime>_test_accuracies`` is (num_splits, iterations): one probe per seed per split.
    PathoROB summarises the seed spread with a Student-t interval, not a normal one -- 20
    seeds is small enough for the difference to be visible in the caps.
    """
    accuracies = _accuracy_matrix(dataset, model, regime) * 100.0
    iterations = accuracies.shape[1]
    half_width = student_t.ppf(0.5 + CI_LEVEL / 2, df=iterations - 1) * (
        accuracies.std(axis=1, ddof=1) / np.sqrt(iterations)
    )
    return accuracies.mean(axis=1), half_width


def normalized_change_summary(accuracies, *, chance):
    """Mean nIPD integrand and paired-repeat t-interval, in percent."""
    accuracies = np.asarray(accuracies, dtype=float)
    baseline_headroom = accuracies[0].mean() - chance
    if baseline_headroom <= 0:
        raise ValueError("mean baseline accuracy must exceed chance")

    changes = (accuracies - accuracies[0]) / baseline_headroom * 100.0
    iterations = changes.shape[1]
    half_width = student_t.ppf(0.5 + CI_LEVEL / 2, df=iterations - 1) * (
        changes.std(axis=1, ddof=1) / np.sqrt(iterations)
    )
    return changes.mean(axis=1), half_width


@cache
def _normalized_mean_and_ci(dataset, model, regime):
    chance = 1.0 / len(DATASETS[dataset]["biological_classes"])
    return normalized_change_summary(_accuracy_matrix(dataset, model, regime), chance=chance)


#: Labelled at a round grid shared by all three columns; the schedule's own V values are
#: marked as unlabelled minor ticks. Eight labelled ticks (Camelyon's schedule) do not fit
#: under a third of a double-column figure, and rotating them buys legibility with a whole
#: row of vertical space. The sampled points are visible as the data points themselves.
XTICKS = [0.0, 0.25, 0.50, 0.75, 1.0]

#: One series style, shared by the drawn curves and the legend's proxy handles.
SERIES_STYLE = dict(marker="o", markersize=2.8, linestyle="--", linewidth=0.8)


def _draw_panel(ax, dataset, regime, *, ylim, correlations, normalized=False,
                models=None):
    summary = _normalized_mean_and_ci if normalized else _mean_and_ci
    for model in _roster(dataset) if models is None else models:
        means, half_width = summary(dataset, model, regime)
        ax.errorbar(
            correlations, means, yerr=half_width,
            color=plotstyle.color_for_model(model),
            markeredgecolor="white", markeredgewidth=0.35, alpha=0.9,
            capsize=1.4, elinewidth=0.6, capthick=0.6,
            zorder=3, **SERIES_STYLE,
        )
    plotstyle.style_axes(ax)
    if normalized:
        ax.axhline(0.0, color=plotstyle.REFERENCE_LINE_COLOR, linestyle="--",
                   linewidth=plotstyle.LW_REFERENCE, zorder=1)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([f"{v:.2f}" for v in XTICKS])
    ax.set_xticks(correlations, minor=True)
    ax.tick_params(axis="x", which="minor", length=2.0,
                   width=plotstyle.LW_SPINE, color=plotstyle.TEXT_COLOR)


def _row_label(ax, text):
    """Rotated regime label outside the y-axis, as in PathoROB's own per-V figure."""
    ax.annotate(
        text, xy=(0, 0.5), xytext=(-38, 0),
        xycoords="axes fraction", textcoords="offset points",
        ha="center", va="center", rotation=90,
        fontsize=plotstyle.FS_TITLE, weight="bold", color=plotstyle.TEXT_COLOR,
    )


def main():
    plotstyle.apply_style()

    correlations = {dataset: training_correlations(dataset) for dataset, _ in BENCHMARKS}
    # Both regimes of a benchmark share a y-axis: the figure's point is that OOD sits lower
    # than ID at the same V, and a per-panel autoscale would hide exactly that.
    curves = {
        (dataset, regime): [_mean_and_ci(dataset, m, regime) for m in _roster(dataset)]
        for dataset, _ in BENCHMARKS
        for regime, _ in REGIMES
    }
    ylims = {}
    for dataset, _ in BENCHMARKS:
        lo = min((mean - ci).min() for r, _ in REGIMES for mean, ci in curves[(dataset, r)])
        hi = max((mean + ci).max() for r, _ in REGIMES for mean, ci in curves[(dataset, r)])
        pad = max(0.8, (hi - lo) * 0.06)
        ylims[dataset] = (lo - pad, hi + pad)

    fig, axes = plt.subplots(
        len(REGIMES), len(BENCHMARKS),
        figsize=(plotstyle.COL_DOUBLE, 5.4),
    )

    for row, (regime, regime_label) in enumerate(REGIMES):
        for col, (dataset, dataset_label) in enumerate(BENCHMARKS):
            ax = axes[row][col]
            _draw_panel(ax, dataset, regime, ylim=ylims[dataset],
                        correlations=correlations[dataset])
            if row == 0:
                plotstyle.set_panel_title(ax, dataset_label)
            if col == 0:
                ax.set_ylabel("LP accuracy [%]")
                _row_label(ax, regime_label)

    # Legend proxies, drawn empty on one axes: an errorbar's own handle carries its caps, so
    # the legend would read "I--o--" once per model. The union over benchmarks, because a
    # model APD has on Camelyon but not Tolkach still belongs in the key.
    legend_models = sorted(
        {model for dataset, _ in BENCHMARKS for model in _roster(dataset)},
        key=lambda m: (plotstyle.model_sort_key(m), m),
    )
    for model in legend_models:
        axes[0][0].plot([], [], color=plotstyle.color_for_model(model), label=model,
                        **SERIES_STYLE)

    fig.text(0.55, 0.147, "Training data correlation (Cramér's $V$)",
             ha="center", va="center", fontsize=plotstyle.FS_LABEL, color=plotstyle.TEXT_COLOR)

    P._finalize_figure(fig, out_path=OUT, legend_axes=[axes[0][0]],
                       top=0.945, bottom=0.195, left=0.115, right=0.985,
                       wspace=0.26, hspace=0.16,
                       legend_y=0.004, legend_ncol=6,
                       legend_fontsize=plotstyle.FS_ANNOT,
                       legend_columnspacing=1.1, legend_handlelength=1.7)
    print(f"wrote {P._pdf_export_path(OUT)} and {P._png_export_path(OUT)}")


if __name__ == "__main__":
    main()
