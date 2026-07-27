"""Cross-benchmark CRoMa ranking figure (fig:cross-benchmark, supplement).

A rank bump chart across the three tile-level PathoROB benchmarks: one line per model,
vertical position = rank by the headline pooled CRoMa (1 = most robust, top). Marker is
filled when the model is biology-dominant (CRoMa >= 0) and hollow when confounder-dominant
(CRoMa < 0), so Camelyon's difficulty and the rank crossings (Midnight-12k rising to the
top) are both visible at a glance.

What is drawn -- the panel, the ranks, the dagger set -- comes from ``_cross_benchmark.load()``,
which the float generator also reads. Neither the models nor the benchmark paths are named here:
this script used to spell its three run directories out at the old protocol, and those runs were
archived when the tile panel was re-run, so it was drawing a benchmark that no longer existed.

Run: python scripts/repro/figures/cross_benchmark_figure.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO, _REPO / "scripts" / "bench", _REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _cross_benchmark import load  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
from plotting import (  # noqa: E402
    DEFAULT_DPI,
    TEXT_COLOR,
    _color_for_model,
    _style_axes,
)

#: Beside the study data it is drawn from, under ``plots/{pdf,png}/``. Nothing writes into
#: ``paper/figures/``; copy what the paper needs by hand. This used to be a bare relative
#: path, so the figure landed wherever the caller happened to be standing.
OUT = _REPO / "output/studies/cross-benchmark/plots/cross_benchmark.pdf"


def main() -> None:
    from croma import plotstyle

    cb = load()
    names = cb.labels
    xs = list(range(len(names)))

    # One row per model: a fixed height crowded the end labels as the panel grew 16 -> 20.
    fig, ax = plt.subplots(figsize=(8.0, 0.40 * cb.n_models))
    _style_axes(ax, grid_axis="y")

    for model in cb.croma.index:
        color = _color_for_model(model)
        exposed = model in cb.exposed
        tag = r"$^\dagger$" if exposed else ""
        ys = [cb.rank_of(model, n) for n in names]
        ax.plot(xs, ys, ("--" if exposed else "-"), color=color,
                lw=plotstyle.LW_SERIES, alpha=0.85, zorder=2)
        for x, n in zip(xs, names):
            robust = cb.croma.loc[model, n] >= 0.0
            ax.plot(
                x, cb.rank_of(model, n), "o", ms=7, zorder=3,
                mfc=(color if robust else "white"),
                mec=color, mew=1.2,
            )
        # end labels with CRoMa value, coloured by model (dagger = TCGA-exposed)
        ax.text(-0.06, cb.rank_of(model, names[0]),
                f"{model}{tag}  {cb.croma.loc[model, names[0]]:.2f}",
                ha="right", va="center", fontsize=plotstyle.FS_ANNOT, color=color)
        ax.text(len(names) - 1 + 0.06, cb.rank_of(model, names[-1]),
                f"{cb.croma.loc[model, names[-1]]:.2f}  {model}{tag}",
                ha="left", va="center", fontsize=plotstyle.FS_ANNOT, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=plotstyle.FS_LABEL, weight="bold")
    ax.set_yticks(range(1, cb.n_models + 1))
    ax.set_ylim(cb.n_models + 0.6, 0.4)  # rank 1 on top
    ax.set_xlim(-1.15, len(names) - 1 + 1.15)
    ax.set_ylabel(rf"Rank by pooled CRoMa ($m{{=}}{int(CROMA_HEADLINE_M)}$)")
    plotstyle.set_panel_title(ax, "Robustness ranking across tile-level benchmarks")

    # marker-style legend (fill = robust, hollow = confounder-dominant)
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", ms=7, mfc=TEXT_COLOR,
                   mec=TEXT_COLOR, label=r"biology-dominant (CRoMa $\geq$ 0)"),
        plt.Line2D([0], [0], marker="o", ls="", ms=7, mfc="white",
                   mec=TEXT_COLOR, mew=1.2,
                   label=r"confounder-dominant (CRoMa $<$ 0)"),
        plt.Line2D([0], [0], ls="--", color=TEXT_COLOR,
                   lw=plotstyle.LW_SERIES,
                   label=r"$\dagger$ TCGA-exposed (pretraining leakage)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13),
              ncol=3, frameon=False, fontsize=plotstyle.FS_ANNOT, handletextpad=0.4)

    # The end labels live inside the axes (xlim is padded), so the left margin only has to
    # clear the rotated y-label -- which 0.04 did not, clipping it against the canvas edge.
    fig.subplots_adjust(left=0.075, right=0.965, top=0.93, bottom=0.10)
    for sub in ("png", "pdf"):
        (OUT.parent / sub).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.parent / "png" / OUT.with_suffix(".png").name, dpi=DEFAULT_DPI)
    fig.savefig(OUT.parent / "pdf" / OUT.name)
    plt.close(fig)

    print("wrote", OUT.parent / "pdf" / OUT.name, "and png/")
    print(f"\n{cb.n_models} ranked models, {len(cb.exposed)} TCGA-exposed")
    print("Ranks (1 = most robust):")
    print(cb.ranks.to_string())


if __name__ == "__main__":
    main()
