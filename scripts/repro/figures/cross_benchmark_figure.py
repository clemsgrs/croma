"""Cross-benchmark CRoMa ranking figure (paper Fig. 3).

A rank bump chart across the three tile-level PathoROB benchmarks: one line per
model, vertical position = rank by the headline pooled CRoMa (1 = most robust, top).
Marker is filled when the model is biology-dominant (CRoMa >= 0) and hollow when
confounder-dominant (CRoMa < 0), so Camelyon's difficulty and the rank crossings
(e.g. Midnight-12k rising to the top) are both visible at a glance.

Run: python scripts/repro/figures/cross_benchmark_figure.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root (croma pkg)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))  # scripts/bench (plotting)
from plotting import (  # noqa: E402
    DEFAULT_DPI,
    REFERENCE_LINE_COLOR,
    TEXT_COLOR,
    _color_for_model,
    _style_axes,
)
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402

BENCHMARKS = [
    ("Camelyon", "output/faithful/pathorob-camelyon-faithful"),
    ("Prostate", "output/prostate-shift-binary-kirumc"),
    ("TCGA (2$\\times$2)", "output/faithful/pathorob-tcga-2x2"),
    ("Tolkach-ESCA", "output/faithful/pathorob-tolkach-esca-faithful"),
]
OUT = Path("paper/figures/cross_benchmark.pdf")


def _load() -> pd.DataFrame:
    cols = {}
    for name, root in BENCHMARKS:
        s = pd.read_csv(Path(root) / "results" / "metrics.csv").set_index("model")["croma"]
        cols[name] = s
    df = pd.DataFrame(cols).dropna()  # models present in all three
    return df


def main() -> None:
    df = _load()
    names = [n for n, _ in BENCHMARKS]
    # rank 1 = highest CRoMa within each benchmark
    ranks = df.rank(ascending=False, method="first").astype(int)
    n_models = len(df)
    xs = list(range(len(names)))

    from croma import plotstyle

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    _style_axes(ax, grid_axis="y")

    for model in df.index:
        color = _color_for_model(model)
        ys = [ranks.loc[model, n] for n in names]
        ax.plot(xs, ys, "-", color=color, lw=plotstyle.LW_SERIES, alpha=0.85, zorder=2)
        for x, n in zip(xs, names):
            robust = df.loc[model, n] >= 0.0
            ax.plot(
                x, ranks.loc[model, n], "o", ms=7, zorder=3,
                mfc=(color if robust else "white"),
                mec=color, mew=1.2,
            )
        # end labels with CRoMa value, coloured by model
        ax.text(-0.06, ranks.loc[model, names[0]],
                f"{model}  {df.loc[model, names[0]]:.2f}",
                ha="right", va="center", fontsize=plotstyle.FS_ANNOT, color=color)
        ax.text(len(names) - 1 + 0.06, ranks.loc[model, names[-1]],
                f"{df.loc[model, names[-1]]:.2f}  {model}",
                ha="left", va="center", fontsize=plotstyle.FS_ANNOT, color=color)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=plotstyle.FS_LABEL, weight="bold")
    ax.set_yticks(range(1, n_models + 1))
    ax.set_ylim(n_models + 0.6, 0.4)  # rank 1 on top
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
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.11),
              ncol=2, frameon=False, fontsize=plotstyle.FS_ANNOT, handletextpad=0.4)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.10)
    for sub in ("png", "pdf"):
        (OUT.parent / sub).mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.parent / "png" / OUT.with_suffix(".png").name, dpi=DEFAULT_DPI)
    fig.savefig(OUT)  # flat pdf in paper/figures/ for \graphicspath
    plt.close(fig)

    print("wrote", OUT, "and png/")
    print("\nRanks (1 = most robust):")
    print(ranks.to_string())


if __name__ == "__main__":
    main()
