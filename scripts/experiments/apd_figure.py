"""CCMR-vs-APD scatter (PathoROB Camelyon) for the APD-validation section.

Two panels share the CCMR x-axis: left vs in-domain APD (isolates shortcut
reliance), right vs out-of-domain APD (cross-centre generalisation). Reuses the
project plotting identity (family palette, square panels, shared model legend).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path("/data/pathology/projects/clement/code/croma")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import plotting as P  # noqa: E402
from croma import plotstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATASET = "camelyon"
OUT = REPO / "paper/figures/results/pathorob-camelyon-faithful/ccmr_vs_apd_scatter.png"


def main():
    plotstyle.apply_style()
    # apd_metrics_joined.csv is written by apd_ccmr_correlation.py with CCMR already
    # on the signed-margin scale (the paper's definition).
    df = pd.read_csv(REPO / "output/apd/apd_metrics_joined.csv")
    df = df[df["dataset"] == DATASET].copy()
    df["apd_id_pct"] = df["apd_id"] * 100.0
    df["apd_ood_pct"] = df["apd_ood"] * 100.0
    rows = df.to_dict("records")

    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.COL_DOUBLE, 5.0))
    panels = [
        (axes[0], "apd_id_pct", "apd_id", "In-domain APD (%)", "CCMR vs in-domain APD"),
        (axes[1], "apd_ood_pct", "apd_ood", "Out-of-domain APD (%)", "CCMR vs out-of-domain APD"),
    ]
    for ax, ykey, raw, ylabel, title in panels:
        ys = df[ykey].to_numpy()
        ypad = max(0.5, (ys.max() - ys.min()) * 0.10)
        P._draw_model_scatter(
            ax, rows, x_key="ccmr", y_key=ykey,
            xlabel="CCMR", ylabel=ylabel, title=title,
            ylim=(ys.min() - ypad, ys.max() + ypad * 1.6),
            hline=0.0, vline=0.0,
        )
        rho, _ = spearmanr(df["ccmr"], df[raw])
        ax.text(0.04, 0.96, f"Spearman $\\rho = {rho:.2f}$",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)

    P._finalize_figure(fig, out_path=OUT, legend_axes=[axes[0]],
                       top=0.93, bottom=0.27, left=0.085, right=0.985, wspace=0.28,
                       legend_y=0.02, legend_ncol=6,
                       legend_fontsize=plotstyle.FS_ANNOT,
                       legend_columnspacing=1.1, legend_handlelength=1.7)
    print(f"wrote {P._pdf_export_path(OUT)} and {P._png_export_path(OUT)}")


if __name__ == "__main__":
    main()
