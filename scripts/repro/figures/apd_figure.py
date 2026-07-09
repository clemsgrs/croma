"""CRoMa-vs-APD scatter grid for the APD-validation section (Fig. croma-vs-apd).

3x2 grid: one row per faithful tile benchmark (Camelyon, TCGA in its four-class
form, Tolkach-ESCA), left column vs in-domain APD (isolates shortcut reliance),
right column vs out-of-domain APD (cross-centre generalisation). Every panel shares
the CRoMa x-axis and the project plotting identity (family palette, square panels,
shared model legend). PANDA is excluded (slide-level, only 4 points).
"""
import sys
from pathlib import Path

from scipy.stats import spearmanr

# The APD loaders live with the rest of the apd study package; reach them across dirs.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "studies" / "apd"))
from loaders import REPO, read_joined  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "bench"))
import plotting as P  # noqa: E402
from croma import plotstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# (dataset key in the joined table, display label). TCGA APD is the four-class form.
BENCHMARKS = [
    ("camelyon", "Camelyon"),
    ("tcga_4x4", r"TCGA (4$\times$4)"),
    ("tolkach", "Tolkach-ESCA"),
]
REGIMES = [
    ("apd_id", "In-domain APD (%)"),
    ("apd_ood", "Out-of-domain APD (%)"),
]
OUT = REPO / "paper/figures/results/pathorob-camelyon-faithful/croma_vs_apd_scatter.png"


def main():
    plotstyle.apply_style()
    joined = read_joined()

    fig, axes = plt.subplots(
        len(BENCHMARKS), len(REGIMES),
        figsize=(plotstyle.COL_DOUBLE, 4.6 * len(BENCHMARKS)),
    )

    for i, (dset, dlabel) in enumerate(BENCHMARKS):
        df = joined[joined["dataset"] == dset].copy()
        df["apd_id"] = df["apd_id"] * 100.0
        df["apd_ood"] = df["apd_ood"] * 100.0
        rows = df.to_dict("records")
        for j, (ykey, ylabel) in enumerate(REGIMES):
            ax = axes[i][j]
            ys = df[ykey].to_numpy()
            ypad = max(0.5, (ys.max() - ys.min()) * 0.10)
            P._draw_model_scatter(
                ax, rows, x_key="croma", y_key=ykey,
                xlabel="CRoMa", ylabel=ylabel,
                title=f"{dlabel}: {'in-domain' if ykey == 'apd_id' else 'out-of-domain'}",
                ylim=(ys.min() - ypad, ys.max() + ypad * 1.6),
                hline=0.0, vline=0.0,
            )
            rho, _ = spearmanr(df["croma"], df[ykey])
            ax.text(0.04, 0.96, f"Spearman $\\rho = {rho:.2f}$",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)

    P._finalize_figure(fig, out_path=OUT, legend_axes=[axes[0][0]],
                       top=0.955, bottom=0.11, left=0.085, right=0.985,
                       wspace=0.28, hspace=0.30,
                       legend_y=0.02, legend_ncol=6,
                       legend_fontsize=plotstyle.FS_ANNOT,
                       legend_columnspacing=1.1, legend_handlelength=1.7)
    print(f"wrote {P._pdf_export_path(OUT)} and {P._png_export_path(OUT)}")


if __name__ == "__main__":
    main()
