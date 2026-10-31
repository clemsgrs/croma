r"""Render ``fig:margin-montage``: the lowest- and highest-CRoMa tiles of one cell per cohort.

For each tile cohort (Camelyon, TCGA-4x4) the encoder, the class x centre cell and the tiles
are chosen by ``_sample_shortcut.montage_choice`` -- the evaluable pathology encoder at the
panel's lower-median within-cell CRoMa AUROC, its cell with the most flipped tiles, and that
cell's ``n`` lowest- and ``n`` highest-CRoMa tiles among those the balanced probe never got
wrong. Each tile carries its CRoMa, whether the fully confounded probe flipped it, and its
colour centre-confidence; a flipped tile is also framed.

The study is read through ``_sample_shortcut.load``, so a stale study cannot be drawn. The
PDF carries no creation date, so a re-render of the same study is byte-for-byte identical.
Writes ``output/studies/sample-shortcut/plots/{pdf,png}/margin_montage.*``; the copy into
``paper/figures/`` stays manual (ADR-0010).

Run: python scripts/repro/figures/margin_montage_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe; figure scripts never show()
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO / "src", REPO / "scripts" / "repro", REPO / "scripts" / "bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _sample_shortcut as basis  # noqa: E402
from plotting import style as plotstyle  # noqa: E402

NAME = "margin_montage"
#: The frame drawn around a flipped tile.
FLIPPED_COLOR = "#b2182b"


def tile_label(row: pd.Series) -> str:
    """The two-line annotation under a tile: CRoMa, flip, colour centre-confidence."""
    croma = f"{float(row['croma_m5']):.2f}".replace("-", "−")
    flipped = "flipped" if bool(row["flipped"]) else "not flipped"
    return f"CRoMa {croma}\n{flipped} · colour {float(row['colour_confidence']):.2f}"


def _read_tile(path: str):
    import matplotlib.image as mpimg

    return mpimg.imread(path)


def _cell_text(cell: str) -> str:
    label, centre = cell.split(" | ", 1)
    return f"{label.replace('_', ' ')} · {centre}"


def render(root: Path = REPO, out_dir: Path | None = None, *, n: int = basis.MONTAGE_TILES) -> Path:
    """Draw the montage from the study under ``root``; returns the PDF path."""
    root = Path(root)
    out_dir = Path(out_dir) if out_dir is not None else root / basis.STUDY_REL / "plots"
    choices = [
        (cohort, basis.montage_choice(basis.load(cohort, root), n=n))
        for cohort in basis.TILE_COHORTS
    ]
    plotstyle.apply_style()

    margin, label_w, tile, gap = 0.08, 0.30, 0.80, 0.05
    title_h, note_h, cohort_gap = 0.30, 0.30, 0.18
    fig_w = margin + label_w + n * tile + (n - 1) * gap + margin
    block_h = title_h + 2 * (tile + note_h)
    fig_h = margin + len(choices) * block_h + (len(choices) - 1) * cohort_gap + margin
    fig = plt.figure(figsize=(fig_w, fig_h))

    for b, (cohort, choice) in enumerate(choices):
        top = margin + b * (block_h + cohort_gap)
        short = basis.load(cohort, root).run.short_name or cohort
        fig.text(
            (margin + label_w) / fig_w,
            1 - (top + 0.04) / fig_h,
            f"{short}: {choice.model}, {_cell_text(choice.cell)}",
            ha="left",
            va="top",
            fontsize=plotstyle.FS_TITLE,
            weight="bold",
            color=plotstyle.TEXT_COLOR,
        )
        for r, (title, tiles) in enumerate((("lowest CRoMa", choice.low), ("highest CRoMa", choice.high))):
            row_top = top + title_h + r * (tile + note_h)
            fig.text(
                (margin + label_w * 0.45) / fig_w,
                1 - (row_top + tile / 2) / fig_h,
                title,
                rotation=90,
                ha="center",
                va="center",
                fontsize=plotstyle.FS_ANNOT,
                color=plotstyle.MUTED_TEXT_COLOR,
            )
            for c, (_, sample) in enumerate(tiles.iterrows()):
                x0 = margin + label_w + c * (tile + gap)
                ax = fig.add_axes(
                    [x0 / fig_w, 1 - (row_top + tile) / fig_h, tile / fig_w, tile / fig_h]
                )
                ax.imshow(_read_tile(sample["image_path"]))
                ax.set_xticks([])
                ax.set_yticks([])
                flipped = bool(sample["flipped"])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(FLIPPED_COLOR if flipped else plotstyle.SPINE_COLOR)
                    spine.set_linewidth(2.0 if flipped else plotstyle.LW_SPINE)
                fig.text(
                    (x0 + tile / 2) / fig_w,
                    1 - (row_top + tile + 0.03) / fig_h,
                    tile_label(sample),
                    ha="center",
                    va="top",
                    fontsize=plotstyle.FS_ANNOT - 1.5,
                    color=FLIPPED_COLOR if flipped else plotstyle.TEXT_COLOR,
                    linespacing=1.1,
                )

    pdf = out_dir / "pdf" / f"{NAME}.pdf"
    png = out_dir / "png" / f"{NAME}.png"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=plotstyle.DEFAULT_DPI, metadata={"Software": None})
    plt.close(fig)
    return pdf


def main() -> int:
    print(f"wrote {render()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
