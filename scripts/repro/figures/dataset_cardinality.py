"""Dataset-cardinality figure (PathoROB card-style): biology x confounder counts.

Renders ``fig:dataset-cardinality`` for ``paper/sections/dataset_summary.tex``: one
block per evaluation benchmark, each a grid of light cells whose text is the number
of samples in that (biological class x confounder) cell, mirroring the composition
card in PathoROB. Row labels are the biological classes, column labels the
confounder levels, and a bold title names the dataset. Every cell in a block is
balanced by construction, so the grid reads as a single count repeated across the
class x confounder product -- the point of the figure is the *shape* (how many
classes, how many centres) and the total sample size, not per-cell variation.

The per-cell counts are dataset-definitional constants (the benchmarks are balanced
by design), so the figure builds anywhere -- it needs no tile images and no run
outputs, exactly like the hand-authored table it replaces. ``--verify`` cross-checks
the hard-coded specs against the actual evaluated runs when ``output/`` is present:

    Camelyon      5,100/cell  (2 classes  x 2 centres)   = 20,400 tiles
    TCGA (4x4)      360/cell   (4 classes  x 4 centres)   =  5,760 tiles
    Tolkach-ESCA    500/cell   (6 classes  x 3 cohorts)   =  9,000 tiles
    PCaBiop         250/cell   (2 classes  x 2 providers) =  1,000 slides

Two paired-protocol cohorts get their own supplement schematics, rendered separately
(``--tcga2x2-out``, ``--pcabiop-isup-out``): TCGA's generic 2x2 template (x 94 quartets)
and PCaBiop-ISUP's 6x2 grade x centre cohort decomposed into its 15 grade-pair quartets.

Run:
    PYTHONPATH=src python scripts/repro/figures/dataset_cardinality.py
    PYTHONPATH=src python scripts/repro/figures/dataset_cardinality.py --verify
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo root: scripts/repro/figures/dataset_cardinality.py -> parents[3].
REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
# paper_manifest lives in scripts/repro/; --verify resolves each run through it.
sys.path.insert(0, str(REPO / "scripts" / "repro"))

# Identity only from croma.plotstyle (registers fonts + rcParams on import).
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting import style as plotstyle  # noqa: E402

# Cell chrome. A single neutral fill for every cell: the benchmarks are balanced,
# so shade carries no information -- the count text does. Matches the card's look.
CELL_FACE = "#eaeaea"
CELL_EDGE = "#c3c3c3"
LW_CELL = 0.6


@dataclass(frozen=True)
class CardSpec:
    """One dataset block: a balanced ``rows x cols`` grid with ``cell`` samples/cell."""

    name: str
    subtitle: str  # muted one-liner: organ . level
    rows: tuple[str, ...]  # biological classes (top -> bottom)
    cols: tuple[str, ...]  # confounder levels (left -> right)
    cell: int  # samples per cell (uniform: balanced by construction)
    unit: str  # "tiles" | "slides"
    benchmark: str  # paper_manifest benchmark key (for --verify)

    @property
    def total(self) -> int:
        return self.cell * len(self.rows) * len(self.cols)


# The four evaluation benchmarks. Row/column order matches the dataset-summary table
# and caption; per-cell counts are verified against the evaluated runs (see --verify).
SPECS: tuple[CardSpec, ...] = (
    CardSpec(
        name="Camelyon",
        subtitle="breast LN · tile",
        rows=("Normal", "Tumour"),
        cols=("RUMC", "UMCU"),
        cell=5100,
        unit="tiles",
        benchmark="pathorob-camelyon",
    ),
    CardSpec(
        name="TCGA (4×4)",
        subtitle="pan-cancer · tile",
        rows=("BRCA", "COAD", "LUAD", "LUSC"),
        cols=("Asterand", "Christiana", "Roswell Park", "U. Pittsburgh"),
        cell=360,
        unit="tiles",
        benchmark="pathorob-tcga-4x4",
    ),
    CardSpec(
        name="Tolkach-ESCA",
        subtitle="oesophagus · tile",
        rows=(
            "Tumour",
            "Regression\ntissue",
            "Oesophageal\nmucosa",
            "Gastric\nmucosa",
            "Muscularis\npropria",
            "Adventitial\ntissue",
        ),
        cols=("UKK", "WNS", "CHA"),
        cell=500,
        unit="tiles",
        benchmark="pathorob-tolkach-esca",
    ),
    CardSpec(
        name="PCaBiop",
        subtitle="prostate · slide",
        rows=("Benign", "Cancer"),
        cols=("KI", "RUMC"),
        cell=250,
        unit="slides",
        benchmark="panda",
    ),
)


# ---------------------------------------------------------------------------
# Layout. Blocks live on one axes in cell units (cell = 1x1) so every cell is the
# same physical size across datasets, as in the card. y increases upward; each
# block is placed by the y of its top edge (``top``) and the x of its left edge.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Placement:
    left: float
    top: float


TOP = 6.0  # top-align Camelyon, TCGA and Tolkach along this edge
PLACEMENTS: dict[str, Placement] = {
    "Camelyon": Placement(left=0.0, top=TOP),
    "PCaBiop": Placement(left=0.0, top=TOP - 4.6),  # stacked under Camelyon, clear of its labels
    "TCGA (4×4)": Placement(left=5.2, top=TOP),
    "Tolkach-ESCA": Placement(left=11.6, top=TOP),
}


def _draw_block(ax, spec: CardSpec, place: Placement) -> None:
    from matplotlib.patches import Rectangle

    nrows, ncols = len(spec.rows), len(spec.cols)
    left, top = place.left, place.top
    mid_x = left + ncols / 2.0
    bottom = top - nrows

    # Title + muted subtitle, stacked above the block.
    ax.text(
        mid_x, top + 0.62, spec.name,
        ha="center", va="bottom",
        fontsize=plotstyle.FS_TITLE, weight="bold", color=plotstyle.TEXT_COLOR,
    )
    ax.text(
        mid_x, top + 0.18, f"{spec.subtitle} · {spec.total:,} {spec.unit}",
        ha="center", va="bottom",
        fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR,
    )

    count = f"{spec.cell:,}"
    for r in range(nrows):
        cy = top - r - 0.5  # cell-centre y (row 0 at the top)
        for c in range(ncols):
            ax.add_patch(
                Rectangle(
                    (left + c, top - r - 1), 1.0, 1.0,
                    facecolor=CELL_FACE, edgecolor=CELL_EDGE, linewidth=LW_CELL,
                )
            )
            ax.text(
                left + c + 0.5, cy, count,
                ha="center", va="center",
                fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR,
            )
        # Row label (biological class), right-aligned in the left gutter.
        ax.text(
            left - 0.22, cy, spec.rows[r],
            ha="right", va="center",
            fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR,
            linespacing=0.9,
        )

    # Column labels (confounder levels), rotated beneath the bottom row.
    for c in range(ncols):
        ax.text(
            left + c + 0.5, bottom - 0.18, spec.cols[c],
            ha="right", va="top", rotation=32, rotation_mode="anchor",
            fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR,
        )


def render(specs: "tuple[CardSpec, ...]", out_path: Path) -> Path:
    """Render the composition card and write a flat PDF (+ PNG sibling under png/)."""
    import matplotlib.pyplot as plt

    plotstyle.apply_style()

    fig = plt.figure(figsize=(plotstyle.COL_DOUBLE, plotstyle.COL_DOUBLE * 0.52))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect("equal")
    ax.axis("off")

    for spec in specs:
        _draw_block(ax, spec, PLACEMENTS[spec.name])

    ax.set_xlim(-2.6, 15.2)
    ax.set_ylim(-2.1, 7.2)

    out_path = Path(out_path)
    (out_path.parent / "png").mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.parent / "png" / out_path.with_suffix(".png").name, dpi=plotstyle.DEFAULT_DPI)
    fig.savefig(out_path)  # flat pdf for \graphicspath
    plt.close(fig)
    return out_path


# TCGA's paired 2x2 protocol (supplement). Unlike the four balanced benchmarks above, its
# biological classes and centres differ from quartet to quartet, so there is no fixed named
# grid: it is one generic two-class x two-centre template, evaluated over 94 quartets and
# pooled. The card therefore shows a single schematic block with a "x 94" bracket, mirroring
# PathoROB's own composition card. Occurrence, not tile: tiles recur across quartets.
TCGA2X2 = {
    "benchmark": "pathorob-tcga-2x2",
    "cell": 300,
    "quartets": 94,
    "total": 300 * 4 * 94,  # 112,800 occurrences
}


def render_tcga2x2(out_path: Path) -> Path:
    """Render the supplement's paired-2x2 schematic (generic 2x2 x 94 quartets)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plotstyle.apply_style()

    fig = plt.figure(figsize=(plotstyle.COL_ONEHALF, plotstyle.COL_ONEHALF * 0.72))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect("equal")
    ax.axis("off")

    rows = ("Class 1", "Class 2")
    cols = ("Centre 1", "Centre 2")
    left, top = 0.0, 2.0
    cell = f"{TCGA2X2['cell']:,}"

    ax.text(left + 1.0, top + 0.62, "TCGA (2×2)", ha="center", va="bottom",
            fontsize=plotstyle.FS_TITLE, weight="bold", color=plotstyle.TEXT_COLOR)
    ax.text(left + 1.0, top + 0.18, "pan-cancer · tile · paired protocol", ha="center",
            va="bottom", fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR)

    for r in range(2):
        cy = top - r - 0.5
        for c in range(2):
            ax.add_patch(Rectangle((left + c, top - r - 1), 1.0, 1.0,
                                   facecolor=CELL_FACE, edgecolor=CELL_EDGE, linewidth=LW_CELL))
            ax.text(left + c + 0.5, cy, cell, ha="center", va="center",
                    fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)
        ax.text(left - 0.22, cy, rows[r], ha="right", va="center",
                fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)
    for c in range(2):
        ax.text(left + c + 0.5, top - 2 - 0.18, cols[c], ha="center", va="top",
                fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR)

    # Right square bracket spanning the two rows, annotated "x 94 quartets".
    xb, ytop, ybot = left + 2 + 0.28, top, top - 2
    ax.plot([xb, xb], [ybot, ytop], color=plotstyle.MUTED_TEXT_COLOR, lw=1.0)
    ax.plot([xb - 0.14, xb], [ytop, ytop], color=plotstyle.MUTED_TEXT_COLOR, lw=1.0)
    ax.plot([xb - 0.14, xb], [ybot, ybot], color=plotstyle.MUTED_TEXT_COLOR, lw=1.0)
    ax.text(xb + 0.22, (ytop + ybot) / 2, f"× {TCGA2X2['quartets']}\nquartets", ha="left",
            va="center", fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR, linespacing=0.95)

    ax.set_xlim(-1.5, 3.9)
    ax.set_ylim(-0.6, 2.95)

    out_path = Path(out_path)
    (out_path.parent / "png").mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.parent / "png" / out_path.with_suffix(".png").name, dpi=plotstyle.DEFAULT_DPI)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Optional verification: the hard-coded counts must equal the evaluated runs.
# ---------------------------------------------------------------------------
def verify(specs: "tuple[CardSpec, ...]", *, repo: Path) -> list[str]:
    """Cross-check each spec against its run's ``per_sample_metrics.csv``.

    Returns a list of human-readable problems; empty means every spec matches (or the
    run is absent, which is reported, not failed -- output/ is gitignored).
    """
    import pandas as pd
    from paper_manifest import by_benchmark

    problems: list[str] = []
    for spec in specs:
        run = repo / by_benchmark(spec.benchmark).run_rel
        path = run / "results" / "per_sample_metrics.csv"
        if not path.exists():
            problems.append(f"{spec.name}: run not present ({path.relative_to(repo)}) -- not checked")
            continue
        df = pd.read_csv(path, usecols=["model", "label", "confounder", "sample_index"], dtype=str)
        one = df[df["model"] == df["model"].iloc[0]]
        cells = one.drop_duplicates(["label", "confounder", "sample_index"])
        counts = cells.groupby(["label", "confounder"]).size()
        n_rows, n_cols = one["label"].nunique(), one["confounder"].nunique()
        want_rows, want_cols = len(spec.rows), len(spec.cols)
        if (n_rows, n_cols) != (want_rows, want_cols):
            problems.append(
                f"{spec.name}: grid is {n_rows}x{n_cols} in the run but the spec is "
                f"{want_rows}x{want_cols}"
            )
        off = counts[counts != spec.cell]
        if not off.empty:
            problems.append(
                f"{spec.name}: {len(off)} cell(s) != {spec.cell:,} "
                f"(min {int(counts.min()):,}, max {int(counts.max()):,})"
            )
    return problems


def verify_tcga2x2(*, repo: Path) -> list[str]:
    """Cross-check the paired-2x2 schematic: 94 quartets, each 2x2 at 300 occurrences/cell."""
    import pandas as pd
    from paper_manifest import by_benchmark

    run = repo / by_benchmark(TCGA2X2["benchmark"]).run_rel
    path = run / "results" / "per_sample_metrics.csv"
    if not path.exists():
        return [f"TCGA (2x2): run not present ({path.relative_to(repo)}) -- not checked"]
    df = pd.read_csv(path, usecols=["model", "subset", "label", "confounder", "sample_index"], dtype=str)
    one = df[df["model"] == df["model"].iloc[0]]
    problems: list[str] = []
    if one["subset"].nunique() != TCGA2X2["quartets"]:
        problems.append(f"TCGA (2x2): {one['subset'].nunique()} quartets in the run, spec says {TCGA2X2['quartets']}")
    per_q = one.groupby("subset").agg(cls=("label", "nunique"), ctr=("confounder", "nunique"))
    if not ((per_q["cls"] == 2).all() and (per_q["ctr"] == 2).all()):
        problems.append("TCGA (2x2): some quartet is not exactly 2 classes x 2 centres")
    cell_counts = one.groupby(["subset", "label", "confounder"]).size()
    off = cell_counts[cell_counts != TCGA2X2["cell"]]
    if not off.empty:
        problems.append(f"TCGA (2x2): {len(off)} cell(s) != {TCGA2X2['cell']:,} occurrences")
    if len(one) != TCGA2X2["total"]:
        problems.append(f"TCGA (2x2): {len(one):,} occurrences in the run, spec says {TCGA2X2['total']:,}")
    return problems


# PCaBiop's ISUP-grading protocol (supplement). Unlike the binary PCaBiop card above, the label
# is the six-level ISUP grade group (0 benign, 1--5 increasingly aggressive), so the cohort is a
# real, named 6 (grade) x 2 (centre) grid at 250 slides/cell -- 3,000 slides. But cell balance
# alone does not balance the typed candidate pools in the full 6x2: for any anchor the same-grade
# other-centre (SO) pool holds 250 slides while the other-grade same-centre (OS) pool holds 1,250.
# So the six grades are decomposed into all C(6,2) = 15 pairs; crossing each pair with the two
# centres yields a balanced 2x2 quartet of 1,000 slides (250 SO + 250 OS per anchor). Each slide
# recurs across the five quartets that include its grade, so the typed evidence pools 15 x 1,000 =
# 15,000 occurrences. The card shows the real 6x2 grid with a bracket for the quartet decomposition.
PCABIOP_ISUP = {
    "benchmark": "panda-isup",
    "cell": 250,
    "grades": 6,  # ISUP 0..5
    "centres": 2,  # KI, RUMC
    "quartets": 15,  # C(6,2) grade pairs
    "slides": 250 * 6 * 2,  # 3,000 physical slides (the cohort)
    "occurrences": 250 * 4 * 15,  # 15,000 typed occurrences (15 quartets x 1,000)
}


def render_pcabiop_isup(out_path: Path) -> Path:
    """Render the supplement's ISUP-grading schematic (6x2 cohort -> 15 grade-pair quartets)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plotstyle.apply_style()

    n = PCABIOP_ISUP["grades"]  # rows
    fig = plt.figure(figsize=(plotstyle.COL_ONEHALF, plotstyle.COL_ONEHALF * 1.12))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect("equal")
    ax.axis("off")

    rows = tuple(f"ISUP {g}" for g in range(n))  # ISUP 0 (benign) .. ISUP 5
    cols = ("KI", "RUMC")
    left, top = 0.0, float(n)  # grid occupies y in [0, n], x in [0, 2]
    cell = f"{PCABIOP_ISUP['cell']:,}"

    ax.text(left + 1.0, top + 0.62, "PCaBiop-ISUP", ha="center", va="bottom",
            fontsize=plotstyle.FS_TITLE, weight="bold", color=plotstyle.TEXT_COLOR)
    ax.text(left + 1.0, top + 0.18, "prostate · slide · paired protocol", ha="center",
            va="bottom", fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR)

    for r in range(n):
        cy = top - r - 0.5
        for c in range(2):
            ax.add_patch(Rectangle((left + c, top - r - 1), 1.0, 1.0,
                                   facecolor=CELL_FACE, edgecolor=CELL_EDGE, linewidth=LW_CELL))
            ax.text(left + c + 0.5, cy, cell, ha="center", va="center",
                    fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)
        ax.text(left - 0.22, cy, rows[r], ha="right", va="center",
                fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR)
    for c in range(2):
        ax.text(left + c + 0.5, top - n - 0.18, cols[c], ha="center", va="top",
                fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR)

    # Right square bracket spanning the six grade rows, annotated with the quartet decomposition.
    xb, ytop, ybot = left + 2 + 0.28, top, top - n
    ax.plot([xb, xb], [ybot, ytop], color=plotstyle.MUTED_TEXT_COLOR, lw=1.0)
    ax.plot([xb - 0.14, xb], [ytop, ytop], color=plotstyle.MUTED_TEXT_COLOR, lw=1.0)
    ax.plot([xb - 0.14, xb], [ybot, ybot], color=plotstyle.MUTED_TEXT_COLOR, lw=1.0)
    ax.text(xb + 0.22, (ytop + ybot) / 2,
            f"→ {PCABIOP_ISUP['quartets']} grade-pair\nquartets", ha="left",
            va="center", fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR, linespacing=0.95)

    ax.set_xlim(-1.9, 5.3)
    ax.set_ylim(-0.6, top + 1.1)

    out_path = Path(out_path)
    (out_path.parent / "png").mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.parent / "png" / out_path.with_suffix(".png").name, dpi=plotstyle.DEFAULT_DPI)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def verify_pcabiop_isup(*, repo: Path) -> list[str]:
    """Cross-check the ISUP schematic: 15 grade-pair quartets, each 2x2 at 250 occurrences/cell."""
    import pandas as pd
    from paper_manifest import by_benchmark

    run = repo / by_benchmark(PCABIOP_ISUP["benchmark"]).run_rel
    path = run / "results" / "per_sample_metrics.csv"
    if not path.exists():
        return [f"PCaBiop-ISUP: run not present ({path.relative_to(repo)}) -- not checked"]
    df = pd.read_csv(path, usecols=["model", "subset", "label", "confounder", "sample_index"], dtype=str)
    one = df[df["model"] == df["model"].iloc[0]]
    problems: list[str] = []
    if one["subset"].nunique() != PCABIOP_ISUP["quartets"]:
        problems.append(
            f"PCaBiop-ISUP: {one['subset'].nunique()} quartets in the run, spec says {PCABIOP_ISUP['quartets']}"
        )
    if one["label"].nunique() != PCABIOP_ISUP["grades"]:
        problems.append(
            f"PCaBiop-ISUP: {one['label'].nunique()} grades in the run, spec says {PCABIOP_ISUP['grades']}"
        )
    per_q = one.groupby("subset").agg(cls=("label", "nunique"), ctr=("confounder", "nunique"))
    if not ((per_q["cls"] == 2).all() and (per_q["ctr"] == 2).all()):
        problems.append("PCaBiop-ISUP: some quartet is not exactly 2 grades x 2 centres")
    cell_counts = one.groupby(["subset", "label", "confounder"]).size()
    off = cell_counts[cell_counts != PCABIOP_ISUP["cell"]]
    if not off.empty:
        problems.append(f"PCaBiop-ISUP: {len(off)} cell(s) != {PCABIOP_ISUP['cell']:,} occurrences")
    if len(one) != PCABIOP_ISUP["occurrences"]:
        problems.append(
            f"PCaBiop-ISUP: {len(one):,} occurrences in the run, spec says {PCABIOP_ISUP['occurrences']:,}"
        )
    return problems


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "paper/figures/dataset_cardinality.pdf",
        help="Main-text card output PDF path (a PNG sibling is written under png/).",
    )
    parser.add_argument(
        "--tcga2x2-out",
        type=Path,
        default=REPO / "paper/figures/dataset_cardinality_tcga2x2.pdf",
        help="Supplement paired-2x2 schematic output PDF path.",
    )
    parser.add_argument(
        "--pcabiop-isup-out",
        type=Path,
        default=REPO / "paper/figures/dataset_cardinality_pcabiop_isup.pdf",
        help="Supplement PCaBiop-ISUP 6x2 schematic output PDF path.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Cross-check the hard-coded counts against the evaluated runs (needs output/).",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO, help="Repo root for run lookup.")
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = _parse_args(argv)
    repo = Path(args.repo_root)

    print("dataset cardinality card:")
    for spec in SPECS:
        print(
            f"  {spec.name:<14} {len(spec.rows)}x{len(spec.cols)} @ {spec.cell:,}/cell "
            f"= {spec.total:,} {spec.unit}"
        )
    print(
        f"  {'TCGA (2×2)':<14} 2x2 @ {TCGA2X2['cell']:,}/cell × {TCGA2X2['quartets']} quartets "
        f"= {TCGA2X2['total']:,} occurrences  [supp]"
    )
    print(
        f"  {'PCaBiop-ISUP':<14} 6x2 @ {PCABIOP_ISUP['cell']:,}/cell = {PCABIOP_ISUP['slides']:,} slides "
        f"→ {PCABIOP_ISUP['quartets']} quartets = {PCABIOP_ISUP['occurrences']:,} occurrences  [supp]"
    )

    if args.verify:
        problems = verify(SPECS, repo=repo) + verify_tcga2x2(repo=repo) + verify_pcabiop_isup(repo=repo)
        if problems:
            print("\nVERIFY:")
            for problem in problems:
                print(f"  {problem}")
            mismatches = [p for p in problems if "not present" not in p]
            if mismatches:
                print("\nspecs disagree with the runs; fix SPECS before rendering.")
                return 1
        else:
            print("\nVERIFY: every spec matches its evaluated run.")

    out = render(SPECS, Path(args.out))
    print(f"\nwrote {out} and {out.parent / 'png' / out.with_suffix('.png').name}")
    supp = render_tcga2x2(Path(args.tcga2x2_out))
    print(f"wrote {supp} and {supp.parent / 'png' / supp.with_suffix('.png').name}")
    isup = render_pcabiop_isup(Path(args.pcabiop_isup_out))
    print(f"wrote {isup} and {isup.parent / 'png' / isup.with_suffix('.png').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
