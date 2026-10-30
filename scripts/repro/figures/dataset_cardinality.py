"""Dataset-cardinality figures (PathoROB card-style): slides and tiles per class x confounder.

Renders ``fig:dataset-cardinality`` (``paper/sections/dataset_summary.tex``): one block per
evaluation benchmark, a grid of light cells with the biological classes as rows and the
confounder levels as columns. Each cell states its slides over its tiles -- "17 slides" over
"5,100 tiles" on Camelyon. Tolkach-ESCA counts cases, because a case there contributes tiles
to several classes. PCaBiop is evaluated per slide, so its cells state slides only.

Two paired-protocol schematics for the supplement are rendered beside it: TCGA-2x2's generic
two-class x two-centre quartet (x its quartet count), and PCaBiop-ISUP's six-grade x two-centre
cohort, which the protocol decomposes into grade-pair quartets.

Every count is read from the evaluated rows, ``views.load_view(benchmark).eval_manifest``, by
``scripts/repro/_cohort_counts.py``: a cell's slides are its distinct ``group_id`` values and
its tiles are its rows. The caption macros (``_panel_macros.cardinality_macros``) read the same
function. ``--verify`` cross-checks every cell against its run's ``per_sample_metrics.csv``.

The figures render beside other standalone studies, under
``output/studies/dataset-cardinality/plots/{pdf,png}/``, and never into ``paper/figures/``;
``check_paper_figures.py`` reports when the paper's copy falls behind.

Run:
    PYTHONPATH=src python scripts/repro/figures/dataset_cardinality.py [--verify]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Repo root: scripts/repro/figures/dataset_cardinality.py -> parents[3].
REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
# _cohort_counts and paper_manifest live in scripts/repro/; views and plotting in scripts/bench/.
for _path in (REPO / "scripts" / "repro", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import _cohort_counts as counts  # noqa: E402
from plotting import style as plotstyle  # noqa: E402

#: Beside the other standalone studies, under ``plots/{pdf,png}/``.
PLOTS = REPO / "output" / "studies" / "dataset-cardinality" / "plots"
MAIN_NAME = "dataset_cardinality"
TCGA2X2_NAME = "dataset_cardinality_tcga2x2"
PCABIOP_ISUP_NAME = "dataset_cardinality_pcabiop_isup"

TCGA2X2_BENCHMARK = "pathorob-tcga-2x2"
PCABIOP_ISUP_BENCHMARK = "panda-isup"

# Cell chrome. A single neutral fill for every cell: the benchmarks are balanced, so shade
# carries no information -- the count text does. Matches the card's look.
CELL_FACE = "#eaeaea"
CELL_EDGE = "#c3c3c3"
LW_CELL = 0.6
#: A cell is ``CELL_W`` wide and one unit tall: wide enough for "5,100 tiles".
CELL_W = 1.5
FS_CELL = plotstyle.FS_ANNOT - 0.5


@dataclass(frozen=True)
class CardSpec:
    """One dataset block: which manifest labels and confounders it shows, and how.

    ``rows`` and ``cols`` pair each manifest value with its display name (top to bottom, left
    to right). ``group`` names the manifest's ``group_id`` unit; ``tile_level`` is False for a
    slide-level benchmark, whose cells then state slides only.
    """

    name: str
    subtitle: str  # muted one-liner: organ . level
    rows: tuple[tuple[str, str], ...]
    cols: tuple[tuple[str, str], ...]
    benchmark: str  # paper_manifest benchmark key
    group: str = "slides"
    tile_level: bool = True


@dataclass(frozen=True)
class Card:
    """A spec with its counted cells: the text lines of each cell and the cohort total."""

    spec: CardSpec
    cells: dict[tuple[str, str], tuple[str, ...]]
    total: str


# The four evaluation benchmarks, in the order of the dataset-summary table.
SPECS: tuple[CardSpec, ...] = (
    CardSpec(
        name="Camelyon",
        subtitle="breast LN · tile",
        rows=(("normal", "Normal"), ("tumor", "Tumour")),
        cols=(("RUMC", "RUMC"), ("UMCU", "UMCU")),
        benchmark="pathorob-camelyon",
    ),
    CardSpec(
        name="TCGA (4×4)",
        subtitle="pan-cancer · tile",
        rows=(
            ("Breast_invasive_carcinoma", "BRCA"),
            ("Colon_adenocarcinoma", "COAD"),
            ("Lung_adenocarcinoma", "LUAD"),
            ("Lung_squamous_cell_carcinoma", "LUSC"),
        ),
        cols=(
            ("Asterand", "Asterand"),
            ("Christiana Healthcare", "Christiana"),
            ("Roswell Park", "Roswell Park"),
            ("University of Pittsburgh", "U. Pittsburgh"),
        ),
        benchmark="pathorob-tcga-4x4",
    ),
    CardSpec(
        name="Tolkach-ESCA",
        subtitle="oesophagus · tile",
        rows=(
            ("TUMOR", "Tumour"),
            ("REGR_TU", "Regression\ntissue"),
            ("SH_OES", "Oesophageal\nmucosa"),
            ("SH_MAG", "Gastric\nmucosa"),
            ("MUSC_PROP", "Muscularis\npropria"),
            ("ADVENT", "Adventitial\ntissue"),
        ),
        cols=(("VALSET1_UKK", "UKK"), ("VALSET2_WNS", "WNS"), ("VALSET4_CHA_FULL", "CHA")),
        benchmark="pathorob-tolkach-esca",
        group="cases",
    ),
    CardSpec(
        name="PCaBiop",
        subtitle="prostate · slide",
        rows=(("0", "Benign"), ("1", "Cancer")),
        cols=(("karolinska", "KI"), ("radboud", "RUMC")),
        benchmark="panda",
        tile_level=False,
    ),
)

#: PCaBiop-ISUP's real six-grade x two-centre grid (grade 0 benign, 1--5 increasingly aggressive).
PCABIOP_ISUP = CardSpec(
    name="PCaBiop-ISUP",
    subtitle="prostate · slide · paired protocol",
    rows=tuple((str(g), f"ISUP {g}") for g in range(6)),
    cols=(("karolinska", "KI"), ("radboud", "RUMC")),
    benchmark=PCABIOP_ISUP_BENCHMARK,
    tile_level=False,
)


def _count(n: int, unit: str) -> str:
    return f"{n:,} {unit if n != 1 else unit[:-1]}"


def _lines(cell: counts.Cell, spec: CardSpec) -> tuple[str, ...]:
    slides = _count(cell.slides, spec.group)
    return (slides, _count(cell.tiles, "tiles")) if spec.tile_level else (slides,)


def card(spec: CardSpec, manifest: pd.DataFrame) -> Card:
    """Count every cell of ``spec`` from ``manifest``; its grid must be exactly the spec's."""
    by_cell = counts.cells(manifest)
    want = {(row, col) for row, _ in spec.rows for col, _ in spec.cols}
    if set(by_cell) != want:
        missing = sorted(want - set(by_cell))
        unexpected = sorted(set(by_cell) - want)
        raise ValueError(
            f"{spec.name}: the view's grid is not the card's -- "
            f"missing {missing}, unexpected {unexpected}"
        )
    cohort = counts.Cell(slides=int(manifest["group_id"].nunique()), tiles=len(manifest))
    return Card(
        spec=spec,
        cells={key: _lines(by_cell[key], spec) for key in sorted(want)},
        total=" · ".join(_lines(cohort, spec)),
    )


@dataclass(frozen=True)
class QuartetCard:
    """TCGA-2x2's generic quartet: one cell's text, the quartet count and the pooled range."""

    cell: tuple[str, ...]
    n_quartets: int
    pooled_slides: tuple[int, int]  # min, max distinct slides of a class x centre, pooled


def quartet_card(manifest: pd.DataFrame) -> QuartetCard:
    """Count the paired cohort: every quartet cell must hold the same slides and tiles."""
    balance = counts.balance(manifest, counts.QUARTET_CELL)
    pooled = [cell.slides for cell in counts.cells(manifest).values()]
    return QuartetCard(
        cell=(_count(balance.cell.slides, "slides"), _count(balance.cell.tiles, "tiles")),
        n_quartets=counts.paired_composition(manifest).n_quartets,
        pooled_slides=(min(pooled), max(pooled)),
    )


# ---------------------------------------------------------------------------
# Drawing. Blocks live on one axes in cell units (a cell is CELL_W x 1) so every cell is the
# same physical size across datasets, as in the card. y increases upward; each block is
# placed by the y of its top edge and the x of its left edge.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Placement:
    left: float
    top: float


TOP = 6.0  # top-align Camelyon, TCGA and Tolkach along this edge
PLACEMENTS: dict[str, Placement] = {
    "Camelyon": Placement(left=0.0, top=TOP),
    "PCaBiop": Placement(left=0.0, top=TOP - 4.6),  # stacked under Camelyon, clear of its labels
    "TCGA (4×4)": Placement(left=CELL_W * 2 + 2.2, top=TOP),
    "Tolkach-ESCA": Placement(left=CELL_W * 6 + 4.6, top=TOP),
}


def _cell(ax, x: float, y_top: float, lines: tuple[str, ...]) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (x, y_top - 1), CELL_W, 1.0,
            facecolor=CELL_FACE, edgecolor=CELL_EDGE, linewidth=LW_CELL,
        )
    )
    # The slide (or case) count leads; the tile count sits beneath it, muted.
    offsets = (0.0,) if len(lines) == 1 else (0.17, -0.17)
    colours = (plotstyle.TEXT_COLOR, plotstyle.MUTED_TEXT_COLOR)
    for line, dy, colour in zip(lines, offsets, colours):
        ax.text(
            x + CELL_W / 2, y_top - 0.5 + dy, line,
            ha="center", va="center", fontsize=FS_CELL, color=colour,
        )


def _title(ax, mid_x: float, top: float, title: str, subtitle: str) -> None:
    ax.text(
        mid_x, top + 0.62, title, ha="center", va="bottom",
        fontsize=plotstyle.FS_TITLE, weight="bold", color=plotstyle.TEXT_COLOR,
    )
    ax.text(
        mid_x, top + 0.18, subtitle, ha="center", va="bottom",
        fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR,
    )


def _draw_block(ax, block: Card, place: Placement, *, rotate_cols: bool = True) -> None:
    spec = block.spec
    left, top = place.left, place.top
    _title(ax, left + len(spec.cols) * CELL_W / 2, top, spec.name, f"{spec.subtitle} · {block.total}")
    for r, (row, row_name) in enumerate(spec.rows):
        for c, (col, _) in enumerate(spec.cols):
            _cell(ax, left + c * CELL_W, top - r, block.cells[(row, col)])
        # Row label (biological class), right-aligned in the left gutter.
        ax.text(
            left - 0.22, top - r - 0.5, row_name, ha="right", va="center",
            fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR, linespacing=0.9,
        )
    # Column labels (confounder levels) beneath the bottom row.
    bottom = top - len(spec.rows)
    for c, (_, col_name) in enumerate(spec.cols):
        x = left + c * CELL_W + CELL_W / 2
        if rotate_cols:
            ax.text(
                x, bottom - 0.18, col_name, ha="right", va="top", rotation=32,
                rotation_mode="anchor", fontsize=plotstyle.FS_ANNOT,
                color=plotstyle.MUTED_TEXT_COLOR,
            )
        else:
            ax.text(
                x, bottom - 0.18, col_name, ha="center", va="top",
                fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR,
            )


def _bracket(ax, x: float, y_top: float, y_bottom: float, label: str) -> None:
    """A right square bracket spanning the rows, annotated with ``label``."""
    colour = plotstyle.MUTED_TEXT_COLOR
    ax.plot([x, x], [y_bottom, y_top], color=colour, lw=1.0)
    ax.plot([x - 0.14, x], [y_top, y_top], color=colour, lw=1.0)
    ax.plot([x - 0.14, x], [y_bottom, y_bottom], color=colour, lw=1.0)
    ax.text(
        x + 0.22, (y_top + y_bottom) / 2, label, ha="left", va="center",
        fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR, linespacing=0.95,
    )


def _new_figure(width: float, height: float):
    import matplotlib.pyplot as plt

    plotstyle.apply_style()
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _save(fig, plots: Path, name: str) -> Path:
    """Write ``plots/pdf/<name>.pdf`` and its ``plots/png/`` sibling; return the PDF."""
    import matplotlib.pyplot as plt

    pdf = Path(plots) / "pdf" / f"{name}.pdf"
    png = Path(plots) / "png" / f"{name}.png"
    for path in (pdf, png):
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=plotstyle.DEFAULT_DPI)
    fig.savefig(pdf)
    plt.close(fig)
    return pdf


def render(cards: "list[Card]", plots: Path = PLOTS) -> Path:
    """Render the main composition card from the four benchmarks' counted cells."""
    fig, ax = _new_figure(plotstyle.COL_DOUBLE, plotstyle.COL_DOUBLE * 0.42)
    for block in cards:
        _draw_block(ax, block, PLACEMENTS[block.spec.name])
    ax.set_xlim(-2.6, CELL_W * 9 + 5.2)
    ax.set_ylim(-1.75, 7.2)
    return _save(fig, plots, MAIN_NAME)


def render_tcga2x2(quartet: QuartetCard, plots: Path = PLOTS) -> Path:
    """Render the paired-2x2 schematic: one generic quartet x the quartet count."""
    fig, ax = _new_figure(plotstyle.COL_ONEHALF, plotstyle.COL_ONEHALF * 0.66)
    rows, cols = ("Class 1", "Class 2"), ("Centre 1", "Centre 2")
    top = 2.0
    _title(ax, CELL_W, top, "TCGA (2×2)", "pan-cancer · tile · paired protocol")
    for r, row_name in enumerate(rows):
        for c in range(2):
            _cell(ax, c * CELL_W, top - r, quartet.cell)
        ax.text(
            -0.22, top - r - 0.5, row_name, ha="right", va="center",
            fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR,
        )
    for c, col_name in enumerate(cols):
        ax.text(
            c * CELL_W + CELL_W / 2, top - 2 - 0.18, col_name, ha="center", va="top",
            fontsize=plotstyle.FS_ANNOT, color=plotstyle.MUTED_TEXT_COLOR,
        )
    _bracket(ax, 2 * CELL_W + 0.28, top, top - 2, f"× {quartet.n_quartets}\nquartets")
    low, high = quartet.pooled_slides
    ax.text(
        CELL_W, top - 2 - 0.62, f"pooled over quartets, a class × centre holds {low}–{high} slides",
        ha="center", va="top", fontsize=plotstyle.FS_ANNOT - 0.5,
        color=plotstyle.MUTED_TEXT_COLOR,
    )
    ax.set_xlim(-1.5, 2 * CELL_W + 1.9)
    ax.set_ylim(-1.0, 2.95)
    return _save(fig, plots, TCGA2X2_NAME)


def render_pcabiop_isup(block: Card, n_quartets: int, plots: Path = PLOTS) -> Path:
    """Render the ISUP schematic: the six-grade x two-centre cohort -> grade-pair quartets."""
    n = len(block.spec.rows)
    fig, ax = _new_figure(plotstyle.COL_ONEHALF, plotstyle.COL_ONEHALF * 1.12)
    _draw_block(ax, block, Placement(left=0.0, top=float(n)), rotate_cols=False)
    _bracket(ax, 2 * CELL_W + 0.28, float(n), 0.0, f"→ {n_quartets} grade-pair\nquartets")
    ax.set_xlim(-1.9, 2 * CELL_W + 3.0)
    ax.set_ylim(-0.6, n + 1.1)
    return _save(fig, plots, PCABIOP_ISUP_NAME)


# ---------------------------------------------------------------------------
# Verification: every cell of the view must match the evaluated run, slide for slide.
# ---------------------------------------------------------------------------
def verify_counts(
    name: str, manifest: pd.DataFrame, run: pd.DataFrame, keys: tuple[str, ...] = counts.CELL
) -> list[str]:
    """Problems where ``run`` (one model's per-sample rows) and ``manifest`` count differently."""
    view_cells = counts.cells(manifest.astype({k: str for k in keys}), keys)
    run_cells = counts.cells(run.astype({k: str for k in keys}), keys)
    problems = []
    for key in sorted(set(view_cells) | set(run_cells)):
        view, scored = view_cells.get(key), run_cells.get(key)
        if view != scored:
            problems.append(f"{name}: cell {key} is {view} in the view but {scored} in the run")
    return problems


def _run_frame(benchmark: str, repo: Path) -> pd.DataFrame | None:
    from paper_manifest import by_benchmark

    path = Path(repo) / by_benchmark(benchmark).per_sample_rel
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype=str)
    return frame[frame["model"] == frame["model"].iloc[0]]


def verify(manifests: dict[str, pd.DataFrame], *, repo: Path = REPO) -> list[str]:
    """Cross-check every benchmark's view against its run; an absent run is reported."""
    problems: list[str] = []
    for benchmark, manifest in manifests.items():
        run = _run_frame(benchmark, repo)
        if run is None:
            problems.append(f"{benchmark}: run not present -- not checked")
            continue
        keys = counts.QUARTET_CELL if "subset" in manifest else counts.CELL
        problems += verify_counts(benchmark, manifest, run, keys)
    return problems


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=PLOTS,
        help="Directory to write pdf/ and png/ into (default: %(default)s).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Cross-check every cell against the evaluated runs before rendering.",
    )
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    import views

    args = _parse_args(argv)
    benchmarks = [spec.benchmark for spec in SPECS] + [TCGA2X2_BENCHMARK, PCABIOP_ISUP_BENCHMARK]
    manifests = {b: views.load_view(b).eval_manifest for b in benchmarks}

    cards = [card(spec, manifests[spec.benchmark]) for spec in SPECS]
    quartet = quartet_card(manifests[TCGA2X2_BENCHMARK])
    isup = manifests[PCABIOP_ISUP_BENCHMARK]
    isup_card = card(PCABIOP_ISUP, isup)
    isup_quartets = counts.paired_composition(isup).n_quartets

    print("dataset cardinality card:")
    for block in [*cards, isup_card]:
        cell = next(iter(block.cells.values()))
        print(f"  {block.spec.name:<14} {' / '.join(cell)} per cell = {block.total}")
    print(f"  {'TCGA (2×2)':<14} {' / '.join(quartet.cell)} per cell x {quartet.n_quartets} quartets")

    if args.verify:
        problems = verify(manifests)
        print("\nVERIFY:" if problems else "\nVERIFY: every cell matches its evaluated run.")
        for problem in problems:
            print(f"  {problem}")
        if any("not present" not in p for p in problems):
            return 1

    for pdf in (
        render(cards, args.plots_dir),
        render_tcga2x2(quartet, args.plots_dir),
        render_pcabiop_isup(isup_card, isup_quartets, args.plots_dir),
    ):
        print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
