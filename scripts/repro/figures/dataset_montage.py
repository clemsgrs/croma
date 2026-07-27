"""Dataset example montage (PathoROB-style): biology x confounder, one block per dataset.

Supplementary figure for the paper. One block per tile benchmark (Camelyon, TCGA,
Tolkach-ESCA, prostate-shift), packed two per row into a wide, short composite;
each block shows two representative biological classes (rows) across *every* confounder level of that
benchmark (columns) -- so the reader sees biology-matched, confounder-different
tiles side by side and can judge how much the acquisition site changes the
picture. The block title is the dataset name, row labels are the biological
classes, and column labels are the confounder levels. Confounder width therefore
varies by benchmark (Camelyon 2 centres, TCGA 4, Tolkach 3, prostate 2), mirroring
the per-cell layout of ``fig:dataset-cardinality`` in
``paper/sections/dataset_summary.tex``.

The class/confounder grid per block is drawn from the manifest that benchmark's
main result is computed on, so the picture matches the reported configuration:

* Camelyon      -- native 2 centres: normal/tumour x RUMC/UMCU (the PathoROB
  in-domain RI set, ``pathorob-camelyon-faithful.csv``, evaluated dataset-wide).
* TCGA (4x4)    -- the headline TCGA configuration reported in the main text:
  four cancer types across four medical centres (``pathorob-tcga-4x4.csv``). We
  show two representative cancer types (breast/colon) across all four headline
  centres (Asterand, Christiana Healthcare, Roswell Park, University of
  Pittsburgh). PathoROB's paired TCGA-2x2 configuration is reported only in the
  supplement, so the example patches follow the 4x4 headline, not that pair.
* Tolkach-ESCA  -- the main result is dataset-wide over 6 classes x 3 cohorts; we
  show a documented representative contrast, Tumour vs Oesophageal mucosa, across
  all three cohorts (UKK, WNS, CHA).
* Prostate      -- native 2 providers: benign/tumour x Karolinska/Radboud
  (``prostate-shift-binary-kirumc.csv``, evaluated dataset-wide).

Tile selection is deterministic from a fixed seed via :func:`select_montage_tiles`
(a pure function over a manifest DataFrame). Actual image loading and PDF
rendering are guarded behind a data-availability check: the module imports and
the selection function runs without the (gitignored) tile images; the montage PDF
is produced only when the data tree is present.

Run:
    PYTHONPATH=src python scripts/repro/figures/dataset_montage.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root: scripts/repro/figures/dataset_montage.py -> parents[3].
REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# Identity only from croma.plotstyle (registers fonts + rcParams on import). We do
# NOT import scripts/bench/plotting.py so this figure stays self-contained.
from croma import plotstyle  # noqa: E402


# ---------------------------------------------------------------------------
# Pure, testable tile selection.
# ---------------------------------------------------------------------------
def _cell_rng(seed: int, klass: str, center: str) -> np.random.Generator:
    """Deterministic, per-cell RNG.

    The generator for a cell depends only on ``(seed, class, center)`` -- never on
    grid iteration order or on the other cells -- so the montage is stable and each
    cell is drawn independently. Python's built-in ``hash`` is salted per process,
    so we derive the seed from a stable SHA-256 digest instead.
    """
    key = f"{int(seed)}|{klass}|{center}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def select_montage_tiles(
    manifest: pd.DataFrame,
    *,
    classes: "list[str] | tuple[str, ...]",
    centers: "list[str] | tuple[str, ...]",
    seed: int,
    class_column: str = "label",
    center_column: str = "medical_center",
    path_column: str = "image_path",
) -> dict[tuple[str, str], str]:
    """Pick one representative tile per (class, center) cell, deterministically.

    Pure function over a manifest DataFrame: it never touches the filesystem, so it
    runs without the (gitignored) tile images. Returns a mapping from
    ``(class, center)`` to a single image path, covering the full ``classes x
    centers`` grid. Selection is deterministic in ``seed`` -- the same seed always
    yields the same tiles.

    Raises ``ValueError`` if a required column is missing, if fewer than two distinct
    classes/centers are requested, or if any requested cell has no candidate tiles
    (i.e. the requested subset is not a complete grid).
    """
    class_values = [str(value) for value in classes]
    center_values = [str(value) for value in centers]
    if len(set(class_values)) < 2:
        raise ValueError("montage requires at least 2 distinct classes")
    if len(set(center_values)) < 2:
        raise ValueError("montage requires at least 2 distinct centers")
    for column in (class_column, center_column, path_column):
        if column not in manifest.columns:
            raise ValueError(f"manifest is missing required column: {column!r}")

    col_class = manifest[class_column].astype(str)
    col_center = manifest[center_column].astype(str)
    col_path = manifest[path_column].astype(str)

    selected: dict[tuple[str, str], str] = {}
    for klass, center in product(class_values, center_values):
        mask = (col_class == klass) & (col_center == center)
        candidates = sorted(col_path[mask].unique().tolist())
        if not candidates:
            raise ValueError(
                f"no tiles for cell (class={klass!r}, center={center!r}); "
                "the requested class/center pair is not a complete grid"
            )
        index = int(_cell_rng(seed, klass, center).integers(len(candidates)))
        selected[(klass, center)] = candidates[index]
    return selected


# ---------------------------------------------------------------------------
# Per-dataset montage specification.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MontageSpec:
    """One dataset block: which manifest, which biology/confounder grid, and labels.

    ``classes`` is always the two representative biology rows; ``centers`` is that
    benchmark's confounder levels (columns) and its length varies by benchmark.
    """

    name: str
    manifest: str  # repo-relative path to the dataset's main-result manifest
    classes: tuple[str, str]  # representative biology pair (rows)
    centers: tuple[str, ...]  # confounder levels (columns); length varies by benchmark
    class_column: str = "label"
    center_column: str = "medical_center"
    path_column: str = "image_path"
    subset: "str | None" = None  # restrict to this manifest 'subset' before selecting
    class_labels: dict[str, str] = field(default_factory=dict)
    center_labels: dict[str, str] = field(default_factory=dict)
    note: str = ""  # provenance of the pair (canonical vs representative/placeholder)

    def class_label(self, value: str) -> str:
        return self.class_labels.get(value, str(value))

    def center_label(self, value: str) -> str:
        return self.center_labels.get(value, str(value))

    def filtered(self, manifest: pd.DataFrame) -> pd.DataFrame:
        """Restrict a raw manifest to this block's subset / class / center grid."""
        working = manifest
        if self.subset is not None:
            if "subset" not in working.columns:
                raise ValueError(
                    f"{self.name}: manifest has no 'subset' column but subset "
                    f"{self.subset!r} was requested"
                )
            working = working[working["subset"].astype(str) == str(self.subset)]
        working = working[
            working[self.class_column].astype(str).isin(self.classes)
            & working[self.center_column].astype(str).isin(self.centers)
        ]
        return working.reset_index(drop=True)


# The four tile benchmarks, each shown as two biology rows across all of its
# confounder levels. Camelyon and prostate are native two-level confounders; TCGA
# uses the 4x4 headline configuration (four centres); Tolkach uses a documented
# representative class contrast across all three cohorts (see module docstring).
SPECS: tuple[MontageSpec, ...] = (
    MontageSpec(
        name="Camelyon",
        manifest="data/pathorob/manifests/pathorob-camelyon-faithful.csv",
        classes=("normal", "tumor"),
        centers=("RUMC", "UMCU"),
        center_column="medical_center",
        class_labels={"normal": "Normal", "tumor": "Tumour"},
        center_labels={"RUMC": "RUMC", "UMCU": "UMCU"},
        note="native 2 centres (PathoROB in-domain RI set)",
    ),
    MontageSpec(
        name=r"TCGA (4$\times$4)",
        manifest="data/pathorob/manifests/pathorob-tcga-4x4.csv",
        classes=("Breast_invasive_carcinoma", "Colon_adenocarcinoma"),
        centers=(
            "Asterand",
            "Christiana Healthcare",
            "Roswell Park",
            "University of Pittsburgh",
        ),
        center_column="medical_center",
        class_labels={
            "Breast_invasive_carcinoma": "Breast IC",
            "Colon_adenocarcinoma": "Colon AC",
        },
        center_labels={
            "Asterand": "Asterand",
            "Christiana Healthcare": "Christiana",
            "Roswell Park": "Roswell",
            "University of Pittsburgh": "U. Pitt",
        },
        note="headline 4x4: 2 representative cancer types x the 4 headline centres",
    ),
    MontageSpec(
        name="Tolkach-ESCA",
        manifest="data/pathorob/manifests/pathorob-tolkach-esca-faithful.csv",
        classes=("TUMOR", "SH_OES"),
        centers=("VALSET1_UKK", "VALSET2_WNS", "VALSET4_CHA_FULL"),
        center_column="medical_center",
        class_labels={"TUMOR": "Tumour", "SH_OES": "Mucosa"},
        center_labels={
            "VALSET1_UKK": "UKK",
            "VALSET2_WNS": "WNS",
            "VALSET4_CHA_FULL": "CHA",
        },
        note="representative class contrast (Tumour vs mucosa) across all 3 cohorts",
    ),
    MontageSpec(
        name="PCaBiop",
        manifest="data/prostate-shift-binary-kirumc.csv",
        classes=("benign", "tumor"),
        centers=("KI", "RUMC"),
        center_column="medical_center",
        class_labels={"benign": "Benign", "tumor": "Cancer"},
        center_labels={"KI": "KI", "RUMC": "RUMC"},
        note="native 2 providers (prostate-shift binary, dataset-wide)",
    ),
)


# ---------------------------------------------------------------------------
# Data-gated collection + rendering.
# ---------------------------------------------------------------------------
@dataclass
class MontageBlock:
    spec: MontageSpec
    tiles: dict[tuple[str, str], str]
    missing_images: list[str]


def collect_blocks(
    specs: "tuple[MontageSpec, ...] | list[MontageSpec]",
    *,
    repo: Path,
    seed: int,
) -> tuple[list[MontageBlock], list[tuple[str, str]]]:
    """Load manifests, run the pure selection, and report data availability.

    Returns ``(blocks, problems)``. ``problems`` collects datasets whose manifest is
    absent or whose canonical grid could not be formed; those never reach the render
    path. Present blocks record any missing image files so the caller can decide
    whether the montage PDF can be drawn.
    """
    blocks: list[MontageBlock] = []
    problems: list[tuple[str, str]] = []
    for spec in specs:
        manifest_path = repo / spec.manifest
        if not manifest_path.exists():
            problems.append((spec.name, f"manifest not found: {manifest_path}"))
            continue
        raw = pd.read_csv(manifest_path, dtype=str)
        try:
            working = spec.filtered(raw)
            tiles = select_montage_tiles(
                working,
                classes=spec.classes,
                centers=spec.centers,
                seed=seed,
                class_column=spec.class_column,
                center_column=spec.center_column,
                path_column=spec.path_column,
            )
        except ValueError as error:
            problems.append((spec.name, str(error)))
            continue
        missing = [path for path in tiles.values() if not Path(path).exists()]
        blocks.append(MontageBlock(spec=spec, tiles=tiles, missing_images=missing))
    return blocks, problems


def _read_tile(path: str):
    """Load a tile as an RGB(A) array (guarded import so the module stays light)."""
    import matplotlib.image as mpimg

    return mpimg.imread(path)


def render_montage(blocks: "list[MontageBlock]", out_path: Path) -> Path:
    """Render the composite montage (one biology x confounder block per dataset).

    Blocks are packed two per row into a wide, short composite (rather than a tall
    single column, which does not sit well on a manuscript page). Layout is in
    fixed inches: every tile is the same square regardless of how many confounder
    columns its benchmark has, and each block is only as wide as its own confounder
    count. Within a row, blocks are packed left-to-right, so a benchmark with fewer
    confounder levels leaves no dead columns -- the next block simply starts sooner.
    Each block reserves a top band for its title + column labels and a left strip
    for the class (row) labels.

    Requires every block's tiles to exist on disk; callers guard this behind a
    data-availability check. Writes a flat PDF at ``out_path`` (for the paper's
    ``\\graphicspath``) plus a PNG sibling under ``png/``.
    """
    import matplotlib.pyplot as plt

    plotstyle.apply_style()

    n = len(blocks)
    if n == 0:
        raise ValueError("no blocks to render")
    n_class = max(len(block.spec.classes) for block in blocks)

    # Layout constants (inches).
    tile = 0.95  # tile edge (square)
    gap = 0.03  # gap between adjacent tiles
    lab_w = 0.42  # left strip per block for the rotated class labels
    title_h = 0.34  # per-block band for the cohort title + column labels
    block_gap_x = 0.50  # horizontal space between blocks in a row
    block_gap_y = 0.35  # vertical space between block rows
    margin = 0.10  # outer figure margin

    def block_width(spec: MontageSpec) -> float:
        c = len(spec.centers)
        return lab_w + c * tile + (c - 1) * gap

    block_cols = 2 if n > 1 else 1
    block_rows = (n + block_cols - 1) // block_cols
    block_row_h = title_h + n_class * tile + (n_class - 1) * gap

    # Figure size: width from the widest packed row, height from the block rows.
    row_width = [0.0] * block_rows
    for i, block in enumerate(blocks):
        row_width[i // block_cols] += block_width(block.spec)
    for r in range(block_rows):
        count = min(block_cols, n - r * block_cols)
        row_width[r] += block_gap_x * (count - 1)
    fig_w = margin + max(row_width) + margin
    fig_h = margin + block_rows * block_row_h + block_gap_y * (block_rows - 1) + margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    x_cursor = [margin] * block_rows  # running left edge for each row
    for i, block in enumerate(blocks):
        spec = block.spec
        r = i // block_cols
        bx = x_cursor[r]
        by_top = margin + r * (block_row_h + block_gap_y)  # block top, from fig top
        x_cursor[r] = bx + block_width(spec) + block_gap_x
        tiles_x0 = bx + lab_w
        tiles_y0_top = by_top + title_h
        for row, klass in enumerate(spec.classes):
            for col, center in enumerate(spec.centers):
                ax = fig.add_axes(
                    [
                        (tiles_x0 + col * (tile + gap)) / fig_w,
                        (fig_h - (tiles_y0_top + row * (tile + gap)) - tile) / fig_h,
                        tile / fig_w,
                        tile / fig_h,
                    ]
                )
                ax.imshow(_read_tile(block.tiles[(klass, center)]))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(plotstyle.SPINE_COLOR)
                    spine.set_linewidth(plotstyle.LW_SPINE)
                if row == 0:  # columns = confounder level
                    ax.set_title(
                        spec.center_label(center),
                        fontsize=plotstyle.FS_ANNOT,
                        color=plotstyle.MUTED_TEXT_COLOR,
                        pad=3,
                    )
                if col == 0:  # rows = biology class
                    ax.set_ylabel(
                        spec.class_label(klass),
                        fontsize=plotstyle.FS_ANNOT,
                        color=plotstyle.TEXT_COLOR,
                    )
        # Cohort title, centred over this block's tiles in the reserved top band.
        block_tiles_w = len(spec.centers) * tile + (len(spec.centers) - 1) * gap
        fig.text(
            (tiles_x0 + block_tiles_w / 2) / fig_w,
            (fig_h - by_top - 0.02) / fig_h,
            spec.name,
            ha="center",
            va="top",
            fontsize=plotstyle.FS_TITLE,
            weight="bold",
            color=plotstyle.TEXT_COLOR,
        )

    out_path = Path(out_path)
    (out_path.parent / "png").mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path.parent / "png" / out_path.with_suffix(".png").name,
        dpi=plotstyle.DEFAULT_DPI,
    )
    fig.savefig(out_path)  # flat pdf for \graphicspath
    plt.close(fig)
    return out_path


def _parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=0, help="Deterministic tile-selection seed."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "paper/figures/dataset_montage.pdf",
        help="Output PDF path (a PNG sibling is written under png/).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO,
        help="Repo root that manifest paths are resolved against.",
    )
    return parser.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = _parse_args(argv)
    repo = Path(args.repo_root)
    blocks, problems = collect_blocks(SPECS, repo=repo, seed=int(args.seed))

    print(f"dataset montage: seed={int(args.seed)}")
    for block in blocks:
        spec = block.spec
        print(f"\n[{spec.name}]  {spec.note}")
        for (klass, center), path in block.tiles.items():
            flag = "" if Path(path).exists() else "  (image missing)"
            print(
                f"  {spec.class_label(klass):<20} x {spec.center_label(center):<14} "
                f"-> {path}{flag}"
            )
    for name, reason in problems:
        print(f"\n[{name}]  SKIPPED: {reason}")

    renderable = [block for block in blocks if not block.missing_images]
    if problems or len(renderable) != len(SPECS):
        print(
            "\nTile images and/or manifests are not all present in this checkout; "
            "the selection ran but the montage PDF was not rendered.\n"
            "Run where the (gitignored) data/ tree is available to produce "
            f"{args.out}."
        )
        return 0

    out = render_montage(renderable, Path(args.out))
    print(f"\nwrote {out} and {out.parent / 'png' / out.with_suffix('.png').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
