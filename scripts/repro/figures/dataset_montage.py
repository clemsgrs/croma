"""Dataset example montage (PathoROB-style): biology x center, one block per dataset.

Composite figure for the paper (Plot 2). One small block per tile benchmark
(Camelyon, TCGA, Tolkach-ESCA, prostate-shift); each block is a 2x2 grid of real
tiles with rows = biology class and columns = center (the confounder). The block
title is the dataset name, row labels are the biological classes, and column
labels are the centers -- mirroring the subcaption composite described in the
paper and the cell layout of ``fig:dataset-cardinality`` in
``paper/sections/dataset_summary.tex``.

The class/center pair per block is the canonical paired subset that dataset's
main result is computed on, so the picture matches the reported numbers:

* Camelyon      -- native 2x2: normal/tumour x RUMC/UMCU (the PathoROB in-domain
  RI set, ``pathorob-camelyon-faithful.csv``, evaluated dataset-wide).
* Prostate      -- native 2x2: benign/tumour x Karolinska/Radboud
  (``prostate-shift-binary-kirumc.csv``, evaluated dataset-wide).
* TCGA (2x2)    -- the main result averages 94 balanced two-class x two-center
  quartets, so there is no single canonical pair; we show one representative
  quartet (the alphabetically-first subset, ``BLCA_BRCA``), matching the paper's
  "one representative quartet is shown".
* Tolkach-ESCA  -- the main result is dataset-wide over 6 classes x 3 cohorts, so
  there is no canonical 2x2 pair; we use a documented placeholder contrast,
  Tumour vs Oesophageal mucosa across the UKK and WNS cohorts.

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
    """One dataset block: which manifest, which canonical 2x2, and pretty labels."""

    name: str
    manifest: str  # repo-relative path to the dataset's main-result manifest
    classes: tuple[str, str]  # canonical biology pair (rows)
    centers: tuple[str, str]  # canonical center pair (columns)
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


# The four tile benchmarks. Camelyon and prostate are native 2x2 (canonical pair
# fully determined). TCGA shows one representative quartet; Tolkach uses a
# documented placeholder contrast (see module docstring).
SPECS: tuple[MontageSpec, ...] = (
    MontageSpec(
        name="Camelyon",
        manifest="data/pathorob/manifests/pathorob-camelyon-faithful.csv",
        classes=("normal", "tumor"),
        centers=("RUMC", "UMCU"),
        center_column="medical_center",
        class_labels={"normal": "Normal", "tumor": "Tumour"},
        center_labels={"RUMC": "RUMC", "UMCU": "UMCU"},
        note="canonical native 2x2 (PathoROB in-domain RI set)",
    ),
    MontageSpec(
        name="TCGA (2x2)",
        manifest="data/pathorob/manifests/pathorob-tcga-2x2.csv",
        classes=("Bladder_Urothelial_Carcinoma", "Breast_invasive_carcinoma"),
        centers=("MD Anderson", "University of Pittsburgh"),
        center_column="medical_center",
        subset="BLCA_BRCA",
        class_labels={
            "Bladder_Urothelial_Carcinoma": "Bladder UC",
            "Breast_invasive_carcinoma": "Breast IC",
        },
        center_labels={
            "MD Anderson": "MD Anderson",
            "University of Pittsburgh": "U. Pittsburgh",
        },
        note="representative quartet (subset BLCA_BRCA); main result averages 94 quartets",
    ),
    MontageSpec(
        name="Tolkach-ESCA",
        manifest="data/pathorob/manifests/pathorob-tolkach-esca-faithful.csv",
        classes=("TUMOR", "SH_OES"),
        centers=("VALSET1_UKK", "VALSET2_WNS"),
        center_column="medical_center",
        class_labels={"TUMOR": "Tumour", "SH_OES": "Oesophageal mucosa"},
        center_labels={"VALSET1_UKK": "UKK", "VALSET2_WNS": "WNS"},
        note="documented placeholder (main result is dataset-wide over 6 classes x 3 cohorts)",
    ),
    MontageSpec(
        name="Prostate (H&E)",
        manifest="data/prostate-shift-binary-kirumc.csv",
        classes=("benign", "tumor"),
        centers=("KI", "RUMC"),
        center_column="medical_center",
        class_labels={"benign": "Benign", "tumor": "Cancer"},
        center_labels={"KI": "Karolinska", "RUMC": "Radboud"},
        note="canonical native 2x2 (prostate-shift binary, dataset-wide)",
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
    """Render the composite montage (one 2x2 biology x center block per dataset).

    Requires every block's four tiles to exist on disk; callers guard this behind a
    data-availability check. Writes a flat PDF at ``out_path`` (for the paper's
    ``\\graphicspath``) plus a PNG sibling under ``png/``.
    """
    import matplotlib.pyplot as plt

    plotstyle.apply_style()

    n = len(blocks)
    if n == 0:
        raise ValueError("no blocks to render")
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(plotstyle.COL_DOUBLE, plotstyle.COL_DOUBLE * 0.58 * nrows))
    subfigs = np.atleast_1d(fig.subfigures(nrows, ncols, wspace=0.05, hspace=0.14)).ravel()

    for panel, block in enumerate(blocks):
        subfig = subfigs[panel]
        spec = block.spec
        subfig.suptitle(
            spec.name,
            fontsize=plotstyle.FS_TITLE,
            weight="bold",
            color=plotstyle.TEXT_COLOR,
        )
        axes = np.atleast_2d(subfig.subplots(2, 2))
        for row, klass in enumerate(spec.classes):
            for col, center in enumerate(spec.centers):
                ax = axes[row, col]
                ax.imshow(_read_tile(block.tiles[(klass, center)]))
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(plotstyle.SPINE_COLOR)
                    spine.set_linewidth(plotstyle.LW_SPINE)
                if row == 0:  # columns = center (confounder)
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

    # Blank any unused panels (e.g. an odd dataset count).
    for panel in range(n, len(subfigs)):
        subfigs[panel].set_facecolor("none")

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
