"""Cohort composition counted from a benchmark view's evaluation manifest.

The dataset-cardinality figures and their caption macros (``_panel_macros.cardinality_macros``)
state how many classes, centres, quartets, slides and tiles a benchmark holds. Those
numbers used to be constants typed into the figure script (``quartets = 94``) and the
captions. They are properties of the evaluated rows, so they are counted here, from the
manifest ``views.load_view(benchmark).eval_manifest`` returns, and every consumer reads the
same function.

The manifest's ``group_id`` is the independence unit -- a slide on the PathoROB tile
benchmarks, a case on Tolkach-ESCA, a slide (one row each) on PCaBiop -- so a "slide" count
here is a count of distinct ``group_id`` values.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PairedComposition:
    """A paired-2x2 cohort: quartets of two classes crossed with two centres."""

    n_classes: int
    n_centres: int
    n_quartets: int


@dataclass(frozen=True)
class Cell:
    """One class x confounder cell: its distinct ``group_id`` values and its rows."""

    slides: int
    tiles: int


#: A cell of an ordinary benchmark; a paired cohort's cell also carries its quartet.
CELL = ("label", "confounder")
QUARTET_CELL = ("subset", "label", "confounder")


def cells(manifest: pd.DataFrame, keys: tuple[str, ...] = CELL) -> dict[tuple, Cell]:
    """Slides (distinct ``group_id``) and tiles (rows) of every cell, keyed by ``keys``."""
    grouped = manifest.groupby(list(keys), sort=True)["group_id"].agg(["nunique", "size"])
    return {
        index if isinstance(index, tuple) else (index,): Cell(slides=int(n), tiles=int(rows))
        for index, n, rows in zip(grouped.index, grouped["nunique"], grouped["size"])
    }


@dataclass(frozen=True)
class Balance:
    """A cohort whose every cell holds the same count, as the cardinality captions state.

    ``tiles_per_slide`` counts rows per ``group_id`` within one class (and quartet), so on
    Tolkach-ESCA it is tiles per case per class. ``slides`` and ``tiles`` are the cohort's
    distinct ``group_id`` values and rows: a Tolkach case contributes to several classes, so
    ``slides`` is not ``cell.slides * n_cells`` there.
    """

    cell: Cell
    n_cells: int
    tiles_per_slide: int
    slides: int
    tiles: int


def balance(manifest: pd.DataFrame, keys: tuple[str, ...] = CELL) -> Balance:
    """The one cell count every cell shares; raises ``ValueError`` if cells differ."""
    by_cell = cells(manifest, keys)
    distinct = set(by_cell.values())
    if len(distinct) != 1:
        raise ValueError(f"unbalanced cohort: cells hold {sorted(distinct, key=str)}")
    per_slide_keys = [k for k in keys if k != "confounder"] + ["group_id"]
    per_slide = set(manifest.groupby(per_slide_keys).size().tolist())
    if len(per_slide) != 1:
        raise ValueError(f"unbalanced cohort: {sorted(per_slide)} tiles per slide")
    return Balance(
        cell=distinct.pop(),
        n_cells=len(by_cell),
        tiles_per_slide=int(per_slide.pop()),
        slides=int(manifest["group_id"].nunique()),
        tiles=len(manifest),
    )


def paired_composition(manifest: pd.DataFrame) -> PairedComposition:
    """Distinct classes, centres and quartets (``subset``) across a paired cohort's rows."""
    return PairedComposition(
        n_classes=int(manifest["label"].nunique()),
        n_centres=int(manifest["confounder"].nunique()),
        n_quartets=int(manifest["subset"].nunique()),
    )
