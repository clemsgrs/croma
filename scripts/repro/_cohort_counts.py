"""Cohort composition counted from a benchmark view's evaluation manifest.

The dataset-cardinality figures, their caption macros and the TCGA-2x2 composition macros
all state how many classes, centres, quartets, slides and tiles a benchmark holds. Those
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


def paired_composition(manifest: pd.DataFrame) -> PairedComposition:
    """Distinct classes, centres and quartets (``subset``) across a paired cohort's rows."""
    return PairedComposition(
        n_classes=int(manifest["label"].nunique()),
        n_centres=int(manifest["confounder"].nunique()),
        n_quartets=int(manifest["subset"].nunique()),
    )
