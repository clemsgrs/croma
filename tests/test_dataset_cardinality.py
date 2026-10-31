"""The dataset-cardinality figures count slides and tiles per cell from the benchmark views.

Every cell of ``fig:dataset-cardinality`` and of the two paired-protocol schematics used to
show one hard-coded count. They now read ``views.load_view(...).eval_manifest``: a cell's
slides are its distinct ``group_id`` values (cases on Tolkach-ESCA) and its tiles are its
rows. The worked manifests below are small enough to count by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts/repro", ROOT / "scripts/repro/figures", ROOT / "scripts/bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _cohort_counts as counts  # noqa: E402


def _manifest(cells: dict[tuple[str, str], tuple[int, int]], subset: str | None = None):
    """``slides`` groups of ``tiles_per_slide`` rows in every (label, confounder) cell."""
    rows = []
    for (label, confounder), (slides, tiles_per_slide) in cells.items():
        for s in range(slides):
            group = f"{label}-{confounder}-{s}"
            for t in range(tiles_per_slide):
                rows.append((f"{group}-{t}", label, confounder, group))
    frame = pd.DataFrame(rows, columns=["sample_id", "label", "confounder", "group_id"])
    if subset is not None:
        frame["subset"] = subset
    return frame


def test_a_cell_counts_distinct_groups_as_slides_and_rows_as_tiles():
    manifest = _manifest({("normal", "A"): (3, 4), ("normal", "B"): (2, 4), ("tumor", "A"): (3, 1)})

    cells = counts.cells(manifest)

    assert cells[("normal", "A")] == counts.Cell(slides=3, tiles=12)
    assert cells[("normal", "B")] == counts.Cell(slides=2, tiles=8)
    assert cells[("tumor", "A")] == counts.Cell(slides=3, tiles=3)


def test_a_balanced_cohort_reports_its_one_cell_its_tiles_per_slide_and_its_groups():
    # Tolkach-shaped: case "c1" contributes 2 tiles to each of two classes, so the four
    # cells hold one case each but the cohort holds three distinct cases, not four.
    manifest = pd.DataFrame(
        {
            "sample_id": [f"t{i}" for i in range(8)],
            "label": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "confounder": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
            "group_id": ["c1", "c1", "c1", "c1", "c2", "c2", "c3", "c3"],
        }
    )

    balance = counts.balance(manifest)

    assert balance.cell == counts.Cell(slides=1, tiles=2)
    assert balance.n_cells == 4
    assert balance.tiles_per_slide == 2  # per case per class
    assert (balance.slides, balance.tiles) == (3, 8)


def test_an_unbalanced_cohort_cannot_claim_one_count_per_cell():
    manifest = _manifest({("normal", "A"): (3, 4), ("normal", "B"): (2, 4)})

    with pytest.raises(ValueError, match="unbalanced"):
        counts.balance(manifest)


def _spec(name: str):
    import dataset_cardinality as cardinality

    return next(spec for spec in cardinality.SPECS if spec.name == name)


def test_a_tile_cell_shows_its_slides_over_its_tiles():
    import dataset_cardinality as cardinality

    manifest = _manifest({(b, c): (17, 300) for b in ("normal", "tumor") for c in ("RUMC", "UMCU")})

    card = cardinality.card(_spec("Camelyon"), manifest)

    assert card.cells[("normal", "RUMC")] == ("17 slides", "5,100 tiles")
    assert card.cells[("tumor", "UMCU")] == ("17 slides", "5,100 tiles")
    assert card.total == "68 slides · 20,400 tiles"


def test_tolkach_counts_cases_and_pcabiop_counts_slides_only():
    import dataset_cardinality as cardinality

    tolkach = _spec("Tolkach-ESCA")
    classes = [raw for raw, _ in tolkach.rows]
    centres = [raw for raw, _ in tolkach.cols]
    pcabiop = _spec("PCaBiop")

    case_card = cardinality.card(
        tolkach, _manifest({(b, c): (5, 100) for b in classes for c in centres})
    )
    slide_card = cardinality.card(
        pcabiop,
        _manifest({(b, c): (250, 1) for b in ("0", "1") for c in ("karolinska", "radboud")}),
    )

    assert case_card.cells[(classes[0], centres[0])] == ("5 cases", "500 tiles")
    assert slide_card.cells[("1", "radboud")] == ("250 slides",)
    assert slide_card.total == "1,000 slides"


def test_a_manifest_whose_grid_differs_from_the_card_fails():
    import dataset_cardinality as cardinality

    manifest = _manifest({("normal", "RUMC"): (17, 300), ("tumor", "RUMC"): (17, 300)})

    with pytest.raises(ValueError, match="UMCU"):
        cardinality.card(_spec("Camelyon"), manifest)


def test_the_paired_schematic_shows_one_quartet_cell_and_the_pooled_slide_range():
    import dataset_cardinality as cardinality

    # Three quartets of 2 slides x 3 tiles per cell. q2 reuses q1's (A, X) slides, so they
    # pool to 2; q3 brings fresh slides for q1's cells, so (A, X) and (B, X) pool to 4.
    q1 = _manifest({(b, c): (2, 3) for b in ("A", "B") for c in ("X", "Y")}, subset="q1")
    q2 = _manifest({(b, c): (2, 3) for b in ("A", "C") for c in ("X", "Z")}, subset="q2")
    q2.loc[q2["label"].eq("C") & q2["confounder"].eq("Z"), "group_id"] += "-new"
    q3 = _manifest({(b, c): (2, 3) for b in ("A", "B") for c in ("X", "Y")}, subset="q3")
    q3["group_id"] += "-q3"  # fresh slides: (A, X) now pools 4 slides across q1-q3

    quartet = cardinality.quartet_card(pd.concat([q1, q2, q3], ignore_index=True))

    assert quartet.cell == ("2 slides", "6 tiles")
    assert quartet.n_quartets == 3
    assert quartet.pooled_slides == (2, 4)


def test_verify_names_the_cell_where_the_run_lost_a_slide():
    import dataset_cardinality as cardinality

    view = _manifest({(b, c): (3, 2) for b in ("A", "B") for c in ("X", "Y")})
    run = view[view["group_id"] != "B-Y-0"]

    assert cardinality.verify_counts("Bench", view, view.copy()) == []
    problems = cardinality.verify_counts("Bench", view, run)
    assert len(problems) == 1 and "('B', 'Y')" in problems[0]
