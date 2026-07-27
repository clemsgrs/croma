"""The APD study driver: does it still own only what a driver owns?

``scripts/studies/apd/`` reproduces the paper's downstream numbers. Since the protocol and
both reductions ship in ``croma.downstream`` (ADR-0011), what is left here is manifests,
dataset configuration, schedules, a model roster, output paths and CSV assembly -- and the
tests below hold it to that. Two of them read the study's source rather than run it,
because what they assert is about the *import graph*: that no metric arrives from a
sibling PathoROB checkout, and that none is defined here instead.

The third is behavioural, and covers the one piece of per-replicate logic the study still
owns: Tolkach-ESCA's train/test case split, which reaches the sweep through its
``arrange_slides`` hook. Its expected ordering comes from the driver loop the rewire
replaced, transcribed below -- so the test asks whether the two agree, not whether the new
code does what the new code does.
"""

from __future__ import annotations

import ast
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "scripts" / "studies" / "apd"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

STUDY_MODULES = sorted(STUDY.glob("*.py"))


def _imported_modules(source: str) -> set[str]:
    """Every module name the source imports, however it spells the import."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", STUDY_MODULES, ids=lambda path: path.name)
def test_the_study_reaches_for_no_pathorob_checkout(module: Path) -> None:
    # The study used to put a sibling PathoROB checkout on sys.path and import the
    # protocol out of it, so the paper's numbers came from code no user of croma could
    # install. Everything it needs is vendored in the package now; a checkout next door is
    # not an input any more, and neither is the variable that used to point at one.
    source = module.read_text(encoding="utf-8")
    imported = _imported_modules(source)

    assert not [name for name in imported if name == "pathorob" or name.startswith("pathorob.")]
    assert "PATHOROB_ROOT" not in source


def test_the_study_defines_neither_reduction_and_imports_both() -> None:
    # Exactly one implementation of each reduction exists in the repository, and it is the
    # shipped one -- otherwise the paper reports numbers from code that is not the code it
    # tells readers to install. The study's own gated nAPD is what this replaced.
    experiment = (STUDY / "apd_experiment.py").read_text(encoding="utf-8")
    tree = ast.parse(experiment)

    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("croma")
        for alias in node.names
    }

    assert not defined & {"apd", "napd", "compute_apd", "compute_napd"}
    assert {"apd", "napd"} <= imported
    assert "NAPD_NORM_SKILL_FLOOR" not in experiment


def _driver_loop(features, centers, n_classes, feasible_splits, num_patches_per_slide, seed):
    """The replicate ordering as ``apd_experiment.compute`` ran it before the rewire.

    Transcribed from the loop this slice deleted, down to the set intersection whose
    iteration order decides which held-out case ends up outermost. It is the reference the
    study's Tolkach arrangement has to reproduce; nothing here may be simplified, because
    the point is that it was not written against the new code.
    """
    random.seed(seed)
    random_train_test_split = (
        random.randint(0, len(feasible_splits["train"][centers[0]]) - 1),
        random.randint(0, len(feasible_splits["train"][centers[1]]) - 1),
    )
    for i, center in enumerate(centers):
        test_cases = feasible_splits["test"][center][random_train_test_split[i]]
        for j in range(n_classes):
            chunked = [
                features[i][j][k : k + num_patches_per_slide]
                for k in range(0, len(features[i][j]), num_patches_per_slide)
            ]
            random.shuffle(chunked)
            case_order = [chunked[k][0][3] for k in range(len(chunked))]
            for test_case in list(set(test_cases) & set(case_order)):
                chunked.append(
                    chunked.pop([chunked[k][0][3] for k in range(len(chunked))].index(test_case))
                )
            features[i][j] = [item for slide in chunked for item in slide]
    return [[[row for row, *_ in cell] for cell in cell_row] for cell_row in features]


CENTERS = ["VALSET2_WNS", "VALSET4_CHA_FULL"]
CASES = ["case-a", "case-b", "case-c", "case-d"]
FEASIBLE = {
    "train": {
        CENTERS[0]: [["case-a", "case-b"], ["case-c", "case-d"], ["case-a", "case-d"]],
        CENTERS[1]: [["case-c"], ["case-a", "case-b", "case-d"]],
    },
    "test": {
        CENTERS[0]: [["case-c", "case-d"], ["case-a", "case-b"], ["case-b", "case-c"]],
        CENTERS[1]: [["case-a", "case-b", "case-d"], ["case-c"]],
    },
}
ROWS_PER_SLIDE = 3
SLIDES_PER_CELL = 8
CLASSES = 3


def _cohort():
    """A 2-centre x 3-class cohort of 8 slides per cell, cycling through four cases."""
    rows, cells, slide_of_row = 0, [], {}
    for confounder in range(len(CENTERS)):
        cells.append([])
        for label in range(CLASSES):
            cell = []
            for slide in range(SLIDES_PER_CELL):
                case = CASES[(confounder + label + slide) % len(CASES)]
                cell.append(list(range(rows, rows + ROWS_PER_SLIDE)))
                slide_of_row.update({row: case for row in cell[-1]})
                rows += ROWS_PER_SLIDE
            cells[-1].append(cell)
    return cells, slide_of_row


@pytest.mark.parametrize("seed", [0, 1, 7, 4242])
def test_the_tolkach_arrangement_reproduces_the_driver_loop_it_replaced(seed: int) -> None:
    # Tolkach-ESCA's cases carry patches of several biological classes, so a case may not
    # be trained on and tested on at once: each replicate draws a feasible split and pushes
    # those cases to the tail the sweep tests from. The study still owns that -- it is one
    # cohort's annotation, not the protocol -- but it must land the same rows in the same
    # places as the loop that used to do it inline, or every Tolkach number moves.
    import loaders

    cells, slide_of_row = _cohort()
    features = [
        [
            [(row, i, j, slide_of_row[row]) for slide in cell for row in slide]
            for j, cell in enumerate(cell_row)
        ]
        for i, cell_row in enumerate(cells)
    ]
    slide_ids = [slide_of_row[row] for row in range(len(slide_of_row))]

    expected = _driver_loop(features, CENTERS, CLASSES, FEASIBLE, ROWS_PER_SLIDE, seed)
    arrange = loaders.case_arrangement(CENTERS, FEASIBLE, slide_ids)
    arranged = arrange(cells, random.Random(seed))

    assert [
        [[row for slide in cell for row in slide] for cell in cell_row] for cell_row in arranged
    ] == expected
