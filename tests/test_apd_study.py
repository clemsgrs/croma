"""The APD study driver: does it still own only what a driver owns?

``scripts/studies/apd/`` reproduces the paper's downstream numbers. Since the protocol and
both reductions ship in ``croma.downstream`` (ADR-0011), what is left here is manifests,
dataset configuration, schedules, a model roster, output paths and CSV assembly -- and the
tests below hold it to that. Two of them read the study's source rather than run it,
because what they assert is about the *import graph*: that no metric arrives from a
sibling PathoROB checkout, and that none is defined here instead.

The rest are behavioural. One covers the one piece of per-replicate logic the study still
owns: Tolkach-ESCA's train/test case split, which reaches the sweep through its
``arrange_slides`` hook. Its expected ordering comes from the driver loop the rewire
replaced, transcribed below -- so the test asks whether the two agree, not whether the new
code does what the new code does. The last two cover the other thing a driver owns, its
reporting: that the CSV it assembles gates nothing, and that the slide-level arm enters
only its own descriptive benchmark scope.
"""

from __future__ import annotations

import ast
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
    # tells readers to install. The retired study-local normalized reduction must stay gone.
    experiment = (STUDY / "apd_experiment.py").read_text(encoding="utf-8")
    tree = ast.parse(experiment)

    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("croma")
        for alias in node.names
    }

    assert not defined & {"apd", "nipd", "compute_apd", "compute_nipd"}
    assert {"apd", "nipd"} <= imported
    assert "NAPD_NORM_SKILL_FLOOR" not in experiment


def _driver_loop(features, centers, n_classes, feasible_splits, num_patches_per_slide, seed):
    """The replicate ordering as ``apd_experiment.compute`` ran it before the rewire.

    Transcribed from the loop the rewire deleted. It is the reference the study's Tolkach
    arrangement has to reproduce; nothing here may be simplified, because the point is that
    it was not written against the new code.

    One line does diverge from that transcription, deliberately and visibly: the intersection
    is ``sorted``. The original left it a bare ``set``, whose iteration order over strings is
    hash-randomised per process, so it decided *which* drawn case ended up outermost and
    therefore which slides the narrow held-out tail actually held (#105). Sorting both sides
    keeps this a real check of everything else the loop does -- the draw, the shuffle, the
    indexing, the pop-and-append -- while pinning the one thing that was never reproducible.
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
            for test_case in sorted(set(test_cases) & set(case_order)):
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
    group_ids = [slide_of_row[row] for row in range(len(slide_of_row))]

    expected = _driver_loop(features, CENTERS, CLASSES, FEASIBLE, ROWS_PER_SLIDE, seed)
    arrange = loaders.case_arrangement(CENTERS, FEASIBLE, group_ids)
    arranged = arrange(cells, random.Random(seed))

    assert [
        [[row for slide in cell for row in slide] for cell in cell_row] for cell_row in arranged
    ] == expected


#: Run one arrangement and print it. Spawned rather than called, because the property under
#: test is a function of ``PYTHONHASHSEED``, which CPython fixes once per process.
_ARRANGEMENT = """
import json, random, sys
sys.path.insert(0, sys.argv[1])
import loaders
centers, feasible, group_ids, cells = json.loads(sys.argv[2])
arrange = loaders.case_arrangement(centers, feasible, group_ids)
print(json.dumps(arrange(cells, random.Random(0))))
"""


def test_the_tolkach_arrangement_does_not_depend_on_the_hash_seed() -> None:
    # The arrangement pushes a replicate's drawn cases to the tail in iteration order, and
    # the tail is narrower than the number of cases drawn -- so if that order is hash
    # randomised, which slides the sweep tests on changes between processes. It did, and
    # Tolkach's stored matrices were never reproducible because of it (#105). The intersection
    # is sorted now; this pins that, and goes red if a bare set comes back.
    cells, slide_of_row = _cohort()
    group_ids = [slide_of_row[row] for row in range(len(slide_of_row))]
    payload = json.dumps([CENTERS, FEASIBLE, group_ids, cells])

    arrangements = {
        seed: subprocess.run(
            [sys.executable, "-c", _ARRANGEMENT, str(STUDY), payload],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        ).stdout
        for seed in range(6)
    }

    assert len(set(arrangements.values())) == 1, "the arrangement moved with PYTHONHASHSEED"


#: One embedding per slide, which is what makes a cohort the slide arm.
SLIDE_LEVEL_ROWS_PER_SLIDE = 1

#: The headroom over chance of the cell the retired skill floor used to blank: the slide
#: arm's OOD baseline for Prov-GigaPath cleared chance by this much, of a possible 0.5.
SUPPRESSED_CELL_HEADROOM = 0.048


def test_the_assembled_csv_reports_a_value_for_every_cell(tmp_path: Path) -> None:
    # The driver used to carry a skill floor and a paired *_gated column, and blanked values
    # for any cell whose baseline retained too little headroom. The shipped reduction has no
    # extra gate (ADR-0018), so a near-chance cell is reported like any other and the reader
    # decides what to lean on. The stand-in below is a cell with the headroom of the one the
    # floor used to blank; it is read out of the resume cache, so no sweep runs for it.
    import apd_experiment
    from loaders import DATASETS, training_correlations

    chance = 1.0 / len(DATASETS["pcabiop"]["biological_classes"])
    cramers_v = training_correlations("pcabiop")
    cached = dict(  # what a swept cell leaves behind; the reductions read the two matrices
        apd_id=-0.1,
        apd_ood=-0.2,
        id_test_accuracies=[[0.9 - 0.2 * v] for v in cramers_v],
        ood_test_accuracies=[[chance + SUPPRESSED_CELL_HEADROOM * (1.0 - v)] for v in cramers_v],
    )
    (tmp_path / "pcabiop").mkdir()
    (tmp_path / "pcabiop" / "Stand-In.json").write_text(json.dumps(cached), encoding="utf-8")

    reported = apd_experiment.run(
        ["pcabiop"], ["Stand-In"], iterations=1, out_dir=tmp_path, overwrite=False
    )
    written = pd.read_csv(tmp_path / "apd.csv")

    assert [column for column in written.columns if column.endswith("_gated")] == []
    assert not written.isna().to_numpy().any()
    assert reported["nipd_ood"].notna().all()
    assert reported["nipd_ood"].iloc[0] == pytest.approx(-0.5)


def test_the_slide_arm_reaches_no_correlation_table_column() -> None:
    """Four slide encoders are too few for a rank correlation to carry a conclusion, so no
    slide-level cohort gets a column in the downstream-correlation table.

    The exclusion is scoped to the *table*, which is the claim the panel size undermines.
    PCaBiop is still computed and still joined: supp/panda.tex quotes both its Spearman
    values in prose under an explicit "these associations are descriptive" caveat, and
    figure_apd_pcabiop.tex plots its scatter. Asserting the exclusion against the join or
    the correlation output instead makes those two macros ungeneratable and leaves the
    manuscript with undefined control sequences -- see the companion test below.
    """
    import loaders

    sys.path.insert(0, str(ROOT / "scripts" / "repro"))
    import _apd

    slide_level = {
        dataset
        for dataset, config in loaders.DATASETS.items()
        if config["num_patches_per_slide"] == SLIDE_LEVEL_ROWS_PER_SLIDE
    }

    assert slide_level, "no slide-level cohort is configured, so the exclusion asserts nothing"
    # The table renders exactly these benchmarks as columns.
    assert not slide_level & set(_apd.FIGURE_DATASETS)


def test_the_correlation_still_computes_the_scopes_the_supplement_quotes() -> None:
    """PCaBiop must survive as far as ``apd_correlation.csv``.

    ``generate_paper_values._apd_macros`` skips any scope missing from that CSV *silently*,
    so dropping PCaBiop here does not fail loudly -- it just omits \\NipdIdCromaPcabiop and
    \\NipdOodCromaPcabiop, which supp/panda.tex cites, and the manuscript stops compiling.
    """
    import apd_croma_correlation

    rows = []
    scopes = {"camelyon", "tcga_4x4", "tolkach", "pcabiop"}
    for dataset in sorted(scopes):
        for i in range(3):
            rows.append(
                {
                    "dataset": dataset,
                    "model": f"model-{i}",
                    "croma": float(i),
                    "ri": float(i),
                    "mari": float(i),
                    "nipd_id": float(i),
                }
            )

    correlations = apd_croma_correlation.corr_block(
        pd.DataFrame(rows),
        target="nipd_id",
    )

    assert set(correlations["scope"]) == scopes
