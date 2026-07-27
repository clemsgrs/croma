"""The confounder-biased probe sweep: is the accuracy matrix it returns well-formed?

The sweep trains a biology probe on a training set the schedule biases more and more
towards a confounder, and scores it on balanced test sets. What it returns is the input
``apd`` and ``napd`` read, so the tests below assert the *shape of that contract* --
dimensions, range, which row is the baseline, and that both reductions accept the matrix
untouched -- and never a particular accuracy. Accuracies here come out of a synthetic
cloud and a seeded shuffle; pinning one would pin the seed, not the protocol.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from croma import apd, napd
from croma.downstream import pathorob_schedule, probe_sweep, probe_sweep_over_test_sets

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "src" / "croma" / "downstream" / "_pathorob.py"
NOTICE = ROOT / "NOTICE"

SCHEDULE_PARITY = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "pathorob_schedule_parity.json").read_text(
        encoding="utf-8"
    )
)
SCHEDULE_CASES = [pytest.param(case, id=case["name"]) for case in SCHEDULE_PARITY["cases"]]

SLIDES_PER_CELL = 10
ROWS_PER_SLIDE = 3
ROWS_PER_CELL = SLIDES_PER_CELL * ROWS_PER_SLIDE


def _synthetic_cells(dim: int = 4, seed: int = 0):
    """A 2-confounder x 2-class tileset: rows grouped by cell, in cell order.

    Biology sits on dimension 0 and the confounder on dimension 1, both above the noise,
    so a probe can learn either one. Which of the two it leans on is what the schedule
    decides -- that is the protocol under test.
    """
    rng = np.random.default_rng(seed)
    rows, confounders, labels = [], [], []
    for confounder in (0, 1):
        for label in (0, 1):
            noise = rng.normal(scale=0.25, size=(ROWS_PER_CELL, dim))
            block = noise + np.array([2.0 * label, 2.0 * confounder, 0.0, 0.0])
            rows.append(block)
            confounders.extend([confounder] * ROWS_PER_CELL)
            labels.extend([label] * ROWS_PER_CELL)
    return np.concatenate(rows), np.array(confounders), np.array(labels)


def _schedule(*, favourable_slides=(4, 2), balanced_load: int = 4, rows_per_slide=ROWS_PER_SLIDE):
    """A 2x2 schedule: split 0 balanced, later splits skewed towards the diagonal.

    ``favourable_slides[s]`` is what a favourable (confounder == class) cell contributes
    at split ``s``; the unfavourable cells take the slides it gave up, so the training
    total is conserved and only the confounder-biology correlation moves.
    """
    max_train_slides = 2 * balanced_load
    schedule = []
    for favourable in favourable_slides:
        unfavourable = 2 * balanced_load - favourable
        split_map = [
            (0, 0, favourable * rows_per_slide),
            (0, 1, unfavourable * rows_per_slide),
            (1, 0, unfavourable * rows_per_slide),
            (1, 1, favourable * rows_per_slide),
        ]
        schedule.append((split_map, max_train_slides))
    return schedule


def test_probe_sweep_returns_one_row_per_split_and_one_column_per_iteration() -> None:
    embeddings, confounders, labels = _synthetic_cells()
    schedule = _schedule(favourable_slides=(4, 3, 2))

    accuracies = probe_sweep(
        embeddings,
        confounders,
        labels,
        schedule=schedule,
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=2,
    )

    assert accuracies.shape == (3, 2)


def test_probe_sweep_reports_balanced_accuracies() -> None:
    embeddings, confounders, labels = _synthetic_cells()

    accuracies = probe_sweep(
        embeddings,
        confounders,
        labels,
        schedule=_schedule(),
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=2,
    )

    assert accuracies.dtype == float
    assert np.all((accuracies >= 0.0) & (accuracies <= 1.0))


def test_probe_sweep_feeds_both_reductions_untouched() -> None:
    # The reason the protocol ships at all: a caller who can run the sweep can obtain the
    # input of both reductions, with no reshaping in between.
    embeddings, confounders, labels = _synthetic_cells()

    accuracies = probe_sweep(
        embeddings,
        confounders,
        labels,
        schedule=_schedule(),
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=2,
    )

    assert np.isfinite(apd(accuracies))
    assert np.isfinite(napd(accuracies, chance=1 / 2))


def test_probe_sweep_row_zero_is_the_split_the_caller_put_first() -> None:
    # "Row 0 is the unbiased baseline" is a claim about *indexing*, not about accuracy:
    # row 0 must read the schedule's first entry -- the balanced split the caller put
    # there -- and nothing later. Two sweeps that agree on their first split and differ
    # everywhere after it must therefore agree on row 0. Asserting the correspondence
    # this way pins it without pinning a seed-dependent accuracy.
    embeddings, confounders, labels = _synthetic_cells()
    common = dict(rows_per_slide=ROWS_PER_SLIDE, iterations=2, seed=7)

    mild = probe_sweep(
        embeddings, confounders, labels, schedule=_schedule(favourable_slides=(4, 3)), **common
    )
    severe = probe_sweep(
        embeddings, confounders, labels, schedule=_schedule(favourable_slides=(4, 0)), **common
    )

    assert mild[0] == pytest.approx(severe[0])
    assert mild[1] != pytest.approx(severe[1])


def test_probe_sweep_over_test_sets_scores_unseen_confounders_from_the_same_probes() -> None:
    # PathoROB scores every probe twice: on the held-out rows of the confounders it
    # trained on, and on an unseen confounder. Both come off one training pass, and a
    # caller who asked for them separately would pay for the sweep twice -- so the
    # multi-test form is what a study driver reaches for. Its in-domain matrix must be
    # the very matrix `probe_sweep` returns: one protocol, two shapes of answer.
    embeddings, confounders, labels = _synthetic_cells()
    unseen_x = np.array([[0.0, 4.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]] * 3, dtype=float)
    unseen_y = np.array([0, 1] * 3)
    common = dict(rows_per_slide=ROWS_PER_SLIDE, iterations=2, seed=11)

    sweep = probe_sweep_over_test_sets(
        embeddings,
        confounders,
        labels,
        schedule=_schedule(),
        test_sets={"unseen_confounder": (unseen_x, unseen_y)},
        **common,
    )

    assert set(sweep) == {"in_domain", "unseen_confounder"}
    assert sweep["unseen_confounder"].shape == (2, 2)
    assert np.all((sweep["unseen_confounder"] >= 0.0) & (sweep["unseen_confounder"] <= 1.0))
    assert sweep["in_domain"] == pytest.approx(
        probe_sweep(embeddings, confounders, labels, schedule=_schedule(), **common)
    )


def test_a_slide_level_sweep_needs_an_explicit_validation_fraction() -> None:
    # PathoROB reserves `1 / max_train_slides` of a cell's *rows* for validation, which is
    # a sane fraction while a slide carries many rows. When a slide is one row -- slide
    # embeddings rather than patch embeddings -- that ratio floors to zero and the
    # validation set vanishes, taking the regularisation search with it. The sweep says so
    # rather than handing an empty set to the probe, and the caller states the fraction.
    embeddings, confounders, labels = _synthetic_cells()
    slide_level = dict(
        schedule=_schedule(favourable_slides=(10, 6), balanced_load=10, rows_per_slide=1),
        rows_per_slide=1,
        iterations=1,
    )

    with pytest.raises(ValueError, match="validation"):
        probe_sweep(embeddings, confounders, labels, **slide_level)

    accuracies = probe_sweep(
        embeddings, confounders, labels, validation_fraction=0.1, **slide_level
    )

    assert accuracies.shape == (2, 1)


def test_probe_sweep_rejects_row_arrays_that_do_not_line_up() -> None:
    embeddings, confounders, labels = _synthetic_cells()

    with pytest.raises(ValueError, match="one entry per embedding row"):
        probe_sweep(
            embeddings,
            confounders[:-1],
            labels,
            schedule=_schedule(),
            rows_per_slide=ROWS_PER_SLIDE,
        )


def test_probe_sweep_rejects_confounder_names_that_are_not_schedule_indices() -> None:
    # A schedule addresses cells by index, so names have to be mapped to indices before
    # the sweep sees them -- otherwise which cell a schedule entry means would depend on
    # an ordering the sweep invented.
    embeddings, confounders, labels = _synthetic_cells()
    named = np.where(confounders == 0, "RUMC", "UMCU")

    with pytest.raises(ValueError, match="integer indices"):
        probe_sweep(embeddings, named, labels, schedule=_schedule(), rows_per_slide=ROWS_PER_SLIDE)


def test_probe_sweep_rejects_a_schedule_with_no_confounded_split() -> None:
    # A baseline on its own is not a sweep: both reductions need a row to compare against
    # row 0, so a matrix with one row could not be reduced at all.
    embeddings, confounders, labels = _synthetic_cells()

    with pytest.raises(ValueError, match="confounded split"):
        probe_sweep(
            embeddings,
            confounders,
            labels,
            schedule=_schedule(favourable_slides=(4,)),
            rows_per_slide=ROWS_PER_SLIDE,
        )


def test_probe_sweep_rejects_a_schedule_the_cells_are_too_small_for() -> None:
    # The held-out tail starts past the widest training block the schedule asks for. Ask
    # for more slides than a cell holds and the tail is empty, so the probe would be
    # scored on nothing -- which sklearn reports as a shape error three frames deeper.
    embeddings, confounders, labels = _synthetic_cells()

    with pytest.raises(ValueError, match="no test rows"):
        probe_sweep(
            embeddings,
            confounders,
            labels,
            schedule=_schedule(favourable_slides=(9, 8), balanced_load=9),
            rows_per_slide=ROWS_PER_SLIDE,
        )


def test_probe_sweep_rejects_a_schedule_naming_a_cell_no_row_carries() -> None:
    embeddings, confounders, labels = _synthetic_cells()
    split_map, max_train_slides = _schedule()[0]

    with pytest.raises(ValueError, match=r"cell \(2, 0\)"):
        probe_sweep(
            embeddings,
            confounders,
            labels,
            schedule=[(split_map, max_train_slides), ([(2, 0, 3)], max_train_slides)],
            rows_per_slide=ROWS_PER_SLIDE,
        )


def test_probe_sweep_over_test_sets_rejects_a_test_set_named_like_the_held_out_one() -> None:
    embeddings, confounders, labels = _synthetic_cells()

    with pytest.raises(ValueError, match="in_domain"):
        probe_sweep_over_test_sets(
            embeddings,
            confounders,
            labels,
            schedule=_schedule(),
            test_sets={"in_domain": (embeddings[:4], labels[:4])},
            rows_per_slide=ROWS_PER_SLIDE,
        )


@pytest.mark.parametrize("case", SCHEDULE_CASES)
def test_pathorob_schedule_reproduces_upstreams_own_schedule_exactly(case: dict) -> None:
    # The schedule is the definition of "progressively confounded": change it and APD is a
    # different quantity under the same name, with every value still computing. So this is
    # equality against what upstream's own helper returns, not a resemblance check.
    schedule = pathorob_schedule(
        case["dataset"],
        rows_per_slide=case["num_patches_per_slide"],
        n_splits=case["num_splits"],
    )

    assert len(schedule) == case["num_splits"]
    for expected, (split_map, max_train_slides) in zip(case["splits"], schedule):
        assert [list(entry) for entry in split_map] == expected["split_map"]
        assert max_train_slides == expected["max_train_slides"]


@pytest.mark.parametrize("case", SCHEDULE_CASES)
def test_pathorob_schedule_knows_how_many_splits_each_dataset_has(case: dict) -> None:
    # PathoROB's split counts live in its loader, not in the helper croma vendored, so the
    # caller would otherwise have to carry three magic numbers to run the reference
    # protocol at all.
    default = pathorob_schedule(case["dataset"], rows_per_slide=case["num_patches_per_slide"])

    assert len(default) == case["num_splits"]


def test_pathorob_schedule_rejects_a_dataset_it_has_no_schedule_for() -> None:
    with pytest.raises(ValueError, match="prostate"):
        pathorob_schedule("prostate", rows_per_slide=1)


def test_schedule_parity_fixture_records_where_it_came_from_and_how_to_recapture_it() -> None:
    source = SCHEDULE_PARITY["source"]

    assert source["url"] == "https://github.com/bifold-pathomics/PathoROB"
    assert len(source["revision"]) == 40
    assert source["helper"] == "pathorob/apd/utils.py :: get_patches_map_to_split"
    assert source["num_splits_from"]
    assert SCHEDULE_PARITY["procedure"]


def test_the_vendored_module_carries_the_protocol_and_names_its_revision() -> None:
    # The reduction alone is not the reference metric: the schedule the probe is trained
    # under and the regularisation search it is trained with decide the accuracies the
    # reduction reduces. All three have to be upstream's, in the file that says so.
    source = VENDORED.read_text(encoding="utf-8")

    assert "DO NOT EDIT" in source
    assert SCHEDULE_PARITY["source"]["revision"] in source
    for function in ("compute_apd", "get_patches_map_to_split", "train_logistic_regression"):
        assert f"def {function}(" in source


def test_notice_covers_the_protocol_under_one_pathorob_heading() -> None:
    # Clause 1 of BSD 3-Clause travels with whatever is redistributed, so the NOTICE has to
    # describe everything vendored -- and describe it once. A second PathoROB block would
    # be a place for the two to drift apart about what croma actually ships.
    notice = NOTICE.read_text(encoding="utf-8")

    assert notice.count("Copyright (c) 2025, BIFOLD Pathomics") == 1
    assert "probe" in notice
    assert "schedule" in notice or "split" in notice


def test_the_downstream_module_reads_nothing_off_disk() -> None:
    # The protocol takes embeddings and a split assignment, and that is the whole of its
    # world: no model to load, no manifest to read, no output layout to know about. That
    # boundary is what ADR-0011's narrowing of ADR-0002 rests on -- the library gained a
    # measurement, not a pipeline -- so it is asserted rather than left to review.
    forbidden = ("open(", "np.load", "read_csv", "read_parquet", "os.environ", "Path(")
    modules = sorted((ROOT / "src" / "croma" / "downstream").glob("*.py"))

    assert modules
    for module in modules:
        code = "\n".join(
            line for line in module.read_text(encoding="utf-8").splitlines() if line.strip()
        )
        for token in forbidden:
            assert token not in code, f"{module.name} reaches for {token}"


def test_the_library_never_imports_torch() -> None:
    # `croma.downstream` is numpy/sklearn-only, which is what makes it free to ship: the
    # protocol adds no install weight, and torch stays in the [repro] extra where it
    # serves embedding extraction. An import here would move that cost onto every user.
    for module in sorted((ROOT / "src" / "croma").rglob("*.py")):
        code = module.read_text(encoding="utf-8")

        assert "import torch" not in code, f"{module} imports torch"
