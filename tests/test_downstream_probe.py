"""The confounder-biased probe sweep: is the accuracy matrix it returns well-formed?

The sweep trains a biology probe on a training set the schedule biases more and more
towards a confounder, and scores it on balanced test sets. What it returns is the input
``apd`` and ``nipd`` read, so the tests below assert the *shape of that contract* --
dimensions, range, which row is the baseline, and that both reductions accept the matrix
untouched -- and never a particular accuracy. Accuracies here come out of a synthetic
cloud and a seeded shuffle; pinning one would pin the seed, not the protocol.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from croma import apd, nipd
from croma.downstream import (
    pathorob_schedule,
    probe_sweep,
    probe_sweep_over_test_sets,
    probe_sweep_with_predictions,
)

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


def _synthetic_cells(dim: int = 4, seed: int = 0, rows_per_cell: int = ROWS_PER_CELL):
    """A 2-confounder x 2-class tileset: rows grouped by cell, in cell order.

    Biology sits on dimension 0 and the confounder on dimension 1, both above the noise,
    so a probe can learn either one. Which of the two it leans on is what the schedule
    decides -- that is the protocol under test.
    """
    rng = np.random.default_rng(seed)
    rows, confounders, labels = [], [], []
    for confounder in (0, 1):
        for label in (0, 1):
            noise = rng.normal(scale=0.25, size=(rows_per_cell, dim))
            block = noise + np.array([2.0 * label, 2.0 * confounder, 0.0, 0.0])
            rows.append(block)
            confounders.extend([confounder] * rows_per_cell)
            labels.extend([label] * rows_per_cell)
    return np.concatenate(rows), np.array(confounders), np.array(labels)


def _cells_with_impostors(rows_per_cell: int = 20):
    """A slide-level 2x2 tileset where the first row of every cell carries the wrong class.

    Biology is the only thing in the features and it is perfectly separable, so a probe
    trained on any majority-clean sample scores 1.0 -- unless the rows it is *scored* on
    are the mislabelled ones. Which rows those are is the arrangement's decision, so the
    impostor row indices come back with the tileset.
    """
    rows, confounders, labels, impostors = [], [], [], set()
    for confounder in (0, 1):
        for label in (0, 1):
            for position in range(rows_per_cell):
                impostor = position == 0
                if impostor:
                    impostors.add(len(rows))
                shown = 1 - label if impostor else label
                rows.append([4.0 * shown - 2.0, 0.0])
                confounders.append(confounder)
                labels.append(label)
    return np.array(rows), np.array(confounders), np.array(labels), impostors


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
    assert np.isfinite(nipd(accuracies, cramers_v=[0.0, 1.0], chance=1 / 2))


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


def test_an_arrangement_that_shuffles_reproduces_the_sweeps_own_replicate_ordering() -> None:
    # The hook replaces the one step of a replicate that decides which slides a split
    # trains on and which sit in the held-out tail. Hand it back the sweep's own step --
    # shuffle each cell's slides with the replicate's generator -- and the matrix has to
    # come out identical, which is what says the cells and the generator a caller receives
    # are the very ones the default path works with. A cohort that needs a different
    # arrangement can then build it from a known starting point.
    embeddings, confounders, labels = _synthetic_cells()
    common = dict(
        schedule=_schedule(favourable_slides=(4, 2)),
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=2,
        seed=7,
    )

    def shuffle_each_cell(cells, rng):
        arranged = []
        for cell_row in cells:
            arranged.append([])
            for slides in cell_row:
                slides = list(slides)
                rng.shuffle(slides)
                arranged[-1].append(slides)
        return arranged

    arranged = probe_sweep_over_test_sets(
        embeddings, confounders, labels, arrange_slides=shuffle_each_cell, **common
    )["in_domain"]

    assert arranged == pytest.approx(probe_sweep(embeddings, confounders, labels, **common))


def test_an_arrangement_decides_which_slides_land_in_the_held_out_tail() -> None:
    # What the hook is for: PathoROB's Tolkach-ESCA sweep draws a train/test *case* split
    # per replicate and pushes those cases' slides to the tail so they are tested on and
    # never trained on. So the observable is exactly that -- rows an arrangement moves to
    # the back are the rows the probe is scored on. Here two rows per cell carry the wrong
    # class's features: kept up front they are trained on and the probe scores perfectly on
    # a clean tail; pushed to the back they are what it is graded against, and cannot be.
    embeddings, confounders, labels, impostors = _cells_with_impostors()
    common = dict(
        schedule=_schedule(favourable_slides=(5, 3), balanced_load=5, rows_per_slide=1),
        rows_per_slide=1,
        iterations=1,
        validation_fraction=0.2,
    )

    def keep_input_order(cells, rng):
        return cells

    def impostors_to_the_tail(cells, rng):
        return [
            [
                [slide for slide in slides if slide[0] not in impostors]
                + [slide for slide in slides if slide[0] in impostors]
                for slides in cell_row
            ]
            for cell_row in cells
        ]

    clean_tail = probe_sweep_over_test_sets(
        embeddings, confounders, labels, arrange_slides=keep_input_order, **common
    )["in_domain"]
    impostor_tail = probe_sweep_over_test_sets(
        embeddings, confounders, labels, arrange_slides=impostors_to_the_tail, **common
    )["in_domain"]

    assert np.all(clean_tail == 1.0)
    assert np.all(impostor_tail < 1.0)


def test_an_arrangement_may_reorder_the_slides_but_not_change_them() -> None:
    # An arrangement is a permutation of a cell's slides and nothing else. Dropping a
    # slide, inventing one, or breaking one apart would silently reshape the cohort under
    # a schedule written for it -- the sweep would still return a matrix, computed on rows
    # the caller never meant to be there.
    embeddings, confounders, labels = _synthetic_cells()
    common = dict(
        schedule=_schedule(favourable_slides=(4, 2)),
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=1,
    )

    def drop_a_slide(cells, rng):
        return [[list(slides)[1:] for slides in cell_row] for cell_row in cells]

    def drop_a_slide_in_place(cells, rng):
        # The sweep's own arrangement rearranges the lists it is handed and returns them,
        # and so does every arrangement written by copying it -- so a guard that compares
        # the return value against those same lists compares each one with itself and
        # passes whatever was done to them.
        for cell_row in cells:
            for slides in cell_row:
                del slides[0]
        return cells

    def split_a_slide(cells, rng):
        return [
            [[[row] for slide in slides for row in slide] for slides in cell_row]
            for cell_row in cells
        ]

    for arrangement in (drop_a_slide, drop_a_slide_in_place, split_a_slide):
        with pytest.raises(ValueError, match="same slides"):
            probe_sweep_over_test_sets(
                embeddings, confounders, labels, arrange_slides=arrangement, **common
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


def test_probe_sweep_rejects_a_schedule_asking_a_cell_for_more_rows_than_it_holds() -> None:
    # Short-serving a cell would be the quiet failure: the sweep still runs, but at a
    # weaker confounder bias than the schedule describes, so the accuracies belong to a
    # protocol nobody specified. Here one cell came up short of the cohort -- four slides
    # rather than ten -- while the schedule was written for the cohort.
    embeddings, confounders, labels = _synthetic_cells()
    short_cell = slice(4 * ROWS_PER_SLIDE, ROWS_PER_CELL)
    kept = np.ones(len(embeddings), dtype=bool)
    kept[short_cell] = False

    with pytest.raises(ValueError, match="cell \\(0, 0\\) holds 12 rows"):
        probe_sweep(
            embeddings[kept],
            confounders[kept],
            labels[kept],
            schedule=_schedule(),
            rows_per_slide=ROWS_PER_SLIDE,
        )


def test_probe_sweep_rejects_a_schedule_built_for_a_different_rows_per_slide() -> None:
    # A schedule counts rows, and how many rows a slide carries is stated twice: once when
    # the schedule is built, once when the sweep is run. Disagree, and the tail offset --
    # which is in slides -- lands somewhere the row counts were never meant for. The
    # disagreement is visible, because no cell may contribute more training rows than
    # `max_train_slides` slides' worth, and every schedule reaches that bound at its most
    # confounded split.
    embeddings, confounders, labels = _synthetic_cells()

    with pytest.raises(ValueError, match="max_train_slides"):
        probe_sweep(
            embeddings,
            confounders,
            labels,
            schedule=_schedule(rows_per_slide=ROWS_PER_SLIDE),
            rows_per_slide=1,
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


def test_pathorob_schedule_drives_the_sweep_without_the_caller_reshaping_it() -> None:
    # The two halves have to meet: what `pathorob_schedule` returns is what `probe_sweep`
    # takes, so running the reference protocol on a 2x2 cohort is two calls and no glue.
    rows_per_slide, slides_per_cell = 5, 17
    embeddings, confounders, labels = _synthetic_cells(
        rows_per_cell=rows_per_slide * slides_per_cell
    )

    accuracies = probe_sweep(
        embeddings,
        confounders,
        labels,
        schedule=pathorob_schedule("camelyon", rows_per_slide=rows_per_slide, n_splits=3),
        rows_per_slide=rows_per_slide,
        iterations=1,
    )

    assert accuracies.shape == (3, 1)
    assert np.all((accuracies >= 0.0) & (accuracies <= 1.0))


def test_pathorob_schedule_rejects_a_dataset_it_has_no_schedule_for() -> None:
    with pytest.raises(ValueError, match="prostate"):
        pathorob_schedule("prostate", rows_per_slide=1)


def test_pathorob_schedule_walks_a_prefix_but_never_past_the_reference_protocol() -> None:
    # A shorter walk is a smoke run, and stays a prefix of the reference: split s means
    # the same training composition either way. Walking *past* upstream's last split is
    # not an extrapolation of the protocol -- the formulas run the favourable cells
    # negative -- so it is refused rather than silently continued.
    prefix = pathorob_schedule("camelyon", rows_per_slide=1, n_splits=3)

    assert prefix == pathorob_schedule("camelyon", rows_per_slide=1)[:3]

    with pytest.raises(ValueError, match="8 splits"):
        pathorob_schedule("camelyon", rows_per_slide=1, n_splits=9)


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


# --- The prediction record -------------------------------------------------------------
#
# A study that asks *which* held-out rows a confounded probe gets wrong needs every probe's
# predictions, not only its score. The record rides on the sweep's own probes, so the first
# property it has to keep is that it changes nothing the sweep already reports.


def _unseen_confounder():
    unseen_x = np.array([[0.0, 4.0, 0.0, 0.0], [2.0, 4.0, 0.0, 0.0]] * 3, dtype=float)
    unseen_y = np.array([0, 1] * 3)
    return unseen_x, unseen_y


def test_recording_predictions_leaves_every_accuracy_matrix_bit_identical() -> None:
    # Recording draws no random numbers and refits nothing, so the matrices must be the very
    # bits the plain sweep returns -- not merely close to them. An approximate match would
    # hide a recorder that perturbed the generator the arrangement reads.
    embeddings, confounders, labels = _synthetic_cells()
    common = dict(
        schedule=_schedule(favourable_slides=(4, 3, 2)),
        test_sets={"unseen_confounder": _unseen_confounder()},
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=3,
        seed=5,
    )

    plain = probe_sweep_over_test_sets(embeddings, confounders, labels, **common)
    recorded = probe_sweep_with_predictions(embeddings, confounders, labels, **common)

    assert set(recorded.accuracies) == set(plain)
    for name, matrix in plain.items():
        assert np.array_equal(recorded.accuracies[name], matrix)


def _balanced_accuracy_from_margins(true_labels, margins, classes):
    """What sklearn's ``predict`` makes of ``decision_function``: sign when binary, else argmax."""
    from sklearn.metrics import balanced_accuracy_score

    if margins.shape[1] == 1:
        predicted = classes[(margins[:, 0] > 0).astype(int)]
    else:
        predicted = classes[margins.argmax(axis=1)]
    return balanced_accuracy_score(true_labels, predicted)


def test_the_held_out_margins_are_the_ones_each_in_domain_accuracy_was_scored_on() -> None:
    # The record is only worth joining to anything if row r of a replicate's margins is the
    # prediction for held-out row held_out[r]. The sweep's own accuracy matrix is the
    # independent witness: re-scoring the recorded margins against the labels of the
    # recorded rows has to give back every entry, split by split and replicate by replicate.
    embeddings, confounders, labels = _synthetic_cells()
    # A noisier cloud than the default, so the probes make mistakes and an accuracy below
    # 1.0 can tell a correct row assignment from a shuffled one.
    embeddings = embeddings + np.random.default_rng(3).normal(scale=0.9, size=embeddings.shape)
    record = probe_sweep_with_predictions(
        embeddings,
        confounders,
        labels,
        schedule=_schedule(favourable_slides=(4, 2, 0)),
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=3,
        seed=5,
    )

    in_domain = record.accuracies["in_domain"]
    assert (in_domain < 1.0).any()
    n_splits, iterations = in_domain.shape
    assert record.held_out.shape[0] == iterations
    assert record.margins["in_domain"].shape[:3] == (n_splits, iterations, record.held_out.shape[1])
    assert record.margins["in_domain"].shape[3] == 1  # binary: one decision column
    for split in range(n_splits):
        for replicate in range(iterations):
            rows = record.held_out[replicate]
            rescored = _balanced_accuracy_from_margins(
                labels[rows], record.margins["in_domain"][split, replicate], record.classes
            )
            assert rescored == in_domain[split, replicate]


def _three_class_cells(rows_per_cell: int = ROWS_PER_CELL, seed: int = 0):
    """A 3-confounder x 3-class tileset, noisy enough that the probes make mistakes."""
    rng = np.random.default_rng(seed)
    rows, confounders, labels = [], [], []
    for confounder in range(3):
        for label in range(3):
            centre = np.zeros(4)
            centre[label] += 1.5
            centre[3] += 1.5 * confounder
            rows.append(centre + rng.normal(scale=0.8, size=(rows_per_cell, 4)))
            confounders.extend([confounder] * rows_per_cell)
            labels.extend([label] * rows_per_cell)
    return np.concatenate(rows), np.array(confounders), np.array(labels)


def _three_class_schedule(favourable_slides=(3, 5), rows_per_slide=ROWS_PER_SLIDE):
    """Split 0 balanced; later splits favour the diagonal and take from the rest."""
    schedule = []
    for favourable in favourable_slides:
        rest = (9 - favourable) // 2
        split_map = [
            (i, j, (favourable if i == j else rest) * rows_per_slide)
            for i in range(3)
            for j in range(3)
        ]
        schedule.append((split_map, 7))
    return schedule


def test_a_multiclass_probe_records_one_decision_column_per_class_on_every_test_set() -> None:
    # Off the binary case, a margin is a vector: the probe's score for every class, whose
    # argmax is its prediction. A named test set gets the same treatment as the held-out
    # rows, row for row in the caller's order, so its accuracy is recoverable from it too.
    embeddings, confounders, labels = _three_class_cells()
    unseen_x = embeddings[::7] + np.array([0.0, 0.0, 0.0, 3.0])
    unseen_y = labels[::7]
    record = probe_sweep_with_predictions(
        embeddings,
        confounders,
        labels,
        schedule=_three_class_schedule(),
        test_sets={"unseen_confounder": (unseen_x, unseen_y)},
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=2,
        seed=9,
    )

    assert record.classes.tolist() == [0, 1, 2]
    unseen = record.margins["unseen_confounder"]
    assert unseen.shape == (2, 2, len(unseen_x), 3)
    assert record.margins["in_domain"].shape[3] == 3
    assert (record.accuracies["unseen_confounder"] < 1.0).any()
    for split in range(2):
        for replicate in range(2):
            rows = record.held_out[replicate]
            assert _balanced_accuracy_from_margins(
                labels[rows], record.margins["in_domain"][split, replicate], record.classes
            ) == (record.accuracies["in_domain"][split, replicate])
            assert _balanced_accuracy_from_margins(
                unseen_y, unseen[split, replicate], record.classes
            ) == (record.accuracies["unseen_confounder"][split, replicate])


def test_the_record_keeps_the_regularisation_each_probe_was_refitted_with() -> None:
    # PathoROB's validation search picks C from a fixed logarithmic grid; the record keeps
    # the pick, per split and replicate, because a margin's scale depends on it.
    embeddings, confounders, labels = _synthetic_cells()
    record = probe_sweep_with_predictions(
        embeddings,
        confounders,
        labels,
        schedule=_schedule(),
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=2,
    )
    grid = 10 ** np.linspace(-8, 4, 15)

    assert record.best_c.shape == (2, 2)
    assert all(np.isclose(grid, c, rtol=1e-12, atol=0).any() for c in record.best_c.ravel())


# --- Fewer training tiles per slide ----------------------------------------------------
#
# A low-data regime keeps the protocol and thins its training set: each training and
# validation slide gives up most of its rows, while which slides train, which are held
# out, and every test row stay exactly where the full-tile sweep put them.


def test_asking_for_every_tile_of_a_slide_reproduces_the_full_tile_sweep_bit_for_bit() -> None:
    # "All of a slide's rows" is the full-tile sweep, so it has to be the full-tile sweep to
    # the bit, through the reduced-tile code path rather than around it.
    embeddings, confounders, labels = _synthetic_cells()
    common = dict(
        schedule=_schedule(favourable_slides=(4, 3, 2)),
        test_sets={"unseen_confounder": _unseen_confounder()},
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=3,
        seed=5,
    )

    full = probe_sweep_over_test_sets(embeddings, confounders, labels, **common)
    every_tile = probe_sweep_over_test_sets(
        embeddings, confounders, labels, train_rows_per_slide=ROWS_PER_SLIDE, **common
    )

    assert set(every_tile) == set(full)
    for name, matrix in full.items():
        assert np.array_equal(every_tile[name], matrix)


def test_fewer_training_tiles_hold_out_the_very_rows_the_full_tile_sweep_holds_out() -> None:
    # The low-data comparison is only a comparison if the probes are graded on the same
    # rows: same held-out in-domain rows replicate by replicate, same named test set row
    # for row. What changes is the probe, which saw a third of each slide.
    embeddings, confounders, labels = _synthetic_cells()
    embeddings = embeddings + np.random.default_rng(3).normal(scale=0.9, size=embeddings.shape)
    unseen_x, unseen_y = _unseen_confounder()
    common = dict(
        schedule=_schedule(favourable_slides=(4, 2, 0)),
        test_sets={"unseen_confounder": (unseen_x, unseen_y)},
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=3,
        seed=5,
    )

    full = probe_sweep_with_predictions(embeddings, confounders, labels, **common)
    thin = probe_sweep_with_predictions(
        embeddings, confounders, labels, train_rows_per_slide=1, **common
    )

    assert np.array_equal(thin.held_out, full.held_out)
    assert thin.margins["unseen_confounder"].shape == full.margins["unseen_confounder"].shape
    for split in range(3):
        for replicate in range(3):
            rows = thin.held_out[replicate]
            assert _balanced_accuracy_from_margins(
                labels[rows], thin.margins["in_domain"][split, replicate], thin.classes
            ) == (thin.accuracies["in_domain"][split, replicate])
            assert _balanced_accuracy_from_margins(
                unseen_y, thin.margins["unseen_confounder"][split, replicate], thin.classes
            ) == (thin.accuracies["unseen_confounder"][split, replicate])
    assert not np.array_equal(thin.margins["in_domain"], full.margins["in_domain"])


def test_the_tiles_a_slide_keeps_are_a_function_of_the_seed() -> None:
    # Reproducible like the rest of the sweep: the same seed keeps the same rows, and
    # nothing else -- the tile draw has no seed of its own to forget to pass.
    embeddings, confounders, labels = _synthetic_cells()
    common = dict(
        schedule=_schedule(),
        rows_per_slide=ROWS_PER_SLIDE,
        iterations=2,
        train_rows_per_slide=1,
    )

    first = probe_sweep_with_predictions(embeddings, confounders, labels, seed=5, **common)
    again = probe_sweep_with_predictions(embeddings, confounders, labels, seed=5, **common)
    other = probe_sweep_with_predictions(embeddings, confounders, labels, seed=6, **common)

    assert np.array_equal(first.margins["in_domain"], again.margins["in_domain"])
    assert not np.array_equal(first.held_out, other.held_out)


def test_a_cohort_with_its_own_arrangement_keeps_it_at_fewer_tiles() -> None:
    # Tolkach-ESCA's arrangement pushes a drawn set of cases to the held-out tail. The
    # rows are thinned after it runs, so the cases it pushed back are still the ones held
    # out -- here, the impostor rows it moves to the tail are what the probe is graded on.
    embeddings, confounders, labels, impostors = _cells_with_impostors()
    embeddings = np.repeat(embeddings, 2, axis=0)
    confounders = np.repeat(confounders, 2)
    labels = np.repeat(labels, 2)
    impostors = {row for impostor in impostors for row in (2 * impostor, 2 * impostor + 1)}

    def impostors_to_the_tail(cells, rng):
        for cell_row in cells:
            for slides in cell_row:
                rng.shuffle(slides)
                slides.sort(key=lambda slide: slide[0] in impostors)
        return cells

    common = dict(
        schedule=_schedule(favourable_slides=(5, 3), balanced_load=5, rows_per_slide=2),
        rows_per_slide=2,
        iterations=2,
        validation_fraction=0.2,
        arrange_slides=impostors_to_the_tail,
    )

    full = probe_sweep_with_predictions(embeddings, confounders, labels, **common)
    thin = probe_sweep_with_predictions(
        embeddings, confounders, labels, train_rows_per_slide=1, **common
    )

    assert np.array_equal(thin.held_out, full.held_out)
    assert impostors <= set(thin.held_out[0].tolist())


def _training_calls(monkeypatch, sweep_kwargs, embeddings, confounders, labels):
    """Every (training rows, validation rows) the sweep hands PathoROB's trainer, in order.

    The trainer is where the sweep's slicing becomes a probe, so it is the one place the
    row counts are observable; the last feature column carries each row's index so the
    rows can be read back off the matrices the trainer receives.
    """
    import croma.downstream.probe as probe_module

    calls = []
    trainer = probe_module.train_logistic_regression

    def spy(train_x, train_y, val_x, val_y, test_xs, test_ys, **kwargs):
        calls.append((train_x[:, -1].astype(int).tolist(), val_x[:, -1].astype(int).tolist()))
        return trainer(train_x, train_y, val_x, val_y, test_xs, test_ys, **kwargs)

    monkeypatch.setattr(probe_module, "train_logistic_regression", spy)
    indexed = np.column_stack([embeddings, np.arange(len(embeddings), dtype=float)])
    probe_sweep(indexed, confounders, labels, **sweep_kwargs)
    monkeypatch.setattr(probe_module, "train_logistic_regression", trainer)
    return calls


def test_every_training_slide_gives_the_same_number_of_tiles_and_validation_shrinks_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The low-data sweep is the full-tile sweep with each training slide thinned: the same
    # slides train, each contributing exactly `train_rows_per_slide` of its rows, so every
    # cell is scaled alike and the schedule's Cramér's V does not move. Validation is cut
    # from the same rows, by the same factor, but never below one row per cell.
    rows_per_slide, kept = 6, 2
    embeddings, confounders, labels = _synthetic_cells(rows_per_cell=10 * rows_per_slide)
    common = dict(
        schedule=_schedule(favourable_slides=(4, 2), rows_per_slide=rows_per_slide),
        rows_per_slide=rows_per_slide,
        iterations=2,
        seed=5,
    )

    full = _training_calls(monkeypatch, common, embeddings, confounders, labels)
    thin = _training_calls(
        monkeypatch,
        dict(common, train_rows_per_slide=kept),
        embeddings,
        confounders,
        labels,
    )

    def cell(row):
        return int(confounders[row]), int(labels[row])

    assert len(thin) == len(full)
    for (full_train, full_val), (thin_train, thin_val) in zip(full, thin):
        # Rows are laid out slide by slide, so a row's slide is its position over the
        # slide length.
        slides = {}
        for row in thin_train:
            slides.setdefault(row // rows_per_slide, []).append(row)
        assert set(slides) == {row // rows_per_slide for row in full_train}
        assert all(len(rows) == kept for rows in slides.values())
        assert set(thin_train) <= set(full_train)

        full_counts = {c: [cell(r) for r in full_train].count(c) for c in map(cell, full_train)}
        thin_counts = {c: [cell(r) for r in thin_train].count(c) for c in map(cell, thin_train)}
        assert thin_counts == {c: n * kept // rows_per_slide for c, n in full_counts.items()}

        assert set(thin_val) <= set(full_val)
        for c in set(map(cell, full_val)):
            full_n = [cell(r) for r in full_val].count(c)
            thin_n = [cell(r) for r in thin_val].count(c)
            assert thin_n == max(1, full_n * kept // rows_per_slide)


@pytest.mark.parametrize("train_rows_per_slide", [0, ROWS_PER_SLIDE + 1])
def test_a_slide_cannot_contribute_fewer_than_one_tile_or_more_than_it_carries(
    train_rows_per_slide: int,
) -> None:
    embeddings, confounders, labels = _synthetic_cells()

    with pytest.raises(ValueError, match="train_rows_per_slide"):
        probe_sweep(
            embeddings,
            confounders,
            labels,
            schedule=_schedule(),
            rows_per_slide=ROWS_PER_SLIDE,
            train_rows_per_slide=train_rows_per_slide,
        )


def test_thinning_needs_training_blocks_made_of_whole_slides() -> None:
    # "Each training slide keeps n rows" has no meaning for a block that ends mid-slide.
    embeddings, confounders, labels = _synthetic_cells()
    (split_map, max_train_slides), confounded = _schedule()
    ragged = [(i, j, take - 1 if (i, j) == (0, 0) else take) for i, j, take in split_map]

    with pytest.raises(ValueError, match="whole number"):
        probe_sweep(
            embeddings,
            confounders,
            labels,
            schedule=[(ragged, max_train_slides), confounded],
            rows_per_slide=ROWS_PER_SLIDE,
            train_rows_per_slide=1,
        )
