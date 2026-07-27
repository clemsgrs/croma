"""The confounder-biased probe sweep: the accuracy matrix both downstream reductions read.

``apd`` and ``napd`` reduce a matrix they cannot produce. This module produces it, and it
is the harder half: the matrix comes from training a biology probe on a training set whose
confounder-biology correlation the caller walks from zero to total, and scoring it on
balanced test sets that never move. The schedule, the probe and the regularisation search
are PathoROB's own code (``croma.downstream._pathorob``); what is authored here is the
sweep that drives them and the argument validation the vendored code deliberately lacks.

The module consumes embeddings and a split assignment. It loads no model, reads no
manifest and knows nothing about where a repository keeps its files -- those belong to
whatever driver assembled the embeddings, and keeping them out is what lets the protocol
ship as library code at all (ADR-0011).

Cohorts differ in one place, and upstream's own loop differs there too: how a replicate
orders each cell's slides before the schedule is cut out of them. PathoROB shuffles, except
on Tolkach-ESCA, where it draws a feasible train/test *case* split per replicate and pushes
those cases to the test tail, because a case there contributes patches to several
biological classes at once. That is a property of one cohort's annotation rather than of
the protocol, so it is not built in -- it is reachable, through the ``arrange_slides`` hook
of :func:`probe_sweep_over_test_sets`, whose default is the shuffle.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from croma.downstream._pathorob import get_patches_map_to_split, train_logistic_regression

#: How many splits each PathoROB dataset's schedule runs for. These are upstream's own
#: numbers, but they live in its loader rather than in the helper croma vendored, so they
#: are restated here -- a caller who had to supply them would be supplying the one thing
#: that decides how far the confounder bias is walked.
PATHOROB_SPLITS = {"camelyon": 8, "tcga": 7, "tolkach_esca": 4}

#: One split of a schedule: how many training rows each ``(confounder, class)`` cell
#: contributes at that split, and the largest number of *slides* any cell contributes at
#: any split. The second number is not redundant -- it is what fixes the validation slice
#: and where the held-out tail starts, and it must not move across splits or the test set
#: would change shape with the confounder bias it is supposed to be independent of.
SplitPlan = tuple[Sequence[tuple[int, int, int]], int]

#: One slide, as row indices into the caller's embeddings.
Slide = Sequence[int]

#: Every cell's slides: ``cells[confounder][class]`` holds that cell's slides, in the order
#: the sweep will lay its rows out -- training taken off the front, held-out rows from the
#: tail. Slides rather than rows, because a slide is the unit the protocol keeps whole.
SlideCells = Sequence[Sequence[Sequence[Slide]]]

#: How one replicate orders each cell's slides. Receives the cells in input order and that
#: replicate's generator, and returns the cells reordered -- the same slides, rearranged.
SlideArrangement = Callable[[SlideCells, random.Random], SlideCells]

#: The key the held-out in-domain matrix is returned under. Named rather than positional
#: because a caller's own test sets sit beside it in the same mapping.
IN_DOMAIN = "in_domain"

#: Upstream draws each replicate's seed from ``range(0, 10000)`` without replacement, so
#: that range also caps how many replicates a sweep can have.
_SEED_POOL = 10000


def pathorob_schedule(
    dataset: str, *, rows_per_slide: int, n_splits: int | None = None
) -> list[SplitPlan]:
    """PathoROB's own schedule for one of its three downstream datasets.

    A schedule is the sequence of training compositions a sweep walks, from balanced to
    fully confounded. This one is not croma's: it comes from the split-mapping helper
    vendored verbatim from PathoROB, so a sweep run on it is the reference protocol rather
    than something resembling it. Pass the result straight to :func:`probe_sweep`.

    Cells are addressed as ``(confounder, class)`` in the order PathoROB lists them --
    ``camelyon`` as ``RUMC, UMCU`` x ``normal, tumor``, and so on -- so the caller's
    ``confounders`` and ``labels`` have to be indexed the same way. Datasets with their own
    schedule, croma's included, build the sequence themselves; nothing here is required.

    Args:
        dataset: ``"camelyon"``, ``"tcga"`` or ``"tolkach_esca"`` -- PathoROB's names, kept
            as PathoROB spells them (note ``"tcga"`` for the 4x4 cohort).
        rows_per_slide: How many rows one slide contributes, which is what the schedule's
            per-slide counts are multiplied by. PathoROB publishes 300 for ``camelyon``,
            30 for ``tcga`` and 100 for ``tolkach_esca``; a slide-level sweep passes ``1``.
        n_splits: How far to walk the schedule. Defaults to the number of splits PathoROB
            runs for this dataset (:data:`PATHOROB_SPLITS`), and cannot exceed it -- past
            that point the formulas run the favourable cells negative, so a longer walk
            would not be more confounded, it would be undefined. A shorter one is a prefix
            of the reference protocol, which is what a smoke run wants.

    Returns:
        One :data:`SplitPlan` per split, balanced baseline first.

    Raises:
        ValueError: If ``dataset`` is not one PathoROB has a schedule for, if
            ``rows_per_slide`` is not positive, or if ``n_splits`` is not positive or runs
            past the schedule's last split.
    """
    if dataset not in PATHOROB_SPLITS:
        raise ValueError(
            f"PathoROB has no schedule for {dataset!r}; it defines "
            f"{sorted(PATHOROB_SPLITS)}. A dataset of your own needs a schedule of its "
            "own -- build the sequence and pass it to probe_sweep directly."
        )
    if rows_per_slide < 1:
        raise ValueError(f"rows_per_slide must be at least 1, got {rows_per_slide}")
    defined = PATHOROB_SPLITS[dataset]
    splits = defined if n_splits is None else n_splits
    if splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {splits}")
    if splits > defined:
        raise ValueError(
            f"PathoROB's {dataset!r} schedule has {defined} splits, so it cannot be walked "
            f"for {splits}; past the last one the favourable cells go negative"
        )
    return [get_patches_map_to_split(dataset, split, rows_per_slide) for split in range(splits)]


def probe_sweep(
    embeddings: ArrayLike,
    confounders: ArrayLike,
    labels: ArrayLike,
    *,
    schedule: Sequence[SplitPlan],
    rows_per_slide: int = 1,
    iterations: int = 20,
    seed: int = 1000,
    validation_fraction: float | None = None,
) -> np.ndarray:
    """Run the confounder-biased probe sweep and return its accuracy matrix.

    One row per split, one column per resampling replicate::

        accuracies[s][i] = balanced accuracy of the probe trained on split s, replicate i

    Row ``0`` is whatever the caller put first in ``schedule`` -- by construction the
    balanced baseline, the split at which the confounder carries no information about the
    biological class. Every later row is a more confounded split. That is exactly the
    matrix :func:`croma.apd` and :func:`croma.napd` reduce, so the return value goes
    into either of them untouched.

    Each replicate reshuffles every ``(confounder, class)`` cell by whole slides, then
    walks the schedule, cutting each split's training rows off the front of the cells, its
    validation rows from directly behind them, and its test rows from a tail that starts
    beyond the widest training block any split asks for. The tail therefore holds the same
    rows for every split, which is what makes accuracies across splits comparable: the
    training composition moves, the test composition does not.

    Args:
        embeddings: ``(n_rows, n_features)`` frozen representations. Rows are grouped into
            slides by position, so they must arrive in the order the slides do.
        confounders: ``(n_rows,)`` confounder index per row -- the medical centre, scanner
            or provider whose influence is being injected. These are the ``i`` of a
            schedule entry, so they index it and must run ``0 .. n_confounders - 1``.
        labels: ``(n_rows,)`` biological class index per row: the ``j`` of a schedule
            entry, running ``0 .. n_classes - 1``. This is what the probe predicts.
        schedule: One :data:`SplitPlan` per split, the balanced baseline first. Two splits
            at minimum, since a sweep with no confounded split reduces to nothing.
        rows_per_slide: How many consecutive rows make up one slide. Slides, not rows, are
            the unit that is shuffled and counted, so train and test never share one.
        iterations: How many resampling replicates to run -- the width of the matrix.
        seed: Seed of the replicate seeds. The sweep is a pure function of it, so two runs
            with the same seed return the same matrix.
        validation_fraction: Size of each cell's validation slice, as a fraction of its
            training rows. ``None`` selects PathoROB's own rule, ``1 / max_train_slides``
            of the cell's training rows, which is the faithful setting for a sweep over
            patches. It underflows to zero when a slide *is* one row, which is why a
            slide-level sweep has to state the fraction instead.

    Returns:
        The ``(n_splits, iterations)`` matrix of balanced accuracies on the held-out
        in-domain rows.

    Raises:
        ValueError: If the inputs are not aligned rectangular arrays of non-negative
            integer indices; if ``rows_per_slide`` or ``iterations`` is not positive, or
            ``iterations`` exceeds the 10,000 seeds a replicate can be drawn from; if
            ``validation_fraction`` is outside ``[0, 1)``; or if the schedule holds fewer
            than two splits, names a cell no row carries, asks a cell for more rows than
            it holds or than ``max_train_slides`` allows, or leaves some split with no
            training, validation or test rows. Those last few are the same fault wearing
            different clothes: a schedule and a cohort that were not written for each
            other, which would otherwise run to completion at a confounder bias nobody
            asked for.
    """
    return probe_sweep_over_test_sets(
        embeddings,
        confounders,
        labels,
        schedule=schedule,
        rows_per_slide=rows_per_slide,
        iterations=iterations,
        seed=seed,
        validation_fraction=validation_fraction,
    )[IN_DOMAIN]


def probe_sweep_over_test_sets(
    embeddings: ArrayLike,
    confounders: ArrayLike,
    labels: ArrayLike,
    *,
    schedule: Sequence[SplitPlan],
    test_sets: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
    rows_per_slide: int = 1,
    iterations: int = 20,
    seed: int = 1000,
    validation_fraction: float | None = None,
    arrange_slides: SlideArrangement | None = None,
) -> dict[str, np.ndarray]:
    """Run one sweep and score every probe it trains on more than one test set.

    Same protocol as :func:`probe_sweep`, and its in-domain matrix is identical -- this
    form only adds test sets that were never trained on, typically rows from confounders
    the sweep never saw. They ride along on the probes the sweep already trained, so
    asking for an unseen-confounder matrix costs one prediction per split and replicate
    rather than a second sweep.

    Args:
        embeddings: As :func:`probe_sweep`.
        confounders: As :func:`probe_sweep`.
        labels: As :func:`probe_sweep`.
        schedule: As :func:`probe_sweep`.
        test_sets: Extra test sets as ``{name: (embeddings, labels)}``. Their labels are
            biological class indices, as ``labels`` is; they carry no confounder index,
            because nothing about a test set is confounder-dependent. Their names come
            back as the keys of the result and cannot be ``"in_domain"``.
        rows_per_slide: As :func:`probe_sweep`.
        iterations: As :func:`probe_sweep`.
        seed: As :func:`probe_sweep`.
        validation_fraction: As :func:`probe_sweep`.
        arrange_slides: How one replicate orders each cell's slides -- the step that
            decides which slides a split trains on and which sit in the held-out tail.
            ``None`` selects the sweep's own step, a shuffle of every cell with the
            replicate's generator, which is the reference protocol. Pass a
            :data:`SlideArrangement` for a cohort whose slides cannot be ordered freely:
            PathoROB's Tolkach-ESCA sweep, for one, draws a train/test *case* split per
            replicate and pushes those cases to the tail, because a case there carries
            patches of several biological classes and may not be trained on and tested on
            at once. An arrangement receives the cells in input order and returns them
            reordered; it may draw from the generator it is given, and *must* draw before
            it shuffles for its sweep to match a driver that does, since the two share one
            stream. It may not add, drop or break up a slide.

    Returns:
        ``{"in_domain": matrix}`` plus one ``(n_splits, iterations)`` matrix per named
        test set. Each is a well-formed input for :func:`croma.apd` and
        :func:`croma.napd`.

    Raises:
        ValueError: Everything :func:`probe_sweep` raises, plus a test set whose
            embeddings and labels disagree in length, whose feature count differs from
            ``embeddings``, that is empty, or that is named ``"in_domain"``; and an
            ``arrange_slides`` that returns anything but a rearrangement of the slides it
            was handed.
    """
    features, confounder_of_row, label_of_row = _validated_rows(embeddings, confounders, labels)
    named_tests = _validated_test_sets(test_sets, n_features=features.shape[1])
    cells = _cells(confounder_of_row, label_of_row)
    plans = _validated_schedule(
        schedule,
        cells=cells,
        rows_per_slide=rows_per_slide,
        validation_fraction=validation_fraction,
        n_classes=len(cells[0]),
    )
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    if iterations > _SEED_POOL:
        raise ValueError(f"iterations must be at most {_SEED_POOL}, got {iterations}")

    arrangement = _shuffle_slides if arrange_slides is None else arrange_slides
    names = [IN_DOMAIN, *named_tests]
    accuracies = {name: np.empty((len(plans), iterations), dtype=float) for name in names}
    replicate_seeds = random.Random(seed).sample(range(0, _SEED_POOL), iterations)

    for column, replicate_seed in enumerate(replicate_seeds):
        _arrange(cells, rows_per_slide, arrangement, random.Random(replicate_seed))
        for row, plan in enumerate(plans):
            train, validation, held_out = plan.slice(cells)
            _, _, scores = train_logistic_regression(
                features[train],
                label_of_row[train],
                features[validation],
                label_of_row[validation],
                [features[held_out], *(test_x for test_x, _ in named_tests.values())],
                [label_of_row[held_out], *(test_y for _, test_y in named_tests.values())],
            )
            for name, score in zip(names, scores):
                accuracies[name][row, column] = score
    return accuracies


class _Plan:
    """One validated split: the row counts it takes, and where its slices fall."""

    def __init__(
        self,
        split_map: Sequence[tuple[int, int, int]],
        max_train_slides: int,
        *,
        rows_per_slide: int,
        validation_fraction: float | None,
    ) -> None:
        self.split_map = [(int(i), int(j), int(take)) for i, j, take in split_map]
        self.max_train_rows = max_train_slides * rows_per_slide
        if validation_fraction is None:
            # PathoROB's rule: each cell reserves the number of rows one training slide
            # would carry if the widest training block were spread over max_train_slides.
            self.validation_take = [int(take / max_train_slides) for _, _, take in self.split_map]
            self.first_held_out = (max_train_slides + 1) * rows_per_slide
        else:
            self.validation_take = [
                round(validation_fraction * take) for _, _, take in self.split_map
            ]
            self.first_held_out = (
                max_train_slides + round(validation_fraction * max_train_slides) + 1
            ) * rows_per_slide

    def slice(self, cells: list[list[list[int]]]) -> tuple[list[int], list[int], list[int]]:
        """Cut this split's training, validation and held-out row indices out of the cells.

        Training rows come off the front of each cell, validation directly behind them, and
        the held-out tail from a fixed offset every split shares -- so the test rows are the
        same rows at every confounder bias, and only the training composition moves.
        """
        train = [row for i, j, take in self.split_map for row in cells[i][j][:take]]
        validation = [
            row
            for (i, j, take), reserve in zip(self.split_map, self.validation_take)
            for row in cells[i][j][take : take + reserve]
        ]
        held_out = [
            row for cell_row in cells for cell in cell_row for row in cell[self.first_held_out :]
        ]
        return train, validation, held_out


def _validated_rows(
    embeddings: ArrayLike, confounders: ArrayLike, labels: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(embeddings, dtype=float)
    if features.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2-D (n_rows, n_features) matrix, got {features.ndim}-D"
        )
    if not np.isfinite(features).all():
        raise ValueError("embeddings must be finite")

    indices = []
    for name, values in (("confounders", confounders), ("labels", labels)):
        column = np.asarray(values)
        if column.ndim != 1 or len(column) != len(features):
            raise ValueError(
                f"{name} must be a 1-D array with one entry per embedding row "
                f"({len(features)}), got shape {column.shape}"
            )
        if not np.issubdtype(column.dtype, np.integer):
            raise ValueError(
                f"{name} must hold integer indices into the schedule, got dtype "
                f"{column.dtype}; map names to indices before calling"
            )
        if len(column) and column.min() < 0:
            raise ValueError(f"{name} must hold non-negative indices, got {column.min()}")
        indices.append(column)
    if not len(features):
        raise ValueError("embeddings must hold at least one row")
    return features, indices[0], indices[1]


def _validated_test_sets(
    test_sets: Mapping[str, tuple[ArrayLike, ArrayLike]] | None, *, n_features: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    validated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (test_embeddings, test_labels) in (test_sets or {}).items():
        if name == IN_DOMAIN:
            raise ValueError(
                f"{IN_DOMAIN!r} is the key the sweep's own held-out matrix comes back "
                "under, so a test set cannot take that name"
            )
        test_x = np.asarray(test_embeddings, dtype=float)
        test_y = np.asarray(test_labels)
        if test_x.ndim != 2 or test_x.shape[1] != n_features:
            raise ValueError(
                f"test set {name!r} must be a 2-D matrix with {n_features} features, "
                f"got shape {test_x.shape}"
            )
        if len(test_y) != len(test_x):
            raise ValueError(
                f"test set {name!r} must hold one label per row, got {len(test_y)} "
                f"labels for {len(test_x)} rows"
            )
        if not len(test_x):
            raise ValueError(f"test set {name!r} must hold at least one row")
        validated[name] = (test_x, test_y)
    return validated


def _validated_schedule(
    schedule: Sequence[SplitPlan],
    *,
    cells: list[list[list[int]]],
    rows_per_slide: int,
    validation_fraction: float | None,
    n_classes: int,
) -> list[_Plan]:
    if rows_per_slide < 1:
        raise ValueError(f"rows_per_slide must be at least 1, got {rows_per_slide}")
    if validation_fraction is not None and not 0.0 <= validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must lie in [0, 1), got {validation_fraction}")
    if len(schedule) < 2:
        raise ValueError(
            "schedule must hold the balanced baseline and at least one confounded split "
            f"after it, got {len(schedule)} split(s)"
        )

    plans = []
    for split, (split_map, max_train_slides) in enumerate(schedule):
        if max_train_slides < 1:
            raise ValueError(
                f"split {split}: max_train_slides must be at least 1, got {max_train_slides}"
            )
        plan = _Plan(
            split_map,
            max_train_slides,
            rows_per_slide=rows_per_slide,
            validation_fraction=validation_fraction,
        )
        for (i, j, take), reserve in zip(plan.split_map, plan.validation_take):
            if not 0 <= i < len(cells) or not 0 <= j < n_classes:
                raise ValueError(
                    f"split {split}: schedule names cell ({i}, {j}), which no row carries"
                )
            if take < 0:
                raise ValueError(f"split {split}: cell ({i}, {j}) asks for {take} rows")
            if take > plan.max_train_rows:
                # Every schedule spends its whole training budget on one cell at its most
                # confounded split, so a cell above the bound is not a schedule croma can
                # read -- in practice it is a schedule built for a different
                # rows_per_slide than the sweep was given, which would also put the
                # held-out tail (counted in slides) in the wrong place.
                raise ValueError(
                    f"split {split}: cell ({i}, {j}) asks for {take} training rows, more "
                    f"than max_train_slides allows ({plan.max_train_rows} rows at "
                    f"{rows_per_slide} row(s) per slide). A schedule and the sweep that "
                    "runs it must agree on how many rows a slide carries."
                )
            if take + reserve > len(cells[i][j]):
                raise ValueError(
                    f"split {split}: cell ({i}, {j}) holds {len(cells[i][j])} rows, too "
                    f"few for the {take} training and {reserve} validation rows the "
                    "schedule asks it for"
                )
        train, validation, held_out = plan.slice(cells)
        for what, rows in (("training", train), ("validation", validation), ("test", held_out)):
            if not rows:
                raise ValueError(
                    f"split {split}: the schedule leaves no {what} rows -- the cells are "
                    "too small for the row counts and held-out tail it asks for"
                )
        plans.append(plan)
    return plans


def _cells(confounders: np.ndarray, labels: np.ndarray) -> list[list[list[int]]]:
    """Row indices grouped by ``(confounder, class)`` cell, each cell in input order."""
    cells: list[list[list[int]]] = [
        [[] for _ in range(int(labels.max()) + 1)] for _ in range(int(confounders.max()) + 1)
    ]
    for row, (confounder, label) in enumerate(zip(confounders, labels)):
        cells[int(confounder)][int(label)].append(row)
    return cells


def _shuffle_slides(cells: list[list[list[list[int]]]], rng: random.Random) -> SlideCells:
    """The sweep's own arrangement, and the reference protocol's: every cell reshuffled.

    The default because it is what makes a split's training rows a random sample of its
    cells rather than whichever rows the caller happened to list first.
    """
    for cell_row in cells:
        for slides in cell_row:
            rng.shuffle(slides)
    return cells


def _arrange(
    cells: list[list[list[int]]],
    rows_per_slide: int,
    arrangement: SlideArrangement,
    rng: random.Random,
) -> None:
    """Reorder every cell for one replicate, in place, by whole slides.

    Whole slides, because a slide's rows are near-duplicates of each other: split one
    across the training and test sets and the probe is scored on tissue it was fitted on.
    Which is why the arrangement is handed slides rather than rows -- it decides their
    order and nothing else. Cells stay arranged between replicates, as upstream's do.
    """
    slides = [
        [
            [cell[start : start + rows_per_slide] for start in range(0, len(cell), rows_per_slide)]
            for cell in cell_row
        ]
        for cell_row in cells
    ]
    arranged = arrangement(slides, rng)
    _check_same_slides(arranged, slides)
    for cell_row, arranged_row in zip(cells, arranged):
        for index, cell in enumerate(arranged_row):
            cell_row[index] = [row for slide in cell for row in slide]


def _check_same_slides(arranged: SlideCells, slides: SlideCells) -> None:
    """Refuse an arrangement that is not a rearrangement.

    A schedule was written for a cohort of a known shape, so an arrangement that loses a
    slide, invents one or breaks one apart quietly re-shapes the cohort underneath it: the
    sweep would still return a matrix, computed on rows the caller never put there.
    """
    complaint = (
        "arrange_slides must return the same slides it was handed, reordered -- "
        "one list per confounder, one per class inside it, holding that cell's slides"
    )
    if len(arranged) != len(slides):
        raise ValueError(f"{complaint}; got {len(arranged)} confounder(s), not {len(slides)}")
    for arranged_row, row in zip(arranged, slides):
        if len(arranged_row) != len(row):
            raise ValueError(f"{complaint}; got {len(arranged_row)} class(es), not {len(row)}")
        for arranged_cell, cell in zip(arranged_row, row):
            if sorted(map(tuple, arranged_cell)) != sorted(map(tuple, cell)):
                raise ValueError(complaint)
