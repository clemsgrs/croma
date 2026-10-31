"""Do low per-sample margins predict the errors a shortcut-trained probe makes?

CRoMa is validated against shortcut susceptibility only between encoders: median CRoMa
against nIPD. This study asks the per-sample question. The confounder-biased probe sweep
(``croma.downstream``) trains a biology probe on a training set walked from balanced
(Cramer's V = 0) to fully confounded (V = 1), and the probe records
(``scripts/studies/apd/probe_records.py``) keep every held-out row's decision margins at
every split. A sample the balanced probe always gets right and the fully confounded one gets
wrong at least half the time has been *flipped* by the shortcut. Is a flipped sample more
likely to carry a low CRoMa than an unflipped one from the same class x centre cell, and
does CRoMa know more than two baselines built from the same embeddings -- how cleanly the
sample sits among its own class (the biology-only margin) and how much of its neighbourhood
comes from its own centre (centre purity)?
"""

from __future__ import annotations

import os

# One BLAS thread per process: the study fans out across encoders instead. Must precede the
# numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (REPO / "src", HERE, HERE / "apd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import probe_records  # noqa: E402
import tile_colour  # noqa: E402

from croma.metrics.bootstrap import (  # noqa: E402
    BootstrapCI,
    _percentile_ci,
    cluster_bootstrap_draws,
    partial_spearman_inference,
    spearman_inference,
)
from croma.metrics.croma import (  # noqa: E402
    CROMA_HEADLINE_M,
    _compute_sample_croma,
    _iterative_typed_neighbor_search,
)
from croma.metrics.neighbors import _prepare_neighbors  # noqa: E402

#: Where the typed search starts and how fast it grows: CRoMa's own defaults.
SEARCH_START_K = 200
SEARCH_GROWTH = 2.0

#: A sample is *flipped* when the balanced probe never errs on it and the fully confounded
#: probe errs on it in at least this share of the replicates that held it out.
FLIP_ERROR_RATE = 0.5


def _true_class_margin(margins: np.ndarray, truth: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """How far a probe's decision leans towards each row's true class.

    Binary probes store one signed decision value that points towards ``classes[1]``, so the
    margin is that value signed by the row's class. Multiclass probes store one score per
    class: the margin is the true class's score minus the strongest other class's.
    """
    if margins.shape[-1] == 1:
        return np.where(truth == classes[1], margins[:, 0], -margins[:, 0])
    position = np.searchsorted(classes, truth)
    own = np.take_along_axis(margins, position[:, None], axis=1)[:, 0]
    others = margins.copy()
    np.put_along_axis(others, position[:, None], -np.inf, axis=1)
    return own - others.max(axis=1)


def sample_targets(record) -> pd.DataFrame:
    """Per held-out row of one probe record: errors and margins at V = 0 and at V = 1.

    ``e0`` and ``e1`` are the shares of the replicates holding the row out in which the
    probe of the first split (Cramer's V = 0, balanced) and of the last (V = 1, fully
    confounded) misclassifies it, exactly as the probe's ``predict`` would
    (``probe_records._predicted``). ``margin0``/``margin1`` are its mean true-class margins
    there and ``margin_drop = margin0 - margin1``. ``flipped`` is ``e0 == 0`` and
    ``e1 >= FLIP_ERROR_RATE``. Rows no replicate held out have no target and are absent;
    ``row`` is the position in the record's ID rows, what ``join_per_sample`` returns.

    Raises:
        ValueError: If the record's schedule does not run from V = 0 to V = 1.
    """
    meta = record["meta"]
    schedule = [float(v) for v in meta["cramers_v"]]
    if len(schedule) != meta["n_splits"] or schedule[0] != 0.0 or schedule[-1] != 1.0:
        raise ValueError(
            f"{meta['model']}: the schedule must run from V = 0 to V = 1, got {schedule}"
        )
    classes = np.asarray(record["classes"])
    by_name = {name: index for index, name in enumerate(meta["biological_classes"])}
    labels = np.array([by_name[label] for label in record["id_label"]])
    held_out = np.asarray(record["held_out"])
    rows = held_out.ravel()
    truth = labels[rows]
    n_rows = len(labels)
    counts = np.bincount(rows, minlength=n_rows)
    out = {"row": np.flatnonzero(counts)}
    out["n_held_out"] = counts[out["row"]]
    for tag, split in (("0", 0), ("1", meta["n_splits"] - 1)):
        margins = np.asarray(record["margins_in_domain"][split], dtype=float)
        margins = margins.reshape(rows.size, margins.shape[-1])
        wrong = probe_records._predicted(margins, classes) != truth
        margin = _true_class_margin(margins, truth, classes)
        out[f"e{tag}"] = np.bincount(rows, wrong, minlength=n_rows)[out["row"]] / out["n_held_out"]
        out[f"margin{tag}"] = (
            np.bincount(rows, margin, minlength=n_rows)[out["row"]] / out["n_held_out"]
        )
    targets = pd.DataFrame(out)
    targets["margin_drop"] = targets["margin0"] - targets["margin1"]
    targets["flipped"] = (targets["e0"] == 0) & (targets["e1"] >= FLIP_ERROR_RATE)
    return targets


#: The per-sample columns the study reads from a run; everything else comes from the record.
PER_SAMPLE_COLUMNS = ("model", "sample_id", "group_id", "croma_m5")


def never_held_out(record) -> int:
    """How many of a record's ID rows no replicate held out: rows without a target."""
    return int(len(record["id_sample_id"]) - len(np.unique(record["held_out"])))


def join_targets(record, per_sample: pd.DataFrame) -> pd.DataFrame:
    """One encoder's held-out rows: tile key, class x centre ``cell``, targets and CRoMa.

    The record's held-out rows are matched to the run's per-sample rows by
    ``probe_records.join_per_sample`` (tile key ``sample_id`` + ``group_id``, the record's
    encoder only), which refuses an unmatched or doubly matched row; the targets are then
    attached by record row. Class names and centres are the record's, so a run that codes
    its classes differently (``panda``'s ``0``/``1``) still yields named cells.
    """
    present = [column for column in PER_SAMPLE_COLUMNS if column in per_sample]
    joined = probe_records.join_per_sample(record, per_sample[present])
    joined = joined.merge(sample_targets(record), on="row", how="inner", validate="one_to_one")
    joined["cell"] = joined["label"].astype(str) + " | " + joined["confounder"].astype(str)
    return joined.sort_values("row", kind="stable").reset_index(drop=True)


def _scan_same_and_other_class(
    *,
    labels: np.ndarray,
    centers: np.ndarray,
    query_indices: np.ndarray,
    neigh_idx: np.ndarray,
    neigh_dist: np.ndarray,
    valid_counts: np.ndarray,
    m: int,
    so_dists: np.ndarray,
    os_dists: np.ndarray,
) -> np.ndarray:
    """Type CRoMa's cross-slide neighbours by class alone, whatever their centre.

    The first array takes the ``m`` nearest same-class neighbours, the second the ``m``
    nearest other-class ones -- the slots ``_iterative_typed_neighbor_search`` fills with
    SO and OS for CRoMa. ``centers`` is accepted and ignored: that is the point.
    """
    del centers
    defined = np.zeros(len(query_indices), dtype=bool)
    for row, sample in enumerate(query_indices.tolist()):
        same = other = 0
        so_dists[sample, :] = np.inf
        os_dists[sample, :] = np.inf
        for pos in range(int(valid_counts[row])):
            j = int(neigh_idx[row, pos])
            if j < 0:
                continue
            if labels[j] == labels[sample]:
                if same < m:
                    so_dists[sample, same] = float(neigh_dist[row, pos])
                    same += 1
            elif other < m:
                os_dists[sample, other] = float(neigh_dist[row, pos])
                other += 1
            if same >= m and other >= m:
                defined[row] = True
                break
    return defined


def _unit_rows(features: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length exactly as CRoMa scales them before its search."""
    features = np.asarray(features)
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)


def biology_only_margin(features, labels, group_ids, *, m: int = CROMA_HEADLINE_M) -> np.ndarray:
    """``(d_diff - d_same) / (d_diff + d_same)`` per sample: the intrinsic-difficulty baseline.

    ``d_same`` is the mean cosine distance to the ``m`` nearest neighbours of the sample's
    own class and ``d_diff`` to the ``m`` nearest of any other class, both from *any* centre
    and never from the sample's own independence group (slide), as in CRoMa. It is CRoMa
    with the centre taken out of the typing, and it runs CRoMa's own adaptive neighbour
    search (``croma.metrics.croma._iterative_typed_neighbor_search``) with a class-only scan.

    Raises:
        RuntimeError: If a sample cannot find ``m`` neighbours of both kinds outside its
            slide, or both distances are zero.
    """
    labels = np.asarray(labels)
    codes = np.unique(labels, return_inverse=True)[1].reshape(-1)
    same, other, _meta = _iterative_typed_neighbor_search(
        features=_unit_rows(features),
        labels=codes,
        centers=np.zeros(len(codes), dtype=int),
        group_ids=np.asarray(group_ids).astype(str),
        m=int(m),
        start_k=SEARCH_START_K,
        k_growth_factor=SEARCH_GROWTH,
        scan=_scan_same_and_other_class,
    )
    margin = _compute_sample_croma(same, other)
    if not np.isfinite(margin).all():
        raise RuntimeError(
            f"{int((~np.isfinite(margin)).sum())} sample(s) have no biology-only margin: fewer "
            f"than {m} same- and other-class neighbours outside their slide"
        )
    return margin


class WithinCellAuroc:
    """The AUROC of ``score`` for ``event`` with pairs formed only inside cells.

    It is the probability that an event outscores a non-event *from the same cell*, ties
    counting half: every (event, non-event) pair that shares a cell is one comparison, and a
    cell lacking either kind contributes none. Pooling cells would let a cell that both
    scores high and holds many events manufacture an association no cell shows.

    The rows are sorted once; ``__call__`` then takes per-row ``weights`` -- how often a
    bootstrap replicate drew each row -- in linear time, so a weighted call equals an
    unweighted call on the rows repeated that often.
    """

    def __init__(self, score, event, cells):
        score = np.asarray(score, dtype=float)
        event = np.asarray(event, dtype=bool)
        if score.shape != event.shape or score.ndim != 1:
            raise ValueError("score and event must be one-dimensional and the same length")
        if not np.isfinite(score).all():
            raise ValueError("score must be finite")
        codes = np.unique(np.asarray(cells), return_inverse=True)[1].reshape(-1)
        if codes.shape != score.shape:
            raise ValueError("cells must hold one label per row")
        self._order = np.lexsort((score, codes))
        s, c = score[self._order], codes[self._order]
        starts = np.ones(len(s), dtype=bool)  # at the rows' own length, so none gives no group
        starts[1:] = (c[1:] != c[:-1]) | (s[1:] != s[:-1])
        self._group = np.cumsum(starts) - 1
        self._group_cell = c[starts]
        self._n_groups = int(self._group[-1]) + 1 if len(s) else 0
        self._n_cells = int(codes.max()) + 1 if len(s) else 0
        self._event = event[self._order]
        self.n = len(score)

    def __call__(self, weights=None) -> float:
        w = np.ones(self.n) if weights is None else np.asarray(weights, dtype=float)[self._order]
        positive = np.bincount(self._group, w * self._event, minlength=self._n_groups)
        negative = np.bincount(self._group, w * ~self._event, minlength=self._n_groups)
        cell_negative = np.bincount(self._group_cell, negative, minlength=self._n_cells)
        cell_positive = np.bincount(self._group_cell, positive, minlength=self._n_cells)
        before_cell = (np.cumsum(cell_negative) - cell_negative)[self._group_cell]
        below = np.cumsum(negative) - negative - before_cell
        wins = float((positive * (below + 0.5 * negative)).sum())
        pairs = float((cell_positive * cell_negative).sum())
        return wins / pairs if pairs > 0 else float("nan")


def within_cell_auroc(score, event, cells, weights=None) -> float:
    """``WithinCellAuroc(score, event, cells)(weights)``: one within-cell AUROC."""
    return WithinCellAuroc(score, event, cells)(weights)


@dataclass(frozen=True)
class AurocBootstrap:
    """Within-cell AUROCs and paired differences, each a point with a percentile CI."""

    auroc: dict[str, BootstrapCI]
    difference: dict[tuple[str, str], BootstrapCI]


def bootstrap_aurocs(
    predictors,
    *,
    event,
    cells,
    groups,
    differences=(),
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> AurocBootstrap:
    """Within-cell AUROCs of several predictors for one event, on one shared cluster draw.

    ``predictors`` maps a name to ``(score, strata)``: the score whose AUROC is wanted and
    the cells its pairs are formed in (the class x centre cells, or a finer partition such
    as cell x colour quintile). Every replicate is one draw of the stratified slide-cluster
    bootstrap (``croma.metrics.bootstrap.cluster_bootstrap_draws``, slides resampled within
    ``cells`` so no replicate loses a cell), applied to every predictor alike, so the CI of
    a difference ``(a, b)`` -- ``AUROC_a - AUROC_b`` -- is paired.
    """
    prepared = {
        name: WithinCellAuroc(score, event, strata) for name, (score, strata) in predictors.items()
    }
    n = len(np.asarray(event))
    replicates = {name: np.empty(int(n_boot)) for name in prepared}
    draws = cluster_bootstrap_draws(
        np.asarray(groups), strata=np.asarray(cells), n_boot=int(n_boot), seed=int(seed)
    )
    for b, rows in enumerate(draws):
        weights = np.bincount(rows, minlength=n)
        for name, auroc in prepared.items():
            replicates[name][b] = auroc(weights)
    points = {name: auroc() for name, auroc in prepared.items()}
    return AurocBootstrap(
        auroc={
            name: _percentile_ci(replicates[name], points[name], level, int(n_boot))
            for name in prepared
        },
        difference={
            (a, b): _percentile_ci(
                replicates[a] - replicates[b], points[a] - points[b], level, int(n_boot)
            )
            for a, b in differences
        },
    )


def _cell_rho(x, y, z=None) -> float:
    """Spearman's rho, or partial rho given ``z``, from the library's #220 estimators.

    Only the point estimate is used: their intervals and p-values treat rows as
    exchangeable encoders, and these rows are tiles clustered in slides. One bootstrap
    replicate and one permutation keep the unused machinery cheap.
    """
    try:
        if z is None:
            return spearman_inference(x, y, n_boot=1, n_permutations=1).rho
        return partial_spearman_inference(x, y, z, n_boot=1, n_permutations=1).rho
    except ValueError:  # fewer than three rows, or a constant vector: no rho
        return float("nan")


def within_cell_spearman(x, y, cells, *, z=None) -> float:
    """Spearman's rho of ``x`` and ``y`` inside each cell, averaged weighting cells by size.

    With ``z``, the partial rho given ``z`` (``croma.metrics.bootstrap``'s rank-residual
    estimator) inside each cell. A cell where rho is undefined (fewer than three rows, or a
    constant variable) is left out of the average.
    """
    x, y, cells = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(cells)
    z = None if z is None else np.asarray(z, dtype=float)
    total = weight = 0.0
    for cell in np.unique(cells):
        rows = cells == cell
        rho = _cell_rho(x[rows], y[rows], None if z is None else z[rows])
        if np.isfinite(rho):
            total += rows.sum() * rho
            weight += rows.sum()
    return total / weight if weight else float("nan")


#: Colour-confidence bins per cell for the colour-conditional AUROC.
COLOUR_BINS = 5


def colour_strata(confidence, cells, *, bins: int = COLOUR_BINS) -> np.ndarray:
    """Each row's cell refined by its colour-confidence quintile *within that cell*.

    Rows are ranked by confidence inside their cell, ties broken by row order, and cut into
    ``bins`` near-equal groups labelled ``"<cell> | q1"`` (least centre-typical colour) to
    ``"<cell> | q<bins>"``. Pairing CRoMa's AUROC within these strata asks whether it still
    separates flips from non-flips among tiles whose colour is equally centre-typical.
    """
    confidence = pd.Series(np.asarray(confidence, dtype=float))
    cells = pd.Series(np.asarray(cells).astype(str))
    rank = confidence.groupby(cells).rank(method="first") - 1
    size = cells.map(cells.value_counts())
    quantile = (rank * bins // size).astype(int) + 1
    return (cells + " | q" + quantile.astype(str)).to_numpy()


#: Fewer flipped samples than this and an encoder's AUROCs get no interval and no verdict.
MIN_FLIPPED = 10

#: Bootstrap replicates and seed of every interval the study reports.
N_BOOT = 2000
BOOT_SEED = 0

#: Each predictor's per-sample column, signed so that a higher score means "more likely
#: flipped": a low margin (CRoMa, biology-only) or a centre-pure neighbourhood.
PREDICTORS = {"croma": ("croma_m5", -1.0), "bio": ("bio_margin", -1.0), "purity": ("purity", 1.0)}

#: The paired differences reported: CRoMa against each baseline.
DIFFERENCES = (("croma", "bio"), ("croma", "purity"))


def encoder_summary(frame: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """One encoder's row of the study: flip counts, within-cell AUROCs, paired differences, rho.

    ``frame`` holds the encoder's held-out rows (``join_targets``) plus ``bio_margin`` and
    ``purity``, and for a tile cohort ``colour_confidence`` and ``colour_stratum`` (cell x
    within-cell colour-confidence quintile). The AUROCs are for ``flipped`` among the rows
    the balanced probe never got wrong (``e0 == 0``): a row it already got wrong cannot be
    flipped, and counting it as a non-flip would reward a predictor for flagging intrinsic
    difficulty instead of the shortcut. Their intervals, and those of the paired differences
    CRoMa - baseline, come from one shared stratified slide-cluster draw
    (``bootstrap_aurocs``) and are left out (``nan``) below ``MIN_FLIPPED`` flips. The
    within-cell Spearman rho of CRoMa with the margin drop, plain and given the V = 0 margin,
    and with the colour centre-confidence, is over every held-out row.
    """
    population = frame[frame["e0"] == 0]
    event = population["flipped"].to_numpy(dtype=bool)
    cells = population["cell"].to_numpy()
    predictors = {
        name: (sign * population[column].to_numpy(dtype=float), cells)
        for name, (column, sign) in PREDICTORS.items()
    }
    colour = "colour_confidence" in frame
    if colour:
        predictors["colour"] = (population["colour_confidence"].to_numpy(dtype=float), cells)
        predictors["croma_given_colour"] = (
            -population["croma_m5"].to_numpy(dtype=float),
            population["colour_stratum"].to_numpy(),
        )
    n_flipped = int(event.sum())
    summary = {
        "n_held_out": int(len(frame)),
        "n_balanced_right": int(len(population)),
        "n_flipped": n_flipped,
        "evaluable": n_flipped >= MIN_FLIPPED,
    }
    if summary["evaluable"]:
        boot = bootstrap_aurocs(
            predictors,
            event=event,
            cells=cells,
            groups=population["group_id"].to_numpy(),
            differences=DIFFERENCES,
            n_boot=n_boot,
            seed=seed,
        )
        estimates = {f"auroc_{name}": ci for name, ci in boot.auroc.items()}
        estimates.update({f"diff_{a}_{b}": ci for (a, b), ci in boot.difference.items()})
        for key, ci in estimates.items():
            summary.update({key: ci.point, f"{key}_lo": ci.lo, f"{key}_hi": ci.hi})
    else:
        points = {
            name: within_cell_auroc(score, event, strata)
            for name, (score, strata) in predictors.items()
        }
        estimates = {f"auroc_{name}": point for name, point in points.items()}
        estimates.update({f"diff_{a}_{b}": points[a] - points[b] for a, b in DIFFERENCES})
        for key, point in estimates.items():
            summary.update({key: point, f"{key}_lo": float("nan"), f"{key}_hi": float("nan")})
    croma, all_cells = frame["croma_m5"].to_numpy(dtype=float), frame["cell"].to_numpy()
    summary["rho_drop"] = within_cell_spearman(croma, frame["margin_drop"], all_cells)
    summary["rho_drop_given_margin0"] = within_cell_spearman(
        croma, frame["margin_drop"], all_cells, z=frame["margin0"]
    )
    if colour:
        summary["rho_croma_colour"] = within_cell_spearman(
            croma, frame["colour_confidence"], all_cells
        )
    return summary


def align_rows(record, manifest: pd.DataFrame) -> np.ndarray:
    """Positions in ``manifest`` (a benchmark view's eval manifest) of each record ID row.

    Matched by tile key (``sample_id`` + ``group_id``), so ``features[align_rows(...)]`` puts
    a view's embedding rows in the record's order. The view's centre must be the record's.

    Raises:
        ValueError: If a record row is absent from the view or matched twice, or a matched
            row names a different centre.
    """
    key = list(probe_records.PER_SAMPLE_JOIN_KEY)
    view = manifest[key + ["confounder"]].astype(str).assign(position=np.arange(len(manifest)))
    if view.duplicated(key).any():
        raise ValueError("the view repeats a tile key")
    wanted = pd.DataFrame({column: record[f"id_{column}"].astype(str) for column in key})
    matched = wanted.merge(view, on=key, how="left", validate="one_to_one")
    if matched["position"].isna().any():
        raise ValueError(
            f"{int(matched['position'].isna().sum())} record row(s) absent from the view"
        )
    if not (matched["confounder"].to_numpy() == record["id_confounder"].astype(str)).all():
        raise ValueError("a record row's centre disagrees with the view's")
    return matched["position"].to_numpy(dtype=int)


def analyse_encoder(
    record,
    per_sample: pd.DataFrame,
    features,
    *,
    k: int,
    m: int = CROMA_HEADLINE_M,
    colour_confidence=None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[pd.DataFrame, dict]:
    """One encoder's held-out rows with every predictor attached, and its summary row.

    ``features`` holds the encoder's embeddings in the record's ID-row order (see
    ``align_rows``). Both baselines are computed over every ID row -- the tiles the run
    scored -- with the biology-only margin at averaging radius ``m`` (the headline CRoMa's)
    and centre purity at the run's operating ``k``. ``colour_confidence``, also in ID-row
    order, adds the colour predictors; its quintiles are taken within each cell over every
    ID row.
    """
    frame = join_targets(record, per_sample)
    labels = record["id_label"].astype(str)
    centres = record["id_confounder"].astype(str)
    slides = record["id_group_id"].astype(str)
    rows = frame["row"].to_numpy()
    frame["bio_margin"] = biology_only_margin(features, labels, slides, m=m)[rows]
    frame["purity"] = centre_purity(features, centres, slides, k=k)[rows]
    if colour_confidence is not None:
        confidence = np.asarray(colour_confidence, dtype=float)
        strata = colour_strata(confidence, np.char.add(np.char.add(labels, " | "), centres))
        frame["colour_confidence"] = confidence[rows]
        frame["colour_stratum"] = strata[rows]
    return frame, encoder_summary(frame, n_boot=n_boot, seed=seed)


def centre_purity(features, centres, group_ids, *, k: int) -> np.ndarray:
    """The share of a sample's ``k`` nearest neighbours, of any class, from its own centre.

    The neighbours are the ones RI and MaRI score at the run's operating ``k``: cosine
    distance, the sample's own independence group (slide) excluded, found by the library's
    ``croma.metrics.neighbors._prepare_neighbors``. Where fewer than ``k`` cross-slide
    neighbours exist the share is over those found.
    """
    centres = np.asarray(centres)
    neighbours, _distances, found = _prepare_neighbors(
        _unit_rows(features), np.asarray(group_ids).astype(str), int(k)
    )
    if (found == 0).any():
        raise RuntimeError(f"{int((found == 0).sum())} sample(s) have no cross-slide neighbour")
    valid = np.arange(neighbours.shape[1]) < found[:, None]
    same = (centres[np.where(valid, neighbours, 0)] == centres[:, None]) & valid
    return same.sum(axis=1) / found


# --------------------------------------------------------------------------------------
# The study: every encoder of each cohort's run, against its full-tile probe records.
# --------------------------------------------------------------------------------------

#: The cohorts and whether each is a tile cohort with colour statistics. PCaBiop's units
#: are slides: it is reported descriptively, without colour.
COHORTS = {"camelyon": True, "tcga_4x4": True, "pcabiop": False}

#: Where the study writes: one directory per cohort, never inside a run directory.
OUT_DIR = REPO / "output" / "studies" / "sample-shortcut"

#: Seed of the colour classifier's fold assignment.
COLOUR_SEED = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    """``path`` relative to the checkout, without following the ``output``/``data`` links."""
    path = Path(os.path.abspath(path))
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


@dataclass(frozen=True)
class CohortRun:
    """What one cohort's analysis reads: the run (via ``paper_manifest``) and its records."""

    cohort: str
    benchmark: str
    metrics_csv: Path
    per_sample_csv: Path
    models: tuple[str, ...]
    k: dict
    records: dict


def cohort_run(cohort: str) -> CohortRun:
    """Resolve a cohort's run through ``paper_manifest`` and its full-tile probe records.

    The roster is the run's ``metrics.csv``; every encoder in it must have a record, and
    the record directory may hold no other encoder.
    """
    from loaders import DATASETS
    from paper_manifest import by_benchmark

    benchmark = DATASETS[cohort]["benchmark"]
    run = by_benchmark(benchmark)
    metrics = pd.read_csv(REPO / run.metrics_rel, keep_default_na=False)
    models = tuple(metrics["model"].astype(str))
    tiles = DATASETS[cohort]["num_patches_per_slide"]
    records = {model: probe_records.Cell(cohort, model, tiles).path() for model in models}
    missing = sorted(model for model, path in records.items() if not path.exists())
    if missing:
        raise FileNotFoundError(
            f"{cohort}: no full-tile probe record for {missing}; run "
            "scripts/studies/apd/probe_records.py"
        )
    directory = next(iter(records.values())).parent
    extra = sorted({path.stem for path in directory.glob("*.npz")} - set(models))
    if extra:
        raise ValueError(f"{cohort}: records for encoders outside the run's roster: {extra}")
    return CohortRun(
        cohort=cohort,
        benchmark=benchmark,
        metrics_csv=REPO / run.metrics_rel,
        per_sample_csv=REPO / run.per_sample_rel,
        models=models,
        k={model: int(k) for model, k in zip(metrics["model"].astype(str), metrics["k"])},
        records=records,
    )


def _encoder_job(job: dict) -> tuple[pd.DataFrame, dict]:
    import views

    record = probe_records.read_record(job["record"])
    view = views.load_view(job["benchmark"])
    features = np.asarray(view.features(job["model"])[job["rows"]], dtype=np.float64)
    frame, summary = analyse_encoder(
        record,
        job["per_sample"],
        features,
        k=job["k"],
        colour_confidence=job["colour_confidence"],
        n_boot=job["n_boot"],
        seed=job["seed"],
    )
    print(f"[done] {job['cohort']}/{job['model']}: {summary['n_flipped']} flipped", flush=True)
    return frame.assign(model=job["model"]), summary


#: The per-sample columns the study writes, in order.
SAMPLE_COLUMNS = (
    "model",
    "row",
    "sample_id",
    "group_id",
    "image_path",
    "cell",
    "n_held_out",
    "e0",
    "e1",
    "flipped",
    "margin0",
    "margin1",
    "margin_drop",
    "croma_m5",
    "bio_margin",
    "purity",
    "colour_confidence",
    "colour_stratum",
)


def run_cohort(
    cohort: str, *, out_dir: Path = OUT_DIR, jobs: int = 1, n_boot: int = N_BOOT
) -> Path:
    """Analyse every encoder of one cohort and write ``<out_dir>/<cohort>/``.

    Writes ``encoders.csv`` (one summary row per encoder), ``samples.csv.gz`` (every
    encoder's held-out rows with all predictors) and ``cohort.json`` (cohort-level counts,
    the colour classifier's balanced accuracy, the constants, and the sha256 of every input
    and output file). A tile cohort also caches its colour statistics in ``colour.csv``.
    """
    import layout
    import views

    run = cohort_run(cohort)
    records = {model: probe_records.read_record(path) for model, path in run.records.items()}
    first = records[run.models[0]]
    for model, record in records.items():
        for column in ("sample_id", "group_id", "label", "confounder", "image_path"):
            if not np.array_equal(record[f"id_{column}"], first[f"id_{column}"]):
                raise ValueError(f"{cohort}/{model}: ID rows differ from {run.models[0]}'s")
        if not np.array_equal(record["held_out"], first["held_out"]):
            raise ValueError(f"{cohort}/{model}: held-out rows differ from {run.models[0]}'s")
    view = views.load_view(run.benchmark)
    rows = align_rows(first, view.eval_manifest)
    per_sample = pd.read_csv(
        run.per_sample_csv,
        usecols=list(PER_SAMPLE_COLUMNS),
        dtype={"model": str, "sample_id": str, "group_id": str},
        keep_default_na=False,
    )
    cohort_dir = Path(out_dir) / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)

    colour_confidence = None
    colour_info = {}
    inputs = {}
    if COHORTS[cohort]:
        stats = tile_colour.cohort_colour_stats(
            first["id_sample_id"], first["id_image_path"], cohort_dir / "colour.csv", workers=jobs
        )
        classifier = tile_colour.colour_centre_classifier(
            stats[list(tile_colour.FEATURES)],
            first["id_confounder"].astype(str),
            first["id_group_id"].astype(str),
            seed=COLOUR_SEED,
        )
        colour_confidence = classifier.confidence
        colour_info = {
            "colour_balanced_accuracy": classifier.balanced_accuracy,
            "colour_folds": tile_colour.CLASSIFIER_FOLDS,
            "colour_seed": COLOUR_SEED,
            "tissue_od_threshold": tile_colour.TISSUE_OD_THRESHOLD,
            "n_tissue_fallback": int(stats["tissue_fallback"].sum()),
            "colour_bins": COLOUR_BINS,
        }
        inputs[_relative(cohort_dir / "colour.csv")] = _sha256(cohort_dir / "colour.csv")

    work = [
        {
            "cohort": cohort,
            "benchmark": run.benchmark,
            "model": model,
            "record": run.records[model],
            "rows": rows,
            "per_sample": per_sample[per_sample["model"] == model],
            "k": run.k[model],
            "colour_confidence": colour_confidence,
            "n_boot": n_boot,
            "seed": BOOT_SEED,
        }
        for model in run.models
    ]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
            results = list(pool.map(_encoder_job, work))
    else:
        results = [_encoder_job(job) for job in work]

    n_never = never_held_out(first)
    encoders = pd.DataFrame(
        [
            {"cohort": cohort, "model": model, "k": run.k[model], **summary}
            for model, (_frame, summary) in zip(run.models, results)
        ]
    )
    samples = pd.concat([frame for frame, _summary in results], ignore_index=True)
    samples = samples.reindex(columns=[c for c in SAMPLE_COLUMNS if c in samples])
    encoders.to_csv(cohort_dir / "encoders.csv", index=False)
    samples.to_csv(
        cohort_dir / "samples.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    spec = view.spec
    for path in [
        run.metrics_csv,
        run.per_sample_csv,
        REPO / spec.manifest,
        layout.tileset_manifest(spec.tileset),
        *run.records.values(),
        *(layout.embedding_path(spec.tileset, model) for model in run.models),
    ]:
        inputs[_relative(path)] = _sha256(path)
    cohort_info = {
        "cohort": cohort,
        "benchmark": run.benchmark,
        "n_tiles": int(len(first["id_sample_id"])),
        "n_held_out": int(len(first["id_sample_id"]) - n_never),
        "n_never_held_out": n_never,
        "n_cells": int(samples["cell"].nunique()),
        "n_encoders": len(run.models),
        "min_flipped": MIN_FLIPPED,
        "flip_error_rate": FLIP_ERROR_RATE,
        "n_boot": n_boot,
        "boot_seed": BOOT_SEED,
        "croma_m": CROMA_HEADLINE_M,
        **colour_info,
        "outputs": {
            name: _sha256(cohort_dir / name) for name in ("encoders.csv", "samples.csv.gz")
        },
        "inputs": dict(sorted(inputs.items())),
    }
    (cohort_dir / "cohort.json").write_text(json.dumps(cohort_info, indent=2) + "\n")
    print(
        f"[{cohort}] {len(run.models)} encoders, {n_never} never held out, "
        f"{int(encoders['evaluable'].sum())} with >= {MIN_FLIPPED} flipped -> {cohort_dir}",
        flush=True,
    )
    return cohort_dir


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    p.add_argument("--jobs", type=int, default=4, help="parallel encoder processes (max 4)")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = get_args(argv)
    for cohort in args.cohorts:
        run_cohort(cohort, out_dir=args.out_dir, jobs=min(args.jobs, 4), n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
