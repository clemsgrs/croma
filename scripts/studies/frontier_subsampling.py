"""Frontier membership under fixed and recomputed margins, on the subsampling draws.

``bootstrap_uncertainty.py`` reports ``p_frontier`` from replicates that reuse the per-sample
CRoMa margins computed once on the full cohort: a retained anchor keeps a margin built from
neighbours on slides the replicate omitted. This study isolates what rebuilding the
neighbourhoods contributes, on draws that need no resampling with replacement: the
subsampling-stability study's (``subsampling_stability.py``). Its replicates keep a stated
number of units in every class x centre cell, without replacement, at 25%, 50% and 75%, 20
paired replicates each, and re-run the production driver on every subsample. Nothing is
redrawn and no neighbour is searched for here; both arms come from existing outputs.

The two arms, on exactly the same anchors of exactly the same draw:

* **fixed**: the retained anchors' full-cohort margins (the production run's per-sample
  ``croma_m{CROMA_HEADLINE_M}``, read as ``bootstrap_uncertainty`` reads them), restricted to
  the rows of the units the replicate retained (its ``units.csv``) and summarised with the
  code and ``ALPHA`` that ``bootstrap_uncertainty`` summarises a replicate with: the pooled
  median and ``LTM_10``.
* **recomputed**: the same anchors' margins recomputed with only the retained units as
  reference -- the median CRoMa (``croma``) and ``LTM_10`` (``croma_ltm_alpha``) in the
  replicate's own ``scores.csv``.

A replicate is used only if the fixed arm's rows are the rows its driver run scored (the
``replicate.json`` row and unit counts) and every cell kept the units the design states.
For each arm the frontier is taken over the ranked encoders (the natural-image control
excluded) with ``plotting.base._pareto_frontier_max_max``, the rule of
``bootstrap_uncertainty.frontier_probability`` and the Pareto figures; the cross-benchmark
mean-rank frontier is ``bootstrap_uncertainty.mean_rank_frontier_probability``'s, combining
replicate ``r`` of every cohort at one fraction. Before any replicate is scored, the fixed
arm on the whole manifest must reproduce the production median and ``LTM_10`` to within
``bootstrap_uncertainty.POINT_ATOL``, and both it and the subsampling study's full-cohort
replicate must reproduce the production frontier.

Reading the frequencies
-----------------------
Every frequency is a **membership frequency under subsampling at that cohort size**: the
share of a fraction's replicates whose frontier holds the encoder. It is not a bootstrap
probability or a confidence interval, and is not to be merged with ``p_frontier``. With 20
replicates it moves in steps of 0.05 (``resolution``), so a difference of one step between
the arms is the resolution, not a finding; ``summary.json`` lists only larger ones.

Anyone comparing with ``p_frontier`` should note that the units differ on Tolkach-ESCA. The
subsampling unit there is a case's block within a class x centre cell (a case contributes a
block to every class it spans), whereas the bootstrap resamples whole cases within centres.
The paired comparison here is internal to the subsampling draws, so it is unaffected.

Outputs, in ``output/studies/frontier-subsampling/``:

``membership.csv``
    Per cohort x fraction x ranked encoder: each arm's frontier hits and frequency, their
    paired difference (``freq_diff``, recomputed minus fixed), the replicates on which only
    one arm holds the encoder, and each arm's median and ``LTM_10`` across replicates (mean,
    sd, min, max).
``cross_benchmark_membership.csv``
    The same membership columns for the mean-rank frontier, per fraction x encoder.
``frontier_sets.csv``
    Every replicate's frontier, per scope (cohort or cross-benchmark) x fraction x arm.
``rank_agreement.csv``
    Spearman's rho between the arms' median CRoMa over the ranked encoders, per replicate.
``replicate_scores.csv``
    Both arms' median and ``LTM_10`` per cohort x fraction x replicate x encoder, with the
    number of units and anchors the replicate retained.
``units_per_cell.csv``
    The units and anchors each replicate retained in each class x centre cell.
``full_cohort_scores.csv`` / ``reproduction.json``
    The full-cohort reproduction: production, fixed and recomputed points and frontiers.
``summary.json``
    The design, the measure, its resolution, the encoders whose membership differs between
    the arms by more than one replicate step, and the rank agreement per cohort x fraction.
``provenance.json``
    The sha256 of every input and output.

Usage:
  python scripts/studies/frontier_subsampling.py [--root .] [--subsampling-root ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import _subsampling  # noqa: E402
import benchmarks  # noqa: E402
import bootstrap_uncertainty as bootstrap  # noqa: E402
import layout  # noqa: E402
import subsampling_stability as subsampling  # noqa: E402
from croma.metrics.bootstrap import _tail_summaries  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402
from plotting.base import _pareto_frontier_max_max  # noqa: E402
from plotting.style import CONTROL_MODEL  # noqa: E402

#: The tail level, as ``bootstrap_uncertainty`` reports it (``LTM_10``).
ALPHA = bootstrap.ALPHA

STUDY = "frontier-subsampling"
PROVENANCE = "provenance.json"
#: What every frequency the study writes is, and is not.
MEASURE = (
    "membership frequency under subsampling at the stated cohort size (units per class x "
    "centre cell): the share of the subsampling study's replicates whose median-versus-"
    "LTM10 Pareto frontier holds the encoder. It is not a bootstrap probability or a "
    "confidence interval, and is not comparable with, or to be merged with, "
    "bootstrap_uncertainty's p_frontier."
)
FIXED, RECOMPUTED = "fixed", "recomputed"
ARMS = (FIXED, RECOMPUTED)
#: A row's identity: each eval-manifest column, and the per-sample column that carries it.
IDENTITY = {
    "sample_id": "sample_id",
    "label": "label",
    "medical_center": "confounder",
    "group_id": "group_id",
}


def retained_rows(manifest: pd.DataFrame, units: pd.DataFrame) -> np.ndarray:
    """Positions of ``manifest``'s rows whose unit (label, centre, group) ``units`` lists."""
    keys = list(subsampling.UNIT)
    kept = pd.MultiIndex.from_frame(units.loc[:, keys].astype(str))
    present = pd.MultiIndex.from_frame(manifest.loc[:, keys].astype(str))
    absent = kept[~kept.isin(present)]
    if len(absent):
        raise ValueError(f"retained units with no row in the full cohort: {list(absent)[:5]}")
    return np.flatnonzero(present.isin(kept))


def fixed_arm(
    values: dict[str, np.ndarray], rows: np.ndarray, *, alpha: float = ALPHA
) -> pd.DataFrame:
    """Each encoder's median and ``LTM_alpha`` over ``rows`` of its full-cohort margins.

    The summaries are ``bootstrap_uncertainty``'s: the pooled median and lower-tail mean of
    the finite values, computed by the same library code its replicates use.
    """
    rows = np.asarray(rows)
    table = {}
    for model, margins in values.items():
        median, _f0, ltm = _tail_summaries(np.asarray(margins, dtype=float)[rows], alpha)
        table[model] = {"median": median, "ltm": ltm}
    return pd.DataFrame.from_dict(table, orient="index", columns=["median", "ltm"])


def frontier_sets(
    models: list[str], medians: np.ndarray, tails: np.ndarray
) -> list[tuple[str, ...]]:
    """Each replicate's median-versus-tail Pareto frontier, in staircase order.

    ``medians`` and ``tails`` are ``(n_replicates, len(models))``. The points and the rule
    are ``bootstrap_uncertainty.frontier_probability``'s: ``_pareto_frontier_max_max``,
    larger better on both axes.
    """
    return [
        tuple(
            _pareto_frontier_max_max(
                [(m, float(x), float(y)) for m, x, y in zip(models, med_row, tail_row)]
            )
        )
        for med_row, tail_row in zip(np.asarray(medians), np.asarray(tails))
    ]


def mean_rank_frontier_sets(
    models: list[str], cohorts: dict[str, tuple[np.ndarray, np.ndarray]]
) -> list[tuple[str, ...]]:
    """Each replicate's cross-benchmark mean-rank frontier.

    ``cohorts`` maps each cohort to its ``(medians, tails)``, ``(n_replicates, len(models))``
    in ``models`` order; replicate ``r`` combines replicate ``r`` of every cohort. Ranks,
    averaging and frontier are ``bootstrap_uncertainty.mean_rank_frontier_probability``'s:
    rank within each cohort (1 = largest, ties in column order), average over cohorts, and
    take the frontier on the two mean ranks, fewer being better.
    """
    pairs = list(cohorts.values())
    median_rank = np.mean([bootstrap._descending_ranks(med) for med, _ in pairs], axis=0)
    tail_rank = np.mean([bootstrap._descending_ranks(tail) for _, tail in pairs], axis=0)
    return frontier_sets(models, -median_rank, -tail_rank)


def paired_membership(
    models: list[str], fixed: list[tuple[str, ...]], recomputed: list[tuple[str, ...]]
) -> pd.DataFrame:
    """Each encoder's frontier membership in both arms over the same replicates.

    ``freq_diff`` is recomputed minus fixed. It moves in steps of one replicate
    (``resolution``); ``n_fixed_only`` and ``n_recomputed_only`` count the replicates whose
    two frontiers disagree about the encoder, in each direction.
    """
    if len(fixed) != len(recomputed) or not fixed:
        raise ValueError("the arms must be scored on the same, non-empty set of replicates")
    n = len(fixed)
    rows = []
    for m in models:
        in_fixed = np.array([m in s for s in fixed])
        in_recomputed = np.array([m in s for s in recomputed])
        rows.append(
            {
                "model": m,
                "n_replicates": n,
                "resolution": 1 / n,
                "hits_fixed": int(in_fixed.sum()),
                "hits_recomputed": int(in_recomputed.sum()),
                "freq_fixed": in_fixed.sum() / n,
                "freq_recomputed": in_recomputed.sum() / n,
                "freq_diff": (int(in_recomputed.sum()) - int(in_fixed.sum())) / n,
                "n_fixed_only": int((in_fixed & ~in_recomputed).sum()),
                "n_recomputed_only": int((in_recomputed & ~in_fixed).sum()),
            }
        )
    return pd.DataFrame(rows)


def beyond_one_step(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of a ``paired_membership`` table whose arms differ by more than one replicate.

    One replicate is the resolution of a membership frequency, so a single-step difference
    is not read as one.
    """
    return table[(table["hits_recomputed"] - table["hits_fixed"]).abs() > 1]


# --------------------------------------------------------------------------- the inputs


class ReproductionError(AssertionError):
    """The inputs do not reproduce the production run, or the arms would not be paired."""


def study_dir() -> Path:
    """``output/studies/frontier-subsampling/``."""
    return layout.OUTPUT_ROOT / "studies" / STUDY


def _read_table(path: Path) -> pd.DataFrame:
    """A manifest or ``units.csv``, every value as its text (a slide id is not a number)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@dataclass(frozen=True, eq=False)
class FullCohort:
    """A production run's per-sample CRoMa at the headline ``m``, row ``i`` the manifest's."""

    benchmark: str
    #: The ranked encoders (the natural-image control dropped), in ``metrics.csv`` order.
    models: list[str]
    values: dict[str, np.ndarray]
    #: The eval manifest the subsampling study drew from.
    manifest: pd.DataFrame
    #: ``metrics.csv``, indexed by model, ranked encoders only.
    published: pd.DataFrame


def load_full_cohort(benchmark: str, root: Path = REPO) -> FullCohort:
    """Read the run's per-sample margins as ``bootstrap_uncertainty.load_cohort`` does.

    The same file (``paper_manifest.by_benchmark(...).per_sample_rel``), column
    (``croma_m{CROMA_HEADLINE_M}``), row order and ranked roster. It also keeps what that
    loader drops and the fixed arm needs: each row's unit. Every encoder's rows must be the
    eval manifest's rows, in its order, so a unit listed in ``units.csv`` selects exactly the
    rows the replicate's driver run scored.
    """
    entry = by_benchmark(benchmark)
    column = f"croma_m{int(CROMA_HEADLINE_M)}"
    per_sample_path = root / entry.per_sample_rel
    metrics_path = root / entry.metrics_rel
    manifest_path = root / benchmarks.get(benchmark).manifest
    cols = ["model", "occurrence_index", "sample_index", "subset", *IDENTITY.values(), column]
    per_sample = pd.read_csv(per_sample_path, usecols=cols, float_precision="round_trip")
    published = pd.read_csv(metrics_path, float_precision="round_trip")
    manifest = _read_table(manifest_path)
    identity = manifest.loc[:, list(IDENTITY)].set_axis(list(IDENTITY.values()), axis=1)
    models = [m for m in published["model"] if m != CONTROL_MODEL]
    values: dict[str, np.ndarray] = {}
    for model, rows in per_sample.groupby("model", sort=False):
        if model not in models:
            continue
        rows = rows.sort_values(["occurrence_index", "sample_index"], kind="stable")
        if rows["subset"].nunique() != 1:
            raise ReproductionError(f"{benchmark}: the run scores several subsets")
        ids = rows.loc[:, list(IDENTITY.values())].astype(str).reset_index(drop=True)
        if not ids.equals(identity):
            raise ReproductionError(
                f"{benchmark}: {model}'s per-sample rows are not {manifest_path}'s rows"
            )
        values[model] = rows[column].to_numpy(dtype=float)
    missing = sorted(set(models) - set(values))
    if missing:
        raise ReproductionError(f"{benchmark}: no per-sample rows for {missing}")
    return FullCohort(
        benchmark=benchmark,
        models=models,
        values={m: values[m] for m in models},
        manifest=manifest,
        published=published.set_index("model").loc[models],
    )


def check_full_cohort(cohort: FullCohort, full_scores: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Both arms on the whole cohort must be the production run, frontier included.

    The fixed arm over every row must give the published median and ``LTM_10`` to within
    ``bootstrap_uncertainty.POINT_ATOL``; its frontier, and that of the subsampling study's
    full-cohort replicate (``full_scores``, already checked against production by that study
    within its cohort's tolerance), must be the production frontier. Returns the record and
    the three sets of points (``production`` / ``fixed`` / ``recomputed`` median and ltm),
    indexed by encoder.
    """
    models = cohort.models
    fixed = fixed_arm(cohort.values, np.arange(len(cohort.manifest)))
    published = cohort.published.loc[models, ["croma", "croma_ltm_alpha"]]
    published = published.set_axis(["median", "ltm"], axis=1)
    gap = (fixed - published).abs()
    for stat in ("median", "ltm"):
        if not (gap[stat] <= bootstrap.POINT_ATOL).all():
            worst = gap[stat].idxmax()
            raise ReproductionError(
                f"{cohort.benchmark}: {worst}'s fixed-arm {stat} over the full cohort is "
                f"{fixed.loc[worst, stat]!r}, but the run published {published.loc[worst, stat]!r}"
            )
    recomputed = full_scores.set_index("model").loc[models, ["croma", "croma_ltm_alpha"]]
    recomputed = recomputed.set_axis(["median", "ltm"], axis=1)
    points = pd.concat({"production": published, FIXED: fixed, RECOMPUTED: recomputed}, axis=1)
    points.columns = [f"{arm}_{stat}" for arm, stat in points.columns]

    def frontier(frame: pd.DataFrame) -> list[str]:
        return list(_frontier(models, frame["median"], frame["ltm"]))

    production = frontier(published)
    record = {
        "n_anchors": int(len(cohort.manifest)),
        "production_frontier": production,
        "fixed_frontier": frontier(fixed),
        "recomputed_frontier": frontier(recomputed),
        "fixed_max_abs_deviation": {s: float(gap[s].max()) for s in ("median", "ltm")},
        "recomputed_max_abs_deviation": {
            s: float((recomputed[s] - published[s]).abs().max()) for s in ("median", "ltm")
        },
        "fixed_tolerance": bootstrap.POINT_ATOL,
        "recomputed_tolerance": subsampling.reproduction_tolerance(cohort.benchmark),
    }
    for arm in ARMS:
        if set(record[f"{arm}_frontier"]) != set(production):
            raise ReproductionError(
                f"{cohort.benchmark}: the {arm} arm's full-cohort frontier "
                f"{record[f'{arm}_frontier']} is not the production frontier {production}"
            )
    return record, points


def _frontier(models: list[str], medians, tails) -> tuple[str, ...]:
    """One frontier from one set of points, each aligned to ``models``."""
    medians = np.asarray(medians, dtype=float)[None, :]
    return frontier_sets(models, medians, np.asarray(tails, dtype=float)[None, :])[0]


# --------------------------------------------------------------------------- the two arms


def _spread(block: pd.DataFrame, column: str) -> pd.DataFrame:
    """``column``'s mean, sd, min and max across replicates, per encoder."""
    stats = block.groupby("model", sort=False)[column].agg(["mean", "std", "min", "max"])
    return stats.rename(columns={"std": "sd"}).add_prefix(f"{column}_")


def score_replicates(
    cohort: FullCohort,
    recomputed: pd.DataFrame,
    meta: pd.DataFrame,
    sub_root: Path,
    design: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both arms on every subsampled replicate of ``cohort``.

    ``recomputed`` and ``meta`` are the subsampling study's scores (ranked encoders only)
    and ``replicate.json`` records for this cohort. Returns one row per replicate x encoder,
    and one per replicate x class x centre cell with the units and anchors it retained.
    A replicate is scored only if the fixed arm's anchors are the rows its driver run scored
    (same count of rows and units) and every cell kept the units the design states.
    """
    rows, cells = [], []
    n_cells = len(cohort.manifest.loc[:, list(subsampling.CELL)].drop_duplicates())
    for fraction in design["fractions"]:
        stated = int(design["units_per_cell"][cohort.benchmark][str(fraction)])
        for replicate in range(int(design["n_replicates"])):
            name = f"{cohort.benchmark} at {fraction}%, replicate {replicate}"
            where = subsampling.replicate_dir(sub_root, cohort.benchmark, fraction, replicate)
            units = _read_table(where / subsampling.UNITS)
            anchors = retained_rows(cohort.manifest, units)
            (record,) = meta[
                (meta["fraction"] == int(fraction)) & (meta["replicate"] == replicate)
            ].to_dict(orient="records")
            if (len(anchors), len(units)) != (int(record["n_rows"]), int(record["n_units"])):
                raise ReproductionError(
                    f"{name}: units.csv selects {len(anchors)} rows in {len(units)} units, "
                    f"but the replicate scored {record['n_rows']} rows in {record['n_units']}"
                )
            per_cell = (
                cohort.manifest.iloc[anchors]
                .groupby(list(subsampling.CELL))
                .agg(n_units=(subsampling.GROUP, "nunique"), n_anchors=(subsampling.GROUP, "size"))
                .reset_index()
            )
            if len(per_cell) != n_cells or set(per_cell["n_units"]) != {stated}:
                raise ReproductionError(
                    f"{name}: the design states {stated} units per cell in {n_cells} cells, "
                    f"the draw kept {sorted(set(per_cell['n_units']))} in {len(per_cell)}"
                )
            cells.append(
                per_cell.assign(
                    cohort=cohort.benchmark, fraction=int(fraction), replicate=replicate
                )
            )
            fixed = fixed_arm(cohort.values, anchors)
            mine = recomputed[
                (recomputed["fraction"] == int(fraction)) & (recomputed["replicate"] == replicate)
            ].set_index("model")
            for model in cohort.models:
                rows.append(
                    {
                        "cohort": cohort.benchmark,
                        "fraction": int(fraction),
                        "replicate": replicate,
                        "model": model,
                        "n_units": int(len(units)),
                        "n_anchors": int(len(anchors)),
                        "fixed_median": fixed.loc[model, "median"],
                        "fixed_ltm": fixed.loc[model, "ltm"],
                        "recomputed_median": float(mine.loc[model, "croma"]),
                        "recomputed_ltm": float(mine.loc[model, "croma_ltm_alpha"]),
                    }
                )
    return pd.DataFrame(rows), pd.concat(cells, ignore_index=True)


def _arm_arrays(scores: pd.DataFrame, models: list[str], arm: str):
    """``(medians, tails)`` of one arm, ``(n_replicates, len(models))`` in ``models`` order."""
    wide = scores.pivot(index="replicate", columns="model")
    return (
        wide[f"{arm}_median"].loc[:, models].to_numpy(dtype=float),
        wide[f"{arm}_ltm"].loc[:, models].to_numpy(dtype=float),
    )


#: The frontier sets of one scope (a cohort, or the cross-benchmark panel) at one fraction:
#: arm -> one frontier per replicate, in replicate order.
ArmSets = dict[str, list[tuple[str, ...]]]
CROSS = "cross-benchmark"


def membership(
    scores: pd.DataFrame, models: list[str], design: dict
) -> tuple[pd.DataFrame, dict[tuple[str, int], ArmSets]]:
    """Per cohort x fraction x encoder: both arms' membership and their median/LTM spread.

    Also returns the replicate-level frontier sets, keyed by ``(cohort, fraction)``.
    """
    tables, sets_by_key = [], {}
    for (cohort, fraction), block in scores.groupby(["cohort", "fraction"], sort=False):
        sets = {arm: frontier_sets(models, *_arm_arrays(block, models, arm)) for arm in ARMS}
        sets_by_key[(cohort, int(fraction))] = sets
        table = paired_membership(models, sets[FIXED], sets[RECOMPUTED])
        stated = design["units_per_cell"][cohort]
        table.insert(0, "cohort", cohort)
        table.insert(1, "fraction", int(fraction))
        table.insert(2, "units_per_cell", int(stated[str(fraction)]))
        table.insert(3, "full_units_per_cell", int(stated["full"]))
        spread = [_spread(block, f"{arm}_{stat}") for arm in ARMS for stat in ("median", "ltm")]
        tables.append(table.join(pd.concat(spread, axis=1), on="model"))
    return pd.concat(tables, ignore_index=True), sets_by_key


def frontier_set_table(sets_by_key: dict[tuple[str, int], ArmSets]) -> pd.DataFrame:
    """One row per scope x fraction x replicate x arm: the frontier, ``;``-joined in staircase
    order (median ascending)."""
    rows = []
    for (scope, fraction), sets in sets_by_key.items():
        for arm in ARMS:
            for replicate, frontier in enumerate(sets[arm]):
                rows.append(
                    {
                        "scope": scope,
                        "fraction": int(fraction),
                        "replicate": replicate,
                        "arm": arm,
                        "n_frontier": len(frontier),
                        "frontier": ";".join(frontier),
                    }
                )
    return pd.DataFrame(rows)


def rank_agreement(scores: pd.DataFrame) -> pd.DataFrame:
    """Spearman's rho between the two arms' median CRoMa over the ranked encoders, per
    cohort x fraction x replicate."""
    rows = []
    keys = ["cohort", "fraction", "replicate"]
    for (cohort, fraction, replicate), block in scores.groupby(keys, sort=False):
        rho = spearmanr(block["fixed_median"], block["recomputed_median"]).statistic
        rows.append(
            {
                "cohort": cohort,
                "fraction": int(fraction),
                "replicate": int(replicate),
                "n_models": int(len(block)),
                "spearman_rho": float(rho),
            }
        )
    return pd.DataFrame(rows)


def cross_benchmark(
    scores: pd.DataFrame, models: list[str]
) -> tuple[pd.DataFrame, dict[tuple[str, int], ArmSets]]:
    """The mean-rank frontier over the cohorts, replicate ``r`` of each at one fraction.

    Returns one row per fraction x encoder (both arms' membership), and the replicate-level
    frontier sets, keyed by ``("cross-benchmark", fraction)``.
    """
    tables, sets_by_key = [], {}
    for fraction, at_fraction in scores.groupby("fraction", sort=True):
        sets = {}
        for arm in ARMS:
            cohorts = {
                cohort: _arm_arrays(block, models, arm)
                for cohort, block in at_fraction.groupby("cohort", sort=False)
            }
            sets[arm] = mean_rank_frontier_sets(models, cohorts)
        table = paired_membership(models, sets[FIXED], sets[RECOMPUTED])
        table.insert(0, "fraction", int(fraction))
        tables.append(table)
        sets_by_key[(CROSS, int(fraction))] = sets
    return pd.concat(tables, ignore_index=True), sets_by_key


def _check_panel(cohorts: list[str], rosters: dict[str, list[str]]) -> list[str]:
    """The cohorts must be ``bootstrap_uncertainty``'s cross-benchmark panel, on one roster."""
    prefixes = {by_benchmark(c).prefix for c in cohorts}
    if prefixes != set(bootstrap.FRONTIER_PANEL):
        raise ReproductionError(
            f"the subsampled cohorts {cohorts} are not the cross-benchmark panel "
            f"{bootstrap.FRONTIER_PANEL}"
        )
    models = rosters[cohorts[0]]
    if any(rosters[c] != models for c in cohorts):
        raise ReproductionError("the cohorts do not share one ranked roster")
    return models


# --------------------------------------------------------------------------- the study


def run(
    *,
    root: Path = REPO,
    subsampling_root: Path | None = None,
    out_dir: Path | None = None,
    expected_design: dict | None = None,
) -> dict:
    """Score both arms on every recorded draw and write the study to ``out_dir``."""
    sub_root = subsampling.study_root() if subsampling_root is None else Path(subsampling_root)
    out = Path(out_dir) if out_dir is not None else study_dir()
    recorded = _subsampling.load(sub_root, expected=expected_design)
    design = recorded.design
    scores, cells, full_points, rosters = [], [], {}, {}
    reproduction: dict = {"cohorts": {}}
    for benchmark in design["cohorts"]:
        cohort = load_full_cohort(benchmark, root)
        rosters[benchmark] = cohort.models
        mine = recorded.scores[recorded.scores["cohort"] == benchmark]
        meta = recorded.meta[recorded.meta["cohort"] == benchmark]
        full = mine[mine["fraction"] == subsampling.FULL]
        record, full_points[benchmark] = check_full_cohort(cohort, full)
        reproduction["cohorts"][benchmark] = record
        replicate_scores, replicate_cells = score_replicates(cohort, mine, meta, sub_root, design)
        scores.append(replicate_scores)
        cells.append(replicate_cells)
    models = _check_panel(list(design["cohorts"]), rosters)
    scores = pd.concat(scores, ignore_index=True)
    reproduction["cross_benchmark"] = {
        f"{arm}_frontier": list(
            mean_rank_frontier_sets(
                models,
                {
                    c: (
                        points[f"{arm}_median"].to_numpy()[None, :],
                        points[f"{arm}_ltm"].to_numpy()[None, :],
                    )
                    for c, points in full_points.items()
                },
            )[0]
        )
        for arm in ("production", *ARMS)
    }
    per_cohort, cohort_sets = membership(scores, models, design)
    cross, cross_sets = cross_benchmark(scores, models)
    agreement = rank_agreement(scores)
    cells = pd.concat(cells, ignore_index=True)
    columns = ["cohort", "fraction", "replicate", *subsampling.CELL, "n_units", "n_anchors"]
    full_points = pd.concat(full_points, names=["cohort", "model"]).reset_index()
    tables = {
        "membership.csv": per_cohort,
        "cross_benchmark_membership.csv": cross,
        "frontier_sets.csv": frontier_set_table({**cohort_sets, **cross_sets}),
        "rank_agreement.csv": agreement,
        "replicate_scores.csv": scores,
        "units_per_cell.csv": cells.loc[:, columns],
        "full_cohort_scores.csv": full_points,
    }
    records = {
        "reproduction.json": reproduction,
        "summary.json": summary(design, per_cohort, cross, agreement),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    for name, payload in records.items():
        _write_json(out / name, payload)
    provenance = {
        "inputs": {_key(p, root): sha256(p) for p in study_inputs(root, sub_root, design)},
        "outputs": {name: sha256(out / name) for name in sorted([*tables, *records])},
    }
    _write_json(out / PROVENANCE, provenance)
    return {"scores": scores, "membership": per_cohort, "cross_benchmark": cross}


def study_inputs(root: Path, sub_root: Path, design: dict) -> list[Path]:
    """Every file the study reads: the production runs and manifests, and every record of
    the subsampling study (its design, each record's scores and metadata, the full-cohort
    reproduction checks, and every subsampled replicate's units)."""
    paths = [Path(sub_root) / subsampling.DESIGN]
    for benchmark in design["cohorts"]:
        entry = by_benchmark(benchmark)
        paths += [root / entry.per_sample_rel, root / entry.metrics_rel]
        paths.append(root / benchmarks.get(benchmark).manifest)
        full = subsampling.replicate_dir(sub_root, benchmark, subsampling.FULL, 0)
        paths += [full / subsampling.SCORES, full / subsampling.REPLICATE_META]
        paths.append(full / subsampling.REPRODUCTION)
        for fraction in design["fractions"]:
            for replicate in range(int(design["n_replicates"])):
                where = subsampling.replicate_dir(sub_root, benchmark, fraction, replicate)
                paths += [where / subsampling.SCORES, where / subsampling.REPLICATE_META]
                paths.append(where / subsampling.UNITS)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` where it lies under it (lexically: ``output/`` may be a
    symlink), else as given."""
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        return str(path)


def summary(
    design: dict, per_cohort: pd.DataFrame, cross: pd.DataFrame, agreement: pd.DataFrame
) -> dict:
    """The study's design, what its frequencies mean, and what differs between the arms."""
    n = int(design["n_replicates"])
    flagged = pd.concat(
        [
            beyond_one_step(per_cohort).rename(columns={"cohort": "scope"}),
            beyond_one_step(cross).assign(scope=CROSS),
        ],
        ignore_index=True,
    )
    columns = ["scope", "fraction", "model", "hits_fixed", "hits_recomputed"]
    columns += ["freq_fixed", "freq_recomputed", "freq_diff"]
    stats = agreement.groupby(["cohort", "fraction"])["spearman_rho"]
    rho = {}
    for (cohort, fraction), values in stats:
        rho.setdefault(cohort, {})[str(fraction)] = {
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return {
        "measure": MEASURE,
        "arms": {
            FIXED: "the retained anchors' full-cohort margins (production per-sample CRoMa), "
            "summarised as bootstrap_uncertainty summarises a replicate",
            RECOMPUTED: "the same anchors' margins recomputed with only the retained units "
            "as reference: the subsampling replicate's scores.csv",
        },
        "alpha": float(ALPHA),
        "headline_m": int(CROMA_HEADLINE_M),
        "cohorts": list(design["cohorts"]),
        "fractions": [int(f) for f in design["fractions"]],
        "units_per_cell": design["units_per_cell"],
        "n_replicates": n,
        "resolution": 1 / n,
        "freq_diff": "recomputed minus fixed, paired replicate by replicate",
        "beyond_one_step": flagged.loc[:, columns].to_dict(orient="records"),
        "rank_agreement": rho,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO, help="where output/ and data/ are read")
    parser.add_argument(
        "--subsampling-root",
        type=Path,
        help="the subsampling-stability study (default: output/studies/subsampling-stability)",
    )
    parser.add_argument("--out-dir", type=Path, help="default: output/studies/frontier-subsampling")
    args = parser.parse_args(argv)
    root = args.root.absolute()
    result = run(
        root=root,
        subsampling_root=args.subsampling_root or root / "output" / "studies" / subsampling.STUDY,
        out_dir=args.out_dir or root / "output" / "studies" / STUDY,
    )
    flagged = beyond_one_step(result["membership"])
    print(
        f"[frontier-subsampling] {len(result['scores'])} replicate x encoder scores; "
        f"{len(flagged)} cohort x fraction x encoder memberships differ by more than one step",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
