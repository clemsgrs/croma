"""Cluster bootstrap utilities for uncertainty quantification on the headline
robustness statistics (CRoMa, and equally RI/MaRI).

The per-sample CRoMa values are not independent: samples from one independence
group (the manifest's ``group_id`` -- a slide, a patient, a specimen) share tissue,
staining and scanner, so an i.i.d. sample-level resample would understate
uncertainty. We therefore resample at the *group* (cluster) level -- draw groups
with replacement and pool all of their samples -- which is the honest (and
conservative) bootstrap for these benchmarks. It degrades gracefully to the i.i.d.
case for slide-level datasets (e.g. PANDA), where each group contributes a single
sample and a cluster is a singleton.

For across-model comparisons (rank stability) all models are evaluated on exactly
the same samples (a paired design), so a single shared cluster resample is applied
to every model within a replicate. This cancels the between-replicate noise common
to all models and isolates genuine model-to-model differences -- the right way to
ask "is model A's lead over model B real?".

The cluster draw can be *stratified* (``strata``): clusters are then resampled within
each stratum, so every replicate keeps each stratum's cluster count and a balanced
design stays balanced -- class x centre strata with slides as clusters, or centre
strata with cases as clusters where a case spans several classes.
:func:`cluster_bootstrap_draws` exposes the draw itself, and
:func:`shared_draw_tail_statistics` bootstraps every encoder's median, ``F(0)`` and
``LTM`` on one shared draw per replicate.

Across encoders, :func:`spearman_inference` and :func:`partial_spearman_inference` give a
correlation with an encoder-bootstrap CI and a two-sided permutation p, and
:func:`benjamini_hochberg` turns a family of p values into q values.

All routines are deterministic given ``seed`` and operate purely on cached
per-sample values, so they never recompute neighbours.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import rankdata, spearmanr

from croma.metrics.croma import _confounder_dominant_fraction
from croma.metrics.tail import compute_tail_metrics

__all__ = [
    "BootstrapCI",
    "RankStability",
    "TailBootstrap",
    "CorrelationInference",
    "TAIL_STATISTICS",
    "EXACT_PERMUTATION_MAX_N",
    "cluster_bootstrap_draws",
    "bootstrap_pooled_median",
    "paired_cluster_bootstrap_delta",
    "paired_rank_stability",
    "shared_draw_tail_statistics",
    "bootstrap_spearman",
    "encoder_bootstrap_draws",
    "spearman_inference",
    "partial_spearman_inference",
    "benjamini_hochberg",
]


@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate with a percentile confidence interval."""

    point: float
    lo: float
    hi: float
    level: float
    n_boot: int


@dataclass(frozen=True)
class RankStability:
    """Bootstrap distribution of model rankings under a paired cluster resample.

    Ranks are 1-based and descending: rank 1 is the most robust model (highest
    pooled-median value).
    """

    models: list[str]
    point_value: dict[str, float]
    point_rank: dict[str, int]
    mean_rank: dict[str, float]
    rank_lo: dict[str, int]
    rank_hi: dict[str, int]
    value_ci: dict[str, BootstrapCI]
    pairwise_win: dict[tuple[str, str], float]
    n_boot: int


def _percentile_ci(boot: np.ndarray, point: float, level: float, n_boot: int) -> BootstrapCI:
    finite = boot[np.isfinite(boot)]
    if finite.size == 0:
        return BootstrapCI(point, float("nan"), float("nan"), level, 0)
    lo_q = (1.0 - level) / 2.0 * 100.0
    hi_q = 100.0 - lo_q
    return BootstrapCI(
        point=float(point),
        lo=float(np.percentile(finite, lo_q, method="linear")),
        hi=float(np.percentile(finite, hi_q, method="linear")),
        level=float(level),
        n_boot=int(n_boot),
    )


def _cluster_row_groups(cluster_ids: np.ndarray) -> list[np.ndarray]:
    """Return, for each unique cluster, the array of row indices it contains."""
    cluster_ids = np.asarray(cluster_ids)
    order = np.argsort(cluster_ids, kind="stable")
    sorted_ids = cluster_ids[order]
    if sorted_ids.size == 0:
        return []
    # boundaries between consecutive distinct cluster ids
    boundaries = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
    return np.split(order, boundaries)


@dataclass(frozen=True)
class _ClusterDesign:
    """Clusters (row indices, in sorted cluster-id order) nested in strata.

    ``strata_members[s]`` holds the positions in ``groups`` of the clusters in stratum
    ``s``. Without strata there is one stratum holding every cluster in order, so a draw
    is the same single ``rng.integers(0, n_clusters, n_clusters)`` call the unstratified
    bootstrap has always made -- the random stream is unchanged.
    """

    groups: list[np.ndarray]
    strata_members: list[np.ndarray]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """Cluster positions for one replicate: each stratum resampled with replacement
        to its own cluster count, strata in sorted-label order."""
        return np.concatenate(
            [
                members[rng.integers(0, members.size, members.size)]
                for members in self.strata_members
            ]
        )


def _cluster_design(cluster_ids: np.ndarray, strata: np.ndarray | None) -> _ClusterDesign:
    cluster_ids = np.asarray(cluster_ids)
    groups = _cluster_row_groups(cluster_ids)
    if strata is None:
        return _ClusterDesign(groups, [np.arange(len(groups))])
    strata = np.asarray(strata)
    if strata.shape != cluster_ids.shape:
        raise ValueError("strata must be one label per row, the same length as cluster_ids")
    _, codes = np.unique(strata, return_inverse=True)
    codes = codes.reshape(-1)
    cluster_stratum = np.empty(len(groups), dtype=int)
    for position, rows in enumerate(groups):
        row_codes = codes[rows]
        if row_codes.min() != row_codes.max():
            raise ValueError(
                f"cluster {cluster_ids[rows[0]]!r} spans several strata; every cluster must "
                "belong to exactly one stratum"
            )
        cluster_stratum[position] = row_codes[0]
    members = [np.flatnonzero(cluster_stratum == code) for code in np.unique(cluster_stratum)]
    # no clusters at all: draw nothing, exactly as the unstratified design does
    return _ClusterDesign(groups, members or [np.arange(0)])


def cluster_bootstrap_draws(
    cluster_ids: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Iterator[np.ndarray]:
    """Yield the row indices of each replicate of a (stratified) cluster bootstrap.

    Clusters are resampled with replacement and every row of a drawn cluster is kept, so
    a cluster drawn twice contributes its rows twice. With ``strata`` (one label per row;
    every cluster must lie in exactly one stratum) clusters are resampled *within* each
    stratum, so every replicate keeps each stratum's cluster count. That keeps a class x
    centre design balanced: with slides as clusters and ``strata`` the class x centre
    label, no replicate can lose a cell or tilt the class mix. Without ``strata`` this is
    the plain cluster draw the other helpers here use, and with the same ``seed`` it is
    the same draw.

    One yielded array is one replicate; apply it to every model to get a paired (shared)
    draw. The design is validated when this is called, before any replicate is requested.
    """
    design = _cluster_design(cluster_ids, strata)

    def replicates() -> Iterator[np.ndarray]:
        rng = np.random.default_rng(seed)
        for _ in range(int(n_boot)):
            pick = design.draw(rng)
            yield np.concatenate([design.groups[i] for i in pick])

    return replicates()


def bootstrap_pooled_median(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Cluster-bootstrap CI for the pooled per-sample median of ``values``.

    Non-finite values (undefined samples) are dropped from each resample, matching
    how the headline pooled median is computed. Clusters are resampled with
    replacement; all rows of a drawn cluster are pooled before taking the median.
    With ``strata`` (one label per row) clusters are resampled within each stratum,
    as in :func:`cluster_bootstrap_draws`; without it the draw is unchanged.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape[0] != cluster_ids.shape[0]:
        raise ValueError("values and cluster_ids must be the same length")

    finite_all = values[np.isfinite(values)]
    point = float(np.median(finite_all)) if finite_all.size else float("nan")

    design = _cluster_design(cluster_ids, strata)
    groups = design.groups
    if len(groups) == 0:
        return BootstrapCI(point, float("nan"), float("nan"), level, 0)

    # pre-extract finite values per cluster so each replicate is a concat + median
    cluster_values = [values[g][np.isfinite(values[g])] for g in groups]

    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        pick = design.draw(rng)
        pooled = np.concatenate([cluster_values[i] for i in pick])
        boot[b] = np.median(pooled) if pooled.size else float("nan")

    return _percentile_ci(boot, point, level, int(n_boot))


def paired_cluster_bootstrap_delta(
    canonical: np.ndarray,
    alternative: np.ndarray,
    group_ids: np.ndarray,
    *,
    subset_ids: np.ndarray | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Paired shared-group bootstrap for an alternative-minus-canonical median.

    One independence-group draw is applied to both aligned occurrence vectors in
    each replicate. The returned point and every replicate are differences between
    the two pooled medians, not medians of the occurrence-level differences.
    """

    canonical = np.asarray(canonical, dtype=float)
    alternative = np.asarray(alternative, dtype=float)
    group_ids = np.asarray(group_ids)
    subsets = None if subset_ids is None else np.asarray(subset_ids)
    if canonical.ndim != 1 or alternative.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("canonical, alternative, and group_ids must be one-dimensional")
    if not (len(canonical) == len(alternative) == len(group_ids)):
        raise ValueError("canonical, alternative, and group_ids must be the same length")
    if subsets is not None and (subsets.ndim != 1 or len(subsets) != len(canonical)):
        raise ValueError("subset_ids must be one-dimensional and match the value length")
    if not np.isfinite(canonical).all() or not np.isfinite(alternative).all():
        raise ValueError("paired CRoMa values must all be finite")

    def balanced_median(values: np.ndarray, rows: np.ndarray) -> float:
        if subsets is None:
            return float(np.median(values[rows]))
        subset_medians = [
            np.median(values[rows][subsets[rows] == subset]) for subset in np.unique(subsets[rows])
        ]
        return float(np.median(subset_medians))

    groups = _cluster_row_groups(group_ids)
    if not groups:
        raise ValueError("paired bootstrap requires at least one independence group")
    all_rows = np.arange(len(canonical), dtype=int)
    point = balanced_median(alternative, all_rows) - balanced_median(canonical, all_rows)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for replicate in range(int(n_boot)):
        picked_groups = rng.integers(0, len(groups), len(groups))
        rows = np.concatenate([groups[index] for index in picked_groups])
        boot[replicate] = balanced_median(alternative, rows) - balanced_median(canonical, rows)
    return _percentile_ci(boot, point, level, int(n_boot))


def paired_rank_stability(
    model_values: dict[str, np.ndarray],
    cluster_ids: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> RankStability:
    """Paired cluster bootstrap of the model ranking by pooled-median value.

    Every model is evaluated on the same rows, so one shared cluster resample per
    replicate is applied to all models. Returns each model's point value/rank, its
    mean and percentile-interval rank across replicates, a CI on its pooled-median
    value, and the pairwise probability ``P(value[A] > value[B])``. With ``strata``
    (one label per row) clusters are resampled within each stratum, as in
    :func:`cluster_bootstrap_draws`; without it the draw is unchanged.
    """
    models = list(model_values)
    if not models:
        raise ValueError("model_values must be non-empty")
    arrs = {m: np.asarray(v, dtype=float) for m, v in model_values.items()}
    n_rows = len(next(iter(arrs.values())))
    for m, a in arrs.items():
        if a.shape[0] != n_rows:
            raise ValueError(f"model '{m}' values length != other models")
    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.shape[0] != n_rows:
        raise ValueError("cluster_ids length must match per-model values length")

    design = _cluster_design(cluster_ids, strata)
    groups = design.groups
    cluster_values = {m: [a[g][np.isfinite(a[g])] for g in groups] for m, a in arrs.items()}

    def _pooled_median(per_cluster: list[np.ndarray], pick: np.ndarray) -> float:
        pooled = np.concatenate([per_cluster[i] for i in pick])
        return float(np.median(pooled)) if pooled.size else float("nan")

    point_value = {
        m: (float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan"))
        for m, a in arrs.items()
    }
    value_draws = np.empty((int(n_boot), len(models)), dtype=float)
    rng = np.random.default_rng(seed)
    for b in range(int(n_boot)):
        pick = design.draw(rng)
        for j, m in enumerate(models):
            value_draws[b, j] = _pooled_median(cluster_values[m], pick)

    ranks, _ = _rank_summary(models, point_value, value_draws, level=level, n_boot=int(n_boot))
    return ranks


def _rank_summary(
    models: list[str],
    point_value: dict[str, float],
    value_draws: np.ndarray,
    *,
    level: float,
    n_boot: int,
) -> tuple[RankStability, np.ndarray]:
    """Rank bookkeeping over replicate values, shape ``(n_boot, len(models))``.

    Returns the :class:`RankStability` and the replicate ranks. A replicate's ranks are
    the stable descending sort: rank 1 is its highest value, an undefined (non-finite)
    value ranks last, and ties keep ``models`` order. ``P(A > B)`` counts the replicates
    where both are defined and A is strictly higher.
    """
    order = sorted(models, key=lambda m: point_value[m], reverse=True)
    point_rank = {m: i + 1 for i, m in enumerate(order)}

    key = np.where(np.isfinite(value_draws), value_draws, -np.inf)
    # [b, i, j] compares model j against model i within replicate b
    above = key[:, None, :] > key[:, :, None]
    tied_before = (key[:, None, :] == key[:, :, None]) & np.tri(len(models), k=-1, dtype=bool)
    rank_draws = 1 + above.sum(axis=2) + tied_before.sum(axis=2)
    # comparisons with nan are False, so an undefined value never wins nor loses
    win = (value_draws[:, :, None] > value_draws[:, None, :]).sum(axis=0) / float(n_boot)

    lo_q = (1.0 - level) / 2.0 * 100.0
    hi_q = 100.0 - lo_q
    idx_of = {m: j for j, m in enumerate(models)}
    stability = RankStability(
        models=models,
        point_value=point_value,
        point_rank=point_rank,
        mean_rank={m: float(rank_draws[:, j].mean()) for m, j in idx_of.items()},
        rank_lo={m: int(np.percentile(rank_draws[:, j], lo_q)) for m, j in idx_of.items()},
        rank_hi={m: int(np.percentile(rank_draws[:, j], hi_q)) for m, j in idx_of.items()},
        value_ci={
            m: _percentile_ci(value_draws[:, j], point_value[m], level, n_boot)
            for m, j in idx_of.items()
        },
        pairwise_win={
            (a, b): float(win[idx_of[a], idx_of[b]]) for a in models for b in models if a != b
        },
        n_boot=n_boot,
    )
    return stability, rank_draws


#: The per-encoder summaries :func:`shared_draw_tail_statistics` bootstraps, in order: the
#: pooled median, ``F(0)`` (the fraction ``<= 0``) and ``LTM_alpha`` (lower-tail mean).
TAIL_STATISTICS: tuple[str, ...] = ("median", "f0", "ltm")


@dataclass(frozen=True, eq=False)
class TailBootstrap:
    """Shared-draw bootstrap of every encoder's median, ``F(0)`` and ``LTM_alpha``.

    ``ci[statistic][model]`` is the point value with its percentile CI, for each
    ``statistic`` in :data:`TAIL_STATISTICS`. ``replicates[statistic]`` holds the replicate
    values, shape ``(n_boot, len(models))``, columns in ``models`` order; row ``b`` of
    every statistic and every encoder comes from the same cluster draw, so per-replicate
    derived quantities (a frontier, a cross-benchmark mean rank) can be built from them.

    ``median_rank`` ranks the encoders by median (point rank, mean rank, rank interval,
    pairwise ``P(A > B)``), exactly as :func:`paired_rank_stability` does on the same
    draws; ``rank_replicates`` holds each replicate's median ranks, shape
    ``(n_boot, len(models))``, rank 1 the highest.
    """

    models: list[str]
    alpha: float
    ci: dict[str, dict[str, BootstrapCI]]
    replicates: dict[str, np.ndarray]
    median_rank: RankStability
    rank_replicates: np.ndarray
    n_boot: int
    level: float

    def difference_ci(self, statistic: str, a: str, b: str) -> BootstrapCI:
        """Paired CI for ``statistic[a] - statistic[b]`` (e.g. a fine-tune minus its parent).

        The point is the difference of the two point values; the interval is the
        percentile interval of the replicate-by-replicate differences, which share one
        draw, at this bootstrap's ``level``.
        """
        if statistic not in TAIL_STATISTICS:
            raise ValueError(f"statistic must be one of {TAIL_STATISTICS}, got {statistic!r}")
        ia, ib = self.models.index(a), self.models.index(b)
        draws = self.replicates[statistic]
        point = self.ci[statistic][a].point - self.ci[statistic][b].point
        return _percentile_ci(draws[:, ia] - draws[:, ib], point, self.level, self.n_boot)


def _tail_summaries(values: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Pooled median, ``F(0)`` and ``LTM_alpha`` of the finite values.

    The median is the pooled per-sample median, as in :func:`bootstrap_pooled_median`;
    ``F(0)`` and ``LTM_alpha`` use the definitions behind
    :class:`~croma.types.CRoMaResult` ``f0`` and ``ltm_alpha``.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(finite)),
        _confounder_dominant_fraction(finite),
        compute_tail_metrics(finite, alpha=alpha).ltm_alpha,
    )


def shared_draw_tail_statistics(
    model_values: dict[str, np.ndarray],
    cluster_ids: np.ndarray,
    *,
    strata: np.ndarray | None = None,
    alpha: float = 0.10,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> TailBootstrap:
    """Bootstrap each encoder's pooled median, ``F(0)`` and ``LTM_alpha`` on shared draws.

    ``model_values`` maps each encoder to its per-sample values (e.g. CRoMa at the headline
    ``m``), every array aligned to the same rows as ``cluster_ids`` (and ``strata``). Each
    replicate is one draw from :func:`cluster_bootstrap_draws` with the same ``strata`` and
    ``seed``, applied to every encoder, so replicate ``b`` is directly comparable across
    encoders: the paired design. Non-finite (undefined) samples are dropped per replicate,
    as in the headline statistics. Margins are taken as given; neighbours are not
    recomputed. Ranks and win probabilities are among the encoders passed, so pass the
    ranked panel only (without the natural-image control).
    """
    models = list(model_values)
    if not models:
        raise ValueError("model_values must be non-empty")
    arrs = {m: np.asarray(v, dtype=float) for m, v in model_values.items()}
    cluster_ids = np.asarray(cluster_ids)
    for m, a in arrs.items():
        if a.shape != cluster_ids.shape:
            raise ValueError(f"model '{m}' values must align with cluster_ids")

    replicates = {s: np.empty((int(n_boot), len(models)), dtype=float) for s in TAIL_STATISTICS}
    draws = cluster_bootstrap_draws(cluster_ids, strata=strata, n_boot=n_boot, seed=seed)
    for b, rows in enumerate(draws):
        for j, m in enumerate(models):
            for s, value in zip(TAIL_STATISTICS, _tail_summaries(arrs[m][rows], alpha)):
                replicates[s][b, j] = value

    points = {m: dict(zip(TAIL_STATISTICS, _tail_summaries(arrs[m], alpha))) for m in models}
    ci = {
        s: {
            m: _percentile_ci(replicates[s][:, j], points[m][s], level, int(n_boot))
            for j, m in enumerate(models)
        }
        for s in TAIL_STATISTICS
    }
    median_rank, rank_replicates = _rank_summary(
        models,
        {m: points[m]["median"] for m in models},
        replicates["median"],
        level=level,
        n_boot=int(n_boot),
    )
    return TailBootstrap(
        models=models,
        alpha=float(alpha),
        ci=ci,
        replicates=replicates,
        median_rank=median_rank,
        rank_replicates=rank_replicates,
        n_boot=int(n_boot),
        level=float(level),
    )


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Spearman rank correlation between two per-model vectors, with a bootstrap CI.

    The CI resamples the paired points (models) with replacement -- the relevant
    uncertainty for a "how redundant are these metrics across models?" question.
    With a small number of models the interval is honestly wide.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must be the same shape")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = x.shape[0]
    if n < 3:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), level, 0)

    point = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            boot[b] = float("nan")
        else:
            boot[b] = float(spearmanr(xb, yb).statistic)
    return _percentile_ci(boot, point, level, int(n_boot))


#: Largest panel whose permutation p enumerates every ordering (8! = 40,320); larger
#: panels use seeded random permutations.
EXACT_PERMUTATION_MAX_N = 8

PermutationMethod = Literal["exact", "monte-carlo"]

# The permutation stream is seeded apart from the bootstrap draw, so passing ``draws``
# explicitly can never change a p value.
_PERMUTATION_STREAM = 1
# |null| >= |observed| up to float noise: a permutation that reproduces the observed
# statistic exactly must count, however its floating-point sum was ordered.
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, eq=False)
class CorrelationInference:
    """A correlation across encoders with its encoder-bootstrap CI and permutation p.

    ``ci.point`` equals ``rho``. ``p_value`` is two-sided: the share of permutations whose
    statistic is at least as extreme as ``rho`` in absolute value. ``permutation`` says how
    the null was formed: ``"exact"`` enumerates all ``n!`` orderings (``n <=``
    :data:`EXACT_PERMUTATION_MAX_N`); ``"monte-carlo"`` draws ``n_permutations`` seeded
    orderings and reports ``(1 + hits) / (1 + n_permutations)``, which is never 0.
    ``replicates`` holds the bootstrap replicates of the correlation (``nan`` where a
    replicate's drawn encoders are constant on a variable); two inferences computed on the
    same ``draws`` have paired replicates, so their difference has a percentile CI too.
    """

    rho: float
    ci: BootstrapCI
    p_value: float
    n: int
    permutation: PermutationMethod
    n_permutations: int
    replicates: np.ndarray


def _permutation_orders(
    n: int, n_permutations: int, seed: int
) -> tuple[np.ndarray, PermutationMethod]:
    """Orderings of ``n`` encoders for a permutation null, and how they were formed."""
    if n <= EXACT_PERMUTATION_MAX_N:
        return np.array(list(itertools.permutations(range(n)))), "exact"
    rng = np.random.default_rng([seed, _PERMUTATION_STREAM])
    identity = np.tile(np.arange(n), (int(n_permutations), 1))
    return rng.permuted(identity, axis=1), "monte-carlo"


def _two_sided_p(observed: float, null: np.ndarray, permutation: PermutationMethod) -> float:
    hits = int(np.count_nonzero(np.abs(null) >= abs(observed) - _TIE_TOLERANCE))
    if permutation == "exact":
        return hits / null.size
    return (hits + 1) / (null.size + 1)


def encoder_bootstrap_draws(n: int, *, n_boot: int = 2000, seed: int = 0) -> np.ndarray:
    """Encoder indices for an encoder bootstrap, shape ``(n_boot, n)``.

    Row ``b`` resamples the ``n`` encoders with replacement. Pass the same array as
    ``draws`` to several correlation helpers to share one encoder draw between them (e.g.
    for the CI of a difference between two correlations on the same panel); without
    ``draws`` each helper makes exactly this draw from its own ``seed``.
    """
    if int(n) < 1:
        raise ValueError("an encoder bootstrap needs at least one encoder")
    return np.random.default_rng(seed).integers(0, int(n), (int(n_boot), int(n)))


def _encoder_vectors(*vectors: np.ndarray) -> list[np.ndarray]:
    """Validate aligned, finite, 1-D per-encoder vectors with at least three encoders."""
    arrays = [np.asarray(v, dtype=float) for v in vectors]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("per-encoder vectors must be one-dimensional and the same length")
    n = arrays[0].shape[0]
    if n < 3:
        raise ValueError("a correlation across encoders needs at least three encoders")
    for a in arrays:
        if not np.isfinite(a).all():
            raise ValueError(
                "per-encoder vectors must be finite; choose the panel before calling, "
                "rather than letting undefined encoders drop out silently"
            )
        if np.ptp(a) == 0:
            raise ValueError("a correlation is undefined for a constant per-encoder vector")
    return arrays


def _row_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis (ties share their mean rank)."""
    return rankdata(values, method="average", axis=-1)


def _row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis; ``nan`` where either row is constant."""
    ac = a - a.mean(axis=-1, keepdims=True)
    bc = b - b.mean(axis=-1, keepdims=True)
    constant = (np.ptp(a, axis=-1) == 0) | (np.ptp(b, axis=-1) == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ac * bc).sum(axis=-1) / np.sqrt((ac**2).sum(axis=-1) * (bc**2).sum(axis=-1))
    return np.where(constant, np.nan, r)


def _resolve_draws(draws: np.ndarray | None, n: int, n_boot: int, seed: int) -> np.ndarray:
    if draws is None:
        return encoder_bootstrap_draws(n, n_boot=n_boot, seed=seed)
    draws = np.asarray(draws)
    if draws.ndim != 2 or draws.shape[1] != n:
        raise ValueError(f"draws must have shape (n_boot, {n}), one row per replicate")
    if draws.size and (draws.min() < 0 or draws.max() >= n):
        raise ValueError(f"draws must index encoders 0..{n - 1}")
    return draws


def spearman_inference(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    n_permutations: int = 10_000,
    draws: np.ndarray | None = None,
) -> CorrelationInference:
    """Spearman's rho between two per-encoder vectors, with a CI and a permutation p.

    ``x[i]`` and ``y[i]`` describe encoder ``i``; both must be finite (choose the panel,
    e.g. drop the control, before calling). The CI is the percentile interval of rho
    over encoder resamples -- an interval over *this* panel of encoders. The two-sided
    p permutes ``y`` against ``x``: exactly, over all ``n!`` orderings, when ``n <=``
    :data:`EXACT_PERMUTATION_MAX_N`, otherwise over ``n_permutations`` seeded orderings.
    ``draws`` (from :func:`encoder_bootstrap_draws`) shares one encoder draw across calls;
    when given it fixes ``n_boot`` to its row count.
    """
    x, y = _encoder_vectors(x, y)
    n = x.shape[0]
    rx, ry = _row_ranks(x), _row_ranks(y)
    rho = float(_row_pearson(rx, ry))

    idx = _resolve_draws(draws, n, n_boot, seed)
    replicates = _row_pearson(_row_ranks(x[idx]), _row_ranks(y[idx]))
    ci = _percentile_ci(replicates, rho, level, int(idx.shape[0]))

    orders, permutation = _permutation_orders(n, n_permutations, seed)
    null = _row_pearson(rx, ry[orders])
    return CorrelationInference(
        rho=rho,
        ci=ci,
        p_value=_two_sided_p(rho, null, permutation),
        n=n,
        permutation=permutation,
        n_permutations=int(orders.shape[0]),
        replicates=replicates,
    )


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini--Hochberg q values for one family of p values, in the input order.

    With the ``m`` p values sorted ascending, ``q_(j) = min_{k >= j} m p_(k) / k``: the
    smallest false-discovery rate at which hypothesis ``j`` would be rejected. The family
    is exactly what is passed in, so call it once per table (or whichever family the
    caption declares). Tied p values get equal q values.
    """
    p = np.asarray(p_values, dtype=float).reshape(-1)
    if not np.all((p >= 0.0) & (p <= 1.0)):  # also rejects nan
        raise ValueError("p values must be finite and within [0, 1]")
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    stepped = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(stepped[::-1])[::-1]
    q = np.empty(m, dtype=float)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def _residualise(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residual of ``a`` on ``[1, z]`` by least squares, row-wise along the last axis;
    ``nan`` where ``z`` is constant (the regression is then undefined)."""
    ac = a - a.mean(axis=-1, keepdims=True)
    zc = z - z.mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (ac * zc).sum(axis=-1, keepdims=True) / (zc**2).sum(axis=-1, keepdims=True)
    residual = ac - slope * zc
    return np.where(np.ptp(z, axis=-1, keepdims=True) == 0, np.nan, residual)


def partial_spearman_inference(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    n_permutations: int = 10_000,
    draws: np.ndarray | None = None,
) -> CorrelationInference:
    """Partial Spearman ``rho(x, y | z)`` across encoders, with a CI and a permutation p.

    ``x``, ``y`` and ``z`` are rank-transformed (average ranks), ``rank(x)`` and
    ``rank(y)`` are residualised on ``rank(z)`` by least squares with an intercept, and
    the partial rho is the Pearson correlation of the residuals. With one covariate this
    equals ``(r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))`` on Spearman's ``r``.

    The CI is an encoder bootstrap: each replicate re-ranks and re-residualises the drawn
    encoders. The two-sided p is Freedman--Lane: ``rank(y)``'s residuals on ``rank(z)``
    are permuted and added back to its fitted values, and each permuted response is
    residualised again and correlated with ``rank(x)``'s residual. Enumeration is exact
    for ``n <=`` :data:`EXACT_PERMUTATION_MAX_N`, as in :func:`spearman_inference`, whose
    ``draws`` and seeding conventions this shares.
    """
    x, y, z = _encoder_vectors(x, y, z)
    n = x.shape[0]
    rx, ry, rz = _row_ranks(x), _row_ranks(y), _row_ranks(z)
    res_x, res_y = _residualise(rx, rz), _residualise(ry, rz)
    rho = float(_row_pearson(res_x, res_y))
    if not np.isfinite(rho):
        raise ValueError("partial rho is undefined: x or y is fully determined by z")

    idx = _resolve_draws(draws, n, n_boot, seed)
    zb = _row_ranks(z[idx])
    replicates = _row_pearson(
        _residualise(_row_ranks(x[idx]), zb), _residualise(_row_ranks(y[idx]), zb)
    )
    ci = _percentile_ci(replicates, rho, level, int(idx.shape[0]))

    orders, permutation = _permutation_orders(n, n_permutations, seed)
    fitted_y = ry - res_y
    null = _row_pearson(res_x, _residualise(fitted_y + res_y[orders], rz))
    return CorrelationInference(
        rho=rho,
        ci=ci,
        p_value=_two_sided_p(rho, null, permutation),
        n=n,
        permutation=permutation,
        n_permutations=int(orders.shape[0]),
        replicates=replicates,
    )
