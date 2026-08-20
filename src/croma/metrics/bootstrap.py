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

All routines are deterministic given ``seed`` and operate purely on cached
per-sample values, so they never recompute neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


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


def bootstrap_pooled_median(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> BootstrapCI:
    """Cluster-bootstrap CI for the pooled per-sample median of ``values``.

    Non-finite values (undefined samples) are dropped from each resample, matching
    how the headline pooled median is computed. Clusters are resampled with
    replacement; all rows of a drawn cluster are pooled before taking the median.
    """
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    if values.shape[0] != cluster_ids.shape[0]:
        raise ValueError("values and cluster_ids must be the same length")

    finite_all = values[np.isfinite(values)]
    point = float(np.median(finite_all)) if finite_all.size else float("nan")

    groups = _cluster_row_groups(cluster_ids)
    n_clusters = len(groups)
    if n_clusters == 0:
        return BootstrapCI(point, float("nan"), float("nan"), level, 0)

    # pre-extract finite values per cluster so each replicate is a concat + median
    cluster_values = [values[g][np.isfinite(values[g])] for g in groups]

    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_clusters, n_clusters)
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
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> RankStability:
    """Paired cluster bootstrap of the model ranking by pooled-median value.

    Every model is evaluated on the same rows, so one shared cluster resample per
    replicate is applied to all models. Returns each model's point value/rank, its
    mean and percentile-interval rank across replicates, a CI on its pooled-median
    value, and the pairwise probability ``P(value[A] > value[B])``.
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

    groups = _cluster_row_groups(cluster_ids)
    n_clusters = len(groups)
    cluster_values = {m: [a[g][np.isfinite(a[g])] for g in groups] for m, a in arrs.items()}

    def _pooled_median(per_cluster: list[np.ndarray], pick: np.ndarray) -> float:
        pooled = np.concatenate([per_cluster[i] for i in pick])
        return float(np.median(pooled)) if pooled.size else float("nan")

    point_value = {
        m: (float(np.median(a[np.isfinite(a)])) if np.isfinite(a).any() else float("nan"))
        for m, a in arrs.items()
    }
    order = sorted(models, key=lambda m: point_value[m], reverse=True)
    point_rank = {m: i + 1 for i, m in enumerate(order)}

    idx_of = {m: i for i, m in enumerate(models)}
    rank_draws = {m: np.empty(int(n_boot), dtype=int) for m in models}
    value_draws = {m: np.empty(int(n_boot), dtype=float) for m in models}
    win = np.zeros((len(models), len(models)), dtype=float)

    rng = np.random.default_rng(seed)
    for b in range(int(n_boot)):
        pick = rng.integers(0, n_clusters, n_clusters)
        meds = {m: _pooled_median(cluster_values[m], pick) for m in models}
        for m in models:
            value_draws[m][b] = meds[m]
        ordb = sorted(
            models,
            key=lambda m: (meds[m] if np.isfinite(meds[m]) else -np.inf),
            reverse=True,
        )
        for i, m in enumerate(ordb):
            rank_draws[m][b] = i + 1
        for a_name in models:
            ma = meds[a_name]
            if not np.isfinite(ma):
                continue
            ia = idx_of[a_name]
            for b_name in models:
                if a_name is b_name:
                    continue
                mb = meds[b_name]
                if np.isfinite(mb) and ma > mb:
                    win[ia, idx_of[b_name]] += 1.0
    win /= float(n_boot)

    lo_q = (1.0 - level) / 2.0 * 100.0
    hi_q = 100.0 - lo_q
    mean_rank = {m: float(rank_draws[m].mean()) for m in models}
    rank_lo = {m: int(np.percentile(rank_draws[m], lo_q)) for m in models}
    rank_hi = {m: int(np.percentile(rank_draws[m], hi_q)) for m in models}
    value_ci = {
        m: _percentile_ci(value_draws[m], point_value[m], level, int(n_boot)) for m in models
    }
    pairwise = {
        (a_name, b_name): float(win[idx_of[a_name], idx_of[b_name]])
        for a_name in models
        for b_name in models
        if a_name != b_name
    }

    return RankStability(
        models=models,
        point_value=point_value,
        point_rank=point_rank,
        mean_rank=mean_rank,
        rank_lo=rank_lo,
        rank_hi=rank_hi,
        value_ci=value_ci,
        pairwise_win=pairwise,
        n_boot=int(n_boot),
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
