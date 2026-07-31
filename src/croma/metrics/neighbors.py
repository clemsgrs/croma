import math
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("croma")

_MIN_NEIGHBOR_BUFFER = 64
_TARGET_EFFECTIVE_K_COVERAGE = 0.90
_GROWTH_FACTOR = 1.5
_MIN_GROWTH_STEP = 32
_REDUCED_EFFECTIVE_K_WARN_THRESHOLD = 0.10


@dataclass(frozen=True)
class _NeighborPreparationMeta:
    final_n_neighbors: int
    coverage: float
    target_coverage: float
    iterations: int
    hit_neighbor_cap: bool


def _max_rows_per_group(group_ids: np.ndarray) -> int:
    if int(group_ids.size) <= 0:
        return 0
    _groups, counts = np.unique(group_ids, return_counts=True)
    if int(counts.size) <= 0:
        return 0
    return int(counts.max())


def _initial_n_neighbors(kmax: int, group_ids: np.ndarray, n_samples: int) -> int:
    inferred_buffer = max(_MIN_NEIGHBOR_BUFFER, _max_rows_per_group(group_ids))
    return int(min(int(kmax) + int(inferred_buffer), int(n_samples) - 1))


def _effective_k_coverage(valid_counts: np.ndarray, kmax: int) -> float:
    n = int(valid_counts.size)
    if n <= 0:
        return 1.0
    reached = int(np.count_nonzero(valid_counts >= int(kmax)))
    return float(reached) / float(n)


def _next_n_neighbors(current: int, n_samples: int, kmax: int) -> int:
    if int(current) >= int(n_samples) - 1:
        return int(n_samples) - 1

    candidate_linear = int(current) + max(_MIN_GROWTH_STEP, int(kmax))
    candidate_geometric = int(math.ceil(float(current) * _GROWTH_FACTOR))
    grown = max(candidate_linear, candidate_geometric)
    next_neighbors = min(int(n_samples) - 1, int(grown))
    if next_neighbors <= int(current):
        return min(int(n_samples) - 1, int(current) + 1)
    return int(next_neighbors)


def _filter_neighbors_excluding_same_group(
    raw_neighbors: np.ndarray,
    group_ids: np.ndarray,
    kmax: int,
    raw_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_k = int(kmax)
    if target_k <= 0:
        raise ValueError("kmax must be > 0")

    n_samples = int(len(group_ids))
    if int(raw_neighbors.shape[0]) != n_samples:
        raise ValueError("raw_neighbors row count must match number of samples")
    if raw_distances is not None and raw_distances.shape != raw_neighbors.shape:
        raise ValueError("raw_distances must have the same shape as raw_neighbors")

    out_idx = np.full((n_samples, target_k), -1, dtype=int)
    out_dist = np.full((n_samples, target_k), np.inf, dtype=float)
    valid_counts = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        vals: list[int] = []
        dists: list[float] = []
        for pos, j in enumerate(raw_neighbors[i].tolist()):
            idx = int(j)
            if idx == i:
                continue
            if group_ids[idx] == group_ids[i]:
                continue
            vals.append(idx)
            if raw_distances is not None:
                dists.append(float(raw_distances[i, pos]))
            if len(vals) == target_k:
                break
        if vals:
            out_idx[i, : len(vals)] = np.asarray(vals, dtype=int)
            if raw_distances is not None:
                out_dist[i, : len(dists)] = np.asarray(dists, dtype=float)
        valid_counts[i] = int(len(vals))

    if raw_distances is None:
        return out_idx, valid_counts
    return out_idx, out_dist, valid_counts


def _filter_query_neighbors_excluding_same_group(
    raw_neighbors: np.ndarray,
    query_indices: np.ndarray,
    group_ids: np.ndarray,
    kmax: int,
    raw_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_k = int(kmax)
    if target_k <= 0:
        raise ValueError("kmax must be > 0")

    n_query = int(raw_neighbors.shape[0])
    if int(query_indices.shape[0]) != n_query:
        raise ValueError("query_indices length must match raw_neighbors rows")
    if raw_distances is not None and raw_distances.shape != raw_neighbors.shape:
        raise ValueError("raw_distances must have the same shape as raw_neighbors")
    if int(len(group_ids)) <= 0:
        raise ValueError("group_ids must be non-empty")

    out_idx = np.full((n_query, target_k), -1, dtype=int)
    out_dist = np.full((n_query, target_k), np.inf, dtype=float)
    valid_counts = np.zeros(n_query, dtype=int)

    for row in range(n_query):
        query_idx = int(query_indices[row])
        vals: list[int] = []
        dists: list[float] = []
        for pos, j in enumerate(raw_neighbors[row].tolist()):
            idx = int(j)
            if idx == query_idx:
                continue
            if group_ids[idx] == group_ids[query_idx]:
                continue
            vals.append(idx)
            if raw_distances is not None:
                dists.append(float(raw_distances[row, pos]))
            if len(vals) == target_k:
                break
        if vals:
            out_idx[row, : len(vals)] = np.asarray(vals, dtype=int)
            if raw_distances is not None:
                out_dist[row, : len(dists)] = np.asarray(dists, dtype=float)
        valid_counts[row] = int(len(vals))

    if raw_distances is None:
        return out_idx, valid_counts
    return out_idx, out_dist, valid_counts


def _warn_if_effective_k_reduced(
    valid_counts: np.ndarray,
    target_k: int,
    context: str,
    threshold: float = _REDUCED_EFFECTIVE_K_WARN_THRESHOLD,
) -> None:
    k = int(target_k)
    n = int(valid_counts.size)
    if k <= 0 or n <= 0:
        return
    reduced = int(np.count_nonzero(valid_counts < k))
    frac = float(reduced) / float(n)
    if frac > float(threshold):
        logger.warning(
            f"[RI/MaRI] {context}: effective k < {k} for {reduced}/{n} samples "
            f"({frac * 100.0:.1f}%) after excluding same-group neighbors."
        )


def _predict_labels_from_neighbors(
    labels: np.ndarray,
    neigh_idx: np.ndarray,
    valid_counts: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_k = int(k)
    pred = np.full((len(labels),), -1, dtype=int)
    used_mask = np.zeros((len(labels),), dtype=bool)

    for i in range(len(labels)):
        eff_k = min(target_k, int(valid_counts[i]))
        if eff_k <= 0:
            continue
        row = neigh_idx[i, :eff_k]
        row = row[row >= 0]
        if row.size <= 0:
            continue
        vals, cnt = np.unique(labels[row], return_counts=True)
        pred[i] = int(vals[np.argmax(cnt)])
        used_mask[i] = True

    return pred, used_mask


def _balanced_accuracy_by_k_from_prepared_neighbors(
    labels: np.ndarray,
    neigh_idx: np.ndarray,
    valid_counts: np.ndarray,
    k_values: Sequence[int],
) -> dict[int, float]:
    candidates = _normalize_k_values(k_values)
    out: dict[int, float] = {}
    for k in candidates:
        pred, used_mask = _predict_labels_from_neighbors(
            labels=labels,
            neigh_idx=neigh_idx,
            valid_counts=valid_counts,
            k=int(k),
        )
        if not bool(np.any(used_mask)):
            continue
        out[int(k)] = float(balanced_accuracy_score(labels[used_mask], pred[used_mask]))
    if not out:
        raise RuntimeError("k-selection failed: no sample has any cross-group neighbor")
    return out


def _prepare_neighbors(
    features: np.ndarray,
    group_ids: np.ndarray,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neigh_idx, neigh_dist, valid_counts, _meta = _prepare_neighbors_with_meta(
        features, group_ids, kmax
    )
    return neigh_idx, neigh_dist, valid_counts


def _prepare_neighbors_with_meta(
    features: np.ndarray,
    group_ids: np.ndarray,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _NeighborPreparationMeta]:
    if int(kmax) <= 0:
        raise ValueError("kmax must be > 0")
    n_samples = int(len(features))
    if n_samples <= 1:
        raise RuntimeError("Need at least two samples to compute neighbors")
    if int(len(group_ids)) != n_samples:
        raise ValueError("group_ids length must match features row count")

    target_k = int(kmax)
    target_coverage = float(_TARGET_EFFECTIVE_K_COVERAGE)
    n_neighbors = _initial_n_neighbors(target_k, group_ids, n_samples)
    iterations = 0

    while True:
        iterations += 1
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn.fit(features)
        distances, neigh = nn.kneighbors(features)
        neigh_idx, neigh_dist, valid_counts = _filter_neighbors_excluding_same_group(
            raw_neighbors=neigh,
            raw_distances=distances,
            group_ids=group_ids,
            kmax=target_k,
        )
        coverage = _effective_k_coverage(valid_counts, target_k)
        hit_neighbor_cap = bool(n_neighbors >= n_samples - 1)
        has_undefined = bool(np.any(valid_counts == 0))
        if (coverage >= target_coverage and not has_undefined) or hit_neighbor_cap:
            meta = _NeighborPreparationMeta(
                final_n_neighbors=int(n_neighbors),
                coverage=float(coverage),
                target_coverage=target_coverage,
                iterations=int(iterations),
                hit_neighbor_cap=hit_neighbor_cap,
            )
            return neigh_idx, neigh_dist, valid_counts, meta
        n_neighbors = _next_n_neighbors(n_neighbors, n_samples, target_k)


def _optimal_k_by_knn_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    k_values: Sequence[int],
    warn_context: str,
) -> int:
    scores = _knn_balanced_accuracy_by_k(
        features=features,
        labels=labels,
        group_ids=group_ids,
        k_values=k_values,
        warn_context=warn_context,
    )
    return _select_k_from_balanced_accuracy(k_values=k_values, scores=scores)


def _normalize_k_values(k_values: Sequence[int]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for raw_k in k_values:
        k = int(raw_k)
        if k <= 0:
            raise ValueError("k_values must be strictly positive")
        if k in seen:
            continue
        seen.add(k)
        normalized.append(k)
    if not normalized:
        raise ValueError("k_values must contain at least one candidate")
    return normalized


def _knn_balanced_accuracy_by_k(
    features: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    k_values: Sequence[int],
    warn_context: str,
) -> dict[int, float]:
    candidates = _normalize_k_values(k_values)
    kmax = int(max(candidates))

    neigh, _dist, valid_counts, prep_meta = _prepare_neighbors_with_meta(features, group_ids, kmax)
    capped = (
        ", capped"
        if prep_meta.hit_neighbor_cap and prep_meta.coverage < prep_meta.target_coverage
        else ""
    )
    warn_context_with_fetch = (
        f"{warn_context} [fetch={prep_meta.final_n_neighbors}/{len(features) - 1}, "
        f"coverage={prep_meta.coverage * 100.0:.1f}%, "
        f"target={prep_meta.target_coverage * 100.0:.1f}%{capped}]"
    )
    _warn_if_effective_k_reduced(
        valid_counts=valid_counts,
        target_k=kmax,
        context=warn_context_with_fetch,
    )
    return _balanced_accuracy_by_k_from_prepared_neighbors(
        labels=labels,
        neigh_idx=neigh,
        valid_counts=valid_counts,
        k_values=candidates,
    )


def _select_k_from_balanced_accuracy(
    k_values: Sequence[int],
    scores: dict[int, float],
) -> int:
    ordered = _normalize_k_values(k_values)
    best_k = int(ordered[0])
    best_score = float("-inf")

    for k in ordered:
        if int(k) not in scores:
            continue
        score = float(scores[int(k)])
        if score > best_score:
            best_score = score
            best_k = int(k)

    if best_score == float("-inf"):
        raise RuntimeError("k-selection failed: no sample has any cross-group neighbor")
    return int(best_k)
