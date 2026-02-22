from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger("mari")

_NEIGHBOR_BUFFER = 64
_REDUCED_EFFECTIVE_K_WARN_THRESHOLD = 0.10


def _require_sklearn():
    try:
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "RI/MaRI requires scikit-learn. Install `scikit-learn` to run these metrics."
        ) from exc
    return balanced_accuracy_score, NearestNeighbors


def _filter_neighbors_excluding_same_slide(
    raw_neighbors: np.ndarray,
    slide_ids: np.ndarray,
    kmax: int,
    raw_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_k = int(kmax)
    if target_k <= 0:
        raise ValueError("kmax must be > 0")

    n_samples = int(len(slide_ids))
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
            if slide_ids[idx] == slide_ids[i]:
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
            "[RI/MaRI] %s: effective k < %d for %d/%d samples (%.1f%%) after excluding same-slide neighbors.",
            str(context),
            k,
            reduced,
            n,
            float(frac * 100.0),
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


def _prepare_neighbors(
    features: np.ndarray,
    slide_ids: np.ndarray,
    kmax: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _balanced_accuracy_score, NearestNeighbors = _require_sklearn()

    if int(kmax) <= 0:
        raise ValueError("kmax must be > 0")
    if len(features) <= 1:
        raise RuntimeError("Need at least two samples to compute neighbors")

    n_neighbors = min(int(kmax) + _NEIGHBOR_BUFFER, len(features) - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(features)
    distances, neigh = nn.kneighbors(features)
    return _filter_neighbors_excluding_same_slide(
        raw_neighbors=neigh,
        raw_distances=distances,
        slide_ids=slide_ids,
        kmax=int(kmax),
    )


def _optimal_k_by_knn_balanced_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    slide_ids: np.ndarray,
    k_values: Sequence[int],
    warn_context: str,
) -> int:
    balanced_accuracy_score, _NearestNeighbors = _require_sklearn()

    candidates = [int(k) for k in k_values]
    if not candidates:
        raise ValueError("k_values must contain at least one candidate")
    if min(candidates) <= 0:
        raise ValueError("k_values must be strictly positive")

    kmax = int(max(candidates))
    neigh, _dist, valid_counts = _prepare_neighbors(features, slide_ids, kmax)
    _warn_if_effective_k_reduced(valid_counts=valid_counts, target_k=kmax, context=warn_context)

    best_k = int(candidates[0])
    best_score = -1.0

    for k in candidates:
        pred, used_mask = _predict_labels_from_neighbors(
            labels=labels,
            neigh_idx=neigh,
            valid_counts=valid_counts,
            k=int(k),
        )
        if not bool(np.any(used_mask)):
            continue
        score = float(balanced_accuracy_score(labels[used_mask], pred[used_mask]))
        if score > best_score:
            best_score = score
            best_k = int(k)

    if best_score < 0.0:
        raise RuntimeError("k-selection failed: no sample has any cross-slide neighbor")

    return best_k

