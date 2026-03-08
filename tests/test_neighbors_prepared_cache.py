import numpy as np

from mari.metrics.neighbors import (
    _balanced_accuracy_by_k_from_prepared_neighbors,
    _knn_balanced_accuracy_by_k,
    _prepare_neighbors,
)


def test_balanced_accuracy_from_prepared_neighbors_matches_direct_knn() -> None:
    features = np.asarray(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.85, 0.15, 0.00, 0.00],
            [0.15, 0.85, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.98, 0.02, 0.00, 0.00],
            [0.82, 0.18, 0.00, 0.00],
            [0.18, 0.82, 0.00, 0.00],
            [0.02, 0.98, 0.00, 0.00],
        ],
        dtype=float,
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
    slide_ids = np.asarray(["sl0", "sl1", "sl2", "sl3", "sl0b", "sl4", "sl5", "sl6"], dtype=object)
    k_values = [1, 3]

    direct = _knn_balanced_accuracy_by_k(
        features=features,
        labels=labels,
        slide_ids=slide_ids,
        k_values=k_values,
        warn_context="toy",
    )
    neigh_idx, _neigh_dist, valid_counts = _prepare_neighbors(features, slide_ids, max(k_values))
    cached = _balanced_accuracy_by_k_from_prepared_neighbors(
        labels=labels,
        neigh_idx=neigh_idx,
        valid_counts=valid_counts,
        k_values=k_values,
    )

    assert cached == direct
