
import numpy as np

from mari import MaRI, RI


def test_ri_pooled_fallback_returns_half_when_no_so_or_os() -> None:
    labels = np.array([0, 1], dtype=int)
    centers = np.array([0, 1], dtype=int)
    neigh_idx = np.array([[1], [0]], dtype=int)
    neigh_dist = np.array([[0.1], [0.1]], dtype=float)
    valid_counts = np.array([1, 1], dtype=int)

    score, sample_scores, informative_mask = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=1,
    )

    assert score == 0.5
    assert np.isnan(sample_scores).all()
    assert informative_mask.tolist() == [False, False]


def test_mari_weights_change_score_when_distances_swap() -> None:
    labels = np.array([0, 0, 1], dtype=int)
    centers = np.array([0, 1, 0], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],  # sample 0 has one SO neighbor (1) and one OS neighbor (2)
            [-1, -1],
            [-1, -1],
        ],
        dtype=int,
    )
    valid_counts = np.array([2, 0, 0], dtype=int)

    close_so = np.array([[0.1, 1.0], [np.inf, np.inf], [np.inf, np.inf]], dtype=float)
    close_os = np.array([[1.0, 0.1], [np.inf, np.inf], [np.inf, np.inf]], dtype=float)

    mari_close_so, _sample_scores, _informative_mask = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=close_so,
        valid_counts=valid_counts,
        k=2,
        tau=0.2,
    )
    mari_close_os, _sample_scores, _informative_mask = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=close_os,
        valid_counts=valid_counts,
        k=2,
        tau=0.2,
    )

    assert mari_close_so > mari_close_os


def test_mari_tau_controls_locality_strength() -> None:
    labels = np.array([0, 0, 1], dtype=int)
    centers = np.array([0, 1, 0], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],
            [-1, -1],
            [-1, -1],
        ],
        dtype=int,
    )
    neigh_dist = np.array([[0.1, 1.0], [np.inf, np.inf], [np.inf, np.inf]], dtype=float)
    valid_counts = np.array([2, 0, 0], dtype=int)

    small_tau, _sample_scores, _informative_mask = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
        tau=0.1,
    )
    large_tau, _sample_scores, _informative_mask = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
        tau=10.0,
    )

    assert small_tau > large_tau
