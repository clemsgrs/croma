import logging

import numpy as np
import pytest

from croma.metrics import neighbors as nb


def test_prepare_neighbors_no_undefined_samples_with_pruning() -> None:
    """Regression: hard samples (tight SS cluster) must not remain undefined
    when --prune-ss-oo is active. The old stopping condition (coverage >= 90%)
    would exit while those samples still had valid_counts == 0 because their
    only SO/OS neighbors were far away. The fix requires valid_counts > 0 for
    all samples before stopping.

    Setup:
    - 100 hard samples: label=0, center=0, tightly clustered at unit vector e_0.
      Their k-nearest neighbours at small n_neighbors are all SS (same label,
      same center) → valid_counts == 0 after pruning.
    - 900 easy samples: spread across other (label, center) combinations at
      random positions, so they quickly accumulate SO/OS neighbours.

    At n_neighbors = 65 (initial buffer), 900/1000 = 90 % coverage is already
    reached, which would trigger the old exit. The hard 100 still have
    valid_counts == 0 at that point. The new stopping condition continues
    until the hard samples' SO/OS neighbours enter the search window.
    """
    rng = np.random.default_rng(42)
    d = 50
    n_hard = 100  # label=0, center=0, clustered at e_0
    n_easy = 900  # diverse, random positions

    # Hard samples: tightly clustered at e_0
    hard_feats = np.zeros((n_hard, d))
    hard_feats[:, 0] = 1.0
    hard_feats += rng.standard_normal((n_hard, d)) * 1e-4
    hard_labels = np.zeros(n_hard, dtype=int)
    hard_centers = np.zeros(n_hard, dtype=int)
    hard_slides = np.array([f"slide-hard-{i}" for i in range(n_hard)])

    # Easy samples: 225 per (label, center) combo that is NOT (0, 0)
    easy_combos = [(0, 1), (1, 0), (0, 2), (1, 2)]
    easy_feats_list, easy_labels_list, easy_centers_list, easy_slides_list = [], [], [], []
    for lbl, ctr in easy_combos:
        n = n_easy // len(easy_combos)
        feats = rng.standard_normal((n, d))
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        easy_feats_list.append(feats)
        easy_labels_list.append(np.full(n, lbl, dtype=int))
        easy_centers_list.append(np.full(n, ctr, dtype=int))
        easy_slides_list.extend([f"slide-easy-{lbl}-{ctr}-{i}" for i in range(n)])

    features = np.vstack([hard_feats, *easy_feats_list])
    labels = np.concatenate([hard_labels, *easy_labels_list])
    centers = np.concatenate([hard_centers, *easy_centers_list])
    slide_ids = np.array(hard_slides.tolist() + easy_slides_list)

    # Normalise all features to unit vectors (cosine metric)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    _, _, valid_counts, meta = nb._prepare_neighbors_with_meta(
        features=features,
        slide_ids=slide_ids,
        kmax=1,
        labels=labels,
        centers=centers,
    )

    assert np.all(valid_counts > 0), (
        f"{np.sum(valid_counts == 0)} samples have valid_counts == 0 after "
        f"_prepare_neighbors_with_meta with pruning"
    )


def test_filter_neighbors_excluding_same_slide_does_not_backfill() -> None:
    raw_neighbors = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 0, 1, 3, 4],
            [3, 0, 1, 2, 4],
            [4, 0, 1, 2, 3],
        ],
        dtype=int,
    )
    slide_ids = np.array(
        ["slide-a", "slide-a", "slide-b", "slide-c", "slide-d"], dtype=object
    )

    neigh, valid_counts = nb._filter_neighbors_excluding_same_slide(
        raw_neighbors=raw_neighbors,
        slide_ids=slide_ids,
        kmax=4,
    )

    assert valid_counts.tolist() == [3, 3, 4, 4, 4]
    assert neigh.shape == (5, 4)
    assert neigh[0].tolist() == [2, 3, 4, -1]
    assert neigh[1].tolist() == [2, 3, 4, -1]

    for sample_idx, row in enumerate(neigh):
        for pos in range(int(valid_counts[sample_idx])):
            assert slide_ids[int(row[pos])] != slide_ids[sample_idx]


def test_filter_query_neighbors_excluding_same_slide_does_not_backfill() -> None:
    raw_neighbors = np.array(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=int,
    )
    raw_distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.1, 0.2, 0.3],
        ],
        dtype=float,
    )
    query_indices = np.array([0, 1], dtype=int)
    slide_ids = np.array(["slide-a", "slide-a", "slide-b", "slide-c"], dtype=object)

    neigh, dist, valid_counts = nb._filter_query_neighbors_excluding_same_slide(
        raw_neighbors=raw_neighbors,
        raw_distances=raw_distances,
        query_indices=query_indices,
        slide_ids=slide_ids,
        kmax=3,
    )

    assert valid_counts.tolist() == [2, 2]
    assert neigh.shape == (2, 3)
    assert dist.shape == (2, 3)
    assert neigh[0].tolist() == [2, 3, -1]
    assert neigh[1].tolist() == [2, 3, -1]
    assert np.isfinite(dist[0, :2]).all()
    assert np.isfinite(dist[1, :2]).all()


def test_predict_labels_uses_per_sample_effective_k() -> None:
    labels = np.array([0, 1, 1, 0, 1], dtype=int)
    neigh = np.array(
        [
            [1, 2, -1],
            [0, 2, 3],
            [0, -1, -1],
            [1, 2, 4],
            [3, 2, -1],
        ],
        dtype=int,
    )
    valid_counts = np.array([2, 3, 1, 3, 2], dtype=int)

    pred, used_mask = nb._predict_labels_from_neighbors(
        labels=labels,
        neigh_idx=neigh,
        valid_counts=valid_counts,
        k=3,
    )

    assert used_mask.tolist() == [True, True, True, True, True]
    assert pred.tolist() == [1, 0, 0, 1, 0]


def test_warn_when_reduced_effective_k_exceeds_ten_percent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="croma")

    nb._warn_if_effective_k_reduced(
        valid_counts=np.array([5] * 9 + [4], dtype=int),
        target_k=5,
        context="toy-dataset",
        threshold=0.10,
    )
    assert not any("effective k < 5" in record.message for record in caplog.records)

    nb._warn_if_effective_k_reduced(
        valid_counts=np.array([5] * 8 + [4, 4], dtype=int),
        target_k=5,
        context="toy-dataset",
        threshold=0.10,
    )
    assert any("effective k < 5" in record.message for record in caplog.records)


def test_initial_neighbor_budget_uses_slide_aware_buffer() -> None:
    slide_ids_small = np.array([f"s{i // 10}" for i in range(100)], dtype=object)
    assert nb._max_tiles_per_slide(slide_ids_small) == 10
    assert (
        nb._initial_n_neighbors(kmax=21, slide_ids=slide_ids_small, n_samples=100) == 85
    )

    slide_ids_large = np.array(["a"] * 70 + [f"s{i}" for i in range(30)], dtype=object)
    assert nb._max_tiles_per_slide(slide_ids_large) == 70
    assert (
        nb._initial_n_neighbors(kmax=21, slide_ids=slide_ids_large, n_samples=100) == 91
    )


def test_prepare_neighbors_grows_when_coverage_below_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class FakeNN:
        def __init__(self, *, n_neighbors: int, metric: str) -> None:
            assert metric == "cosine"
            self.n_neighbors = int(n_neighbors)
            calls.append(self.n_neighbors)

        def fit(self, features: np.ndarray) -> "FakeNN":
            return self

        def kneighbors(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = int(features.shape[0])
            distances = np.zeros((n, self.n_neighbors), dtype=float)
            neighbors = np.zeros((n, self.n_neighbors), dtype=int)
            return distances, neighbors

    def fake_filter(
        raw_neighbors: np.ndarray,
        slide_ids: np.ndarray,
        kmax: int,
        raw_distances: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(raw_neighbors.shape[0])
        width = int(raw_neighbors.shape[1])
        valid_counts = (
            np.full((n,), kmax, dtype=int)
            if width >= 99
            else np.full((n,), kmax - 1, dtype=int)
        )
        neigh = np.full((n, kmax), -1, dtype=int)
        dist = np.full((n, kmax), np.inf, dtype=float)
        return neigh, dist, valid_counts

    monkeypatch.setattr(nb, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(nb, "_filter_neighbors_excluding_same_slide", fake_filter)

    features = np.zeros((100, 2), dtype=float)
    slide_ids = np.array([f"slide-{i}" for i in range(100)], dtype=object)
    _neigh, _dist, valid_counts, meta = nb._prepare_neighbors_with_meta(
        features, slide_ids, kmax=21
    )

    assert calls == [85, 99]
    assert meta.final_n_neighbors == 99
    assert meta.iterations == 2
    assert meta.hit_neighbor_cap
    assert nb._effective_k_coverage(valid_counts, 21) == pytest.approx(1.0)


def test_prepare_neighbors_stops_at_n_minus_one_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class FakeNN:
        def __init__(self, *, n_neighbors: int, metric: str) -> None:
            self.n_neighbors = int(n_neighbors)
            calls.append(self.n_neighbors)

        def fit(self, features: np.ndarray) -> "FakeNN":
            return self

        def kneighbors(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = int(features.shape[0])
            distances = np.zeros((n, self.n_neighbors), dtype=float)
            neighbors = np.zeros((n, self.n_neighbors), dtype=int)
            return distances, neighbors

    def fake_filter(
        raw_neighbors: np.ndarray,
        slide_ids: np.ndarray,
        kmax: int,
        raw_distances: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(raw_neighbors.shape[0])
        valid_counts = np.full((n,), kmax - 1, dtype=int)
        neigh = np.full((n, kmax), -1, dtype=int)
        dist = np.full((n, kmax), np.inf, dtype=float)
        return neigh, dist, valid_counts

    monkeypatch.setattr(nb, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(nb, "_filter_neighbors_excluding_same_slide", fake_filter)

    features = np.zeros((50, 2), dtype=float)
    slide_ids = np.array([f"slide-{i}" for i in range(50)], dtype=object)
    _neigh, _dist, valid_counts, meta = nb._prepare_neighbors_with_meta(
        features, slide_ids, kmax=21
    )

    assert calls == [49]
    assert meta.final_n_neighbors == 49
    assert meta.hit_neighbor_cap
    assert nb._effective_k_coverage(valid_counts, 21) == pytest.approx(0.0)


def test_warning_emitted_when_final_effective_k_still_reduced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="croma")

    nb._warn_if_effective_k_reduced(
        valid_counts=np.array([21] * 8 + [20, 20], dtype=int),
        target_k=21,
        context="dataset k-selection [fetch=99/99, coverage=80.0%, target=90.0%, capped]",
        threshold=0.10,
    )

    assert any("fetch=99/99" in record.message for record in caplog.records)
    assert any("coverage=80.0%" in record.message for record in caplog.records)


def test_no_growth_when_initial_budget_is_enough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    class FakeNN:
        def __init__(self, *, n_neighbors: int, metric: str) -> None:
            self.n_neighbors = int(n_neighbors)
            calls.append(self.n_neighbors)

        def fit(self, features: np.ndarray) -> "FakeNN":
            return self

        def kneighbors(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = int(features.shape[0])
            distances = np.zeros((n, self.n_neighbors), dtype=float)
            neighbors = np.zeros((n, self.n_neighbors), dtype=int)
            return distances, neighbors

    def fake_filter(
        raw_neighbors: np.ndarray,
        slide_ids: np.ndarray,
        kmax: int,
        raw_distances: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(raw_neighbors.shape[0])
        valid_counts = np.full((n,), kmax, dtype=int)
        valid_counts[-5:] = kmax - 1
        neigh = np.full((n, kmax), -1, dtype=int)
        dist = np.full((n, kmax), np.inf, dtype=float)
        return neigh, dist, valid_counts

    monkeypatch.setattr(nb, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(nb, "_filter_neighbors_excluding_same_slide", fake_filter)

    features = np.zeros((100, 2), dtype=float)
    slide_ids = np.array([f"slide-{i}" for i in range(100)], dtype=object)
    _neigh, _dist, valid_counts, meta = nb._prepare_neighbors_with_meta(
        features, slide_ids, kmax=21
    )

    assert calls == [85]
    assert meta.final_n_neighbors == 85
    assert not meta.hit_neighbor_cap
    assert nb._effective_k_coverage(valid_counts, 21) == pytest.approx(0.95)


def test_knn_balanced_accuracy_by_k_matches_optimal_selection() -> None:
    features = np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.00, 1.00],
            [0.01, 0.99],
            [0.02, 0.98],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    slide_ids = np.array([f"slide-{i}" for i in range(6)], dtype=object)
    k_values = [1, 2, 3]

    scores = nb._knn_balanced_accuracy_by_k(
        features=features,
        labels=labels,
        slide_ids=slide_ids,
        k_values=k_values,
        warn_context="toy",
    )

    assert set(scores) == set(k_values)
    assert all(0.0 <= float(v) <= 1.0 for v in scores.values())

    best_from_scores = nb._select_k_from_balanced_accuracy(
        k_values=k_values, scores=scores
    )
    best_from_existing = nb._optimal_k_by_knn_balanced_accuracy(
        features=features,
        labels=labels,
        slide_ids=slide_ids,
        k_values=k_values,
        warn_context="toy",
    )
    assert best_from_scores == best_from_existing


def test_select_k_from_balanced_accuracy_breaks_ties_by_input_order() -> None:
    scores = {3: 0.90, 1: 0.90, 5: 0.80}
    ordered_candidates = [1, 3, 5]

    selected = nb._select_k_from_balanced_accuracy(
        k_values=ordered_candidates,
        scores=scores,
    )

    assert selected == 1
