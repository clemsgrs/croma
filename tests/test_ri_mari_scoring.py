import numpy as np
import pandas as pd
import pytest

from croma import MaRI, RI
from croma.metrics.neighbors import _filter_neighbors_informative_only


def test_ri_pooled_fallback_returns_half_when_no_so_or_os() -> None:
    labels = np.array([0, 1], dtype=int)
    centers = np.array([0, 1], dtype=int)
    neigh_idx = np.array([[1], [0]], dtype=int)
    neigh_dist = np.array([[0.1], [0.1]], dtype=float)
    valid_counts = np.array([1, 1], dtype=int)

    score, sample_scores, informative_mask, _undefined_type = RI._score_from_neighbors(
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

    mari_close_so, _sample_scores, _informative_mask, _undef_type = (
        MaRI._score_from_neighbors(
            labels=labels,
            centers=centers,
            neigh_idx=neigh_idx,
            neigh_dist=close_so,
            valid_counts=valid_counts,
            k=2,
            tau=0.2,
        )
    )
    mari_close_os, _sample_scores, _informative_mask, _undef_type = (
        MaRI._score_from_neighbors(
            labels=labels,
            centers=centers,
            neigh_idx=neigh_idx,
            neigh_dist=close_os,
            valid_counts=valid_counts,
            k=2,
            tau=0.2,
        )
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

    small_tau, _sample_scores, _informative_mask, _undef_type = (
        MaRI._score_from_neighbors(
            labels=labels,
            centers=centers,
            neigh_idx=neigh_idx,
            neigh_dist=neigh_dist,
            valid_counts=valid_counts,
            k=2,
            tau=0.1,
        )
    )
    large_tau, _sample_scores, _informative_mask, _undef_type = (
        MaRI._score_from_neighbors(
            labels=labels,
            centers=centers,
            neigh_idx=neigh_idx,
            neigh_dist=neigh_dist,
            valid_counts=valid_counts,
            k=2,
            tau=10.0,
        )
    )

    assert small_tau > large_tau


# ---------------------------------------------------------------------------
# Tests for prune_ss_oo
# ---------------------------------------------------------------------------

def _make_manifest_with_ss_oo_dominated_samples() -> pd.DataFrame:
    """
    8 samples: 4 label=A / 4 label=B, 2 confounders (V1, V2), each on its own slide.

    Samples 0-3: label=A, confounder=V1 → purely SS neighbours among themselves
    Samples 4-7: label=B, confounder=V2 → purely OO neighbours relative to A/V1 group

    With k=2 and no pruning:
      - Sample 0's 2 nearest neighbours are samples 1,2 (all A/V1 → SS) → undefined
    With k=2 and prune_ss_oo=True:
      - SS/OO slots are skipped; sample 0 looks further until it finds SO/OS neighbours
    """
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "scanner_vendor": ["V1", "V1", "V1", "V1", "V2", "V2", "V2", "V2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )


def _make_features_two_tight_clusters() -> np.ndarray:
    """
    Cluster 1 (samples 0-3, A/V1): tightly packed near [1, 0]
    Cluster 2 (samples 4-7, B/V2): tightly packed near [0, 1]

    With k=2 the 2 NNs of any A/V1 sample are other A/V1 samples (SS).
    With k=3+ the NNs start including B/V2 samples (OO) and eventually SO/OS.

    Note: SO = same-label, other-center. With only V1 and V2 there are no SO/OS
    cross-confounder same-label pairs in these two tight clusters, so we need a
    richer manifest. Use dataset_wide with 4 confounders instead.
    """
    rng = np.random.default_rng(42)
    cluster1 = np.column_stack([np.ones(4) + rng.normal(0, 0.01, 4),
                                 rng.normal(0, 0.01, 4)])
    cluster2 = np.column_stack([rng.normal(0, 0.01, 4),
                                 np.ones(4) + rng.normal(0, 0.01, 4)])
    return np.vstack([cluster1, cluster2]).astype(float)


def _make_manifest_four_confounders() -> pd.DataFrame:
    """
    12 samples with 4 confounders (V1-V4) and 2 labels (A/B).
    Arranged so that near neighbours within each label group are SS (same confounder)
    but further neighbours are SO (same label, other confounder).

    layout: 3 blocks, each block = 2 confounders × 2 labels = 4 samples
    """
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "image_path": [f"/tmp/{i}.png" for i in range(12)],
            "label":      ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "scanner_vendor": ["V1", "V1", "V1", "V1", "V2", "V2", "V2", "V2",
                               "V3", "V3", "V3", "V3"],
            "slide_id": [f"slide-{i}" for i in range(12)],
            "dataset": ["toy"] * 12,
        }
    )


def _make_features_informative_neighbors() -> np.ndarray:
    """
    Features for _make_manifest_four_confounders:
    All A samples cluster near [1,0,0], all B samples near [0,1,0].
    Within each label, V1 < V2 < V3 in the 3rd dimension so that nearest neighbours
    within a label group are same-confounder (SS) for small k.
    With prune_ss_oo the SS neighbours are skipped and SO neighbours are found.
    """
    # A samples (indices 0,2,4,6,8,10): near [1,0,z] with z = confounder offset
    # B samples (indices 1,3,5,7,9,11): near [0,1,z]
    conf_offsets = {"V1": 0.0, "V2": 0.05, "V3": 0.10}
    manifest = _make_manifest_four_confounders()
    rows = []
    for i, row in manifest.iterrows():
        z = conf_offsets[row["scanner_vendor"]]
        if row["label"] == "A":
            rows.append([1.0, 0.0, z])
        else:
            rows.append([0.0, 1.0, z])
    return np.array(rows, dtype=float)


def test_filter_neighbors_informative_only_excludes_ss_oo() -> None:
    """
    Given 5 candidate neighbours for sample 0:
      idx 1: same label, same center → SS → skip
      idx 2: same label, other center → SO → keep
      idx 3: other label, same center → OS → keep
      idx 4: other label, other center → OO → skip
      idx 5: same label, other center → SO → keep
    Expect out_idx[0] = [2, 3, 5], valid_counts[0] = 3
    """
    n = 6
    slide_ids = np.array([f"sl{i}" for i in range(n)])
    labels   = np.array([0, 0, 0, 1, 1, 0], dtype=int)   # 0=A, 1=B
    centers  = np.array([0, 0, 1, 0, 1, 1], dtype=int)   # 0=V1, 1=V2

    # shape (n_samples, n_candidates) — only sample 0 row matters for this assertion
    raw_neighbors = np.array(
        [[1, 2, 3, 4, 5, -1],
         [0, 2, 3, 4, 5, -1],
         [0, 1, 3, 4, 5, -1],
         [0, 1, 2, 4, 5, -1],
         [0, 1, 2, 3, 5, -1],
         [0, 1, 2, 3, 4, -1]],
        dtype=int,
    )
    raw_distances = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
         [0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
         [0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
         [0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
         [0.1, 0.2, 0.3, 0.4, 0.5, np.inf],
         [0.1, 0.2, 0.3, 0.4, 0.5, np.inf]],
        dtype=float,
    )

    out_idx, out_dist, valid_counts = _filter_neighbors_informative_only(
        raw_neighbors=raw_neighbors,
        slide_ids=slide_ids,
        labels=labels,
        centers=centers,
        kmax=3,
        raw_distances=raw_distances,
    )

    assert valid_counts[0] == 3
    assert out_idx[0, 0] == 2   # SO
    assert out_idx[0, 1] == 3   # OS
    assert out_idx[0, 2] == 5   # SO
    assert out_dist[0, 0] == pytest.approx(0.2)
    assert out_dist[0, 1] == pytest.approx(0.3)
    assert out_dist[0, 2] == pytest.approx(0.5)


def test_prune_ss_oo_valid_counts_reflect_so_os_only() -> None:
    """
    With prune_ss_oo=True, _prepare_neighbors must return valid_counts that count
    only SO/OS neighbours — so valid_counts[i] <= k and all filled slots are informative.
    """
    from croma.metrics.neighbors import _prepare_neighbors

    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()
    labels  = (manifest["label"] == "B").astype(int).to_numpy()
    centers = manifest["scanner_vendor"].map({"V1": 0, "V2": 1, "V3": 2}).to_numpy()
    slide_ids = manifest["slide_id"].to_numpy()

    kmax = 3
    neigh_idx, neigh_dist, valid_counts = _prepare_neighbors(
        features=features,
        slide_ids=slide_ids,
        kmax=kmax,
        labels=labels,
        centers=centers,
    )

    # Every filled slot must be SO or OS (not SS or OO)
    for i in range(len(features)):
        for pos in range(int(valid_counts[i])):
            j = int(neigh_idx[i, pos])
            assert j >= 0
            same_label  = labels[j] == labels[i]
            same_center = centers[j] == centers[i]
            is_so = same_label and not same_center
            is_os = not same_label and same_center
            assert is_so or is_os, (
                f"sample {i} slot {pos}: neighbour {j} is neither SO nor OS "
                f"(same_label={same_label}, same_center={same_center})"
            )


def test_prune_ss_oo_eliminates_undefined_dataset_wide() -> None:
    """
    With prune_ss_oo=True and dataset_wide evaluation, undefined_frac should be 0
    when each sample has at least one SO/OS neighbour available.
    """
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()

    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=[2, 3],
        evaluation_design="dataset_wide",
        prune_ss_oo=True,
    )

    assert result.undefined_frac == 0.0, (
        f"Expected no undefined samples with prune_ss_oo=True, "
        f"got undefined_frac={result.undefined_frac}"
    )


def test_prune_ss_oo_false_default_unchanged() -> None:
    """prune_ss_oo=False (default) must give the same result as not passing it."""
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()

    kwargs = dict(
        confounder_column="scanner_vendor",
        k_candidates=[2, 3],
        evaluation_design="dataset_wide",
    )
    result_default = RI.compute(features, manifest, **kwargs)
    result_explicit_false = RI.compute(features, manifest, **kwargs, prune_ss_oo=False)

    assert result_default.value == pytest.approx(result_explicit_false.value)
    assert result_default.undefined_frac == pytest.approx(result_explicit_false.undefined_frac)


def test_prune_ss_oo_uses_k_max_not_knn_selected_k() -> None:
    """With prune_ss_oo=True, the selected k must always equal max(k_candidates),
    regardless of which k would have been chosen by kNN biological accuracy."""
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()

    k_candidates = [2, 3, 4]
    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=k_candidates,
        evaluation_design="dataset_wide",
        prune_ss_oo=True,
    )

    assert result.k == max(k_candidates), (
        f"Expected k={max(k_candidates)} (k_max) with prune_ss_oo=True, got k={result.k}"
    )


# ---------------------------------------------------------------------------
# Tests for summarize_by_mean
# ---------------------------------------------------------------------------


def test_summarize_by_mean_value_equals_curve_mean() -> None:
    """result.value must equal the arithmetic mean of the RI curve values."""
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()
    k_candidates = [2, 3, 4]

    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=k_candidates,
        evaluation_design="dataset_wide",
        summarize_by_mean=True,
    )
    curve = RI.compute_curve(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_values=k_candidates,
        evaluation_design="dataset_wide",
    )

    expected = float(np.mean(list(curve.values())))
    assert result.value == pytest.approx(expected)


def test_summarize_by_mean_std_equals_curve_std() -> None:
    """result.std must equal the std of the RI curve values."""
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()
    k_candidates = [2, 3, 4]

    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=k_candidates,
        evaluation_design="dataset_wide",
        summarize_by_mean=True,
    )
    curve = RI.compute_curve(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_values=k_candidates,
        evaluation_design="dataset_wide",
    )

    expected_std = float(np.std(list(curve.values())))
    assert result.std == pytest.approx(expected_std)


def test_summarize_by_mean_k_equals_kmax() -> None:
    """result.k must equal max(k_candidates) when summarize_by_mean=True."""
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()
    k_candidates = [2, 3, 4]

    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=k_candidates,
        evaluation_design="dataset_wide",
        summarize_by_mean=True,
    )

    assert result.k == max(k_candidates)


def test_summarize_by_mean_false_default_unchanged() -> None:
    """summarize_by_mean=False (default) must give the same result as not passing it."""
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()

    kwargs = dict(
        confounder_column="scanner_vendor",
        k_candidates=[2, 3, 4],
        evaluation_design="dataset_wide",
    )
    result_default = RI.compute(features, manifest, **kwargs)
    result_explicit_false = RI.compute(features, manifest, **kwargs, summarize_by_mean=False)

    assert result_default.value == pytest.approx(result_explicit_false.value)
    assert result_default.std == pytest.approx(result_explicit_false.std)
    assert result_default.k == result_explicit_false.k


def test_compute_mean_from_curve_returns_arithmetic_mean() -> None:
    """_compute_mean_from_curve returns the arithmetic mean of curve values.

    Mocked curve: {1: 0.4, 2: 0.6, 3: 0.8, 4: 0.6}
      mean = (0.4 + 0.6 + 0.8 + 0.6) / 4 = 0.6
    """
    from croma.metrics.base import BaseRobustnessIndex

    curve = {1: 0.4, 2: 0.6, 3: 0.8, 4: 0.6}
    mean_val, _ = BaseRobustnessIndex._compute_mean_from_curve(curve)

    assert mean_val == pytest.approx((0.4 + 0.6 + 0.8 + 0.6) / 4)
