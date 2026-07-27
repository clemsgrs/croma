import numpy as np
import pandas as pd
import pytest

from croma import MaRI, RI


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

    mari_close_so, _sample_scores, _informative_mask, _undef_type = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=close_so,
        valid_counts=valid_counts,
        k=2,
        tau=0.2,
    )
    mari_close_os, _sample_scores, _informative_mask, _undef_type = MaRI._score_from_neighbors(
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

    small_tau, _sample_scores, _informative_mask, _undef_type = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
        tau=0.1,
    )
    large_tau, _sample_scores, _informative_mask, _undef_type = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
        tau=10.0,
    )

    assert small_tau > large_tau


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
            "label": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "scanner_vendor": [
                "V1",
                "V1",
                "V1",
                "V1",
                "V2",
                "V2",
                "V2",
                "V2",
                "V3",
                "V3",
                "V3",
                "V3",
            ],
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


# ---------------------------------------------------------------------------
# Per-sample median and tail metrics
# ---------------------------------------------------------------------------


def test_per_sample_median_value_equals_numpy_median() -> None:
    """result.median_value must equal np.median(result.sample_values)."""
    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()

    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=[2, 3, 4],
        evaluation_design="dataset_wide",
    )

    expected = float(np.median(result.sample_values))
    assert result.median_value == pytest.approx(expected)


def test_per_sample_q_alpha_from_tail_metrics() -> None:
    """result.q_alpha must match compute_tail_metrics(result.sample_values).q_alpha."""
    from croma.metrics.tail import compute_tail_metrics

    manifest = _make_manifest_four_confounders()
    features = _make_features_informative_neighbors()

    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=[2, 3, 4],
        evaluation_design="dataset_wide",
    )

    expected_q = compute_tail_metrics(result.sample_values, alpha=0.10).q_alpha
    assert result.q_alpha == pytest.approx(expected_q)


def test_per_sample_median_nan_when_no_defined_samples() -> None:
    """When no sample is defined, result.median_value must be NaN."""
    # All samples share the same label AND confounder → every neighbor is SS or OO
    # → no defined per-sample RI values
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(4)],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "A", "A"],
            "scanner_vendor": ["V1", "V1", "V1", "V1"],
            "slide_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
        }
    )
    features = np.eye(4, dtype=float)

    result = RI.compute(
        features,
        manifest,
        confounder_column="scanner_vendor",
        k_candidates=[1, 2],
        evaluation_design="dataset_wide",
    )

    assert len(result.sample_values) == 0
    assert np.isnan(result.median_value)
