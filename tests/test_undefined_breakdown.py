import numpy as np
import pytest

from mari import MaRI, RI


def _single_subset_manifest():
    import pandas as pd

    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "subset": ["pair0"] * 8,
            "dataset": ["toy"] * 8,
        }
    )


def _single_subset_features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.92, 0.08, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.05, 0.95, 0.00, 0.00],
            [0.08, 0.92, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00],
        ],
        dtype=float,
    )


def _multi_subset_manifest():
    import pandas as pd

    base = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "image_path": [f"/tmp/{i}.png" for i in range(12)],
            "label": ["A"] * 6 + ["B"] * 6,
            "medical_center": [
                "C1",
                "C1",
                "C2",
                "C2",
                "C3",
                "C3",
                "C1",
                "C1",
                "C2",
                "C2",
                "C3",
                "C3",
            ],
            "slide_id": [f"slide-{i}" for i in range(12)],
            "dataset": ["toy3"] * 12,
        }
    )
    subset_map = {
        "C1": ("pair12", "pair13"),
        "C2": ("pair12", "pair23"),
        "C3": ("pair13", "pair23"),
    }
    rows: list[dict] = []
    for row in base.to_dict(orient="records"):
        for subset_id in subset_map[str(row["medical_center"])]:
            rows.append({**row, "subset": subset_id})
    return pd.DataFrame(rows)


def _multi_subset_features() -> np.ndarray:
    base = np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.96, 0.04],
            [0.95, 0.05],
            [0.00, 1.00],
            [0.01, 0.99],
            [0.02, 0.98],
            [0.03, 0.97],
            [0.04, 0.96],
            [0.05, 0.95],
        ],
        dtype=float,
    )
    return np.repeat(base, repeats=2, axis=0)


def _dataset_wide_undefined_manifest():
    import pandas as pd

    return pd.DataFrame(
        {
            "sample_id": [f"u{i}" for i in range(4)],
            "image_path": [f"/tmp/u{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2"],
            "slide_id": [f"u-slide-{i}" for i in range(4)],
            "dataset": ["toy_dw_undef"] * 4,
        }
    )


def _dataset_wide_undefined_features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.00, 1.00],
            [0.01, 0.99],
        ],
        dtype=float,
    )


def test_ss_dominated_undefined_samples() -> None:
    labels = np.array([0, 0, 0, 1], dtype=int)
    centers = np.array([0, 0, 0, 1], dtype=int)
    neigh_idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=int)
    neigh_dist = np.full((4, 2), 0.1, dtype=float)
    valid_counts = np.array([2, 2, 2, 2], dtype=int)

    _score, _sample_scores, informative_mask, undefined_type = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
    )

    assert informative_mask.tolist() == [False, False, False, False]
    assert undefined_type.tolist() == [1, 1, 1, 2]


def test_oo_dominated_undefined_samples() -> None:
    labels = np.array([0, 1, 1], dtype=int)
    centers = np.array([0, 1, 1], dtype=int)
    neigh_idx = np.array([[1, 2], [2, 0], [1, 0]], dtype=int)
    neigh_dist = np.full((3, 2), 0.1, dtype=float)
    valid_counts = np.array([2, 2, 2], dtype=int)

    _score, _sample_scores, informative_mask, undefined_type = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
    )

    assert informative_mask[0] is np.bool_(False)
    assert undefined_type[0] == 2


def test_mixed_undefined_samples() -> None:
    labels = np.array([0, 0, 1], dtype=int)
    centers = np.array([0, 0, 1], dtype=int)
    neigh_idx = np.array([[1, 2], [0, 2], [0, 1]], dtype=int)
    neigh_dist = np.full((3, 2), 0.1, dtype=float)
    valid_counts = np.array([2, 2, 2], dtype=int)

    _score, _sample_scores, informative_mask, undefined_type = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
    )

    assert informative_mask[0] is np.bool_(False)
    assert undefined_type[0] == 3


def test_defined_samples_have_undefined_type_zero() -> None:
    labels = np.array([0, 0, 1], dtype=int)
    centers = np.array([0, 1, 0], dtype=int)
    neigh_idx = np.array([[1, 2], [0, 2], [0, 1]], dtype=int)
    neigh_dist = np.full((3, 2), 0.1, dtype=float)
    valid_counts = np.array([2, 2, 2], dtype=int)

    _score, _sample_scores, informative_mask, undefined_type = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
    )

    assert informative_mask[0] is np.bool_(True)
    assert undefined_type[0] == 0


def test_occurrence_weighted_breakdown_fracs_are_reported() -> None:
    manifest = _single_subset_manifest()
    features = _single_subset_features()

    result = RI.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1, 3],
        random_state=0,
    )

    assert 0.0 <= result.undefined_frac <= 1.0
    assert 0.0 <= result.ss_dominated_undefined_frac <= 1.0
    assert 0.0 <= result.oo_dominated_undefined_frac <= 1.0
    assert 0.0 <= result.mixed_undefined_frac <= 1.0
    assert result.undefined_frac == np.sum(
        [
            result.ss_dominated_undefined_frac,
            result.oo_dominated_undefined_frac,
            result.mixed_undefined_frac,
        ]
    )


def test_mari_undefined_breakdown_matches_ri() -> None:
    labels = np.array([0, 0, 0, 1], dtype=int)
    centers = np.array([0, 0, 0, 1], dtype=int)
    neigh_idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=int)
    neigh_dist = np.full((4, 2), 0.1, dtype=float)
    valid_counts = np.array([2, 2, 2, 2], dtype=int)

    _score_ri, _, _, undef_ri = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
    )
    _score_mari, _, _, undef_mari = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
        tau=0.2,
    )

    np.testing.assert_array_equal(undef_ri, undef_mari)


def test_robustness_result_has_breakdown_fields() -> None:
    manifest = _single_subset_manifest()
    features = _single_subset_features()

    result = RI.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1, 3],
        random_state=0,
    )

    assert hasattr(result, "undefined_frac")
    assert hasattr(result, "ss_dominated_undefined_frac")
    assert hasattr(result, "oo_dominated_undefined_frac")
    assert hasattr(result, "mixed_undefined_frac")
    assert hasattr(result, "occurrence_defined_mask")
    assert hasattr(result, "occurrence_subsets")
    assert hasattr(result, "occurrence_source_indices")
    assert 0.0 <= result.undefined_frac <= 1.0
    assert 0.0 <= result.ss_dominated_undefined_frac <= 1.0
    assert 0.0 <= result.oo_dominated_undefined_frac <= 1.0
    assert 0.0 <= result.mixed_undefined_frac <= 1.0


def test_occurrence_weighted_undefined_fraction_counts_repeated_memberships() -> None:
    manifest = _multi_subset_manifest()
    features = _multi_subset_features()

    result = RI.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1, 3],
        random_state=0,
    )

    occurrence_total = int(result.sample_values_aligned.shape[0])
    undefined_total = int(np.count_nonzero(~result.occurrence_defined_mask))
    assert occurrence_total == 24
    assert result.undefined_frac == np.divide(undefined_total, occurrence_total)
    assert result.undefined_frac == np.mean(~result.occurrence_defined_mask)
    assert result.ss_dominated_undefined_frac == np.mean(result.sample_undefined_types == 1)
    assert result.oo_dominated_undefined_frac == np.mean(result.sample_undefined_types == 2)
    assert result.mixed_undefined_frac == np.mean(result.sample_undefined_types == 3)


def test_no_valid_neighbors_are_classified_as_mixed() -> None:
    labels = np.array([0, 1], dtype=int)
    centers = np.array([0, 1], dtype=int)
    neigh_idx = np.array([[-1], [-1]], dtype=int)
    neigh_dist = np.array([[np.inf], [np.inf]], dtype=float)
    valid_counts = np.array([0, 0], dtype=int)

    _score, _sample_scores, informative_mask, undefined_type = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=1,
    )

    assert informative_mask.tolist() == [False, False]
    assert undefined_type.tolist() == [3, 3]


def test_dataset_wide_undefined_fraction_uses_sample_denominator() -> None:
    manifest = _dataset_wide_undefined_manifest()
    features = _dataset_wide_undefined_features()

    result = RI.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1],
        evaluation_design="dataset_wide",
    )

    assert result.evaluation_design == "dataset_wide"
    assert result.evaluation_unit == "sample"
    assert result.sample_values.shape == (0,)
    assert result.sample_values_aligned.shape == (len(manifest),)
    assert result.undefined_frac == pytest.approx(1.0)
    assert result.ss_dominated_undefined_frac == pytest.approx(1.0)
    assert result.oo_dominated_undefined_frac == pytest.approx(0.0)
    assert result.mixed_undefined_frac == pytest.approx(0.0)
    assert np.array_equal(result.sample_undefined_types, np.ones(len(manifest), dtype=int))
