import numpy as np
import pandas as pd
import pytest

from mari import MaRI, RI


def _paired_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
            "subset": ["pair0"] * 8,
        }
    )


def _paired_features() -> np.ndarray:
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


def _dataset_wide_undefined_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a1", "a2", "b1", "b2"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
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


def test_undefined_samples_are_bucketed_as_ss_oo_or_mixed() -> None:
    labels = np.array([0, 0, 1, 1], dtype=int)
    centers = np.array([0, 0, 1, 0], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],  # SS + OO -> mixed
            [0, 3],  # SS + OS -> defined
            [0, 3],  # OO + SO -> defined
            [2, 0],  # SS + OO -> mixed
        ],
        dtype=int,
    )
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

    assert informative_mask.tolist() == [False, True, True, True]
    assert undefined_type[0] == 3

    ss_only_score = RI._score_from_neighbors(
        labels=np.array([0, 0, 0], dtype=int),
        centers=np.array([0, 0, 0], dtype=int),
        neigh_idx=np.array([[1, 2], [0, 2], [0, 1]], dtype=int),
        neigh_dist=np.full((3, 2), 0.1, dtype=float),
        valid_counts=np.array([2, 2, 2], dtype=int),
        k=2,
    )[3]
    oo_only_score = RI._score_from_neighbors(
        labels=np.array([0, 1, 1], dtype=int),
        centers=np.array([0, 1, 1], dtype=int),
        neigh_idx=np.array([[1, 2], [2, 0], [1, 0]], dtype=int),
        neigh_dist=np.full((3, 2), 0.1, dtype=float),
        valid_counts=np.array([2, 2, 2], dtype=int),
        k=2,
    )[3]

    assert ss_only_score[0] == 1
    assert oo_only_score[0] == 2


def test_no_valid_neighbors_are_classified_as_mixed() -> None:
    _score, _sample_scores, informative_mask, undefined_type = RI._score_from_neighbors(
        labels=np.array([0, 1], dtype=int),
        centers=np.array([0, 1], dtype=int),
        neigh_idx=np.array([[-1], [-1]], dtype=int),
        neigh_dist=np.array([[np.inf], [np.inf]], dtype=float),
        valid_counts=np.array([0, 0], dtype=int),
        k=1,
    )

    assert informative_mask.tolist() == [False, False]
    assert undefined_type.tolist() == [3, 3]


def test_dataset_wide_undefined_fraction_uses_sample_denominator() -> None:
    result = RI.compute(
        features=_dataset_wide_undefined_features(),
        manifest=_dataset_wide_undefined_manifest(),
        evaluation_design="dataset_wide",
        k_candidates=[1],
    )

    assert result.evaluation_design == "dataset_wide"
    assert result.evaluation_unit == "sample"
    assert result.undefined_frac == pytest.approx(1.0)
    assert result.undefined_frac == pytest.approx(
        result.ss_dominated_undefined_frac
        + result.oo_dominated_undefined_frac
        + result.mixed_undefined_frac
    )


def test_mari_uses_the_same_undefined_type_classification_as_ri() -> None:
    labels = np.array([0, 0, 0, 1], dtype=int)
    centers = np.array([0, 0, 0, 1], dtype=int)
    neigh_idx = np.array([[1, 2], [0, 2], [0, 1], [0, 1]], dtype=int)
    neigh_dist = np.full((4, 2), 0.1, dtype=float)
    valid_counts = np.array([2, 2, 2, 2], dtype=int)

    undef_ri = RI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
    )[3]
    undef_mari = MaRI._score_from_neighbors(
        labels=labels,
        centers=centers,
        neigh_idx=neigh_idx,
        neigh_dist=neigh_dist,
        valid_counts=valid_counts,
        k=2,
        tau=0.2,
    )[3]

    np.testing.assert_array_equal(undef_ri, undef_mari)


def test_paired_breakdown_sums_to_occurrence_weighted_undefined_fraction() -> None:
    result = RI.compute(
        features=_paired_features(),
        manifest=_paired_manifest(),
        evaluation_design="paired_2x2",
        k_candidates=[1, 3],
        random_state=0,
    )

    assert result.evaluation_design == "paired_2x2"
    assert result.evaluation_unit == "occurrence"
    assert result.undefined_frac == pytest.approx(
        result.ss_dominated_undefined_frac
        + result.oo_dominated_undefined_frac
        + result.mixed_undefined_frac
    )
