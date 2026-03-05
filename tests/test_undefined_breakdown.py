
import numpy as np
import pytest

from mari import MaRI, RI


def test_ss_dominated_undefined_samples() -> None:
    """Samples whose k-neighborhood has only SS neighbors should be classified as SS-dominated."""
    # 4 samples: labels [0, 0, 0, 1], centers [0, 0, 0, 1]
    # Sample 0's neighbors at k=2: samples 1,2 -> same label, same center -> SS
    # SO+OS=0 -> undefined, SS-dominated
    labels = np.array([0, 0, 0, 1], dtype=int)
    centers = np.array([0, 0, 0, 1], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],  # neighbors of sample 0: both label=0, center=0 -> SS
            [0, 2],  # neighbors of sample 1: both label=0, center=0 -> SS
            [0, 1],  # neighbors of sample 2: both label=0, center=0 -> SS
            [0, 1],  # neighbors of sample 3: label=0!=1, center=0!=1 -> OO
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

    # Samples 0, 1, 2 are undefined (all SS neighbors) -> SS-dominated (type 1)
    # Sample 3 is undefined (all OO neighbors) -> OO-dominated (type 2)
    assert informative_mask.tolist() == [False, False, False, False]
    assert undefined_type[0] == 1  # SS-dominated
    assert undefined_type[1] == 1  # SS-dominated
    assert undefined_type[2] == 1  # SS-dominated
    assert undefined_type[3] == 2  # OO-dominated


def test_oo_dominated_undefined_samples() -> None:
    """Samples whose k-neighborhood has only OO neighbors should be classified as OO-dominated."""
    # 3 samples: labels [0, 1, 1], centers [0, 1, 1]
    # Sample 0's neighbors: samples 1,2 -> different label, different center -> OO
    labels = np.array([0, 1, 1], dtype=int)
    centers = np.array([0, 1, 1], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],  # OO, OO
            [2, 0],  # SS, OO
            [1, 0],  # SS, OO
        ],
        dtype=int,
    )
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

    # Sample 0: all OO -> OO-dominated
    assert informative_mask[0] is np.bool_(False)
    assert undefined_type[0] == 2  # OO-dominated


def test_mixed_undefined_samples() -> None:
    """Samples with equal SS and OO neighbors (no SO/OS) should be classified as mixed."""
    # Sample 0 has 1 SS neighbor and 1 OO neighbor, no SO/OS
    labels = np.array([0, 0, 1], dtype=int)
    centers = np.array([0, 0, 1], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],  # SS (label=0,center=0), OO (label=1,center=1)
            [0, 2],
            [0, 1],
        ],
        dtype=int,
    )
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

    # Sample 0: 1 SS + 1 OO -> mixed (type 3)
    assert informative_mask[0] is np.bool_(False)
    assert undefined_type[0] == 3  # mixed


def test_defined_samples_have_undefined_type_zero() -> None:
    """Samples with SO+OS > 0 (informative) should have undefined_type = 0."""
    labels = np.array([0, 0, 1], dtype=int)
    centers = np.array([0, 1, 0], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],  # SO (same label, diff center), OS (diff label, same center)
            [0, 2],
            [0, 1],
        ],
        dtype=int,
    )
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

    # Sample 0 has SO and OS neighbors -> defined
    assert informative_mask[0] is np.bool_(True)
    assert undefined_type[0] == 0


def test_undefined_frac_equals_sum_of_breakdown_fracs() -> None:
    """undefined_frac must equal the sum of the unconditional subtype fractions."""
    # Create a dataset where some samples are SS-dominated, some OO-dominated, some defined
    # 8 samples: 4 label=A, 4 label=B
    # Centers: A=[C1,C1,C2,C2], B=[C1,C1,C2,C2]
    # At small k, some samples will have only SS neighbors
    import pandas as pd

    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )
    features = np.array(
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

    result = RI.compute(
        features=features,
        manifest=manifest,
        mode="global",
        k_candidates=[1, 3],
        random_state=0,
    )

    assert result.undefined_frac == pytest.approx(
        result.ss_dominated_undefined_frac
        + result.oo_dominated_undefined_frac
        + result.mixed_undefined_frac
    )


def test_mari_undefined_breakdown_matches_ri() -> None:
    """MaRI should produce the same undefined_type classification as RI for the same neighborhood."""
    labels = np.array([0, 0, 0, 1], dtype=int)
    centers = np.array([0, 0, 0, 1], dtype=int)
    neigh_idx = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 1],
            [0, 1],
        ],
        dtype=int,
    )
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
    """RobustnessResult should expose the unconditional undefined subtype fractions."""
    import pandas as pd

    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )
    features = np.array(
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

    result = RI.compute(
        features=features,
        manifest=manifest,
        mode="global",
        k_candidates=[1, 3],
        random_state=0,
    )

    assert hasattr(result, "undefined_frac")
    assert hasattr(result, "ss_dominated_undefined_frac")
    assert hasattr(result, "oo_dominated_undefined_frac")
    assert hasattr(result, "mixed_undefined_frac")
    assert 0.0 <= result.ss_dominated_undefined_frac <= 1.0
    assert 0.0 <= result.oo_dominated_undefined_frac <= 1.0
    assert 0.0 <= result.mixed_undefined_frac <= 1.0
    assert 0.0 <= result.undefined_frac <= 1.0


def test_no_valid_neighbors_are_classified_as_mixed() -> None:
    """Rows with zero valid neighbors must still land in an undefined bucket."""
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


def test_paired_mode_breakdown_is_not_reported() -> None:
    """Paired-mode subtype fractions are intentionally undefined until aggregation semantics are specified."""
    import pandas as pd

    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )
    features = np.array(
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

    result = RI.compute(
        features=features,
        manifest=manifest,
        mode="paired",
        k_candidates=[1, 3],
        random_state=0,
    )

    assert np.isnan(result.ss_dominated_undefined_frac)
    assert np.isnan(result.oo_dominated_undefined_frac)
    assert np.isnan(result.mixed_undefined_frac)
