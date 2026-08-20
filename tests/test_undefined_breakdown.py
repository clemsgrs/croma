import inspect

import numpy as np
import pandas as pd
import pytest

from croma import MaRI, RI
from croma.types import RobustnessResult
from support_schema import RETIRED_AGGREGATE_FIELD


def _paired_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "scanner_vendor": ["V1", "V1", "V2", "V2", "V1", "V1", "V2", "V2"],
            "group_id": [f"slide-{i}" for i in range(8)],
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


def _all_rows_undefined_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a1", "a2", "b1", "b2"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V1", "V2", "V2"],
            "group_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
        }
    )


def _all_rows_undefined_features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.00, 1.00],
            [0.01, 0.99],
        ],
        dtype=float,
    )


def _partial_support_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a-v1", "a-v2", "b-v1", "b-v2"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2"],
            "group_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
        }
    )


def _partial_support_features() -> np.ndarray:
    # Nearest-neighbour types by row are OO, OS, OS, OO, respectively.
    angles = np.array([0.0, 0.2, -0.2, 0.05], dtype=float)
    return np.column_stack((np.cos(angles), np.sin(angles)))


@pytest.mark.parametrize("metric", [RI, MaRI])
def test_sample_support_is_the_defined_sample_fraction(metric) -> None:
    result = metric.compute(
        features=_partial_support_features(),
        manifest=_partial_support_manifest(),
        confounder_column="scanner_vendor",
        evaluation_design="all",
        k_candidates=[1],
    )

    assert result.occurrence_defined_mask.tolist() == [False, True, True, False]
    assert result.support == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [RI, MaRI])
def test_result_exposes_support_without_retired_aggregate(metric) -> None:
    result = metric.compute(
        features=_partial_support_features(),
        manifest=_partial_support_manifest(),
        confounder_column="scanner_vendor",
        evaluation_design="all",
        k_candidates=[1],
    )

    assert result.support == pytest.approx(0.5)
    assert not hasattr(result, RETIRED_AGGREGATE_FIELD)


def test_result_construction_requires_support() -> None:
    assert (
        inspect.signature(RobustnessResult).parameters["support"].default is inspect.Parameter.empty
    )


def test_result_construction_rejects_legacy_positional_coverage() -> None:
    positional = (
        "toy",
        1,
        0.5,
        0.0,
        1,
        np.asarray([0.5]),
        np.asarray([0.5]),
        np.asarray([0.5]),
        np.asarray([True]),
        np.asarray([0]),
        np.asarray(["dataset"]),
        np.asarray([0]),
    )

    with pytest.raises(TypeError):
        RobustnessResult(*positional, 0.5)

    result = RobustnessResult(*positional, support=0.5)
    assert result.support == pytest.approx(0.5)


@pytest.mark.parametrize("metric", [RI, MaRI])
def test_occurrence_support_is_the_defined_occurrence_fraction(metric) -> None:
    result = metric.compute(
        features=_partial_support_features(),
        manifest=_partial_support_manifest().assign(subset="pair"),
        confounder_column="scanner_vendor",
        evaluation_design="paired_2x2",
        k_candidates=[1],
    )

    assert result.occurrence_defined_mask.tolist() == [False, True, True, False]
    assert result.support == pytest.approx(0.5)


def test_partial_sample_support_warning_is_reported_positively() -> None:
    with pytest.warns(RuntimeWarning, match=r"RI/MaRI support is 50\.0% across samples\."):
        RI.compute(
            features=_partial_support_features(),
            manifest=_partial_support_manifest(),
            confounder_column="scanner_vendor",
            evaluation_design="all",
            k_candidates=[1],
        )


def test_partial_occurrence_support_warning_names_subset_occurrences() -> None:
    with pytest.warns(
        RuntimeWarning,
        match=r"RI/MaRI support is 50\.0% across subset occurrences\.",
    ):
        RI.compute(
            features=_partial_support_features(),
            manifest=_partial_support_manifest().assign(subset="pair"),
            confounder_column="scanner_vendor",
            evaluation_design="paired_2x2",
            k_candidates=[1],
        )


def test_low_support_warning_preserves_the_dominant_undefined_cause() -> None:
    with pytest.warns(RuntimeWarning) as caught:
        RI.compute(
            features=_partial_support_features(),
            manifest=_partial_support_manifest(),
            confounder_column="scanner_vendor",
            evaluation_design="all",
            k_candidates=[1],
        )

    assert any(
        str(warning.message)
        == "toy: undefined RI/MaRI samples are predominantly OO-dominated (50.0%)."
        for warning in caught
    )


@pytest.mark.parametrize("metric", [RI, MaRI])
@pytest.mark.parametrize("evaluation_design", ["all", "paired_2x2"])
def test_undefined_causes_keep_the_all_unit_denominator(metric, evaluation_design: str) -> None:
    manifest = _partial_support_manifest()
    if evaluation_design == "paired_2x2":
        manifest = manifest.assign(subset="pair")

    result = metric.compute(
        features=_partial_support_features(),
        manifest=manifest,
        confounder_column="scanner_vendor",
        evaluation_design=evaluation_design,
        k_candidates=[1],
    )

    cause_fractions = (
        result.ss_dominated_undefined_frac,
        result.oo_dominated_undefined_frac,
        result.mixed_undefined_frac,
    )
    assert cause_fractions == pytest.approx((0.0, 0.5, 0.0))
    assert result.support + sum(cause_fractions) == pytest.approx(1.0)


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


def test_all_rows_support_uses_sample_denominator() -> None:
    result = RI.compute(
        features=_all_rows_undefined_features(),
        manifest=_all_rows_undefined_manifest(),
        confounder_column="scanner_vendor",
        evaluation_design="all",
        k_candidates=[1],
    )

    assert result.evaluation_design == "all"
    assert result.evaluation_unit == "sample"
    assert result.support == pytest.approx(0.0)
    assert (
        result.support
        + result.ss_dominated_undefined_frac
        + result.oo_dominated_undefined_frac
        + result.mixed_undefined_frac
    ) == pytest.approx(1.0)


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


def test_paired_breakdown_sums_to_the_unsupported_occurrence_fraction() -> None:
    result = RI.compute(
        features=_paired_features(),
        manifest=_paired_manifest(),
        confounder_column="scanner_vendor",
        evaluation_design="paired_2x2",
        k_candidates=[1, 3],
    )

    assert result.evaluation_design == "paired_2x2"
    assert result.evaluation_unit == "occurrence"
    assert (
        result.support
        + result.ss_dominated_undefined_frac
        + result.oo_dominated_undefined_frac
        + result.mixed_undefined_frac
    ) == pytest.approx(1.0)
