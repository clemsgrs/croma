import numpy as np
import pandas as pd
import pytest

from croma import MaRI, RI


def _all_rows_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a_c1", "a_c2", "b_c1", "b_c2"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2"],
            "group_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
        }
    )


def _all_rows_features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.00, 1.00],
            [0.01, 0.99],
        ],
        dtype=float,
    )


def _paired_manifest_single_subset() -> pd.DataFrame:
    manifest = pd.DataFrame(
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
    return manifest


def _paired_manifest_repeated_subset_memberships() -> pd.DataFrame:
    base = _paired_manifest_single_subset().copy()
    other = base.copy()
    other["subset"] = "pair1"
    return pd.concat([base, other], ignore_index=True)


def _paired_features_single_subset() -> np.ndarray:
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


def _paired_features_repeated_subset_memberships() -> np.ndarray:
    base = _paired_features_single_subset()
    return np.vstack([base, base])


def _metric_kwargs(metric_cls) -> dict:
    kwargs = {"confounder_column": "scanner_vendor"}
    if metric_cls is MaRI:
        kwargs["tau"] = 0.2
    return kwargs


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_compute_returns_occurrence_aligned_outputs(metric_cls) -> None:
    manifest = _paired_manifest_single_subset()
    result = metric_cls.compute(
        features=_paired_features_single_subset(),
        manifest=manifest,
        evaluation_design="paired_2x2",
        k_candidates=[1, 3],
        **_metric_kwargs(metric_cls),
    )

    assert result.dataset == "toy"
    assert result.evaluation_design == "paired_2x2"
    assert result.evaluation_unit == "occurrence"
    assert result.n_pairs == 1
    assert result.pair_values.shape == (1,)
    assert result.sample_values_aligned.shape == (len(manifest),)
    assert result.occurrence_defined_mask.shape == (len(manifest),)
    assert result.sample_undefined_types.shape == (len(manifest),)
    assert result.occurrence_subsets.tolist() == ["pair0"] * len(manifest)
    assert result.occurrence_source_indices.tolist() == list(range(len(manifest)))


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_keeps_repeated_subset_memberships_as_distinct_occurrences(
    metric_cls,
) -> None:
    manifest = _paired_manifest_repeated_subset_memberships()
    result = metric_cls.compute(
        features=_paired_features_repeated_subset_memberships(),
        manifest=manifest,
        evaluation_design="paired_2x2",
        k_candidates=[1],
        **_metric_kwargs(metric_cls),
    )

    assert result.evaluation_unit == "occurrence"
    assert result.n_pairs == 2
    assert result.sample_values_aligned.shape == (len(manifest),)
    assert sorted(set(result.occurrence_subsets.tolist())) == ["pair0", "pair1"]
    assert result.occurrence_source_indices.tolist() == list(range(len(manifest)))


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_all_rows_compute_returns_expected_exact_value(metric_cls) -> None:
    result = metric_cls.compute(
        features=_all_rows_features(),
        manifest=_all_rows_manifest(),
        evaluation_design="all",
        k_candidates=[1],
        **_metric_kwargs(metric_cls),
    )

    assert result.k == 1
    assert result.value == pytest.approx(1.0)
    assert result.std == pytest.approx(0.0)
    assert result.n_pairs == 1
    assert result.evaluation_design == "all"
    assert result.evaluation_unit == "sample"
    assert result.sample_values_aligned.shape == (4,)
    assert result.occurrence_defined_mask.tolist() == [True, True, True, True]
    assert result.sample_undefined_types.tolist() == [0, 0, 0, 0]
    assert result.occurrence_subsets.tolist() == ["dataset"] * 4
    assert result.occurrence_source_indices.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("evaluation_design", ["paired_2x2", "all"])
def test_compute_curve_matches_single_k_compute(metric_cls, evaluation_design: str) -> None:
    if evaluation_design == "paired_2x2":
        manifest = _paired_manifest_single_subset()
        features = _paired_features_single_subset()
    else:
        manifest = _all_rows_manifest()
        features = _all_rows_features()

    curve_kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    curve = metric_cls.compute_curve(
        features=features,
        manifest=manifest,
        confounder_column="scanner_vendor",
        evaluation_design=evaluation_design,
        k_values=[1],
        **curve_kwargs,
    )

    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        evaluation_design=evaluation_design,
        k_candidates=[1],
        **_metric_kwargs(metric_cls),
    )

    assert curve == {1: pytest.approx(float(result.value))}


def test_compute_rejects_unknown_evaluation_design() -> None:
    with pytest.raises(ValueError, match="evaluation_design"):
        RI.compute(
            features=_all_rows_features(),
            manifest=_all_rows_manifest(),
            confounder_column="scanner_vendor",
            evaluation_design="auto",
            k_candidates=[1],
        )


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_requires_subset_metadata(metric_cls) -> None:
    with pytest.raises(ValueError, match="subset"):
        metric_cls.compute(
            features=_all_rows_features(),
            manifest=_all_rows_manifest(),
            evaluation_design="paired_2x2",
            k_candidates=[1],
            **_metric_kwargs(metric_cls),
        )


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_exclude_centers_is_not_supported(metric_cls) -> None:
    with pytest.raises(TypeError, match="exclude_confounders"):
        metric_cls.compute(
            features=_all_rows_features(),
            manifest=_all_rows_manifest(),
            evaluation_design="all",
            k_candidates=[1],
            exclude_confounders=["V1"],
            **_metric_kwargs(metric_cls),
        )


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_compute_requires_explicit_confounder_column(metric_cls) -> None:
    with pytest.raises(TypeError, match="confounder_column"):
        metric_cls.compute(
            features=_all_rows_features(),
            manifest=_all_rows_manifest(),
            evaluation_design="all",
            k_candidates=[1],
            **({"tau": 0.2} if metric_cls is MaRI else {}),
        )
