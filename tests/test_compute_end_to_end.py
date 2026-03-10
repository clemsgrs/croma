import numpy as np
import pandas as pd
import pytest

from croma import MaRI, RI


def _dataset_wide_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a_c1", "a_c2", "b_c1", "b_c2"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "medical_center": ["C1", "C2", "C1", "C2"],
            "slide_id": [f"slide-{i}" for i in range(4)],
            "dataset": ["toy"] * 4,
        }
    )


def _dataset_wide_features() -> np.ndarray:
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
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
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
    return {"tau": 0.2} if metric_cls is MaRI else {}


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_compute_returns_occurrence_aligned_outputs(metric_cls) -> None:
    manifest = _paired_manifest_single_subset()
    result = metric_cls.compute(
        features=_paired_features_single_subset(),
        manifest=manifest,
        evaluation_design="paired_2x2",
        k_candidates=[1, 3],
        random_state=0,
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
def test_paired_keeps_repeated_subset_memberships_as_distinct_occurrences(metric_cls) -> None:
    manifest = _paired_manifest_repeated_subset_memberships()
    result = metric_cls.compute(
        features=_paired_features_repeated_subset_memberships(),
        manifest=manifest,
        evaluation_design="paired_2x2",
        k_candidates=[1],
        random_state=0,
        **_metric_kwargs(metric_cls),
    )

    assert result.evaluation_unit == "occurrence"
    assert result.n_pairs == 2
    assert result.sample_values_aligned.shape == (len(manifest),)
    assert sorted(set(result.occurrence_subsets.tolist())) == ["pair0", "pair1"]
    assert result.occurrence_source_indices.tolist() == list(range(len(manifest)))


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_dataset_wide_compute_returns_expected_exact_value(metric_cls) -> None:
    result = metric_cls.compute(
        features=_dataset_wide_features(),
        manifest=_dataset_wide_manifest(),
        evaluation_design="dataset_wide",
        k_candidates=[1],
        **_metric_kwargs(metric_cls),
    )

    assert result.k == 1
    assert result.value == pytest.approx(1.0)
    assert result.std == pytest.approx(0.0)
    assert result.n_pairs == 1
    assert result.evaluation_design == "dataset_wide"
    assert result.evaluation_unit == "sample"
    assert result.sample_values_aligned.shape == (4,)
    assert result.occurrence_defined_mask.tolist() == [True, True, True, True]
    assert result.sample_undefined_types.tolist() == [0, 0, 0, 0]
    assert result.occurrence_subsets.tolist() == ["dataset"] * 4
    assert result.occurrence_source_indices.tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("evaluation_design", ["paired_2x2", "dataset_wide"])
def test_compute_curve_matches_single_k_compute(metric_cls, evaluation_design: str) -> None:
    if evaluation_design == "paired_2x2":
        manifest = _paired_manifest_single_subset()
        features = _paired_features_single_subset()
    else:
        manifest = _dataset_wide_manifest()
        features = _dataset_wide_features()

    curve_kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    curve = metric_cls.compute_curve(
        features=features,
        manifest=manifest,
        evaluation_design=evaluation_design,
        k_values=[1],
        random_state=0,
        **curve_kwargs,
    )

    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        evaluation_design=evaluation_design,
        k_candidates=[1],
        random_state=0,
        **_metric_kwargs(metric_cls),
    )

    assert curve == {1: pytest.approx(float(result.value))}


def test_compute_rejects_unknown_evaluation_design() -> None:
    with pytest.raises(ValueError, match="evaluation_design"):
        RI.compute(
            features=_dataset_wide_features(),
            manifest=_dataset_wide_manifest(),
            evaluation_design="auto",
            k_candidates=[1],
        )


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_requires_subset_metadata(metric_cls) -> None:
    with pytest.raises(ValueError, match="subset"):
        metric_cls.compute(
            features=_dataset_wide_features(),
            manifest=_dataset_wide_manifest(),
            evaluation_design="paired_2x2",
            k_candidates=[1],
            **_metric_kwargs(metric_cls),
        )


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_exclude_centers_matches_manual_manifest_filter_for_dataset_wide(metric_cls) -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A"] * 4 + ["B"] * 4,
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )
    features = np.array(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.00, 1.00],
            [0.01, 0.99],
            [0.02, 0.98],
            [0.03, 0.97],
        ],
        dtype=float,
    )
    keep_mask = manifest["medical_center"] != "C2"

    expected = metric_cls.compute(
        features=features[keep_mask.to_numpy()],
        manifest=manifest.loc[keep_mask].reset_index(drop=True),
        evaluation_design="dataset_wide",
        k_candidates=[1],
        **_metric_kwargs(metric_cls),
    )
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        evaluation_design="dataset_wide",
        k_candidates=[1],
        exclude_centers=["C2"],
        **_metric_kwargs(metric_cls),
    )

    assert result.value == pytest.approx(expected.value)
    assert result.undefined_frac == pytest.approx(expected.undefined_frac)
    np.testing.assert_allclose(result.sample_values_aligned, expected.sample_values_aligned, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_exclude_centers_raises_when_all_rows_removed(metric_cls) -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "medical_center": ["C1", "C1"],
            "slide_id": ["slide-0", "slide-1"],
            "dataset": ["toy"] * 2,
        }
    )
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="No samples remain"):
        metric_cls.compute(
            features=features,
            manifest=manifest,
            evaluation_design="dataset_wide",
            k_candidates=[1],
            exclude_centers=["C1"],
            **_metric_kwargs(metric_cls),
        )
