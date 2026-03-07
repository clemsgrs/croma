import numpy as np
import pandas as pd
import pytest

from mari import MaRI, RI
from mari.metrics.pairs import load_manifest


def _single_subset_manifest() -> pd.DataFrame:
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


def _dataset_wide_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"dw{i}" for i in range(4)],
            "image_path": [f"/tmp/dw{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "medical_center": ["C1", "C2", "C1", "C2"],
            "slide_id": [f"dw-slide-{i}" for i in range(4)],
            "dataset": ["toy_dw"] * 4,
        }
    )


def _dataset_wide_features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00],
            [0.90, 0.10],
            [0.00, 1.00],
            [0.10, 0.90],
        ],
        dtype=float,
    )


def _multi_subset_manifest() -> pd.DataFrame:
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


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_compute_end_to_end(metric_cls) -> None:
    manifest = _single_subset_manifest()
    features = _single_subset_features()

    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )

    assert result.dataset == "toy"
    assert result.k in {1, 3}
    assert 0.0 <= result.value <= 1.0
    assert result.n_pairs == 1
    assert result.pair_values.shape == (1,)
    assert result.sample_values.shape[0] <= len(manifest)
    assert result.sample_values_aligned.shape == (len(manifest),)
    assert result.occurrence_defined_mask.shape == (len(manifest),)
    assert result.sample_undefined_types.shape == (len(manifest),)
    assert result.occurrence_subsets.shape == (len(manifest),)
    assert result.occurrence_source_indices.shape == (len(manifest),)
    assert set(result.occurrence_subsets.tolist()) == {"pair0"}
    assert set(result.occurrence_source_indices.tolist()) == set(range(len(manifest)))


def test_compute_requires_subset_metadata() -> None:
    manifest = _single_subset_manifest().drop(columns=["subset"])
    features = _single_subset_features()

    with pytest.raises(ValueError, match="subset membership"):
        RI.compute(
            features=features,
            manifest=manifest,
            k_candidates=[1],
        )


def test_paired_compute_rejects_manifest_without_subset_even_with_sidecar(tmp_path) -> None:
    manifest_path = tmp_path / "toy.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "medical_center": ["C1", "C2"],
            "slide_id": ["slide-0", "slide-1"],
            "patch_id": ["p0", "p1"],
        }
    ).to_csv(manifest_path, index=False)
    pd.DataFrame(
        {
            "subset": ["pair0", "pair0"],
            "slide_id": ["slide-0", "slide-1"],
            "patch_id": ["p0", "p1"],
        }
    ).to_csv(tmp_path / "toy-subsets.csv", index=False)

    manifest = load_manifest(str(manifest_path), dataset_name="toy")

    assert "subset" not in manifest.columns

    with pytest.raises(ValueError, match="subset'.*column"):
        RI.compute(
            features=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            manifest=manifest,
            k_candidates=[1],
        )


def test_compute_rejects_invalid_subset_definition() -> None:
    manifest = _single_subset_manifest().copy()
    manifest.loc[manifest["sample_id"] == "s7", "medical_center"] = "C3"
    features = _single_subset_features()

    with pytest.raises(ValueError, match="invalid subsets"):
        RI.compute(
            features=features,
            manifest=manifest,
            k_candidates=[1],
        )


def test_compute_rejects_unknown_evaluation_design() -> None:
    manifest = _single_subset_manifest()
    features = _single_subset_features()

    with pytest.raises(ValueError, match="evaluation_design"):
        RI.compute(
            features=features,
            manifest=manifest,
            k_candidates=[1],
            evaluation_design="unknown",
        )


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_dataset_wide_compute_works_without_subset_metadata(metric_cls) -> None:
    manifest = _dataset_wide_manifest()
    features = _dataset_wide_features()

    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1],
        evaluation_design="dataset_wide",
        **kwargs,
    )

    assert result.dataset == "toy_dw"
    assert result.evaluation_design == "dataset_wide"
    assert result.evaluation_unit == "sample"
    assert result.k == 1
    assert result.value == pytest.approx(1.0)
    assert result.undefined_frac == pytest.approx(0.0)
    assert result.sample_values.shape == (len(manifest),)
    assert result.sample_values_aligned.shape == (len(manifest),)
    assert result.occurrence_defined_mask.shape == (len(manifest),)
    assert set(result.occurrence_subsets.tolist()) == {"dataset"}
    assert np.array_equal(result.occurrence_source_indices, np.arange(len(manifest), dtype=int))


def test_ri_compute_curve_matches_single_k_compute() -> None:
    manifest = _single_subset_manifest()
    features = _single_subset_features()
    k_values = [1, 3]

    curve = RI.compute_curve(
        features=features,
        manifest=manifest,
        k_values=k_values,
        random_state=0,
    )
    assert set(curve) == set(k_values)

    for k in k_values:
        one_k = RI.compute(
            features=features,
            manifest=manifest,
            k_candidates=[k],
            random_state=0,
        )
        assert float(curve[k]) == pytest.approx(float(one_k.value))


def test_mari_compute_curve_matches_single_k_compute() -> None:
    manifest = _single_subset_manifest()
    features = _single_subset_features()
    k_values = [1, 3]

    curve = MaRI.compute_curve(
        features=features,
        manifest=manifest,
        k_values=k_values,
        tau=0.2,
        random_state=0,
    )
    assert set(curve) == set(k_values)

    for k in k_values:
        one_k = MaRI.compute(
            features=features,
            manifest=manifest,
            k_candidates=[k],
            tau=0.2,
            random_state=0,
        )
        assert float(curve[k]) == pytest.approx(float(one_k.value))


def test_dataset_wide_curve_matches_single_k_compute() -> None:
    manifest = _dataset_wide_manifest()
    features = _dataset_wide_features()

    curve = RI.compute_curve(
        features=features,
        manifest=manifest,
        k_values=[1],
        evaluation_design="dataset_wide",
    )

    result = RI.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1],
        evaluation_design="dataset_wide",
    )

    assert curve == {1: pytest.approx(result.value)}


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_occurrence_outputs_expand_for_multi_subset_memberships(metric_cls) -> None:
    manifest = _multi_subset_manifest()
    features = _multi_subset_features()

    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )

    assert result.n_pairs == 3
    assert result.sample_values_aligned.shape == (24,)
    assert result.occurrence_defined_mask.shape == (24,)
    assert result.sample_undefined_types.shape == (24,)
    assert result.occurrence_subsets.shape == (24,)
    assert result.occurrence_source_indices.shape == (24,)
    assert set(result.occurrence_subsets.tolist()) == {"pair12", "pair13", "pair23"}
    assert set(result.occurrence_source_indices.tolist()) == set(range(len(manifest)))


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_exclude_centers_matches_manual_manifest_filter(metric_cls) -> None:
    manifest = _multi_subset_manifest()
    features = _multi_subset_features()
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    expected_mask = (manifest["medical_center"] != "C3") & (manifest["subset"] == "pair12")
    expected_manifest = manifest.loc[expected_mask].reset_index(drop=True)
    expected_features = features[expected_mask.to_numpy()]

    expected = metric_cls.compute(
        features=expected_features,
        manifest=expected_manifest,
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        k_candidates=[1, 3],
        exclude_centers=["C3"],
        random_state=0,
        **kwargs,
    )

    assert result.k == expected.k
    assert result.n_pairs == expected.n_pairs
    assert result.value == pytest.approx(expected.value)
    assert result.std == pytest.approx(expected.std)
    assert np.allclose(result.pair_values, expected.pair_values, atol=1e-12, rtol=0.0)
    assert np.allclose(result.sample_values, expected.sample_values, atol=1e-12, rtol=0.0)
    assert np.array_equal(result.occurrence_subsets, expected.occurrence_subsets)
    assert result.occurrence_source_indices.tolist() == np.flatnonzero(
        (
            (
                manifest.loc[manifest["medical_center"] != "C3", "subset"].reset_index(drop=True)
                == "pair12"
            ).to_numpy()
        )
    ).tolist()


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_exclude_centers_raises_when_all_rows_removed(metric_cls) -> None:
    manifest = _single_subset_manifest()
    features = _single_subset_features()
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}

    with pytest.raises(ValueError, match="No samples remain"):
        metric_cls.compute(
            features=features,
            manifest=manifest,
            k_candidates=[1],
            exclude_centers=["C1", "C2"],
            **kwargs,
        )
