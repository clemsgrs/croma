from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mari import MaRI, RI


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("mode", ["paired", "global"])
def test_compute_end_to_end(metric_cls, mode: str) -> None:
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

    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        mode=mode,
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )

    assert result.dataset == "toy"
    assert result.k in {1, 3}
    assert 0.0 <= result.value <= 1.0
    assert result.n_pairs == 1
    assert result.pair_values.shape == (1,)
    assert result.sample_values.shape[0] <= 8


def test_invalid_mode_rejected() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["/tmp/0.png", "/tmp/1.png"],
            "label": ["A", "B"],
            "medical_center": ["C1", "C2"],
            "slide_id": ["slide-0", "slide-1"],
        }
    )
    features = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="mode"):
        RI.compute(
            features=features,
            manifest=manifest,
            mode="auto",
            k_candidates=[1],
        )


@pytest.mark.parametrize("mode", ["paired", "global"])
def test_ri_compute_curve_matches_single_k_compute(mode: str) -> None:
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
    k_values = [1, 3]

    curve = RI.compute_curve(
        features=features,
        manifest=manifest,
        mode=mode,
        k_values=k_values,
        random_state=0,
    )
    assert set(curve) == set(k_values)

    for k in k_values:
        one_k = RI.compute(
            features=features,
            manifest=manifest,
            mode=mode,
            k_candidates=[k],
            random_state=0,
        )
        assert float(curve[k]) == pytest.approx(float(one_k.value))


@pytest.mark.parametrize("mode", ["paired", "global"])
def test_mari_compute_curve_matches_single_k_compute(mode: str) -> None:
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
    k_values = [1, 3]

    curve = MaRI.compute_curve(
        features=features,
        manifest=manifest,
        mode=mode,
        k_values=k_values,
        tau=0.2,
        random_state=0,
    )
    assert set(curve) == set(k_values)

    for k in k_values:
        one_k = MaRI.compute(
            features=features,
            manifest=manifest,
            mode=mode,
            k_candidates=[k],
            tau=0.2,
            random_state=0,
        )
        assert float(curve[k]) == pytest.approx(float(one_k.value))


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_mode_sample_values_are_aggregated_per_original_sample(metric_cls) -> None:
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
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        mode="paired",
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )

    # For this manifest there is one valid 2x2 pair; sample_values should still
    # report at most one informative value per original sample.
    assert result.n_pairs == 1
    assert result.sample_values.shape[0] <= len(manifest)


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_mode_sample_values_not_multiplied_by_pair_count(metric_cls) -> None:
    manifest = pd.DataFrame(
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
    features = np.array(
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
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        mode="paired",
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )
    # With 2 classes and 3 centers there are 3 valid center pairs, so a concat-style
    # implementation would produce 36 sample scores. Aggregated-per-sample should
    # remain bounded by original sample count.
    assert result.n_pairs == 3
    assert result.sample_values.shape[0] <= len(manifest)


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("mode", ["paired", "global"])
def test_exclude_centers_matches_manual_manifest_filter(metric_cls, mode: str) -> None:
    manifest = pd.DataFrame(
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
    features = np.array(
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
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    expected_mask = manifest["medical_center"] != "C3"
    expected_manifest = manifest.loc[expected_mask].reset_index(drop=True)
    expected_features = features[expected_mask.to_numpy()]

    expected = metric_cls.compute(
        features=expected_features,
        manifest=expected_manifest,
        mode=mode,
        k_candidates=[1, 3],
        random_state=0,
        **kwargs,
    )
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        mode=mode,
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
    kwargs = {"tau": 0.2} if metric_cls is MaRI else {}

    with pytest.raises(ValueError, match="No samples remain"):
        metric_cls.compute(
            features=features,
            manifest=manifest,
            mode="global",
            k_candidates=[1],
            exclude_centers=["C1"],
            **kwargs,
        )
