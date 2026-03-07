import numpy as np
import pandas as pd
import pytest

from mari import MaRI, RI


def _manifest_two_centers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )


def _features_two_centers() -> np.ndarray:
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


def _manifest_three_centers() -> pd.DataFrame:
    return pd.DataFrame(
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


def _features_three_centers() -> np.ndarray:
    return np.array(
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


def _metric_kwargs(metric_cls) -> dict:
    return {"tau": 0.2} if metric_cls is MaRI else {}


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("mode", ["paired", "global"])
def test_compute_returns_expected_shapes(metric_cls, mode: str) -> None:
    result = metric_cls.compute(
        features=_features_two_centers(),
        manifest=_manifest_two_centers(),
        mode=mode,
        k_candidates=[1, 3],
        random_state=0,
        **_metric_kwargs(metric_cls),
    )

    assert result.dataset == "toy"
    assert result.k in {1, 3}
    assert 0.0 <= result.value <= 1.0
    assert result.n_pairs == 1
    assert result.pair_values.shape == (1,)
    assert result.sample_values.shape[0] <= 8
    assert result.sample_values_aligned.shape == (8,)
    assert result.sample_undefined_types.shape == (8,)


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("mode", ["paired", "global"])
def test_compute_curve_matches_single_k_compute(metric_cls, mode: str) -> None:
    curve_kwargs = {"tau": 0.2} if metric_cls is MaRI else {}
    curve = metric_cls.compute_curve(
        features=_features_two_centers(),
        manifest=_manifest_two_centers(),
        mode=mode,
        k_values=[1, 3],
        random_state=0,
        **curve_kwargs,
    )

    assert set(curve) == {1, 3}
    for k in (1, 3):
        result = metric_cls.compute(
            features=_features_two_centers(),
            manifest=_manifest_two_centers(),
            mode=mode,
            k_candidates=[k],
            random_state=0,
            **_metric_kwargs(metric_cls),
        )
        assert float(curve[k]) == pytest.approx(float(result.value))


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
def test_paired_mode_aggregates_sample_values_to_original_samples(metric_cls) -> None:
    result = metric_cls.compute(
        features=_features_three_centers(),
        manifest=_manifest_three_centers(),
        mode="paired",
        k_candidates=[1, 3],
        random_state=0,
        **_metric_kwargs(metric_cls),
    )

    assert result.n_pairs == 3
    assert result.sample_values.shape[0] <= len(_manifest_three_centers())


@pytest.mark.parametrize("metric_cls", [RI, MaRI])
@pytest.mark.parametrize("mode", ["paired", "global"])
def test_exclude_centers_matches_manual_manifest_filter(metric_cls, mode: str) -> None:
    manifest = _manifest_three_centers()
    features = _features_three_centers()
    keep_mask = manifest["medical_center"] != "C3"

    expected = metric_cls.compute(
        features=features[keep_mask.to_numpy()],
        manifest=manifest.loc[keep_mask].reset_index(drop=True),
        mode=mode,
        k_candidates=[1, 3],
        random_state=0,
        **_metric_kwargs(metric_cls),
    )
    result = metric_cls.compute(
        features=features,
        manifest=manifest,
        mode=mode,
        k_candidates=[1, 3],
        exclude_centers=["C3"],
        random_state=0,
        **_metric_kwargs(metric_cls),
    )

    assert result.k == expected.k
    assert result.n_pairs == expected.n_pairs
    assert result.value == pytest.approx(expected.value)
    assert result.std == pytest.approx(expected.std)
    assert np.allclose(result.pair_values, expected.pair_values, atol=1e-12, rtol=0.0)
    assert np.allclose(result.sample_values, expected.sample_values, atol=1e-12, rtol=0.0)


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        RI.compute(
            features=_features_two_centers(),
            manifest=_manifest_two_centers(),
            mode="auto",
            k_candidates=[1],
        )


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
            mode="global",
            k_candidates=[1],
            exclude_centers=["C1"],
            **_metric_kwargs(metric_cls),
        )
