
import inspect

import numpy as np
import pandas as pd
import pytest

from mari import CCRR
import mari.metrics.ccrr as ccrr_mod
from mari.metrics.ccrr import (
    CrossConfounderRetrievalRatio,
    _compute_sample_ccrr,
)


def _make_manifest(
    n: int,
    labels: list[str],
    centers: list[str],
    slide_ids: list[str] | None = None,
) -> pd.DataFrame:
    if slide_ids is None:
        slide_ids = [f"slide-{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/tmp/{i}.png" for i in range(n)],
            "label": labels,
            "medical_center": centers,
            "slide_id": slide_ids,
            "dataset": ["toy"] * n,
        }
    )


def _toy_features_so_closer() -> tuple[np.ndarray, pd.DataFrame]:
    manifest = _make_manifest(
        n=8,
        labels=["A", "A", "A", "A", "B", "B", "B", "B"],
        centers=["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
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
    return features, manifest


class TestComputeSampleCCRR:

    def test_basic_ratio(self) -> None:
        so_dists = np.array([[0.1], [0.3]])
        os_dists = np.array([[0.3], [0.1]])

        ccrr = _compute_sample_ccrr(so_dists, os_dists)

        assert ccrr[0] == pytest.approx(3.0)
        assert ccrr[1] == pytest.approx(1.0 / 3.0)

    def test_inf_produces_nan(self) -> None:
        so_dists = np.array([[0.1], [np.inf]])
        os_dists = np.array([[0.3], [0.2]])

        ccrr = _compute_sample_ccrr(so_dists, os_dists)

        assert np.isfinite(ccrr[0])
        assert np.isnan(ccrr[1])


class TestCCRRCompute:

    def test_default_k_growth_factor_is_two(self) -> None:
        sig = inspect.signature(CrossConfounderRetrievalRatio.compute)
        assert sig.parameters["k_growth_factor"].default == 2.0

    def test_compute_list_m_matches_individual_compute(self) -> None:
        features, manifest = _toy_features_so_closer()
        by_m = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=[1, 2],
        )
        assert isinstance(by_m, dict)

        single_m1 = CCRR.compute(features=features, manifest=manifest, mode="global", m=1)
        single_m2 = CCRR.compute(features=features, manifest=manifest, mode="global", m=2)

        assert by_m[1].value == pytest.approx(single_m1.value)
        assert by_m[1].std == pytest.approx(single_m1.std)
        assert by_m[1].undefined_frac == pytest.approx(single_m1.undefined_frac)
        assert by_m[1].q_alpha == pytest.approx(single_m1.q_alpha)
        assert by_m[1].ltm_alpha == pytest.approx(single_m1.ltm_alpha)

        assert by_m[2].value == pytest.approx(single_m2.value)
        assert by_m[2].std == pytest.approx(single_m2.std)
        assert by_m[2].undefined_frac == pytest.approx(single_m2.undefined_frac)
        assert by_m[2].q_alpha == pytest.approx(single_m2.q_alpha)
        assert by_m[2].ltm_alpha == pytest.approx(single_m2.ltm_alpha)

    def test_compute_list_m_uses_single_neighbor_search_for_multiple_m(self, monkeypatch: pytest.MonkeyPatch) -> None:
        features, manifest = _toy_features_so_closer()
        calls = {"n": 0}
        original = ccrr_mod._iterative_typed_neighbor_search

        def wrapped(**kwargs):
            calls["n"] += 1
            return original(**kwargs)

        monkeypatch.setattr(ccrr_mod, "_iterative_typed_neighbor_search", wrapped)
        by_m = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=[1, 2],
        )
        assert isinstance(by_m, dict)

        assert set(by_m) == {1, 2}
        assert calls["n"] == 1

    @pytest.mark.parametrize("mode", ["paired", "global"])
    def test_compute_returns_valid_result(self, mode: str) -> None:
        features, manifest = _toy_features_so_closer()
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode=mode,
            m=1,
        )

        assert result.dataset == "toy"
        assert result.m == 1
        assert np.isfinite(result.value)
        assert result.n_pairs >= 1
        assert result.sample_values.shape[0] <= 8
        assert result.acceptance_met
        assert result.undefined_frac == pytest.approx(0.0)

    def test_so_closer_yields_ccrr_above_one(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = CCRR.compute(features=features, manifest=manifest, mode="global", m=1)
        assert result.value > 1.0

    def test_os_closer_yields_ccrr_below_one(self) -> None:
        manifest = _make_manifest(
            n=8,
            labels=["A", "A", "B", "B", "A", "A", "B", "B"],
            centers=["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"],
        )
        features = np.array(
            [
                [1.00, 0.00],
                [0.00, 1.00],
                [0.95, 0.05],
                [0.05, 0.95],
                [0.92, 0.08],
                [0.08, 0.92],
                [0.90, 0.10],
                [0.10, 0.90],
            ],
            dtype=float,
        )
        result = CCRR.compute(features=features, manifest=manifest, mode="global", m=1)
        assert result.value < 1.0

    def test_all_same_label_and_center_are_undefined(self) -> None:
        manifest = _make_manifest(
            n=4,
            labels=["A", "A", "A", "A"],
            centers=["C1", "C1", "C1", "C1"],
        )
        features = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=float)

        result = CCRR.compute(features=features, manifest=manifest, mode="global", m=1)

        assert result.undefined_frac == pytest.approx(1.0)
        assert result.sample_values.shape[0] == 0

    def test_relaxed_acceptance_threshold_stops_earlier(self) -> None:
        features, manifest = _toy_features_so_closer()

        strict = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
            start_k=1,
            acceptance_threshold=0.0,
            k_growth_factor=1.5,
        )
        relaxed = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
            start_k=1,
            acceptance_threshold=1.0,
            k_growth_factor=1.5,
        )

        assert relaxed.k_final == 1
        assert relaxed.retries == 0
        assert strict.k_final >= relaxed.k_final
        assert strict.retries >= relaxed.retries

    def test_unmet_acceptance_threshold_at_cap_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        labels = ["A", "A", "B", "B"]
        centers = ["C1", "C2", "C1", "C2"]
        slides = ["s1", "s1", "s2", "s2"]
        manifest = _make_manifest(n=4, labels=labels, centers=centers, slide_ids=slides)
        features = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ],
            dtype=float,
        )

        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
            start_k=1,
            acceptance_threshold=0.0,
            k_growth_factor=1.5,
        )

        assert not result.acceptance_met
        assert result.k_final == 3
        assert result.undefined_frac > 0.0
        assert any("undefined threshold unmet" in rec.message for rec in caplog.records)

    def test_start_k_is_clamped(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = CCRR.compute(features=features, manifest=manifest, mode="global", m=1, start_k=200)
        assert result.k_start == len(manifest) - 1

    def test_growth_schedule_follows_factor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        labels = ["A", "A", "A", "B", "B", "B", "A", "A", "B", "B"]
        centers = ["C1", "C2", "C1", "C2", "C1", "C2", "C2", "C1", "C2", "C1"]
        manifest = _make_manifest(n=10, labels=labels, centers=centers)
        features = np.column_stack(
            [
                np.linspace(0.0, 1.0, 10, dtype=float),
                np.linspace(1.0, 0.0, 10, dtype=float),
            ]
        )

        class _FakeNN:
            def __init__(self, metric: str) -> None:
                self.metric = metric

            def fit(self, _x: np.ndarray) -> "_FakeNN":
                return self

            def kneighbors(self, x: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
                q = int(x.shape[0])
                d = np.zeros((q, n_neighbors), dtype=float)
                idx = np.tile(np.arange(n_neighbors, dtype=int) % 10, (q, 1))
                return d, idx

        def _never_define(**kwargs) -> np.ndarray:
            query_indices = kwargs["query_indices"]
            return np.zeros((len(query_indices),), dtype=bool)

        monkeypatch.setattr(ccrr_mod, "NearestNeighbors", _FakeNN)
        monkeypatch.setattr(ccrr_mod, "_scan_typed_neighbors_for_query_rows", _never_define)

        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
            start_k=2,
            acceptance_threshold=0.0,
            k_growth_factor=1.5,
        )

        assert result.k_start == 2
        assert result.k_final == 9
        assert result.retries == 4
        assert not result.acceptance_met

    def test_queries_only_unresolved_samples_over_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        labels = ["A", "A", "A", "A", "B", "B", "B", "B"]
        centers = ["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"]
        manifest = _make_manifest(n=8, labels=labels, centers=centers)
        features = np.column_stack(
            [
                np.linspace(0.0, 1.0, 8, dtype=float),
                np.linspace(1.0, 0.0, 8, dtype=float),
            ]
        )

        query_sizes: list[int] = []

        class _FakeNN:
            def __init__(self, metric: str) -> None:
                self.metric = metric

            def fit(self, _x: np.ndarray) -> "_FakeNN":
                return self

            def kneighbors(self, x: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
                q = int(x.shape[0])
                query_sizes.append(q)
                d = np.zeros((q, n_neighbors), dtype=float)
                idx = np.tile(np.arange(n_neighbors, dtype=int) % 8, (q, 1))
                return d, idx

        call_count = {"n": 0}

        def _define_half_then_all(**kwargs) -> np.ndarray:
            query_indices = kwargs["query_indices"]
            n = int(len(query_indices))
            out = np.zeros((n,), dtype=bool)
            call_count["n"] += 1
            if call_count["n"] == 1:
                out[: max(1, n // 2)] = True
            else:
                out[:] = True
            return out

        monkeypatch.setattr(ccrr_mod, "NearestNeighbors", _FakeNN)
        monkeypatch.setattr(ccrr_mod, "_scan_typed_neighbors_for_query_rows", _define_half_then_all)

        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
            start_k=1,
            acceptance_threshold=0.0,
            k_growth_factor=1.5,
        )

        assert len(query_sizes) >= 2
        assert query_sizes[1] < query_sizes[0]
        assert result.acceptance_met

    def test_invalid_mode_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="mode"):
            CCRR.compute(features=features, manifest=manifest, mode="auto", m=1)

    def test_m_zero_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="m must be >= 1"):
            CCRR.compute(features=features, manifest=manifest, mode="global", m=0)

    def test_acceptance_threshold_bounds_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="acceptance_threshold"):
            CCRR.compute(features=features, manifest=manifest, mode="global", m=1, acceptance_threshold=-0.1)
        with pytest.raises(ValueError, match="acceptance_threshold"):
            CCRR.compute(features=features, manifest=manifest, mode="global", m=1, acceptance_threshold=1.1)

    def test_growth_factor_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="k_growth_factor"):
            CCRR.compute(features=features, manifest=manifest, mode="global", m=1, k_growth_factor=1.0)

    def test_start_k_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="start_k"):
            CCRR.compute(features=features, manifest=manifest, mode="global", m=1, start_k=0)

    def test_exclude_centers(self) -> None:
        manifest = _make_manifest(
            n=12,
            labels=["A"] * 6 + ["B"] * 6,
            centers=["C1", "C1", "C2", "C2", "C3", "C3", "C1", "C1", "C2", "C2", "C3", "C3"],
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
        mask = manifest["medical_center"] != "C3"
        result_excluded = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
            exclude_centers=["C3"],
        )
        result_manual = CCRR.compute(
            features=features[mask.to_numpy()],
            manifest=manifest.loc[mask].reset_index(drop=True),
            mode="global",
            m=1,
        )
        assert result_excluded.value == pytest.approx(result_manual.value)

    def test_api_alias(self) -> None:
        assert CCRR is CrossConfounderRetrievalRatio

    def test_manual_kmax_not_supported(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(TypeError):
            CCRR.compute(features=features, manifest=manifest, mode="global", m=1, kmax=3)  # type: ignore[call-arg]
