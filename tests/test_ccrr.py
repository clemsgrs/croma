from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mari import CCRR
from mari.metrics.ccrr import (
    CrossConfounderRetrievalRatio,
    _compute_sample_ccrr,
    _find_typed_neighbor_distances,
)


def _make_manifest(n: int, labels: list[str], centers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/tmp/{i}.png" for i in range(n)],
            "label": labels,
            "medical_center": centers,
            "slide_id": [f"slide-{i}" for i in range(n)],
            "dataset": ["toy"] * n,
        }
    )


class TestFindTypedNeighborDistances:

    def test_finds_correct_so_os_distances(self) -> None:
        labels = np.array([0, 0, 0, 1, 1])
        centers = np.array([0, 1, 0, 0, 1])
        # Neighbors of sample 0: [1, 2, 3, 4] with distances [0.1, 0.2, 0.3, 0.4]
        neigh_idx = np.array(
            [
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [4, 0, 1, 2],
                [3, 0, 1, 2],
            ]
        )
        neigh_dist = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        valid_counts = np.array([4, 4, 4, 4, 4])

        so_dists, os_dists = _find_typed_neighbor_distances(
            labels, centers, neigh_idx, neigh_dist, valid_counts, m=1
        )

        # Sample 0: label=0, center=0
        #   neighbor 1: label=0, center=1 -> SO, dist=0.1
        #   neighbor 3: label=1, center=0 -> OS, dist=0.3
        assert so_dists[0, 0] == pytest.approx(0.1)
        assert os_dists[0, 0] == pytest.approx(0.3)

    def test_m_greater_than_1(self) -> None:
        labels = np.array([0, 0, 0, 1, 1])
        centers = np.array([0, 1, 1, 0, 0])
        neigh_idx = np.array(
            [
                [1, 2, 3, 4],
                [0, 2, 3, 4],
                [0, 1, 3, 4],
                [4, 0, 1, 2],
                [3, 0, 1, 2],
            ]
        )
        neigh_dist = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        valid_counts = np.array([4, 4, 4, 4, 4])

        so_dists, os_dists = _find_typed_neighbor_distances(
            labels, centers, neigh_idx, neigh_dist, valid_counts, m=2
        )

        # Sample 0: label=0, center=0
        #   neighbor 1: label=0, center=1 -> SO, dist=0.1
        #   neighbor 2: label=0, center=1 -> SO, dist=0.2
        #   neighbor 3: label=1, center=0 -> OS, dist=0.3
        #   neighbor 4: label=1, center=0 -> OS, dist=0.4
        assert so_dists[0, 0] == pytest.approx(0.1)
        assert so_dists[0, 1] == pytest.approx(0.2)
        assert os_dists[0, 0] == pytest.approx(0.3)
        assert os_dists[0, 1] == pytest.approx(0.4)

    def test_unfound_neighbors_are_inf(self) -> None:
        # All neighbors have same label and same center -> no SO or OS
        labels = np.array([0, 0, 0])
        centers = np.array([0, 0, 0])
        neigh_idx = np.array([[1, 2], [0, 2], [0, 1]])
        neigh_dist = np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
        valid_counts = np.array([2, 2, 2])

        so_dists, os_dists = _find_typed_neighbor_distances(
            labels, centers, neigh_idx, neigh_dist, valid_counts, m=1
        )

        assert np.all(np.isinf(so_dists))
        assert np.all(np.isinf(os_dists))


class TestComputeSampleCCRR:

    def test_basic_ratio(self) -> None:
        so_dists = np.array([[0.1], [0.3]])
        os_dists = np.array([[0.3], [0.1]])

        ccrr = _compute_sample_ccrr(so_dists, os_dists)

        # sample 0: os/so = 0.3/0.1 = 3.0 (robust)
        # sample 1: os/so = 0.1/0.3 ≈ 0.333 (not robust)
        assert ccrr[0] == pytest.approx(3.0)
        assert ccrr[1] == pytest.approx(1.0 / 3.0)

    def test_inf_produces_nan(self) -> None:
        so_dists = np.array([[0.1], [np.inf]])
        os_dists = np.array([[0.3], [0.2]])

        ccrr = _compute_sample_ccrr(so_dists, os_dists)

        assert np.isfinite(ccrr[0])
        assert np.isnan(ccrr[1])

    def test_m_greater_than_1_averages(self) -> None:
        so_dists = np.array([[0.1, 0.2]])
        os_dists = np.array([[0.4, 0.6]])

        ccrr = _compute_sample_ccrr(so_dists, os_dists)

        # mean_so = 0.15, mean_os = 0.5 -> ratio = 0.5/0.15 ≈ 3.333
        assert ccrr[0] == pytest.approx(0.5 / 0.15)


class TestCCRRCompute:

    def _toy_features_so_closer(self) -> tuple[np.ndarray, pd.DataFrame]:
        """Features where SO neighbors are closer than OS neighbors."""
        manifest = _make_manifest(
            n=8,
            labels=["A", "A", "A", "A", "B", "B", "B", "B"],
            centers=["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
        )
        # A-cluster near [1,0], B-cluster near [0,1] -> SO closer than OS
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

    @pytest.mark.parametrize("mode", ["paired", "global"])
    def test_compute_returns_valid_result(self, mode: str) -> None:
        features, manifest = self._toy_features_so_closer()
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

    def test_so_closer_yields_ccrr_above_one(self) -> None:
        features, manifest = self._toy_features_so_closer()
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
        )
        # When SO is closer than OS, ccrr = d(OS)/d(SO) > 1
        assert result.value > 1.0

    def test_os_closer_yields_ccrr_below_one(self) -> None:
        """Features where OS neighbors are closer than SO neighbors."""
        manifest = _make_manifest(
            n=8,
            labels=["A", "A", "B", "B", "A", "A", "B", "B"],
            centers=["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"],
        )
        # Cluster by center, not by label:
        # C1 near [1,0], C2 near [0,1]
        features = np.array(
            [
                [1.00, 0.00],  # A, C1
                [0.00, 1.00],  # A, C2
                [0.95, 0.05],  # B, C1
                [0.05, 0.95],  # B, C2
                [0.92, 0.08],  # A, C1
                [0.08, 0.92],  # A, C2
                [0.90, 0.10],  # B, C1
                [0.10, 0.90],  # B, C2
            ],
            dtype=float,
        )
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
        )
        # When OS (same center, different label) is closer than SO
        assert result.value < 1.0

    def test_all_same_label_and_center_undefined(self) -> None:
        manifest = _make_manifest(
            n=4,
            labels=["A", "A", "A", "A"],
            centers=["C1", "C1", "C1", "C1"],
        )
        features = np.array(
            [[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=float
        )
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=1,
        )
        assert result.undefined_frac == pytest.approx(1.0)
        assert result.sample_values.shape[0] == 0

    def test_invalid_mode_rejected(self) -> None:
        features, manifest = self._toy_features_so_closer()
        with pytest.raises(ValueError, match="mode"):
            CCRR.compute(features=features, manifest=manifest, mode="auto", m=1)

    def test_m_zero_rejected(self) -> None:
        features, manifest = self._toy_features_so_closer()
        with pytest.raises(ValueError, match="m must be >= 1"):
            CCRR.compute(features=features, manifest=manifest, mode="global", m=0)

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

    def test_m_greater_than_1(self) -> None:
        features, manifest = self._toy_features_so_closer()
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode="global",
            m=2,
        )
        assert result.m == 2
        assert np.isfinite(result.value)

    @pytest.mark.parametrize("mode", ["paired", "global"])
    def test_paired_vs_global_both_work(self, mode: str) -> None:
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
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode=mode,
            m=1,
        )
        assert np.isfinite(result.value)
        if mode == "paired":
            assert result.n_pairs == 3
        else:
            assert result.n_pairs == 1
