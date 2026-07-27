import inspect
import re

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa
import croma.metrics.croma as croma_mod
from croma.metrics.croma import (
    CROMA_HEADLINE_M,
    CrossConfounderRobustnessMargin,
    _compute_sample_croma,
    _sample_croma_with_causes,
)
from metric_harness import constant_embedding


def _croma_warning(caplog: pytest.LogCaptureFixture) -> str:
    """The single ``[CRoMa]`` warning the run logged, as text."""
    messages = [rec.message for rec in caplog.records if rec.message.startswith("[CRoMa]")]
    assert len(messages) == 1, messages
    return messages[0]


def _make_manifest(
    n: int,
    labels: list[str],
    centers: list[str],
    slide_ids: list[str] | None = None,
    subset: str | None = None,
) -> pd.DataFrame:
    if slide_ids is None:
        slide_ids = [f"slide-{i}" for i in range(n)]
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/tmp/{i}.png" for i in range(n)],
            "label": labels,
            "scanner_vendor": centers,
            "slide_id": slide_ids,
            "dataset": ["toy"] * n,
        }
    )
    if subset is not None:
        manifest["subset"] = str(subset)
    return manifest


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


def _compute_croma(**kwargs):
    return CRoMa.compute(confounder_column="scanner_vendor", **kwargs)


class TestComputeSampleCRoMa:

    def test_basic_margin(self) -> None:
        # CRoMa is the signed normalized margin (d_OS - d_SO) / (d_OS + d_SO).
        so_dists = np.array([[0.1], [0.3]])
        os_dists = np.array([[0.3], [0.1]])

        croma = _compute_sample_croma(so_dists, os_dists)

        assert croma[0] == pytest.approx(0.5)  # (0.3 - 0.1) / (0.3 + 0.1)
        assert croma[1] == pytest.approx(-0.5)  # (0.1 - 0.3) / (0.1 + 0.3)

    def test_inf_produces_nan(self) -> None:
        so_dists = np.array([[0.1], [np.inf]])
        os_dists = np.array([[0.3], [0.2]])

        croma = _compute_sample_croma(so_dists, os_dists)

        assert np.isfinite(croma[0])
        assert np.isnan(croma[1])

    def test_the_two_causes_partition_the_undefined_samples(self) -> None:
        """An unfilled slot is the search's failure; an all-zero one is the denominator's.

        The masks have to be disjoint and cover the NaN set exactly, because the warning
        reports counts taken from them: any overlap double-counts a sample and any gap
        leaves one undefined for a reason nobody is told.
        """
        so_dists = np.array([[0.1], [np.inf], [0.0], [0.3]])
        os_dists = np.array([[0.3], [0.2], [0.0], [np.inf]])

        scored = _sample_croma_with_causes(so_dists, os_dists)

        assert scored.unresolved.tolist() == [False, True, False, True]
        assert scored.zero_distance.tolist() == [False, False, True, False]
        assert not np.any(scored.unresolved & scored.zero_distance)
        np.testing.assert_array_equal(
            scored.unresolved | scored.zero_distance, np.isnan(scored.values)
        )


class TestCRoMaCompute:

    def test_default_k_growth_factor_is_two(self) -> None:
        sig = inspect.signature(CrossConfounderRobustnessMargin.compute)
        assert sig.parameters["k_growth_factor"].default == 2.0

    def test_headline_m_default_is_five(self) -> None:
        # The headline averaging radius is m=5: the smallest window that removes
        # single-neighbour outlier sensitivity while staying in the local typed
        # shell (model ranking and sign are m-invariant; see paper croma.tex).
        assert CROMA_HEADLINE_M == 5
        sig = inspect.signature(CrossConfounderRobustnessMargin.compute)
        assert sig.parameters["m"].default == CROMA_HEADLINE_M

    def test_compute_list_m_matches_individual_compute(self) -> None:
        features, manifest = _toy_features_so_closer()
        by_m = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="dataset_wide",
            m=[1, 2],
        )
        assert isinstance(by_m, dict)

        single_m1 = _compute_croma(
            features=features, manifest=manifest, evaluation_design="dataset_wide", m=1
        )
        single_m2 = _compute_croma(
            features=features, manifest=manifest, evaluation_design="dataset_wide", m=2
        )

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

    def test_compute_list_m_uses_single_neighbor_search_for_multiple_m(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        features, manifest = _toy_features_so_closer()
        calls = {"n": 0}
        original = croma_mod._iterative_typed_neighbor_search

        def wrapped(**kwargs):
            calls["n"] += 1
            return original(**kwargs)

        monkeypatch.setattr(croma_mod, "_iterative_typed_neighbor_search", wrapped)
        by_m = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="dataset_wide",
            m=[1, 2],
        )
        assert isinstance(by_m, dict)

        assert set(by_m) == {1, 2}
        assert calls["n"] == 1

    @pytest.mark.parametrize("evaluation_design", ["paired_2x2", "dataset_wide"])
    def test_compute_returns_valid_result(self, evaluation_design: str) -> None:
        features, manifest = _toy_features_so_closer()
        if evaluation_design == "paired_2x2":
            manifest = manifest.copy()
            manifest["subset"] = "pair0"
        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design=evaluation_design,
            m=1,
        )

        assert result.dataset == "toy"
        assert result.m == 1
        assert np.isfinite(result.value)
        assert result.n_pairs >= 1
        assert result.sample_values.shape[0] <= 8
        assert result.sample_values_aligned.shape == (len(manifest),)
        assert result.occurrence_defined_mask.shape == (len(manifest),)
        assert result.undefined_frac == pytest.approx(0.0)
        assert result.occurrence_defined_mask.tolist() == [True] * len(manifest)

    def test_so_closer_yields_croma_above_zero_dataset_wide(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = _compute_croma(
            features=features, manifest=manifest, evaluation_design="dataset_wide", m=1
        )
        assert result.value > 0.0

    def test_os_closer_yields_croma_below_zero(self) -> None:
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
        result = _compute_croma(
            features=features, manifest=manifest, evaluation_design="dataset_wide", m=1
        )
        assert result.value < 0.0

    def test_all_same_label_and_center_are_undefined(self) -> None:
        manifest = _make_manifest(
            n=4,
            labels=["A", "A", "A", "A"],
            centers=["C1", "C1", "C1", "C1"],
        )
        features = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=float)

        result = _compute_croma(
            features=features, manifest=manifest, evaluation_design="dataset_wide", m=1
        )

        assert result.undefined_frac == pytest.approx(1.0)
        assert result.sample_values.shape[0] == 0
        assert result.sample_values_aligned.shape == (len(manifest),)
        assert result.occurrence_defined_mask.shape == (len(manifest),)
        assert result.occurrence_defined_mask.tolist() == [False] * len(manifest)
        assert np.isnan(result.sample_values_aligned).all()

    def test_unresolved_rows_at_cap_warn(self, caplog: pytest.LogCaptureFixture) -> None:
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

        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="dataset_wide",
            m=1,
            start_k=1,
            k_growth_factor=1.5,
        )

        assert result.k_final == 3
        assert result.undefined_frac > 0.0
        message = _croma_warning(caplog)
        assert "dataset 'toy' (dataset_wide)" in message
        # The cause here really is the search: it ran out of radius before finding both
        # types. The message must say so, and must not offer the other cause as well.
        assert "the neighbour search could not find 1 SO and 1 OS neighbor(s)" in message
        assert "denominator" not in message

    def test_a_collapsed_embedding_does_not_blame_the_neighbour_search(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every distance zero: the search resolves completely, and every sample is NaN.

        The two causes of a NaN sample come apart here. The search finds its ``m`` SO and
        ``m`` OS neighbours for every sample -- nothing about it failed -- and CRoMa is
        undefined anyway, because ``d_OS + d_SO`` is zero. Reading the cause off the NaN
        output cannot tell these apart, and blaming the search sends the reader to grow
        ``start_k`` on an embedding whose radius was never the problem.
        """
        features, manifest = constant_embedding()

        result = CRoMa.compute(
            features,
            manifest,
            confounder_column="scanner_vendor",
            evaluation_design="dataset_wide",
            m=1,
        )

        assert result.undefined_frac == pytest.approx(1.0)
        message = _croma_warning(caplog)
        assert "could not find" not in message
        assert "denominator" in message
        assert "collapsed" in message

    def test_undefined_causes_account_for_every_undefined_sample(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Whichever causes are reported, together they must add up to ``undefined_frac``.

        A cause breakdown that does not close is a third way to misreport: it would leave
        some NaN samples silently unaccounted for.
        """
        features, manifest = constant_embedding()

        result = CRoMa.compute(
            features,
            manifest,
            confounder_column="scanner_vendor",
            evaluation_design="dataset_wide",
            m=1,
        )

        n_undefined = int(round(result.undefined_frac * len(manifest)))
        counted = sum(int(n) for n in re.findall(r"(\d+) where ", _croma_warning(caplog)))
        assert counted == n_undefined == len(manifest)

    def test_start_k_is_clamped(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="dataset_wide",
            m=1,
            start_k=200,
        )
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

        monkeypatch.setattr(croma_mod, "NearestNeighbors", _FakeNN)
        monkeypatch.setattr(croma_mod, "_scan_typed_neighbors_for_query_rows", _never_define)

        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="dataset_wide",
            m=1,
            start_k=2,
            k_growth_factor=1.5,
        )

        assert result.k_start == 2
        assert result.k_final == 9
        assert result.retries == 4

    def test_queries_only_unresolved_samples_over_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        monkeypatch.setattr(croma_mod, "NearestNeighbors", _FakeNN)
        monkeypatch.setattr(
            croma_mod, "_scan_typed_neighbors_for_query_rows", _define_half_then_all
        )

        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="dataset_wide",
            m=1,
            start_k=1,
            k_growth_factor=1.5,
        )

        assert len(query_sizes) >= 2
        assert query_sizes[1] < query_sizes[0]

    def test_invalid_evaluation_design_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="evaluation_design"):
            _compute_croma(features=features, manifest=manifest, evaluation_design="auto", m=1)

    def test_paired_requires_subset_metadata(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="subset"):
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="paired_2x2",
                m=1,
            )

    def test_m_zero_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="m must be >= 1"):
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="dataset_wide",
                m=0,
            )

    def test_growth_factor_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="k_growth_factor"):
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="dataset_wide",
                m=1,
                k_growth_factor=1.0,
            )

    def test_start_k_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="start_k"):
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="dataset_wide",
                m=1,
                start_k=0,
            )

    def test_api_alias(self) -> None:
        assert CRoMa is CrossConfounderRobustnessMargin

    def test_manual_kmax_not_supported(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(TypeError):
            _compute_croma(features=features, manifest=manifest, evaluation_design="dataset_wide", m=1, kmax=3)  # type: ignore[call-arg]
