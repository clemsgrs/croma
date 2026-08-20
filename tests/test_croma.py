import inspect

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa
import croma.metrics.croma as croma_mod
from croma.metrics.croma import (
    CROMA_HEADLINE_M,
    CrossConfounderRobustnessMargin,
    _compute_sample_croma,
    _confounder_dominant_fraction,
    _sample_croma_with_causes,
)
from croma.types import CRoMaResult
from metric_harness import constant_embedding, contested
from support_schema import RETIRED_AGGREGATE_FIELD


def _make_manifest(
    n: int,
    labels: list[str],
    centers: list[str],
    group_ids: list[str] | None = None,
    subset: str | None = None,
) -> pd.DataFrame:
    if group_ids is None:
        group_ids = [f"slide-{i}" for i in range(n)]
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/tmp/{i}.png" for i in range(n)],
            "label": labels,
            "scanner_vendor": centers,
            "group_id": group_ids,
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


def _paired_subsets_sharing_one_sample() -> tuple[np.ndarray, pd.DataFrame]:
    """Two complete 2x2 subsets in which one source sample occurs in both.

    ``p0`` clusters by confounder (every row's impostor is nearer than its biological
    match) and ``p1`` clusters by label (the reverse), so the two subsets contribute eight
    negative and eight positive occurrences. Sample ``s0`` carries the same feature vector
    in both, which is how a paired manifest represents one sample scored twice.
    """
    labels = ["A", "B", "A", "B"] * 2
    centers = ["C1", "C1", "C2", "C2"] * 2
    manifest = _make_manifest(n=16, labels=labels * 2, centers=centers * 2)
    manifest["sample_id"] = (
        ["s0"] + [f"s{i}" for i in range(1, 8)] + ["s0"] + [f"s{i}" for i in range(8, 15)]
    )
    manifest["group_id"] = list(manifest["sample_id"])
    manifest["subset"] = ["p0"] * 8 + ["p1"] * 8

    shared = [1.00, 0.00, 0.05]
    # p0: geometry follows the confounder -- C1 rows near axis 0, C2 rows near axis 1.
    confounder_clustered = [
        shared,
        [0.98, 0.02, 0.00],
        [0.02, 0.98, 0.00],
        [0.00, 1.00, 0.00],
        [0.97, 0.03, 0.00],
        [0.96, 0.04, 0.00],
        [0.03, 0.97, 0.00],
        [0.01, 0.99, 0.00],
    ]
    # p1: the same rows, clustered by label instead -- A rows near axis 0, B near axis 1.
    label_clustered = [
        shared,
        [0.02, 0.98, 0.00],
        [0.98, 0.02, 0.05],
        [0.00, 1.00, 0.00],
        [0.97, 0.03, 0.04],
        [0.03, 0.97, 0.00],
        [0.96, 0.04, 0.03],
        [0.01, 0.99, 0.00],
    ]
    features = np.array(confounder_clustered + label_clustered, dtype=float)
    return features, manifest


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

        The masks have to be disjoint and cover the NaN set exactly, because the
        total-support error reports counts taken from them: any overlap double-counts a
        sample and any gap leaves one unscoreable for a reason nobody is told.
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

    def test_result_construction_rejects_legacy_positional_coverage(self) -> None:
        positional = (
            "toy",
            1,
            0.1,
            0.0,
            1,
            np.asarray([0.1]),
            np.asarray([0.1]),
            np.asarray([0.1]),
        )

        with pytest.raises(TypeError):
            CRoMaResult(*positional, np.asarray([True]), 0.0)

        result = CRoMaResult(*positional, evaluation_design="all")
        assert result.evaluation_design == "all"

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
            evaluation_design="all",
            m=[1, 2],
        )
        assert isinstance(by_m, dict)

        single_m1 = _compute_croma(
            features=features, manifest=manifest, evaluation_design="all", m=1
        )
        single_m2 = _compute_croma(
            features=features, manifest=manifest, evaluation_design="all", m=2
        )

        assert by_m[1].value == pytest.approx(single_m1.value)
        assert by_m[1].std == pytest.approx(single_m1.std)
        assert by_m[1].q_alpha == pytest.approx(single_m1.q_alpha)
        assert by_m[1].ltm_alpha == pytest.approx(single_m1.ltm_alpha)

        assert by_m[2].value == pytest.approx(single_m2.value)
        assert by_m[2].std == pytest.approx(single_m2.std)
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
            evaluation_design="all",
            m=[1, 2],
        )
        assert isinstance(by_m, dict)

        assert set(by_m) == {1, 2}
        assert calls["n"] == 1

    @pytest.mark.parametrize("evaluation_design", ["paired_2x2", "all"])
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
        assert not hasattr(result, "occurrence_defined_mask")
        assert not hasattr(result, RETIRED_AGGREGATE_FIELD)

    def test_so_closer_yields_croma_above_zero_all_rows(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = _compute_croma(features=features, manifest=manifest, evaluation_design="all", m=1)
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
        result = _compute_croma(features=features, manifest=manifest, evaluation_design="all", m=1)
        assert result.value < 0.0

    def test_unresolved_rows_at_cap_raise_with_evaluation_context_and_cause(self) -> None:
        labels = ["A", "A", "B", "B"]
        centers = ["C1", "C2", "C1", "C2"]
        slides = ["s1", "s1", "s2", "s2"]
        manifest = _make_manifest(n=4, labels=labels, centers=centers, group_ids=slides)
        features = np.array(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
            ],
            dtype=float,
        )

        with pytest.raises(RuntimeError) as exc_info:
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="all",
                m=1,
                start_k=1,
                k_growth_factor=1.5,
            )

        message = str(exc_info.value)
        assert "dataset 'toy' (all)" in message
        assert "subset 'dataset'" in message
        assert "after adaptive neighbour search reached k=3" in message
        assert "could not find 1 SO and 1 OS neighbour(s)" in message
        assert "denominator" not in message

    def test_a_collapsed_embedding_raises_with_zero_denominator_cause(self) -> None:
        """Every distance zero: the search resolves completely, and every sample is NaN.

        The two causes of a NaN sample come apart here. The search finds its ``m`` SO and
        ``m`` OS neighbours for every sample -- nothing about it failed -- and CRoMa is
        undefined anyway, because ``d_OS + d_SO`` is zero. Reading the cause off the NaN
        output cannot tell these apart, and blaming the search sends the reader to grow
        ``start_k`` on an embedding whose radius was never the problem.
        """
        features, manifest = constant_embedding()

        with pytest.raises(RuntimeError) as exc_info:
            CRoMa.compute(
                features,
                manifest,
                confounder_column="scanner_vendor",
                evaluation_design="all",
                m=1,
            )

        message = str(exc_info.value)
        assert "dataset 'toy' (all)" in message
        assert "subset 'dataset'" in message
        assert "could not find" not in message
        assert "zero margin denominator (d_OS + d_SO = 0)" in message
        assert "collapsed embedding" in message

    def test_fully_scoreable_values_alignment_and_tail_statistics_are_preserved(self) -> None:
        features, manifest = contested()

        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="all",
            m=1,
            alpha=0.25,
        )

        assert result.value == pytest.approx(0.0, abs=1e-12)
        assert result.sample_values == pytest.approx(np.zeros(len(manifest)), abs=1e-12)
        assert result.sample_values_aligned == pytest.approx(np.zeros(len(manifest)), abs=1e-12)
        assert result.q_alpha == pytest.approx(0.0, abs=1e-12)
        assert result.ltm_alpha == pytest.approx(0.0, abs=1e-12)
        assert result.f0 == pytest.approx(1.0)

    def test_start_k_is_clamped(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="all",
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

        with pytest.raises(RuntimeError, match="adaptive neighbour search reached k=9"):
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="all",
                m=1,
                start_k=2,
                k_growth_factor=1.5,
            )

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
            resolved_indices = query_indices[out]
            kwargs["so_dists"][resolved_indices, :] = 0.1
            kwargs["os_dists"][resolved_indices, :] = 0.2
            return out

        monkeypatch.setattr(croma_mod, "NearestNeighbors", _FakeNN)
        monkeypatch.setattr(
            croma_mod, "_scan_typed_neighbors_for_query_rows", _define_half_then_all
        )

        result = _compute_croma(
            features=features,
            manifest=manifest,
            evaluation_design="all",
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
                evaluation_design="all",
                m=0,
            )

    def test_growth_factor_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="k_growth_factor"):
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="all",
                m=1,
                k_growth_factor=1.0,
            )

    def test_start_k_rejected(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(ValueError, match="start_k"):
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="all",
                m=1,
                start_k=0,
            )

    def test_api_alias(self) -> None:
        assert CRoMa is CrossConfounderRobustnessMargin

    def test_manual_kmax_not_supported(self) -> None:
        features, manifest = _toy_features_so_closer()
        with pytest.raises(TypeError):
            _compute_croma(features=features, manifest=manifest, evaluation_design="all", m=1, kmax=3)  # type: ignore[call-arg]


class TestConfounderDominantFraction:
    """``F(0)``: the empirical CDF of the per-sample margin at zero.

    The boundary is ``<= 0`` -- an exactly contested sample is counted as
    confounder-dominant -- and the denominator is the *defined* values only, the same
    distribution ``q_alpha`` and ``ltm_alpha`` are read off.
    """

    def test_zero_counts_as_confounder_dominant_and_nan_leaves_the_denominator(self) -> None:
        values = np.array([-0.5, 0.0, 0.25, np.nan], dtype=float)
        assert _confounder_dominant_fraction(values) == pytest.approx(2.0 / 3.0)

    def test_no_defined_values_is_nan(self) -> None:
        assert np.isnan(_confounder_dominant_fraction(np.array([np.nan, np.nan])))
        assert np.isnan(_confounder_dominant_fraction(np.array([], dtype=float)))


class TestCRoMaF0:

    def test_all_counts_every_defined_manifest_sample_once(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = _compute_croma(features=features, manifest=manifest, evaluation_design="all", m=1)

        defined = result.sample_values_aligned
        assert len(defined) == len(manifest)
        assert result.f0 == pytest.approx(float(np.mean(defined <= 0.0)))

    def test_unscoreable_rows_prevent_partial_pooled_and_tail_statistics(self) -> None:
        # Four rows are scoreable, while two lack both required neighbour types. The public
        # call must fail instead of returning pooled CRoMa or tail fields for only four rows.
        manifest = _make_manifest(
            n=6,
            labels=["A", "A", "B", "B", "C", "C"],
            centers=["C1", "C2", "C1", "C2", "C3", "C3"],
        )
        features = np.array(
            [
                [1.00, 0.00],
                [0.00, 1.00],
                [0.95, 0.05],
                [0.05, 0.95],
                [-1.00, 0.00],
                [0.00, -1.00],
            ],
            dtype=float,
        )
        with pytest.raises(RuntimeError) as exc_info:
            _compute_croma(
                features=features,
                manifest=manifest,
                evaluation_design="all",
                m=1,
            )

        message = str(exc_info.value)
        assert "2/6 sample(s) could not find 1 SO and 1 OS neighbour(s)" in message
        assert "no pooled or tail statistics were computed" in message

    def test_paired_counts_a_repeated_source_sample_once_per_occurrence(self) -> None:
        """One sample scored in two subsets contributes an occurrence to each.

        ``s0`` appears in both subsets, with the same feature vector and different
        neighbourhoods: confounder-dominant in ``p0`` and biology-dominant in ``p1``. The
        denominator is the 16 defined occurrences, not the 15 distinct samples, so
        ``F(0)`` is exactly ``8/16`` and not ``8/15``.
        """
        features, manifest = _paired_subsets_sharing_one_sample()

        result = _compute_croma(
            features=features, manifest=manifest, evaluation_design="paired_2x2", m=1
        )

        assert result.evaluation_unit == "occurrence"
        defined = result.sample_values_aligned
        assert len(defined) == 16
        assert manifest["sample_id"].nunique() == 15
        assert result.f0 == pytest.approx(0.5)
        assert result.f0 == pytest.approx(float(np.mean(defined <= 0.0)))

    def test_f0_does_not_disturb_the_existing_statistics(self) -> None:
        features, manifest = _toy_features_so_closer()
        result = _compute_croma(features=features, manifest=manifest, evaluation_design="all", m=1)

        finite = result.sample_values
        assert result.value == pytest.approx(float(np.median(finite)))
        assert result.q_alpha == pytest.approx(float(np.percentile(finite, 10)))
