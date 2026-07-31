"""Tests for the lower-tail statistics.

The assertions here are deliberately *relational*: each one states a property
that any correct definition of the lower tail must satisfy, rather than
recomputing ``q_alpha``/``ltm_alpha`` with the same formula the implementation
uses. A rewrite of :func:`compute_tail_metrics` with a subtly different notion
of "the worst alpha fraction" has to break one of these relations.
"""

import numpy as np
import pandas as pd
import pytest

from croma import CRoMa
from croma.metrics.tail import TailMetrics, compute_tail_metrics

# Floating-point slack for inequalities that are exact in exact arithmetic.
TOL = 1e-12


def _signed_skewed_sample(size: int = 400) -> np.ndarray:
    """A signed, left-skewed sample standing in for per-sample CRoMa margins."""
    rng = np.random.default_rng(11)
    return rng.lognormal(mean=0.0, sigma=1.5, size=size) - 2.0


@pytest.fixture
def toy_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "scanner_vendor": ["V1", "V1", "V2", "V2", "V1", "V1", "V2", "V2"],
            "group_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )


@pytest.fixture
def toy_features() -> np.ndarray:
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


class TestComputeTailMetrics:

    def test_ltm_is_non_decreasing_in_alpha(self) -> None:
        # Widening the tail can only admit values at or above the current
        # threshold, so the tail mean cannot fall as alpha grows.
        rng = np.random.default_rng(20260727)
        values = rng.normal(loc=0.2, scale=1.0, size=500)
        alphas = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

        results = [compute_tail_metrics(values, alpha=a) for a in alphas]

        for lower, higher in zip(results, results[1:]):
            assert lower.ltm_alpha <= higher.ltm_alpha + TOL
            assert lower.q_alpha <= higher.q_alpha + TOL
            assert lower.n_tail_samples <= higher.n_tail_samples

    def test_ltm_at_alpha_one_is_the_mean_of_the_finite_values(self) -> None:
        # At alpha = 1 the whole distribution is the tail.
        values = np.array([3.0, -1.0, np.nan, 7.5, 0.25, np.inf, -4.0])
        finite = np.array([3.0, -1.0, 7.5, 0.25, -4.0])

        result = compute_tail_metrics(values, alpha=1.0)

        assert result.ltm_alpha == pytest.approx(float(finite.mean()))
        assert result.n_tail_samples == len(finite)

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.25, 0.50])
    def test_ltm_at_most_quantile_at_most_median(self, alpha: float) -> None:
        # Every tail member sits at or below q_alpha, and for alpha <= 0.5 the
        # alpha-quantile sits at or below the median. The sample is signed and
        # skewed, like a per-sample CRoMa margin.
        values = _signed_skewed_sample()
        median = float(np.median(values))

        result = compute_tail_metrics(values, alpha=alpha)

        assert result.ltm_alpha <= result.q_alpha + TOL
        assert result.q_alpha <= median + TOL

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.25, 0.50])
    def test_tail_holds_about_the_worst_alpha_fraction(self, alpha: float) -> None:
        # The tail is the worst alpha fraction of the sample, so its size is
        # alpha * n up to the granularity of a single order statistic.
        values = _signed_skewed_sample()

        result = compute_tail_metrics(values, alpha=alpha)

        assert abs(result.n_tail_samples - alpha * len(values)) <= 1
        # It must hold the smallest value and leave out the largest.
        assert result.q_alpha >= float(values.min())
        assert result.q_alpha < float(values.max())

    def test_all_equal_collapses_the_tail_onto_the_median(self) -> None:
        values = np.full(20, 5.0)

        result = compute_tail_metrics(values, alpha=0.10)

        median = float(np.median(values))
        assert result.q_alpha == pytest.approx(median)
        assert result.ltm_alpha == pytest.approx(median)
        assert result.n_tail_samples == len(values)

    def test_empty_array(self) -> None:
        result = compute_tail_metrics(np.array([]), alpha=0.10)

        assert result.alpha == 0.10
        assert np.isnan(result.q_alpha)
        assert np.isnan(result.ltm_alpha)
        assert result.n_tail_samples == 0

    def test_all_nan_behaves_like_the_empty_case(self) -> None:
        all_nan = compute_tail_metrics(np.full(8, np.nan), alpha=0.10)
        empty = compute_tail_metrics(np.array([]), alpha=0.10)

        assert np.isnan(all_nan.q_alpha)
        assert np.isnan(all_nan.ltm_alpha)
        assert all_nan.n_tail_samples == empty.n_tail_samples == 0
        assert all_nan.alpha == empty.alpha
        # NaN != NaN, so the sentinel fields are compared through isnan above.
        assert np.isnan(empty.q_alpha) and np.isnan(empty.ltm_alpha)

    def test_non_finite_values_are_dropped_not_ranked(self) -> None:
        # Filtering must be equivalent to never having seen the bad entries.
        finite = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        padded = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.inf, 5.0, -np.inf])

        assert compute_tail_metrics(padded, alpha=0.20) == compute_tail_metrics(finite, alpha=0.20)

    def test_returns_frozen_dataclass(self) -> None:
        result = compute_tail_metrics(np.array([1.0, 2.0, 3.0]))
        assert isinstance(result, TailMetrics)
        with pytest.raises(AttributeError):
            result.alpha = 0.5  # type: ignore[misc]


class TestCRoMaTailIntegration:

    @pytest.mark.parametrize("alpha", [0.10, 0.25])
    def test_result_tail_fields_satisfy_the_ordering(
        self,
        toy_manifest: pd.DataFrame,
        toy_features: np.ndarray,
        alpha: float,
    ) -> None:
        # The tail reported on the result object must obey the same relations
        # as the free function; a mis-wiring between the two shows up here.
        result = CRoMa.compute(
            features=toy_features,
            manifest=toy_manifest,
            confounder_column="scanner_vendor",
            evaluation_design="all",
            m=1,
            alpha=alpha,
        )

        finite = result.sample_values[np.isfinite(result.sample_values)]
        assert len(finite) > 0
        median = float(np.median(finite))

        assert result.alpha == alpha
        assert np.isfinite(result.q_alpha)
        assert np.isfinite(result.ltm_alpha)
        assert result.ltm_alpha <= result.q_alpha + TOL
        assert result.q_alpha <= median + TOL
        assert result.sample_values_aligned.shape == (len(toy_manifest),)

    def test_result_ltm_is_non_decreasing_in_alpha(
        self,
        toy_manifest: pd.DataFrame,
        toy_features: np.ndarray,
    ) -> None:
        ltms = []
        for alpha in [0.10, 0.25, 0.50, 1.00]:
            result = CRoMa.compute(
                features=toy_features,
                manifest=toy_manifest,
                confounder_column="scanner_vendor",
                evaluation_design="all",
                m=1,
                alpha=alpha,
            )
            assert result.alpha == alpha
            ltms.append(result.ltm_alpha)

        for lower, higher in zip(ltms, ltms[1:]):
            assert lower <= higher + TOL
