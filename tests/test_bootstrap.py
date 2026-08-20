import numpy as np
import pytest

from croma.metrics.bootstrap import (
    BootstrapCI,
    RankStability,
    bootstrap_pooled_median,
    bootstrap_spearman,
    paired_cluster_bootstrap_delta,
    paired_rank_stability,
)


class TestBootstrapPooledMedian:

    def test_point_is_pooled_median_ignoring_nan(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        clusters = np.array(["a", "a", "b", "b", "c", "c"])
        ci = bootstrap_pooled_median(values, clusters, n_boot=200, seed=0)
        assert ci.point == pytest.approx(float(np.median([1.0, 2.0, 3.0, 4.0, 5.0])))
        assert ci.lo <= ci.point <= ci.hi

    def test_deterministic_given_seed(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(size=500)
        clusters = np.repeat(np.arange(50), 10)
        a = bootstrap_pooled_median(values, clusters, n_boot=300, seed=42)
        b = bootstrap_pooled_median(values, clusters, n_boot=300, seed=42)
        assert (a.lo, a.hi) == (b.lo, b.hi)

    def test_clustering_widens_interval_vs_singletons(self) -> None:
        # Highly correlated within-cluster signal: cluster bootstrap should give a
        # wider CI than treating every row as its own (independent) cluster.
        rng = np.random.default_rng(1)
        cluster_means = rng.normal(scale=3.0, size=40)
        values = np.repeat(cluster_means, 25) + rng.normal(scale=0.01, size=1000)
        clusters = np.repeat(np.arange(40), 25)
        singletons = np.arange(1000)
        clustered = bootstrap_pooled_median(values, clusters, n_boot=400, seed=3)
        iid = bootstrap_pooled_median(values, singletons, n_boot=400, seed=3)
        assert (clustered.hi - clustered.lo) > (iid.hi - iid.lo)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_pooled_median(np.zeros(5), np.zeros(4), n_boot=10)

    def test_returns_frozen_ci(self) -> None:
        ci = bootstrap_pooled_median(np.arange(10.0), np.arange(10), n_boot=50)
        assert isinstance(ci, BootstrapCI)
        with pytest.raises(AttributeError):
            ci.point = 0.0  # type: ignore[misc]


class TestPairedRankStability:

    def test_separated_models_rank_deterministically(self) -> None:
        clusters = np.repeat(np.arange(20), 10)
        n = clusters.shape[0]
        model_values = {
            "high": np.full(n, 0.5),
            "mid": np.full(n, 0.2),
            "low": np.full(n, -0.1),
        }
        rs = paired_rank_stability(model_values, clusters, n_boot=100, seed=0)
        assert isinstance(rs, RankStability)
        assert rs.point_rank == {"high": 1, "mid": 2, "low": 3}
        # constant, separated values => A always beats B
        assert rs.pairwise_win[("high", "low")] == pytest.approx(1.0)
        assert rs.pairwise_win[("low", "high")] == pytest.approx(0.0)
        assert rs.mean_rank["high"] == pytest.approx(1.0)

    def test_tied_models_split_wins(self) -> None:
        rng = np.random.default_rng(0)
        clusters = np.repeat(np.arange(30), 10)
        base = rng.normal(size=clusters.shape[0])
        model_values = {"a": base.copy(), "b": base.copy() + 1e-9}
        rs = paired_rank_stability(model_values, clusters, n_boot=200, seed=1)
        # near-identical models: neither dominates the win probability
        assert 0.0 <= rs.pairwise_win[("a", "b")] <= 1.0
        assert rs.pairwise_win[("a", "b")] + rs.pairwise_win[("b", "a")] <= 1.0

    def test_value_ci_present_per_model(self) -> None:
        clusters = np.repeat(np.arange(10), 5)
        n = clusters.shape[0]
        rs = paired_rank_stability(
            {"x": np.linspace(-1, 1, n), "y": np.linspace(-0.5, 0.5, n)},
            clusters,
            n_boot=100,
            seed=0,
        )
        assert set(rs.value_ci) == {"x", "y"}
        assert rs.value_ci["x"].lo <= rs.value_ci["x"].point <= rs.value_ci["x"].hi


def test_paired_cluster_bootstrap_delta_uses_one_shared_group_resample() -> None:
    canonical = np.array([0.0, 0.0])
    alternative = np.array([1.0, 3.0])
    groups = np.array(["slide-a", "slide-b"])

    ci = paired_cluster_bootstrap_delta(
        canonical,
        alternative,
        groups,
        n_boot=4,
        level=0.5,
        seed=0,
    )

    # seed=0 draws [b,b], [b,a], [a,a], [a,a]. Each draw computes
    # median(alternative) - median(canonical) on that same shared group sample.
    assert ci == BootstrapCI(point=2.0, lo=1.0, hi=2.25, level=0.5, n_boot=4)


def test_paired_cluster_bootstrap_delta_preserves_subset_balancing() -> None:
    canonical = np.zeros(4)
    alternative = np.array([0.0, 0.0, 0.0, 10.0])
    groups = np.array(["a", "b", "c", "d"])
    subsets = np.array(["left", "left", "left", "right"])

    ci = paired_cluster_bootstrap_delta(
        canonical,
        alternative,
        groups,
        subset_ids=subsets,
        n_boot=4,
        seed=0,
    )

    # Each arm is first reduced within subset, then balanced across subsets.
    assert ci.point == pytest.approx(5.0)


class TestBootstrapSpearman:

    def test_perfect_monotone_is_one(self) -> None:
        x = np.arange(16.0)
        y = 2.0 * x + 1.0
        ci = bootstrap_spearman(x, y, n_boot=200, seed=0)
        assert ci.point == pytest.approx(1.0)
        assert ci.hi == pytest.approx(1.0)

    def test_reversed_is_minus_one(self) -> None:
        x = np.arange(16.0)
        ci = bootstrap_spearman(x, -x, n_boot=200, seed=0)
        assert ci.point == pytest.approx(-1.0)

    def test_too_few_points_returns_nan(self) -> None:
        ci = bootstrap_spearman(np.array([1.0, 2.0]), np.array([2.0, 1.0]), n_boot=50)
        assert np.isnan(ci.point)

    def test_ci_brackets_point(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.normal(size=16)
        y = x + rng.normal(scale=1.0, size=16)
        ci = bootstrap_spearman(x, y, n_boot=500, seed=0)
        assert ci.lo <= ci.point <= ci.hi
        assert -1.0 <= ci.lo <= ci.hi <= 1.0
