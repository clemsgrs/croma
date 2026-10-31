import itertools

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

from croma.metrics.bootstrap import (
    BootstrapCI,
    RankStability,
    benjamini_hochberg,
    bootstrap_pooled_median,
    bootstrap_spearman,
    cluster_bootstrap_draws,
    encoder_bootstrap_draws,
    paired_cluster_bootstrap_delta,
    paired_rank_stability,
    partial_spearman_inference,
    shared_draw_tail_statistics,
    spearman_inference,
)
from croma.metrics.bootstrap import _residualise, _two_sided_p
from croma.metrics.tail import compute_tail_metrics


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

    def test_strata_keep_the_design_balanced(self) -> None:
        # Five single-tile slides per stratum, all 0 in one and all 1 in the other. A
        # balanced replicate always holds five of each, so its pooled median is 0.5.
        values = np.repeat([0.0, 1.0], 5)
        clusters = np.arange(10)
        strata = np.repeat(["normal", "tumor"], 5)
        stratified = bootstrap_pooled_median(values, clusters, strata=strata, n_boot=300, seed=0)
        plain = bootstrap_pooled_median(values, clusters, n_boot=300, seed=0)
        assert (stratified.lo, stratified.point, stratified.hi) == (0.5, 0.5, 0.5)
        assert plain.lo < 0.5 < plain.hi

    def test_one_stratum_is_the_unstratified_draw(self) -> None:
        values = TestUnstratifiedDrawsAreUnchanged.VALUES
        clusters = TestUnstratifiedDrawsAreUnchanged.CLUSTERS
        one = np.full(values.shape, "all")
        assert bootstrap_pooled_median(
            values, clusters, strata=one, n_boot=50, seed=11
        ) == bootstrap_pooled_median(values, clusters, n_boot=50, seed=11)

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

    def test_strata_keep_the_design_balanced(self) -> None:
        # "a" scores 0 on every normal slide and 1 on every tumor slide: its median is 0.5
        # in any balanced replicate, so it beats the constant 0.4 of "b" every time. An
        # unstratified draw that over-samples normal slides hands the lead to "b".
        panel = {"a": np.repeat([0.0, 1.0], 5), "b": np.full(10, 0.4)}
        clusters = np.arange(10)
        strata = np.repeat(["normal", "tumor"], 5)
        stratified = paired_rank_stability(panel, clusters, strata=strata, n_boot=300, seed=0)
        plain = paired_rank_stability(panel, clusters, n_boot=300, seed=0)
        assert stratified.pairwise_win[("a", "b")] == 1.0
        assert (stratified.rank_lo["a"], stratified.rank_hi["a"]) == (1, 1)
        assert plain.pairwise_win[("a", "b")] < 1.0

    def test_one_stratum_is_the_unstratified_draw(self) -> None:
        values = TestUnstratifiedDrawsAreUnchanged.VALUES
        clusters = TestUnstratifiedDrawsAreUnchanged.CLUSTERS
        panel = {"a": values, "b": values[::-1].copy()}
        one = np.full(values.shape, "all")
        assert paired_rank_stability(
            panel, clusters, strata=one, n_boot=50, seed=11
        ) == paired_rank_stability(panel, clusters, n_boot=50, seed=11)

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


def _toy_stratified_design() -> tuple[np.ndarray, np.ndarray]:
    """Slides of unequal size nested in class x centre strata of unequal cluster counts.

    Rows are shuffled so no helper can rely on a cluster's rows being contiguous.
    """
    layout = {  # stratum -> sizes of its slides
        "tumor|rumc": [3, 1],
        "tumor|umcu": [2, 2, 5],
        "normal|rumc": [1, 4, 2, 3],
    }
    clusters, strata = [], []
    for stratum, sizes in layout.items():
        for i, size in enumerate(sizes):
            clusters += [f"{stratum}/slide{i}"] * size
            strata += [stratum] * size
    order = np.random.default_rng(99).permutation(len(clusters))
    return np.array(clusters)[order], np.array(strata)[order]


def _drawn_clusters_per_stratum(
    rows: np.ndarray, clusters: np.ndarray, strata: np.ndarray
) -> dict[str, int]:
    """Count drawn clusters from a replicate's rows: each draw of a cluster repeats its
    first row exactly once."""
    counts: dict[str, int] = {}
    for cluster in np.unique(clusters):
        first_row = int(np.flatnonzero(clusters == cluster)[0])
        stratum = str(strata[first_row])
        counts[stratum] = counts.get(stratum, 0) + int(np.sum(rows == first_row))
    return counts


class TestClusterBootstrapDraws:

    def test_every_replicate_keeps_each_stratums_cluster_count(self) -> None:
        clusters, strata = _toy_stratified_design()
        expected = {"tumor|rumc": 2, "tumor|umcu": 3, "normal|rumc": 4}
        replicates = list(cluster_bootstrap_draws(clusters, strata=strata, n_boot=200, seed=3))
        assert len(replicates) == 200
        for rows in replicates:
            assert _drawn_clusters_per_stratum(rows, clusters, strata) == expected
            # whole clusters are drawn: a stratum's rows are only ever its own clusters'
            assert set(strata[rows]) <= set(expected)

    def test_a_cluster_spanning_two_strata_is_rejected_at_the_call(self) -> None:
        clusters = np.array(["case1", "case1", "case2", "case2"])
        strata = np.array(["UKK", "WNS", "WNS", "WNS"])
        # rejected before any replicate is requested, not at the first iteration
        with pytest.raises(ValueError, match="case1"):
            cluster_bootstrap_draws(clusters, strata=strata, n_boot=10)

    def test_strata_must_label_every_row(self) -> None:
        with pytest.raises(ValueError, match="strata"):
            cluster_bootstrap_draws(np.array(["a", "b", "c"]), strata=np.array(["x", "y"]))


def _toy_panel() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Four encoders scored on the same rows of the toy stratified design."""
    clusters, strata = _toy_stratified_design()
    rng = np.random.default_rng(5)
    panel = {
        name: rng.normal(loc, 0.3, size=clusters.size)
        for name, loc in [("enc-a", 0.15), ("enc-b", 0.05), ("enc-c", -0.02), ("enc-d", 0.1)]
    }
    panel["enc-c"][[2, 7]] = np.nan  # undefined samples are dropped, as in the headline
    return panel, clusters, strata


def _lowest_decile_mean(values: np.ndarray) -> float:
    """LTM10 by its definition: the mean of the values at or below the 10th percentile."""
    q = np.percentile(values, 10)
    return float(values[values <= q].mean())


class TestSharedDrawTailStatistics:

    def test_points_are_the_headline_statistics(self) -> None:
        panel, clusters, strata = _toy_panel()
        tail = shared_draw_tail_statistics(panel, clusters, strata=strata, n_boot=20, seed=1)
        for name, values in panel.items():
            finite = values[np.isfinite(values)]
            assert tail.ci["median"][name].point == pytest.approx(np.median(finite))
            assert tail.ci["f0"][name].point == pytest.approx(np.mean(finite <= 0.0))
            assert tail.ci["ltm"][name].point == pytest.approx(
                compute_tail_metrics(values, alpha=0.10).ltm_alpha
            )

    def test_every_encoder_is_scored_on_one_shared_draw_per_replicate(self) -> None:
        panel, clusters, strata = _toy_panel()
        tail = shared_draw_tail_statistics(panel, clusters, strata=strata, n_boot=60, seed=4)
        draws = cluster_bootstrap_draws(clusters, strata=strata, n_boot=60, seed=4)
        for b, rows in enumerate(draws):
            for j, name in enumerate(tail.models):
                drawn = panel[name][rows]
                drawn = drawn[np.isfinite(drawn)]
                assert tail.replicates["median"][b, j] == pytest.approx(np.median(drawn))
                assert tail.replicates["f0"][b, j] == pytest.approx(np.mean(drawn <= 0.0))
                assert tail.replicates["ltm"][b, j] == pytest.approx(_lowest_decile_mean(drawn))

    def test_cis_are_percentile_intervals_of_the_replicates(self) -> None:
        panel, clusters, strata = _toy_panel()
        tail = shared_draw_tail_statistics(
            panel, clusters, strata=strata, n_boot=80, level=0.9, seed=2
        )
        for statistic in ("median", "f0", "ltm"):
            for j, name in enumerate(tail.models):
                lo, hi = np.percentile(tail.replicates[statistic][:, j], [5.0, 95.0])
                ci = tail.ci[statistic][name]
                assert (ci.lo, ci.hi, ci.level, ci.n_boot) == pytest.approx((lo, hi, 0.9, 80))

    def test_median_ranks_and_wins_agree_with_a_direct_computation(self) -> None:
        panel, clusters, strata = _toy_panel()
        tail = shared_draw_tail_statistics(panel, clusters, strata=strata, n_boot=150, seed=6)
        models, medians = tail.models, tail.replicates["median"]
        n = len(models)
        # rank 1 is the highest median of that replicate (continuous toy values: no ties)
        direct = np.array(
            [[1 + sum(row[k] > row[j] for k in range(n)) for j in range(n)] for row in medians]
        )
        np.testing.assert_array_equal(tail.rank_replicates, direct)

        ranks = tail.median_rank
        point_order = sorted(models, key=lambda m: tail.ci["median"][m].point, reverse=True)
        assert ranks.point_rank == {m: i + 1 for i, m in enumerate(point_order)}
        for j, m in enumerate(models):
            assert ranks.mean_rank[m] == pytest.approx(direct[:, j].mean())
            assert ranks.rank_lo[m] == int(np.percentile(direct[:, j], 2.5))
            assert ranks.rank_hi[m] == int(np.percentile(direct[:, j], 97.5))
            assert ranks.value_ci[m] == tail.ci["median"][m]
            for k, other in enumerate(models):
                if other != m:
                    expected = np.mean(medians[:, j] > medians[:, k])
                    assert ranks.pairwise_win[(m, other)] == pytest.approx(expected)

    def test_paired_difference_of_a_shifted_encoder_is_exact(self) -> None:
        # The fine-tune scores every sample 0.25 above its parent. On a shared draw its
        # median and lower-tail mean are 0.25 higher in every replicate, so the paired CI
        # collapses onto 0.25; independent draws would scatter it.
        _, clusters, strata = _toy_panel()
        parent = np.random.default_rng(0).normal(size=clusters.size)
        panel = {"parent": parent, "finetune": parent + 0.25}
        tail = shared_draw_tail_statistics(panel, clusters, strata=strata, n_boot=100, seed=0)
        for statistic in ("median", "ltm"):
            diff = tail.difference_ci(statistic, "finetune", "parent")
            assert (diff.point, diff.lo, diff.hi) == pytest.approx((0.25, 0.25, 0.25))
            assert diff.n_boot == 100

    def test_paired_difference_is_the_percentile_interval_of_replicate_differences(
        self,
    ) -> None:
        panel, clusters, strata = _toy_panel()
        tail = shared_draw_tail_statistics(panel, clusters, strata=strata, n_boot=90, seed=3)
        a, b = tail.models.index("enc-a"), tail.models.index("enc-c")
        diff = tail.difference_ci("f0", "enc-a", "enc-c")
        replicate_diff = tail.replicates["f0"][:, a] - tail.replicates["f0"][:, b]
        assert diff.point == pytest.approx(
            tail.ci["f0"]["enc-a"].point - tail.ci["f0"]["enc-c"].point
        )
        assert (diff.lo, diff.hi) == pytest.approx(
            tuple(np.percentile(replicate_diff, [2.5, 97.5]))
        )
        with pytest.raises(ValueError, match="statistic"):
            tail.difference_ci("mean", "enc-a", "enc-c")

    def test_median_ranking_matches_paired_rank_stability(self) -> None:
        panel, clusters, strata = _toy_panel()
        tail = shared_draw_tail_statistics(panel, clusters, strata=strata, n_boot=120, seed=8)
        assert tail.median_rank == paired_rank_stability(
            panel, clusters, strata=strata, n_boot=120, seed=8
        )


def _toy_encoder_metrics(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two noisy, positively associated per-encoder vectors."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return x, x + rng.normal(scale=1.0, size=n)


class TestSpearmanInference:

    def test_rho_and_ci_agree_with_a_direct_encoder_bootstrap(self) -> None:
        x, y = _toy_encoder_metrics(14)
        result = spearman_inference(x, y, n_boot=300, level=0.9, seed=7)
        assert result.rho == pytest.approx(spearmanr(x, y).statistic)
        assert result.n == 14

        draws = encoder_bootstrap_draws(14, n_boot=300, seed=7)
        assert draws.shape == (300, 14)
        direct = np.array([spearmanr(x[idx], y[idx]).statistic for idx in draws])
        np.testing.assert_allclose(result.replicates, direct)
        lo, hi = np.percentile(direct[np.isfinite(direct)], [5.0, 95.0])
        assert (result.ci.point, result.ci.lo, result.ci.hi) == pytest.approx((result.rho, lo, hi))

    def test_exact_permutation_p_on_a_hand_worked_case(self) -> None:
        # n = 4: rho = 1 - sum(d^2) / 10. One adjacent swap gives sum(d^2) = 2, rho = 0.8.
        # |rho| >= 0.8 iff sum(d^2) <= 2 or >= 18: the identity, the 3 adjacent swaps, the
        # reversal and the reversal of each adjacent swap -- 8 of the 24 orderings.
        result = spearman_inference(np.array([1.0, 2, 3, 4]), np.array([1.0, 2, 4, 3]), n_boot=10)
        assert result.rho == pytest.approx(0.8)
        assert result.permutation == "exact"
        assert result.n_permutations == 24
        assert result.p_value == pytest.approx(8 / 24)

    def test_exact_permutation_p_of_a_perfect_ranking(self) -> None:
        # only the identity and the reversal reach |rho| = 1 among the 5! orderings
        result = spearman_inference(np.arange(5.0), np.arange(5.0) ** 2, n_boot=10)
        assert result.rho == pytest.approx(1.0)
        assert (result.permutation, result.n_permutations) == ("exact", 120)
        assert result.p_value == pytest.approx(2 / 120)

    def test_exact_enumeration_stops_at_eight_encoders(self) -> None:
        x8, y8 = _toy_encoder_metrics(8)
        x9, y9 = _toy_encoder_metrics(9)
        eight = spearman_inference(x8, y8, n_boot=10)
        nine = spearman_inference(x9, y9, n_boot=10)
        assert (eight.permutation, eight.n_permutations) == ("exact", 40_320)
        assert (nine.permutation, nine.n_permutations) == ("monte-carlo", 10_000)

    def test_monte_carlo_p_never_reaches_zero(self) -> None:
        # No random ordering of 12 encoders reproduces a perfect ranking in practice, so
        # the count of extreme orderings is 0 and p is (0 + 1) / (10,000 + 1).
        result = spearman_inference(np.arange(12.0), np.arange(12.0), n_boot=10, seed=1)
        assert result.p_value == pytest.approx(1 / 10_001)

    def test_monte_carlo_p_is_seeded_and_close_to_the_exact_p(self) -> None:
        x, y = _toy_encoder_metrics(9, seed=3)
        result = spearman_inference(x, y, n_boot=10, seed=5)
        assert result.p_value == spearman_inference(x, y, n_boot=10, seed=5).p_value
        # exact p from the untied closed form rho = 1 - 6 sum(d^2) / (n (n^2 - 1))
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        orders = np.array(list(itertools.permutations(range(9))), dtype=np.int8)
        rho_null = 1 - 6 * ((rx - ry[orders]) ** 2).sum(axis=1) / (9 * 80)
        exact = np.mean(np.abs(rho_null) >= abs(result.rho) - 1e-12)
        assert result.p_value == pytest.approx(exact, abs=0.015)

    def test_explicit_draws_reproduce_the_seeded_draw_and_leave_p_alone(self) -> None:
        x, y = _toy_encoder_metrics(12, seed=2)
        implicit = spearman_inference(x, y, n_boot=200, seed=9)
        explicit = spearman_inference(
            x, y, seed=9, draws=encoder_bootstrap_draws(12, n_boot=200, seed=9)
        )
        np.testing.assert_array_equal(explicit.replicates, implicit.replicates)
        assert (explicit.ci, explicit.p_value) == (implicit.ci, implicit.p_value)

    def test_one_encoder_draw_shared_between_two_correlations(self) -> None:
        # Head-to-head: rho(a, target) - rho(b, target) on one shared draw. Each replicate
        # of both correlations is computed on the same drawn encoders.
        rng = np.random.default_rng(4)
        target = rng.normal(size=15)
        a = target + rng.normal(scale=0.5, size=15)
        b = target + rng.normal(scale=2.0, size=15)
        draws = encoder_bootstrap_draws(15, n_boot=100, seed=0)
        ra = spearman_inference(a, target, draws=draws)
        rb = spearman_inference(b, target, draws=draws)
        assert ra.ci.n_boot == rb.ci.n_boot == 100
        for replicate, idx in enumerate(draws[:5]):
            difference = spearmanr(a[idx], target[idx]).statistic
            difference -= spearmanr(b[idx], target[idx]).statistic
            assert ra.replicates[replicate] - rb.replicates[replicate] == pytest.approx(difference)

    def test_difference_of_two_correlations_is_paired_on_their_shared_draw(self) -> None:
        # The CI of rho(a, target) - rho(b, target) is the percentile interval of the
        # replicate-by-replicate differences, each replicate re-ranking the same drawn
        # encoders for both correlations.
        rng = np.random.default_rng(4)
        target = rng.normal(size=15)
        a = target + rng.normal(scale=0.5, size=15)
        b = target + rng.normal(scale=2.0, size=15)
        draws = encoder_bootstrap_draws(15, n_boot=400, seed=0)
        ra = spearman_inference(a, target, draws=draws)
        rb = spearman_inference(b, target, draws=draws)

        difference = ra.difference_ci(rb)

        direct = np.array(
            [
                spearmanr(a[i], target[i]).statistic - spearmanr(b[i], target[i]).statistic
                for i in draws
            ]
        )
        lo, hi = np.nanpercentile(direct, [2.5, 97.5])
        point = spearmanr(a, target).statistic - spearmanr(b, target).statistic
        assert difference.point == pytest.approx(point)
        assert (difference.lo, difference.hi) == pytest.approx((lo, hi))
        assert (difference.level, difference.n_boot) == (0.95, 400)

    def test_draws_of_the_same_size_from_different_seeds_are_not_paired(self) -> None:
        # Same n, same replicate count, different encoders drawn: replicate b of one is not
        # replicate b of the other, so no paired difference exists. Accepting it would give
        # a self-difference a wide interval instead of exactly zero.
        x, y = _toy_encoder_metrics(12, seed=2)
        seed0 = spearman_inference(x, y, n_boot=500, seed=0)
        with pytest.raises(ValueError, match="draw"):
            seed0.difference_ci(spearman_inference(x, y, n_boot=500, seed=1))
        itself = seed0.difference_ci(spearman_inference(x, y, n_boot=500, seed=0))
        assert (itself.point, itself.lo, itself.hi) == (0.0, 0.0, 0.0)

    def test_pairing_follows_the_draw_not_the_array_object(self) -> None:
        # A seeded draw and the same draw passed explicitly (in another integer dtype) are one
        # draw; a partial on that draw pairs with a plain correlation on it, too.
        x, y = _toy_encoder_metrics(12, seed=2)
        z = np.random.default_rng(5).normal(size=12)
        draws = encoder_bootstrap_draws(12, n_boot=200, seed=3)
        implicit = spearman_inference(x, y, n_boot=200, seed=3)
        explicit = spearman_inference(x, y, draws=draws.astype(np.int32))
        partial = partial_spearman_inference(x, y, z, draws=draws)
        assert implicit.draws_sha256 == explicit.draws_sha256 == partial.draws_sha256
        assert implicit.difference_ci(explicit).point == 0.0
        implicit.difference_ci(partial)  # paired: no error
        other = partial_spearman_inference(x, y, z, n_boot=200, seed=4)
        assert other.draws_sha256 != partial.draws_sha256
        with pytest.raises(ValueError, match="draw"):
            partial.difference_ci(other)

    def test_differencing_unpaired_inferences_is_rejected(self) -> None:
        x, y = _toy_encoder_metrics(12, seed=2)
        with pytest.raises(ValueError, match="draw"):
            spearman_inference(x, y, n_boot=100).difference_ci(spearman_inference(x, y, n_boot=50))
        with pytest.raises(ValueError, match="level"):
            spearman_inference(x, y, n_boot=50).difference_ci(
                spearman_inference(x, y, n_boot=50, level=0.9)
            )

    def test_malformed_draws_are_rejected(self) -> None:
        x, y = _toy_encoder_metrics(6)
        with pytest.raises(ValueError, match="shape"):
            spearman_inference(x, y, draws=np.zeros((10, 5), dtype=int))
        with pytest.raises(ValueError, match="index"):
            spearman_inference(x, y, draws=np.full((10, 6), 6))

    def test_undefined_inputs_are_rejected_not_dropped(self) -> None:
        x, y = _toy_encoder_metrics(6)
        y[2] = np.nan
        with pytest.raises(ValueError, match="finite"):
            spearman_inference(x, y)
        with pytest.raises(ValueError, match="constant"):
            spearman_inference(x, np.ones(6))
        with pytest.raises(ValueError, match="three"):
            spearman_inference(np.array([1.0, 2.0]), np.array([2.0, 1.0]))


def _closed_form_partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rxy, rxz, ryz = (spearmanr(a, b).statistic for a, b in [(x, y), (x, z), (y, z)])
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))


class TestPartialSpearman:

    @pytest.mark.parametrize("ties", [False, True])
    def test_matches_the_closed_form_on_ranks(self, ties: bool) -> None:
        rng = np.random.default_rng(11)
        z = rng.normal(size=20)
        x = z + rng.normal(size=20)
        y = 0.5 * z + 0.5 * x + rng.normal(size=20)
        if ties:  # coarse values: average ranks, as Spearman's rho uses
            x, y, z = np.round(x), np.round(y), np.round(z)
        result = partial_spearman_inference(x, y, z, n_boot=50)
        assert result.rho == pytest.approx(_closed_form_partial_spearman(x, y, z))
        assert result.ci.point == result.rho

    def test_ci_agrees_with_a_direct_bootstrap_of_the_closed_form(self) -> None:
        rng = np.random.default_rng(12)
        z = rng.normal(size=16)
        x, y = z + rng.normal(size=16), z + rng.normal(size=16)
        result = partial_spearman_inference(x, y, z, n_boot=200, seed=3)
        draws = encoder_bootstrap_draws(16, n_boot=200, seed=3)
        direct = np.array([_closed_form_partial_spearman(x[i], y[i], z[i]) for i in draws])
        np.testing.assert_allclose(result.replicates, direct, atol=1e-12)
        lo, hi = np.percentile(direct, [2.5, 97.5])
        assert (result.ci.lo, result.ci.hi) == pytest.approx((lo, hi))

    def test_freedman_lane_p_agrees_with_a_direct_enumeration(self) -> None:
        # Freedman-Lane, step by step: fit rank(y) on [1, rank(z)], permute its
        # residuals, add them back to the fitted values, refit, and correlate the new
        # residual with rank(x)'s residual. n = 6, so all 720 orderings are enumerated.
        rng = np.random.default_rng(21)
        z = rng.normal(size=6)
        x, y = z + rng.normal(size=6), z + rng.normal(size=6)
        result = partial_spearman_inference(x, y, z, n_boot=10)
        assert (result.permutation, result.n_permutations) == ("exact", 720)

        design = np.column_stack([np.ones(6), rankdata(z)])

        def residual(v: np.ndarray) -> np.ndarray:
            return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

        res_x, ry = residual(rankdata(x)), rankdata(y)
        fitted, res_y = ry - residual(ry), residual(ry)
        observed = np.corrcoef(res_x, res_y)[0, 1]
        null = [
            np.corrcoef(res_x, residual(fitted + res_y[list(order)]))[0, 1]
            for order in itertools.permutations(range(6))
        ]
        expected = np.mean(np.abs(null) >= abs(observed) - 1e-12)
        assert result.rho == pytest.approx(observed)
        assert result.p_value == pytest.approx(expected)

    def test_undefined_permutations_leave_the_exact_p_denominator(self) -> None:
        # rank(y) is uncorrelated with z = 0..4, so rho = 0 and every defined ordering is
        # at least as extreme; 2 of the 120 permuted residuals lie in span(1, rank z),
        # re-residualise to zero and have no statistic. p is 118 / 118, not 118 / 120.
        result = partial_spearman_inference([1, 0, 4, 3, 2], [0, 4, 3, 2, 1], np.arange(5))
        assert result.rho == pytest.approx(0.0, abs=1e-12)
        assert (result.permutation, result.n_permutations) == ("exact", 118)
        assert result.p_value == 1.0

    def test_undefined_permutations_with_tied_ranks(self) -> None:
        # 4 of the 24 orderings are undefined: p = 12 / 20, not 12 / 24
        result = partial_spearman_inference([0, 2, 1, 3], [1, 1, 2, 0], [3, 0, 1, 1])
        assert (result.permutation, result.n_permutations) == ("exact", 20)
        assert result.p_value == pytest.approx(0.6)

    def test_a_residual_at_float_noise_is_undefined(self) -> None:
        # a = 5.5 - 1.2 z exactly, but its least-squares residual is ~3e-16, not 0
        residual = _residualise(np.array([1.9, 0.7, 3.1, 4.3]), np.array([3.0, 4.0, 2.0, 1.0]))
        assert np.isnan(residual).all()

    def test_monte_carlo_p_counts_only_defined_statistics(self) -> None:
        # one hit among two defined statistics: (1 + 1) / (2 + 1), not (1 + 1) / (3 + 1)
        p, n_defined = _two_sided_p(0.5, np.array([0.6, np.nan, 0.1]), "monte-carlo")
        assert (p, n_defined) == (pytest.approx(2 / 3), 2)

    def test_conditioning_removes_an_association_driven_only_by_z(self) -> None:
        rng = np.random.default_rng(0)
        z = rng.normal(size=26)
        x, y = z + 0.3 * rng.normal(size=26), z + 0.3 * rng.normal(size=26)
        assert spearman_inference(x, y, n_boot=10).p_value < 0.001
        assert partial_spearman_inference(x, y, z, n_boot=10).p_value > 0.05

    def test_conditioning_keeps_a_genuine_association(self) -> None:
        rng = np.random.default_rng(1)
        z = rng.normal(size=26)
        x = z + rng.normal(size=26)
        y = x + 0.2 * rng.normal(size=26)
        result = partial_spearman_inference(x, y, z, n_boot=10)
        assert result.rho > 0.5
        assert (result.permutation, result.n_permutations) == ("monte-carlo", 10_000)
        assert result.p_value < 0.001


class TestBenjaminiHochberg:

    def test_hand_worked_example(self) -> None:
        # sorted p: .005 .01 .03 .04 .20 -> m p_(j) / j: .025 .025 .05 .05 .20, already
        # monotone; q is returned in the input order.
        q = benjamini_hochberg([0.01, 0.04, 0.03, 0.005, 0.20])
        np.testing.assert_allclose(q, [0.025, 0.05, 0.05, 0.025, 0.20])

    def test_step_up_takes_the_running_minimum_from_the_top(self) -> None:
        # m p_(j) / j = .03 .03 .025: the largest p pulls the two smaller q down to .025
        np.testing.assert_allclose(benjamini_hochberg([0.01, 0.02, 0.025]), [0.025] * 3)

    def test_tied_p_values_share_a_q_value(self) -> None:
        np.testing.assert_allclose(benjamini_hochberg([0.02, 0.5, 0.02]), [0.03, 0.5, 0.03])

    def test_rejects_values_that_are_not_p_values(self) -> None:
        for bad in ([0.1, np.nan], [0.1, 1.5], [-0.01]):
            with pytest.raises(ValueError, match="p values"):
                benjamini_hochberg(bad)

    def test_empty_family(self) -> None:
        assert benjamini_hochberg([]).shape == (0,)


def test_every_helper_is_exported_beside_the_existing_ones_and_documented() -> None:
    from croma.metrics import bootstrap

    existing = {
        "BootstrapCI",
        "RankStability",
        "bootstrap_pooled_median",
        "paired_cluster_bootstrap_delta",
        "paired_rank_stability",
        "bootstrap_spearman",
    }
    added = {
        "cluster_bootstrap_draws",
        "TAIL_STATISTICS",
        "TailBootstrap",
        "shared_draw_tail_statistics",
        "EXACT_PERMUTATION_MAX_N",
        "CorrelationInference",
        "encoder_bootstrap_draws",
        "spearman_inference",
        "partial_spearman_inference",
        "benjamini_hochberg",
    }
    assert set(bootstrap.__all__) == existing | added
    for name in bootstrap.__all__:
        exported = getattr(bootstrap, name)
        if callable(exported):
            assert (exported.__doc__ or "").strip(), f"{name} has no docstring"


class TestUnstratifiedDrawsAreUnchanged:
    """Omitting ``strata`` must reproduce the pre-stratification output exactly.

    The literals were captured from the implementation before ``strata`` existed, so any
    drift in the unstratified random stream or in the rank/win bookkeeping fails here.
    """

    CLUSTERS = np.repeat(np.arange(6), 3)
    VALUES = np.array(
        [0.31, -0.12, 0.05, np.nan, 0.44, -0.27, 0.18, 0.02, -0.09]
        + [0.36, -0.41, 0.11, 0.07, 0.29, -0.15, 0.22, -0.03, 0.14]
    )

    def test_pooled_median(self) -> None:
        ci = bootstrap_pooled_median(self.VALUES, self.CLUSTERS, n_boot=50, seed=11)
        assert ci == BootstrapCI(point=0.07, lo=0.05, hi=0.136625, level=0.95, n_boot=50)

    def test_rank_stability(self) -> None:
        panel = {
            "a": self.VALUES,
            "b": self.VALUES[::-1].copy(),
            "c": np.roll(self.VALUES, 4) - 0.05,
        }
        rs = paired_rank_stability(panel, self.CLUSTERS, n_boot=50, seed=11)
        assert rs.point_rank == {"a": 1, "b": 2, "c": 3}
        assert rs.mean_rank == pytest.approx({"a": 1.5, "b": 1.7, "c": 2.8})
        assert rs.rank_lo == {"a": 1, "b": 1, "c": 1}
        assert rs.rank_hi == {"a": 2, "b": 3, "c": 3}
        assert rs.value_ci["b"] == BootstrapCI(0.07, 0.046125000000000006, 0.125, 0.95, 50)
        assert rs.value_ci["c"] == BootstrapCI(
            0.020000000000000004, -0.08, 0.07500000000000001, 0.95, 50
        )
        assert rs.pairwise_win == pytest.approx(
            {
                ("a", "b"): 0.48,
                ("a", "c"): 0.94,
                ("b", "a"): 0.44,
                ("b", "c"): 0.86,
                ("c", "a"): 0.06,
                ("c", "b"): 0.14,
            }
        )


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
