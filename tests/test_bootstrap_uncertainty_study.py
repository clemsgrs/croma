"""The per-encoder uncertainty study: stratified slide draws over fixed per-sample margins.

Expected values are worked by hand from a few literal per-sample vectors. ``LTM_10`` of five
values is the lowest one: the 10th percentile of five sorted values sits 0.4 of the way from
the first to the second, and only the first is at or below it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
if str(STUDIES) not in sys.path:
    sys.path.insert(0, str(STUDIES))

import bootstrap_uncertainty as study  # noqa: E402

#: Two encoders over two subsets (three and two rows), each row its own slide.
#:   A: subset s1 [1, 2, 3], s2 [10, 20]  -> subset medians 2, 15  -> balanced median 8.5
#:   B: subset s1 [-1, 0, 4], s2 [5, 6]   -> subset medians 0, 5.5 -> balanced median 2.75
#: Pooled, A's median is 3 and B's is 4: balancing over subsets reverses the order.
VALUES = {
    "A": np.array([1.0, 2.0, 3.0, 10.0, 20.0]),
    "B": np.array([-1.0, 0.0, 4.0, 5.0, 6.0]),
}
SUBSETS = np.array(["s1", "s1", "s1", "s2", "s2"])
SLIDES = np.array(["p", "q", "r", "s", "t"])


def test_strata_are_class_by_centre_when_every_slide_sits_in_one_cell():
    labels = np.array(["tumour", "tumour", "normal", "normal"])
    centres = np.array(["c1", "c2", "c1", "c2"])
    strata, design = study.stratum_labels(labels, centres, np.array(["a", "b", "c", "d"]))
    assert design == "class x centre"
    assert len(set(strata)) == 4


def test_strata_fall_back_to_centre_when_a_case_spans_classes():
    labels = np.array(["tumour", "normal", "tumour", "normal"])
    centres = np.array(["c1", "c1", "c2", "c2"])
    strata, design = study.stratum_labels(labels, centres, np.array(["a", "a", "b", "b"]))
    assert design == "centre"
    assert strata.tolist() == ["c1", "c1", "c2", "c2"]


def test_point_values_balance_the_median_over_subsets_and_pool_the_tail():
    result = study.encoder_uncertainty(
        VALUES, SLIDES, strata=SLIDES, subsets=SUBSETS, n_boot=20, seed=0
    )
    table = result.table.set_index("model")
    assert table.loc["A", "median"] == 8.5
    assert table.loc["B", "median"] == 2.75
    # F(0) and LTM_10 pool every sample, as the published columns do.
    assert table.loc["A", "f0"] == 0.0
    assert table.loc["B", "f0"] == 0.4
    assert table.loc["A", "ltm"] == 1.0
    assert table.loc["B", "ltm"] == -1.0
    assert table.loc["A", "point_rank"] == 1


def test_without_subsets_the_median_is_pooled():
    result = study.encoder_uncertainty(VALUES, SLIDES, strata=SLIDES, n_boot=20, seed=0)
    table = result.table.set_index("model")
    assert table.loc["A", "median"] == 3.0
    assert table.loc["B", "median"] == 4.0
    assert table.loc["B", "point_rank"] == 1


def test_one_slide_per_stratum_leaves_nothing_to_resample():
    # Every stratum holds a single slide, so every replicate redraws the full cohort: each
    # interval collapses on its point and every encoder keeps its rank.
    result = study.encoder_uncertainty(
        VALUES, SLIDES, strata=SLIDES, subsets=SUBSETS, n_boot=50, seed=3
    )
    table = result.table.set_index("model")
    for stat in ("median", "f0", "ltm"):
        assert (table[f"{stat}_lo"] == table[stat]).all()
        assert (table[f"{stat}_hi"] == table[stat]).all()
    assert table.loc["A", "rank_lo"] == table.loc["A", "rank_hi"] == 1
    assert table.loc["B", "mean_rank"] == 2.0
    # A beats B on both median and tail in every replicate, so only A is on the frontier.
    assert table.loc["A", "p_frontier"] == 1.0
    assert table.loc["B", "p_frontier"] == 0.0
    assert result.pairwise_win[("A", "B")] == 1.0


def test_frontier_probability_is_the_share_of_replicates_an_encoder_is_undominated():
    models = ["A", "B", "C"]
    # replicate 0: A (1, 0), B (0, 1), C (0, 0)   -> frontier {A, B}
    # replicate 1: A (1, 1), B (0, 0), C (0, 0.5) -> frontier {A}
    medians = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tails = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.5]])
    p = study.frontier_probability(models, medians, tails)
    assert p == {"A": 1.0, "B": 0.5, "C": 0.0}


def test_separated_adjacent_pairs_count_wins_at_or_above_the_threshold():
    order = ["A", "B", "C", "D"]
    wins = {("A", "B"): 0.99, ("B", "C"): 0.975, ("C", "D"): 0.97}
    assert study.separated_adjacent_pairs(order, wins) == 2


def test_mean_rank_frontier_combines_replicate_b_of_every_benchmark():
    models = ["A", "B", "C"]
    # Replicate 0. X: medians A > B > C, tails C > A > B (ranks A1 B2 C3; A2 B3 C1).
    #              Y: medians B > C > A, tails C > B > A (ranks A3 B1 C2; A3 B2 C1).
    # Mean median ranks A 2, B 1.5, C 2.5; mean tail ranks A 2.5, B 2.5, C 1: B and C are
    # undominated, A is dominated by B.
    # Replicate 1: A is highest on everything in both benchmarks, so A alone.
    x = (
        np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
        np.array([[2.0, 1.0, 3.0], [3.0, 2.0, 1.0]]),
    )
    y = (
        np.array([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]]),
        np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    )
    p = study.mean_rank_frontier_probability(models, {"X": x, "Y": y})
    assert p == {"A": 0.5, "B": 0.5, "C": 0.5}


def test_cross_benchmark_rho_shares_one_encoder_draw_across_pairs():
    medians = pd.DataFrame(
        {"X": [1.0, 2.0, 3.0, 4.0, 5.0], "Y": [10.0, 20.0, 30.0, 40.0, 50.0]},
        index=list("abcde"),
    )
    medians["Z"] = -medians["X"]
    rho, mean = study.cross_benchmark_rho(
        medians, headline=[("X", "Y"), ("X", "Z")], n_boot=200, seed=0
    )
    by_pair = rho.set_index(["a", "b"])
    assert by_pair.loc[("X", "Y"), "rho"] == 1.0
    assert by_pair.loc[("X", "Z"), "rho"] == -1.0
    assert by_pair.loc[("Y", "Z"), "rho"] == -1.0
    assert by_pair.loc[("X", "Y"), ["ci_lo", "ci_hi"]].tolist() == [1.0, 1.0]
    assert by_pair["headline"].tolist() == [True, True, False]
    # With one shared draw every replicate's mean over the headline pairs is (1 - 1) / 2.
    assert mean == {"rho": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n_pairs": 2, "n_models": 5}
