"""Roster-dependent paper macros: frontiers, counts, leaders, operating points and agreement.

Every number here used to be hand-typed in the text and went stale as the tile panel grew.
``scripts/repro/_panel_macros.py`` now derives them from the runs. The pure builders are
driven with small worked frames whose expected values are computed by hand; the loader is
driven against a temporary root to show that a missing input fails the build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts/repro", ROOT / "scripts/bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _panel_macros as panel  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402


def _frame(**overrides) -> pd.DataFrame:
    """Five ranked encoders and the natural-image control, worked by hand below.

    Median-vs-LTM frontier over the ranked five: A (0.30, -0.10) has the highest median and
    B (0.20, 0.05) the mildest tail; C is beaten by A on both axes, D and E by everything.
    The control (0.10, 0.20) would join the frontier if it were not excluded.
    """
    frame = pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "DINOv2-B"],
            "croma": [0.30, 0.20, 0.25, -0.05, 0.00, 0.10],
            "croma_ltm_alpha": [-0.10, 0.05, -0.20, -0.50, -0.40, 0.20],
            "croma_f0": [0.05, 0.00, 0.05, 0.60, 0.50, 0.30],
            "bio_knn_bacc": [0.90, 0.85, 0.95, 0.70, 0.80, 0.75],
            "k": [11] * 6,
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


def test_frontier_lists_ranked_members_in_staircase_order_with_their_count():
    lines = panel.ranking_macros("Bench", _frame())

    assert r"\newcommand{\BenchParetoFrontierModels}{B, A}" in lines
    assert r"\newcommand{\BenchParetoFrontierNModels}{2}" in lines


def test_counts_cover_the_ranked_encoders_only():
    lines = panel.ranking_macros("Bench", _frame())

    assert r"\newcommand{\BenchNCromaNonPositive}{2}" in lines  # D (-0.05), E (0.00)
    assert r"\newcommand{\BenchNLtmNegative}{4}" in lines  # all but B
    assert r"\newcommand{\BenchNFZeroPositive}{4}" in lines  # all but B


def test_f0_extremes_name_their_encoders_and_every_tied_one():
    lines = panel.ranking_macros("Bench", _frame(croma_f0=[0.05, 0.00, 0.05, 0.60, 0.60, 0.90]))

    assert r"\newcommand{\BenchFZeroMin}{$0.000$}" in lines
    assert r"\newcommand{\BenchFZeroMinModels}{B}" in lines
    # D and E tie at the maximum; the control's 0.90 is not a ranked extreme.
    assert r"\newcommand{\BenchFZeroMax}{$0.600$}" in lines
    assert r"\newcommand{\BenchFZeroMaxModels}{D, E}" in lines


def test_top_three_by_median_croma_carry_their_values():
    lines = panel.ranking_macros("Bench", _frame())

    assert r"\newcommand{\BenchTopOneModel}{A}" in lines
    assert r"\newcommand{\BenchTopOneCroma}{$0.30$}" in lines
    assert r"\newcommand{\BenchTopTwoModel}{C}" in lines
    assert r"\newcommand{\BenchTopTwoCroma}{$0.25$}" in lines
    assert r"\newcommand{\BenchTopThreeModel}{B}" in lines
    assert r"\newcommand{\BenchTopThreeCroma}{$0.20$}" in lines


def test_an_empty_frontier_fails():
    with pytest.raises(CaptionClaimError, match="frontier"):
        panel.ranking_macros("Bench", _frame().iloc[[5]])


def test_operating_k_is_the_runs_single_shared_k():
    lines = panel.operating_point_macros("Bench", _frame(k=[71] * 6))

    assert r"\newcommand{\BenchOperatingK}{71}" in lines


def test_a_run_without_one_shared_k_fails():
    with pytest.raises(CaptionClaimError, match="shared k"):
        panel.operating_point_macros("Bench", _frame(k=[11, 11, 11, 11, 11, 13]))


def test_control_macros_report_its_scores_and_how_many_encoders_fall_below_it():
    lines = panel.operating_point_macros("Bench", _frame())

    assert r"\newcommand{\BenchControlCroma}{$0.10$}" in lines
    assert r"\newcommand{\BenchControlBioBacc}{$0.750$}" in lines
    assert r"\newcommand{\BenchNBelowControl}{2}" in lines  # D, E


def test_tcga2x2_composition_is_counted_from_the_view():
    manifest = pd.DataFrame(
        {
            "subset": ["q1"] * 4 + ["q2"] * 4,
            "label": ["BRCA", "BRCA", "LUAD", "LUAD", "BRCA", "BRCA", "COAD", "COAD"],
            "confounder": ["X", "Y", "X", "Y", "X", "Z", "X", "Z"],
        }
    )

    lines = panel.tcga2x2_composition_macros(manifest)

    assert r"\newcommand{\TcgaTwoByTwoNCancerTypes}{3}" in lines
    assert r"\newcommand{\TcgaTwoByTwoNCentres}{3}" in lines
    assert r"\newcommand{\TcgaTwoByTwoNQuartets}{2}" in lines


def test_tcga2x2_rank_agreement_is_reported_apart_from_the_cross_cohort_mean():
    index = ["A", "B", "C", "D", "DINOv2-B"]
    croma = {
        "TcgaTwoByTwo": pd.Series([1, 2, 3, 4, 9], index=index, dtype=float),
        "TcgaFourByFour": pd.Series([10, 20, 30, 40, -9], index=index, dtype=float),
        "Camelyon": pd.Series([4, 3, 2, 1, 9], index=index, dtype=float),
        # One swapped pair: rho = 1 - 6 * 2 / (4 * 15) = 0.8.
        "Tolkach": pd.Series([1, 2, 4, 3, -9], index=index, dtype=float),
    }

    lines = panel.tcga2x2_agreement_macros(croma)

    assert r"\newcommand{\TcgaTwoByTwoRhoTcgaFourByFour}{$1.00$}" in lines
    assert r"\newcommand{\TcgaTwoByTwoRhoOtherRange}{$[-1.00, 0.80]$}" in lines


def test_apd_and_nipd_rankings_agree_per_benchmark_and_regime():
    joined = pd.DataFrame(
        {
            "dataset": ["camelyon"] * 5,
            "model": ["A", "B", "C", "D", "DINOv2-B"],
            "apd_id": [0.1, 0.2, 0.3, 0.4, 0.9],
            "nipd_id": [1.0, 2.0, 3.0, 4.0, -5.0],
            "apd_ood": [0.1, 0.2, 0.3, 0.4, 0.9],
            "nipd_ood": [4.0, 3.0, 2.0, 1.0, 9.0],
        }
    )

    lines = panel.apd_nipd_macros(joined, datasets=("camelyon",))

    assert r"\newcommand{\CamelyonApdNipdRhoId}{$1.00$}" in lines
    assert r"\newcommand{\CamelyonApdNipdRhoOod}{$-1.00$}" in lines


def test_a_missing_run_fails_the_build(tmp_path):
    with pytest.raises(FileNotFoundError, match="metrics.csv"):
        panel.panel_macros(tmp_path)
