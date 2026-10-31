"""The bootstrap-uncertainty supplement and its macros, rendered from hand-built study outputs.

Every expected string is written out by hand from the toy frames below, never recomputed the
way the renderer computes it.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "scripts" / "repro"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

import _uncertainty as unc  # noqa: E402
import build_paper  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402
from paper_manifest import by_prefix  # noqa: E402


def _benchmark(prefix: str, **meta) -> unc.Benchmark:
    table = pd.DataFrame(
        {
            "model": ["Beta", "Alpha", "Gamma"],
            "median": [0.30, 0.20, -0.05],
            "median_lo": [0.28, 0.17, -0.08],
            "median_hi": [0.33, 0.24, -0.01],
            "f0": [0.041, 0.120, 0.555],
            "f0_lo": [0.030, 0.100, 0.500],
            "f0_hi": [0.055, 0.140, 0.610],
            "ltm": [-0.02, -0.10, -0.40],
            "ltm_lo": [-0.05, -0.13, -0.45],
            "ltm_hi": [0.01, -0.07, -0.35],
            "point_rank": [1, 2, 3],
            "mean_rank": [1.0, 2.0, 3.0],
            "rank_lo": [1, 2, 3],
            "rank_hi": [1, 2, 3],
            "on_frontier": [True, False, False],
            "p_frontier": [0.9995, 0.0125, 0.0],
        }
    )
    base = dict(
        n_boot=2000,
        alpha=0.1,
        strata="class x centre",
        n_strata=4,
        n_clusters=68,
        n_subsets=1,
        n_models=3,
        separation_threshold=0.975,
        n_separated_adjacent=2,
        n_adjacent=2,
    )
    return unc.Benchmark(prefix, table, {**base, **meta})


def _delta(prefix: str, k: int) -> unc.Delta:
    table = pd.DataFrame(
        {
            "model": ["Alpha", "Beta", "DINOv2-B"],
            "is_control": [False, False, True],
            "delta": [0.0123, -0.0004, 0.05],
            "delta_lo": [0.0091, -0.0031, 0.04],
            "delta_hi": [0.0150, 0.0022, 0.06],
            "excludes_zero": [True, False, True],
        }
    )
    meta = dict(n_boot=2000, k=k, n_ranked=2, n_excluding_zero=1)
    return unc.Delta(prefix, table, meta)


def _study(**cross_meta) -> unc.Study:
    cross = pd.DataFrame(
        {
            "a": ["Camelyon", "Camelyon", "TcgaTwoByTwo", "TcgaFourByFour"],
            "b": ["TcgaTwoByTwo", "Tolkach", "TcgaFourByFour", "Tolkach"],
            "n": [3, 3, 3, 3],
            "rho": [0.5, 1.0, 0.5, 0.5],
            "ci_lo": [-0.5, 0.5, -1.0, 0.0],
            "ci_hi": [1.0, 1.0, 1.0, 1.0],
            "p_value": [0.3, 0.1, 0.3, 0.3],
            "headline": [False, True, False, True],
        }
    )
    meta = dict(
        rho=0.75,
        ci_lo=0.25,
        ci_hi=1.0,
        n_pairs=3,
        n_models=3,
        n_boot=2000,
        pairs=[
            ["Camelyon", "TcgaFourByFour"],
            ["Camelyon", "Tolkach"],
            ["TcgaFourByFour", "Tolkach"],
        ],
    )
    return unc.Study(
        benchmarks=tuple(_benchmark(p) for p in unc.PREFIXES),
        deltas=tuple(_delta(p, k) for p, k in zip(unc.DELTA_PREFIXES, (71, 91, 71, 61))),
        cross=cross,
        cross_meta={**meta, **cross_meta},
        mean_rank=pd.DataFrame({"model": ["Alpha", "Beta", "Gamma"], "p_frontier": [0.4, 0.97, 0]}),
    )


def test_a_benchmark_table_lists_ranked_encoders_with_their_intervals():
    tex = unc.render_benchmark(_benchmark("Camelyon"), 2000)
    rows = [line for line in tex.splitlines() if line.endswith(r"\\") and "Model" not in line]
    assert rows[0] == (
        r"\textbf{Beta} & $0.30$\,{\scriptsize$[0.28, 0.33]$} & $1$\,{\scriptsize$[1, 1]$} & "
        r"$0.041$\,{\scriptsize$[0.030, 0.055]$} & $-0.02$\,{\scriptsize$[-0.05, 0.01]$} & "
        r"$1.000$ \\"
    )
    assert rows[1].startswith(r"Alpha & $0.20$")
    assert rows[1].endswith(r"$0.013$ \\")
    assert r"\label{tab:bootstrap-uncertainty-camelyon}" in tex
    assert "$2000$ stratified cluster-bootstrap replicates" in tex
    assert r"the $68$ slides with replacement within their $4$ class $\times$ centre strata" in tex
    assert "$2$ of the $2$ adjacent pairs" in tex
    assert r"at least 97.5\% of replicates" in tex


def test_tolkach_resamples_cases_within_centres_and_tcga_2x2_balances_its_subsets():
    tolkach = unc.render_benchmark(
        _benchmark("Tolkach", strata="centre", n_strata=3, n_clusters=43), 2000
    )
    assert "the $43$ cases with replacement within their $3$ centre strata" in tolkach
    tcga = unc.render_benchmark(_benchmark("TcgaTwoByTwo", n_subsets=94), 2000)
    assert "the median of the $94$ paired subsets' medians" in tcga
    assert "paired subsets" not in unc.render_benchmark(_benchmark("Camelyon"), 2000)


def test_the_slide_table_is_named_pcabiop_and_stratifies_by_provider():
    b = _benchmark("Panda")
    tex = unc.render_benchmark(unc.Benchmark(b.prefix, b.table, b.meta, has_control=False), 2000)
    assert "on PCaBiop.}" in tex
    assert r"class $\times$ data provider strata" in tex
    assert "ranked slide-level encoders, the median" in tex
    assert "control" not in tex
    assert r"\label{tab:bootstrap-uncertainty-pcabiop}" in tex
    assert "the natural-image control is excluded" in unc.render_benchmark(_benchmark("Tolkach"), 1)


def test_the_delta_table_excludes_the_control_and_counts_intervals_off_zero():
    tex = unc.render_delta(_study())
    assert "DINOv2-B" not in tex
    # \textbf does not reach into math mode, so a bold cell sets \boldmath.
    assert r"Alpha & {\boldmath$+0.012$\,{\scriptsize$[+0.009, +0.015]$}} & " in tex
    # A point that rounds to zero carries no sign; a bound keeps the side of zero it is on.
    assert r"Beta & $0.000$\,{\scriptsize$[-0.003, +0.002]$} & " in tex
    assert r"CI excludes $0$ & $1/2$ & $1/2$ & $1/2$ & $1/2$ \\" in tex
    assert r"TCGA-2x2 ($k{=}91$)" in tex
    assert r"\label{tab:mari-ri-delta}" in tex


def test_the_cross_benchmark_table_marks_the_pairs_in_the_mean():
    tex = unc.render_cross(_study())
    assert r"Camelyon--Tolkach-ESCA$^{*}$ & $3$ & $1.00$\,{\scriptsize$[0.50, 1.00]$} \\" in tex
    assert r"Camelyon--TCGA-2x2 & $3$ & " in tex
    assert r"Mean$^{*}$ & $3$ & $0.75$\,{\scriptsize$[0.25, 1.00]$} \\" in tex
    assert "among Camelyon, TCGA-4x4 and Tolkach-ESCA" in tex


def test_a_mean_over_other_benchmarks_cannot_keep_the_caption():
    with pytest.raises(CaptionClaimError, match="leaves TCGA-2x2 out"):
        unc.render_cross(_study(pairs=[["Camelyon", "TcgaTwoByTwo"]]))


def test_macros_quote_the_frontier_members_separations_and_delta_counts():
    lines = unc.macros(_study(), rank_frontier=("Beta",))
    assert r"\newcommand{\UncertaintyNBoot}{2000}" in lines
    assert r"\newcommand{\CamelyonFrontierProbBeta}{$1.00$}" in lines
    assert not any("FrontierProbAlpha" in line for line in lines if "TileRank" not in line)
    assert r"\newcommand{\TolkachSeparatedAdjacentPairs}{2}" in lines
    assert r"\newcommand{\PandaAdjacentPairs}{2}" in lines
    assert r"\newcommand{\TileRankFrontierProbBeta}{$0.97$}" in lines
    assert r"\newcommand{\TcgaTwoByTwoDeltaCiExcludesZero}{1}" in lines
    assert r"\newcommand{\TcgaTwoByTwoDeltaNRanked}{2}" in lines


def test_a_rank_frontier_member_the_study_never_scored_fails():
    with pytest.raises(CaptionClaimError, match="Delta"):
        unc.macros(_study(), rank_frontier=("Delta",))


def test_cross_cohort_cis_are_checked_against_the_quoted_rhos():
    rhos = {frozenset(("Camelyon", "Tolkach")): 1.0, frozenset(("TcgaFourByFour", "Tolkach")): 0.5}
    lines = unc.cross_cohort_ci_macros(_study(), rhos)
    assert r"\newcommand{\CrossCohortRhoMeanCi}{$[0.25, 1.00]$}" in lines
    assert r"\newcommand{\CrossCohortRhoCamelyonTolkachCi}{$[0.50, 1.00]$}" in lines
    with pytest.raises(CaptionClaimError, match="differs"):
        unc.cross_cohort_ci_macros(_study(), {**rhos, frozenset(("Camelyon", "Tolkach")): 0.9})


def test_a_missing_study_output_fails_naming_the_remedy(tmp_path):
    with pytest.raises(FileNotFoundError, match="bootstrap_uncertainty.py"):
        unc.load(tmp_path)
    for prefix in unc.PREFIXES:
        base = tmp_path / by_prefix(prefix).studies_rel
        base.mkdir(parents=True)
        b = _benchmark(prefix)
        b.table.to_csv(base / "bootstrap_uncertainty.csv", index=False)
        (base / "bootstrap_uncertainty.json").write_text(json.dumps(b.meta))
        metrics = tmp_path / by_prefix(prefix).metrics_rel
        metrics.parent.mkdir(parents=True)
        b.table[["model"]].to_csv(metrics, index=False)
    with pytest.raises(FileNotFoundError, match="mari_ri_delta.py"):
        unc.load(tmp_path)


def test_the_section_is_a_core_generator_so_a_missing_study_fails_the_build():
    core = [script for script, _, _ in build_paper.CORE]
    assert "generate_uncertainty_supp_table.py" in core
    assert "generate_uncertainty_supp_table.py" not in [s for s, _, _ in build_paper.STUDIES]
    # The headline rho CIs are macros of the downstream block, so it cannot be skipped either.
    assert "generate_apd_floats.py" in core


def test_results_captions_point_to_their_benchmarks_uncertainty_table():
    import generate_results_table as grt

    clause = grt.uncertainty_clause(by_prefix("Tolkach"))
    assert r"Supplementary Table~\ref{tab:bootstrap-uncertainty-tolkach}" in clause
    assert "rank" in clause
    # prostate-shift-binary is not in the uncertainty study: no pointer, no dangling \ref.
    assert grt.uncertainty_clause(by_prefix("Prostate")) == ""


def _headline_rows(**two_level) -> pd.DataFrame:
    rows = [
        dict(
            target="nipd_id",
            metric="croma",
            scope=scope,
            spearman=rho,
            ci_lo=lo,
            ci_hi=hi,
            two_level_lo=lo - 0.05,
            two_level_hi=hi - 0.01,
        )
        for scope, rho, lo, hi in (
            ("camelyon", 0.946, 0.8344, 0.9774),
            ("tcga_4x4", 0.904, 0.7469, 0.9560),
            ("tolkach", 0.931, 0.7952, 0.9752),
        )
    ]
    return pd.DataFrame(rows)


def test_the_headline_rho_ci_macros_follow_the_table_switch():
    import generate_paper_values as gpv

    encoder = gpv._apd_ci_macros(_headline_rows(), design="encoder")
    assert r"\newcommand{\NipdIdCromaCamelyonCi}{$[0.83, 0.98]$}" in encoder
    assert r"\newcommand{\NipdIdCromaTcgaCiLo}{0.75}" in encoder
    assert r"\newcommand{\NipdIdCromaTolkachCiHi}{0.98}" in encoder
    two_level = gpv._apd_ci_macros(_headline_rows(), design="two-level")
    assert r"\newcommand{\NipdIdCromaCamelyonCi}{$[0.78, 0.97]$}" in two_level


def test_a_missing_downstream_study_fails_the_macros(tmp_path):
    import generate_paper_values as gpv

    with pytest.raises(FileNotFoundError, match="apd_croma_correlation.py"):
        gpv._apd_block(tmp_path)
