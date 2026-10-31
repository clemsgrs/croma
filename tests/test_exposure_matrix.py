"""Pretraining exposure: the matrix, the drop-exposed sensitivity and their renderings.

``scripts/repro/_exposure.py`` resolves every cell of ``tab:exposure-matrix`` through
``_model_provenance.exposed_models_for_domain``, the resolver the Pareto and results-table
daggers use, and the drop-exposed sensitivity (``tab:drop-exposed``) removes exactly the cells
the matrix marks. The pure builders are driven with small worked frames whose expected values
are written out by hand; the committed metadata drives the agreement tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts/repro", ROOT / "scripts/bench"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _exposure as ex  # noqa: E402


def _metadata(*rows: dict) -> pd.DataFrame:
    """A model_metadata.csv-shaped frame; omitted fields are empty, as in the CSV."""
    defaults = {
        "panel": "tile",
        "panel_order": 0,
        "disclosed": "yes",
        "corpus_domains": "",
        "institutional_domains": "",
        "cite": "",
        "benchmark_selection": "",
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["panel_order"] = range(1, len(frame) + 1)
    return frame


def test_a_cell_is_exposed_not_exposed_or_undisclosed_and_never_inferred():
    md = _metadata(
        {"model": "Tcga-only", "corpus_domains": "tcga"},
        {"model": "Public-elsewhere", "corpus_domains": ""},
        {"model": "Proprietary", "disclosed": "no"},
    )

    matrix = ex.exposure_matrix(md)

    assert matrix.status("Tcga-only", "tcga") == "exposed"
    assert matrix.status("Tcga-only", "camelyon") == "not exposed"
    assert matrix.status("Public-elsewhere", "tcga") == "not exposed"
    assert matrix.status("Proprietary", "tcga") == "undisclosed"
    assert matrix.status("Proprietary", "tolkach-esca") == "undisclosed"


def test_a_shared_institution_is_marked_with_its_centres_and_source_not_as_dataset_exposure():
    md = _metadata(
        {
            "model": "Prostate",
            "corpus_domains": "tcga",
            "institutional_domains": "rumc",
            "cite": "prostkey",
        },
        {"model": "Elsewhere"},
    )

    matrix = ex.exposure_matrix(md)

    (mark,) = matrix.institutions("Prostate")
    assert mark.institution == "rumc"
    assert ("pathorob-camelyon", "RUMC") in mark.centres
    assert ("panda", "radboud") in mark.centres
    assert mark.source == "prostkey"
    # Sharing Camelyon's RUMC centre is not exposure to the CAMELYON16/17 dataset.
    assert matrix.status("Prostate", "camelyon") == "not exposed"
    assert matrix.institutions("Elsewhere") == ()


def test_an_institution_marker_without_a_source_fails():
    md = _metadata({"model": "Unsourced", "institutional_domains": "charite"})

    with pytest.raises(ValueError, match="Unsourced"):
        ex.exposure_matrix(md)


def test_an_institution_token_no_benchmark_centre_is_mapped_to_fails():
    md = _metadata({"model": "Unknown", "institutional_domains": "nowhere", "cite": "key"})

    with pytest.raises(ValueError, match="nowhere"):
        ex.exposure_matrix(md)


@pytest.mark.parametrize(
    "row",
    [
        {"model": "Crossed", "institutional_domains": "camelyon", "cite": "key"},
        {"model": "Crossed", "corpus_domains": "charite", "cite": "key"},
    ],
    ids=["dataset-token-as-institution", "institution-token-as-corpus"],
)
def test_dataset_and_institution_tokens_never_share_a_namespace(row):
    """The resolver unions both columns, so a token in the wrong one would silently turn an
    institution marker into dataset exposure (or the reverse)."""
    with pytest.raises(ValueError, match="Crossed"):
        ex.exposure_matrix(_metadata(row))


def test_pathorob_use_during_development_is_its_own_column_with_its_source():
    md = _metadata(
        {
            "model": "Selected",
            "disclosed": "no",
            "benchmark_selection": "pathorob-ri",
            "cite": "selkey",
        },
        {"model": "Plain"},
    )

    matrix = ex.exposure_matrix(md)

    assert matrix.selection("Selected") == ex.Selection("PathoROB RI", "selkey")
    assert matrix.selection("Plain") is None
    # A different kind of exposure: it never touches a dataset cell.
    assert matrix.status("Selected", "tcga") == "undisclosed"


@pytest.mark.parametrize(
    "row",
    [
        {"model": "Odd", "benchmark_selection": "some-other-benchmark", "cite": "key"},
        {"model": "Odd", "benchmark_selection": "pathorob-ri"},
    ],
    ids=["unworded-disclosure", "unsourced"],
)
def test_a_selection_disclosure_needs_wording_and_a_source(row):
    with pytest.raises(ValueError, match="Odd"):
        ex.exposure_matrix(_metadata(row))


def test_counts_are_exposed_and_undisclosed_per_column_and_panel_without_the_control():
    md = _metadata(
        {"model": "T1", "corpus_domains": "tcga"},
        {"model": "T2", "corpus_domains": "tcga;camelyon"},
        {"model": "T3", "disclosed": "no"},
        {"model": "DINOv2-B"},
        {"model": "S1", "panel": "slide", "disclosed": "no"},
        {"model": "S2", "panel": "slide", "corpus_domains": "panda"},
    )

    counts = ex.exposure_matrix(md).counts()

    assert counts["tile"]["tcga"] == ex.Count(exposed=2, undisclosed=1, n=3)
    assert counts["tile"]["camelyon"] == ex.Count(exposed=1, undisclosed=1, n=3)
    assert counts["tile"]["par"] == ex.Count(exposed=0, undisclosed=1, n=3)
    assert counts["slide"]["panda"] == ex.Count(exposed=1, undisclosed=1, n=2)
    assert counts["slide"]["tcga"] == ex.Count(exposed=0, undisclosed=1, n=2)
    assert set(counts) == {"tile", "slide"}


def test_a_disclosure_state_other_than_yes_or_no_fails_rather_than_being_guessed():
    md = _metadata({"model": "Blank", "disclosed": ""})

    with pytest.raises(ValueError, match="Blank"):
        ex.exposure_matrix(md)


# --- rendering the matrix -------------------------------------------------------------------


def _cells(tex: str, first: str) -> list[str]:
    """The ``&``-separated cells of the one tabular row that starts with ``first``."""
    (line,) = [ln for ln in tex.splitlines() if ln.startswith(first + " &")]
    return [cell.strip() for cell in line.removesuffix(r"\\").split("&")]


def _render_worked() -> str:
    md = _metadata(
        {"model": "Tcga-only", "corpus_domains": "tcga"},
        {
            "model": "Proprietary",
            "disclosed": "no",
            "benchmark_selection": "pathorob-ri",
            "cite": "selkey",
        },
        {
            "model": "Prostate",
            "corpus_domains": "tcga",
            "institutional_domains": "rumc",
            "cite": "prostkey",
        },
        {"model": "DINOv2-B"},
        {"model": "Slide-one", "panel": "slide", "corpus_domains": "panda"},
    )
    return ex.render_matrix(ex.exposure_matrix(md), [("scripts/bench/model_metadata.csv", "ab12")])


def test_matrix_rows_read_the_cell_values_in_dataset_column_order():
    tex = _render_worked()

    # TCGA, CAMELYON16/17, PANDA, PAR, Tolkach-ESCA, institution, PathoROB in development
    assert _cells(tex, "Tcga-only") == [
        "Tcga-only",
        r"\textbf{exposed}",
        "not exposed",
        "not exposed",
        "not exposed",
        "not exposed",
        "--",
        "--",
    ]
    assert _cells(tex, "Proprietary")[1:6] == ["undisclosed"] * 5
    assert _cells(tex, "Slide-one")[3] == r"\textbf{exposed}"
    assert r"\label{tab:exposure-matrix}" in tex


def test_matrix_institution_and_development_cells_carry_their_source():
    tex = _render_worked()

    institution = _cells(tex, "Prostate")[6]
    assert institution.startswith("Radboudumc (") and institution.endswith(r"~\cite{prostkey}")
    assert "Camelyon RUMC" in institution and "PANDA radboud" in institution
    assert _cells(tex, "Proprietary")[7] == r"PathoROB RI~\cite{selkey}"


def test_matrix_counts_rows_give_exposed_and_undisclosed_per_panel():
    tex = _render_worked()

    assert _cells(tex, r"\textit{Tile: exposed / undisclosed}")[1:6] == [
        "2 / 1",
        "0 / 1",
        "0 / 1",
        "0 / 1",
        "0 / 1",
    ]
    assert _cells(tex, r"\textit{Slide: exposed / undisclosed}")[1:6] == [
        "0 / 0",
        "0 / 0",
        "1 / 0",
        "0 / 0",
        "0 / 0",
    ]


def test_the_control_sits_below_both_panels_and_is_not_counted():
    tex = _render_worked()
    lines = tex.splitlines()

    control = next(i for i, ln in enumerate(lines) if ln.startswith("DINOv2-B &"))
    slide = next(i for i, ln in enumerate(lines) if ln.startswith("Slide-one &"))
    assert control > slide


def test_a_rendered_table_records_the_fingerprint_of_every_input():
    tex = _render_worked()

    assert "% input scripts/bench/model_metadata.csv sha256=ab12" in tex.splitlines()


# --- the drop-exposed sensitivity ----------------------------------------------------------


def _scores() -> pd.DataFrame:
    """Five ranked encoders and the control. Median-vs-LTM frontier, by hand: D has the mildest
    tail, B beats A's tail and C on both axes, A has the top median; C and E are dominated."""
    return pd.DataFrame(
        {
            "model": ["A", "B", "C", "D", "E", "DINOv2-B"],
            "croma": [0.5, 0.4, 0.3, 0.2, 0.1, 0.9],
            "croma_ltm_alpha": [-0.5, -0.1, -0.2, 0.0, -0.3, 0.9],
            "nipd_id": [5.0, 4.0, 3.0, 2.0, 1.0, 9.0],
            "nipd_ood": [1.0, 2.0, 3.0, 4.0, 5.0, 9.0],
        }
    )


def test_each_variant_recomputes_rho_frontier_and_top_three_on_the_encoders_it_keeps():
    full, drop, disclosed = ex.sensitivity(_scores(), drop={"A"}, keep={"C", "D", "E"})

    assert full.kept == ("A", "B", "C", "D", "E") and full.dropped == ()
    assert full.frontier == ("D", "B", "A")
    assert full.top == ("A", "B", "C")
    assert (full.rho_id.rho, full.rho_ood.rho) == (1.0, -1.0)
    assert full.rho_id.p_value == pytest.approx(2 / 120)  # exact: 2 of 5! orderings

    assert drop.kept == ("B", "C", "D", "E") and drop.dropped == ("A",)
    assert drop.frontier == ("D", "B")
    assert drop.top == ("B", "C", "D")
    assert drop.rho_id.n == 4 and drop.rho_id.permutation == "exact"
    assert drop.rho_id.p_value == pytest.approx(2 / 24)

    assert disclosed.kept == ("C", "D", "E") and disclosed.dropped == ("A", "B")
    assert disclosed.frontier == ("D", "C")
    assert disclosed.top == ("C", "D", "E")
    assert disclosed.rho_ood.rho == -1.0
    assert disclosed.rho_ood.p_value == pytest.approx(2 / 6)


def test_dropping_encoders_never_changes_a_remaining_encoders_score():
    full, drop, disclosed = ex.sensitivity(_scores(), drop={"A"}, keep={"C", "D", "E"})

    for variant in (drop, disclosed):
        for model in variant.kept:
            assert variant.scores.loc[model].equals(full.scores.loc[model])


def test_a_variant_with_nothing_to_drop_is_the_full_panel_and_says_so():
    full, drop, _ = ex.sensitivity(_scores(), drop=set(), keep={"C", "D", "E"})

    assert drop.dropped == () and drop.unchanged
    assert (drop.kept, drop.frontier, drop.top) == (full.kept, full.frontier, full.top)


@pytest.mark.parametrize(
    "drop, keep, match",
    [
        ({"Z"}, {"C", "D", "E"}, "Z"),
        (set(), {"C", "Z"}, "Z"),
        (set(), {"C", "D"}, "three"),
    ],
    ids=["drop-unknown", "keep-unknown", "too-few-kept"],
)
def test_a_variant_that_cannot_be_computed_fails(drop, keep, match):
    with pytest.raises(ex.CaptionClaimError, match=match):
        ex.sensitivity(_scores(), drop=drop, keep=keep)


def _rank_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two benchmarks, four encoders: median CRoMa and LTM_10 values. Tail ranks are 3, 2, 1, 4
    on both. Mean ranks by hand, full panel (median, tail):
    P (1.5, 3), Q (2, 2), R (3.5, 1), S (3, 4) -> frontier R, Q, P.
    Without S and re-ranked: P (1.5, 3), Q (1.5, 2), R (3, 1) -> Q now ties P's median and has
    the milder tail, so the frontier is R, Q."""
    models = ["P", "Q", "R", "S"]
    medians = pd.DataFrame({"b1": [0.9, 0.7, 0.6, 0.8], "b2": [0.8, 0.9, 0.7, 0.6]}, index=models)
    tails = pd.DataFrame(
        {"b1": [-0.3, -0.2, -0.1, -0.4], "b2": [-0.3, -0.2, -0.1, -0.4]}, index=models
    )
    return medians, tails


def test_the_mean_rank_frontier_is_re_ranked_among_the_encoders_kept():
    from _rank_pareto import RankPareto

    medians, tails = _rank_inputs()
    rp = RankPareto(
        medians=medians,
        median_ranks=medians.rank(ascending=False, method="first").astype(int),
        tail_ranks=tails.rank(ascending=False, method="first").astype(int),
        exposed=frozenset({"S"}),
        adversarial="b2",
    )

    assert ex.rank_frontier(medians, tails, {"P", "Q", "R", "S"}) == ("R", "Q", "P") == rp.frontier
    assert ex.rank_frontier(medians, tails, {"P", "Q", "R"}) == ("R", "Q")


def test_a_rank_frontier_over_unknown_encoders_fails():
    with pytest.raises(ex.CaptionClaimError, match="T"):
        ex.rank_frontier(*_rank_inputs(), {"P", "Q", "T"})


def test_rank_inputs_are_each_benchmarks_median_and_tail_over_the_ranked_panel():
    frame = _scores()
    medians, tails = ex.rank_inputs({"b1": frame, "b2": frame.iloc[::-1]})

    assert list(medians.index) == ["A", "B", "C", "D", "E"]  # the control is a floor
    assert list(medians.columns) == ["b1", "b2"]
    assert medians.loc["B", "b2"] == 0.4 and tails.loc["D", "b1"] == 0.0


def test_rank_inputs_over_benchmarks_that_rank_different_panels_fail():
    frame = _scores()

    with pytest.raises(ex.CaptionClaimError, match="E"):
        ex.rank_inputs({"b1": frame, "b2": frame[frame["model"] != "E"]})


def _write_inputs(
    root: Path,
    *,
    join_croma: list[float] | None = None,
    run_croma: list[float] | None = None,
) -> None:
    """A Camelyon run and the APD join under ``root``, in registry spellings.

    ``croma`` is the margin both files carry by default; ``run_croma`` overrides what the run
    stores (e.g. the same margins on the ratio scale) and ``join_croma`` what the join stores.
    """
    from _panel_macros import APD_JOINED_CSV
    from paper_manifest import by_prefix

    models = ["Alpha", "RudolfV 2", "Gamma", "DINOv2-B"]
    croma = [0.3, -0.1, 0.2, -0.4]
    metrics = root / by_prefix("Camelyon").metrics_rel
    metrics.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": models,
            "croma": run_croma or croma,
            "croma_ltm_alpha": [-0.2, -0.5, 0.1, -0.9],
        }
    ).to_csv(metrics, index=False)
    joined = root / APD_JOINED_CSV
    joined.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset": ["camelyon"] * 4 + ["tolkach"],
            "model": models + ["Alpha"],
            "croma": (join_croma or croma) + [0.9],
            "nipd_id": [-0.01, -0.05, -0.02, -0.2, 0.0],
            "nipd_ood": [-0.02, -0.04, -0.03, -0.3, 0.0],
        }
    ).to_csv(joined, index=False)


def test_scores_join_the_runs_median_and_tail_to_its_benchmarks_nipd(tmp_path):
    from paper_manifest import by_prefix

    _write_inputs(tmp_path)

    scores = ex.load_scores(tmp_path, by_prefix("Camelyon")).set_index("model")

    assert list(scores.index) == ["Alpha", "RudolfV-2", "Gamma", "DINOv2-B"]
    assert scores.loc["RudolfV-2", "croma"] == -0.1
    assert scores.loc["Gamma", "croma_ltm_alpha"] == 0.1
    assert scores.loc["Alpha", "nipd_id"] == -0.01  # Camelyon's row, not Tolkach's
    assert scores.loc["DINOv2-B", "nipd_ood"] == -0.3


def test_a_join_whose_croma_is_not_the_runs_fails_as_stale(tmp_path):
    from paper_manifest import by_prefix

    _write_inputs(tmp_path, join_croma=[0.3, -0.1, 0.25, -0.4])

    with pytest.raises(ex.CaptionClaimError, match="Gamma"):
        ex.load_scores(tmp_path, by_prefix("Camelyon"))


def test_a_ratio_scale_run_is_checked_against_the_join_on_the_margin_scale(tmp_path):
    """``loaders.load_joined`` writes the join on the margin scale whatever the run stores, so
    a ratio-scale run is compared, and returned, as margins rather than refused as stale."""
    from paper_manifest import by_prefix

    margins = [0.3, -0.1, 0.2, -0.4]
    _write_inputs(tmp_path, run_croma=[(1 + m) / (1 - m) for m in margins])

    scores = ex.load_scores(tmp_path, by_prefix("Camelyon")).set_index("model")

    assert list(scores.index) == ["Alpha", "RudolfV-2", "Gamma", "DINOv2-B"]
    assert scores["croma"].tolist() == pytest.approx(margins, abs=1e-12)


def test_a_missing_run_or_join_fails_the_build(tmp_path):
    from _panel_macros import APD_JOINED_CSV
    from paper_manifest import by_prefix

    with pytest.raises(FileNotFoundError, match="metrics.csv"):
        ex.load_scores(tmp_path, by_prefix("Camelyon"))
    _write_inputs(tmp_path)
    (tmp_path / APD_JOINED_CSV).unlink()
    with pytest.raises(FileNotFoundError, match="apd_metrics_joined"):
        ex.load_scores(tmp_path, by_prefix("Camelyon"))


# --- rendering the sensitivity --------------------------------------------------------------


def _render_sensitivity() -> list[str]:
    changed = ex.BenchmarkSensitivity(
        "Changed", ex.sensitivity(_scores(), drop={"A"}, keep={"C", "D", "E"})
    )
    unchanged = ex.BenchmarkSensitivity(
        "Unchanged", ex.sensitivity(_scores(), drop=set(), keep={"C", "D", "E"})
    )
    ranks = (
        ex.RankVariant("full", ("P", "Q", "R", "S"), ("R", "Q", "P")),
        ex.RankVariant("drop-exposed", ("P", "Q", "R"), ("R", "Q")),
        ex.RankVariant("disclosed-unexposed", ("P", "Q", "R"), ("R", "Q")),
    )
    tex = ex.render_drop_exposed([changed, unchanged], ranks, [("output/x.csv", "cd34")])
    return tex.splitlines()


def _row(lines: list[str], group: str, offset: int) -> list[str]:
    start = next(i for i, ln in enumerate(lines) if ln.startswith(group + " &"))
    return [c.strip() for c in lines[start + offset].removesuffix(r"\\").split("&")]


def test_each_benchmark_reports_n_both_rhos_with_ci_and_p_the_frontier_and_top_three():
    lines = _render_sensitivity()

    assert _row(lines, "Changed", 0) == [
        "Changed",
        "All ranked",
        "5",
        r"$1.00$ $[1.00, 1.00]$, $p=0.017$",
        r"$-1.00$ $[-1.00, -1.00]$, $p=0.017$",
        "D, B, A",
        "A, B, C",
    ]
    assert _row(lines, "Changed", 1)[1:3] == ["Drop exposed", "4"]
    assert _row(lines, "Changed", 1)[3] == r"$1.00$ $[1.00, 1.00]$, $p=0.083$"
    assert _row(lines, "Changed", 1)[5:] == ["D, B", "B, C, D"]
    assert _row(lines, "Changed", 2)[1:3] == ["Disclosed, not exposed", "3"]
    assert _row(lines, "Changed", 2)[4:] == [
        r"$-1.00$ $[-1.00, -1.00]$, $p=0.333$",
        "D, C",
        "C, D, E",
    ]


def test_a_benchmark_with_nothing_exposed_states_it_rather_than_omitting_the_row():
    lines = _render_sensitivity()

    drop = _row(lines, "Unchanged", 1)
    assert drop[1:3] == ["Drop exposed", "5"]
    assert "none exposed" in drop[3] and "identical to all ranked" in drop[3]


def test_the_cross_benchmark_rank_frontier_is_reported_per_variant():
    lines = _render_sensitivity()

    assert _row(lines, "Mean rank", 0)[1:3] == ["All ranked", "4"]
    assert _row(lines, "Mean rank", 0)[5] == "R, Q, P"
    assert _row(lines, "Mean rank", 1)[1:3] == ["Drop exposed", "3"]
    assert _row(lines, "Mean rank", 1)[5] == "R, Q"


def test_the_sensitivity_caption_says_dropping_moves_no_remaining_score():
    tex = "\n".join(_render_sensitivity())

    assert r"\label{tab:drop-exposed}" in tex
    assert "does not change any remaining encoder's score" in tex
    assert "% input output/x.csv sha256=cd34" in tex


def test_p_values_below_a_thousandth_are_bounded_not_rounded_to_zero():
    assert ex.p_text(0.00009999) == "p<0.001"
    assert ex.p_text(0.0012) == "p=0.001"
    assert ex.p_text(0.5) == "p=0.500"


def test_fingerprints_are_sha256_of_each_input_relative_to_the_root(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")

    import hashlib

    assert ex.fingerprints(tmp_path, ["a.csv"]) == [
        ("a.csv", hashlib.sha256(b"x\n1\n").hexdigest())
    ]


def test_an_input_that_changed_after_it_was_fingerprinted_is_refused(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    recorded = ex.fingerprints(tmp_path, ["a.csv"])
    (tmp_path / "a.csv").write_text("x\n2\n")

    with pytest.raises(ex.CaptionClaimError, match="a.csv"):
        ex.check_fingerprints(tmp_path, recorded)


def test_a_missing_input_fails_the_whole_build(tmp_path):
    with pytest.raises(FileNotFoundError):
        ex.build(tmp_path)


#: A six-encoder panel in registry spellings, one (median, LTM_10) per encoder on every
#: benchmark. UNI leads both axes and Mascaret, which leads the checkout's mean-rank panel,
#: trails both; among the disclosed-unexposed encoders RudolfV 2 leads.
_PANEL = {
    "UNI": (0.6, 0.1),
    "RudolfV 2": (0.5, 0.0),
    "CONCH": (0.4, -0.1),
    "CONCHv1.5": (0.3, -0.2),
    "GPFM": (0.2, -0.3),
    "Mascaret": (0.1, -0.4),
    "DINOv2-B": (-0.2, -0.9),
}


def _write_panel(root: Path) -> None:
    """The Camelyon, TCGA-4x4 and Tolkach-ESCA runs and their APD join, under ``root`` only."""
    from _panel_macros import APD_JOINED_CSV
    from loaders import DATASETS
    from paper_manifest import by_prefix

    models = list(_PANEL)
    croma = [_PANEL[m][0] for m in models]
    joined = []
    for prefix in ex.SENSITIVITY_PREFIXES:
        entry = by_prefix(prefix)
        metrics = root / entry.metrics_rel
        metrics.parent.mkdir(parents=True)
        pd.DataFrame(
            {"model": models, "croma": croma, "croma_ltm_alpha": [_PANEL[m][1] for m in models]}
        ).to_csv(metrics, index=False)
        (dataset,) = [k for k, spec in DATASETS.items() if spec["benchmark"] == entry.benchmark]
        joined.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "model": models,
                    "croma": croma,
                    "nipd_id": [-0.01, -0.03, -0.02, -0.05, -0.04, -0.06, -0.2],
                    "nipd_ood": [-0.02, -0.01, -0.04, -0.03, -0.06, -0.05, -0.3],
                }
            )
        )
    path = root / APD_JOINED_CSV
    path.parent.mkdir(parents=True)
    pd.concat(joined).to_csv(path, index=False)


def test_the_mean_rank_rows_read_the_runs_under_the_build_root(tmp_path):
    _write_panel(tmp_path)

    built = ex.build(tmp_path)

    lines = built.drop_exposed_tex.splitlines()
    assert _row(lines, "Mean rank", 0)[1:3] == ["All ranked", "6"]
    assert _row(lines, "Mean rank", 0)[5] == "UNI"
    # GPFM and Mascaret are exposed on TCGA-4x4 (GPFM on Camelyon too), so both leave.
    assert _row(lines, "Mean rank", 1)[1:3] == ["Drop exposed", "4"]
    assert _row(lines, "Mean rank", 1)[5] == "UNI"
    assert _row(lines, "Mean rank", 2)[1:3] == ["Disclosed, not exposed", "3"]
    assert _row(lines, "Mean rank", 2)[5] == "RudolfV-2"
    assert r"\newcommand{\TileRankFrontierDisclosedUnexposedModels}{RudolfV-2}" in built.macros


def test_a_run_replaced_after_its_scores_were_read_is_refused(tmp_path, monkeypatch):
    """The header must hash the bytes the scores came from: an input is fingerprinted before
    it is read, so a replacement between the read and the hash cannot pass as the source."""
    _write_panel(tmp_path)
    read = ex.load_scores

    def read_then_replace(root, entry):
        frame = read(root, entry)
        if entry.prefix == "Camelyon":
            path = Path(root) / entry.metrics_rel
            path.write_text(path.read_text().replace("0.6", "0.65"))
        return frame

    monkeypatch.setattr(ex, "load_scores", read_then_replace)

    with pytest.raises(ex.CaptionClaimError, match="changed while"):
        ex.build(tmp_path)


def test_the_full_panel_mean_rank_frontier_is_the_overview_figures(monkeypatch):
    """tab:drop-exposed and the mean-rank overview figure name the same frontier."""
    from _rank_pareto import load as load_rank_pareto

    monkeypatch.chdir(ROOT)  # ``_cross_benchmark.load`` reads its runs relative to the cwd
    try:
        built = ex.build(ROOT)
        expected = load_rank_pareto().frontier
    except FileNotFoundError as missing:
        pytest.skip(f"output/ is not tracked: {missing}")

    lines = built.drop_exposed_tex.splitlines()
    assert _row(lines, "Mean rank", 0)[5] == ", ".join(expected)


# --- macros ----------------------------------------------------------------------------------


def _macros() -> list[str]:
    md = _metadata(
        {"model": "T1", "corpus_domains": "tcga"},
        {"model": "T2", "disclosed": "no"},
        {"model": "S1", "panel": "slide", "corpus_domains": "tcga;panda"},
    )
    blocks = {
        "Bench": ex.BenchmarkSensitivity(
            "Bench", ex.sensitivity(_scores(), drop={"A"}, keep={"C", "D", "E"})
        )
    }
    ranks = (
        ex.RankVariant("full", ("P", "Q", "R", "S"), ("R", "Q", "P")),
        ex.RankVariant("drop-exposed", ("P", "Q", "R"), ("R", "Q")),
        ex.RankVariant("disclosed-unexposed", ("P", "Q"), ("Q",)),
    )
    return ex.exposure_macros(ex.exposure_matrix(md), blocks, ranks)


def test_count_macros_give_exposed_and_undisclosed_per_dataset_and_panel():
    lines = _macros()

    assert r"\newcommand{\ExposureTileTcgaNExposed}{1}" in lines
    assert r"\newcommand{\ExposureTileTcgaNUndisclosed}{1}" in lines
    assert r"\newcommand{\ExposureTileTolkachEscaNExposed}{0}" in lines
    assert r"\newcommand{\ExposureSlidePandaNExposed}{1}" in lines
    assert r"\newcommand{\ExposureSlideParNUndisclosed}{0}" in lines


def test_sensitivity_macros_carry_n_rho_ci_p_and_frontier_numbers_not_verdicts():
    lines = _macros()

    assert r"\newcommand{\BenchDropExposedN}{4}" in lines
    assert r"\newcommand{\BenchDropExposedNipdIdRho}{$1.00$}" in lines
    assert r"\newcommand{\BenchDropExposedNipdIdCi}{$[1.00, 1.00]$}" in lines
    assert r"\newcommand{\BenchDropExposedNipdIdP}{$p=0.083$}" in lines
    assert r"\newcommand{\BenchDisclosedUnexposedNipdOodRho}{$-1.00$}" in lines
    assert r"\newcommand{\BenchDisclosedUnexposedParetoFrontierModels}{D, C}" in lines
    assert r"\newcommand{\BenchDisclosedUnexposedParetoFrontierNModels}{2}" in lines
    assert r"\newcommand{\TileRankFrontierDropExposedModels}{Q, R}" in lines
    assert r"\newcommand{\TileRankFrontierDropExposedNKept}{3}" in lines
    assert r"\newcommand{\TileRankFrontierDisclosedUnexposedNModels}{1}" in lines


def test_no_macro_is_defined_twice():
    names = [ln.split("}")[0] for ln in _macros() if ln.startswith(r"\newcommand")]

    assert len(names) == len(set(names))


# --- the committed metadata --------------------------------------------------------------

#: The tile-panel grouping on TCGA, as tabulated by hand from model_metadata.csv.
TCGA_EXPOSED = {
    "H0-mini",
    "Midnight-12k",
    "Prost40M",
    "Phikon",
    "Phikon-v2",
    "mSTAR",
    "GPFM",
    "MUSK",
    "GenBio-PathFM",
    "Mascaret",
    "Phaet",
}
UNDISCLOSED = {
    "Virchow",
    "Virchow2",
    "UNI",
    "UNI2-h",
    "H-optimus-0",
    "H-optimus-1",
    "Prov-GigaPath",
    "Hibou-B",
    "Hibou-L",
    "Mettle",
}
DISCLOSED_UNEXPOSED = {"CONCH", "CONCHv1.5", "RudolfV-2", "RudolfV-2-B", "RudolfV-2-S"}


def _committed() -> "ex.ExposureMatrix":
    return ex.exposure_matrix(ex.load_metadata())


def test_tile_rows_come_first_then_the_slide_panel_then_the_control():
    rows = _committed().rows

    panels = [row.panel for row in rows]
    assert panels == ["tile"] * 26 + ["slide"] * 5 + ["control"]
    assert rows[-1].model == "DINOv2-B"
    # Published spellings: the registry's "RudolfV 2" is printed "RudolfV-2".
    assert {row.model for row in rows if row.panel == "tile"} == (
        TCGA_EXPOSED | UNDISCLOSED | DISCLOSED_UNEXPOSED
    )


def test_committed_tile_panel_groups_on_tcga_as_tabulated():
    matrix = _committed()

    by_status: dict[str, set[str]] = {}
    for row in matrix.rows:
        if row.panel == "tile":
            by_status.setdefault(matrix.status(row.model, "tcga"), set()).add(row.model)

    assert by_status == {
        "exposed": TCGA_EXPOSED,
        "undisclosed": UNDISCLOSED,
        "not exposed": DISCLOSED_UNEXPOSED,
    }


def test_committed_exposure_outside_tcga_is_gpfm_on_camelyon_and_nobody_on_tolkach_or_par():
    matrix = _committed()
    tile = [row.model for row in matrix.rows if row.panel == "tile"]

    def exposed(domain: str) -> set[str]:
        return {m for m in tile if matrix.status(m, domain) == "exposed"}

    assert exposed("camelyon") == {"GPFM"}
    assert exposed("panda") == {"GPFM"}
    assert exposed("tolkach-esca") == set()
    assert exposed("par") == set()


def test_committed_institution_markers_name_the_shared_institution_and_carry_a_source():
    matrix = _committed()

    marked = {
        row.model: {(m.institution, m.source) for m in matrix.institutions(row.model)}
        for row in matrix.rows
        if matrix.institutions(row.model)
    }

    assert marked == {
        # Prost40M's LEOPARD/RUMC prostate data; Radboudumc runs the RUMC/radboud centres.
        "Prost40M": {("rumc", "grisi2026bcr")},
        # The Charite + LMU corpus; CHA is one of Tolkach-ESCA's three cohorts.
        "RudolfV-2": {("charite", "rudolfv2")},
        "RudolfV-2-B": {("charite", "rudolfv2")},
        "RudolfV-2-S": {("charite", "rudolfv2")},
        # MSKCC's own archive; MSKCC is a TCGA-2x2 tissue-source site.
        "Virchow": {("mskcc", "Vorontsov2024-te")},
        "Virchow2": {("mskcc", "zimmermann2024")},
        "PRISM2": {("mskcc", "prism2")},
    }


def test_every_institution_centre_is_a_label_its_benchmark_view_carries():
    """The map is curated by hand; a renamed or mistyped centre would silently mark nothing."""
    views = pytest.importorskip("views")
    for token, institution in ex.INSTITUTIONS.items():
        for benchmark, label in institution.centres:
            try:
                labels = set(views.load_view(benchmark).eval_manifest["confounder"].astype(str))
            except FileNotFoundError:
                pytest.skip(f"{benchmark}: view not available (output/ is not tracked)")
            assert label in labels, f"{token}: {benchmark} has no centre {label!r}"


def _tile_roster(matrix) -> set[str]:
    return {row.model for row in matrix.rows if row.panel == "tile"}


def test_matrix_daggers_and_drop_sets_agree_on_every_benchmark_with_an_exposure_domain():
    """One resolver: each figure/table dagger set is what the matrix flags for that domain, and
    each drop set is the matrix's exposed cells in the benchmark's scored-dataset column."""
    from _distributions import metadata_exposed_models
    from generate_results_table import benchmark_exposed
    from paper_manifest import TABLES

    matrix = _committed()
    roster = _tile_roster(matrix)

    checked = set()
    for entry in TABLES:
        if not entry.exposure_domain:
            continue
        daggers = set(metadata_exposed_models(entry, roster))
        assert daggers == matrix.flagged(entry.exposure_domain, roster), entry.prefix
        if entry.mark_exposure_in_table:
            table = benchmark_exposed(entry, pd.DataFrame({"model": sorted(roster)}))
            assert table == daggers, entry.prefix
        checked.add(entry.prefix)
    assert checked == {"Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach"}

    for prefix, daggered in (("Camelyon", True), ("TcgaFourByFour", True), ("Tolkach", False)):
        entry = next(t for t in TABLES if t.prefix == prefix)
        drop = matrix.drop_exposed(entry.benchmark, roster)
        column = ex.SCORED_DATASET[entry.benchmark]
        assert drop == {m for m in roster if matrix.status(m, column) == "exposed"}
        if daggered:
            assert drop == set(metadata_exposed_models(entry, roster)), prefix

    assert matrix.drop_exposed("pathorob-camelyon", roster) == {"GPFM"}
    assert matrix.drop_exposed("pathorob-tcga-4x4", roster) == TCGA_EXPOSED
    # The scored Tolkach-ESCA cohorts are in no disclosed corpus; the RudolfV-2 daggers mark
    # possible institutional overlap with CHA, which is flagged and never dropped.
    assert matrix.drop_exposed("pathorob-tolkach-esca", roster) == set()
    assert matrix.flagged("charite", roster) == {"RudolfV-2", "RudolfV-2-B", "RudolfV-2-S"}


def test_drop_exposed_and_undisclosed_keeps_only_the_disclosed_unexposed_encoders():
    matrix = _committed()
    roster = _tile_roster(matrix)

    keep = matrix.disclosed_unexposed("pathorob-tcga-4x4", roster)

    assert keep == DISCLOSED_UNEXPOSED
    assert matrix.disclosed_unexposed("pathorob-camelyon", roster) == (
        DISCLOSED_UNEXPOSED | TCGA_EXPOSED - {"GPFM"}
    )
