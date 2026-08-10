"""The published ``results/`` tree: how it is assembled, and that it is not stale.

Two kinds of test live here. The assembly tests drive the exporter's pure functions with
synthetic frames and never touch a run. The freshness test regenerates the whole tree from
``output/`` and asserts byte-equality with what is committed -- it is the guard that stops
a benchmark re-run from silently leaving the public site asserting numbers no run
produced. ``output/`` is git-ignored, so that test skips on any machine without the runs
(including CI) and guards on the one machine that has them.
"""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "tools"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_results as er  # noqa: E402

RESULTS = ROOT / "results"

#: The committed payload is fetched by every visitor to the results page. 200 bins x 26
#: models x 3 cohorts lands near 43 KB; the budget catches a change that quietly makes it
#: an order of magnitude bigger (per-sample values, say, or a much finer grid).
PAYLOAD_BUDGET_BYTES = 100 * 1024


# --------------------------------------------------------------------------------------
# Fixtures: minimal frames shaped like a run's metrics.csv.
# --------------------------------------------------------------------------------------


def _metrics(**columns) -> pd.DataFrame:
    """A metrics.csv-shaped frame; every column defaults to something inert."""
    models = columns.pop("model")
    n = len(models)
    frame = {
        "model": models,
        "bio_knn_bacc": [0.9] * n,
        "confounder_knn_bacc": [0.5] * n,
        "ri": [0.5] * n,
        "mari": [0.5] * n,
        "croma": [0.1] * n,
        "croma_ltm_alpha": [-0.2] * n,
        "croma_f0": [0.3] * n,
        "ri_undefined_frac": [0.25] * n,
    }
    frame.update(columns)
    return pd.DataFrame(frame)


def _cohort(models, croma, ltm) -> pd.DataFrame:
    """A built cohort table, as ``build_aggregate_table`` consumes it."""
    return pd.DataFrame({"model": models, "croma": croma, "croma_ltm10": ltm})


# --------------------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------------------


def test_croma_as_margin_passes_through_a_margin_column():
    margin = pd.Series([-0.3, 0.0, 0.4])
    pd.testing.assert_series_equal(er.croma_as_margin(margin), margin)


def test_croma_as_margin_converts_a_ratio_column():
    # A ratio of 3 is a margin of 0.5; 1.0 is the neutral point on both scales.
    converted = er.croma_as_margin(pd.Series([3.0, 1.0]))
    assert converted.tolist() == pytest.approx([0.5, 0.0])


def test_croma_as_margin_treats_an_all_unit_interval_column_as_a_margin():
    """Ambiguous columns default to margin -- the scale the manuscript reports."""
    ambiguous = pd.Series([0.1, 0.9])
    pd.testing.assert_series_equal(er.croma_as_margin(ambiguous), ambiguous)


# --------------------------------------------------------------------------------------
# Per-cohort table
# --------------------------------------------------------------------------------------


def test_cohort_table_has_the_published_columns_in_order():
    metrics = _metrics(model=["A", "B"])
    table = er.build_cohort_table(metrics)
    assert list(table.columns) == er.COHORT_COLUMNS


def test_cohort_table_sorts_by_croma_descending():
    metrics = _metrics(model=["low", "high", "mid"], croma=[0.1, 0.9, 0.5])
    table = er.build_cohort_table(metrics)
    assert table["model"].tolist() == ["high", "mid", "low"]


def test_cohort_table_derives_delta_and_support():
    metrics = _metrics(model=["A"], ri=[0.40], mari=[0.55], ri_undefined_frac=[0.30])
    row = er.build_cohort_table(metrics).iloc[0]
    assert row["delta"] == pytest.approx(0.15)
    assert row["support"] == pytest.approx(0.70)


def test_cohort_table_flags_the_control_but_ranks_it_inline():
    """The control competes in the sort; only the flag sets it apart.

    It is a floor, not a competitor, but on these cohorts it lands mid-panel -- banding it
    off beneath a rule would hide where the floor actually is. The site's footnote hangs
    on this flag instead.
    """
    metrics = _metrics(model=["A", er.CONTROL_MODEL, "B"], croma=[0.9, 0.5, 0.1])
    table = er.build_cohort_table(metrics)
    assert table["model"].tolist() == ["A", er.CONTROL_MODEL, "B"]
    assert table["is_control"].tolist() == [False, True, False]


def test_cohort_table_publishes_the_run_s_own_f0():
    """F(0) is CRoMa's, read straight off the run -- the exporter never recomputes it."""
    metrics = _metrics(model=["A", "B"], croma_f0=[0.11, 0.22], croma=[0.9, 0.1])
    table = er.build_cohort_table(metrics)
    assert table["croma_f0"].tolist() == [0.11, 0.22]


def test_cohort_table_refuses_a_run_without_a_stored_f0():
    """A run predating canonical ``croma_f0`` must fail loudly, not publish a NaN column."""
    metrics = _metrics(model=["A", "B"]).drop(columns=["croma_f0"])
    with pytest.raises(KeyError, match="croma_f0"):
        er.build_cohort_table(metrics)


# --------------------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------------------


def test_aggregate_ranks_are_the_mean_of_the_per_cohort_ranks():
    per_cohort = {
        # A is 1st, B 2nd, C 3rd on margin;  reversed on the tail.
        "one": _cohort(["A", "B", "C"], [0.9, 0.5, 0.1], [-0.9, -0.5, -0.1]),
        # A is 3rd, B 2nd, C 1st on margin;  reversed on the tail.
        "two": _cohort(["A", "B", "C"], [0.1, 0.5, 0.9], [-0.1, -0.5, -0.9]),
    }
    out = er.build_aggregate_table(per_cohort).set_index("model")
    # margin: A = mean(1, 3) = 2;  tail (1 = mildest = largest LTM): A = mean(3, 1) = 2.
    assert out.loc["A", "croma_rank"] == pytest.approx(2.0)
    assert out.loc["A", "ltm_rank"] == pytest.approx(2.0)
    assert out.loc["B", "croma_rank"] == pytest.approx(2.0)


def test_aggregate_excludes_the_natural_image_control_from_pathology_ranks():
    """The public panel is 25 ranked pathology encoders plus one unranked control.

    Put the control between two pathology models on both axes so including it would move
    B from rank 2 to rank 3. Its measurements remain visible, but it receives no rank and
    cannot enter the pathology Pareto frontier.
    """
    per_cohort = {
        "one": _cohort(
            ["A", er.CONTROL_MODEL, "B"],
            [0.9, 0.5, 0.1],
            [-0.1, -0.5, -0.9],
        )
    }

    out = er.build_aggregate_table(per_cohort).set_index("model")

    assert out.loc["B", "croma_rank"] == pytest.approx(2.0)
    assert out.loc["B", "ltm_rank"] == pytest.approx(2.0)
    assert pd.isna(out.loc[er.CONTROL_MODEL, "croma_rank"])
    assert pd.isna(out.loc[er.CONTROL_MODEL, "ltm_rank"])
    assert pd.isna(out.loc[er.CONTROL_MODEL, "mean_rank"])
    assert not out.loc[er.CONTROL_MODEL, "on_frontier"]


def test_aggregate_breaks_a_mean_rank_tie_on_the_croma_rank():
    """A and B both mean 1.5 (1st/2nd against 2nd/1st); the margin rank orders them."""
    per_cohort = {
        "one": _cohort(["A", "B", "C"], [0.9, 0.5, 0.1], [-0.5, -0.1, -0.9]),
        "two": _cohort(["A", "B", "C"], [0.9, 0.5, 0.1], [-0.5, -0.1, -0.9]),
    }
    out = er.build_aggregate_table(per_cohort)
    assert out["model"].tolist() == ["A", "B", "C"]
    assert out["croma_rank"].tolist() == [1.0, 2.0, 3.0]


def test_aggregate_marks_the_undominated_set_and_nothing_else():
    """Frontier = no other model is at least as good on both axes and better on one.

    ``best_margin`` wins on margin, ``best_tail`` wins on the tail, and ``dominated`` is
    beaten by ``best_margin`` on both -- so exactly the first two are undominated.
    """
    per_cohort = {
        "one": _cohort(
            ["best_margin", "best_tail", "dominated"],
            [0.9, 0.1, 0.5],
            [-0.5, -0.1, -0.9],
        )
    }
    out = er.build_aggregate_table(per_cohort).set_index("model")
    assert out.loc["best_margin", "on_frontier"]
    assert out.loc["best_tail", "on_frontier"]
    assert not out.loc["dominated", "on_frontier"]


def test_aggregate_publishes_the_mean_rank_beside_the_two_it_averages():
    """The aggregate rank is a reading order, so it ships with its own inputs.

    Publishing it alone would make it a composite scalar -- a number whose disagreement
    between margin and tail the reader cannot recover. Both columns it averages stay in
    the same row.
    """
    out = er.build_aggregate_table({"one": _cohort(["A", "B"], [0.9, 0.1], [-0.1, -0.9])})
    assert list(out.columns) == [
        "model",
        "is_control",
        "on_frontier",
        "mean_rank",
        "croma_rank",
        "ltm_rank",
        "croma_one",
        "ltm_one",
    ]


def test_aggregate_publishes_each_cohort_s_tail_beside_its_margin():
    """A cohort's median margin ships with the tail its second rank was taken over.

    The site prints the pair in one cell (``0.19/-0.05``), and it can only do that because
    both are in the row. Publishing the margin alone would leave the table asserting a
    two-axis reading it could not show per cohort.
    """
    out = er.build_aggregate_table(
        {"one": _cohort(["A", "B"], [0.9, 0.1], [-0.1, -0.9])}
    ).set_index("model")
    assert out.loc["A", "croma_one"] == pytest.approx(0.9)
    assert out.loc["A", "ltm_one"] == pytest.approx(-0.1)
    assert out.loc["B", "ltm_one"] == pytest.approx(-0.9)


def test_mean_rank_is_re_derivable_from_the_two_published_ranks():
    """Averaged from the *published* ranks, not the raw ones, so a reader doing the
    arithmetic on the two visible columns gets the third back exactly. Averaging the raw
    ranks would put the column off its own inputs for exactly the encoders whose two axes
    disagree -- the rows the aggregate is worth checking on.

    Exact rather than to-within-rounding because ``MEAN_RANK_PRECISION`` carries the extra
    decimal that halving a three-decimal sum produces.
    """
    per_cohort = {
        # A, B, C fixed on margin; the tail order varies so the tail rank is fractional
        # and the two axes disagree for B and C.
        "one": _cohort(["A", "B", "C"], [0.9, 0.5, 0.1], [-0.1, -0.5, -0.9]),
        "two": _cohort(["A", "B", "C"], [0.9, 0.5, 0.1], [-0.1, -0.9, -0.5]),
        "three": _cohort(["A", "B", "C"], [0.9, 0.5, 0.1], [-0.1, -0.5, -0.9]),
    }
    out = er.build_aggregate_table(per_cohort)
    assert out["croma_rank"].ne(out["ltm_rank"]).any(), "axes must disagree to be a test"
    assert out["ltm_rank"].mod(1).ne(0).any(), "a rank must be fractional to be a test"
    for _, row in out.iterrows():
        assert row["mean_rank"] == pytest.approx((row["croma_rank"] + row["ltm_rank"]) / 2)


def test_aggregate_sorts_by_mean_rank_not_by_either_axis_alone():
    """``margin_king`` leads on margin and ``allrounder`` on neither, but ``allrounder``
    has the better mean -- so a table still sorted by ``croma_rank`` would fail this."""
    per_cohort = {
        "one": _cohort(
            ["margin_king", "allrounder", "laggard"],
            [0.9, 0.5, 0.1],
            [-0.9, -0.1, -0.5],
        )
    }
    out = er.build_aggregate_table(per_cohort)
    assert out["model"].tolist() == ["allrounder", "margin_king", "laggard"]
    assert out["mean_rank"].tolist() == [1.5, 2.0, 2.5]


def test_aggregate_drops_a_model_missing_from_a_cohort():
    """A model scored on two of three cohorts has no comparable mean rank."""
    per_cohort = {
        "one": _cohort(["A", "B"], [0.9, 0.5], [-0.1, -0.5]),
        "two": _cohort(["A"], [0.9], [-0.1]),
    }
    assert er.build_aggregate_table(per_cohort)["model"].tolist() == ["A"]


# --------------------------------------------------------------------------------------
# Distribution binning
# --------------------------------------------------------------------------------------


def test_bin_distribution_conserves_count():
    values = [-0.4, -0.1, 0.0, 0.05, 0.3, 0.3, 0.9]
    assert sum(er.bin_distribution(values, -1.0, 1.0, n_bins=20)) == len(values)


def test_bin_distribution_clips_rather_than_drops_out_of_range_values():
    """The explorer answers "how many samples fall in this range"; lost tail mass would
    make it answer wrongly, which is the one thing the tail statistics exist to prevent."""
    counts = er.bin_distribution([-5.0, 5.0], 0.0, 1.0, n_bins=4)
    assert counts == [1, 0, 0, 1]


def test_bin_distribution_places_values_in_the_expected_bin():
    # Four bins over [0, 1] are [0, .25), [.25, .5), [.5, .75), [.75, 1].
    counts = er.bin_distribution([0.0, 0.3, 0.6, 0.99], 0.0, 1.0, n_bins=4)
    assert counts == [1, 1, 1, 1]


def test_bin_distribution_rejects_an_empty_range():
    with pytest.raises(ValueError, match="empty range"):
        er.bin_distribution([0.0], 1.0, 1.0)


# --------------------------------------------------------------------------------------
# The committed tree
# --------------------------------------------------------------------------------------


def _provenance() -> dict:
    return json.loads((RESULTS / "PROVENANCE.json").read_text())


def test_provenance_names_every_committed_artifact_and_no_others():
    listed = set(_provenance()["files"])
    on_disk = {f"results/{p.name}" for p in RESULTS.iterdir() if p.name != "PROVENANCE.json"}
    assert listed == on_disk


def test_provenance_checksums_match_the_committed_files():
    for path, digest in _provenance()["files"].items():
        content = (ROOT / path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == digest, f"{path} does not match its checksum"


def test_the_payload_has_the_shape_the_explorer_reads():
    """docs/_static/explorer.js reads this file directly. Nothing else type-checks the
    contract between them, so a rename here would break the widget silently -- the page
    would render, the fetch would succeed, and the histogram would be empty."""
    payload = json.loads((RESULTS / "distributions.json").read_text())
    assert {"m", "n_bins", "cohorts"} <= set(payload)
    for slug, cohort in payload["cohorts"].items():
        assert {"label", "lo", "hi", "models"} <= set(cohort), slug
        assert cohort["hi"] > cohort["lo"], slug
        for model, counts in cohort["models"].items():
            assert len(counts) == payload["n_bins"], f"{slug}/{model}"


def test_the_payload_counts_every_sample_in_the_run():
    """Binning clips out-of-range values into the edge bins rather than dropping them, so
    each encoder's histogram totals the cohort's sample count exactly. The explorer reports
    "N of M samples"; a histogram that lost mass would answer that wrongly."""
    payload = json.loads((RESULTS / "distributions.json").read_text())
    for slug, cohort in payload["cohorts"].items():
        totals = {sum(counts) for counts in cohort["models"].values()}
        assert len(totals) == 1, f"{slug}: encoders disagree on the sample count: {totals}"


def test_published_payload_stays_within_its_size_budget():
    size = (RESULTS / "distributions.json").stat().st_size
    assert size < PAYLOAD_BUDGET_BYTES, f"payload grew to {size // 1024} KB"


def test_the_readme_carries_the_generated_results_region():
    readme = (ROOT / "README.md").read_text()
    assert er.README_START in readme and er.README_END in readme
    block = readme.split(er.README_START)[1].split(er.README_END)[0]
    assert "| Model | mean rank | CRoMa rank | tail rank |" in block


def test_the_readme_cohort_cells_carry_the_tail_beside_the_margin():
    """Same pairing as the site: a caption about hidden tails over a margin-only table
    would leave its own point unillustrated."""
    aggregate = er.build_aggregate_table(
        {"camelyon": _cohort(["A", "B"], [0.9, 0.1], [-0.1, -0.9])}
    )
    block = er.render_readme(aggregate, {"camelyon": {"label": "Camelyon"}})
    block = block.split(er.README_START)[1].split(er.README_END)[0]
    assert "Camelyon<br>CRoMa/LTM₁₀" in block
    assert "| 0.90/-0.10 |" in block


def test_the_readme_table_shows_the_truncation_rule_rather_than_a_selection():
    """Cutting the panel at eight is a judgement call. Stating it mechanically -- with the
    total, and a link to the rest -- is what keeps it a rule rather than a shortlist."""
    block = (ROOT / "README.md").read_text().split(er.README_START)[1]
    assert f"Top {er.README_TOP} of 25 ranked pathology encoders" in block
    assert "DINOv2-B control is shown unranked" in block
    assert "/results/" in block


def _read_csv(name: str) -> list[str]:
    return (RESULTS / name).read_text().strip().splitlines()[1:]


def test_the_readme_region_is_replaced_without_touching_the_prose_around_it():
    """The exporter owns the block, never the file. Everything outside the markers has to
    survive an export byte for byte, or a rewrite would quietly eat hand-written text."""
    readme = (ROOT / "README.md").read_text()
    aggregate = er.build_aggregate_table(
        {"camelyon": _cohort(["A", "B"], [0.9, 0.1], [-0.1, -0.9])}
    )
    rewritten = er.render_readme(aggregate, {"camelyon": {"label": "Camelyon"}})
    assert rewritten.split(er.README_START)[0] == readme.split(er.README_START)[0]
    assert rewritten.split(er.README_END)[1] == readme.split(er.README_END)[1]


def test_provenance_does_not_checksum_the_readme():
    """It is mostly prose; a hash over it would go stale on every wording change while
    saying nothing about the numbers."""
    assert all(path.startswith("results/") for path in _provenance()["files"])


def test_every_published_cohort_has_a_committed_table():
    for cohort in er.COHORTS:
        assert (RESULTS / f"{cohort.slug}.csv").exists()


@pytest.mark.skipif(
    not all(c.metrics_csv.exists() for c in er.COHORTS),
    reason="output/ is git-ignored; the runs are absent on this machine",
)
def test_committed_results_match_a_fresh_export():
    """Re-run the export and compare. Fails when a benchmark was re-run without a
    republish -- the state in which the public site describes a run that no longer
    exists."""
    # A run predating canonical ``croma_f0`` cannot be exported at all, so there is
    # nothing to compare against: the tree is unreproducible until the benchmarks are
    # re-run. Say that, rather than failing as though the committed tree were wrong.
    stale_runs = [
        c.slug for c in er.COHORTS if "croma_f0" not in pd.read_csv(c.metrics_csv, nrows=0).columns
    ]
    if stale_runs:
        pytest.skip(f"runs predate croma_f0 ({', '.join(stale_runs)}); re-run and republish")

    for path, content in er.export().items():
        if path.endswith("PROVENANCE.json"):
            continue  # carries an export date; its checksums are compared above
        assert (
            ROOT / path
        ).read_text() == content, f"{path} is stale; run python scripts/tools/export_results.py"
