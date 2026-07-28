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

#: The committed payload is fetched by every visitor to the results page. 200 bins x 21
#: models x 3 cohorts lands near 35 KB; the budget catches a change that quietly makes it
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
    table = er.build_cohort_table(metrics, {"A": 0.1, "B": 0.2})
    assert list(table.columns) == er.COHORT_COLUMNS


def test_cohort_table_sorts_by_croma_descending():
    metrics = _metrics(model=["low", "high", "mid"], croma=[0.1, 0.9, 0.5])
    table = er.build_cohort_table(metrics, {"low": 0.0, "high": 0.0, "mid": 0.0})
    assert table["model"].tolist() == ["high", "mid", "low"]


def test_cohort_table_derives_delta_and_support():
    metrics = _metrics(model=["A"], ri=[0.40], mari=[0.55], ri_undefined_frac=[0.30])
    row = er.build_cohort_table(metrics, {"A": 0.0}).iloc[0]
    assert row["delta"] == pytest.approx(0.15)
    assert row["support"] == pytest.approx(0.70)


def test_cohort_table_flags_the_control_but_ranks_it_inline():
    """The control competes in the sort; only the flag sets it apart.

    It is a floor, not a competitor, but on these cohorts it lands mid-panel -- banding it
    off beneath a rule would hide where the floor actually is. The site's footnote hangs
    on this flag instead.
    """
    metrics = _metrics(model=["A", er.CONTROL_MODEL, "B"], croma=[0.9, 0.5, 0.1])
    table = er.build_cohort_table(metrics, dict.fromkeys(["A", er.CONTROL_MODEL, "B"], 0.0))
    assert table["model"].tolist() == ["A", er.CONTROL_MODEL, "B"]
    assert table["is_control"].tolist() == [False, True, False]


def test_cohort_table_refuses_a_model_without_a_per_sample_distribution():
    """F(0) cannot be read off metrics.csv, so a missing one is a silent NaN column."""
    metrics = _metrics(model=["A", "B"])
    with pytest.raises(KeyError, match="B"):
        er.build_cohort_table(metrics, {"A": 0.1})


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


def test_aggregate_sorts_by_croma_rank_then_tail_rank():
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


def test_aggregate_has_no_combined_rank_column():
    """The mean of a margin rank and a tail rank is the composite scalar the two-axis
    framing exists to refuse. A column named for it must never appear."""
    out = er.build_aggregate_table({"one": _cohort(["A", "B"], [0.9, 0.1], [-0.1, -0.9])})
    assert list(out.columns) == [
        "model",
        "is_control",
        "on_frontier",
        "croma_rank",
        "ltm_rank",
        "croma_one",
    ]
    assert not any("mean" in c or "combined" in c or c == "rank" for c in out.columns)


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


def test_provenance_records_the_version_that_produced_the_numbers():
    from croma import __version__

    assert _provenance()["croma_version"] == __version__


def test_published_payload_stays_within_its_size_budget():
    size = (RESULTS / "distributions.json").stat().st_size
    assert size < PAYLOAD_BUDGET_BYTES, f"payload grew to {size // 1024} KB"


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
    for path, content in er.export().items():
        if path.endswith("PROVENANCE.json"):
            continue  # carries an export date; its checksums are compared above
        assert (
            ROOT / path
        ).read_text() == content, f"{path} is stale; run python scripts/tools/export_results.py"
