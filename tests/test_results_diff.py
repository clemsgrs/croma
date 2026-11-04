"""The rename guard: two ``results/`` trees compared by registry identity.

A published-name change must move no number. ``scripts/tools/diff_results.py`` joins two
exports on the registry identity behind each printed name, so a renamed model is the same
row on both sides, and reports the largest numeric change and every other difference.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "tools"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diff_results as dr  # noqa: E402

COHORT = "model,is_control,croma,croma_ltm10\nUNI,False,0.2,-0.1\n{name},False,0.11,-0.02\n"
NIPD = {"cohorts": [{"slug": "camelyon", "models": [{"model": "{name}", "nipd": 0.25}]}]}
DISTRIBUTIONS = {"cohorts": {"camelyon": {"lo": -0.8, "models": {"{name}": [0, 3, 1]}}}}


def _tree(name: str, *, croma: str = "0.11", bins=(0, 3, 1)) -> dict[str, str]:
    distributions = json.loads(json.dumps(DISTRIBUTIONS).replace("{name}", name))
    distributions["cohorts"]["camelyon"]["models"][name] = list(bins)
    return {
        "results/camelyon.csv": COHORT.format(name=name).replace("0.11", croma),
        "results/nipd.json": json.dumps(NIPD).replace("{name}", name),
        "results/distributions.json": json.dumps(distributions),
    }


def test_a_rename_is_the_same_row_and_moves_nothing():
    comparison = dr.compare(_tree("Mettle"), _tree("Mettle-1 (Pre-release)"))

    assert comparison.max_abs_delta == 0.0
    assert comparison.problems == []
    assert comparison.renamed == {("Mettle", "Mettle-1 (Pre-release)")}
    assert comparison.name_only


def test_a_moved_number_is_reported_with_its_size():
    comparison = dr.compare(_tree("Mettle"), _tree("Mettle-1 (Pre-release)", croma="0.12"))

    assert abs(comparison.max_abs_delta - 0.01) < 1e-12
    assert not comparison.name_only


def test_a_moved_histogram_count_is_reported():
    comparison = dr.compare(_tree("Mettle"), _tree("Mettle-1 (Pre-release)", bins=(0, 2, 2)))

    assert comparison.max_abs_delta == 1.0
    assert not comparison.name_only


def test_a_name_that_maps_to_a_different_identity_is_a_missing_row_not_a_rename():
    comparison = dr.compare(_tree("Mettle"), _tree("Mettle-2"))

    assert comparison.problems
    assert not comparison.name_only


def test_two_json_keys_for_one_identity_are_reported_not_merged():
    before = {"models": {"Mettle": [0, 3, 1]}}
    after = {"models": {"Mettle": [999, 0, 0], "Mettle-1 (Pre-release)": [0, 3, 1]}}

    comparison = dr.compare(
        {"results/distributions.json": json.dumps(before)},
        {"results/distributions.json": json.dumps(after)},
    )

    assert not comparison.name_only
    assert any("two keys for 'Mettle'" in problem for problem in comparison.problems)


def test_provenance_differs_only_in_its_export_date_and_checksums_by_design():
    before = {"exported": "2026-10-05", "files": {"results/a.csv": "x"}, "k": {"UNI": 3}}
    after = {"exported": "2026-11-04", "files": {"results/a.csv": "y"}, "k": {"UNI": 3}}

    comparison = dr.compare(
        {"results/PROVENANCE.json": json.dumps(before)},
        {"results/PROVENANCE.json": json.dumps(after)},
    )

    assert comparison.name_only
    after["k"]["UNI"] = 5
    moved = dr.compare(
        {"results/PROVENANCE.json": json.dumps(before)},
        {"results/PROVENANCE.json": json.dumps(after)},
    )
    assert moved.max_abs_delta == 2.0
