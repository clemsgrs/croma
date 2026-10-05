"""The exposure tinting the results-table directives apply to published tables.

The row classes are derived by a pure function so the mapping is testable without
invoking docutils; the directive walk that applies them is exercised by the ``sphinx -W``
build, whose pages carry the directives.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "docs" / "_ext"
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))

# The extension imports docutils at module level; the test job installs no docs deps.
pytest.importorskip("docutils")

import resultstable as rt  # noqa: E402


def test_exposure_row_classes_tints_exposed_rows_only():
    """The tint is binary, matching the paper's dagger: exposed rows are marked, the
    rest are the unmarked default. Colour alone carries no text, so the tinted class
    pairs with a screen-reader label."""
    exposure = {"A": True, "B": False}
    assert rt.exposure_row_classes(["A", "B"], exposure) == ["croma-exposure-exposed", None]
    assert rt.EXPOSURE_LABELS == {
        "croma-exposure-exposed": " (pretraining overlaps this cohort's source)"
    }


def test_exposure_row_classes_raises_on_a_model_without_a_state():
    """A cohort row missing from the model-level export is an inconsistency, not a blank."""
    with pytest.raises(KeyError):
        rt.exposure_row_classes(["A", "unknown"], {"A": True})


def test_exposure_map_raises_on_a_value_that_is_not_a_boolean():
    """A corrupted ``tcga_exposed`` cell must fail the ``-W`` build, not silently
    render as unmarked."""
    with pytest.raises(ValueError):
        rt._parse_flag("maybe", "tcga_exposed")


def test_the_published_export_covers_every_cohort_table_row():
    """The cohort directives join on model name; the join must be total for every cohort."""
    exposure = rt._exposure_map()
    for slug in ("camelyon", "tcga-4x4", "tolkach-esca"):
        models = [row["model"] for row in rt._read(f"{slug}.csv")]
        assert rt.exposure_row_classes(models, exposure) is not None


def test_every_cohort_table_has_an_exposure_legend_and_a_selection_state():
    """Each cohort page shades by its own source; the join with the model-level selection
    flag must be total for every tile cohort, and every cohort needs a legend wording."""
    selected = rt.selection_map()
    assert {model for model, flag in selected.items() if flag} == {"Mettle"}
    for slug in ("camelyon", "tcga-4x4", "tolkach-esca", "pcabiop"):
        assert slug in rt.EXPOSURE_LEGENDS
        exposure = rt.cohort_exposure_map(slug)
        models = [row["model"] for row in rt._read(f"{slug}.csv")]
        assert len(rt.exposure_row_classes(models, exposure)) == len(models)
    assert "aggregate" in rt.EXPOSURE_LEGENDS


def test_the_site_and_readme_word_the_selection_legend_alike():
    """The README marks with ‡ what the site shades yellow; the sentence is the same."""
    tools = str(ROOT / "scripts" / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import export_results as er

    assert er.SELECTION_LEGEND == f"‡ {rt.SELECTION_LEGEND}"
