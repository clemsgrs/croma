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

import resultstable as rt  # noqa: E402


def test_exposure_row_classes_tints_two_of_the_three_states():
    """Exposed and undisclosed rows are tinted; disclosed-clean is the unmarked default.

    Colour alone carries no text, so each tinted class pairs with a screen-reader label.
    """
    exposure = {"A": "exposed", "B": "disclosed-clean", "C": "undisclosed"}
    assert rt.exposure_row_classes(["A", "B", "C"], exposure) == [
        "croma-exposure-exposed",
        None,
        "croma-exposure-undisclosed",
    ]
    assert set(rt.EXPOSURE_LABELS) == {"croma-exposure-exposed", "croma-exposure-undisclosed"}


def test_exposure_row_classes_raises_on_a_model_without_a_state():
    """A cohort row missing from the model-level export is an inconsistency, not a blank."""
    with pytest.raises(KeyError):
        rt.exposure_row_classes(["A", "unknown"], {"A": "exposed"})


def test_the_published_export_covers_every_cohort_table_row():
    """The cohort directives join on model name; the join must be total for every cohort."""
    exposure = rt._exposure_map()
    for slug in ("camelyon", "tcga-4x4", "tolkach-esca"):
        models = [row["model"] for row in rt._read(f"{slug}.csv")]
        assert rt.exposure_row_classes(models, exposure) is not None
