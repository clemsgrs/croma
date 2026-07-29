"""APD: the reduction's behaviour, plus the frozen parity fixture that keeps it faithful.

``apd`` is PathoROB's metric, not croma's, so the tests come in two kinds. The behavioural
ones below drive the public function on small hand-built matrices and assert values that
can be read off by hand. The parity ones drive it on matrices PathoROB published and assert
the values PathoROB published for them -- neither side computed here, so a mismatch is
evidence that croma drifted from upstream rather than a snapshot going stale.

The parity fixture is deliberately hermetic: it needs no PathoROB checkout, no
``PATHOROB_ROOT`` and takes no skip path, because CI is the branch that gates merges and a
test that silently skips there protects nothing. See
``tests/fixtures/pathorob_apd_parity.json`` for its provenance and recapture procedure.

A third, smaller kind covers the obligations that come with redistributing someone else's
code: the upstream licence stays in the vendored file and the attribution stays in
``NOTICE``. Those are conditions of the BSD 3-Clause grant, so they are asserted rather
than trusted to review.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from croma import apd

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "src" / "croma" / "downstream" / "_pathorob.py"
NOTICE = ROOT / "NOTICE"

PARITY_PATH = Path(__file__).resolve().parent / "fixtures" / "pathorob_apd_parity.json"
PARITY = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
PARITY_CASES = [pytest.param(case, id=case["name"]) for case in PARITY["cases"]]


def test_apd_is_zero_when_no_split_degrades() -> None:
    # Every confounded split scores exactly what the balanced baseline scores, so the
    # ratio is 1 in every cell and the drop is zero.
    accuracies = np.array(
        [
            [0.80, 0.80],
            [0.80, 0.80],
            [0.80, 0.80],
        ],
        dtype=float,
    )

    assert apd(accuracies) == pytest.approx(0.0)


def test_apd_is_the_mean_relative_change_against_the_balanced_split() -> None:
    # Baseline 0.90, both confounded splits 0.72: every ratio is 0.8, so APD is -0.2.
    accuracies = np.array(
        [
            [0.90, 0.90],
            [0.72, 0.72],
            [0.72, 0.72],
        ],
        dtype=float,
    )

    assert apd(accuracies) == pytest.approx(-0.2)


def test_apd_takes_the_ratio_per_replicate_and_averages_afterwards() -> None:
    # The reduction order is visible, not cosmetic. Replicate 0 has a baseline of 0.20
    # and a confounded split of 0.30 (ratio 1.5); replicate 1 has 0.90 and 0.75 (ratio
    # 0.833). Mean-of-ratios -- PathoROB's order, which apd keeps -- averages +0.5 and
    # -0.167 to +1/6, so it reports that the confounder *helped*. Ratio-of-means, the
    # order nIPD uses, would average the replicates first (0.55 against 0.55) and report
    # -1/22. The two disagree in sign here, so this pins which one apd runs.
    accuracies = np.array(
        [
            [0.20, 0.90],
            [0.30, 0.75],
        ],
        dtype=float,
    )

    assert apd(accuracies) == pytest.approx(1 / 6)
    assert apd(accuracies) > 0.0


def test_apd_orders_two_matrices_by_how_much_accuracy_they_lose() -> None:
    baseline = [0.90, 0.90]
    mild = np.array([baseline, [0.85, 0.85], [0.85, 0.85]], dtype=float)
    severe = np.array([baseline, [0.60, 0.60], [0.60, 0.60]], dtype=float)

    assert apd(severe) < apd(mild) < 0.0


def test_apd_returns_a_plain_float() -> None:
    value = apd([[0.90, 0.90], [0.70, 0.70]])

    assert isinstance(value, float)
    assert np.isfinite(value)


def test_apd_accepts_a_plain_nested_list() -> None:
    # Anything array-like is coerced -- a caller who assembled the sweep in Python lists
    # should not have to convert it first.
    assert apd([[0.90, 0.90], [0.72, 0.72]]) == pytest.approx(-0.2)


def test_apd_rejects_a_matrix_with_no_confounded_split() -> None:
    # A baseline row on its own averages an empty set of ratios; numpy would answer nan.
    baseline_only = np.array([[0.80, 0.80]], dtype=float)

    with pytest.raises(ValueError, match="confounded split"):
        apd(baseline_only)


def test_apd_rejects_a_matrix_that_is_not_two_dimensional() -> None:
    one_split_flat = np.array([0.80, 0.70, 0.60], dtype=float)

    with pytest.raises(ValueError, match="2-D"):
        apd(one_split_flat)


def test_apd_rejects_a_matrix_with_no_replicates() -> None:
    no_iterations = np.zeros((3, 0), dtype=float)

    with pytest.raises(ValueError, match="replicate"):
        apd(no_iterations)


def test_apd_rejects_a_matrix_holding_a_non_finite_score() -> None:
    accuracies = np.array([[0.80, np.nan], [0.70, 0.70]], dtype=float)

    with pytest.raises(ValueError, match="finite"):
        apd(accuracies)


def test_apd_rejects_a_baseline_replicate_that_is_not_positive() -> None:
    # Mean-of-ratios divides by each replicate's own baseline, so it is a single
    # replicate -- not the mean -- that has to be positive. A zero has no ratio at all;
    # a negative one is not a balanced accuracy and would invert that replicate's sign.
    # This is the reduction's domain running out, not a judgement about precision.
    zero_baseline = np.array([[0.80, 0.00], [0.70, 0.70]], dtype=float)
    negative_baseline = np.array([[0.80, -0.10], [0.70, 0.70]], dtype=float)

    for accuracies in (zero_baseline, negative_baseline):
        with pytest.raises(ValueError, match="baseline"):
            apd(accuracies)


@pytest.mark.parametrize("case", PARITY_CASES)
def test_apd_reproduces_pathorobs_published_value_exactly(case: dict) -> None:
    # Bit-identity, not approximation: the whole reason the reduction is vendored rather
    # than reimplemented is that a number reported as "APD" has to be the number PathoROB
    # would report. `approx` here would let a slow drift through.
    assert apd(case["accuracies"]) == case["apd"]


def test_apd_parity_fixture_covers_more_than_one_split_count() -> None:
    # A single matrix shape would let a reduction that collapses the wrong axis pass.
    split_counts = {len(case["accuracies"]) for case in PARITY["cases"]}

    assert len(split_counts) > 1


def test_apd_parity_fixture_records_where_it_came_from_and_how_to_recapture_it() -> None:
    # An intentional re-vendor needs a documented path, so provenance is part of the
    # fixture's contract rather than a courtesy comment.
    source = PARITY["source"]

    assert source["url"] == "https://github.com/bifold-pathomics/PathoROB"
    assert len(source["revision"]) == 40
    assert source["reduction"] == "pathorob/apd/utils.py :: compute_apd"
    assert source["accuracies_from"] and source["apd_from"]
    assert PARITY["procedure"]


def test_the_vendored_module_retains_the_upstream_licence_in_file() -> None:
    # Clause 1 of BSD 3-Clause: a source redistribution must retain the copyright notice,
    # the conditions and the disclaimer. Retaining them *in the file that carries the
    # code* is what keeps that true no matter how the file is copied onward.
    source = VENDORED.read_text(encoding="utf-8")

    assert "BSD 3-Clause License" in source
    assert "Copyright (c) 2025, BIFOLD Pathomics" in source
    assert "Redistribution and use in source and binary forms" in source
    assert "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS" in source


def test_the_vendored_module_says_it_is_frozen_and_names_its_revision() -> None:
    # The file is the only place a reader lands when they follow the import, so it has to
    # carry both halves of the deal: do not edit this, and here is what it was taken from.
    source = VENDORED.read_text(encoding="utf-8")

    assert "DO NOT EDIT" in source
    assert PARITY["source"]["revision"] in source


def test_notice_credits_the_project_the_reduction_is_vendored_from() -> None:
    notice = NOTICE.read_text(encoding="utf-8")

    assert "PathoROB" in notice
    assert "BIFOLD Pathomics" in notice
    assert "BSD 3-Clause" in notice
    assert "https://github.com/bifold-pathomics/PathoROB" in notice
