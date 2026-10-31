"""The probe-record reductions: coverage, sensitivity and specificity, and low data.

``scripts/studies/probe_reductions.py`` reduces the confounder-biased sweep's probe records
(``scripts/studies/apd/probe_records.py``) to study CSVs. These tests drive its public
reductions on hand-built arrays whose answers can be worked out by hand. They need no paper
tree, so they stay in the source distribution; the end-to-end run, which resolves each
cohort's run through ``paper_manifest``, is tested in ``test_low_data_paper.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "scripts" / "studies"
for _p in (STUDIES, STUDIES / "apd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import probe_reductions as pr  # noqa: E402


class TestCoverage:
    def test_counts_the_replicates_each_in_domain_row_was_held_out_in(self):
        # Three replicates over five rows: row 1 is held out every time, row 4 never.
        held_out = np.array([[0, 1], [1, 2], [1, 3]])

        coverage = pr.coverage(held_out, n_rows=5)

        assert coverage.counts.tolist() == [1, 3, 1, 1, 0]
        assert coverage.never_held_out == 1
        assert coverage.share_never_held_out == pytest.approx(0.2)


class TestSensitivitySpecificity:
    def test_scores_each_probe_on_its_own_held_out_rows(self):
        # One split, two replicates of four held-out rows; a binary probe's single margin
        # column points toward the positive class, and a zero margin predicts negative.
        margins = np.array([[[1.0, -1.0, -2.0, 0.0], [3.0, 0.5, -1.0, 2.0]]])[..., None]
        positive = np.array([[True, True, False, False], [True, False, False, False]])

        sensitivity, specificity = pr.sensitivity_specificity(margins, positive)

        assert sensitivity.tolist() == [[0.5, 1.0]]
        assert specificity.shape == (1, 2)
        assert specificity[0].tolist() == pytest.approx([1.0, 1 / 3])


def _binary_record(id_margins, ood_margins, *, accuracies):
    """A binary record: four in-domain rows (two tumour), two held out per replicate, and
    three out-of-domain rows (one tumour), with the balanced accuracies it stored."""
    return {
        "meta": {"biological_classes": ["normal", "tumor"]},
        # The probe is fitted on class indices into ``biological_classes``.
        "classes": np.array([0, 1]),
        "held_out": np.array([[0, 2], [1, 3]]),
        "id_label": np.array(["tumor", "tumor", "normal", "normal"]),
        "ood_label": np.array(["tumor", "normal", "normal"]),
        "margins_in_domain": np.asarray(id_margins, dtype=float)[..., None],
        "margins_out_of_domain": np.asarray(ood_margins, dtype=float)[..., None],
        "accuracies_in_domain": np.asarray(accuracies[0], dtype=float),
        "accuracies_out_of_domain": np.asarray(accuracies[1], dtype=float),
    }


class TestRecordSensitivitySpecificity:
    def test_the_disease_class_is_positive_and_each_replicate_scores_its_held_out_rows(self):
        # Replicate 0 holds out rows 0 (tumour) and 2 (normal); replicate 1 rows 1 and 3.
        # One split: replicate 0 gets both right, replicate 1 calls both rows tumour.
        record = _binary_record(
            id_margins=[[[2.0, -1.0], [1.0, 1.0]]],
            ood_margins=[[[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]],
            accuracies=([[1.0, 0.5]], [[0.75, 0.5]]),
        )

        scores = pr.record_sensitivity_specificity(record)

        sens_id, spec_id = scores["in_domain"]
        assert sens_id.tolist() == [[1.0, 1.0]]
        assert spec_id.tolist() == [[1.0, 0.0]]
        sens_ood, spec_ood = scores["out_of_domain"]
        assert sens_ood.tolist() == [[1.0, 0.0]]
        assert spec_ood.tolist() == [[0.5, 1.0]]

    def test_margins_that_do_not_reproduce_the_stored_accuracy_are_refused(self):
        record = _binary_record(
            id_margins=[[[2.0, -1.0], [1.0, 1.0]]],
            ood_margins=[[[1.0, 1.0, -1.0], [-1.0, -1.0, -1.0]]],
            accuracies=([[1.0, 0.5]], [[0.75, 0.6]]),
        )

        with pytest.raises(ValueError, match="out_of_domain"):
            pr.record_sensitivity_specificity(record)


class TestAssociation:
    def test_an_undefined_encoder_leaves_the_panel_and_is_named(self):
        import pandas as pd

        panel = pd.DataFrame(
            {
                "model": ["A", "B", "C", "D", "E"],
                "croma": [0.1, 0.2, 0.3, 0.4, 0.5],
                "nipd": [-0.5, -0.4, np.nan, -0.2, -0.1],
            }
        )

        row = pr.associate(panel, "nipd", n_boot=50, n_permutations=100, seed=0)

        assert row["n"] == 4
        assert row["undefined"] == "C"
        assert row["rho"] == pytest.approx(1.0)
        # Four encoders: the p is exact over all 24 orderings, two of which reach |rho| = 1.
        assert row["permutation"] == "exact"
        assert row["n_permutations"] == 24
        assert row["p_value"] == pytest.approx(2 / 24)


class TestNipd:
    V = [0.0, 0.5, 1.0]

    def test_a_linear_fall_to_chance_integrates_to_minus_one_half(self):
        falling = np.array([[0.9, 0.9], [0.7, 0.7], [0.5, 0.5]])

        assert pr.nipd_or_undefined(falling, self.V, chance=0.5) == pytest.approx(-0.5)

    def test_is_undefined_when_the_balanced_baseline_has_no_skill_above_chance(self):
        # A probe that calls almost everything positive has a specificity below 0.5 at
        # V = 0: there is no skill above chance to normalise the degradation by.
        below = np.array([[0.30, 0.26], [0.20, 0.20], [0.10, 0.10]])

        assert np.isnan(pr.nipd_or_undefined(below, self.V, chance=0.5))
