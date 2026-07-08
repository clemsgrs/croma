"""Tests for the benchmark's dataset-level tau-scale summary lines."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts" / "bench"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark import (
    _tau_assessment_from_payload,
    _tau_assessment_to_payload,
    _tau_summary_lines,
)
from croma.metrics.tau import TauAssessment


def _assessment(regime: str, recommended: float) -> TauAssessment:
    return TauAssessment(
        tau=0.2,
        n_typed=100,
        median_typed_distance=recommended,
        recommended_tau=recommended,
        low=recommended / 4.0,
        high=recommended * 4.0,
        factor=4.0,
        regime=regime,
    )


def test_no_lines_without_assessments() -> None:
    assert _tau_summary_lines(0.2, {}) == []


def test_reports_recommended_range_and_no_offscale_when_all_principled() -> None:
    assessments = {
        "A": _assessment("principled", 0.18),
        "B": _assessment("principled", 0.31),
    }
    lines = _tau_summary_lines(0.2, assessments)
    assert len(lines) == 1
    assert "0.18" in lines[0] and "0.31" in lines[0]
    assert "off-scale" not in lines[0]


def test_flags_offscale_models_by_name() -> None:
    assessments = {
        "good": _assessment("principled", 0.2),
        "sharp": _assessment("too_sharp", 0.5),
        "flat": _assessment("too_flat", 0.02),
    }
    lines = _tau_summary_lines(0.2, assessments)
    offscale_line = next(line for line in lines if "off-scale" in line)
    assert "2/3 model" in offscale_line
    assert "sharp" in offscale_line and "flat" in offscale_line
    assert "good" not in offscale_line


def test_undetermined_excluded_from_recommended_range() -> None:
    assessments = {
        "A": _assessment("principled", 0.25),
        "B": _assessment("undetermined", float("nan")),
    }
    lines = _tau_summary_lines(0.2, assessments)
    assert "0.25" in lines[0]
    assert "nan" not in lines[0].lower()


def test_tau_assessment_payload_round_trips() -> None:
    original = _assessment("too_flat", 0.0358)
    restored = _tau_assessment_from_payload(_tau_assessment_to_payload(original))
    assert restored == original


def test_tau_assessment_from_payload_rejects_incomplete_or_invalid() -> None:
    assert _tau_assessment_from_payload(None) is None
    assert _tau_assessment_from_payload({}) is None
    # Missing a required field.
    payload = _tau_assessment_to_payload(_assessment("too_sharp", 0.5))
    payload.pop("regime")
    assert _tau_assessment_from_payload(payload) is None
