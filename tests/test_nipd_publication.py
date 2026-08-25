"""The committed nIPD payload is the publication boundary for every site consumer."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "results" / "nipd.json"
SUMMARY = ROOT / "results" / "nipd.csv"
EXPORTER = ROOT / "scripts" / "tools" / "export_nipd.py"


def _exporter():
    spec = importlib.util.spec_from_file_location("export_nipd", EXPORTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


def test_committed_payload_is_complete_and_valid() -> None:
    published = _payload()
    _exporter().validate_payload(published)

    assert published["schema_version"] == 1
    assert [cohort["slug"] for cohort in published["cohorts"]] == [
        "camelyon",
        "tcga-4x4",
        "tolkach-esca",
        "pcabiop",
    ]
    assert "prostate" not in PAYLOAD.read_text(encoding="utf-8").lower()
    for cohort in published["cohorts"]:
        assert cohort["cramers_v"][0] == 0.0
        assert cohort["cramers_v"][-1] == 1.0
        assert {regime for model in cohort["models"] for regime in model["regimes"]} == {
            "id",
            "ood",
        }

    for cohort in published["cohorts"][:3]:
        control = [model for model in cohort["models"] if model["is_control"]]
        assert [(model["model"], model["panel"], model["ranked"]) for model in control] == [
            ("DINOv2-B", "tile", False)
        ]
        assert "RudolfV-2" in [model["model"] for model in cohort["models"]]
        assert "RudolfV 2" not in [model["model"] for model in cohort["models"]]
    assert all(model["model"] != "DINOv2-B" for model in published["cohorts"][3]["models"])


def test_representative_values_have_an_independent_float_basis() -> None:
    published = _payload()
    camelyon = next(cohort for cohort in published["cohorts"] if cohort["slug"] == "camelyon")
    conch = next(model for model in camelyon["models"] if model["model"] == "CONCH")
    id_result = conch["regimes"]["id"]

    # Frozen manuscript-derived values, not recomputed from the implementation under test.
    assert camelyon["chance"] == 0.5
    assert id_result["baseline_balanced_accuracy"] == pytest.approx(0.9714166667)
    assert id_result["baseline_skill"] == pytest.approx(0.4714166667)
    assert id_result["nipd"] == pytest.approx(-0.0427315084)

    # Independently apply the trapezoidal definition to the exported float basis.
    area = sum(
        (right_v - left_v) * (left_y + right_y) / 2
        for left_v, right_v, left_y, right_y in zip(
            camelyon["cramers_v"][:-1],
            camelyon["cramers_v"][1:],
            id_result["mean_normalized_trajectory"][:-1],
            id_result["mean_normalized_trajectory"][1:],
        )
    )
    assert area == pytest.approx(-0.0427315084, abs=2e-10)


def test_summary_csv_is_the_complete_tabular_view_of_the_payload() -> None:
    payload = _payload()
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected_rows = 2 * sum(len(cohort["models"]) for cohort in payload["cohorts"])
    assert len(rows) == expected_rows == 166
    assert list(rows[0]) == [
        "cohort",
        "regime",
        "model",
        "panel",
        "ranked",
        "is_control",
        "croma_median_m5",
        "chance",
        "nipd",
        "baseline_balanced_accuracy",
        "baseline_skill",
    ]


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda p: p["cohorts"].append(copy.deepcopy(p["cohorts"][0])), "cohorts"),
        (lambda p: p["cohorts"][0]["models"].pop(), "roster"),
        (
            lambda p: p["cohorts"][0]["models"][0].update(is_control=True),
            "control",
        ),
        (
            lambda p: p["cohorts"][0]["models"][0]["regimes"]["id"].update(nipd=math.nan),
            "finite",
        ),
        (
            lambda p: p["cohorts"][0]["models"][0]["regimes"]["id"].update(
                mean_normalized_trajectory=[0.0]
            ),
            "shape",
        ),
        (lambda p: p["provenance"].update(study_revision="stale"), "provenance"),
        (
            lambda p: p["provenance"].update(apd_summary_sha256="0" * 64),
            "provenance",
        ),
        (lambda p: p["cohorts"][0].update(chance=0.123), "chance"),
    ],
)
def test_validation_fails_closed(corrupt, message: str) -> None:
    payload = _payload()
    corrupt(payload)
    with pytest.raises(ValueError, match=message):
        _exporter().validate_payload(payload)


def test_validation_rejects_an_oversized_payload() -> None:
    payload = _payload()
    payload["padding"] = "x" * _exporter().MAX_PAYLOAD_BYTES
    with pytest.raises(ValueError, match="oversized"):
        _exporter().validate_payload(payload)


def test_committed_summary_is_fresh_from_the_committed_float_basis() -> None:
    _exporter().check_committed(ROOT / "results")


@pytest.mark.skipif(
    not (ROOT / "output" / "studies" / "apd" / "apd.csv").exists(),
    reason="canonical ignored study output is not present in this checkout",
)
def test_committed_publication_is_fresh_against_canonical_study_output(tmp_path: Path) -> None:
    destination = tmp_path / "results"
    _exporter().export(ROOT / "output" / "studies" / "apd", destination)
    assert (destination / "nipd.json").read_bytes() == PAYLOAD.read_bytes()
    assert (destination / "nipd.csv").read_bytes() == SUMMARY.read_bytes()
