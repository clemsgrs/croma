"""Semantic contract for the public encoder-request GitHub Issue Forms."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FORMS = Path(__file__).resolve().parents[1] / ".github/ISSUE_TEMPLATE"


def _fields_by_id(form: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        field["id"]: field for field in form["body"] if isinstance(field, dict) and "id" in field
    }


@pytest.mark.parametrize(
    ("encoder_level", "title", "encoder_label", "contract_terms", "notice_terms"),
    [
        pytest.param(
            "tile",
            "[Tile encoder request] ",
            "tile-encoder",
            (
                "checkpoint loading",
                "inference preprocessing or augmentation",
                "input size",
                "supported slide spacing",
                "output representation",
                "pooling",
                "dimension",
                "normalization",
                "recommended precision",
                "custom dependencies",
                "remote code",
                "documented facts need not be repeated",
            ),
            (),
            id="tile",
        ),
        pytest.param(
            "slide",
            "[Slide encoder request] ",
            "slide-encoder",
            (
                "slide checkpoint and loader",
                "required tile encoder",
                "tile output variant and dimension",
                "tile size and spacing",
                "slide sampling or maximum-tile assumptions",
                "coordinate, order, and level-zero geometry inputs",
                "aggregation method",
                "final output representation and dimension",
                "recommended precision",
                "custom dependencies",
                "remote code",
                "documented facts need not be repeated",
            ),
            (
                "not yet supported by slide2vec remain requestable",
                "linked prerequisite",
                "does not perform that work",
            ),
            id="slide",
        ),
    ],
)
def test_encoder_form_has_the_complete_manual_request_contract(
    encoder_level: str,
    title: str,
    encoder_label: str,
    contract_terms: tuple[str, ...],
    notice_terms: tuple[str, ...],
) -> None:
    form = yaml.safe_load(
        (FORMS / f"{encoder_level}-encoder-request.yml").read_text(encoding="utf-8")
    )
    fields = _fields_by_id(form)

    assert form["title"] == title
    assert form["labels"] == ["model-request", encoder_label, "needs-triage"]

    required = {
        field_id
        for field_id, field in fields.items()
        if field.get("validations", {}).get("required") is True
    }
    assert required == {"model_name", "checkpoint_url", "rationale", "encoder_contract"}
    assert set(fields) - required == {
        "preferred_revision",
        "paper_project_url",
        "relationship",
        "additional_context",
    }

    assert fields["relationship"]["attributes"]["options"] == [
        "Author or maintainer",
        "Contributor or collaborator",
        "Unaffiliated requester",
    ]

    checkpoint_guidance = fields["checkpoint_url"]["attributes"]["description"].lower()
    assert "standard gated hugging face checkpoint" in checkpoint_guidance
    assert "source repository" in checkpoint_guidance
    assert "cannot replace the checkpoint url" in checkpoint_guidance

    contract_guidance = fields["encoder_contract"]["attributes"]["description"].lower()
    for expected in contract_terms:
        assert expected in contract_guidance

    notice = " ".join(
        field["attributes"]["value"] for field in form["body"] if field.get("type") == "markdown"
    ).lower()
    assert "manual triage" in notice
    assert "does not run the model or start an evaluation" in notice
    for expected in notice_terms:
        assert expected in notice
