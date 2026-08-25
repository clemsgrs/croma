"""Semantic contract for the public tile-encoder GitHub Issue Form."""

from __future__ import annotations

from pathlib import Path

import yaml

FORM = Path(__file__).resolve().parents[1] / ".github/ISSUE_TEMPLATE/tile-encoder-request.yml"


def _fields_by_id(form: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        field["id"]: field for field in form["body"] if isinstance(field, dict) and "id" in field
    }


def test_tile_encoder_form_has_the_complete_manual_request_contract() -> None:
    form = yaml.safe_load(FORM.read_text(encoding="utf-8"))
    fields = _fields_by_id(form)

    assert form["title"] == "[Tile encoder request] "
    assert form["labels"] == ["model-request", "tile-encoder", "needs-triage"]

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
    for expected in (
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
    ):
        assert expected in contract_guidance
