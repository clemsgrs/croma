"""Render static nIPD tables from the committed publication payload."""

from __future__ import annotations

import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

RESULT = Path(__file__).resolve().parents[2] / "results" / "nipd.json"


def _payload() -> dict:
    if not RESULT.exists():
        raise FileNotFoundError(
            f"{RESULT} is missing; publish results/nipd.json before building the docs"
        )
    return json.loads(RESULT.read_text(encoding="utf-8"))


class NipdTable(Directive):
    """One complete cohort/regime view of the nIPD publication payload."""

    required_arguments = 2
    has_content = False

    def run(self) -> list[nodes.Node]:
        slug, regime = self.arguments
        if regime not in {"id", "ood"}:
            raise ValueError(f"nIPD regime must be id or ood, got {regime!r}")
        cohort = next(item for item in _payload()["cohorts"] if item["slug"] == slug)
        heading = "ID" if regime == "id" else "OOD"
        lines = [
            f".. list-table:: {cohort['label']} — {heading}",
            "   :header-rows: 1",
            "   :class: croma-results",
            "",
            "   * - Model",
            "     - ``nIPD``",
            "     - Baseline balanced accuracy",
            "     - Baseline skill",
        ]
        for model in cohort["models"]:
            result = model["regimes"][regime]
            mark = " †" if model["is_control"] else ""
            lines.extend(
                [
                    f"   * - {model['model']}{mark}",
                    f"     - {result['nipd']:.3f}",
                    f"     - {result['baseline_balanced_accuracy']:.3f}",
                    f"     - {result['baseline_skill']:.3f}",
                ]
            )
        container = nodes.container()
        self.state.nested_parse(StringList(lines, source=str(RESULT)), self.content_offset, container)
        return container.children


def setup(app):
    app.add_directive("nipd-table", NipdTable)
    return {"parallel_read_safe": True}

