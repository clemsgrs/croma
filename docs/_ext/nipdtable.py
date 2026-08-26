"""Render static nIPD tables from the committed publication payload."""

from __future__ import annotations

import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

RESULT = Path(__file__).resolve().parents[2] / "results" / "nipd.json"
# A normalized change of -1 is chance level: the probe has lost its whole above-chance
# margin. Curves ending at or below this are named as a collapse, as in the explorer.
COLLAPSE = -0.9


def _payload() -> dict:
    if not RESULT.exists():
        raise FileNotFoundError(
            f"{RESULT} is missing; publish results/nipd.json before building the docs"
        )
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _end_of_range(result: dict) -> str:
    """The trajectory endpoint, named when it reaches chance level."""
    change = result["mean_normalized_trajectory"][-1]
    return f"{change:.3f}" + (" ≈ chance" if change <= COLLAPSE else "")


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
        association = cohort["association"]
        association_result = association["regimes"][regime]
        descriptor = "; descriptive" if association["descriptive"] else ""
        lines = [
            f".. list-table:: {cohort['label']} — {heading}; Spearman ρ = "
            f"{association_result['spearman_rho']:.2f}; n={association_result['n']} "
            f"ranked pathology encoders{descriptor}",
            "   :header-rows: 1",
            "   :class: croma-results",
            "",
            "   * - Model",
            "     - Median CRoMa (m=5)",
            "     - ``nIPD``",
            "     - Change at ``V`` = 1",
            "     - Baseline balanced accuracy",
        ]
        # Higher nIPD (less net degradation) first, matching the explorer's ordering;
        # the unranked natural-image control sits last whatever its value.
        models = sorted(
            cohort["models"],
            key=lambda model: (model["is_control"], -model["regimes"][regime]["nipd"]),
        )
        for model in models:
            result = model["regimes"][regime]
            mark = " †" if model["is_control"] else ""
            lines.extend(
                [
                    f"   * - {model['model']}{mark}",
                    f"     - {model['croma_median_m5']:.3f}",
                    f"     - {result['nipd']:.3f}",
                    f"     - {_end_of_range(result)}",
                    f"     - {result['baseline_balanced_accuracy']:.3f}",
                ]
            )
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source=str(RESULT)), self.content_offset, container
        )
        return container.children


def setup(app):
    app.add_directive("nipd-table", NipdTable)
    return {"parallel_read_safe": True}
