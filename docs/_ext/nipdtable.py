"""Render static nIPD tables from the committed publication payload."""

from __future__ import annotations

import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from resultstable import shade_cohort_table

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


def _leader(values: list[float]) -> float:
    """The best value of a higher-is-better column, at printed precision.

    Rounding before the comparison keeps ties visible: two rows that print the same
    value are both marked, rather than one winning on invisible digits.
    """
    return max(round(value, 3) for value in values)


def _mark(value: float, leader: float, text: str) -> str:
    return f"**{text}**" if round(value, 3) == leader else text


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
            "     - Change at ``V`` = 1",
            "     - ``nIPD``",
            "     - Baseline balanced accuracy",
        ]
        # Ranked on the endpoint, not the pooled area, and matching the explorer's
        # ordering. The signed area lets an early gain pay for a late collapse, so a curve
        # ending at chance can outrank one that never moved; the endpoint cannot cancel
        # with itself. The unranked natural-image control sits last whatever its value.
        models = sorted(
            cohort["models"],
            key=lambda model: (
                model["is_control"],
                -model["regimes"][regime]["mean_normalized_trajectory"][-1],
            ),
        )
        # Bold the leader of each higher-is-better column over the ranked encoders, so a
        # column that disagrees with the nIPD ordering shows it at a glance. Baseline
        # balanced accuracy has no leader: a higher baseline is not a better one.
        ranked = [model for model in models if not model["is_control"]]
        leaders = {
            "croma": _leader([model["croma_median_m5"] for model in ranked]),
            "nipd": _leader([model["regimes"][regime]["nipd"] for model in ranked]),
            "end": _leader(
                [model["regimes"][regime]["mean_normalized_trajectory"][-1] for model in ranked]
            ),
        }
        for model in models:
            result = model["regimes"][regime]
            control = model["is_control"]

            def cell(column: str, value: float, text: str, control: bool = control) -> str:
                return text if control else _mark(value, leaders[column], text)

            end = result["mean_normalized_trajectory"][-1]
            lines.extend(
                [
                    f"   * - {model['model']}{' †' if control else ''}",
                    "     - "
                    + cell("croma", model["croma_median_m5"], f"{model['croma_median_m5']:.3f}"),
                    "     - " + cell("end", end, _end_of_range(result)),
                    "     - " + cell("nipd", result["nipd"], f"{result['nipd']:.3f}"),
                    f"     - {result['baseline_balanced_accuracy']:.3f}",
                ]
            )
        container = nodes.container()
        self.state.nested_parse(
            StringList(lines, source=str(RESULT)), self.content_offset, container
        )
        rendered = list(container.children)
        # The same row shading as the cohort's results table above it.
        return rendered + shade_cohort_table(rendered, slug, [model["model"] for model in models])


def setup(app):
    app.add_directive("nipd-table", NipdTable)
    return {"parallel_read_safe": True}
