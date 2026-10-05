"""Render the committed ``results/`` CSVs as documentation tables.

The site's numbers come from ``results/``, which a tracked exporter writes from the
benchmark runs (ADR-0016). Two directives read those CSVs at build time:

.. code-block:: rst

   .. results-table:: camelyon

   .. aggregate-table::
      :top: 8

Reading the CSV rather than hand-writing the table is the whole point: a benchmark re-run
that gets republished updates every page at once, and one that does not gets caught by the
freshness test instead of quietly leaving stale numbers on a public site.

Each directive builds ``list-table`` reStructuredText and hands it back to the parser, so
inline markup (bold, literals) works exactly as it would if the table had been typed by
hand, and ``sphinx -W`` reports a malformed table against the page that used it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList

RESULTS = Path(__file__).resolve().parents[2] / "results"

#: Appended to the natural-image control's name. Its cohort measurements remain visible,
#: but it is excluded from the pathology ranks and frontier.
CONTROL_MARK = " †"


@dataclass(frozen=True)
class Column:
    """One published column: where it comes from and how it reads.

    ``best`` is ``"max"``, ``"min"`` or ``None``. ``None`` marks a *diagnostic* rather
    than a score -- confounder accuracy's maximum marks the least robust model, and
    bolding it would assert a ranking the column does not carry.
    """

    key: str
    header: str
    fmt: str
    best: str | None = None

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.key,)

    def render(self, row: dict[str, str], best: dict[str, float | None]) -> str:
        return _value(row, self.key, self.fmt, best[self.key])


@dataclass(frozen=True)
class PairColumn:
    """A cohort cell carrying both quantities its two ranks are built from.

    ``0.19/-0.05`` reads as median ``CRoMa`` over LTM10. They share a cell rather than
    taking two columns because they are one cohort's answer, and because the pairing is
    the point the whole aggregate makes: a strong median beside a severe tail is the case
    the table exists to keep visible, and a reader scanning margins alone would miss it.
    """

    croma_key: str
    ltm_key: str
    header: str
    fmt: str = "{:.2f}"

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.croma_key, self.ltm_key)

    def render(self, row: dict[str, str], best: dict[str, float | None]) -> str:
        return "/".join(
            _value(row, key, self.fmt, best[key]) for key in (self.croma_key, self.ltm_key)
        )


COHORT_COLUMNS: tuple[Column, ...] = (
    Column("bio_bacc", "bio bacc", "{:.3f}", "max"),
    Column("conf_bacc", "conf bacc", "{:.3f}"),
    Column("ri", "``RI``", "{:.3f}", "max"),
    Column("mari", "``MaRI``", "{:.3f}", "max"),
    Column("croma", "``CRoMa``", "{:.2f}", "max"),
    Column("croma_f0", "*F*\\ (0)", "{:.3f}", "min"),
    Column("croma_ltm10", "LTM₁₀", "{:.2f}", "max"),
    Column("support", "support", "{:.1%}", "max"),
)

#: ``mean_rank`` leads because the table is sorted by it, and the two ranks it averages
#: follow immediately, so the aggregate is always read next to its own inputs.
AGGREGATE_RANKS: tuple[Column, ...] = (
    Column("mean_rank", "mean rank", "{:.1f}", "min"),
    Column("croma_rank", "``CRoMa`` rank", "{:.1f}", "min"),
    Column("ltm_rank", "tail rank", "{:.1f}", "min"),
)


#: Row shading carries what a reader should discount a row for. Each tint is binary --
#: shaded or the unshaded default -- and colour alone carries no text, so every shaded row
#: also gets a visually-hidden label for screen readers, print and copy-paste, and every
#: table showing a tint is followed by a legend naming it. Unshaded means *not disclosed*,
#: not an audited absence; the legends say so.
#:
#: Orange: the encoder's disclosed pretraining corpus or institutional provenance overlaps
#: the cohort's source (per cohort: CAMELYON, TCGA, Charité, PANDA; TCGA on the aggregate).
EXPOSED_CLASS = "croma-exposure-exposed"
#: Yellow: the encoder's authors disclosed using PathoROB RI on these cohorts during its
#: development, e.g. in checkpoint selection (ADR-0020). Never a rank adjustment.
SELECTED_CLASS = "croma-selection-flagged"

EXPOSURE_LABELS = {EXPOSED_CLASS: " (pretraining overlaps this cohort's source)"}
SELECTION_LABEL = " (PathoROB RI used during development)"

#: What the orange tint asserts on each table, in the legend beneath it.
EXPOSURE_LEGENDS = {
    "aggregate": "disclosed pretraining overlaps TCGA, the source of TCGA-4×4.",
    "camelyon": "disclosed pretraining includes CAMELYON data.",
    "tcga-4x4": "disclosed pretraining or institutional provenance includes TCGA.",
    "tolkach-esca": (
        "institutional provenance overlaps Charité, one of the cohort's sources — a "
        "source-domain overlap, not evidence that any scored slide was seen in pretraining."
    ),
    "pcabiop": "disclosed pretraining includes PANDA data.",
}
SELECTION_LEGEND = (
    "PathoROB RI on these cohorts was used during the encoder's development, e.g. in "
    "checkpoint selection, as disclosed by its authors. Ranks are shown unadjusted."
)


def _parse_flag(value: str, column: str) -> bool:
    """Strict boolean parse: a corrupted cell must fail the build, not render unshaded."""
    if value not in ("True", "False"):
        raise ValueError(f"{column} must be True or False, got {value!r}")
    return value == "True"


def _exposure_map() -> dict[str, bool]:
    """Model -> TCGA exposure, from the model-level export (the aggregate's tint)."""
    return {
        row["model"]: _parse_flag(row["tcga_exposed"], "tcga_exposed")
        for row in _read("cross_benchmark.csv")
    }


def cohort_exposure_map(slug: str) -> dict[str, bool]:
    """Model -> exposure to *this* cohort's source, from the cohort's own export."""
    return {row["model"]: _parse_flag(row["exposed"], "exposed") for row in _read(f"{slug}.csv")}


def selection_map() -> dict[str, bool]:
    """Model -> author-disclosed use of PathoROB RI in development, model-level.

    The slide cohort's encoders are not in that export; nothing was disclosed for them, so
    a lookup there falls back to ``False``.
    """
    return {
        row["model"]: _parse_flag(row["benchmark_selected"], "benchmark_selected")
        for row in _read("cross_benchmark.csv")
    }


def exposure_row_classes(models: list[str], exposure: dict[str, bool]) -> list[str | None]:
    """The exposure class (or ``None``) for each model, in table order.

    Raises on a model without a state: it means the table and its export disagree, which
    should fail the ``-W`` build rather than publish an unshaded row.
    """
    return [EXPOSED_CLASS if exposure[model] else None for model in models]


def _body_rows(rendered: list[nodes.Node], n_models: int) -> list[nodes.row]:
    """The body rows of the one table in ``rendered``, checked against the model count."""
    tables = [n for node in rendered for n in node.findall(nodes.table)]
    if len(tables) != 1:
        raise ValueError(f"expected one rendered table, found {len(tables)}")
    body = next(tables[0].findall(nodes.tbody))
    row_nodes = list(body.findall(nodes.row))
    if len(row_nodes) != n_models:
        raise ValueError(f"{n_models} models but {len(row_nodes)} body rows")
    return row_nodes


def _append_hidden_label(row_node: nodes.row, label: str) -> None:
    first_cell = next(row_node.findall(nodes.entry))
    first_cell += nodes.inline(label, label, classes=["croma-sr-only"])


def shade_rows(
    rendered: list[nodes.Node],
    models: list[str],
    exposure: dict[str, bool],
    selected: dict[str, bool],
    legend: str,
) -> list[nodes.Node]:
    """Shade a rendered table's rows and return the legend for the tints it shows.

    ``legend`` keys :data:`EXPOSURE_LEGENDS`. A row can carry both tints; the stylesheet
    splits it between the two colours.
    """
    row_nodes = _body_rows(rendered, len(models))
    shown = {EXPOSED_CLASS: False, SELECTED_CLASS: False}
    for row_node, model, exposed_class in zip(
        row_nodes, models, exposure_row_classes(models, exposure)
    ):
        if exposed_class is not None:
            row_node["classes"].append(exposed_class)
            _append_hidden_label(row_node, EXPOSURE_LABELS[exposed_class])
            shown[EXPOSED_CLASS] = True
        if selected.get(model, False):
            row_node["classes"].append(SELECTED_CLASS)
            _append_hidden_label(row_node, SELECTION_LABEL)
            shown[SELECTED_CLASS] = True
    return _legend(shown, EXPOSURE_LEGENDS[legend])


def _legend(shown: dict[str, bool], exposure_text: str) -> list[nodes.Node]:
    items = []
    if shown[EXPOSED_CLASS]:
        items.append(
            ("croma-swatch-exposed", "Orange", f"{exposure_text} Unshaded means no disclosed "
             "overlap, not an audited absence.")
        )
    if shown[SELECTED_CLASS]:
        items.append(("croma-swatch-selected", "Yellow", SELECTION_LEGEND))
    if not items:
        return []
    legend = nodes.bullet_list(classes=["croma-legend"])
    for swatch, name, text in items:
        paragraph = nodes.paragraph()
        paragraph += nodes.inline("", "", classes=["croma-swatch", swatch])
        paragraph += nodes.strong(name, name)
        paragraph += nodes.Text(f" — {text}")
        legend += nodes.list_item("", paragraph)
    return [legend]


def shade_cohort_table(rendered: list[nodes.Node], slug: str, models: list[str]) -> list[nodes.Node]:
    """Shade a table of one cohort's encoders by that cohort's exposure and selection."""
    return shade_rows(rendered, models, cohort_exposure_map(slug), selection_map(), slug)


def _read(name: str) -> list[dict[str, str]]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run python scripts/tools/export_results.py, or check that "
            f"results/ was committed -- the docs build cannot see output/."
        )
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _value(row: dict[str, str], key: str, fmt: str, best: float | None) -> str:
    if not str(row[key]).strip():
        return "—"
    value = float(row[key])
    text = fmt.format(value)
    if best is not None and abs(value - best) < 1e-9:
        return f"**{text}**"
    return text


def _render_row(row: dict[str, str], columns, best: dict[str, float | None]) -> list[str]:
    """One model's cells. The control is never bolded, whatever it happens to lead on."""
    blank = dict.fromkeys(best)
    return [
        column.render(row, blank if _is_true(row["is_control"]) else best) for column in columns
    ]


def _best_values(rows: list[dict[str, str]], columns) -> dict[str, float | None]:
    """The bolded value per scored key, over the pathology encoders only.

    The control is excluded from the comparison, not merely from being bolded: it is a
    floor rather than a competitor, and on at least one cohort it holds the highest
    support in the panel -- a bold there would read as an endorsement of a model that has
    never seen tissue.
    """
    ranked = [r for r in rows if not _is_true(r["is_control"])]
    best: dict[str, float | None] = {}
    for column in columns:
        direction = getattr(column, "best", "max")
        for key in column.keys:
            if direction is None or not ranked:
                best[key] = None
                continue
            values = [float(r[key]) for r in ranked]
            best[key] = min(values) if direction == "min" else max(values)
    return best


def _model_cell(row: dict[str, str], *, emphasise: bool = False) -> str:
    name = row["model"]
    if emphasise:
        name = f"**{name}**"
    return name + (CONTROL_MARK if _is_true(row["is_control"]) else "")


def operating_point(slug: str) -> str:
    """The sentence naming the ``k`` a cohort's k-dependent columns are read at.

    Under ``median-k`` that ``k`` is the lower median of every encoder's own best ``k``,
    so it can move whenever an encoder joins the panel; it is stated beside the numbers
    it shapes rather than left for a reader to look up. The two kNN accuracies, ``RI``,
    ``MaRI`` and support are read at it; ``CRoMa``, *F*\ (0) and LTM₁₀ do not depend on it.
    """
    with (RESULTS / "PROVENANCE.json").open() as handle:
        cohort = json.load(handle)["cohorts"][slug]
    k = cohort["k"]
    scope = "every column except ``CRoMa``, *F*\\ (0) and LTM₁₀"
    if isinstance(k, dict):
        return f"Each encoder's own best ``k`` for {scope}."
    return f"``k`` = {k}, the median of the {cohort['n_models']} encoders' best ``k``, for {scope}."


def _list_table(headers: list[str], body: list[list[str]], *, name: str) -> list[str]:
    lines = [f".. list-table:: {name}", "   :header-rows: 1", "   :class: croma-results", ""]
    for record in [headers, *body]:
        for index, cell in enumerate(record):
            lines.append(f"   {'*' if index == 0 else ' '} - {cell}")
    lines.append("")
    return lines


class _TableDirective(Directive):
    """Shared plumbing: build reST lines, then let the parser handle them."""

    has_content = False

    def _render(self, lines: list[str]) -> list[nodes.Node]:
        container = nodes.container()
        self.state.nested_parse(StringList(lines, source=""), self.content_offset, container)
        return container.children


class ResultsTable(_TableDirective):
    """One cohort's full column set, best CRoMa first."""

    required_arguments = 1
    option_spec = {"caption": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        slug = self.arguments[0]
        rows = _read(f"{slug}.csv")
        best = _best_values(rows, COHORT_COLUMNS)
        headers = ["Model", *(c.header for c in COHORT_COLUMNS)]
        body = [[_model_cell(row), *_render_row(row, COHORT_COLUMNS, best)] for row in rows]
        title = self.options.get("caption", f"{slug} — {len(rows)} encoders")
        title = f"{title} {operating_point(slug)}"
        rendered = self._render(_list_table(headers, body, name=title))
        # Every cohort shades by its own source: GPFM on Camelyon, the TCGA-trained
        # encoders on TCGA-4×4, and so on.
        return rendered + shade_cohort_table(rendered, slug, [row["model"] for row in rows])


class AggregateTable(_TableDirective):
    """The cross-cohort aggregate: three ranks, then each cohort's margin/tail pair.

    ``:top:`` truncates to the first *n* rows. The truncation is a rule, not a selection,
    and the page that uses it has to say so -- which is why the count lands in the table
    caption rather than being left for a reader to notice.
    """

    option_spec = {"top": directives.positive_int, "caption": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        rows = _read("cross_benchmark.csv")
        ranks = {c.key for c in AGGREGATE_RANKS}
        cohorts = [
            k.removeprefix("croma_") for k in rows[0] if k.startswith("croma_") and k not in ranks
        ]
        columns = list(AGGREGATE_RANKS) + [
            PairColumn(f"croma_{slug}", f"ltm_{slug}", _cohort_header(slug)) for slug in cohorts
        ]
        best = _best_values(rows, columns)

        ranked_total = sum(not _is_true(row["is_control"]) for row in rows)
        top = self.options.get("top")
        shown = rows[:top] if top else rows
        body = [
            [
                _model_cell(row, emphasise=_is_true(row["on_frontier"])),
                *_render_row(row, columns, best),
            ]
            for row in shown
        ]
        headers = ["Model", *(c.header for c in columns)]
        default = (
            f"Top {len(shown)} of {ranked_total} ranked pathology encoders"
            if top and top < ranked_total
            else f"{ranked_total} ranked pathology encoders plus control"
        )
        rendered = self._render(
            _list_table(headers, body, name=self.options.get("caption", default))
        )
        # Shaded wherever the aggregate renders (results page and landing page): one of
        # its three cohorts is TCGA, so the caveat travels with the ranks.
        legend = shade_rows(
            rendered,
            [row["model"] for row in shown],
            _exposure_map(),
            selection_map(),
            "aggregate",
        )
        return rendered + legend


def _cohort_header(key: str) -> str:
    """``tcga_4x4`` -> ``TCGA-4×4``. The CSV column carries the cohort slug."""
    slug = key.replace("_", "-")
    special = {"tcga-4x4": "TCGA-4×4", "tolkach-esca": "Tolkach-ESCA", "camelyon": "Camelyon"}
    return special.get(slug, slug)


def setup(app):
    app.add_directive("results-table", ResultsTable)
    app.add_directive("aggregate-table", AggregateTable)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
