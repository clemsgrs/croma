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
    than a score -- confounder accuracy's maximum marks the least robust model, and the
    RI/MaRI difference is informative in its sign rather than ordered by its size. Bolding
    either would assert a ranking the column does not carry.
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
    Column("delta", "Δ", "{:+.3f}"),
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


#: Row class per exposure state. Disclosed-clean is deliberately absent: it is the
#: unmarked default, so the tint marks only what a reader must discount (exposed) or
#: cannot verify (undisclosed) -- and colour alone carries no text, so each tinted class
#: pairs with a visually-hidden label for screen readers, print, and copy-paste.
EXPOSURE_CLASSES = {
    "exposed": "croma-exposure-exposed",
    "disclosed-clean": None,
    "undisclosed": "croma-exposure-undisclosed",
}

EXPOSURE_LABELS = {
    "croma-exposure-exposed": " (TCGA-exposed pretraining)",
    "croma-exposure-undisclosed": " (pretraining corpus undisclosed)",
}


def _exposure_map() -> dict[str, str]:
    """Model -> exposure state, from the model-level export."""
    return {row["model"]: row["tcga_exposure"] for row in _read("cross_benchmark.csv")}


def exposure_row_classes(models: list[str], exposure: dict[str, str]) -> list[str | None]:
    """The row class (or ``None``) for each model, in table order.

    Raises on a model without a state or with an unknown state: either means the cohort
    CSV and the model-level export disagree, which should fail the ``-W`` build rather
    than publish an unmarked row.
    """
    classes = []
    for model in models:
        state = exposure[model]
        if state not in EXPOSURE_CLASSES:
            raise KeyError(f"unknown exposure state {state!r} for {model!r}")
        classes.append(EXPOSURE_CLASSES[state])
    return classes


def _tint_rows(rendered: list[nodes.Node], models: list[str], exposure: dict[str, str]) -> None:
    """Apply the exposure classes and screen-reader labels to a rendered table's body."""
    tables = [n for node in rendered for n in node.findall(nodes.table)]
    if len(tables) != 1:
        raise ValueError(f"expected one rendered table, found {len(tables)}")
    body = next(tables[0].findall(nodes.tbody))
    row_nodes = list(body.findall(nodes.row))
    if len(row_nodes) != len(models):
        raise ValueError(f"{len(models)} models but {len(row_nodes)} body rows")
    for row_node, row_class in zip(row_nodes, exposure_row_classes(models, exposure)):
        if row_class is None:
            continue
        row_node["classes"].append(row_class)
        first_cell = next(row_node.findall(nodes.entry))
        label = EXPOSURE_LABELS[row_class]
        first_cell += nodes.inline(label, label, classes=["croma-sr-only"])


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
        rendered = self._render(_list_table(headers, body, name=title))
        # Only the cohort whose scored domain the exposure states are derived for; on the
        # TCGA-free cohorts the tint would imply a caveat their scores do not carry.
        if slug == "tcga-4x4":
            _tint_rows(rendered, [row["model"] for row in rows], _exposure_map())
        return rendered


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
        # Tinted wherever the aggregate renders (results page and landing page): one of
        # its three cohorts is TCGA, so the caveat travels with the ranks.
        _tint_rows(rendered, [row["model"] for row in shown], _exposure_map())
        return rendered


def _cohort_header(key: str) -> str:
    """``tcga_4x4`` -> ``TCGA-4×4``. The CSV column carries the cohort slug."""
    slug = key.replace("_", "-")
    special = {"tcga-4x4": "TCGA-4×4", "tolkach-esca": "Tolkach-ESCA", "camelyon": "Camelyon"}
    return special.get(slug, slug)


def setup(app):
    app.add_directive("results-table", ResultsTable)
    app.add_directive("aggregate-table", AggregateTable)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
