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

#: Appended to the natural-image control's name. It is ranked inline like every other
#: entry -- it earns its position -- but a reader who does not know it is a control would
#: draw the wrong conclusion from that position, so the pages footnote it beneath.
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

AGGREGATE_RANKS: tuple[Column, ...] = (
    Column("croma_rank", "``CRoMa`` rank", "{:.1f}", "min"),
    Column("ltm_rank", "tail rank", "{:.1f}", "min"),
)


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


def _cell(row: dict[str, str], column: Column, best: float | None) -> str:
    value = float(row[column.key])
    text = column.fmt.format(value)
    if best is not None and abs(value - best) < 1e-9:
        return f"**{text}**"
    return text


def _best_values(rows: list[dict[str, str]], columns) -> dict[str, float | None]:
    """The bolded value per scored column, over the pathology encoders only.

    The control is excluded from the comparison, not merely from being bolded: it is a
    floor rather than a competitor, and on at least one cohort it holds the highest
    support in the panel -- a bold there would read as an endorsement of a model that has
    never seen tissue.
    """
    ranked = [r for r in rows if not _is_true(r["is_control"])]
    best: dict[str, float | None] = {}
    for column in columns:
        if column.best is None or not ranked:
            best[column.key] = None
            continue
        values = [float(r[column.key]) for r in ranked]
        best[column.key] = min(values) if column.best == "min" else max(values)
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
        body = [
            [_model_cell(row)]
            + [
                _cell(row, column, None if _is_true(row["is_control"]) else best[column.key])
                for column in COHORT_COLUMNS
            ]
            for row in rows
        ]
        title = self.options.get("caption", f"{slug} — {len(rows)} encoders")
        return self._render(_list_table(headers, body, name=title))


class AggregateTable(_TableDirective):
    """The cross-cohort aggregate: two ranks, then the margins behind them.

    ``:top:`` truncates to the first *n* rows. The truncation is a rule, not a selection,
    and the page that uses it has to say so -- which is why the count lands in the table
    caption rather than being left for a reader to notice.
    """

    option_spec = {"top": directives.positive_int, "caption": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        rows = _read("cross_benchmark.csv")
        cohorts = [k for k in rows[0] if k.startswith("croma_") and k not in {c.key for c in AGGREGATE_RANKS}]
        columns = list(AGGREGATE_RANKS) + [
            Column(key, _cohort_header(key), "{:.2f}", "max") for key in cohorts
        ]
        best = _best_values(rows, columns)

        total = len(rows)
        top = self.options.get("top")
        shown = rows[:top] if top else rows
        body = [
            [_model_cell(row, emphasise=_is_true(row["on_frontier"]))]
            + [
                _cell(row, column, None if _is_true(row["is_control"]) else best[column.key])
                for column in columns
            ]
            for row in shown
        ]
        headers = ["Model", *(c.header for c in columns)]
        default = (
            f"Top {len(shown)} of {total} by ``CRoMa`` rank"
            if top and top < total
            else f"All {total} encoders, by ``CRoMa`` rank"
        )
        return self._render(
            _list_table(headers, body, name=self.options.get("caption", default))
        )


def _cohort_header(key: str) -> str:
    """``croma_tcga_4x4`` -> ``TCGA-4×4``. The CSV column carries the cohort slug."""
    slug = key.removeprefix("croma_").replace("_", "-")
    special = {"tcga-4x4": "TCGA-4×4", "tolkach-esca": "Tolkach-ESCA", "camelyon": "Camelyon"}
    return special.get(slug, slug)


def setup(app):
    app.add_directive("results-table", ResultsTable)
    app.add_directive("aggregate-table", AggregateTable)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
