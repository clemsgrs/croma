"""Inline run-derived numbers, computed from the committed ``results/`` exports.

Prose on the results pages cites values that move when a benchmark is re-run -- roster
counts, the shared ``k``, how many encoders sit below zero, a support floor, a contrast
between two named models. Hand-typing them is how a public page kept claiming a number a
month after the run that produced it was corrected. This role computes them at build time
from the same committed exports the tables render, so re-exported results update the prose
and the tables in the same build:

.. code-block:: rst

   The shared operating point is k = :results-value:`k(camelyon)`, and
   :results-value:`below_zero(camelyon)` of the :results-value:`ranked()` ranked
   encoders score below zero.

An unknown function, cohort, model or column is a build error, not a dash: with
``sphinx -W`` a typo fails the build instead of publishing a blank.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from docutils import nodes

RESULTS = Path(__file__).resolve().parents[2] / "results"

#: How each CSV column renders inline. Kept consistent with the table directives.
FORMATS = {
    "bio_bacc": "{:.3f}",
    "conf_bacc": "{:.3f}",
    "ri": "{:.3f}",
    "mari": "{:.3f}",
    "delta": "{:+.3f}",
    "croma": "{:.2f}",
    "croma_f0": "{:.3f}",
    "croma_ltm10": "{:.2f}",
    "support": "{:.1%}",
}

_CALL = re.compile(r"^\s*([a-z_]+)\s*\(\s*(.*?)\s*\)\s*$")


def _read_cohort(slug: str) -> list[dict[str, str]]:
    path = RESULTS / f"{slug}.csv"
    if not path.exists():
        known = sorted(p.stem for p in RESULTS.glob("*.csv"))
        raise ValueError(f"unknown cohort {slug!r}; results/ has {known}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _ranked(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if str(r["is_control"]).strip().lower() != "true"]


def _row(rows: list[dict[str, str]], model: str) -> dict[str, str]:
    for row in rows:
        if row["model"] == model:
            return row
    raise ValueError(f"unknown model {model!r}")


def _column(rows: list[dict[str, str]], key: str, model: str) -> float:
    if key not in FORMATS:
        raise ValueError(f"unknown column {key!r}; one of {sorted(FORMATS)}")
    return float(_row(rows, model)[key])


def _provenance() -> dict:
    with (RESULTS / "PROVENANCE.json").open() as handle:
        return json.load(handle)


def roster() -> str:
    """Every encoder the panel scores, control included."""
    return str(len(_read_cohort("cross_benchmark")))


def ranked() -> str:
    """The pathology encoders: the roster minus the natural-image control."""
    return str(len(_ranked(_read_cohort("cross_benchmark"))))


def models(slug: str) -> str:
    """How many encoders the cohort's table holds, control included."""
    return str(len(_read_cohort(slug)))


def k(slug: str) -> str:
    """The cohort's shared ``median-k`` operating point, from ``PROVENANCE.json``."""
    value = _provenance()["cohorts"][slug]["k"]
    if isinstance(value, dict):
        raise ValueError(
            f"{slug!r} is a k-star cohort with per-model k; use k_range({slug}) instead"
        )
    return str(value)


def k_range(slug: str) -> str:
    """A ``k-star`` cohort's span of per-model operating points, as ``3–13``."""
    value = _provenance()["cohorts"][slug]["k"]
    if not isinstance(value, dict):
        raise ValueError(f"{slug!r} has one shared k; use k({slug}) instead")
    ks = sorted(int(v) for v in value.values())
    return f"{ks[0]}–{ks[-1]}"


def below_zero(slug: str) -> str:
    """Ranked encoders whose pooled ``CRoMa`` is negative on the cohort."""
    return str(sum(float(r["croma"]) < 0 for r in _ranked(_read_cohort(slug))))


def support_min(slug: str) -> str:
    """The cohort's worst RI/MaRI support fraction over the ranked panel."""
    return "{:.0%}".format(min(float(r["support"]) for r in _ranked(_read_cohort(slug))))


def support_max(slug: str) -> str:
    """The cohort's best RI/MaRI support fraction over the ranked panel."""
    return "{:.0%}".format(max(float(r["support"]) for r in _ranked(_read_cohort(slug))))


def count_above(slug: str, key: str, threshold: str) -> str:
    """Ranked encoders strictly above ``threshold`` on one column."""
    if key not in FORMATS:
        raise ValueError(f"unknown column {key!r}; one of {sorted(FORMATS)}")
    limit = float(threshold)
    return str(sum(float(r[key]) > limit for r in _ranked(_read_cohort(slug))))


def value(slug: str, model: str, key: str) -> str:
    """One cell, formatted as the tables format it."""
    return FORMATS[key].format(_column(_read_cohort(slug), key, model))


def gap(slug: str, key: str, a: str, b: str) -> str:
    """The absolute difference between two models on one column.

    A difference needs more precision than a level: two ``CRoMa`` values that print as
    ``0.20`` can sit ``0.003`` apart, and that near-tie is exactly what a gap is quoted
    to establish.
    """
    rows = _read_cohort(slug)
    fmt = "{:.3f}" if FORMATS[key] == "{:.2f}" else FORMATS[key]
    return fmt.format(abs(_column(rows, key, a) - _column(rows, key, b))).lstrip("+")


def ratio(slug: str, key: str, a: str, b: str) -> str:
    """How many times *a* exceeds *b* on one column, as ``1.7×``."""
    rows = _read_cohort(slug)
    denominator = _column(rows, key, b)
    if denominator == 0:
        raise ValueError(f"{b!r} is zero on {key!r}; a ratio against it is undefined")
    return "{:.1f}×".format(abs(_column(rows, key, a) / denominator))


FUNCTIONS = {
    fn.__name__: fn
    for fn in (
        roster,
        ranked,
        models,
        k,
        k_range,
        below_zero,
        support_min,
        support_max,
        count_above,
        value,
        gap,
        ratio,
    )
}


def results_value_role(name, rawtext, text, lineno, inliner, options=None, content=None):
    match = _CALL.match(text)
    if not match:
        return _error(inliner, rawtext, lineno, f"cannot parse {text!r}; expected name(args)")
    fn_name, arg_text = match.groups()
    fn = FUNCTIONS.get(fn_name)
    if fn is None:
        return _error(
            inliner, rawtext, lineno, f"unknown value {fn_name!r}; one of {sorted(FUNCTIONS)}"
        )
    args = [a.strip() for a in arg_text.split(",")] if arg_text else []
    try:
        rendered = fn(*args)
    except (TypeError, ValueError, KeyError, FileNotFoundError) as error:
        return _error(inliner, rawtext, lineno, f"{text!r}: {error}")
    return [nodes.Text(rendered)], []


def _error(inliner, rawtext, lineno, message):
    error = inliner.reporter.error(f"results-value: {message}", line=lineno)
    return [inliner.problematic(rawtext, rawtext, error)], [error]


def setup(app):
    app.add_role("results-value", results_value_role)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
