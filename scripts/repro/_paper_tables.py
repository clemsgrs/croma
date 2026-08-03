"""Shared helpers for the paper's LaTeX table/value generators.

The generators (`generate_paper_values.py`, `generate_results_table.py`,
`generate_supp_rank_table.py`, `generate_uncertainty_supp_table.py`,
`generate_pretraining_overlap_table.py`) all need two things: the CRoMa headline
radius constant from the installed package, and a handful of LaTeX number/CI
formatting idioms. Both used to be re-implemented (or the package re-imported via a
``sys.path`` shim) in each generator. This module centralises them.

Every formatter is a pure function whose output is byte-for-byte identical to the
inline expression it replaces, so importing it here does not change any emitted
``.tex``.
"""

import sys
from pathlib import Path

import pandas as pd

# The generators live in ``scripts/repro``; ``croma`` lives under ``src/`` and is not
# pip-installed. Do the shim once here so each generator can import the constant from this
# module rather than repeating the ``sys.path`` dance. This pointed at the repo root, which
# holds no ``croma/`` package -- so every generator importing it needed ``PYTHONPATH=src``
# from the caller, and ``python scripts/repro/build_paper.py`` on its own could not run.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402,F401  (re-exported)


class CaptionClaimError(AssertionError):
    """A rendered sentence asserts something the data no longer supports. See ADR-0010.

    Six generators each declared their own copy of this class, so they were six unrelated
    types: ``pytest.raises(generate_results_table.CaptionClaimError)`` would sail straight
    past a ``generate_cross_benchmark_float.CaptionClaimError``, and a caller wanting to
    catch "any broken claim" had no type to name. One definition, re-exported by each
    generator so ``<module>.CaptionClaimError`` still resolves.
    """


def detect_croma_scale(croma: pd.Series) -> str:
    """Return ``"margin"`` or ``"ratio"`` from the value range.

    Margin lives in ``(-1, 1)`` (neutral at 0); ratio lives in ``(0, inf)`` (neutral at 1).
    A negative value is decisive for margin; a value above 1 is decisive for ratio; an
    all-``[0, 1]`` column (every model biology-dominant) is ambiguous and defaults to
    margin, the paper's canonical scale.
    """
    if (croma < 0.0).any():
        return "margin"
    if (croma > 1.0 + 1e-9).any():
        return "ratio"
    return "margin"


def to_margin(croma: pd.Series, scale: str) -> pd.Series:
    """Map a CRoMa column onto the paper's canonical margin scale."""
    if scale == "ratio":
        return (croma - 1.0) / (croma + 1.0)
    return croma


def croma_as_margin(croma: pd.Series) -> pd.Series:
    """Detect the stored scale and normalise. Every emitter must go through this.

    Historically the value generator normalised while the table generator printed the raw
    column, so a benchmark stored as ratio would have put ``1.50`` in a table cell and
    ``0.20`` in the prose macro beside it. Both now share this one path.
    """
    croma = croma.astype(float)
    return to_margin(croma, detect_croma_scale(croma))


def num_math(value: float, decimals: int = 2) -> str:
    """Math-mode literal, e.g. ``$-0.46$``; ``-0.00`` is normalised to ``$0.00$``."""
    body = f"{value:.{decimals}f}"
    if float(body) == 0.0:
        body = f"{0.0:.{decimals}f}"  # strip a spurious leading minus
    return f"${body}$"


def bare_num(value: float, decimals: int = 2) -> str:
    """Bare number (no math delimiters), e.g. ``0.46``; ``-0.00`` normalised to ``0.00``.

    Used where a value nests inside a surrounding ``$...$`` supplied by the prose (the
    math minus is then provided by that outer math mode).
    """
    body = f"{value:.{decimals}f}"
    if float(body) == 0.0:
        body = f"{0.0:.{decimals}f}"
    return body


def pct_round(value: float) -> str:
    r"""Percent literal from a fraction, rounded to a whole percent, e.g. ``70\%``."""
    return rf"{int(round(value * 100))}\%"


def ci_bracket(lo: float, hi: float, decimals: int = 2) -> str:
    """Bracketed CI in math mode, e.g. ``$[0.17, 0.23]$``."""
    return rf"$[{lo:.{decimals}f}, {hi:.{decimals}f}]$"


def scriptsize_ci(lo: float, hi: float, decimals: int = 2) -> str:
    r"""Inline ``\scriptsize`` bracketed CI annotation, e.g. ``{\scriptsize$[0.17, 0.23]$}``.

    This is the in-cell CI idiom the result and uncertainty tables append after a point
    estimate (optionally preceded by a thin space ``\,``).
    """
    return rf"{{\scriptsize$[{lo:.{decimals}f}, {hi:.{decimals}f}]$}}"
