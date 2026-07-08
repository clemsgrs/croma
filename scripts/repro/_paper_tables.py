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

# The generators live in ``scripts/repro``; the repo root is two levels up. Do
# the shim once here so each generator can import the constant from this module rather
# than repeating the ``sys.path`` dance.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402,F401  (re-exported)


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
