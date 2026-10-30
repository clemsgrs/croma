"""Shared basis for the robustness fine-tune vs parent artifacts (``tab:finetune-pairs``).

The table and the ``\\FineTune...`` macros both read the study CSV through :func:`load`, which
fails when the study was never run or no longer matches the pairs ``model_metadata.csv``
declares. A missing input therefore fails the paper build instead of silently dropping the
macros a sentence cites.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "studies"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import finetune_pairs as study  # noqa: E402
from _paper_tables import CaptionClaimError, bare_num  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

CSV = study.OUT_CSV
REMEDY = "run scripts/studies/finetune_pairs.py"

#: Differences cited by macro: (macro stem, study column stem, decimals, has an interval).
#: The decimals match the results tables; biological accuracy is a point difference.
MACRO_STATS = (
    ("Croma", "croma", 2, True),
    ("FZero", "f0", 3, True),
    ("Ltm", "ltm", 2, True),
    ("BioBacc", "bio_bacc", 3, False),
    ("NipdId", "nipd_id", 3, True),
)
_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def load(path: Path = CSV, *, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """The study CSV, checked against the pairs and cohorts the metadata now declares."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}: {REMEDY}")
    frame = pd.read_csv(path)
    pairs = study.finetune_pairs(study.read_metadata() if metadata is None else metadata)
    expected = [(p.finetune, p.parent, b) for p in pairs for b in study.BENCHMARKS]
    found = list(zip(frame["finetune"], frame["parent"], frame["benchmark"]))
    if found != expected:
        raise CaptionClaimError(
            f"{path} holds (fine-tune, parent, cohort) rows {found}, but model_metadata.csv "
            f"now declares {expected}: {REMEDY}"
        )
    return frame


def signed(value: float, decimals: int) -> str:
    """Signed math literal, ``$+0.18$`` / ``$-0.31$``; a value that rounds to zero is ``$0.00$``."""
    body = f"{value:+.{decimals}f}"
    if float(body) == 0.0:
        body = f"{0.0:.{decimals}f}"
    return f"${body}$"


def interval(lo: float, hi: float, decimals: int) -> str:
    """Bracketed interval in math mode, ``$[0.15, 0.20]$``."""
    return f"$[{bare_num(lo, decimals)}, {bare_num(hi, decimals)}]$"


def token(name: str) -> str:
    """A TeX-macro-safe token: letters only, each digit spelled out (``Robust-2`` -> ``RobustTwo``)."""
    parts = re.findall(r"[A-Za-z]+|[0-9]", name)
    return "".join(_DIGITS[int(p)] if p.isdigit() else p[0].upper() + p[1:] for p in parts)


def macro_lines(frame: pd.DataFrame) -> list[str]:
    """``\\newcommand`` lines: each pair x cohort difference, and two counts over all cells."""
    lines = []
    for _, row in frame.iterrows():
        stem = f"FineTune{token(row['finetune'])}{by_benchmark(row['benchmark']).prefix}"
        for macro, column, decimals, has_interval in MACRO_STATS:
            delta = float(row[f"delta_{column}"])
            lines.append(rf"\newcommand{{\{stem}{macro}Delta}}{{{signed(delta, decimals)}}}")
            if has_interval:
                ci = interval(row[f"delta_{column}_lo"], row[f"delta_{column}_hi"], decimals)
                lines.append(rf"\newcommand{{\{stem}{macro}DeltaCi}}{{{ci}}}")
    lines += [
        rf"\newcommand{{\FineTuneNCells}}{{{len(frame)}}}",
        rf"\newcommand{{\FineTuneFZeroFallsCount}}{{{int((frame['delta_f0'] < 0).sum())}}}",
        rf"\newcommand{{\FineTuneLtmRisesCount}}{{{int((frame['delta_ltm'] > 0).sum())}}}",
    ]
    return lines
