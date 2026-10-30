"""Basis for the grade-matched SO-pool study on PCaBiop (``tab:grade-matched`` and macros).

``scripts/studies/grade_matched_pools.py`` restricts each cancer anchor's same-label,
other-centre (SO) pool to slides of its own ISUP grade and compares the margin with a
size-matched random pool. Two consumers read its output and must agree:

* ``generate_grade_matched_table.py`` renders ``tab:grade-matched``;
* ``generate_paper_values.py`` emits the per-encoder medians, the grade effect and the
  grade counts as macros.

Both go through :func:`load`, which fails when the study has not been run, was run on a
different benchmark run than the paper reports, or no longer supports the caption's
claims. It never lets a consumer warn and skip.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _paper_tables import CaptionClaimError

REPO = Path(__file__).resolve().parents[2]
# plotting.style (the published model names) lives with the benchmark plot library.
if str(REPO / "scripts" / "bench") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "bench"))
STUDY_REL = "output/studies/grade-matched"
_RERUN = "Run scripts/studies/grade_matched_pools.py"


_DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def macro_suffix(name: str) -> str:
    """A LaTeX-command-safe spelling of a model or provider name.

    ``Prov-GigaPath`` -> ``ProvGigaPath``, ``PRISM2`` -> ``PrismTwo``, ``karolinska`` ->
    ``Karolinska``: separators drop, all-caps words are title-cased, digits are spelled out
    (command names take letters only), the spelling the PCaBiop nIPD macros already use.
    """
    out = []
    for token in re.findall(r"[A-Za-z]+|\d", name):
        if token.isdigit():
            out.append(_DIGITS[int(token)])
        elif token.isupper() or token.islower():
            out.append(token.capitalize())
        else:
            out.append(token[0].upper() + token[1:])
    return "".join(out)


@dataclass(frozen=True)
class GradeMatched:
    #: One row per encoder over all cancer anchors, published model names.
    pooled: pd.DataFrame
    #: One row per (encoder, ISUP grade), published model names.
    per_grade: pd.DataFrame
    meta: dict


def load(root: Path = REPO) -> GradeMatched:
    """The study's summary and metadata under ``root``; raises rather than guessing."""
    study = Path(root) / STUDY_REL
    summary_path, meta_path = study / "summary.csv", study / "meta.json"
    missing = [p for p in (summary_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(map(str, missing))} absent: tab:grade-matched and its macros "
            f"cannot be built. {_RERUN}."
        )
    from paper_manifest import by_benchmark
    from plotting.style import published_models

    meta = json.loads(meta_path.read_text())
    entry = by_benchmark(meta["benchmark"])
    if meta["run_rel"] != entry.run_rel:
        raise CaptionClaimError(
            f"the grade-matched study read {meta['run_rel']}, but the paper reports "
            f"{meta['benchmark']} from {entry.run_rel}. {_RERUN}."
        )
    if not meta.get("full_pool_reproduces_run"):
        raise CaptionClaimError(f"{meta_path} does not record the full-pool check. {_RERUN}.")
    counts = meta["cancer_slides_per_grade"]
    unbalanced = {
        provider: by_grade
        for provider, by_grade in counts.items()
        if set(by_grade.values()) != {meta["pool_size"]}
    }
    if unbalanced:
        raise CaptionClaimError(
            f"the caption states {meta['pool_size']} cancer slides per ISUP grade per "
            f"provider, but the study counted {unbalanced}"
        )

    summary = pd.read_csv(summary_path, dtype={"grade": str})
    if set(summary["model"]) != set(meta["roster"]):
        raise CaptionClaimError(
            f"the study summarises {sorted(summary['model'].unique())}, but its run's roster is "
            f"{sorted(meta['roster'])}. {_RERUN}."
        )
    summary = published_models(summary)
    pooled = summary[summary["grade"] == "all"].set_index("model")
    per_grade = summary[summary["grade"] != "all"].reset_index(drop=True)
    return GradeMatched(pooled=pooled, per_grade=per_grade, meta=meta)
