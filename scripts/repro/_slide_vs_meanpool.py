"""Float basis for the slide-encoder versus mean-pooling study on PCaBiop.

``scripts/studies/slide_vs_meanpool.py`` writes three artifacts under
``output/studies/slide-vs-meanpool/``: ``summary.csv`` (one row per pair and arm),
``differences.csv`` (slide minus mean-pool, per pair and statistic) and
``provenance.json`` (pairing, reproduction, bootstrap and probe settings). Two consumers
read them and must agree:

* ``generate_slide_vs_meanpool_table.py`` renders ``tab:slide-vs-meanpool``;
* ``generate_paper_values.py`` emits the paired differences as macros.

Which pairs are rendered is declared once, in :data:`RENDERED_PAIRS`. A declared pair the
study has no output for is an error, never a silently shorter table.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _paper_tables import CaptionClaimError, ci_bracket, num_math

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "studies"))
from slide_vs_meanpool import PAIRS, PAIRS_BY_MODEL, STUDY_NAME  # noqa: E402

__all__ = ["CaptionClaimError", "RENDERED_PAIRS", "STUDY_REL", "Basis", "load", "macros"]

#: The pairs ``tab:slide-vs-meanpool`` renders, in row order. Every one must have study
#: output; the caption lists the declared pairs (``slide_vs_meanpool.PAIRS``) left out.
RENDERED_PAIRS: tuple[str, ...] = ("TITAN", "MOOZY", "PRISM2")

STUDY_REL = f"output/studies/{STUDY_NAME}"

#: How each tile encoder is named in the paper.
TILE_DISPLAY = {
    "conchv15": "CONCHv1.5",
    "lunit": "Lunit ViT-S/8",
    "virchow2": "Virchow2",
    "virchow": "Virchow",
    "gigapath": "Prov-GigaPath",
}

#: Letters-only macro stem per slide encoder (LaTeX command names take no digits).
MACRO_STEM = {
    "TITAN": "Titan",
    "MOOZY": "Moozy",
    "PRISM2": "PrismTwo",
    "PRISM": "Prism",
    "Prov-GigaPath": "ProvGigaPath",
}

ARMS = ("slide", "meanpool")
CROMA_STATISTICS = ("median_croma", "f0", "ltm")
NIPD_STATISTICS = ("nipd_id", "nipd_ood")


@dataclass(frozen=True)
class Basis:
    summary: pd.DataFrame  # indexed by (pair, arm)
    differences: pd.DataFrame  # indexed by (pair, statistic)
    provenance: dict
    rendered: tuple[str, ...]

    @property
    def missing(self) -> list[str]:
        """Declared pairs that the table does not render."""
        return [p.slide_model for p in PAIRS if p.slide_model not in self.rendered]


def tile_display(pair: str) -> str:
    return TILE_DISPLAY[PAIRS_BY_MODEL[pair].tile_encoder]


def load(study_dir: Path | None = None, rendered: tuple[str, ...] = RENDERED_PAIRS) -> Basis:
    """Read the study's artifacts and check that every rendered pair is in all three."""
    study_dir = Path(study_dir) if study_dir is not None else REPO / STUDY_REL
    paths = [study_dir / n for n in ("summary.csv", "differences.csv", "provenance.json")]
    absent = [str(p) for p in paths if not p.exists()]
    if absent:
        raise FileNotFoundError(
            f"{absent} absent: run scripts/studies/slide_vs_meanpool.py evaluate"
        )
    summary = pd.read_csv(paths[0])
    differences = pd.read_csv(paths[1])
    provenance = json.loads(paths[2].read_text())

    problems = []
    for pair in rendered:
        if pair not in PAIRS_BY_MODEL:
            problems.append(f"{pair} is not a declared pair")
            continue
        arms = set(summary.loc[summary["pair"] == pair, "arm"])
        if arms != set(ARMS):
            problems.append(f"{pair}: summary.csv has arms {sorted(arms)}")
        stats = set(differences.loc[differences["pair"] == pair, "statistic"])
        if stats != set(CROMA_STATISTICS + NIPD_STATISTICS):
            problems.append(f"{pair}: differences.csv has {sorted(stats)}")
        if pair not in provenance.get("pairs", {}):
            problems.append(f"{pair}: provenance.json has no record")
    if problems:
        raise CaptionClaimError(
            "tab:slide-vs-meanpool declares pairs the study has no output for: "
            + "; ".join(problems)
        )
    return Basis(
        summary=summary.set_index(["pair", "arm"]),
        differences=differences.set_index(["pair", "statistic"]),
        provenance=provenance,
        rendered=tuple(rendered),
    )


#: Decimals per CRoMa statistic, matching the slide results table.
DECIMALS = {"median_croma": 2, "f0": 3, "ltm": 2}


def pct(value: float) -> str:
    """nIPD in percent (points for a difference), one decimal, math mode."""
    return num_math(100.0 * float(value), 1)


#: How an nIPD spread prints when it is undefined: fewer than two paired replicates on
#: which both arms' nIPD is defined (``slide_vs_meanpool.NIPD_SPREAD_RULE``).
UNDEFINED_SPREAD = "---"


def spread_pct(value: float) -> str:
    """An nIPD spread in points, as :func:`pct`; :data:`UNDEFINED_SPREAD` when NaN."""
    return UNDEFINED_SPREAD if pd.isna(value) else pct(value)


def macros(basis: Basis) -> list[str]:
    r"""``\SlideMeanpool<Pair><Statistic>Diff`` (+ ``CI`` / ``Sd``): slide minus mean-pool."""
    names = {
        "median_croma": "Croma",
        "f0": "Fzero",
        "ltm": "Ltm",
        "nipd_id": "NipdId",
        "nipd_ood": "NipdOod",
    }
    lines = []
    for pair in basis.rendered:
        stem = f"SlideMeanpool{MACRO_STEM[pair]}"
        for statistic in CROMA_STATISTICS:
            row = basis.differences.loc[(pair, statistic)]
            d = DECIMALS[statistic]
            lines.append(rf"\newcommand{{\{stem}{names[statistic]}Diff}}{{{num_math(row['point'], d)}}}")
            lines.append(
                rf"\newcommand{{\{stem}{names[statistic]}DiffCI}}{{{ci_bracket(row['lo'], row['hi'], d)}}}"
            )
        for statistic in NIPD_STATISTICS:
            row = basis.differences.loc[(pair, statistic)]
            lines.append(rf"\newcommand{{\{stem}{names[statistic]}Diff}}{{{pct(row['point'])}}}")
            lines.append(
                rf"\newcommand{{\{stem}{names[statistic]}DiffSd}}{{{spread_pct(row['spread'])}}}"
            )
    return lines
