"""Supplementary table ``tab:finetune-pairs``: each robustness fine-tune against its parent.

One block per pair, one row group per tile cohort: parent -> fine-tune, the difference
(fine-tune minus parent) and its interval. Every number comes from the study CSV
(``scripts/studies/finetune_pairs.py``) through ``_finetune_pairs.load``, which fails when the
study is missing, no longer matches the pairs ``model_metadata.csv`` declares, or predates a
re-run of the benchmark runs, APD table or APD cells it read. The caption's disclosure,
development-selection and exposure statements are generated from the metadata, so none of
them can outlive the fact behind it.

Run: python scripts/repro/generate_finetune_pairs_table.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from _finetune_pairs import interval, load, signed
from _model_provenance import exposed_models_for_domain
from _paper_tables import CROMA_HEADLINE_M, CaptionClaimError, bare_num
from paper_manifest import by_benchmark

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "bench"))
from plotting.style import published_model_name  # noqa: E402

import finetune_pairs as study  # noqa: E402  (on sys.path via _finetune_pairs)

OUT = Path("paper/sections/supp/finetune_pairs.tex")
BIBLIOGRAPHY = Path("paper/references.bib")

#: Table columns: (study column stem, decimals, has an interval). Decimals follow the
#: results tables; biological accuracy and rank are point values.
COLUMNS = (
    ("croma", 2, True),
    ("f0", 3, True),
    ("ltm", 2, True),
    ("bio_bacc", 3, False),
    ("nipd_id", 3, True),
    ("nipd_ood", 3, True),
)
HEADER = (
    r"Cohort & \code{CRoMa} & $F(0)$ & $\mcode{LTM}_{10}$ & bio bacc & "
    r"\code{nIPD}$_\mathrm{ID}$ & \code{nIPD}$_\mathrm{OOD}$ & Rank \\"
)
N_COLUMNS = 2 + len(COLUMNS)

#: How each disclosed kind of benchmark use during development is worded (ADR-0020). A kind
#: with no wording here fails the build instead of being published under the wrong legend.
SELECTION_LEGENDS = {
    "pathorob-ri": (
        r"$\ddagger$ PathoROB \code{RI} on these cohorts was used during the fine-tune's "
        r"development, e.g.\ in checkpoint selection, as disclosed by its authors; its "
        r"differences are shown unadjusted."
    ),
}
#: Reader-facing names of the bootstrap strata columns on the PathoROB tile cohorts.
STRATUM_NAMES = {"label": "class", "confounder": "centre"}


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _single(frame: pd.DataFrame, column: str) -> object:
    values = frame[column].unique()
    if len(values) != 1:
        raise CaptionClaimError(f"the caption states one {column}, the study holds {values}")
    return values[0]


def _source(source: str | None, bib_keys: set[str]) -> str:
    if source is None:
        return "undisclosed"
    if re.match(r"https?://", source):
        return rf"\url{{{source}}}"
    if source not in bib_keys:
        raise CaptionClaimError(
            f"disclosure source {source!r} is neither a URL nor a key of {BIBLIOGRAPHY}"
        )
    return rf"\cite{{{source}}}"


def _disclosure(pairs: list[study.FinetunePair], bib_keys: set[str]) -> str:
    silent = [
        published_model_name(p.finetune)
        for p in pairs
        if p.method_source is None and p.data_source is None
    ]
    parts = [f"both undisclosed for {_and(silent)}"] if silent else []
    parts += [
        f"{published_model_name(p.finetune)}, method {_source(p.method_source, bib_keys)} "
        f"and data {_source(p.data_source, bib_keys)}"
        for p in pairs
        if p.method_source is not None or p.data_source is not None
    ]
    return (
        "Adaptation method and adaptation data as disclosed by each fine-tune's authors: "
        f"{'; '.join(parts)}."
    )


def _strata(frame: pd.DataFrame) -> str:
    groups: dict[str, list[str]] = {}
    for benchmark, strata in zip(frame["benchmark"], frame["strata"]):
        names = groups.setdefault(strata, [])
        short = by_benchmark(benchmark).short_name
        if short not in names:
            names.append(short)
    phrases = []
    for strata, cohorts in groups.items():
        unknown = [s for s in strata.split("+") if s not in STRATUM_NAMES]
        if unknown:
            raise CaptionClaimError(f"no reader-facing name for bootstrap strata {unknown}")
        named = r"$\times$".join(STRATUM_NAMES[s] for s in strata.split("+"))
        phrases.append(f"within {named} on {_and(cohorts)}")
    return _and(phrases)


def _exposure(frame: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, list[str]]:
    """Cohort -> the pair members (registry names, pair order) exposed to its source domain."""
    exposed: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        members = [row["parent"], row["finetune"]]
        domain = by_benchmark(row["benchmark"]).exposure_domain
        hit = exposed_models_for_domain(metadata, domain, set(members))
        if hit:
            exposed.setdefault(row["benchmark"], []).extend(m for m in members if m in hit)
    return exposed


def _rows(row: pd.Series, daggered: bool, percent: str) -> list[str]:
    """A cohort's three rows; ``percent`` is the caption's level, so the CI label shares it."""
    cohort = by_benchmark(row["benchmark"]).short_name + (r"$^{\dagger}$" if daggered else "")
    values, deltas, intervals = [cohort], [r"\quad $\Delta$"], [rf"\quad {percent} CI"]
    for stem, decimals, has_interval in COLUMNS:
        parent = bare_num(row[f"parent_{stem}"], decimals)
        finetune = bare_num(row[f"finetune_{stem}"], decimals)
        values.append(rf"${parent} \to {finetune}$")
        deltas.append(signed(row[f"delta_{stem}"], decimals))
        intervals.append(
            interval(row[f"delta_{stem}_lo"], row[f"delta_{stem}_hi"], decimals)
            if has_interval
            else ""
        )
    values.append(rf"${int(row['parent_rank'])} \to {int(row['finetune_rank'])}$")
    deltas.append(signed(int(row["rank_change"]), 0))
    intervals.append("")
    return [" & ".join(cells) + r" \\" for cells in (values, deltas, intervals)]


def build(frame: pd.DataFrame, metadata: pd.DataFrame, *, bib_keys: set[str]) -> str:
    """The ``tab:finetune-pairs`` float for a loaded study frame and the model metadata."""
    pairs = study.finetune_pairs(metadata)
    if list(dict.fromkeys(frame["finetune"])) != [p.finetune for p in pairs]:
        raise CaptionClaimError("the study's fine-tunes are not the pairs the metadata declares")
    exposed = _exposure(frame, metadata)
    level = float(_single(frame, "level"))
    percent = f"{level * 100:g}\\%"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "c" * (N_COLUMNS - 1) + "}",
        r"\hline",
        HEADER,
        r"\hline",
    ]
    for pair in pairs:
        mark = r"$^{\ddagger}$" if pair.benchmark_selection else ""
        title = (
            f"{published_model_name(pair.parent)} $\\rightarrow$ "
            f"{published_model_name(pair.finetune)}{mark}"
        )
        lines.append(rf"\multicolumn{{{N_COLUMNS}}}{{l}}{{\textbf{{{title}}}}} \\")
        for _, row in frame[frame["finetune"] == pair.finetune].iterrows():
            members = exposed.get(row["benchmark"], [])
            daggered = any(m in members for m in (pair.parent, pair.finetune))
            lines += _rows(row, daggered, percent)

    selections = list(dict.fromkeys(p.benchmark_selection for p in pairs if p.benchmark_selection))
    unworded = [s for s in selections if s not in SELECTION_LEGENDS]
    if unworded:
        raise CaptionClaimError(f"no legend wording for benchmark_selection {unworded}")

    caption = [
        r"\textbf{Each robustness-targeted fine-tune against its parent encoder.}",
        "The pairs are read from the model metadata: every tile encoder declared a "
        "robustness-targeted fine-tune, with its declared parent, whose architecture and "
        "weights it starts from.",
        r"For each cohort the first row gives parent $\to$ fine-tune, the $\Delta$ row the "
        f"difference (fine-tune minus parent) and the CI row its {percent} interval.",
        r"\code{CRoMa} is the median per-sample "
        rf"$\mcode{{CRoMa}}(m{{=}}{int(CROMA_HEADLINE_M)})$, $F(0)$ the share of samples with "
        r"$\mcode{CRoMa}\le0$ and $\mcode{LTM}_{10}$ the mean of the lowest decile; bio bacc "
        r"is biological $k$-NN balanced accuracy, whose difference carries no interval; "
        r"\code{nIPD} is the normalised integrated performance degradation of a linear probe "
        "in and out of domain.",
        r"The three \code{CRoMa} intervals are percentile intervals over "
        rf"$B{{=}}{int(_single(frame, 'n_boot'))}$ replicates of one stratified cluster "
        "bootstrap applied to both encoders of a pair, resampling independence groups "
        f"{_strata(frame)}.",
        rf"\code{{nIPD}} intervals resample the ${int(_single(frame, 'nipd_runs'))}$ probe runs "
        "jointly for both encoders, so they reflect training-run variation on a fixed cohort.",
        r"Rank is by median \code{CRoMa} among the "
        rf"${int(_single(frame, 'n_ranked'))}$ ranked pathology encoders, $1$ the highest, and "
        r"its $\Delta$ counts places gained.",
        _disclosure(pairs, bib_keys),
    ]
    caption += [SELECTION_LEGENDS[s] for s in selections]
    if exposed:
        named = "; ".join(
            f"{by_benchmark(b).short_name} for "
            f"{_and([published_model_name(m) for m in members])}"
            for b, members in exposed.items()
        )
        caption.append(
            r"$\dagger$ marks a cohort whose source domain is in the pretraining provenance of "
            rf"the named encoders (Table~\ref{{tab:model-summary}}): {named}."
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{" + " ".join(caption) + "}",
        r"\label{tab:finetune-pairs}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def bibliography_keys(path: Path = BIBLIOGRAPHY) -> set[str]:
    """The entry keys of a BibTeX file."""
    return set(re.findall(r"^@\w+\{([^,\s]+),", path.read_text(), flags=re.MULTILINE))


def main() -> None:
    tex = build(load(), study.read_metadata(), bib_keys=bibliography_keys())
    OUT.write_text(tex)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
