"""Supplementary table: per-model CRoMa on the non-TCGA cohorts vs the TCGA cohort
of Tolkach-ESCA, isolating the pretraining-domain-overlap confound.

Tolkach-ESCA mixes one TCGA cohort (VALSET3_TCGA) with three non-TCGA cohorts,
so the per-sample CRoMa difference between them is a within-dataset test for an
in-distribution advantage. We report each cohort's median CRoMa and the TCGA
"boost": the ratio of typed-distance odds (1+CRoMa)/(1-CRoMa) between the TCGA
and non-TCGA cohorts. The odds ratio (not the raw CRoMa gap) is the fair
cross-model comparator, because CRoMa's bounded curve compresses differences at
the biology-dominant end and inflates them near the contested boundary -- so a
raw gap conflates the TCGA effect with where a model sits on the curve. Sorted by
boost (descending); the TCGA-only-pretrained Midnight-12k is the clear outlier.

The per-sample CSV is produced by ``scripts/studies/pretraining_overlap.py`` and stores
the bounded CRoMa margin in (-1, 1) directly in its ``croma_m*`` columns -- NOT the legacy
typed-distance ratio r = dbar^OS/dbar^SO. We aggregate the margin as-is; a guard fires if a
legacy ratio-scale file (values outside (-1, 1)) is ever fed in, which would be
double-converted by ``_odds``.

Run: python scripts/repro/generate_pretraining_overlap_table.py
"""

from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M

PER_SAMPLE = Path("output/studies/pretraining-overlap/per_sample_metrics.csv")
TCGA_CENTER = "VALSET3_TCGA"
HEADLINE_COL = f"croma_m{int(CROMA_HEADLINE_M)}"
OUT = Path("paper/sections/supp_pretraining_overlap.tex")


def _odds(croma: float) -> float:
    """Typed-distance odds (1+CRoMa)/(1-CRoMa) == dbar^OS/dbar^SO."""
    return (1.0 + croma) / (1.0 - croma)


def build() -> str:
    df = pd.read_csv(PER_SAMPLE, usecols=["model", "confounder", HEADLINE_COL])
    assert df[HEADLINE_COL].abs().max() < 1.0, (
        f"{PER_SAMPLE} column {HEADLINE_COL} is out of (-1, 1): it looks like a legacy "
        "typed-distance ratio r, not the bounded CRoMa margin. Feeding a ratio-scale file "
        "here would double-convert it in _odds -- regenerate via "
        "scripts/studies/pretraining_overlap.py."
    )

    rows = []
    for model, g in df.groupby("model"):
        tcga = g.loc[g.confounder == TCGA_CENTER, HEADLINE_COL].median()
        rest = g.loc[g.confounder != TCGA_CENTER, HEADLINE_COL].median()
        rows.append({"model": model, "tcga": tcga, "rest": rest,
                     "boost": _odds(tcga) / _odds(rest)})
    r = pd.DataFrame(rows).sort_values("boost", ascending=False).reset_index(drop=True)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & \code{CRoMa} (non-TCGA) & \code{CRoMa} (TCGA) & TCGA boost \\",
        r"\hline",
    ]
    for _, x in r.iterrows():
        bold = x["model"] == "Midnight-12k"
        name = rf"\textbf{{{x['model']}}}" if bold else x["model"]
        cells = f"{name} & {x['rest']:.2f} & {x['tcga']:.2f} & {x['boost']:.2f}$\\times$"
        lines.append(cells + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{\textbf{The pretraining-domain-overlap confound, localised within "
        rf"Tolkach-ESCA.}} Median per-sample $\mcode{{CRoMa}}(m{{=}}{int(CROMA_HEADLINE_M)})$ on the "
        r"three non-TCGA cohorts versus the TCGA cohort (\texttt{VALSET3\_TCGA}), per model, with "
        r"the \emph{TCGA boost}---the ratio of typed-distance odds "
        r"$(1{+}\mcode{CRoMa})/(1{-}\mcode{CRoMa})$ between the two cohorts---sorted descending. "
        r"The odds ratio, not the raw \code{CRoMa} gap, is the fair cross-model comparator: "
        r"\code{CRoMa}'s bounded scale compresses differences at the biology-dominant end, so a "
        r"raw gap would conflate the TCGA effect with a model's position on the curve. Every model "
        r"scores somewhat higher on the TCGA cohort---it is intrinsically cleaner---but "
        r"\code{Midnight-12k}, pretrained \emph{exclusively} on TCGA~\cite{midnight}, is the clear "
        r"outlier: its odds of a biology-dominant margin are $2.5\times$ higher on the TCGA cohort, "
        r"against at most ${\sim}1.5\times$ for every other model. The advantage is an "
        r"in-distribution effect of pretraining-domain overlap, not general robustness; it is "
        r"invisible on the out-of-distribution Camelyon benchmark, where \code{Midnight-12k} ranks "
        r"only sixth.}",
        r"\label{tab:pretraining-overlap}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(build())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
