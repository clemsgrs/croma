"""Supplementary table: per-model CCMR on the TCGA cohort vs the non-TCGA cohorts
of Tolkach-ESCA, isolating the pretraining-domain-overlap confound.

Tolkach-ESCA mixes one TCGA cohort (VALSET3_TCGA) with three non-TCGA cohorts,
so the per-sample CCMR gap between them is a within-dataset test for an
in-distribution advantage. Sorted by the TCGA-favouring gap (descending); the
TCGA-only-pretrained Midnight-12k is the extreme outlier.

Run: python scripts/experiments/generate_pretraining_overlap_table.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from croma.metrics.ccmr import CCMR_HEADLINE_M  # noqa: E402

PER_SAMPLE = Path("output/pathorob-tolkach-esca/results/per_sample_metrics.csv")
TCGA_CENTER = "VALSET3_TCGA"
HEADLINE_COL = f"ccmr_m{int(CCMR_HEADLINE_M)}"
OUT = Path("paper/sections/supp_pretraining_overlap.tex")


def build() -> str:
    df = pd.read_csv(PER_SAMPLE, usecols=["model", "confounder", HEADLINE_COL])
    rows = []
    for model, g in df.groupby("model"):
        tcga = g.loc[g.confounder == TCGA_CENTER, HEADLINE_COL].median()
        rest = g.loc[g.confounder != TCGA_CENTER, HEADLINE_COL].median()
        rows.append({"model": model, "tcga": tcga, "rest": rest,
                     "gap": tcga - rest, "ratio": tcga / rest})
    r = pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Model & \code{CCMR} (TCGA) & \code{CCMR} (non-TCGA) & gap & ratio \\",
        r"\hline",
    ]
    for _, x in r.iterrows():
        bold = x["model"] == "Midnight-12k"
        name = rf"\textbf{{{x['model']}}}" if bold else x["model"]
        cells = (f"{name} & {x['tcga']:.2f} & {x['rest']:.2f} & "
                 f"{x['gap']:.2f} & {x['ratio']:.2f}$\\times$")
        lines.append(cells + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{\textbf{The pretraining-domain-overlap confound, localised within "
        rf"Tolkach-ESCA.}} Median per-sample $\mcode{{CCMR}}(m{{=}}{int(CCMR_HEADLINE_M)})$ on the TCGA cohort "
        r"(\texttt{VALSET3\_TCGA}) versus the three non-TCGA cohorts, per model, sorted by the "
        r"TCGA-favouring gap. Every model scores somewhat higher on the TCGA cohort---it is "
        r"intrinsically cleaner---but \code{Midnight-12k}, pretrained \emph{exclusively} on "
        r"TCGA~\cite{midnight}, is a stark outlier: its TCGA-vs-rest gap is the largest of all "
        r"$16$ models and more than ten times the next. The advantage is an in-distribution "
        r"effect of pretraining-domain overlap, not general robustness; it is invisible on the "
        r"out-of-distribution Camelyon benchmark, where \code{Midnight-12k} ranks only sixth.}",
        r"\label{tab:pretraining-overlap}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(build())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
