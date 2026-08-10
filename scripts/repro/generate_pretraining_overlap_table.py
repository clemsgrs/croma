"""Supplementary table: per-model CRoMa on the non-TCGA cohorts vs the TCGA cohort
of Tolkach-ESCA, isolating the pretraining-domain-overlap confound.

Tolkach-ESCA mixes one TCGA cohort (VALSET3_TCGA) with three non-TCGA cohorts,
so the per-sample CRoMa difference between them is a within-dataset test for an
in-distribution advantage. We report each cohort's median CRoMa and the TCGA
"boost": the ratio of typed-distance odds (1+CRoMa)/(1-CRoMa) between the TCGA
and non-TCGA cohorts. The odds ratio (not the raw CRoMa gap) is the fair
cross-model comparator, because CRoMa's bounded curve compresses differences at
the biology-dominant end and inflates them near the contested boundary -- so a
raw gap conflates the TCGA effect with where a model sits on the curve. Rows are sorted by
boost (descending).

The medians, the boost, and the scale guard live in ``_overlap.py``: this table, the
Section 3.4's inline scalars all read that one basis.

Run: python scripts/repro/generate_pretraining_overlap_table.py
"""

import sys
from pathlib import Path

import pandas as pd

from _overlap import rows as overlap_rows
from _model_provenance import exposed_models_for_domain
from _paper_tables import CROMA_HEADLINE_M

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench")
)  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting.style import CONTROL_MODEL  # noqa: E402

OUT = Path("paper/sections/supp/pretraining_overlap.tex")
METADATA = Path(__file__).resolve().parent.parent / "bench" / "model_metadata.csv"


def build() -> str:
    r = overlap_rows()  # every encoder incl. the control, ordered by boost (descending)
    ranked = r[r["model"] != CONTROL_MODEL]
    control = r[r["model"] == CONTROL_MODEL]
    # The TCGA-exposed encoders carry a dagger, exactly as in the two TCGA results tables
    # and the rank-aggregate Pareto -- one source (model_metadata.csv), so the sets never drift.
    md = pd.read_csv(METADATA)
    exposed = set(exposed_models_for_domain(md, "tcga", set(ranked["model"])))

    def _row(x: pd.Series) -> str:
        name = x["model"] + (r"$^{\dagger}$" if x["model"] in exposed else "")
        return f"{name} & {x['rest']:.2f} & {x['tcga']:.2f} & {x['boost']:.2f}$\\times$" + r" \\"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Model & Tolkach-ESCA & TCGA extension & Boost \\",
        r"\hline",
    ]
    lines += [_row(x) for _, x in ranked.iterrows()]
    # The natural-image control sits in its own band beneath the rule, as in the results tables.
    if not control.empty:
        lines.append(r"\hline")
        lines += [_row(x) for _, x in control.iterrows()]

    caption = (
        r"\caption{\textbf{Median \code{CRoMa} on Tolkach-ESCA and on its TCGA extension.} "
        rf"Median per-sample $\mcode{{CRoMa}}(m{{=}}{int(CROMA_HEADLINE_M)})$ for the "
        rf"historical {len(ranked)}-encoder pathology subset, computed separately over the "
        r"three original Tolkach-ESCA cohorts and over the held-out TCGA cases. The \emph{boost} "
        r"measures how much further the nearest same-confounder biological distractors sit "
        r"(relative to the nearest cross-confounder biological matches) on the TCGA cases than on "
        r"the original cohorts. It is the between-cohort ratio of "
        r"$r=d^{\mcode{OS}}_m/d^{\mcode{SO}}_m$, the typed-distance ratio on which \code{CRoMa} is "
        r"defined (Section~\ref{sec:methods-croma}). "
        rf"$\dagger$ marks the ${len(exposed)}$ TCGA-exposed encoders (Table~\ref{{tab:model-summary}}). "
        rf"The natural-image control \code{{{CONTROL_MODEL}}} is shown separately.}}"
    )

    lines += [
        r"\hline",
        r"\end{tabular}",
        caption,
        r"\label{tab:pretraining-overlap}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(build())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
