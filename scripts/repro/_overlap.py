import sys
"""Float basis for the pretraining-domain-overlap study (Tolkach-ESCA).

Tolkach-ESCA mixes one TCGA cohort (``VALSET3_TCGA``) with three non-TCGA cohorts, so the
per-sample CRoMa difference between them is a within-dataset test for an in-distribution
advantage. Three consumers ask this artifact the same question and must get the same answer:

* ``generate_pretraining_overlap_table.py`` renders ``tab:pretraining-overlap``;
* ``_cross_benchmark.tolkach_median_without_tcga`` asks whether the leader's Tolkach rank
  survives dropping the TCGA cohort, for ``fig:cross-benchmark``'s caption;
* ``generate_paper_values.py`` emits the scalars Section 3.4 cites inline.

The first two each had their own median-over-non-TCGA-cohorts groupby. They agreed, which is
the dangerous case: nothing would have flagged it if one had been changed. One basis, three
renderings -- the rule CONTEXT.md states for a *float* basis, applied to a study.
"""

from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)

REPO = Path(__file__).resolve().parents[2]

PER_SAMPLE = REPO / "output/studies/pretraining-overlap/per_sample_metrics.csv"
TCGA_COHORT = "VALSET3_TCGA"
HEADLINE_COL = f"croma_m{int(CROMA_HEADLINE_M)}"


def odds(croma: float) -> float:
    """Typed-distance odds ``(1+CRoMa)/(1-CRoMa)``, i.e. ``dbar^OS / dbar^SO``."""
    return (1.0 + croma) / (1.0 - croma)


def rows(include_control: bool = True) -> pd.DataFrame:
    """Per-model ``(model, rest, tcga, boost)``, sorted by TCGA boost, best first.

    ``rest`` is the median per-sample CRoMa over the three non-TCGA cohorts, ``tcga`` the
    median over the TCGA cohort, and ``boost`` the ratio of their typed-distance odds. The
    odds ratio -- not the raw CRoMa gap -- is the fair cross-model comparator, because
    CRoMa's bounded curve compresses differences at the biology-dominant end, so a raw gap
    would conflate the TCGA effect with where a model sits on the curve.

    Raises when the study has not been run. A caption or a macro that quietly rewords itself
    around a missing artifact is how a rendered table once grew confidence intervals nobody
    asked for; ``build_paper.py`` names the skip instead.
    """
    if not PER_SAMPLE.exists():
        raise FileNotFoundError(
            f"{PER_SAMPLE} is absent, so nothing can say whether the Tolkach lead survives "
            "dropping its TCGA cohort. Run scripts/studies/pretraining_overlap.py."
        )
    df = pd.read_csv(PER_SAMPLE, usecols=["model", "confounder", HEADLINE_COL])
    if df[HEADLINE_COL].abs().max() >= 1.0:
        raise ValueError(
            f"{PER_SAMPLE} column {HEADLINE_COL} is out of (-1, 1): it looks like a legacy "
            "typed-distance ratio r, not the bounded CRoMa margin. Feeding a ratio-scale "
            "file here would double-convert it in `odds` -- regenerate via "
            "scripts/studies/pretraining_overlap.py."
        )
    if not include_control:
        from plotting.style import CONTROL_MODEL

        df = df[df["model"] != CONTROL_MODEL]

    out = []
    for model, g in df.groupby("model"):
        tcga = g.loc[g.confounder == TCGA_COHORT, HEADLINE_COL].median()
        rest = g.loc[g.confounder != TCGA_COHORT, HEADLINE_COL].median()
        out.append({"model": model, "tcga": tcga, "rest": rest,
                    "boost": odds(tcga) / odds(rest)})
    return pd.DataFrame(out).sort_values("boost", ascending=False).reset_index(drop=True)
