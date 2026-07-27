"""Correlate CRoMa (and RI/MaRI) with the downstream drop across models.

Answers feedback concern #2 ("does CRoMa predict a downstream robustness
outcome?"): join the faithful representation metrics with the downstream
performance-drop targets and report rank correlation across models, per dataset
and pooled.

Primary target is nAPD (normalized Average Performance Drop), our skill-normalized
refinement of PathoROB's APD; APD itself is reported alongside for reference. On the
tile benchmarks every baseline is well above chance, so nAPD and APD rank-agree
(Spearman >= 0.94) and the correlation with CRoMa is essentially the same for both --
which is the point: the refinement does not disturb the downstream validation, it only
changes what happens in the near-chance regime (the slide OOD arm, excluded here).

Hypothesis (pre-registered): a more confounder-robust representation (higher
CRoMa) gives a downstream probe fewer exploitable shortcuts, so it suffers a
smaller drop -> nAPD closer to 0. nAPD is negative, so we expect a POSITIVE
Spearman(CRoMa, nAPD), strongest for the OOD probe (cross-centre generalisation,
the conceptual sibling of CRoMa's cross-confounder contrast).

RI/MaRI are reported alongside purely as a sanity check that the drop is a sensible
yardstick — no claim that CRoMa predicts it better than RI.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, pearsonr

from loaders import (
    CORR_METRICS as METRICS,
    DATASET_KEYS as DATASETS,
    HEADLINE_DATASETS,
    REPO,
    load_joined,
    ranked,
)


def corr_block(df, target):
    """Spearman + Pearson of each metric vs APD target, per dataset and pooled.

    Computed on the ranked panel. The control would be doubly flattered here: its CRoMa is
    high because its biological neighbourhoods are poor, and APD is a *relative* drop, so a
    model with little accuracy to lose is scored leniently. Both effects push the same way,
    and including it would inflate every rho in the table.
    """
    df = ranked(df)
    rows = []
    for metric in METRICS:
        for scope in [*DATASETS, "headline", "pooled"]:
            if scope == "pooled":
                sub = df
            elif scope == "headline":
                sub = df[df["dataset"].isin(HEADLINE_DATASETS)]
            else:
                sub = df[df["dataset"] == scope]
            sub = sub[[metric, target]].dropna()
            if len(sub) < 3:
                continue
            rho, p_s = spearmanr(sub[metric], sub[target])
            r, p_p = pearsonr(sub[metric], sub[target])
            rows.append(dict(target=target, metric=metric, scope=scope, n=len(sub),
                             spearman=rho, spearman_p=p_s, pearson=r, pearson_p=p_p))
    return pd.DataFrame(rows)


def main(apd_csv, out_dir):
    out_dir = Path(out_dir)
    df = load_joined(apd_csv)
    df.to_csv(out_dir / "apd_metrics_joined.csv", index=False)

    # nAPD (primary) first, APD (reference) second.
    targets = ["napd_ood", "napd_id", "apd_ood", "apd_id"]
    res = pd.concat([corr_block(df, t) for t in targets], ignore_index=True)
    res.to_csv(out_dir / "apd_correlation.csv", index=False)

    pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.3f}")
    for target in targets:
        ref = " (reference)" if target.startswith("apd_") else " (primary)"
        print(f"\n===== Spearman( metric , {target} ){ref} — expect POSITIVE =====")
        piv = (res[res["target"] == target]
               .pivot(index="metric", columns="scope", values="spearman")
               .reindex(index=METRICS, columns=[*DATASETS, "headline", "pooled"]))
        print(piv.to_string())
    print(f"\nwrote {out_dir/'apd_correlation.csv'} and {out_dir/'apd_metrics_joined.csv'}")


if __name__ == "__main__":
    apd_csv = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "output/studies/apd/apd.csv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "output/studies/apd")
    main(apd_csv, out_dir)
