"""Correlate CRoMa (and RI/MaRI) with the downstream drop across models.

Join the representation metrics with downstream performance degradation and report
rank correlation across models within each benchmark.

The primary target is nIPD (normalized Integrated Performance Degradation), which
normalizes by above-chance baseline skill and integrates over Cramér's V. PathoROB's
APD is retained as a continuity analysis.

Hypothesis (pre-registered): a more confounder-robust representation (higher
CRoMa) gives a downstream probe fewer exploitable shortcuts, so it suffers a
smaller drop -> nIPD closer to 0. Because nIPD is signed, we expect a positive
Spearman(CRoMa, nIPD), with ID as the cleaner shortcut-susceptibility endpoint.
OOD additionally contains transfer to an unseen acquisition distribution.

RI/MaRI are reported alongside purely as a sanity check that the drop is a sensible
yardstick — no claim that CRoMa predicts it better than RI.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from loaders import (
    CORR_METRICS as METRICS,
    DATASET_KEYS as DATASETS,
    REPO,
    load_joined,
    ranked,
)


def corr_block(df, target):
    """Spearman correlation of each metric with one target, per benchmark.

    Computed on the ranked panel. The control would be doubly flattered here: its CRoMa is
    high because its biological neighbourhoods are poor, and a downstream reduction can
    flatter a model with little signal to lose. Both effects push the same way, and
    including it would inflate the association.
    """
    df = ranked(df)
    rows = []
    for metric in METRICS:
        for scope in DATASETS:
            sub = df[df["dataset"] == scope]
            sub = sub[[metric, target]].dropna()
            if len(sub) < 3:
                continue
            rho, p_s = spearmanr(sub[metric], sub[target])
            rows.append(dict(target=target, metric=metric, scope=scope, n=len(sub),
                             spearman=rho, spearman_p=p_s))
    return pd.DataFrame(rows)


def main(apd_csv, out_dir):
    out_dir = Path(out_dir)
    df = load_joined(apd_csv)
    df.to_csv(out_dir / "apd_metrics_joined.csv", index=False)

    # nIPD (primary, ID first) followed by PathoROB APD (reference).
    targets = ["nipd_id", "nipd_ood", "apd_id", "apd_ood"]
    res = pd.concat([corr_block(df, t) for t in targets], ignore_index=True)
    res.to_csv(out_dir / "apd_correlation.csv", index=False)

    pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.3f}")
    for target in targets:
        ref = " (reference)" if target.startswith("apd_") else " (primary)"
        print(f"\n===== Spearman( metric , {target} ){ref} — expect POSITIVE =====")
        piv = (res[res["target"] == target]
               .pivot(index="metric", columns="scope", values="spearman")
               .reindex(index=METRICS, columns=DATASETS))
        print(piv.to_string())
    print(f"\nwrote {out_dir/'apd_correlation.csv'} and {out_dir/'apd_metrics_joined.csv'}")


if __name__ == "__main__":
    apd_csv = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "output/studies/apd/apd.csv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "output/studies/apd")
    main(apd_csv, out_dir)
