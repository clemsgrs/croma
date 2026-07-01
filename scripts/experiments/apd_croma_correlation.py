"""Correlate CRoMa (and RI/MaRI) with downstream APD across models.

Answers feedback concern #2 ("does CRoMa predict a downstream robustness
outcome?"): join the faithful representation metrics with PathoROB's Average
Performance Drop and report rank correlation across models, per dataset and
pooled.

Hypothesis (pre-registered): a more confounder-robust representation (higher
CRoMa) gives a downstream probe fewer exploitable shortcuts, so it suffers a
smaller drop -> APD closer to 0. APD is negative, so we expect a POSITIVE
Spearman(CRoMa, APD), strongest for the OOD probe (cross-centre generalisation,
the conceptual sibling of CRoMa's cross-confounder contrast).

RI/MaRI are reported alongside purely as a sanity check that APD is a sensible
yardstick — no claim that CRoMa predicts APD better than RI.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

REPO = Path("/data/pathology/projects/clement/code/croma")
# The first three are the faithful PathoROB benchmarks that constitute the headline
# validation table (tab:apd-correlation). `prostate` is the caveated second-organ
# extension: its APD_ID corroborates the validation, but its APD_OOD rests on a single
# small out-of-domain centre (NUS: 300/class, no Gleason-3) and is reported separately
# with that caveat -- do NOT read the 4-benchmark `pooled` APD_OOD as the headline.
DATASETS = ["camelyon", "tcga_4x4", "tolkach", "prostate"]
# The headline validation table (tab:apd-correlation) pools over ONLY the three faithful
# PathoROB benchmarks (48 model-benchmark pairs). Prostate is excluded from this pool: its
# single-centre OOD arm cannot be pooled into a cross-benchmark APD_OOD statistic (see the
# `prostate` caveat above). `headline` is that 3-benchmark pool; `pooled` is all four.
HEADLINE_DATASETS = ["camelyon", "tcga_4x4", "tolkach"]
METRICS = ["croma", "ri", "mari"]

# Canonical faithful (n=20,400) metric dirs — the exact CSVs the paper's tables and
# figures read, with CRoMa already on the signed-margin scale the paper defines. The
# `dataset` column there holds the dir name, so we override it with the APD key below.
METRIC_DIR = {
    "camelyon": "output/faithful/pathorob-camelyon-faithful",
    "tcga_4x4": "output/faithful/pathorob-tcga-4x4",
    "tolkach": "output/faithful/pathorob-tolkach-esca-faithful",
    "prostate": "output/prostate-shift-binary-kirumc",
}


def load_joined(apd_csv):
    apd = pd.read_csv(apd_csv)
    frames = []
    for ds in DATASETS:
        m = pd.read_csv(REPO / METRIC_DIR[ds] / "results/metrics.csv")
        m["dataset"] = ds  # align with APD's dataset key (CSV stores the dir name)
        # Defensive: the canonical dirs are already signed-margin CRoMa; only convert
        # if a legacy ratio-scale CSV (any value > 1) is ever pointed at here.
        if m["croma"].max() > 1.0:
            m["croma"] = (m["croma"] - 1.0) / (m["croma"] + 1.0)
        frames.append(m)
    faith = pd.concat(frames, ignore_index=True)
    df = apd.merge(faith[["dataset", "model", *METRICS, "croma_ltm_alpha"]], on=["dataset", "model"], how="inner")
    missing = set(zip(apd["dataset"], apd["model"])) - set(zip(df["dataset"], df["model"]))
    if missing:
        print(f"[warn] {len(missing)} (dataset,model) APD rows had no faithful metric and were dropped: {sorted(missing)}")
    return df


def corr_block(df, target):
    """Spearman + Pearson of each metric vs APD target, per dataset and pooled."""
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

    res = pd.concat([corr_block(df, "apd_ood"), corr_block(df, "apd_id")], ignore_index=True)
    res.to_csv(out_dir / "apd_correlation.csv", index=False)

    pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.3f}")
    for target in ["apd_ood", "apd_id"]:
        print(f"\n===== Spearman( metric , {target} ) — expect POSITIVE =====")
        piv = (res[res["target"] == target]
               .pivot(index="metric", columns="scope", values="spearman")
               .reindex(index=METRICS, columns=[*DATASETS, "headline", "pooled"]))
        print(piv.to_string())
    print(f"\nwrote {out_dir/'apd_correlation.csv'} and {out_dir/'apd_metrics_joined.csv'}")


if __name__ == "__main__":
    apd_csv = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "output/apd/apd.csv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "output/apd")
    main(apd_csv, out_dir)
