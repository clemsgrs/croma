"""Compare prostate-shift robustness at two biological label resolutions.

Reads three KI+RUMC runs on a shared patch pool and answers: does measuring robustness
at finer biological granularity (4-class Gleason) change what we conclude versus the
coarse benign-vs-tumour binary?

  natural-binary   output/prostate-shift-binary-kirumc            (dataset_wide, 1,850/cell)
  gradebal-binary  output/prostate-shift-gradebal-binary-kirumc-paired (1 pair, 1,440/cell)
  4-class          output/prostate-shift-4class-kirumc-paired      (6 grade-pairs, 480/cell)

Outputs (printed + CSV under the 4-class results dir):
  1. per-model CCMR / RI / MaRI / support across the three settings;
  2. rank-stability: Spearman of model rankings between settings, per metric
     (pre-registered expectation: RI most reordered, CCMR least);
  3. per-grade-pair breakdown of the 4-class run (pooled over models) -- where the
     confounder dominance actually concentrates, which the binary cannot resolve.

Levels are NOT absolute-comparable across granularities (the 4-class folds in grade-vs-
grade pairs the binary cannot express, and restricts each pair's candidate pool); the
comparison is of rankings and of the per-pair structure, not raw magnitudes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
SETTINGS = {
    "natbin": "output/prostate-shift-binary-kirumc",
    "gradebal": "output/prostate-shift-gradebal-binary-kirumc-paired",
    "fourclass": "output/prostate-shift-4class-kirumc-paired",
}
OUT_DIR = REPO / SETTINGS["fourclass"] / "results"
PAIR_LABEL = {
    "benign+gleason_3__KI_RUMC": "benign-G3",
    "benign+gleason_4__KI_RUMC": "benign-G4",
    "benign+gleason_5__KI_RUMC": "benign-G5",
    "gleason_3+gleason_4__KI_RUMC": "G3-G4",
    "gleason_3+gleason_5__KI_RUMC": "G3-G5",
    "gleason_4+gleason_5__KI_RUMC": "G4-G5",
}


def _to_margin(s: pd.Series) -> pd.Series:
    """Signed-margin CCMR is canonical here; convert only a stray ratio-scale column."""
    return (s - 1.0) / (s + 1.0) if float(s.max()) > 1.0 else s


def _load_metrics(rel: str) -> pd.DataFrame:
    m = pd.read_csv(REPO / rel / "results/metrics.csv")
    m = m[["model", "ccmr", "ri", "mari", "ri_undefined_frac", "bio_knn_bacc",
           "confounder_knn_bacc"]].copy()
    m["ccmr"] = _to_margin(m["ccmr"].astype(float))
    m["support"] = 1.0 - m["ri_undefined_frac"].astype(float)
    return m.drop(columns="ri_undefined_frac")


def per_model_table() -> pd.DataFrame:
    frames = {k: _load_metrics(v).set_index("model") for k, v in SETTINGS.items()}
    out = pd.DataFrame(index=frames["fourclass"].index)
    for metric in ("ccmr", "ri", "mari", "support"):
        for setting in SETTINGS:
            out[f"{metric}_{setting}"] = frames[setting][metric]
    # sort by the fully-defined 4-class CCMR (house style: CCMR is primary, k-free)
    return out.sort_values("ccmr_fourclass", ascending=False)


def rank_stability(tbl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("ccmr", "ri", "mari"):
        for a, b in (("natbin", "gradebal"), ("natbin", "fourclass"), ("gradebal", "fourclass")):
            sub = tbl[[f"{metric}_{a}", f"{metric}_{b}"]].dropna()
            rho, p = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
            rows.append(dict(metric=metric, pair=f"{a}->{b}", n=len(sub),
                             spearman=round(float(rho), 3), p=float(p)))
    return pd.DataFrame(rows)


def per_pair_breakdown() -> pd.DataFrame:
    d = pd.read_csv(REPO / SETTINGS["fourclass"] / "results/per_sample_metrics.csv")
    rows = []
    for subset, g in d.groupby("subset"):
        ri_def = g[g["ri_defined"]]
        mari_def = g[g["mari_defined"]]
        rows.append(dict(
            pair=PAIR_LABEL.get(subset, subset),
            ccmr_m5=round(float(g["ccmr_m5"].mean()), 3),            # mean per-sample margin, m=5
            ri=round(float(ri_def["ri"].mean()), 3) if len(ri_def) else np.nan,
            mari=round(float(mari_def["mari"].mean()), 3) if len(mari_def) else np.nan,
            support=round(float(g["ri_defined"].mean()), 3),
            kind="benign-vs-grade" if subset.startswith("benign") else "grade-vs-grade",
        ))
    order = ["benign-G3", "benign-G4", "benign-G5", "G3-G4", "G3-G5", "G4-G5"]
    out = pd.DataFrame(rows)
    return out.set_index("pair").reindex(order).reset_index()


def main() -> int:
    tbl = per_model_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl.round(3).to_csv(OUT_DIR / "granularity_comparison.csv")
    pd.set_option("display.width", 200, "display.max_columns", 40,
                  "display.float_format", lambda x: f"{x:.3f}")

    print("===== per-model CCMR / support across granularities (sorted by 4-class CCMR) =====")
    show = tbl[["ccmr_natbin", "ccmr_gradebal", "ccmr_fourclass",
                "support_natbin", "support_gradebal", "support_fourclass"]]
    print(show.to_string())

    print("\n===== rank stability (Spearman of model rankings between settings) =====")
    rs = rank_stability(tbl)
    print(rs.to_string(index=False))
    print("\nmean |Spearman| per metric (higher = ranking more preserved across granularity):")
    print(rs.groupby("metric")["spearman"].mean().round(3).to_string())

    print("\n===== 4-class per-grade-pair breakdown (pooled over 16 models) =====")
    pp = per_pair_breakdown()
    print(pp.to_string(index=False))
    pp.to_csv(OUT_DIR / "granularity_per_pair.csv", index=False)

    bvg = pp[pp["kind"] == "benign-vs-grade"]["ccmr_m5"].mean()
    gvg = pp[pp["kind"] == "grade-vs-grade"]["ccmr_m5"].mean()
    print(f"\nmean CCMR(m=5): benign-vs-grade={bvg:.3f}  grade-vs-grade={gvg:.3f}")
    print(f"wrote {OUT_DIR/'granularity_comparison.csv'} and {OUT_DIR/'granularity_per_pair.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
