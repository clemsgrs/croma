"""Rebuild CCMR artifacts (margin) from per_sample_metrics as the single source
of truth, for a dataset_wide benchmark dir. Idempotent: per_sample is transformed
to margin in-memory if it is still ratio-valued.

The headline pooled CCMR / q_alpha / LTM and the saved per-sample distribution are
taken at the headline averaging radius m (croma.metrics.ccmr.CCMR_HEADLINE_M; the
full m-sweep columns ccmr_m1..ccmr_mM are left untouched).

Usage: python scripts/experiments/rebuild_ccmr_from_persample.py <output_dir> [alpha] [headline_m]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from croma.metrics.ccmr import CCMR_HEADLINE_M  # noqa: E402


def _m(r):
    return (r - 1.0) / (r + 1.0)


def _trapz(y, x):
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def main() -> int:
    base = Path(sys.argv[1])
    base = base if base.is_absolute() else ROOT / base
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    headline_m = int(sys.argv[3]) if len(sys.argv) > 3 else int(CCMR_HEADLINE_M)
    results = base / "results"

    pdf = pd.read_csv(results / "per_sample_metrics.csv")
    m_cols = sorted((c for c in pdf.columns if c.startswith("ccmr_m") and c[6:].isdigit()),
                    key=lambda c: int(c[6:]))
    m_values = [int(c[6:]) for c in m_cols]
    # per_sample is ratio (all >= 0) or already margin (has negatives)
    is_margin = any((pdf[c].dropna() < 0).any() for c in m_cols)
    if not is_margin:
        for c in m_cols:
            pdf[c] = _m(pdf[c].astype(float))
        pdf.to_csv(results / "per_sample_metrics.csv", index=False)
        print("per_sample transformed ratio -> margin")
    else:
        print("per_sample already margin")

    def stats(vals):
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return float("nan"), float("nan"), float("nan")
        med = float(np.median(v))
        q = float(np.percentile(v, alpha * 100))
        tail = v[v <= q]
        ltm = float(tail.mean()) if tail.size else q
        return med, q, ltm

    pooled: dict[str, dict[int, tuple]] = {}
    for model, g in pdf.groupby("model"):
        pooled[model] = {m: stats(g[f"ccmr_m{m}"].to_numpy(dtype=float)) for m in m_values}

    # metrics.json / csv
    metrics = json.loads((results / "metrics.json").read_text())
    df = pd.read_csv(results / "metrics.csv")
    for row in metrics:
        model = row["model"]
        ps = pooled[model]
        curve = [ps[m][0] for m in m_values]
        finite = [c for c in curve if np.isfinite(c)]
        med_h, q_h, ltm_h = ps[headline_m]
        row["ccmr"] = med_h
        row["ccmr_q_alpha"] = q_h
        row["ccmr_ltm_alpha"] = ltm_h
        row["ccmr_auc"] = _trapz(curve, m_values) / (m_values[-1] - m_values[0]) if len(curve) > 1 else curve[0]
        row["ccmr_min"] = float(min(finite)) if finite else float("nan")
        row["ccmr_delta"] = float(curve[-1] - curve[0]) if len(curve) > 1 else 0.0
        i = df.index[df["model"] == model][0]
        for c in ("ccmr", "ccmr_q_alpha", "ccmr_ltm_alpha", "ccmr_auc", "ccmr_min", "ccmr_delta"):
            df.at[i, c] = row[c]
        # rewrite the per-sample distribution npy (finite margin values, headline m)
        finite_h = pdf.loc[pdf["model"] == model, f"ccmr_m{headline_m}"].to_numpy(dtype=float)
        finite_h = finite_h[np.isfinite(finite_h)]
        for p in {Path(row["ccmr_samples_path"]), results / "sample_distributions" / f"ccmr.{model}.npy"}:
            p = p if p.is_absolute() else ROOT / p
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, finite_h)
        print(f"{model:14s} ccmr={med_h:+.4f} q={q_h:+.3f} ltm={ltm_h:+.3f} "
              f"auc={row['ccmr_auc']:+.3f} min={row['ccmr_min']:+.3f} delta={row['ccmr_delta']:+.3f}")
    (results / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    df.to_csv(results / "metrics.csv", index=False)

    # m-sweep
    ms = json.loads((results / "ccmr_m_sweep_metrics.json").read_text())
    for r in ms:
        med, q, ltm = pooled[r["model"]][int(r["m"])]
        r["ccmr"], r["ccmr_q_alpha"], r["ccmr_ltm_alpha"] = med, q, ltm
    (results / "ccmr_m_sweep_metrics.json").write_text(json.dumps(ms, indent=1) + "\n")
    sdf = pd.read_csv(results / "ccmr_m_sweep_metrics.csv")
    for i in range(len(sdf)):
        med, q, ltm = pooled[sdf.at[i, "model"]][int(sdf.at[i, "m"])]
        sdf.at[i, "ccmr"], sdf.at[i, "ccmr_q_alpha"], sdf.at[i, "ccmr_ltm_alpha"] = med, q, ltm
    sdf.to_csv(results / "ccmr_m_sweep_metrics.csv", index=False)
    print(f"DONE {base.name} (headline m={headline_m})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
