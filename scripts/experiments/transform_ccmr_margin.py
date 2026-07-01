"""Fast, exact CCMR ratio -> margin transform for a benchmark output dir.

Pooled CCMR is a median and Q_alpha a percentile, both of which commute with the
monotone map ``margin = (r-1)/(r+1)``; so they transform exactly without any
neighbour search. LTM (mean of the lower tail) does not commute, so it is
recomputed from the per-sample distribution (transformed elementwise). AUC/min/
delta are recomputed from the transformed m-sweep curve.

Usage: python scripts/experiments/transform_ccmr_margin.py <output_dir> [alpha]
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _m(r):
    return (r - 1.0) / (r + 1.0)


def _trapz(y, x):
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def _ltm_of(margin: np.ndarray, alpha: float) -> float:
    fin = margin[np.isfinite(margin)]
    if fin.size == 0:
        return float("nan")
    q = float(np.percentile(fin, alpha * 100))
    tail = fin[fin <= q]
    return float(tail.mean()) if tail.size else q


def main() -> int:
    base = Path(sys.argv[1])
    if not base.is_absolute():
        base = ROOT / base
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    results = base / "results"

    metrics = json.loads((results / "metrics.json").read_text())
    if all(-1.0 < float(r.get("ccmr", 2)) < 1.0 for r in metrics):
        print(f"SKIP {base.name}: ccmr already in (-1,1) — looks transformed already")
        return 0

    # transform the per-sample distributions on disk (once) and recompute LTM
    ltm_by_model: dict[str, float] = {}
    for row in metrics:
        paths = {Path(row["ccmr_samples_path"]), results / "sample_distributions" / f"ccmr.{row['model']}.npy"}
        paths = {p if p.is_absolute() else ROOT / p for p in paths}
        src = next((p for p in paths if p.exists()), None)
        if src is None:
            ltm_by_model[row["model"]] = float("nan")
            continue
        margin = _m(np.load(src))
        for p in paths:
            np.save(p, margin)
        ltm_by_model[row["model"]] = _ltm_of(margin, alpha)

    # m-sweep: transform pooled ccmr + q exactly; recompute per-model curve for auc/min/delta
    m_sweep = json.loads((results / "ccmr_m_sweep_metrics.json").read_text())
    curve: dict[str, dict[int, float]] = {}
    for r in m_sweep:
        r["ccmr"] = _m(float(r["ccmr"]))
        if "ccmr_q_alpha" in r and r["ccmr_q_alpha"] is not None:
            r["ccmr_q_alpha"] = _m(float(r["ccmr_q_alpha"]))
        curve.setdefault(r["model"], {})[int(r["m"])] = float(r["ccmr"])
    (results / "ccmr_m_sweep_metrics.json").write_text(json.dumps(m_sweep, indent=1) + "\n")
    sdf = pd.read_csv(results / "ccmr_m_sweep_metrics.csv")
    sdf["ccmr"] = _m(sdf["ccmr"].astype(float))
    if "ccmr_q_alpha" in sdf:
        sdf["ccmr_q_alpha"] = _m(sdf["ccmr_q_alpha"].astype(float))
    sdf.to_csv(results / "ccmr_m_sweep_metrics.csv", index=False)

    # metrics.json / metrics.csv
    df = pd.read_csv(results / "metrics.csv")
    for row in metrics:
        model = row["model"]
        ms = curve.get(model, {})
        m_vals = sorted(ms)
        cvals = [ms[m] for m in m_vals]
        auc = _trapz(cvals, m_vals) / (m_vals[-1] - m_vals[0]) if len(cvals) > 1 else (cvals[0] if cvals else float("nan"))
        finite = [c for c in cvals if np.isfinite(c)]
        row["ccmr"] = _m(float(row["ccmr"]))
        row["ccmr_q_alpha"] = _m(float(row["ccmr_q_alpha"]))
        row["ccmr_ltm_alpha"] = ltm_by_model[model]
        row["ccmr_auc"] = auc
        row["ccmr_min"] = float(min(finite)) if finite else float("nan")
        row["ccmr_delta"] = float(cvals[-1] - cvals[0]) if len(cvals) > 1 else 0.0
        i = df.index[df["model"] == model][0]
        for c in ("ccmr", "ccmr_q_alpha", "ccmr_ltm_alpha", "ccmr_auc", "ccmr_min", "ccmr_delta"):
            df.at[i, c] = row[c]
        print(f"{model:14s} ccmr={row['ccmr']:+.4f} q={row['ccmr_q_alpha']:+.3f} ltm={row['ccmr_ltm_alpha']:+.3f} "
              f"auc={row['ccmr_auc']:+.3f} min={row['ccmr_min']:+.3f} delta={row['ccmr_delta']:+.3f}")
    (results / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    df.to_csv(results / "metrics.csv", index=False)

    # transform per_sample_metrics columns if present (consumed by analyze_results)
    ps = results / "per_sample_metrics.csv"
    if ps.exists():
        pdf = pd.read_csv(ps)
        ccmr_cols = [c for c in pdf.columns if c.startswith("ccmr_m") and c[6:].isdigit()]
        # ratio is always >= 0; presence of negatives means already transformed.
        already = any((pdf[c].dropna() < 0).any() for c in ccmr_cols)
        if already:
            print("per_sample ccmr_m* already transformed — skipping")
        else:
            for c in ccmr_cols:
                pdf[c] = _m(pdf[c].astype(float))
            pdf.to_csv(ps, index=False)
            print(f"transformed {len(ccmr_cols)} per_sample ccmr_m* columns")

    print(f"DONE {base.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
