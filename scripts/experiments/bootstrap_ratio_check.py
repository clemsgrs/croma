"""Sanity check (feedback Q&A "leaderboard noise"): does the top-3 statistical-tie
conclusion survive on the *distance ratio* d_OS/d_SO instead of the signed margin?

CCMR is reported as the signed normalized margin g = (d_OS - d_SO)/(d_OS + d_SO).
The original "ratio" framing is r = d_OS/d_SO, related by the strictly monotone map
r = (1 + g)/(1 - g). Per-sample ranking is therefore identical, but the pooled
*median* does not commute exactly with a nonlinear map on even sample counts, and a
percentile CI on r is not just g^-1 of the margin CI -- so we recompute the paired
slide-level cluster bootstrap directly on r and compare rank intervals / win probs.

Usage: python scripts/experiments/bootstrap_ratio_check.py [n_boot]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from croma.metrics.bootstrap import paired_rank_stability  # noqa: E402
from croma.metrics.ccmr import CCMR_HEADLINE_M  # noqa: E402

REL = "output/faithful/pathorob-camelyon-faithful"
SEED = 12345  # same seed as bootstrap_uncertainty.py -> identical slide resamples


def load_aligned(ps_path: Path, m: int):
    col = f"ccmr_m{m}"
    ps = pd.read_csv(ps_path, usecols=["model", "occurrence_index", "sample_index", "slide_id", col])
    models = sorted(ps["model"].unique())
    ref = None
    margins, ratios = {}, {}
    for model in models:
        sub = ps[ps["model"] == model].sort_values(["occurrence_index", "sample_index"])
        slides = sub["slide_id"].to_numpy()
        if ref is None:
            ref = slides
        elif not np.array_equal(ref, slides):
            raise RuntimeError(f"slide sequence misaligned for '{model}'")
        g = sub[col].to_numpy(dtype=float)
        margins[model] = g
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (1.0 + g) / (1.0 - g)  # = d_OS / d_SO; >1 biology-dominant
        r[~np.isfinite(g)] = np.nan
        ratios[model] = r
    return margins, ratios, ref


def main() -> int:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    m = int(CCMR_HEADLINE_M)
    ps_path = ROOT / REL / "results" / "per_sample_metrics.csv"
    margins, ratios, slides = load_aligned(ps_path, m)

    rs_m = paired_rank_stability(margins, slides, n_boot=n_boot, seed=SEED)
    rs_r = paired_rank_stability(ratios, slides, n_boot=n_boot, seed=SEED)

    order_r = sorted(rs_r.models, key=lambda x: rs_r.point_value[x], reverse=True)

    print(f"=== Camelyon distance-ratio r=d_OS/d_SO at m={m} "
          f"(n_boot={n_boot}, slides={len(np.unique(slides))}) ===")
    print(f"{'model':14s} {'ratio r':>8s} {'95% CI':>20s}  rank  [rank CI]   "
          f"margin rank [CI]")
    for model in order_r:
        ci = rs_r.value_ci[model]
        rm = rs_m  # margin reference
        print(f"{model:14s} {ci.point:8.3f} "
              f"[{ci.lo:6.3f}, {ci.hi:6.3f}]  "
              f"{rs_r.point_rank[model]:>2d}   [{rs_r.rank_lo[model]},{rs_r.rank_hi[model]}]"
              f"        {rm.point_rank[model]:>2d}   [{rm.rank_lo[model]},{rm.rank_hi[model]}]")

    # top-tie span and closest adjacent pair on the ratio
    top_tie = [x for x in order_r if rs_r.rank_lo[x] == 1]
    print(f"\ntop models whose 95% rank interval includes rank 1: "
          f"{len(top_tie)} -> {', '.join(top_tie)}")
    adj = [(order_r[i], order_r[i + 1],
            rs_r.pairwise_win[(order_r[i], order_r[i + 1])])
           for i in range(len(order_r) - 1)]
    a, b, p = min(adj, key=lambda t: abs(t[2] - 0.5))
    print(f"closest adjacent pair (ratio):  {a} > {b}   P={p:.3f}")
    # same pair on the margin, for direct comparison
    print(f"   same pair on margin:         {a} > {b}   "
          f"P={rs_m.pairwise_win[(a, b)]:.3f}")

    # do the two orderings agree, and is the sign-equivalent boundary (r=1) preserved?
    order_m = sorted(rs_m.models, key=lambda x: rs_m.point_value[x], reverse=True)
    print(f"\nrank order identical to margin? {order_r == order_m}")
    biology_dom = sum(1 for x in order_r if rs_r.point_value[x] > 1.0)
    biology_dom_g = sum(1 for x in order_m if rs_m.point_value[x] > 0.0)
    print(f"models with r>1 (ratio): {biology_dom}   |  models with g>0 (margin): {biology_dom_g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
