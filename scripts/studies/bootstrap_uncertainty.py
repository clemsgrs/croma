"""Uncertainty quantification for the headline CRoMa (closes feedback point 3) and
the CRoMa-vs-RI/MaRI redundancy question (point 6).

For each benchmark we read the cached per-sample CRoMa (no neighbour recompute) and:

  1. a slide-level *cluster* bootstrap CI on the pooled-median headline CRoMa per
     model -- tiles within a slide are correlated (pure slides, one scanner), so an
     i.i.d. resample would understate uncertainty;
  2. a *paired* rank-stability bootstrap (one shared slide resample applied to all
     models) giving each model's rank interval and the pairwise win probability
     P(CRoMa_A > CRoMa_B) -- this is the honest answer to "is model A's lead real?";
  3. Spearman(CRoMa, RI) and Spearman(CRoMa, MaRI) across models with a bootstrap CI
     over models -- the redundancy evidence for metric_complementarity.tex.

Outputs, per benchmark, under <dir>/results/:
  - bootstrap_uncertainty.csv   (per-model: croma + CI, point/mean rank + CI)
  - bootstrap_uncertainty.json  (correlations, adjacent-pair win probs, meta)

Usage:
  python scripts/studies/bootstrap_uncertainty.py [n_boot] [bench1 bench2 ...]
  (defaults: n_boot=2000, all benchmarks)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from croma.metrics.bootstrap import (  # noqa: E402
    bootstrap_spearman,
    paired_rank_stability,
)
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402

# name -> benchmark output directory (relative to repo root)
BENCHMARKS: dict[str, str] = {
    "camelyon": "output/faithful/k-star/pathorob-camelyon-faithful",
    "tcga2x2": "output/faithful/k-star/pathorob-tcga-2x2",
    "tcga4x4": "output/faithful/k-star/pathorob-tcga-4x4",
    "tolkach": "output/faithful/k-star/pathorob-tolkach-esca-faithful",
    "panda": "output/panda-wsi-isup-paired-2x2",
}

SEED = 12345
MIN_MODELS_FOR_CORR = 8  # Spearman over models is meaningless for tiny suites


def _load_aligned(ps_path: Path, m: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-model CRoMa(m) arrays aligned to one shared slide vector.

    Rows are sorted by (occurrence_index, sample_index); every model shares this
    order and the same slide sequence (asserted), so a single slide vector clusters
    all models for the paired bootstrap.
    """
    cols = ["model", "occurrence_index", "sample_index", "slide_id", f"croma_m{m}"]
    ps = pd.read_csv(ps_path, usecols=cols)
    models = sorted(ps["model"].unique())
    ref_slides: np.ndarray | None = None
    model_values: dict[str, np.ndarray] = {}
    for model in models:
        sub = ps[ps["model"] == model].sort_values(["occurrence_index", "sample_index"])
        slides = sub["slide_id"].to_numpy()
        if ref_slides is None:
            ref_slides = slides
        elif not np.array_equal(ref_slides, slides):
            raise RuntimeError(f"slide sequence misaligned for model '{model}'")
        model_values[model] = sub[f"croma_m{m}"].to_numpy(dtype=float)
    assert ref_slides is not None
    return model_values, ref_slides


def _correlations(metrics_csv: Path, n_boot: int) -> dict:
    df = pd.read_csv(metrics_csv).set_index("model")
    n_models = len(df)
    if n_models < MIN_MODELS_FOR_CORR or not {"ri", "mari", "croma"} <= set(df.columns):
        return {"n_models": int(n_models), "skipped": True}
    out: dict = {"n_models": int(n_models), "skipped": False}
    for other in ("ri", "mari"):
        ci = bootstrap_spearman(
            df["croma"].to_numpy(float),
            df[other].to_numpy(float),
            n_boot=n_boot,
            seed=SEED,
        )
        out[f"croma_vs_{other}"] = {
            "rho": ci.point,
            "lo": ci.lo,
            "hi": ci.hi,
        }
    ci_rm = bootstrap_spearman(
        df["ri"].to_numpy(float), df["mari"].to_numpy(float), n_boot=n_boot, seed=SEED
    )
    out["ri_vs_mari"] = {"rho": ci_rm.point, "lo": ci_rm.lo, "hi": ci_rm.hi}
    return out


def run_benchmark(name: str, rel_dir: str, n_boot: int) -> None:
    base = ROOT / rel_dir
    results = base / "results"
    ps_path = results / "per_sample_metrics.csv"
    metrics_csv = results / "metrics.csv"
    if not ps_path.exists():
        print(f"[skip] {name}: {ps_path} not found")
        return

    m = int(CROMA_HEADLINE_M)
    model_values, slides = _load_aligned(ps_path, m)
    rs = paired_rank_stability(model_values, slides, n_boot=n_boot, seed=SEED)

    order = sorted(rs.models, key=lambda mm: rs.point_value[mm], reverse=True)
    rows = []
    for model in order:
        ci = rs.value_ci[model]
        rows.append(
            {
                "model": model,
                "croma": ci.point,
                "croma_lo": ci.lo,
                "croma_hi": ci.hi,
                "point_rank": rs.point_rank[model],
                "mean_rank": round(rs.mean_rank[model], 2),
                "rank_lo": rs.rank_lo[model],
                "rank_hi": rs.rank_hi[model],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(results / "bootstrap_uncertainty.csv", index=False)

    # adjacent-pair win probabilities along the headline ordering
    adjacent = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        adjacent.append(
            {
                "higher": a,
                "lower": b,
                "p_higher_beats_lower": round(rs.pairwise_win[(a, b)], 3),
            }
        )

    corr = _correlations(metrics_csv, n_boot) if metrics_csv.exists() else {"skipped": True}

    summary = {
        "benchmark": name,
        "headline_m": m,
        "n_boot": n_boot,
        "n_models": len(rs.models),
        "n_slides": int(len(np.unique(slides))),
        "n_samples_per_model": int(len(slides)),
        "level": rs.value_ci[order[0]].level,
        "correlations": corr,
        "adjacent_pair_win": adjacent,
    }
    (results / "bootstrap_uncertainty.json").write_text(json.dumps(summary, indent=2) + "\n")

    # console report
    print(f"\n=== {name}  (m={m}, n_boot={n_boot}, slides={summary['n_slides']}) ===")
    for r in rows:
        print(
            f"  {r['model']:14s} CRoMa={r['croma']:+.3f} "
            f"[{r['croma_lo']:+.3f}, {r['croma_hi']:+.3f}]  "
            f"rank {r['point_rank']:>2d} (mean {r['mean_rank']:.2f}, "
            f"[{r['rank_lo']},{r['rank_hi']}])"
        )
    if adjacent:
        tightest = min(adjacent, key=lambda d: abs(d["p_higher_beats_lower"] - 0.5))
        print(
            f"  closest adjacent pair: {tightest['higher']} > {tightest['lower']} "
            f"with P={tightest['p_higher_beats_lower']:.3f}"
        )
    if not corr.get("skipped"):
        for key in ("croma_vs_ri", "croma_vs_mari", "ri_vs_mari"):
            c = corr[key]
            print(f"  Spearman {key:14s} rho={c['rho']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
    else:
        print(f"  correlations skipped (n_models={corr.get('n_models', '?')})")


def main() -> int:
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    chosen = sys.argv[2:] if len(sys.argv) > 2 else list(BENCHMARKS)
    for name in chosen:
        if name not in BENCHMARKS:
            print(f"[skip] unknown benchmark '{name}'")
            continue
        run_benchmark(name, BENCHMARKS[name], n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
