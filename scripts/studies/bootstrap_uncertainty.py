"""Uncertainty quantification for the headline CRoMa (closes feedback point 3) and
the CRoMa-vs-RI/MaRI redundancy question (point 6).

For each benchmark we read the cached per-sample CRoMa (no neighbour recompute) and:

  1. a *cluster* bootstrap CI on the pooled-median headline CRoMa per model, clustered
     on the manifest's ``group_id`` -- samples in one independence group are correlated
     (for these cohorts, one slide: pure tissue, one scanner), so an i.i.d. resample
     would understate uncertainty;
  2. a *paired* rank-stability bootstrap (one shared group resample applied to all
     models) giving each model's rank interval and the pairwise win probability
     P(CRoMa_A > CRoMa_B) -- this is the honest answer to "is model A's lead real?";
  3. Spearman(CRoMa, RI) and Spearman(CRoMa, MaRI) across models with a bootstrap CI
     over models -- the redundancy evidence for metric_complementarity.tex.

Outputs, per benchmark, under <dir>/results/:
  - bootstrap_uncertainty.csv   (per-model: croma + CI, point/mean rank + CI)
  - bootstrap_uncertainty.json  (correlations, adjacent-pair win probs, meta)

Note the correlations are protocol-dependent while the CIs are not: CRoMa is k-free, so its
per-sample values are bit-identical across protocols, but RI and MaRI move with k (by 0.13 on
PANDA). Spearman(CRoMa, RI) must therefore be computed at the protocol whose RI the paper
prints, which is why the run directory comes from ``paper_manifest`` rather than a constant.

Usage:
  python scripts/studies/bootstrap_uncertainty.py [n_boot] [Prefix1 Prefix2 ...]
  (defaults: n_boot=2000, every benchmark in the manifest)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# ``croma`` lives under src/ and is not installed; ROOT alone never resolved it.
for _p in (ROOT / "src", ROOT / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from croma.metrics.bootstrap import (  # noqa: E402
    bootstrap_spearman,
    paired_rank_stability,
)
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting.style import CONTROL_MODEL  # noqa: E402
from paper_manifest import TABLES  # noqa: E402

#: macro prefix -> run directory, at each benchmark's reported protocol. See ADR-0010.
BENCHMARKS: dict[str, str] = {t.prefix: t.run_rel for t in TABLES}

SEED = 12345
MIN_MODELS_FOR_CORR = 8  # Spearman over models is meaningless for tiny suites


def _load_aligned(ps_path: Path, m: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-model CRoMa(m) arrays aligned to one shared independence-group vector.

    Rows are sorted by (occurrence_index, sample_index); every model shares this
    order and the same ``group_id`` sequence (asserted), so a single group vector
    clusters all models for the paired bootstrap.
    """
    cols = ["model", "occurrence_index", "sample_index", "group_id", f"croma_m{m}"]
    ps = pd.read_csv(ps_path, usecols=cols)
    # Both outputs of this study rank models against each other, so both are computed on the
    # ranked panel: the natural-image control is a floor, not a competitor (see CONTEXT.md).
    # Leaving it in would give it a rank interval among the pathology encoders and would pull
    # every Spearman, since it sits off-trend on support and biological accuracy alike.
    models = sorted(m_ for m_ in ps["model"].unique() if m_ != CONTROL_MODEL)
    ref_groups: np.ndarray | None = None
    model_values: dict[str, np.ndarray] = {}
    for model in models:
        sub = ps[ps["model"] == model].sort_values(["occurrence_index", "sample_index"])
        groups = sub["group_id"].to_numpy()
        if ref_groups is None:
            ref_groups = groups
        elif not np.array_equal(ref_groups, groups):
            raise RuntimeError(f"group_id sequence misaligned for model '{model}'")
        model_values[model] = sub[f"croma_m{m}"].to_numpy(dtype=float)
    assert ref_groups is not None
    return model_values, ref_groups


def _correlations(metrics_csv: Path, n_boot: int) -> dict:
    df = pd.read_csv(metrics_csv).set_index("model")
    df = df.drop(index=CONTROL_MODEL, errors="ignore")  # ranked panel only
    n_models = len(df)
    # The artifact carries the floor it was computed under, so the table generator can
    # explain a blank row without keeping its own copy of the threshold.
    base = {"n_models": int(n_models), "min_models": MIN_MODELS_FOR_CORR}
    if n_models < MIN_MODELS_FOR_CORR or not {"ri", "mari", "croma"} <= set(df.columns):
        return {**base, "skipped": True}
    out: dict = {**base, "skipped": False}
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
    model_values, groups = _load_aligned(ps_path, m)
    rs = paired_rank_stability(model_values, groups, n_boot=n_boot, seed=SEED)

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
        "n_groups": int(len(np.unique(groups))),
        "n_samples_per_model": int(len(groups)),
        "level": rs.value_ci[order[0]].level,
        "correlations": corr,
        "adjacent_pair_win": adjacent,
    }
    (results / "bootstrap_uncertainty.json").write_text(json.dumps(summary, indent=2) + "\n")

    # console report
    print(f"\n=== {name}  (m={m}, n_boot={n_boot}, groups={summary['n_groups']}) ===")
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
