"""Regenerate CCMR artifacts after switching the metric from distance ratio to
the signed normalized margin ``(d_OS - d_SO) / (d_OS + d_SO)``.

CCMR is k-free, so its values are identical across k-selection variants; this
script recomputes them with the (edited) real metric on cached embeddings and
rewrites only the ``ccmr_*`` fields/artifacts, leaving RI/MaRI untouched. The
paper's figures are unified on the faithful run, so the default also re-renders
the CCMR figures into ``paper/figures`` from faithful.

Usage:
    python scripts/experiments/regen_ccmr_margin.py            # faithful (+figures)
    python scripts/experiments/regen_ccmr_margin.py reduced    # reduced data values only
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from croma.metrics.ccmr import CCMR_HEADLINE_M  # noqa: E402
from croma.metrics.ccmr import CrossConfounderMarginRatio as CCMR  # noqa: E402
import plotting  # noqa: E402

M_MAX = 20
ALPHA = 0.10
START_K = 200
GROWTH = 2.0

FIGURES = [
    "ccmr_m_sweep",
    "ccmr_sample_distributions",
    "ccmr_ltm_scatter",
    "ccmr_ltm_bars",
    "ccmr_vs_mari_scatter",
    "q_alpha_vs_ccmr_scatter",
]


def _trapz(y, x):
    fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(fn(y, x))


def regen(base: Path, *, paper_fig: Path | None = None) -> None:
    results = base / "results"
    manifest = pd.read_csv(base / "embedding_source_manifest.csv")
    metrics = json.loads((results / "metrics.json").read_text())
    m_sweep = json.loads((results / "ccmr_m_sweep_metrics.json").read_text())
    m_values = list(range(1, M_MAX + 1))
    per_model: dict[str, dict] = {}

    for row in metrics:
        model = row["model"]
        emb = np.load(base / "embeddings" / f"{model}.npy")
        by_m = CCMR.compute(
            emb, manifest, confounder_column="confounder",
            evaluation_design="dataset_wide", m=m_values,
            alpha=ALPHA, start_k=START_K, k_growth_factor=GROWTH,
        )
        curve = [float(by_m[m].value) for m in m_values]
        finite = [c for c in curve if np.isfinite(c)]
        auc = _trapz(curve, m_values) / (m_values[-1] - m_values[0]) if len(curve) > 1 else curve[0]
        per_model[model] = {
            "by_m": by_m, "auc": auc,
            "min": float(min(finite)) if finite else float("nan"),
            "delta": float(curve[-1] - curve[0]) if len(curve) > 1 else 0.0,
        }
        res_h = by_m[CCMR_HEADLINE_M]
        for t in {Path(row["ccmr_samples_path"]), results / "sample_distributions" / f"ccmr.{model}.npy"}:
            t = t if t.is_absolute() else ROOT / t
            t.parent.mkdir(parents=True, exist_ok=True)
            np.save(t, np.asarray(res_h.sample_values_aligned, dtype=float))
        print(f"{model:14s} ccmr={res_h.value:+.4f} q={res_h.q_alpha:+.3f} ltm={res_h.ltm_alpha:+.3f} "
              f"auc={per_model[model]['auc']:+.3f} min={per_model[model]['min']:+.3f} delta={per_model[model]['delta']:+.3f}")

    for row in metrics:
        pm = per_model[row["model"]]
        r_h = pm["by_m"][CCMR_HEADLINE_M]
        row.update({
            "ccmr": float(r_h.value), "ccmr_std": float(r_h.std),
            "ccmr_undefined_frac": float(r_h.undefined_frac),
            "ccmr_k_start": int(r_h.k_start), "ccmr_k_final": int(r_h.k_final),
            "ccmr_retries": int(r_h.retries), "ccmr_q_alpha": float(r_h.q_alpha),
            "ccmr_ltm_alpha": float(r_h.ltm_alpha), "ccmr_auc": float(pm["auc"]),
            "ccmr_min": float(pm["min"]), "ccmr_delta": float(pm["delta"]),
        })
    (results / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")

    df = pd.read_csv(results / "metrics.csv")
    bym = {r["model"]: r for r in metrics}
    ccmr_cols = ["ccmr", "ccmr_std", "ccmr_undefined_frac", "ccmr_k_start", "ccmr_k_final",
                 "ccmr_retries", "ccmr_q_alpha", "ccmr_ltm_alpha", "ccmr_auc", "ccmr_min", "ccmr_delta"]
    for i, model in enumerate(df["model"]):
        for c in ccmr_cols:
            df.at[i, c] = bym[model][c]
    df.to_csv(results / "metrics.csv", index=False)

    for row in m_sweep:
        res = per_model[row["model"]]["by_m"][int(row["m"])]
        row.update({
            "ccmr": float(res.value), "ccmr_std": float(res.std),
            "ccmr_undefined_frac": float(res.undefined_frac),
            "ccmr_q_alpha": float(res.q_alpha), "ccmr_ltm_alpha": float(res.ltm_alpha),
            "ccmr_k_start": int(res.k_start), "ccmr_k_final": int(res.k_final),
            "ccmr_retries": int(res.retries),
        })
    (results / "ccmr_m_sweep_metrics.json").write_text(json.dumps(m_sweep, indent=1) + "\n")
    sdf = pd.read_csv(results / "ccmr_m_sweep_metrics.csv")
    skey = {(r["model"], int(r["m"])): r for r in m_sweep}
    for i in range(len(sdf)):
        k = (sdf.at[i, "model"], int(sdf.at[i, "m"]))
        for c in ["ccmr", "ccmr_std", "ccmr_undefined_frac", "ccmr_q_alpha", "ccmr_ltm_alpha",
                  "ccmr_k_start", "ccmr_k_final", "ccmr_retries"]:
            sdf.at[i, c] = skey[k][c]
    sdf.to_csv(results / "ccmr_m_sweep_metrics.csv", index=False)

    if paper_fig is not None:
        plots = results / "plots_regen"
        plots.mkdir(parents=True, exist_ok=True)
        plotting.plot_ccmr_m_sweep(rows=m_sweep, out_path=plots / "ccmr_m_sweep.png")
        plotting.plot_ccmr_sample_distributions(rows=metrics, out_path=plots / "ccmr_sample_distributions.png")
        plotting.plot_ccmr_ltm_scatter(rows=metrics, out_path=plots / "ccmr_ltm_scatter.png")
        plotting.plot_ccmr_ltm_bars(rows=metrics, out_path=plots / "ccmr_ltm_bars.png")
        plotting.plot_ccmr_vs_mari_scatter(rows=metrics, out_path=plots / "ccmr_vs_mari_scatter.png")
        plotting.plot_q_alpha_vs_ccmr_scatter(rows=metrics, out_path=plots / "q_alpha_vs_ccmr_scatter.png")
        for name in FIGURES:
            for sub, ext in (("pdf", "pdf"), ("png", "png")):
                src = plots / sub / f"{name}.{ext}" if sub == "pdf" else plots / f"{name}.{ext}"
                src_png = plots / "png" / f"{name}.png"
                src = (plots / "pdf" / f"{name}.pdf") if sub == "pdf" else src_png
                dst = paper_fig / sub / f"{name}.{ext}"
                if src.exists():
                    shutil.copyfile(src, dst)
        print(f"figures -> {paper_fig}")


if __name__ == "__main__":
    # The paper's figures are unified on the faithful run, so figure rendering is
    # tied to faithful. The reduced-kstar branch only recomputes the reduced data
    # values (consumed by render_one.py for style review) and must NOT write into
    # the paper figure directory, or it would re-introduce the n=4,000 vs n=20,400
    # figure/table mismatch.
    if len(sys.argv) > 1 and sys.argv[1] == "reduced":
        regen(ROOT / "output/pathorob-camelyon-reduced-kstar")
    else:
        regen(ROOT / "output/faithful/pathorob-camelyon-faithful",
              paper_fig=ROOT / "paper/figures/results/pathorob-camelyon-faithful")
