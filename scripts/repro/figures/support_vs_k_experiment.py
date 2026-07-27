"""Support-vs-k curve: defined (non-undefined) fraction of RI/MaRI as a function of k.

The single-k* support bar chart (`ri_mari_support.pdf`) confounds two things: a model
can look "low support" either because it genuinely entangles biology and confounder
(few SO/OS neighbours ever appear) or simply because it was assigned a small k*
(little neighbourhood scanned). Support and k* co-vary, so the bar chart cannot tell
them apart.

This script turns that single number into an honest 2D picture: for every model it
sweeps k and reports the fraction of samples whose RI/MaRI is *defined* (at least one
SO or OS neighbour within the top-k), then marks each model's k* on its own curve. A
model whose curve stays low even at large k is truly entangled; a model whose curve is
already high but was clipped to a small k* just got an unlucky k*.

Definition (identical to the production metric): a sample is defined at k iff, among its
first k cross-slide neighbours (self and same-slide excluded), at least one is SO (same
biology, other confounder) or OS (other biology, same confounder). RI and MaRI share
this exact support set -- the distance weighting changes the score value, not whether
the SO/OS denominator is positive -- so this is a single curve per model. The undefined
fraction it produces is verified to reproduce `metrics.csv` (e.g. Virchow2 at k=3:
undefined 0.79225) because it calls the same core neighbour helpers the benchmark uses.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from _neighbor_analysis import REPO, load_meta, prepare_embedding  # noqa: E402
import views  # noqa: E402

from croma import RI
from croma.metrics.neighbors import _prepare_neighbors

PROTOCOL = "k-star"
view = views.load_view("pathorob-camelyon")  # row-view over the pathorob-camelyon tileset
METRICS = view.results_dir(PROTOCOL) / "metrics.csv"
STUDIES = view.studies_dir(PROTOCOL)
# Beside the run this study reads, never in the manuscript tree (see check_paper_figures.py).
# Derived from STUDIES, so it follows PROTOCOL instead of restating it.
FIGDIR = STUDIES / "plots" / "pdf"

K_GRID = list(range(1, 101))  # spans every model's k* (faithful k* range 5..61)

df = view.eval_manifest
labels, centers, slide = load_meta(df)
n = len(df)

kstar = (
    pd.read_csv(METRICS).set_index("model")["k"].astype(int).to_dict()
)  # per-model k* selected by the benchmark

models = view.models
print(f"{n} samples, {len(models)} models\n")

kmax = max(K_GRID)
curves: dict[str, np.ndarray] = {}
for model in models:
    X = prepare_embedding(view.features(model))
    nidx, ndist, vc = _prepare_neighbors(X, slide, kmax)
    scored = RI._score_all_k_from_neighbors(labels, centers, nidx, ndist, vc, K_GRID)
    # scored[k] = (pooled, sample_scores, informative_mask, undefined_type, so, os)
    defined = np.array([float(scored[k][2].mean()) for k in K_GRID])
    curves[model] = defined

# ---- tidy summary (long format) + per-model JSON ----
rows = []
summary_json: dict[str, dict] = {}
for model in models:
    ks = kstar.get(model)
    defined = curves[model]
    frac_at_kstar = float(defined[K_GRID.index(int(ks))]) if ks in K_GRID else float("nan")
    for k, frac in zip(K_GRID, defined):
        rows.append(
            {
                "model": model,
                "k": int(k),
                "defined_frac": float(frac),
                "is_kstar": bool(ks is not None and int(ks) == int(k)),
            }
        )
    summary_json[model] = {
        "kstar": int(ks) if ks is not None else None,
        "defined_frac_at_kstar": frac_at_kstar,
        "defined_frac_curve": {str(int(k)): float(f) for k, f in zip(K_GRID, defined)},
    }

summary = pd.DataFrame(rows)
STUDIES.mkdir(parents=True, exist_ok=True)
out_csv = STUDIES / "support_vs_k_summary.csv"
summary.to_csv(out_csv, index=False)
json.dump(summary_json, open(STUDIES / "support_vs_k_summary.json", "w"), indent=1)
print(f"wrote {out_csv}")

# quick console view: defined fraction at k* vs at k=50 (the entanglement gap)
view = pd.DataFrame(
    {
        "model": models,
        "kstar": [kstar.get(m) for m in models],
        "defined@kstar": [summary_json[m]["defined_frac_at_kstar"] for m in models],
        "defined@50": [float(curves[m][-1]) for m in models],
    }
).sort_values("defined@50")
pd.set_option("display.width", 200)
print(view.round(3).to_string(index=False))

# ---- figure: one line per model, k* marked ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from croma import plotstyle
from croma.plotstyle import color_for_model, model_sort_key

order = sorted(models, key=lambda m: (model_sort_key(m), m))

fig, ax = plt.subplots(figsize=(plotstyle.COL_DOUBLE, 4.6))
plotstyle.style_axes(ax)
for model in order:
    c = color_for_model(model)
    ax.plot(K_GRID, curves[model], color=c, lw=plotstyle.LW_SERIES, alpha=0.95, label=model)
    ks = kstar.get(model)
    if ks in K_GRID:
        ax.plot(
            ks,
            curves[model][K_GRID.index(int(ks))],
            marker="*",
            ms=9,
            color=c,
            mec="white",
            mew=0.7,
            zorder=5,
        )
ax.set_xlabel(r"neighbourhood size $k$")
ax.set_ylabel("Defined fraction (RI/MaRI support)")
ax.set_ylim(0, 1)
ax.set_xlim(1, max(K_GRID))
plotstyle.title_with_subtitle(
    ax,
    r"RI/MaRI support vs $k$  ($\bigstar$ = each model's $k^\star$)",
    r"low-at-large-$k$ = entangled; high-but-clipped = small $k^\star$",
)
ax.legend(
    fontsize=plotstyle.FS_ANNOT, ncol=6, loc="upper center",
    bbox_to_anchor=(0.5, -0.16), frameon=False, columnspacing=1.1, handlelength=1.7,
)
figdir = FIGDIR
figdir.mkdir(parents=True, exist_ok=True)
(figdir.parent / "png").mkdir(parents=True, exist_ok=True)
figpath = figdir / "support_vs_k.pdf"
fig.savefig(figpath, bbox_inches="tight")
fig.savefig(figdir.parent / "png" / "support_vs_k.png", dpi=plotstyle.DEFAULT_DPI, bbox_inches="tight")
print(f"wrote {figpath}")
