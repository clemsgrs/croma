"""Top-k neighbourhood composition: does OO dominate the immediate neighbourhood?

Objection to CRoMa: by reading only SO and OS typed neighbours it "ignores" the SS
and OO neighbours. The SS half of that objection is already answered (SS dominates the
immediate neighbourhood, which is exactly why typed neighbours are reached deep; see
typed_neighbor_rank_experiment.py). This script answers the OO half: it measures, for
every sample and model, the fraction of the top-k neighbours that are OO (other biology,
other confounder), pooled across samples and models.

Type definitions (consistent with the manuscript and the rank experiment):
  SS = same label,  same confounder      OO = other label, other confounder
  SO = same label,  other confounder     OS = other label, same confounder
Self and same-slide neighbours are excluded (as in CRoMa), which is the conservative
choice: it removes guaranteed-SS same-slide tiles and so gives OO its best chance to
look large.

If OO stays a small minority across the immediate neighbourhood, the "CRoMa ignores
informative OO neighbours" objection is closed with evidence: there is little OO signal
to ignore near each sample.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from _neighbor_analysis import load_meta, prepare_embedding  # noqa: E402
import views  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # lock a headless backend before plotstyle/pyplot is imported
from croma import plotstyle  # noqa: E402

# The natural-image control (DINOv2-B) is not a pathology encoder; exclude it so the pooled
# composition matches the manuscript's 20-pathology-model roster (see typed_neighbor_rank).
CONTROL_MODEL = plotstyle.CONTROL_MODEL

PROTOCOL = "k-star"
view = views.load_view("pathorob-camelyon")  # row-view over the pathorob-camelyon tileset
STUDIES = view.studies_dir(PROTOCOL)
# Beside the run this study reads, never in the manuscript tree (see check_paper_figures.py).
# Derived from STUDIES, so it follows PROTOCOL instead of restating it.
FIGDIR = STUDIES / "plots" / "pdf"

K_GRID = [3, 5, 10, 20, 30, 50, 75, 100, 150, 200]
COL_CAP = 400  # leading sorted columns scanned to collect up to max(K_GRID) valid ones

df = view.eval_manifest
labels, conf, slide = load_meta(df, compact=True)
n = len(df)

models = [m for m in view.models if m != CONTROL_MODEL]  # 20 pathology encoders (drop control)
print(f"{n} samples, {len(models)} pathology models (control {CONTROL_MODEL} excluded)\n")

self_idx = np.arange(n)
# pooled per-sample type fractions at each k, stacked over all models
pooled = {t: {k: [] for k in K_GRID} for t in ("SS", "SO", "OS", "OO")}

for model in models:
    X = prepare_embedding(view.features(model), np.float32)
    D = 1.0 - X @ X.T
    order = np.argsort(D, axis=1, kind="stable")[:, :COL_CAP]  # (n, COL_CAP)

    L = labels[order]
    C = conf[order]
    S = slide[order]
    di = labels[:, None]
    ci = conf[:, None]
    si = slide[:, None]

    valid = (order != self_idx[:, None]) & (S != si)
    same_lab = L == di
    same_conf = C == ci
    types = {
        "SS": valid & same_lab & same_conf,
        "SO": valid & same_lab & ~same_conf,
        "OS": valid & ~same_lab & same_conf,
        "OO": valid & ~same_lab & ~same_conf,
    }

    cum_valid = np.cumsum(valid, axis=1)  # # valid neighbours within first j+1 columns
    cum_type = {t: np.cumsum(mask, axis=1) for t, mask in types.items()}

    for k in K_GRID:
        # first column at which the k-th valid neighbour appears, per sample
        reached = cum_valid >= k
        has_k = reached.any(axis=1)
        jstar = np.argmax(reached, axis=1)  # first True; 0 for rows without k valid
        rows = self_idx[has_k]
        cols = jstar[has_k]
        for t in types:
            frac = cum_type[t][rows, cols] / float(k)
            pooled[t][k].append(frac)

# aggregate pooled fractions across all samples and models
summary_rows = []
for k in K_GRID:
    row = {"k": k}
    for t in ("SS", "SO", "OS", "OO"):
        a = np.concatenate(pooled[t][k])
        row[f"{t}_mean"] = float(a.mean())
        row[f"{t}_q25"] = float(np.percentile(a, 25))
        row[f"{t}_q75"] = float(np.percentile(a, 75))
        row[f"{t}_q90"] = float(np.percentile(a, 90))
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
pd.set_option("display.width", 200, "display.max_columns", 40)
print(summary.round(3).to_string(index=False))

oo_max_mean = float(summary["OO_mean"].max())
oo_max_q90 = float(summary["OO_q90"].max())
print(f"\nOO fraction: max mean over k = {oo_max_mean:.3f}, max q90 over k = {oo_max_q90:.3f}")
print(f"SS fraction: min mean over k = {float(summary['SS_mean'].min()):.3f}")

STUDIES.mkdir(parents=True, exist_ok=True)
out_csv = STUDIES / "oo_fraction_summary.csv"
summary.to_csv(out_csv, index=False)
json.dump(
    {
        "oo_mean_by_k": {str(r["k"]): r["OO_mean"] for r in summary_rows},
        "oo_q90_by_k": {str(r["k"]): r["OO_q90"] for r in summary_rows},
        "oo_max_mean": oo_max_mean,
        "oo_max_q90": oo_max_q90,
        "ss_min_mean": float(summary["SS_mean"].min()),
    },
    open(STUDIES / "oo_fraction_summary.json", "w"),
    indent=1,
)
print(f"wrote {out_csv}")

# ---- figure: top-k composition by type, OO never dominates ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from croma import plotstyle

# Semantic neighbour-type colours from the shared palette.
COLORS = plotstyle.NEIGHBOR_TYPE_COLOR
LABELS = {
    "SS": "SS (same bio, same conf.)",
    "SO": "SO (same bio, other conf.)",
    "OS": "OS (other bio, same conf.)",
    "OO": "OO (other bio, other conf.)",
}

ks = summary["k"].to_numpy()
fig, ax = plt.subplots(figsize=(plotstyle.COL_ONEHALF, 4.0))
plotstyle.style_axes(ax)
for t in ("SS", "SO", "OS", "OO"):
    ax.plot(
        ks, summary[f"{t}_mean"], color=COLORS[t], lw=plotstyle.LW_SERIES,
        marker="o", ms=2.5, label=LABELS[t],
    )
# IQR band on OO to show even its upper spread stays low
ax.fill_between(ks, summary["OO_q25"], summary["OO_q75"], color=COLORS["OO"], alpha=0.18, lw=0)
ax.set_xscale("log")
ax.set_xlabel(r"neighbourhood size $k$")
ax.set_ylabel("Mean fraction of top-$k$ neighbours")
ax.set_ylim(0, 1)
plotstyle.title_with_subtitle(
    ax,
    "OO neighbours never dominate the local neighbourhood",
    f"pooled, {len(models)} pathology models; OO band = IQR",
)
ax.legend(fontsize=plotstyle.FS_ANNOT, loc="center right")
fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
(FIGDIR.parent / "png").mkdir(parents=True, exist_ok=True)
figpath = FIGDIR / "oo_fraction.pdf"
fig.savefig(figpath, bbox_inches="tight")
fig.savefig(FIGDIR.parent / "png" / "oo_fraction.png", dpi=plotstyle.DEFAULT_DPI, bbox_inches="tight")
print(f"wrote {figpath}")
