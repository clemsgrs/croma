"""tau-sensitivity sweep for MaRI: is the MaRI-vs-RI reordering stable across tau?

MaRI weights neighbour evidence by exp(-distance / tau). The pipeline sets tau per model
to its median typed-neighbour distance (auto); this script checks that the headline claim
-- that switching from RI to MaRI *reorders* the model ranking -- is not an artefact of
the chosen tau, by sweeping tau across a wide grid that brackets the per-model auto values.

This script sweeps tau and checks two things across all 16 models, each scored at its own
k* (the same k* the benchmark selected, read from metrics.csv):

  1. The MaRI model ranking is essentially invariant to tau: pairwise Spearman rho
     between MaRI(tau_i) and MaRI(tau_j) stays ~1 over the whole grid. If true, "tau=0.2"
     is not load-bearing -- any tau gives the same ordering.
  2. That ranking genuinely differs from the RI ranking (Spearman rho_{MaRI,RI} < 1) and
     differs by the *same* reordering at every tau. So the MaRI reshuffle is a stable
     property of margin-awareness, not a knob-tuning accident.

MaRI and RI values are computed by calling the production CRoMa/RI metric code
(`MaRI.compute`, `RI.compute`, dataset_wide), verified to reproduce metrics.csv
(Virchow2: RI 0.814118, MaRI@0.2 0.845230).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
from _neighbor_analysis import REPO, prepare_embedding  # noqa: E402
import views  # noqa: E402

from croma import MaRI, RI

PROTOCOL = "k-star"
view = views.load_view("camelyon")  # row-view over the pathorob-camelyon tileset
METRICS = view.results_dir(PROTOCOL) / "metrics.csv"
STUDIES = view.studies_dir(PROTOCOL)
FIGDIR = REPO / "paper/figures/results/pathorob-camelyon-faithful/pdf"

TAUS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
TAU_REF = 0.2


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (rank then Pearson); deterministic, no scipy."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


df = view.eval_manifest
_metrics = pd.read_csv(METRICS).set_index("model")
kstar = _metrics["k"].astype(int).to_dict()
tau_auto = _metrics["tau"].astype(float)  # per-model auto tau (median typed-neighbour dist)
TAU_MIN, TAU_MED, TAU_MAX = float(tau_auto.min()), float(tau_auto.median()), float(tau_auto.max())
models = view.models
print(f"{len(df)} samples, {len(models)} models, taus={TAUS}\n")

# The MaRI(tau) curves are deterministic from the embeddings; reuse the cached
# summary when it matches the current taus/models so a re-render (e.g. for a style
# refresh) skips the expensive recompute. Delete the JSON to force recomputation.
STUDIES.mkdir(parents=True, exist_ok=True)
CACHE_JSON = STUDIES / "tau_sensitivity_summary.json"
_cache = json.loads(CACHE_JSON.read_text()) if CACHE_JSON.exists() else None
_cache_ok = bool(
    _cache
    and [float(t) for t in _cache.get("taus", [])] == TAUS
    and set(_cache.get("ri", {})) == set(models)
)

ri_vals: dict[str, float] = {}
mari_vals: dict[float, dict[str, float]] = {t: {} for t in TAUS}
if _cache_ok:
    print("loaded MaRI(tau) curves from cache (skipping recompute)")
    ri_vals = {m: float(v) for m, v in _cache["ri"].items()}
    mari_vals = {
        t: {m: float(_cache["mari_by_tau"][str(t)][m]) for m in models} for t in TAUS
    }
else:
    for model in models:
        X = prepare_embedding(view.features(model), np.float64, normalize=False)
        k = int(kstar[model])
        ri_vals[model] = float(
            RI.compute(
                X, df, confounder_column="confounder", k_candidates=[k],
                evaluation_design="dataset_wide",
            ).value
        )
        for t in TAUS:
            mari_vals[t][model] = float(
                MaRI.compute(
                    X, df, confounder_column="confounder", k_candidates=[k], tau=t,
                    evaluation_design="dataset_wide",
                ).value
            )

# ---- per-model table: RI + MaRI(tau) ----
table = pd.DataFrame(
    {"model": models, "kstar": [kstar[m] for m in models], "RI": [ri_vals[m] for m in models]}
)
for t in TAUS:
    table[f"MaRI@{t}"] = [mari_vals[t][m] for m in models]
table = table.sort_values("RI", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 220, "display.max_columns", 30)
print(table.round(3).to_string(index=False))

# ---- rank-stability statistics ----
ri_arr = np.array([ri_vals[m] for m in models])
mari_arr = {t: np.array([mari_vals[t][m] for m in models]) for t in TAUS}

rho_mari_vs_ref = {t: spearman(mari_arr[t], mari_arr[TAU_REF]) for t in TAUS}
rho_mari_vs_ri = {t: spearman(mari_arr[t], ri_arr) for t in TAUS}
pairwise = [
    spearman(mari_arr[ti], mari_arr[tj])
    for i, ti in enumerate(TAUS)
    for tj in TAUS[i + 1 :]
]
min_pairwise_rho = float(min(pairwise))

print(f"\nSpearman(MaRI(tau), MaRI@{TAU_REF}) by tau:")
for t in TAUS:
    print(f"  tau={t:<5} rho={rho_mari_vs_ref[t]:.4f}")
print("Spearman(MaRI(tau), RI) by tau:")
for t in TAUS:
    print(f"  tau={t:<5} rho={rho_mari_vs_ri[t]:.4f}")
print(f"\nmin pairwise Spearman across all tau pairs = {min_pairwise_rho:.4f}")
print(
    f"MaRI-vs-RI Spearman range = "
    f"[{min(rho_mari_vs_ri.values()):.4f}, {max(rho_mari_vs_ri.values()):.4f}]  (<1 => genuine reorder)"
)

json.dump(
    {
        "taus": TAUS,
        "tau_ref": TAU_REF,
        "ri": ri_vals,
        "mari_by_tau": {str(t): mari_vals[t] for t in TAUS},
        "spearman_mari_vs_ref": {str(t): rho_mari_vs_ref[t] for t in TAUS},
        "spearman_mari_vs_ri": {str(t): rho_mari_vs_ri[t] for t in TAUS},
        "min_pairwise_spearman": min_pairwise_rho,
    },
    open(STUDIES / "tau_sensitivity_summary.json", "w"),
    indent=1,
)
table.to_csv(STUDIES / "tau_sensitivity_summary.csv", index=False)
print(f"\nwrote {STUDIES / 'tau_sensitivity_summary.csv'}")

# ---- figure: MaRI(tau) per model; flat, non-crossing lines = tau-stable ordering ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from croma import plotstyle
from croma.plotstyle import color_for_model, model_sort_key

order = sorted(models, key=lambda m: (model_sort_key(m), m))

fig, ax = plt.subplots(figsize=(plotstyle.COL_DOUBLE, 4.6))
plotstyle.style_axes(ax)
for model in order:
    ax.plot(
        TAUS, [mari_vals[t][model] for t in TAUS],
        color=color_for_model(model), lw=plotstyle.LW_SERIES, marker="o", ms=2.5,
        alpha=0.95, label=model,
    )
ax.axvspan(TAU_MIN, TAU_MAX, color=plotstyle.MUTED_TEXT_COLOR, alpha=0.08, lw=0)
ax.axvline(
    TAU_MED, color=plotstyle.REFERENCE_LINE_COLOR, ls="--",
    lw=plotstyle.LW_REFERENCE, alpha=0.7,
)
ax.text(
    TAU_MED, 0.96, r"per-model $\tau$ (auto)", fontsize=plotstyle.FS_ANNOT,
    ha="center", va="top", color=plotstyle.MUTED_TEXT_COLOR,
)
ax.set_xscale("log")
ax.set_xlabel(r"MaRI temperature $\tau$")
ax.set_ylabel(r"MaRI ($k^\star$, dataset-wide)")
ax.set_ylim(0, 1)
plotstyle.title_with_subtitle(
    ax,
    r"MaRI model ordering is stable across $\tau$",
    f"min pairwise Spearman $\\rho={min_pairwise_rho:.3f}$; "
    f"vs RI $\\rho={rho_mari_vs_ri[TAU_REF]:.3f}$ within the per-model $\\tau$ band; "
    f"diverges only at $\\tau=0.05$ ($\\rho={rho_mari_vs_ri[0.05]:.3f}$)",
)
ax.legend(
    fontsize=plotstyle.FS_ANNOT, ncol=6, loc="upper center",
    bbox_to_anchor=(0.5, -0.16), frameon=False, columnspacing=1.1, handlelength=1.7,
)
figdir = FIGDIR
figdir.mkdir(parents=True, exist_ok=True)
(figdir.parent / "png").mkdir(parents=True, exist_ok=True)
figpath = figdir / "tau_sensitivity.pdf"
fig.savefig(figpath, bbox_inches="tight")
fig.savefig(figdir.parent / "png" / "tau_sensitivity.png", dpi=plotstyle.DEFAULT_DPI, bbox_inches="tight")
print(f"wrote {figpath}")
