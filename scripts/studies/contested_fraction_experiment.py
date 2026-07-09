"""Contested-fraction analysis: why pooled MaRI tracks RI at the operating k*.

MaRI can only differ from RI on *contested* samples -- those whose fixed-k neighbourhood
contains at least one SO (same biology, other confounder) AND at least one OS (other
biology, same confounder). If a sample's defined neighbourhood is "pure" (only SO, or only
OS), its MaRI denominator has a single term and MaRI_i == RI_i exactly, regardless of tau.

This script measures, for every model at its selected k* (read from metrics.csv), the
fraction of defined samples that are pure-SO / pure-OS / contested, and pairs it with the
pooled RI and MaRI@0.2 (also from metrics.csv) so the pooled gap MaRI - RI can be read off.

The two findings it backs in the manuscript:
  (1) At local k*, the contested fraction is tiny (a few percent), so almost every defined
      sample has MaRI_i == RI_i by construction -- the per-sample reason pooled MaRI ~ RI.
      This is the typed_neighbor_rank result from the other lens (both an SO and an OS are
      reached only deep, median rank ~314), so a local k* neighbourhood rarely holds both.
  (2) The pooled gap MaRI - RI is nonetheless not identically zero: where favorable (SO)
      neighbours are *systematically* closer than unfavorable (OS) ones, the asymmetry
      accumulates instead of cancelling and MaRI rises above RI (e.g. UNI2-h), and vice
      versa (e.g. Midnight-12k). Pooled MaRI carries a weak global directional-margin
      signal; the per-sample margin story is left to CRoMa (deep search, near-full coverage).

Contested counts come from the same core neighbour helpers the benchmark uses
(_prepare_neighbors), so "defined" matches metrics.csv exactly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

from _neighbor_analysis import REPO, load_meta, prepare_embedding

sys.path.insert(0, str(REPO / "scripts" / "bench"))
import views  # noqa: E402

from croma.metrics.neighbors import _prepare_neighbors

PROTOCOL = "k-star"
view = views.load_view("pathorob-camelyon")  # row-view over the pathorob-camelyon tileset
METRICS = view.results_dir(PROTOCOL) / "metrics.csv"
STUDIES = view.studies_dir(PROTOCOL)

df = view.eval_manifest
labels, centers, slide = load_meta(df)
n = len(df)

mdf = pd.read_csv(METRICS).set_index("model")
kstar = mdf["k"].astype(int).to_dict()
ri_pooled = mdf["ri"].astype(float).to_dict()
mari_pooled = mdf["mari"].astype(float).to_dict()

models = view.models
print(f"{n} samples, {len(models)} models\n")

rows = []
for model in models:
    X = prepare_embedding(view.features(model))
    k = int(kstar[model])
    nidx, ndist, vc = _prepare_neighbors(X, slide, k)
    so = np.zeros(n, int)
    os = np.zeros(n, int)
    for i in range(n):
        for pos in range(int(vc[i])):
            j = int(nidx[i, pos])
            if j < 0:
                continue
            same_label = labels[j] == labels[i]
            same_center = centers[j] == centers[i]
            if same_label and not same_center:
                so[i] += 1
            elif (not same_label) and same_center:
                os[i] += 1
    defined = (so + os) > 0
    nd = int(defined.sum())
    n_pure_so = int(((so > 0) & (os == 0)).sum())
    n_pure_os = int(((os > 0) & (so == 0)).sum())
    n_contested = int(((so > 0) & (os > 0)).sum())
    gap = float(mari_pooled[model]) - float(ri_pooled[model])
    rows.append(
        {
            "model": model,
            "kstar": k,
            "defined_frac": nd / n,
            "pure_so_frac_of_defined": n_pure_so / nd if nd else float("nan"),
            "pure_os_frac_of_defined": n_pure_os / nd if nd else float("nan"),
            "contested_frac_of_defined": n_contested / nd if nd else float("nan"),
            "ri": float(ri_pooled[model]),
            "mari": float(mari_pooled[model]),
            "mari_minus_ri": gap,
        }
    )

summary = pd.DataFrame(rows).sort_values("contested_frac_of_defined", ascending=False)
pd.set_option("display.width", 220, "display.max_columns", 20)
view = summary.copy()
for c in ["defined_frac", "pure_so_frac_of_defined", "pure_os_frac_of_defined", "contested_frac_of_defined"]:
    view[c] = (100 * view[c]).round(1)
view[["ri", "mari", "mari_minus_ri"]] = view[["ri", "mari", "mari_minus_ri"]].round(3)
print(view.to_string(index=False))

cf = summary["contested_frac_of_defined"]
print(
    f"\ncontested fraction of defined samples: min {100*cf.min():.1f}%  "
    f"max {100*cf.max():.1f}%  median {100*cf.median():.1f}%"
)
gaps = summary["mari_minus_ri"]
top_pos = summary.loc[gaps.idxmax()]
top_neg = summary.loc[gaps.idxmin()]
print(
    f"pooled gap MaRI-RI range: [{gaps.min():+.3f} ({top_neg['model']}), "
    f"{gaps.max():+.3f} ({top_pos['model']})]"
)

STUDIES.mkdir(parents=True, exist_ok=True)
out_csv = STUDIES / "contested_fraction_summary.csv"
summary.to_csv(out_csv, index=False)
json.dump(
    {
        "by_model": {r["model"]: r for r in rows},
        "contested_frac_min": float(cf.min()),
        "contested_frac_max": float(cf.max()),
        "contested_frac_median": float(cf.median()),
        "pooled_gap_min": {"model": str(top_neg["model"]), "gap": float(gaps.min())},
        "pooled_gap_max": {"model": str(top_pos["model"]), "gap": float(gaps.max())},
    },
    open(STUDIES / "contested_fraction_summary.json", "w"),
    indent=1,
)
print(f"\nwrote {out_csv}")
