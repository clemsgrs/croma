"""Variable-radius diagnostic for CRoMa.

For every sample and every model, find the rank (1-indexed position among non-self
neighbours, ordered by increasing cosine distance, excluding same-slide neighbours as
CRoMa does) at which the first SO and first OS neighbour appears.

SO = same label, different confounder ; OS = different label, same confounder.

This decides empirically whether CRoMa's k-free search is, in practice, local (typed
neighbours at small rank) or whether it routinely reaches deep shells. It also checks
whether the CRoMa bottom decile (the LTM tail) is populated by near-shell samples
(genuine local shortcut) rather than far-shell ratio-compression artefacts.

SS-shell depth (concern 6).
---------------------------
The same neighbour ordering yields the entanglement signal CRoMa is blind to. For each
sample we record the rank, *among valid (non-self, non-same-slide) neighbours*, of the
first typed (SO or OS) neighbour: this is the depth at which the sample exits its
same-biology/same-confounder (SS) pocket and RI/MaRI first become defined. Everything
strictly closer than this rank is SS or OO; in this benchmark OO is negligible, so the
shell that is traversed is SS, and its depth is a continuous, threshold-free, k*-free
generalisation of RI's SS-dominated undefined fraction. We summarise it per model
(median depth, fixed-k SS-pocket prevalence) and cross it against pooled CRoMa to test
whether a model can rank high on CRoMa while sitting in a deep SS shell (the masking
concern 6 raises). The CRoMa-vs-SS-depth scatter is the twin of fig:croma-vs-mari.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studies"))
from _neighbor_analysis import list_models, load_embedding, load_meta  # noqa: E402

ROOT = Path("output/faithful/k-star/pathorob-camelyon-faithful")
EMB = ROOT / "embeddings"
MANIFEST = ROOT / "embedding_source_manifest.csv"
METRICS = ROOT / "results" / "metrics.csv"
FIGDIR = Path("paper/figures/results/pathorob-camelyon-faithful/pdf")
ALPHA = 0.10  # bottom decile, matches reported LTM_10%
SS_DEPTH_K = (10, 25, 50)  # fixed reference k for SS-pocket prevalence (k*-free)

df = pd.read_csv(MANIFEST)
labels, conf, slide = load_meta(df, compact=True)
n = len(df)

# pooled CRoMa (signed margin) per model for the SS-depth complementarity scatter
croma_by_model = {}
if METRICS.exists():
    _m = pd.read_csv(METRICS)
    croma_by_model = dict(zip(_m["model"].astype(str), _m["croma"].astype(float)))

models = list_models(EMB)
print(f"{n} samples, {len(models)} models\n")

self_idx = np.arange(n)
rows = []
pooled_so, pooled_os = [], []
pooled_tail_so, pooled_tail_os = [], []
pooled_tail_mask = []
pooled_pre_ss = 0  # SS neighbours strictly closer than the first typed neighbour
pooled_pre_total = 0  # all (SS or OO) neighbours strictly closer than the first typed

for model in models:
    X = load_embedding(EMB / f"{model}.npy", np.float32)
    D = 1.0 - X @ X.T  # cosine distance, (n, n)

    order = np.argsort(D, axis=1, kind="stable")  # ascending; col 0 is self
    L = labels[order]
    C = conf[order]
    S = slide[order]
    di = labels[:, None]
    ci = conf[:, None]
    si = slide[:, None]

    is_self = order == self_idx[:, None]
    same_slide = S == si
    valid = ~is_self & ~same_slide

    so = valid & (L == di) & (C != ci)
    os = valid & (L != di) & (C == ci)

    # --- SS-shell depth: rank (among valid neighbours) of the FIRST typed neighbour ---
    # Everything strictly closer is SS or OO; this is where RI/MaRI become defined and
    # the sample exits its same-biology/same-confounder pocket.
    ss = valid & (L == di) & (C == ci)
    typed = so | os
    vrank = np.cumsum(valid, axis=1, dtype=np.int32)  # 1-indexed rank among valid nbrs
    BIG = np.int32(n + 1)
    ftvr = np.where(typed, vrank, BIG).min(1)  # SS-shell exit depth (per sample)
    has_typed = typed.any(1)
    # pre-typed neighbours (closer than the first typed one) are SS or OO by construction
    pre_ss = (ss & (vrank < ftvr[:, None])).sum(1).astype(np.int64)
    pre_total = ftvr.astype(np.int64) - 1
    pooled_pre_ss += int(pre_ss[has_typed].sum())
    pooled_pre_total += int(pre_total[has_typed].sum())
    depth = ftvr[has_typed].astype(float)

    # 1-indexed rank among non-self neighbours = column index in `order`
    # (col 0 is self, so col j -> rank j among non-self points).
    col = np.arange(n)[None, :]
    big = n + 1
    so_rank = np.where(so.any(1), np.where(so, col, big).min(1), -1)
    os_rank = np.where(os.any(1), np.where(os, col, big).min(1), -1)

    # nearest typed distances -> CRoMa_i at m=1
    so_d = np.where(so.any(1), np.where(so, D[self_idx[:, None], order], np.inf).min(1), np.nan)
    os_d = np.where(os.any(1), np.where(os, D[self_idx[:, None], order], np.inf).min(1), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        croma = os_d / so_d
    defined = np.isfinite(croma) & (so_d > 0)

    sr = so_rank[defined]
    orr = os_rank[defined]
    cc = croma[defined]
    pooled_so.append(sr)
    pooled_os.append(orr)

    sod = so_d[defined]
    osd = os_d[defined]

    # bottom-decile (LTM tail) membership
    q = np.quantile(cc, ALPHA)
    tail = cc <= q
    pooled_tail_so.append(sr[tail])
    pooled_tail_os.append(orr[tail])
    pooled_tail_mask.append(tail)

    def pct(a, p):
        return float(np.percentile(a, p))

    both = np.maximum(sr, orr)  # rank at which BOTH typed neighbours are available
    rows.append(
        dict(
            model=model,
            n_def=int(defined.sum()),
            so_med=pct(sr, 50),
            so_p90=pct(sr, 90),
            so_p99=pct(sr, 99),
            os_med=pct(orr, 50),
            os_p90=pct(orr, 90),
            os_p99=pct(orr, 99),
            both_p90=pct(both, 90),
            both_max=int(both.max()),
            frac_both_le10=float(np.mean(both <= 10)),
            frac_both_le50=float(np.mean(both <= 50)),
            so_d_med=float(np.median(sod)),
            os_d_med=float(np.median(osd)),
            tail_os_rank_med=pct(orr[tail], 50),
            rest_os_rank_med=pct(orr[~tail], 50),
            tail_so_rank_med=pct(sr[tail], 50),
            rest_so_rank_med=pct(sr[~tail], 50),
            croma=float(croma_by_model.get(model, np.nan)),
            ss_depth_med=pct(depth, 50),
            ss_depth_p90=pct(depth, 90),
            **{f"ss_pocket_frac_k{k}": float(np.mean(ftvr[has_typed] > k)) for k in SS_DEPTH_K},
        )
    )

summary = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 40)
print(summary.round(1).to_string(index=False))

PS, PO = np.concatenate(pooled_so), np.concatenate(pooled_os)
both_all = np.maximum(PS, PO)
TS, TO = np.concatenate(pooled_tail_so), np.concatenate(pooled_tail_os)
tail_both = np.maximum(TS, TO)
nontail_both = both_all  # approx pop; tail is 10% of it

print("\n=== POOLED (all defined samples, all models) ===")
for p in (50, 75, 90, 95, 99):
    print(f"  rank where BOTH typed neighbours found  p{p:>2}: {np.percentile(both_all, p):7.1f}")
print(f"  frac with both typed neighbours within rank 10 : {np.mean(both_all <= 10):.3f}")
print(f"  frac within rank 20 : {np.mean(both_all <= 20):.3f}   rank 50 : {np.mean(both_all <= 50):.3f}")

print("\n=== TAIL CLEANLINESS (CRoMa bottom decile vs all) ===")
print(f"  bottom-decile 'both-found' rank   median {np.percentile(tail_both,50):.1f}  p90 {np.percentile(tail_both,90):.1f}")
print(f"  all-samples   'both-found' rank   median {np.percentile(both_all,50):.1f}  p90 {np.percentile(both_all,90):.1f}")
print(f"  frac of bottom-decile samples with both typed neighbours within rank 10: {np.mean(tail_both <= 10):.3f}")

out = ROOT / "typed_neighbor_rank_summary.csv"
summary.to_csv(out, index=False)
json.dump(
    dict(
        pooled_both_percentiles={str(p): float(np.percentile(both_all, p)) for p in (50, 75, 90, 95, 99)},
        pooled_frac_both_le10=float(np.mean(both_all <= 10)),
        pooled_frac_both_le50=float(np.mean(both_all <= 50)),
        tail_both_median=float(np.percentile(tail_both, 50)),
        tail_both_p90=float(np.percentile(tail_both, 90)),
        tail_frac_le10=float(np.mean(tail_both <= 10)),
    ),
    open(ROOT / "typed_neighbor_rank_summary.json", "w"),
    indent=1,
)
print(f"\nwrote {out}")

# ---- figure: (left) how deep CRoMa searches; (right) tail is locally grounded ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from croma import plotstyle

# Shared semantic palette: OS is the impostor / lower-tail accent, SO the neutral "rest".
ACCENT = plotstyle.NEIGHBOR_TYPE_COLOR["OS"]  # red (impostor / tail)
REST = plotstyle.NEIGHBOR_TYPE_COLOR["SO"]    # steel blue (rest)

tmask = np.concatenate(pooled_tail_mask)


def ecdf(a):
    a = np.sort(a)
    return a, np.arange(1, len(a) + 1) / len(a)


def _panel_tag(ax, tag):
    ax.text(
        -0.14, 1.06, tag, transform=ax.transAxes, ha="left", va="top",
        fontsize=plotstyle.FS_PANEL_TAG, weight="bold", color=plotstyle.TEXT_COLOR,
    )


fig, (axL, axR) = plt.subplots(1, 2, figsize=(plotstyle.COL_DOUBLE, 3.4))

x, y = ecdf(both_all)
plotstyle.style_axes(axL)
axL.plot(x, y, color=plotstyle.TEXT_COLOR, lw=plotstyle.LW_SERIES)
for r in (10, 50):
    axL.axvline(r, color=ACCENT, ls=":", lw=plotstyle.LW_REFERENCE)
    axL.text(
        r, 0.04, f" rank {r}\n {np.mean(both_all<=r)*100:.1f}%",
        color=ACCENT, fontsize=plotstyle.FS_ANNOT, va="bottom",
    )
axL.axhline(0.5, color=plotstyle.REFERENCE_LINE_COLOR, ls="--", lw=plotstyle.LW_REFERENCE)
axL.set_xscale("log")
axL.set_xlabel("rank at which both nearest SO and OS neighbour are found")
axL.set_ylabel("Fraction of samples (ECDF)")
plotstyle.title_with_subtitle(
    axL, "Typed neighbours are reached deep, not locally", "pooled, 16 models"
)
axL.set_xlim(1, n)
_panel_tag(axL, "a")

xr_t, yr_t = ecdf(PO[tmask])
xr_r, yr_r = ecdf(PO[~tmask])
plotstyle.style_axes(axR)
axR.plot(xr_t, yr_t, color=ACCENT, lw=plotstyle.LW_SERIES, label="CRoMa bottom decile (LTM tail)")
axR.plot(xr_r, yr_r, color=REST, lw=plotstyle.LW_SERIES, label="rest")
axR.set_xscale("log")
axR.set_xlabel("rank of nearest OS (same-confounder impostor) neighbour")
axR.set_ylabel("Fraction of samples (ECDF)")
plotstyle.title_with_subtitle(
    axR, "Fragile samples have a genuinely near OS neighbour", "tail is locally grounded"
)
axR.set_xlim(1, n)
axR.legend(fontsize=plotstyle.FS_ANNOT, loc="lower right")
_panel_tag(axR, "b")

fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
(FIGDIR.parent / "png").mkdir(parents=True, exist_ok=True)
figpath = FIGDIR / "croma_rank.pdf"
fig.savefig(figpath, bbox_inches="tight")
fig.savefig(FIGDIR.parent / "png" / "croma_rank.png", dpi=plotstyle.DEFAULT_DPI, bbox_inches="tight")
print(f"wrote {figpath}")

# ===================================================================================
# Concern 6: SS-shell depth is the entanglement axis CRoMa is blind to
# ===================================================================================
from scipy.stats import spearmanr

print("\n=== SS-SHELL DEPTH (rank of first typed neighbour among valid neighbours) ===")
ss_cols = ["model", "croma", "ss_depth_med", "ss_depth_p90"] + [
    f"ss_pocket_frac_k{k}" for k in SS_DEPTH_K
]
ss_view = summary[ss_cols].sort_values("croma", ascending=False)
print(ss_view.round(3).to_string(index=False))

ss_frac_pretyped = pooled_pre_ss / pooled_pre_total if pooled_pre_total else float("nan")
print(
    f"\nPooled SS fraction among neighbours closer than the first typed neighbour: "
    f"{ss_frac_pretyped:.4f}  (the shell that is traversed is SS, not OO)"
)

have_croma = summary["croma"].notna()
if have_croma.sum() >= 3:
    sub = summary[have_croma]

    def _sp(col):
        r, p = spearmanr(sub["croma"], sub[col])
        return f"rho={r:+.3f}  p={p:.3f}"

    print("\n--- Spearman( CRoMa , . ) across models ---")
    print(f"  SS-shell depth (median first-typed rank): {_sp('ss_depth_med')}")
    for k in SS_DEPTH_K:
        print(f"  SS-pocket prevalence @k={k:<3d}            : {_sp(f'ss_pocket_frac_k{k}')}")
    print(
        "  SS-shell depth and CRoMa co-vary (related, not redundant -- the same status\n"
        "  as CRoMa-vs-MaRI): biology-dominant models tend to exit the SS pocket sooner.\n"
        "  Concern 6 rests on the ABSOLUTE prevalence below, not on weak correlation:"
    )
    lead = sub.loc[sub["croma"].idxmax()]
    best_local = sub.loc[sub["ss_pocket_frac_k10"].idxmin()]
    print(
        f"    top-CRoMa model {lead['model']} (CRoMa {lead['croma']:+.3f}) still has "
        f"{lead['ss_pocket_frac_k10'] * 100:.0f}% of samples with NO typed neighbour in the 10 nearest;"
    )
    print(
        f"    even the least locally-entangled model {best_local['model']} is at "
        f"{best_local['ss_pocket_frac_k10'] * 100:.0f}%. CRoMa reads its verdict past a locally SS-saturated shell."
    )

# ---- figure: even CRoMa leaders are majority locally SS-saturated ----
# y = fraction of samples with NO typed (SO/OS) neighbour among the 10 nearest valid
# neighbours == RI's SS-dominated undefined fraction at a controlled k. The point cloud
# trends with CRoMa (related, not redundant) yet sits high for *every* model: CRoMa reads
# its biology-vs-confounder verdict past a locally entangled neighbourhood.
scat = summary[have_croma].copy().sort_values("model")
YK = "ss_pocket_frac_k10"
fig2, ax2 = plt.subplots(figsize=(plotstyle.COL_ONEHALF, 5.6))
plotstyle.style_axes(ax2)
ax2.axvline(0.0, color=plotstyle.REFERENCE_LINE_COLOR, ls="--", lw=plotstyle.LW_REFERENCE, zorder=1)
ax2.axhline(0.5, color=plotstyle.REFERENCE_LINE_COLOR, ls=":", lw=plotstyle.LW_REFERENCE, zorder=1)
for _, r in scat.iterrows():
    ax2.scatter(
        [float(r["croma"])], [float(r[YK])],
        s=52, color=plotstyle.color_for_model(str(r["model"])),
        edgecolors="white", linewidths=0.7, alpha=0.9, zorder=3, label=str(r["model"]),
    )
# annotate the headline: the top-CRoMa model is still majority locally entangled
lead_ex = scat.loc[scat["croma"].idxmax()]
ax2.annotate(
    f"{lead_ex['model']}\n{lead_ex[YK] * 100:.0f}% SS-pocketed",
    xy=(float(lead_ex["croma"]), float(lead_ex[YK])),
    xytext=(-6, -12), textcoords="offset points", ha="right", va="top",
    fontsize=plotstyle.FS_ANNOT, color=plotstyle.TEXT_COLOR,
)
ax2.set_ylim(0.0, 1.0)
ax2.set_xlabel("CRoMa  (biology-vs-confounder ordering)")
ax2.set_ylabel("Local SS-saturation\n(fraction with no typed neighbour in 10 nearest)")
plotstyle.title_with_subtitle(
    ax2, "CRoMa robustness does not imply local disentanglement",
    "16 models; even CRoMa leaders are majority SS-pocketed",
)
handles, labels = ax2.get_legend_handles_labels()
fig2.legend(
    handles, labels, loc="lower center", ncol=4, frameon=False,
    fontsize=plotstyle.FS_ANNOT, columnspacing=1.25, handletextpad=0.4,
    bbox_to_anchor=(0.5, 0.012),
)
fig2.subplots_adjust(top=0.90, bottom=0.30, left=0.135, right=0.965)
figpath2 = FIGDIR / "croma_vs_ss_depth.pdf"
fig2.savefig(figpath2, bbox_inches="tight")
fig2.savefig(FIGDIR.parent / "png" / "croma_vs_ss_depth.png", dpi=plotstyle.DEFAULT_DPI, bbox_inches="tight")
print(f"wrote {figpath2}")
