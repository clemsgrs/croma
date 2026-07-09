"""Uncapped k* sweep on the FAITHFUL (PathoROB-comparable) datasets.

Reuses already-extracted embeddings (no re-extraction): Camelyon-faithful and
Tolkach-faithful are subset from the full runs; TCGA-2x2/4x4 already are faithful.

Per model it sweeps k to KMAX and reports, all at the model's own biological k*
(argmax bio-kNN balanced accuracy, PathoROB-style sparse grid) and at the
median-k* across models (the PathoROB reporting convention):
  RI@k*, RI@median ; MaRI@k*, MaRI@median ; support@k* ; and the
  cross-model correlation RI@k* vs RI@median (and same for MaRI).

Pooling matches the pipeline: RI = sum(SO)/sum(SO+OS) over all (sub)sets;
paired datasets aggregate SO/OS counts across the 2x2 subsets before the ratio.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

# Repo root autodetected from this file's location: scripts/studies/kstar_faithful_sweep.py
# -> parents[2] is the croma repo root, so the reproduction works from any checkout.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from croma import RI, MaRI
from croma.metrics.neighbors import _prepare_neighbors, _knn_balanced_accuracy_by_k

TAU = 0.2
CFG = {
    "camelyon":  dict(design="dataset_wide", kmax=80,  pathorob_k=11,
                      src="output/pathorob-camelyon", sel=("confounder", {"RUMC", "UMCU"})),
    "tolkach":   dict(design="dataset_wide", kmax=100, pathorob_k=46,
                      src="output/pathorob-tolkach-esca",
                      sel=("__faithful__", "data/pathorob/manifests/pathorob-tolkach-esca-faithful.csv")),
    "tcga_4x4":  dict(design="dataset_wide", kmax=100, pathorob_k=None,
                      src="output/pathorob-tcga-4x4", sel=None),
    "tcga_2x2":  dict(design="paired", kmax=100, pathorob_k=61,
                      src="output/pathorob-tcga-2x2",
                      input_manifest="data/pathorob/manifests/pathorob-tcga-2x2.csv"),
}


def load_meta_and_features_index(cfg):
    """Return (meta DataFrame with cols label/confounder/slide_id[/subset],
    row->feature-index array, n_features)."""
    esm = pd.read_csv(REPO / cfg["src"] / "embedding_source_manifest.csv")
    sid2idx = {s: i for i, s in enumerate(esm["sample_id"])}
    if cfg["design"] == "dataset_wide":
        meta = esm.rename(columns={}).copy()
        if cfg["sel"] is not None:
            kind, arg = cfg["sel"]
            if kind == "confounder":
                mask = meta["confounder"].isin(arg).to_numpy()
            else:  # __faithful__ : keep sample_ids present in a faithful manifest
                keep = set(pd.read_csv(REPO / arg)["sample_id"])
                mask = meta["sample_id"].isin(keep).to_numpy()
            meta = meta[mask].reset_index(drop=True)
        feat_idx = meta["sample_id"].map(sid2idx).to_numpy()
        return meta, feat_idx, len(esm)
    # paired: occurrences from the input manifest, mapped to unique-patch embeddings
    inp = pd.read_csv(REPO / cfg["input_manifest"])
    inp = inp.rename(columns={"medical_center": "confounder"})
    inp["__fidx"] = inp["sample_id"].map(sid2idx)
    assert inp["__fidx"].notna().all(), "some occurrences have no embedding"
    return inp, inp["__fidx"].to_numpy().astype(int), len(esm)


def sweep_model(emb_full, feat_idx, meta, design, kmax, ri_grid, acc_grid):
    """Return acc{k}, ri{k}, mari{k}, support{k} pooled across subset(s)."""
    if design == "dataset_wide":
        groups = [np.arange(len(meta))]
    else:
        groups = [np.asarray(idx, dtype=int)
                  for idx in meta.groupby("subset", sort=True).indices.values()]

    SO = {k: 0.0 for k in ri_grid}; OS = {k: 0.0 for k in ri_grid}
    mSO = {k: 0.0 for k in ri_grid}; mOS = {k: 0.0 for k in ri_grid}
    inf = {k: 0 for k in ri_grid}; ntot = {k: 0 for k in ri_grid}
    acc_subs = []
    typed_d = []   # cosine distances of SO/OS neighbours -> per-model tau (median)
    stored = []    # neighbour data per subset, for the MaRI pass at the per-model tau
    # --- pass 1: neighbours, accuracy, RI (tau-independent), collect typed distances ---
    for g in groups:
        if len(g) <= 3:
            continue
        X = emb_full[feat_idx[g]].astype(np.float64)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        labels = pd.factorize(meta.iloc[g]["label"])[0].astype(int)
        centers = pd.factorize(meta.iloc[g]["confounder"])[0].astype(int)
        slide = meta.iloc[g]["slide_id"].astype(str).to_numpy()
        km = min(kmax, len(g) - 1)
        nidx, ndist, vc = _prepare_neighbors(X, slide, km)
        nl = labels[np.clip(nidx, 0, None)]
        nc = centers[np.clip(nidx, 0, None)]
        valid = nidx >= 0
        typed = valid & ((nl == labels[:, None]) ^ (nc == centers[:, None]))
        td = ndist[typed]
        typed_d.append(td[np.isfinite(td)])
        gri = [k for k in ri_grid if k <= km]
        sri = RI._score_all_k_from_neighbors(labels, centers, nidx, ndist, vc, gri)
        for k in gri:
            SO[k] += float(sri[k][4].sum()); OS[k] += float(sri[k][5].sum())
            inf[k] += int(sri[k][2].sum()); ntot[k] += len(g)
        ag = [k for k in acc_grid if k < len(g)]
        acc_subs.append(_knn_balanced_accuracy_by_k(features=X, labels=labels,
                        slide_ids=slide, k_values=ag, warn_context="sweep"))
        stored.append((labels, centers, nidx, ndist, vc, gri, len(g)))
    # per-model tau = median typed-neighbour distance over the whole evaluation
    alltd = np.concatenate(typed_d) if typed_d else np.array([np.nan])
    rec_tau = float(np.median(alltd))
    # --- pass 2: MaRI scored at the per-model tau ---
    for labels, centers, nidx, ndist, vc, gri, n in stored:
        sma = MaRI._score_all_k_from_neighbors(labels, centers, nidx, ndist, vc, gri, tau=rec_tau)
        for k in gri:
            mSO[k] += float(sma[k][4].sum()); mOS[k] += float(sma[k][5].sum())
    acc = {}
    allk = sorted({k for d in acc_subs for k in d})
    for k in allk:
        v = [d[k] for d in acc_subs if k in d]
        if v: acc[k] = float(np.mean(v))
    ri = {k: (SO[k] / (SO[k] + OS[k]) if (SO[k] + OS[k]) > 0 else 0.5) for k in ri_grid if ntot[k]}
    mari = {k: (mSO[k] / (mSO[k] + mOS[k]) if (mSO[k] + mOS[k]) > 0 else 0.5) for k in ri_grid if ntot[k]}
    supp = {k: inf[k] / ntot[k] for k in ri_grid if ntot[k]}
    return acc, ri, mari, supp, rec_tau


def main(ds):
    cfg = CFG[ds]
    meta, feat_idx, nfeat = load_meta_and_features_index(cfg)
    kmax = cfg["kmax"]
    ri_grid = list(range(1, kmax + 1))
    acc_grid = [1, 3, 5, 7, 9] + list(range(11, kmax + 1, 10))
    emb_dir = REPO / cfg["src"] / "embeddings"
    models = sorted(p.stem for p in emb_dir.glob("*.npy"))
    print(f"[{ds}] design={cfg['design']} n_eval={len(meta)} models={len(models)} kmax={kmax}", flush=True)

    rec = {}
    for m in models:
        emb = np.load(emb_dir / f"{m}.npy")
        acc, ri, mari, supp, rec_tau = sweep_model(emb, feat_idx, meta, cfg["design"], kmax, ri_grid, acc_grid)
        kstar = max(acc, key=lambda k: acc[k])
        rec[m] = dict(kstar=int(kstar), acc_kstar=round(acc[kstar], 4),
                      ri=ri, mari=mari, supp=supp, rec_tau=rec_tau)
        print(f"  {m:14} k*={kstar:3d}  RI@k*={ri[kstar]:.3f}  MaRI@k*={mari[kstar]:.3f}  "
              f"supp@k*={supp[kstar]:.3f}  rec_tau={rec_tau:.3f}", flush=True)

    kstars = [rec[m]["kstar"] for m in models]
    medk = int(np.median(kstars))
    ri_ks = np.array([rec[m]["ri"][rec[m]["kstar"]] for m in models])
    ri_md = np.array([rec[m]["ri"].get(medk, np.nan) for m in models])
    ma_ks = np.array([rec[m]["mari"][rec[m]["kstar"]] for m in models])
    ma_md = np.array([rec[m]["mari"].get(medk, np.nan) for m in models])
    rec_taus = np.array([rec[m]["rec_tau"] for m in models])
    out = dict(
        dataset=ds, design=cfg["design"], pathorob_k=cfg["pathorob_k"],
        tau_policy="per-model (median typed-neighbour distance)",
        tau_median=round(float(np.median(rec_taus)), 4),
        tau_range=f"{rec_taus.min():.3f}-{rec_taus.max():.3f}",
        median_kstar=medk, kstar_min=int(min(kstars)), kstar_max=int(max(kstars)),
        kstar_spread=f"{min(kstars)}-{max(kstars)}",
        per_model={m: dict(kstar=rec[m]["kstar"],
                           RI_at_kstar=round(float(rec[m]["ri"][rec[m]["kstar"]]), 4),
                           RI_at_medk=round(float(rec[m]["ri"].get(medk, np.nan)), 4),
                           MaRI_at_kstar=round(float(rec[m]["mari"][rec[m]["kstar"]]), 4),
                           MaRI_at_medk=round(float(rec[m]["mari"].get(medk, np.nan)), 4),
                           support_at_kstar=round(float(rec[m]["supp"][rec[m]["kstar"]]), 4))
                   for m in models},
        RI_kstar_vs_medk=dict(pearson=round(float(pearsonr(ri_ks, ri_md)[0]), 4),
                              spearman=round(float(spearmanr(ri_ks, ri_md)[0]), 4),
                              dRI_mean=round(float(np.mean(np.abs(ri_ks - ri_md))), 4),
                              dRI_max=round(float(np.max(np.abs(ri_ks - ri_md))), 4)),
        MaRI_kstar_vs_medk=dict(pearson=round(float(pearsonr(ma_ks, ma_md)[0]), 4),
                                spearman=round(float(spearmanr(ma_ks, ma_md)[0]), 4),
                                dMaRI_mean=round(float(np.mean(np.abs(ma_ks - ma_md))), 4),
                                dMaRI_max=round(float(np.max(np.abs(ma_ks - ma_md))), 4)),
    )
    outdir = REPO / "output/faithful/k-star" / ds
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outdir / "kstar_sweep.json", "w"), indent=2)
    # tidy CSV
    pd.DataFrame(out["per_model"]).T.to_csv(outdir / "kstar_sweep.csv")
    print(f"\n[{ds}] median k*={medk} (spread {out['kstar_spread']}; PathoROB={cfg['pathorob_k']})")
    print(f"  MaRI tau: per-model (median typed-dist) median={out['tau_median']} "
          f"(range {out['tau_range']})")
    print(f"  RI   @k* vs @median-k: r={out['RI_kstar_vs_medk']['pearson']}  "
          f"|dRI|max={out['RI_kstar_vs_medk']['dRI_max']}")
    print(f"  MaRI @k* vs @median-k: r={out['MaRI_kstar_vs_medk']['pearson']}  "
          f"|dMaRI|max={out['MaRI_kstar_vs_medk']['dMaRI_max']}")
    print(f"  wrote {outdir}/kstar_sweep.json")


if __name__ == "__main__":
    main(sys.argv[1])
