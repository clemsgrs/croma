"""Assemble the full faithful-dataset metrics.csv under the new reporting default:
per-model k* (uncapped, k_max from CFG), per-model tau, no SS/OO pruning, report support.

RI/MaRI/support/bio-acc come from the in-memory sweep (reusing existing embeddings,
no re-extraction); CCMR (headline m) + LTM are k-free and computed directly with
croma.CCMR on the same subset features. Output columns match generate_results_table.py.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/data/pathology/projects/clement/code/croma")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/experiments"))
from kstar_faithful_sweep import CFG, load_meta_and_features_index, sweep_model
from croma import CCMR
from croma.metrics.ccmr import CCMR_HEADLINE_M

PROD = {  # production run -> source of confounder_display_name
    "camelyon": "output/pathorob-camelyon-reduced-kstar",
    "tolkach": "output/pathorob-tolkach-esca",
    "tcga_2x2": "output/pathorob-tcga-2x2",
    "tcga_4x4": "output/pathorob-tcga-4x4",
}
# datasets whose faithful set == the production run: CCMR/LTM are k-free and identical,
# so reuse them (validated bit-exact) instead of recomputing the slow 112k paired search.
CCMR_FROM_PROD = {"tcga_2x2", "tcga_4x4"}
DESIGN_FOR_CCMR = {"paired": "paired_2x2", "dataset_wide": "dataset_wide"}


def run(ds):
    cfg = CFG[ds]
    meta, feat_idx, _ = load_meta_and_features_index(cfg)
    kmax = cfg["kmax"]
    ri_grid = list(range(1, kmax + 1))
    acc_grid = [1, 3, 5, 7, 9] + list(range(11, kmax + 1, 10))
    emb_dir = REPO / cfg["src"] / "embeddings"
    prod_metrics = pd.read_csv(REPO / PROD[ds] / "results/metrics.csv")
    cdn = prod_metrics["confounder_display_name"].iloc[0]
    prod_ccmr = prod_metrics.set_index("model")[["ccmr", "ccmr_ltm_alpha"]] if ds in CCMR_FROM_PROD else None
    cc_design = DESIGN_FOR_CCMR[cfg["design"]]
    models = sorted(p.stem for p in emb_dir.glob("*.npy"))
    print(f"[{ds}] design={cfg['design']} n_eval={len(meta)} models={len(models)} kmax={kmax}"
          f" ccmr_from_prod={ds in CCMR_FROM_PROD}", flush=True)
    rows = []
    for m in models:
        emb = np.load(emb_dir / f"{m}.npy")
        acc, ri, mari, supp, rec_tau = sweep_model(emb, feat_idx, meta, cfg["design"], kmax, ri_grid, acc_grid)
        ks = max(acc, key=lambda k: acc[k])
        if prod_ccmr is not None:
            ccmr_v, ltm_v = float(prod_ccmr.loc[m, "ccmr"]), float(prod_ccmr.loc[m, "ccmr_ltm_alpha"])
        else:
            res = CCMR.compute(features=emb[feat_idx], manifest=meta, confounder_column="confounder",
                               evaluation_design=cc_design, m=CCMR_HEADLINE_M, alpha=0.1, start_k=200, k_growth_factor=2.0)
            ccmr_v, ltm_v = float(res.value), float(res.ltm_alpha)
        rows.append(dict(dataset=ds, model=m, confounder_display_name=cdn, k=int(ks),
                         bio_knn_bacc=float(acc[ks]), ri=float(ri[ks]), mari=float(mari[ks]),
                         ccmr=ccmr_v, ccmr_ltm_alpha=ltm_v,
                         ri_undefined_frac=float(1.0 - supp[ks]), tau=float(rec_tau)))
        print(f"  {m:14} k*={ks:3d} RI={ri[ks]:.3f} MaRI={mari[ks]:.3f} "
              f"CCMR={ccmr_v:.3f} LTM={ltm_v:.3f} tau={rec_tau:.3f} supp={supp[ks]:.3f}", flush=True)
    out = REPO / "output/faithful" / ds / "results"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("ccmr", ascending=False)
    df.to_csv(out / "metrics.csv", index=False)
    print(f"wrote {out}/metrics.csv ({len(df)} models)")


if __name__ == "__main__":
    run(sys.argv[1])
