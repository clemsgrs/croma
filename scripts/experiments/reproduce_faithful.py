"""Single-command reproduction of the paper's FAITHFUL (PathoROB-comparable) results.

For each faithful dataset it (1) materialises the model embeddings by subsetting the
already-extracted full-run embeddings -- no GPU / re-extraction -- writing them with a
sidecar whose fingerprint matches what benchmark.py expects, then (2) runs the canonical
benchmark.py with the paper's reporting defaults (k_max=100; per-model auto tau; no
SS/OO pruning), then (3) regenerates the LaTeX results table. Because the numbers come
straight from benchmark.py, they are guaranteed identical to a fresh from-embeddings run.

Usage:
    python scripts/experiments/reproduce_faithful.py            # all datasets
    python scripts/experiments/reproduce_faithful.py camelyon   # one dataset

Faithful datasets (exactly PathoROB's RI sets):
    camelyon  : RUMC+UMCU, 20,400 patches, dataset_wide
    tolkach   : 3 cohorts (TCGA excluded), 9,000 patches, dataset_wide
    tcga_2x2  : 112,800 paired occurrences, paired_2x2
    tcga_4x4  : 8,160 patches, dataset_wide (supplementary)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from benchmark import _prepare_eval_manifest  # noqa: E402
from croma.alignment import build_embedding_source_manifest  # noqa: E402
from croma.metrics.pairs import load_manifest  # noqa: E402
from input_fingerprint import manifest_fingerprint  # noqa: E402

MODELS = ("CONCH,CONCHv1.5,H-optimus-0,H-optimus-1,H0-mini,Hibou-B,Hibou-L,"
          "Midnight-12k,Phikon,Phikon-v2,Prost40M,Prov-GigaPath,UNI,UNI2-h,Virchow,Virchow2")

CFG = {
    # camelyon is the primary table (tab:main-results, the column-defining one): we
    # reproduce its metrics.csv here but regenerate its .tex separately so its
    # self-contained caption is preserved. Its body carries in-table bootstrap CCMR
    # CIs: regenerate with generate_results_table.py --with-ci (needs the sibling
    # bootstrap_uncertainty.csv from bootstrap_uncertainty.py), then restore the caption.
    "camelyon": dict(manifest="data/pathorob/manifests/pathorob-camelyon-faithful.csv",
                     design="dataset_wide", prod="output/pathorob-camelyon",
                     name=r"PathoROB Camelyon", label="tab:main-results",
                     out_tex=None),
    "tolkach": dict(manifest="data/pathorob/manifests/pathorob-tolkach-esca-faithful.csv",
                    design="dataset_wide", prod="output/pathorob-tolkach-esca",
                    name=r"PathoROB Tolkach-ESCA", label="tab:main-results-tolkach",
                    out_tex="paper/sections/results_table_tolkach.tex"),
    "tcga_2x2": dict(manifest="data/pathorob/manifests/pathorob-tcga-2x2.csv",
                     design="paired_2x2", prod="output/pathorob-tcga-2x2",
                     name=r"PathoROB TCGA (2$\times$2)", label="tab:main-results-tcga",
                     out_tex="paper/sections/results_table_tcga.tex"),
    "tcga_4x4": dict(manifest="data/pathorob/manifests/pathorob-tcga-4x4.csv",
                     design="dataset_wide", prod="output/pathorob-tcga-4x4",
                     name=r"PathoROB TCGA (4$\times$4)", label="tab:main-results-tcga4x4",
                     out_tex="paper/sections/results_table_tcga4x4.tex"),
}
CONFOUNDER = "medical_center"
OUT_ROOT = REPO / "output/faithful"


def materialise_embeddings(ds, cfg):
    """Subset full-run embeddings into the faithful run dir with benchmark-compatible
    sidecars, so benchmark.py reuses them instead of re-extracting."""
    stem = Path(cfg["manifest"]).stem
    dataset_dir = OUT_ROOT / stem
    emb_out = dataset_dir / "embeddings"
    emb_out.mkdir(parents=True, exist_ok=True)

    mdf = load_manifest(str(REPO / cfg["manifest"]), confounder_column=CONFOUNDER)
    eval_manifest = _prepare_eval_manifest(manifest_df=mdf, dataset_name=stem,
                                           evaluation_design=cfg["design"])
    emb_manifest, _keep = build_embedding_source_manifest(eval_manifest)
    fp = manifest_fingerprint(emb_manifest)
    n = int(len(emb_manifest))

    prod = REPO / cfg["prod"]
    prod_esm = pd.read_csv(prod / "embedding_source_manifest.csv")
    sid2i = {s: i for i, s in enumerate(prod_esm["sample_id"])}
    rows = emb_manifest["sample_id"].map(sid2i)
    assert rows.notna().all(), f"{ds}: {int(rows.isna().sum())} unique patches missing from {prod}"
    rows = rows.to_numpy(dtype=int)

    made = 0
    for m in MODELS.split(","):
        src_npy = prod / "embeddings" / f"{m}.npy"
        if not src_npy.exists():
            continue
        sub = np.load(src_npy)[rows]
        out_npy = emb_out / f"{m}.npy"
        np.save(out_npy, sub)
        src_meta = json.loads((src_npy.with_suffix(".npy.json")).read_text())
        sidecar = dict(src_meta)
        sidecar.update(manifest=str(dataset_dir / "embedding_source_manifest.csv"),
                       manifest_fingerprint=fp, n_samples=n,
                       embedding_dim=int(sub.shape[1]))
        (out_npy.with_suffix(".npy.json")).write_text(json.dumps(sidecar, indent=2))
        made += 1
    print(f"[{ds}] materialised {made} embeddings ({n} unique patches) -> {emb_out}", flush=True)
    return dataset_dir


def run(ds):
    cfg = CFG[ds]
    materialise_embeddings(ds, cfg)
    cmd = [sys.executable, str(REPO / "scripts/benchmark.py"),
           "--manifest", str(REPO / cfg["manifest"]),
           "--confounder-column", CONFOUNDER,
           "--evaluation-design", cfg["design"],
           "--models", MODELS,
           "--k-max", "100",
           "--output-dir", str(OUT_ROOT),
           "--device", "cpu", "--progress", "off"]
    print(f"[{ds}] running benchmark: {' '.join(cmd[-12:])}", flush=True)
    subprocess.run(cmd, check=True)
    stem = Path(cfg["manifest"]).stem
    metrics = OUT_ROOT / stem / "results" / "metrics.csv"
    print(f"[{ds}] metrics -> {metrics}", flush=True)
    if cfg["out_tex"]:
        gen = [sys.executable, str(REPO / "scripts/experiments/generate_results_table.py"),
               "--metrics", str(metrics), "--name", cfg["name"],
               "--label", cfg["label"], "--out", str(REPO / cfg["out_tex"])]
        subprocess.run(gen, check=True)
        print(f"[{ds}] regenerated {cfg['out_tex']}", flush=True)


if __name__ == "__main__":
    targets = [sys.argv[1]] if len(sys.argv) > 1 else list(CFG)
    for ds in targets:
        run(ds)
    # Refresh the inline derived scalars (\<Bench>CcmrSpan, ...) so prose stays in
    # sync with the regenerated tables instead of being hand-typed.
    subprocess.run([sys.executable, str(REPO / "scripts/experiments/generate_paper_values.py"),
                    "--root", str(REPO),
                    "--out", str(REPO / "paper/sections/generated_values.tex")],
                   check=True)
    print("regenerated paper/sections/generated_values.tex", flush=True)
