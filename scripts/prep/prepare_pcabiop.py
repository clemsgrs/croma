"""Build the PCaBiop slide-level APD tileset: PANDA (ID) + PAR (OOD).

PCaBiop reuses PANDA's slide-level cancer-detection cohort as the in-domain (ID)
pool and a newly curated external prostate-biopsy cohort, PAR (Leica-scanned), as
the held-out out-of-domain (OOD) pool -- structurally the slide-level analogue of
prostate-shift's KI+RUMC (ID) / NUS (OOD) design.

This writes two row-aligned artifacts, exactly the pair every APD dataset needs
(cf. ``prostate-shift`` + ``data/prostate/metadata/prostate_shift.csv``):

1. a combined tileset ``output/embeddings/pcabiop/`` with one ``<model>.npy`` per
   slide FM (ID rows stacked on top of OOD rows) plus a ``manifest.csv`` row-order
   contract, and
2. the APD metadata ``data/pcabiop/metadata/pcabiop.csv`` (columns ``subset``,
   ``slide_id``, ``biological_class``, ``medical_center``) that ``loaders.load_data``
   reads, row-aligned index-for-index with every ``<model>.npy``.

ID cohort = the 1,000 slides of ``data/benchmarks/panda.csv`` (the published PANDA
cancer-detection view): 250 benign + 250 cancer per provider (radboud, karolinska),
a balanced 2x2 with 250 slides/cell. Their embeddings already live in the
``panda-wsi`` tileset, so we slice them out rather than re-embed.

OOD cohort = PAR, balanced to 162 benign / 162 cancer. PAR is 162 benign / 177
cancer; the 15 surplus cancer slides are dropped stratified by ISUP (largest
remainder, fixed seed) so the retained cancer pool keeps its grade mix rather than
accidentally shedding all low- or high-grade cases. Balancing the OOD pool matches
PathoROB's curated OOD sets (camelyon ~50/50, tcga exactly balanced) and croma's own
NUS (300/300); the probe scores balanced accuracy either way, so this is for
protocol consistency, not to change the numbers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "bench"))
import layout  # noqa: E402

# --- inputs ----------------------------------------------------------------
WSI = Path("/data/pathology/projects/clement/croma/wsi")
PANDA_ID_MANIFEST = REPO / "data/benchmarks/panda.csv"          # 1,000 balanced cancer slides
PANDA_TILESET = "panda-wsi"                                      # holds the ID embeddings
PAR_LABELS = WSI / "par-labels.csv"

MODELS = ["PRISM", "TITAN", "Prov-GigaPath", "MOOZY"]
# Each slide FM's PAR slide_embeddings live under a hash-named slide2vec run dir.
PAR_FEATURE_DIR = {
    "PRISM": "552fd85i",
    "TITAN": "n02n1k2t",
    "Prov-GigaPath": "aetsexkb",
    "MOOZY": "nchyscec",
}
SEED = 1000            # PathoROB's own APD seed; used here only for the OOD drop
OOD_PER_CLASS = 162    # min(benign, cancer) in PAR

# --- outputs ---------------------------------------------------------------
TILESET = "pcabiop"
META_PATH = REPO / "data/pcabiop/metadata/pcabiop.csv"


def _stratified_keep(df, n_keep, strata_col, seed):
    """Return ``df`` down-sampled to exactly ``n_keep`` rows, proportional per stratum.

    Largest-remainder allocation over ``strata_col`` gives an integer quota per
    stratum summing to ``n_keep``; rows within a stratum are chosen with a seeded RNG.
    Deterministic given (df row order, seed).
    """
    counts = df[strata_col].value_counts().sort_index()
    quota = counts * n_keep / len(df)
    base = np.floor(quota).astype(int)
    remainder = n_keep - int(base.sum())
    # hand the leftover seats to the strata with the largest fractional parts
    order = (quota - base).sort_values(ascending=False, kind="stable").index
    for stratum in order[:remainder]:
        base[stratum] += 1
    rng = np.random.default_rng(seed)
    kept = []
    for stratum, k in base.items():
        rows = df[df[strata_col] == stratum]
        idx = rng.choice(rows.index.to_numpy(), size=int(k), replace=False)
        kept.append(rows.loc[sorted(idx)])
    return pd.concat(kept).sort_values("sample_id", kind="stable")


def build_id_rows():
    """ID metadata rows (radboud/karolinska x benign/cancer), deterministically ordered,
    each tagged with its positional row index in the panda-wsi tileset."""
    panda = pd.read_csv(PANDA_ID_MANIFEST)                       # sample_id,label(0/1),data_provider,...
    man = pd.read_csv(layout.tileset_manifest(PANDA_TILESET))    # panda-wsi row-order contract
    pos = {sid: i for i, sid in enumerate(man["sample_id"])}
    missing = [s for s in panda["sample_id"] if s not in pos]
    assert not missing, f"{len(missing)} panda.csv slides absent from {PANDA_TILESET} tileset"

    rows = pd.DataFrame({
        "subset": "ID",
        "slide_id": panda["sample_id"],
        "biological_class": np.where(panda["label"].to_numpy() == 0, "benign", "cancer"),
        "medical_center": panda["data_provider"],
        "sample_id": panda["sample_id"],
        "tile_row": [pos[s] for s in panda["sample_id"]],
    })
    return rows.sort_values(["medical_center", "biological_class", "sample_id"],
                            kind="stable").reset_index(drop=True)


def build_ood_rows():
    """OOD (PAR) metadata rows, balanced to OOD_PER_CLASS per class, deterministically ordered."""
    par = pd.read_csv(PAR_LABELS)
    benign = par[par["biological_class"] == "benign"]
    cancer = par[par["biological_class"] == "cancer"]
    assert len(benign) >= OOD_PER_CLASS and len(cancer) >= OOD_PER_CLASS
    cancer_kept = _stratified_keep(cancer, OOD_PER_CLASS, "isup_r1", SEED)
    kept = pd.concat([benign, cancer_kept])

    rows = pd.DataFrame({
        "subset": "OOD",
        "slide_id": kept["sample_id"],
        "biological_class": kept["biological_class"],
        "medical_center": "PAR",
        "sample_id": kept["sample_id"],
        "tile_row": -1,  # OOD embeddings are stacked from .pt, not sliced from a tileset
    })
    return rows.sort_values(["biological_class", "sample_id"], kind="stable").reset_index(drop=True)


def load_ood_embeddings(model, ood_rows):
    """Stack PAR slide embeddings for ``model`` in ``ood_rows`` order."""
    sd = WSI / "features" / PAR_FEATURE_DIR[model] / "slide_embeddings"
    vecs = []
    for sid in ood_rows["sample_id"]:
        t = torch.load(sd / f"{sid}.pt", map_location="cpu")
        vecs.append(np.asarray(t, dtype=np.float32).reshape(-1))
    return np.stack(vecs)


def main():
    id_rows, ood_rows = build_id_rows(), build_ood_rows()
    meta = pd.concat([id_rows, ood_rows]).reset_index(drop=True)

    out_dir = layout.embeddings_dir(TILESET)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        id_emb = np.load(layout.embedding_path(PANDA_TILESET, model))[id_rows["tile_row"].to_numpy()]
        ood_emb = load_ood_embeddings(model, ood_rows)
        assert id_emb.shape[1] == ood_emb.shape[1], f"{model}: ID dim {id_emb.shape} vs OOD {ood_emb.shape}"
        emb = np.vstack([id_emb.astype(np.float32), ood_emb]).astype(np.float32)
        assert len(emb) == len(meta)
        np.save(layout.embedding_path(TILESET, model), emb)
        (out_dir / f"{model}.npy.json").write_text(json.dumps(
            {"tileset": TILESET, "model": model, "dim": int(emb.shape[1]),
             "n_id": int(len(id_rows)), "n_ood": int(len(ood_rows))}, indent=2))
        print(f"  {model:16s} -> {emb.shape} (ID {len(id_rows)} + OOD {len(ood_rows)})")

    # tileset row-order contract (provenance; APD reads META_PATH, not this)
    pd.DataFrame({
        "sample_id": meta["sample_id"],
        "label": meta["biological_class"],
        "confounder": meta["medical_center"],
        "slide_id": meta["slide_id"],
        "subset": meta["subset"],
    }).to_csv(layout.tileset_manifest(TILESET), index=False)

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta[["subset", "slide_id", "biological_class", "medical_center"]].to_csv(META_PATH, index=False)

    print(f"\nwrote {out_dir}/ and {META_PATH}")
    print("ID cells (medical_center x biological_class):")
    print(pd.crosstab(id_rows["medical_center"], id_rows["biological_class"]))
    print("OOD pool (biological_class):")
    print(ood_rows["biological_class"].value_counts().to_string())


if __name__ == "__main__":
    main()
