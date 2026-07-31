"""Bridge the standalone full-8,000 prostate embeddings into a benchmark-ready cache
for any KI+RUMC manifest -- WITHOUT re-embedding.

Generalises ``bridge_kirumc_cache.py`` (which only handled the all-rows binary) to
arbitrary manifests and either evaluation design. benchmark.py gets an embedding cache
HIT iff ``output/<manifest-stem>/embeddings/{model}.npy`` has ``len(embedding_manifest)``
rows in ``build_embedding_source_manifest`` order, plus a sidecar whose
``manifest_fingerprint`` equals ``manifest_fingerprint(embedding_manifest)``.

To guarantee the fingerprint matches by construction we reuse benchmark.py's OWN
manifest-preparation functions (``_prepare_eval_manifest`` -> ``_build_aligned_manifest``
-> ``build_embedding_source_manifest``): the embedding manifest is then byte-identical to
what benchmark recomputes. Rows are sourced from the full 8,000-row stacked arrays via
image_path (the unique join key), so no GPU / re-embed is needed.

Usage:
  python scripts/prep/bridge_prostate_cache.py \
      --manifest data/prostate-shift-4class-kirumc-paired.csv \
      --evaluation-design paired_2x2 --output-dir output
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "bench"))

import layout  # noqa: E402
from croma.alignment import build_embedding_source_manifest  # noqa: E402
from croma.metrics.pairs import load_manifest  # noqa: E402
from input_fingerprint import manifest_fingerprint  # noqa: E402
from extract_embeddings import _output_path_in_dir  # noqa: E402
from benchmark import _prepare_eval_manifest, _build_aligned_manifest  # noqa: E402

CONFOUNDER = "medical_center"
FULL_CSV = REPO / "data/prostate-shift-binary.csv"  # 8,000 rows, the embedded order
# The full 8,000-row prostate embeddings now live as the `prostate-shift` tileset.
FULL_EMB_DIR = layout.embeddings_dir("prostate-shift")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--evaluation-design", default="paired_2x2",
                   choices=["all", "paired_2x2"])
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Benchmark output root; cache lands in <output-dir>/<manifest-stem>/.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest)
    dataset_name = manifest_path.stem

    # 1. Replicate benchmark's manifest pipeline -> the exact embedding manifest it expects.
    manifest_df = load_manifest(str(manifest_path), confounder_column=CONFOUNDER)
    eval_manifest = _prepare_eval_manifest(
        manifest_df=manifest_df,
        dataset_name=dataset_name,
        evaluation_design=str(args.evaluation_design),
    )
    aligned_manifest = _build_aligned_manifest(
        eval_manifest=eval_manifest,
        evaluation_design=str(args.evaluation_design),
    )
    embedding_manifest, _keep = build_embedding_source_manifest(aligned_manifest)
    emb_fp = manifest_fingerprint(embedding_manifest)
    n_emb = len(embedding_manifest)
    print(f"[manifest] {dataset_name}: aligned rows={len(aligned_manifest)} "
          f"unique embedding rows={n_emb} fp={emb_fp[:16]}...")

    # 2. full-manifest image_path -> .npy row index (extract order == load_manifest order).
    full_manifest = load_manifest(str(FULL_CSV), confounder_column=CONFOUNDER).reset_index(drop=True)
    if len(full_manifest) != 8000:
        raise AssertionError(f"expected 8000 full rows, got {len(full_manifest)}")
    path_to_idx = {p: i for i, p in enumerate(full_manifest["image_path"].astype(str))}
    if len(path_to_idx) != 8000:
        raise AssertionError("full image_path not unique after normalize")

    sel = np.array([path_to_idx[p] for p in embedding_manifest["image_path"].astype(str)], dtype=int)
    if len(set(sel.tolist())) != n_emb:
        raise AssertionError("row mapping not bijective")
    sel_centers = set(full_manifest.iloc[sel][CONFOUNDER].unique().tolist())
    if not sel_centers <= {"KI", "RUMC"}:
        raise AssertionError(f"unexpected centres in selection: {sel_centers}")

    # 3. write per-model sliced caches + sidecars in embedding_manifest order.
    dataset_dir = Path(args.output_dir) / dataset_name
    emb_dir = dataset_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    embedding_manifest.to_csv(dataset_dir / "embedding_source_manifest.csv", index=False)

    models = sorted(p.stem for p in FULL_EMB_DIR.glob("*.npy"))
    print(f"[models] {len(models)}")
    for model in models:
        full = np.load(FULL_EMB_DIR / f"{model}.npy")
        if full.shape[0] != 8000:
            raise AssertionError(f"{model}: expected 8000 rows, got {full.shape}")
        stacked = full[sel]
        out_path = _output_path_in_dir(manifest_path, emb_dir, model)
        np.save(out_path, stacked)
        out_path.with_suffix(out_path.suffix + ".json").write_text(
            json.dumps(
                {
                    "manifest": str(dataset_dir / "embedding_source_manifest.csv"),
                    "manifest_fingerprint": emb_fp,
                    "n_samples": int(stacked.shape[0]),
                    "embedding_dim": int(stacked.shape[1]),
                    "model_id": model,
                    "extract": "precomputed",
                    "mixed_precision": False,
                    "source": "bridge_prostate_cache (sliced from tileset prostate-shift)",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"DONE. cache at {emb_dir} ({len(models)} models, {n_emb} rows each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
