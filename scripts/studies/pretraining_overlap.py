"""Per-sample CRoMa over the *full* Tolkach-ESCA tileset, for the pretraining-overlap study.

Tolkach-ESCA mixes one TCGA cohort (``VALSET3_TCGA``) with three non-TCGA cohorts.
Comparing per-sample CRoMa on the TCGA cohort against the rest is a within-dataset test
for a pretraining-domain-overlap advantage (see ``generate_pretraining_overlap_table.py``,
which consumes this study's output).

This reads the *whole* tileset -- every tile of all four cohorts, 16,300 rows -- not the
RI eval view (which subsets rows per PathoROB's schedule). The row set therefore differs
from the ``pathorob-tolkach-esca`` benchmark, which is why this lives in ``scripts/studies``
and reads the tileset matrix directly, exactly as ``scripts/studies/apd/loaders.py`` does:
the full ``<Model>.npy`` matrix aligned row-for-row to the all-tiles manifest.

CRoMa is computed ``dataset_wide`` at ``m = CROMA_HEADLINE_M`` with the confounder column
``medical_center`` (the cohort) and auto neighbour search (no fixed tau). The emitted
``croma_m{M}`` column is the bounded CRoMa margin in (-1, 1) -- NOT the legacy typed-distance
ratio.

Run: python scripts/studies/pretraining_overlap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root autodetected: scripts/studies/pretraining_overlap.py -> parents[2] is the root.
REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout  # noqa: E402  (on-disk output layout: output/embeddings/<tileset>/...)

from croma import CRoMa  # noqa: E402
from croma.metrics.base import EVALUATION_DESIGN_DATASET_WIDE  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
from croma.metrics.pairs import load_manifest  # noqa: E402

TILESET = "pathorob-tolkach-esca"
# The all-tiles manifest: every tile of all four cohorts (incl. VALSET3_TCGA), row-aligned
# to the tileset's <Model>.npy matrices.
MANIFEST = REPO / "data/pathorob/manifests/pathorob-tolkach-esca.csv"
CONFOUNDER_COLUMN = "medical_center"
HEADLINE_COL = f"croma_m{int(CROMA_HEADLINE_M)}"
OUT_DIR = REPO / "output/studies/pretraining-overlap"
OUT_CSV = OUT_DIR / "per_sample_metrics.csv"


def _models(tileset: str) -> list[str]:
    """Every model with a ``.npy`` in the tileset, sorted for determinism."""
    return sorted(p.stem for p in layout.embeddings_dir(tileset).glob("*.npy"))


def _per_sample(model: str, manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-sample CRoMa(m=CROMA_HEADLINE_M) for one model over the full tileset."""
    emb = np.load(layout.embedding_path(TILESET, model))
    if len(emb) != len(manifest):
        raise RuntimeError(
            f"{model}: {len(emb)} embedding rows vs {len(manifest)} manifest rows"
        )

    result = CRoMa.compute(
        features=np.asarray(emb, dtype=float),
        manifest=manifest,
        confounder_column=CONFOUNDER_COLUMN,
        evaluation_design=EVALUATION_DESIGN_DATASET_WIDE,
        m=int(CROMA_HEADLINE_M),
    )

    # dataset_wide -> occurrences are the samples, in manifest row order. Align each
    # per-occurrence CRoMa value back to its source row via occurrence_source_indices.
    src = np.asarray(result.occurrence_source_indices, dtype=int)
    vals = np.asarray(result.sample_values_aligned, dtype=float)  # NaN where undefined
    rows = manifest.iloc[src]
    return pd.DataFrame(
        {
            "model": model,
            "confounder": rows["confounder"].to_numpy(),
            "group_id": rows["group_id"].to_numpy(),
            "label": rows["label"].to_numpy(),
            "sample_id": rows["sample_id"].to_numpy(),
            HEADLINE_COL: vals,
        }
    )


def build() -> pd.DataFrame:
    manifest = load_manifest(str(MANIFEST), confounder_column=CONFOUNDER_COLUMN)
    frames = []
    for model in _models(TILESET):
        print(f"[pretraining-overlap] {model}", flush=True)
        frames.append(_per_sample(model, manifest))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows, {df['model'].nunique()} models)")


if __name__ == "__main__":
    main()
