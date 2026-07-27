"""The benchmark no longer embeds anything (embeddings are produced once per tileset).

What used to be "embed each unique source sample once" is now a pure index composition:
an eval manifest may repeat a tile across paired subsets, so the row-view it borrows from
the tileset must resolve every occurrence of a repeated tile to the *same* embedding row.
These tests pin that composition and its end-to-end effect on the per-sample artifact.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm
from croma.alignment import build_embedding_source_manifest, build_view_row_index


def _repeated_subset_manifest() -> pd.DataFrame:
    # Each tile (fixed sample_id + image_path) appears in both pair1 and pair2.
    base_rows = [
        ("s0", "/tmp/s0.png", "A", "V1", "sl0"),
        ("s1", "/tmp/s1.png", "A", "V2", "sl1"),
        ("s2", "/tmp/s2.png", "B", "V1", "sl2"),
        ("s3", "/tmp/s3.png", "B", "V2", "sl3"),
    ]
    rows: list[dict[str, str]] = []
    for subset in ("pair1", "pair2"):
        for sample_id, image_path, label, confounder, slide_id in base_rows:
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "label": label,
                    "scanner_vendor": confounder,
                    "slide_id": slide_id,
                    "subset": subset,
                    "dataset": "toy",
                }
            )
    return pd.DataFrame(rows)


def _tileset_from(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in manifest.columns if c not in ("subset", "dataset")]
    return (
        manifest.loc[:, cols]
        .drop_duplicates(subset=["sample_id", "image_path"])
        .reset_index(drop=True)
    )


def _features() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def test_repeated_tile_gathers_one_embedding_row_per_occurrence(tmp_path: Path) -> None:
    """The two-hop index composition: eval row -> unique source row -> tileset row."""
    manifest_path = tmp_path / "toy.csv"
    _repeated_subset_manifest().to_csv(manifest_path, index=False)

    manifest = bm.load_manifest(str(manifest_path), confounder_column="scanner_vendor")
    eval_manifest = bm._prepare_eval_manifest(
        manifest_df=manifest,
        dataset_name="toy",
        evaluation_design="paired_2x2",
    )
    assert len(eval_manifest) == 8

    # Hop 1: collapse repeated tiles to a per-source-row unique manifest.
    embedding_manifest, view_keep_indices = build_embedding_source_manifest(eval_manifest)
    assert len(embedding_manifest) == 4  # four distinct tiles
    assert view_keep_indices.shape == (8,)
    # s0 appears at eval rows 0 and 4; both collapse to the same unique source row.
    assert view_keep_indices[0] == view_keep_indices[4]

    # Hop 2: map each unique source row to its tileset embedding row.
    tileset_manifest = _tileset_from(_repeated_subset_manifest())
    view_to_tileset = build_view_row_index(embedding_manifest, tileset_manifest)
    assert view_to_tileset.shape == (4,)

    # Composed: one tileset embedding row per eval row, with repeats resolving equally.
    embedding_keep_indices = view_to_tileset[view_keep_indices]
    assert embedding_keep_indices.shape == (8,)
    assert embedding_keep_indices[0] == embedding_keep_indices[4]  # same s0 tile
    # Every distinct tile is represented, and each occurrence points at a valid row.
    assert set(embedding_keep_indices.tolist()) == set(range(4))


def test_benchmark_run_over_repeated_tile_emits_per_occurrence_rows(bench_env) -> None:
    manifest = _repeated_subset_manifest()
    tileset = _tileset_from(manifest)
    bench_env.write_tileset("repeat-tiles", tileset, {"M1": _features()[: len(tileset)]})
    bench_env.register(
        "toy",
        tileset="repeat-tiles",
        manifest=manifest,
        design="paired_2x2",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    per_sample_df = pd.read_csv(bench_env.results_dir("toy") / "per_sample_metrics.csv")
    assert len(per_sample_df) == 8
    assert set(per_sample_df["subset"]) == {"pair1", "pair2"}
    # The repeated tile s0 yields one per-sample row per occurrence (one per subset).
    assert int((per_sample_df["sample_id"] == "s0").sum()) == 2
