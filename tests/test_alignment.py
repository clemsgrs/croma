from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from croma import expand_features_to_manifest as top_level_expand_features_to_manifest
from croma.alignment import (
    build_embedding_source_manifest,
    expand_features_to_manifest,
)
from croma.metrics.pairs import load_manifest


def _write_repeated_subset_manifest(path: Path) -> pd.DataFrame:
    rows = [
        ("s0", "/tmp/0.png", "A", "VendorA", "sl0"),
        ("s1", "/tmp/1.png", "A", "VendorB", "sl1"),
        ("s2", "/tmp/2.png", "B", "VendorA", "sl2"),
        ("s3", "/tmp/3.png", "B", "VendorB", "sl3"),
    ]
    payload = ["sample_id,image_path,label,scanner_vendor,slide_id,subset"]
    for subset in ("pair1", "pair2"):
        for sample_id, image_path, label, confounder, slide_id in rows:
            payload.append(",".join([sample_id, image_path, label, confounder, slide_id, subset]))
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")
    return pd.read_csv(path, dtype=str)


def test_expand_features_to_manifest_repeats_deduplicated_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    _write_repeated_subset_manifest(manifest_path)
    manifest = load_manifest(str(manifest_path), confounder_column="scanner_vendor")
    embedding_manifest, _ = build_embedding_source_manifest(manifest)
    features = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=float,
    )

    expanded = expand_features_to_manifest(
        features=features,
        manifest=manifest,
        embedding_manifest=embedding_manifest,
    )

    expected = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(expanded, expected)


def test_expand_features_to_manifest_rejects_length_mismatch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    _write_repeated_subset_manifest(manifest_path)
    manifest = load_manifest(str(manifest_path), confounder_column="scanner_vendor")
    embedding_manifest, _ = build_embedding_source_manifest(manifest)
    features = np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="embeddings rows must match embedding manifest rows"):
        expand_features_to_manifest(
            features=features,
            manifest=manifest,
            embedding_manifest=embedding_manifest,
        )


def test_top_level_expand_features_helper_is_exposed() -> None:
    assert top_level_expand_features_to_manifest is expand_features_to_manifest


def test_build_embedding_source_manifest_tracks_selected_confounder(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _write_repeated_subset_manifest(manifest_path)
    manifest = load_manifest(str(manifest_path), confounder_column="scanner_vendor")

    embedding_manifest, _ = build_embedding_source_manifest(manifest)

    assert embedding_manifest.columns.tolist() == [
        "sample_id",
        "image_path",
        "label",
        "confounder",
        "slide_id",
    ]
    assert embedding_manifest["confounder"].tolist() == [
        "VendorA",
        "VendorB",
        "VendorA",
        "VendorB",
    ]
