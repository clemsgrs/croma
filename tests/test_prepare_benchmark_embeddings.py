import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "bench", ROOT / "scripts" / "prep"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout
import prepare_benchmark_embeddings as pbe
from input_fingerprint import manifest_fingerprint


def _repeated_subset_manifest() -> pd.DataFrame:
    base_rows = [
        ("s0", "/tmp/s0.png", "A", "V1", "sl0"),
        ("s1", "/tmp/s1.png", "A", "V2", "sl1"),
        ("s2", "/tmp/s2.png", "B", "V1", "sl2"),
        ("s3", "/tmp/s3.png", "B", "V2", "sl3"),
    ]
    rows: list[dict[str, str]] = []
    for subset in ("pair1", "pair2"):
        for sample_id, image_path, label, confounder, group_id in base_rows:
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "label": label,
                    "scanner_vendor": confounder,
                    "group_id": group_id,
                    "subset": subset,
                    "dataset": "toy",
                }
            )
    return pd.DataFrame(rows)


def _write_npz(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embedding=values)


def test_prepare_benchmark_embeddings_writes_tileset(bench_env, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    mapping_path = tmp_path / "mapping.csv"
    _repeated_subset_manifest().to_csv(manifest_path, index=False)

    image_order = ["/tmp/s0.png", "/tmp/s1.png", "/tmp/s2.png", "/tmp/s3.png"]
    uni_values = [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        np.asarray([[3.0, 0.0]], dtype=np.float32),
        np.asarray([[4.0, 0.0]], dtype=np.float32),
    ]
    virchow_values = [
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        np.asarray([[0.0, 2.0]], dtype=np.float32),
        np.asarray([[0.0, 3.0]], dtype=np.float32),
        np.asarray([[0.0, 4.0]], dtype=np.float32),
    ]
    mapping_rows: list[dict[str, str]] = []
    for model, values_by_image in (("UNI", uni_values), ("Virchow2", virchow_values)):
        for image_path, values in zip(image_order, values_by_image, strict=True):
            embedding_path = tmp_path / "embeddings" / model / f"{Path(image_path).stem}.npz"
            _write_npz(embedding_path, values)
            mapping_rows.append(
                {
                    "image_path": image_path,
                    "model": model,
                    "embedding_path": str(embedding_path),
                }
            )
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    summary = pbe.prepare_benchmark_embeddings(
        manifest_path=manifest_path,
        confounder_column="scanner_vendor",
        mapping_csv=mapping_path,
        tileset="toy-tiles",
        models=["UNI", "Virchow2"],
        evaluation_design="paired_2x2",
    )

    tileset_dir = layout.embeddings_dir("toy-tiles")
    tileset_manifest = pd.read_csv(layout.tileset_manifest("toy-tiles"), dtype=str)
    # The tileset manifest holds one row per distinct tile, in embedding row order.
    assert tileset_manifest["image_path"].tolist() == image_order
    assert summary["embedding_manifest_rows"] == 4
    assert summary["models"] == ["UNI", "Virchow2"]
    assert summary["tileset"] == "toy-tiles"

    uni_path = layout.embedding_path("toy-tiles", "UNI")
    virchow_path = layout.embedding_path("toy-tiles", "Virchow2")
    assert uni_path == tileset_dir / "UNI.npy"
    np.testing.assert_allclose(
        np.load(uni_path),
        np.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.load(virchow_path),
        np.asarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]], dtype=np.float32),
    )

    # The sidecar pins each matrix to the tileset manifest's row-order contract.
    manifest_fp = manifest_fingerprint(tileset_manifest)
    for path, model in ((uni_path, "UNI"), (virchow_path, "Virchow2")):
        sidecar = json.loads(path.with_suffix(".npy.json").read_text(encoding="utf-8"))
        assert sidecar["extract"] == "precomputed"
        assert sidecar["model_id"] == model
        assert sidecar["n_samples"] == 4
        assert sidecar["manifest_fingerprint"] == manifest_fp
        assert sidecar["manifest"] == str(layout.tileset_manifest("toy-tiles"))


def test_prepare_benchmark_embeddings_rejects_missing_rows(bench_env, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    mapping_path = tmp_path / "mapping.csv"
    _repeated_subset_manifest().to_csv(manifest_path, index=False)

    rows = [
        {
            "image_path": "/tmp/s0.png",
            "model": "UNI",
            "embedding_path": str(tmp_path / "s0.npz"),
        },
        {
            "image_path": "/tmp/s1.png",
            "model": "UNI",
            "embedding_path": str(tmp_path / "s1.npz"),
        },
        {
            "image_path": "/tmp/s2.png",
            "model": "UNI",
            "embedding_path": str(tmp_path / "s2.npz"),
        },
    ]
    for idx in range(3):
        _write_npz(tmp_path / f"s{idx}.npz", np.asarray([[float(idx), 0.0]], dtype=np.float32))
    pd.DataFrame(rows).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match="missing embeddings for model 'UNI'"):
        pbe.prepare_benchmark_embeddings(
            manifest_path=manifest_path,
            confounder_column="scanner_vendor",
            mapping_csv=mapping_path,
            tileset="toy-tiles",
            models=["UNI"],
            evaluation_design="paired_2x2",
        )
