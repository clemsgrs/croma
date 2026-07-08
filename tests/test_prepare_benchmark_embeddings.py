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

import benchmark as bm
import prepare_benchmark_embeddings as pbe


def _repeated_subset_manifest() -> pd.DataFrame:
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


def _write_npz(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embedding=values)


def test_prepare_benchmark_embeddings_writes_benchmark_cache_files(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "toy.csv"
    mapping_path = tmp_path / "mapping.csv"
    output_dir = tmp_path / "out"
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
        output_dir=output_dir,
        models=["UNI", "Virchow2"],
        evaluation_design="paired_2x2",
    )

    dataset_dir = output_dir / manifest_path.stem
    embedding_manifest_path = dataset_dir / "embedding_source_manifest.csv"
    embedding_manifest = pd.read_csv(embedding_manifest_path, dtype=str)
    assert embedding_manifest["image_path"].tolist() == image_order
    assert summary["embedding_manifest_rows"] == 4
    assert summary["models"] == ["UNI", "Virchow2"]

    uni_path = dataset_dir / "embeddings" / "UNI.npy"
    virchow_path = dataset_dir / "embeddings" / "Virchow2.npy"
    np.testing.assert_allclose(
        np.load(uni_path),
        np.asarray(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        np.load(virchow_path),
        np.asarray(
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]],
            dtype=np.float32,
        ),
    )

    manifest_fp = bm.manifest_fingerprint(embedding_manifest)
    assert bm._embedding_cache_matches_expected(
        uni_path,
        expected_n_samples=4,
        expected_manifest_fingerprint=manifest_fp,
        expected_manifest_path=embedding_manifest_path,
    )
    assert bm._embedding_cache_matches_expected(
        virchow_path,
        expected_n_samples=4,
        expected_manifest_fingerprint=manifest_fp,
        expected_manifest_path=embedding_manifest_path,
    )

    uni_sidecar = json.loads(uni_path.with_suffix(".npy.json").read_text(encoding="utf-8"))
    assert uni_sidecar["extract"] == "precomputed"
    assert uni_sidecar["model_id"] == "UNI"


def test_prepare_benchmark_embeddings_rejects_missing_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    mapping_path = tmp_path / "mapping.csv"
    output_dir = tmp_path / "out"
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

    with pytest.raises(
        ValueError, match="missing embeddings for model 'UNI'"
    ):
        pbe.prepare_benchmark_embeddings(
            manifest_path=manifest_path,
            confounder_column="scanner_vendor",
            mapping_csv=mapping_path,
            output_dir=output_dir,
            models=["UNI"],
            evaluation_design="paired_2x2",
        )
