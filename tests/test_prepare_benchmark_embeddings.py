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


def _write_slide2vec_run(
    run_dir: Path, vectors: dict[str, np.ndarray], *, version: str
) -> dict[str, Path]:
    """A slide2vec run as the launcher and slide2vec 7.1.4 leave it."""
    torch = pytest.importorskip("torch")
    import yaml

    (run_dir / "slide_embeddings").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"name": "gigapath-slide"},
                "tiling": {"params": {"requested_spacing_um": 0.5, "requested_tile_size_px": 224}},
            }
        )
    )
    (run_dir / "run_record.json").write_text(
        json.dumps({"slide2vec_version": version, "slide_encoder_eval_mode": True})
    )
    paths = {}
    for sid, vector in vectors.items():
        path = run_dir / "slide_embeddings" / f"{sid}.pt"
        torch.save(torch.as_tensor(vector), path)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"sample_id": sid, "compatibility": {"aggregation_revision": 1}})
        )
        paths[sid] = path
    return paths


def test_sidecar_records_the_slide2vec_runs_behind_a_model(bench_env, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    mapping_path = tmp_path / "mapping.csv"
    _repeated_subset_manifest().to_csv(manifest_path, index=False)
    image_order = ["/tmp/s0.png", "/tmp/s1.png", "/tmp/s2.png", "/tmp/s3.png"]

    # Prov-GigaPath's rows come from two runs (as a tileset spanning two cohorts would);
    # UNI's rows are plain per-image arrays, not a slide2vec run.
    run_a = _write_slide2vec_run(
        tmp_path / "features" / "gigapath-a",
        {"s0": np.ones(3, np.float32), "s1": np.ones(3, np.float32)},
        version="7.1.4",
    )
    run_b = _write_slide2vec_run(
        tmp_path / "features" / "gigapath-b",
        {"s2": np.ones(3, np.float32), "s3": np.ones(3, np.float32)},
        version="7.1.4",
    )
    gigapath_paths = {**run_a, **run_b}
    mapping_rows = []
    for image_path in image_order:
        sid = Path(image_path).stem
        mapping_rows.append(
            {
                "image_path": image_path,
                "model": "Prov-GigaPath",
                "embedding_path": str(gigapath_paths[sid]),
            }
        )
        uni_path = tmp_path / "uni" / f"{sid}.npz"
        _write_npz(uni_path, np.zeros((1, 2), np.float32))
        mapping_rows.append(
            {"image_path": image_path, "model": "UNI", "embedding_path": str(uni_path)}
        )
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    pbe.prepare_benchmark_embeddings(
        manifest_path=manifest_path,
        confounder_column="scanner_vendor",
        mapping_csv=mapping_path,
        tileset="toy-slides",
        models=["Prov-GigaPath", "UNI"],
        evaluation_design="paired_2x2",
    )

    def _sidecar(model):
        path = layout.embedding_path("toy-slides", model)
        return json.loads(path.with_suffix(".npy.json").read_text(encoding="utf-8"))

    common = {
        "slide2vec_version": "7.1.4",
        "encoder_preset": "gigapath-slide",
        "tile_size_px": 224,
        "spacing_um": 0.5,
        "aggregation_revision": 1,
        "slide_encoder_eval_mode": True,
        "n_samples": 2,
    }
    # Two runs supplied equally, so the first used is the main one; the other names the
    # rows it supplied and how its config.yaml differs (here: not at all).
    assert _sidecar("Prov-GigaPath")["source_runs"] == [
        {"run_id": "gigapath-a", "run_dir": str(tmp_path / "features" / "gigapath-a"), **common},
        {
            "run_id": "gigapath-b",
            "run_dir": str(tmp_path / "features" / "gigapath-b"),
            **common,
            "sample_ids": ["s2", "s3"],
            "config_vs_main_run": {"main_run_id": "gigapath-a", "differences": {}},
        },
    ]
    assert "source_runs" not in _sidecar("UNI")


def test_rejected_provenance_leaves_the_previous_matrix_in_place(bench_env, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    mapping_path = tmp_path / "mapping.csv"
    _repeated_subset_manifest().to_csv(manifest_path, index=False)
    paths = _write_slide2vec_run(
        tmp_path / "features" / "gigapath",
        {f"s{i}": np.full(3, float(i), np.float32) for i in range(4)},
        version="7.1.4",
    )
    pd.DataFrame(
        {
            "image_path": [f"/tmp/{sid}.png" for sid in paths],
            "model": "Prov-GigaPath",
            "embedding_path": [str(path) for path in paths.values()],
        }
    ).to_csv(mapping_path, index=False)
    kwargs = dict(
        manifest_path=manifest_path,
        confounder_column="scanner_vendor",
        mapping_csv=mapping_path,
        tileset="toy-slides",
        models=["Prov-GigaPath"],
        evaluation_design="paired_2x2",
    )
    pbe.prepare_benchmark_embeddings(**kwargs)
    matrix_path = layout.embedding_path("toy-slides", "Prov-GigaPath")
    before = matrix_path.read_bytes()

    # One slide is re-embedded under another aggregation revision: the run's
    # provenance is now contradictory, so the rebuild must refuse before writing.
    import torch

    torch.save(torch.full((3,), 9.0), paths["s3"])
    paths["s3"].with_suffix(".meta.json").write_text(
        json.dumps({"sample_id": "s3", "compatibility": {"aggregation_revision": 2}})
    )
    with pytest.raises(ValueError, match="disagree on aggregation_revision"):
        pbe.prepare_benchmark_embeddings(**kwargs)
    assert matrix_path.read_bytes() == before


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
