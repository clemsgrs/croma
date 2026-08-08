import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import embedding_artifacts as artifacts
import model_registry as mr


def _write_manifest(path: Path) -> Path:
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "image_path": ["0.png", "1.png"],
            "label": ["A", "B"],
            "confounder": ["X", "Y"],
            "group_id": ["g0", "g1"],
        }
    ).to_csv(path, index=False)
    return path


def _contract() -> artifacts.EmbeddingArtifactContract:
    return artifacts.EmbeddingArtifactContract(
        checkpoint_revision="a" * 40,
        extraction_contract={
            "version": 1,
            "backend": "timm",
            "model_id": "owner/model",
            "extract": "cls",
            "timm_kwargs": {},
        },
        precision="float32",
        manifest_fingerprint="manifest-v1",
        batch_size=8,
        output_dtype="float32",
        output_shape=(2, 2),
    )


def test_failed_publication_cannot_look_complete_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "model.npy"
    sidecar = output.with_suffix(".npy.json")
    real_replace = os.replace

    def fail_sidecar_commit(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == sidecar:
            raise OSError("simulated crash before completion marker")
        real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_sidecar_commit)

    with pytest.raises(OSError, match="simulated crash"):
        artifacts.publish_embedding_artifact(
            output,
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            _contract(),
        )

    assert output.exists()
    assert not sidecar.exists()
    with pytest.raises(artifacts.ArtifactCompatibilityError, match="incomplete"):
        artifacts.artifact_is_reusable(output, _contract())


def test_failed_durability_sync_removes_the_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "model.npy"
    metadata = output.with_suffix(".npy.json")
    sync_count = 0

    def fail_final_sync(path: Path) -> None:
        nonlocal sync_count
        sync_count += 1
        if sync_count == 3:
            raise OSError("simulated directory fsync failure")

    monkeypatch.setattr(artifacts, "_fsync_directory", fail_final_sync)

    with pytest.raises(OSError, match="fsync failure"):
        artifacts.publish_embedding_artifact(
            output,
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            _contract(),
        )

    assert output.exists()
    assert not metadata.exists()
    with pytest.raises(artifacts.ArtifactCompatibilityError, match="incomplete"):
        artifacts.artifact_is_reusable(output, _contract())


@pytest.mark.parametrize("orphan", ["matrix", "sidecar"])
def test_orphaned_artifact_members_are_rejected(tmp_path: Path, orphan: str) -> None:
    output = tmp_path / "model.npy"
    sidecar = output.with_suffix(".npy.json")
    if orphan == "matrix":
        np.save(output, np.zeros((2, 2), dtype=np.float32))
    else:
        sidecar.write_text("{}\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactCompatibilityError, match="incomplete"):
        artifacts.artifact_is_reusable(output, _contract())


def test_compatible_artifact_is_reusable_without_modification(tmp_path: Path) -> None:
    output = tmp_path / "model.npy"
    sidecar = output.with_suffix(".npy.json")
    artifacts.publish_embedding_artifact(
        output,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        _contract(),
    )
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output, sidecar)}

    assert artifacts.artifact_is_reusable(output, _contract()) is True
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output, sidecar)
    } == before


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("checkpoint_revision", "b" * 40),
        (
            "extraction_contract",
            {
                "version": 1,
                "backend": "timm",
                "model_id": "owner/model",
                "extract": "cls_and_patch",
                "timm_kwargs": {},
            },
        ),
        ("precision", "mixed-float16"),
        ("manifest_fingerprint", "manifest-v2"),
        ("batch_size", 16),
        ("output_dtype", "float64"),
        ("output_shape", (2, 3)),
    ],
)
def test_incompatible_artifact_is_rejected_for_reuse(
    tmp_path: Path, field: str, replacement: object
) -> None:
    output = tmp_path / "model.npy"
    artifacts.publish_embedding_artifact(
        output,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        _contract(),
    )
    changed = dataclasses.replace(_contract(), **{field: replacement})

    with pytest.raises(artifacts.ArtifactCompatibilityError, match=field):
        artifacts.artifact_is_reusable(output, changed)


def test_published_sidecar_records_the_complete_provenance_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "model.npy"
    artifacts.publish_embedding_artifact(
        output,
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        _contract(),
    )

    assert output.with_suffix(".npy.json").read_text(encoding="utf-8") == (
        "{\n"
        '  "checkpoint_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",\n'
        '  "extraction_contract": {\n'
        '    "version": 1,\n'
        '    "backend": "timm",\n'
        '    "model_id": "owner/model",\n'
        '    "extract": "cls",\n'
        '    "timm_kwargs": {}\n'
        "  },\n"
        '  "precision": "float32",\n'
        '  "manifest_fingerprint": "manifest-v1",\n'
        '  "batch_size": 8,\n'
        '  "output_dtype": "float32",\n'
        '  "output_shape": [\n'
        "    2,\n"
        "    2\n"
        "  ]\n"
        "}\n"
    )


def test_model_and_manifest_define_the_expected_artifact_contract(
    tmp_path: Path,
    extraction_module,
) -> None:
    ee = extraction_module
    manifest_path = _write_manifest(tmp_path / "manifest.csv")
    spec = mr.ModelSpec(
        backend="timm",
        model_id="owner/model",
        checkpoint_revision="a" * 40,
        embedding_dim=2,
        extract="cls",
        timm_kwargs={"dynamic_img_size": True},
        mixed_precision=True,
    )

    contract = ee.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=8,
        device_arg="cuda",
    )

    assert contract == artifacts.EmbeddingArtifactContract(
        checkpoint_revision="a" * 40,
        extraction_contract={
            "version": 1,
            "backend": "timm",
            "model_id": "owner/model",
            "extract": "cls",
            "timm_kwargs": {"dynamic_img_size": True},
        },
        precision="mixed-float16",
        manifest_fingerprint="a0f6ad0283bf29e08eff11bb730ca1e790f6cb74faf825d195631a0a44ac989b",
        batch_size=8,
        output_dtype="float32",
        output_shape=(2, 2),
    )


def test_unpinned_cpu_extraction_records_honest_revision_and_precision(
    tmp_path: Path,
    extraction_module,
) -> None:
    ee = extraction_module
    manifest_path = _write_manifest(tmp_path / "manifest.csv")
    spec = mr.ModelSpec(
        backend="timm",
        model_id="owner/model",
        mixed_precision=True,
    )

    contract = ee.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=8,
        device_arg="cpu",
    )

    assert contract.checkpoint_revision is None
    assert contract.precision == "float32"
    assert contract.output_shape == (2, None)


def _write_unpinned_uni_artifact(ee, batch_size: int) -> tuple[Path, Path]:
    manifest_path = ee.layout.tileset_manifest("toy")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path)
    spec = ee._resolve_specs(["UNI"])[0][1]
    expected = ee.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=batch_size,
        device_arg="cpu",
    )
    assert expected.checkpoint_revision is None
    output = ee.layout.embedding_path("toy", "UNI")
    ee.publish_embedding_artifact(
        output,
        np.zeros((2, 2), dtype=np.float32),
        dataclasses.replace(expected, output_shape=(2, 2)),
    )
    return output, output.with_suffix(".npy.json")


def test_extraction_cli_rejects_incompatible_resume(
    bench_env, monkeypatch: pytest.MonkeyPatch, extraction_module
) -> None:
    ee = extraction_module
    _write_unpinned_uni_artifact(ee, batch_size=8)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_embeddings.py",
            "--tileset",
            "toy",
            "--models",
            "UNI",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--progress",
            "off",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        ee.main()

    assert excinfo.value.code == 1


def test_extraction_cli_skips_compatible_artifact_without_modification(
    bench_env, monkeypatch: pytest.MonkeyPatch, extraction_module
) -> None:
    ee = extraction_module
    output, metadata = _write_unpinned_uni_artifact(ee, batch_size=8)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output, metadata)}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_embeddings.py",
            "--tileset",
            "toy",
            "--models",
            "UNI",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--progress",
            "off",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        ee.main()

    assert excinfo.value.code == 0
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (output, metadata)
    } == before
