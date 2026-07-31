import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import croma.cli as cli
from croma.types import CRoMaResult


class _FakeResult:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.k = 1
        self.value = 0.5
        self.std = 0.0
        self.n_pairs = 1
        self.undefined_frac = 0.0
        self.evaluation_design = "all"
        self.evaluation_unit = "sample"


def _write_manifest(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "sample_id,image_path,label,scanner_vendor,group_id,dataset",
                "s0,/tmp/0.png,A,VendorA,sl0,wrong_name",
                "s1,/tmp/1.png,B,VendorB,sl1,wrong_name",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_repeated_subset_manifest(path: Path) -> None:
    rows = [
        ("s0", "/tmp/0.png", "A", "VendorA", "sl0"),
        ("s1", "/tmp/1.png", "A", "VendorB", "sl1"),
        ("s2", "/tmp/2.png", "B", "VendorA", "sl2"),
        ("s3", "/tmp/3.png", "B", "VendorB", "sl3"),
    ]
    payload = [
        "sample_id,image_path,label,scanner_vendor,group_id,subset,dataset",
    ]
    for subset in ("pair1", "pair2"):
        for sample_id, image_path, label, confounder, group_id in rows:
            payload.append(
                ",".join(
                    [
                        sample_id,
                        image_path,
                        label,
                        confounder,
                        group_id,
                        subset,
                        "wrong_name",
                    ]
                )
            )
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def _write_embedding_manifest(
    path: Path, *, duplicate_first_row: bool = False, drop_last_row: bool = False
) -> None:
    rows = [
        ("s0", "/tmp/0.png", "A", "VendorA", "sl0"),
        ("s1", "/tmp/1.png", "A", "VendorB", "sl1"),
        ("s2", "/tmp/2.png", "B", "VendorA", "sl2"),
        ("s3", "/tmp/3.png", "B", "VendorB", "sl3"),
    ]
    if drop_last_row:
        rows = rows[:-1]
    if duplicate_first_row:
        rows = [rows[0], *rows]
    path.write_text(
        "\n".join(
            [
                "sample_id,image_path,label,confounder,group_id",
                *[",".join(row) for row in rows],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_cli_uses_manifest_stem_for_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "camelyon.csv"
    embeddings_path = tmp_path / "embeddings.npy"
    _write_manifest(manifest_path)
    np.save(embeddings_path, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float))

    def fake_compute(*, manifest, **kwargs):
        assert manifest["dataset"].tolist() == ["camelyon", "camelyon"]
        return _FakeResult(dataset=str(manifest["dataset"].iloc[0]))

    monkeypatch.setattr(cli.RI, "compute", fake_compute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "ri",
            "--manifest",
            str(manifest_path),
            "--embeddings",
            str(embeddings_path),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "all",
            "--k-candidates",
            "1",
        ],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "camelyon"


def test_cli_builds_deduplicated_embedding_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "toy.csv"
    out_path = tmp_path / "toy.embedding_sources.csv"
    _write_repeated_subset_manifest(manifest_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "build-embedding-manifest",
            "--manifest",
            str(manifest_path),
            "--confounder-column",
            "scanner_vendor",
            "--out",
            str(out_path),
        ],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    built = pd.read_csv(out_path, dtype=str)
    assert payload["manifest_rows"] == 8
    assert payload["embedding_manifest_rows"] == 4
    assert built.columns.tolist() == [
        "sample_id",
        "image_path",
        "label",
        "confounder",
        "group_id",
    ]
    assert built["sample_id"].tolist() == ["s0", "s1", "s2", "s3"]


def test_cli_expand_embeddings_writes_manifest_aligned_npy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "toy.csv"
    embedding_manifest_path = tmp_path / "toy.embedding_sources.csv"
    embeddings_path = tmp_path / "embeddings.npy"
    out_path = tmp_path / "expanded.npy"
    _write_repeated_subset_manifest(manifest_path)
    _write_embedding_manifest(embedding_manifest_path)
    np.save(
        embeddings_path,
        np.asarray(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=float,
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "expand-embeddings",
            "--manifest",
            str(manifest_path),
            "--confounder-column",
            "scanner_vendor",
            "--embedding-manifest",
            str(embedding_manifest_path),
            "--embeddings",
            str(embeddings_path),
            "--out",
            str(out_path),
        ],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    expanded = np.load(out_path)
    assert payload["manifest_rows"] == 8
    assert payload["embedding_manifest_rows"] == 4
    assert tuple(expanded.shape) == (8, 2)
    np.testing.assert_allclose(
        expanded,
        np.asarray(
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
        ),
    )


def test_cli_metric_commands_fail_fast_on_unaligned_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    embeddings_path = tmp_path / "embeddings.npy"
    _write_repeated_subset_manifest(manifest_path)
    np.save(
        embeddings_path,
        np.asarray(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ],
            dtype=float,
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "ri",
            "--manifest",
            str(manifest_path),
            "--embeddings",
            str(embeddings_path),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "paired_2x2",
            "--k-candidates",
            "1",
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        cli.main()

    message = str(excinfo.value)
    assert "embeddings rows must match manifest rows" in message
    assert "manifest-aligned embeddings" in message
    assert "build-embedding-manifest" in message
    assert "expand-embeddings" in message


@pytest.mark.parametrize(
    "argv",
    [
        [
            "cli.py",
            "ri",
            "--manifest",
            "m.csv",
            "--embeddings",
            "e.npy",
            "--confounder-column",
            "confounder",
            "--evaluation-design",
            "all",
            "--dataset-name",
            "toy",
        ],
        [
            "cli.py",
            "ri",
            "--manifest",
            "m.csv",
            "--embeddings",
            "e.npy",
            "--confounder-column",
            "confounder",
            "--evaluation-design",
            "all",
            "--exclude-confounder",
            "C1",
        ],
        [
            "cli.py",
            "ri",
            "--manifest",
            "m.csv",
            "--embedding-manifest",
            "m.embed.csv",
            "--embeddings",
            "e.npy",
            "--confounder-column",
            "confounder",
            "--evaluation-design",
            "all",
        ],
        [
            "cli.py",
            "croma",
            "--manifest",
            "m.csv",
            "--embeddings",
            "e.npy",
            "--confounder-column",
            "confounder",
            "--evaluation-design",
            "all",
            "--acceptance-threshold",
            "0.5",
        ],
    ],
)
def test_cli_rejects_removed_flags(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2


def test_cli_requires_confounder_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "ri",
            "--manifest",
            "m.csv",
            "--embeddings",
            "e.npy",
            "--evaluation-design",
            "all",
            "--k-candidates",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 2


def test_cli_croma_payload_reports_the_canonical_f0(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The JSON carries CRoMa's own ``f0``; a consumer never recomputes it."""
    manifest_path = tmp_path / "toy.csv"
    embeddings_path = tmp_path / "embeddings.npy"
    _write_manifest(manifest_path)
    np.save(embeddings_path, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float))

    def fake_compute(*, manifest, **kwargs):
        return CRoMaResult(
            dataset="toy",
            m=5,
            value=0.25,
            std=0.0,
            n_pairs=1,
            pair_values=np.asarray([0.25]),
            sample_values=np.asarray([-0.5, 0.0, 0.25]),
            sample_values_aligned=np.asarray([-0.5, 0.0, 0.25, np.nan]),
            occurrence_defined_mask=np.asarray([True, True, True, False]),
            undefined_frac=0.25,
            f0=2.0 / 3.0,
        )

    monkeypatch.setattr(cli.CRoMa, "compute", fake_compute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "croma",
            "--manifest",
            str(manifest_path),
            "--embeddings",
            str(embeddings_path),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "all",
        ],
    )

    cli.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["f0"] == pytest.approx(2.0 / 3.0)
    assert payload["undefined_frac"] == pytest.approx(0.25)


def test_cli_croma_f0_matches_a_direct_computation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end, on a real computation: the JSON's ``f0`` is the library's own."""
    manifest_path = tmp_path / "toy.csv"
    embeddings_path = tmp_path / "embeddings.npy"
    _write_repeated_subset_manifest(manifest_path)
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.98, 0.02], [0.02, 0.98]] * 2, dtype=float)
    np.save(embeddings_path, features)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "croma",
            "--manifest",
            str(manifest_path),
            "--embeddings",
            str(embeddings_path),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "paired_2x2",
            "--m",
            "1",
        ],
    )

    cli.main()
    payload = json.loads(capsys.readouterr().out)

    manifest = pd.read_csv(manifest_path)
    expected = cli.CRoMa.compute(
        features=features,
        manifest=manifest,
        confounder_column="scanner_vendor",
        evaluation_design="paired_2x2",
        m=1,
    )
    assert payload["f0"] == pytest.approx(expected.f0)
    assert payload["f0"] == pytest.approx(
        float(np.mean(expected.sample_values_aligned[expected.occurrence_defined_mask] <= 0.0))
    )
