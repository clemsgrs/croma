import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "prep"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_paired_manifest as ppm
from croma.metrics.pairs import load_manifest, validate_subset_manifest


def _balanced_manifest() -> pd.DataFrame:
    labels = ["A", "B", "C"]
    confounders = ["X", "Y", "Z"]
    rows: list[dict[str, str]] = []
    for label in labels:
        for confounder in confounders:
            sample_id = f"{label}_{confounder}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"/tmp/{sample_id}.png",
                    "label": label,
                    "scanner_vendor": confounder,
                    "slide_id": f"slide_{sample_id}",
                }
            )
    return pd.DataFrame(rows)


def test_prepare_paired_manifest_builds_all_label_and_confounder_pairs(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "toy.csv"
    out_path = tmp_path / "toy_paired.csv"
    _balanced_manifest().to_csv(manifest_path, index=False)

    summary = ppm.prepare_paired_manifest(
        manifest_path=manifest_path,
        confounder_column="scanner_vendor",
        out_path=out_path,
    )

    out_df = pd.read_csv(out_path, dtype=str)
    normalized = load_manifest(str(out_path), confounder_column="scanner_vendor")
    validate_subset_manifest(normalized, str(out_path))

    assert len(out_df) == 36
    assert out_df["sample_id"].nunique() == 9
    assert sorted(out_df.groupby("sample_id").size().unique().tolist()) == [4]
    assert out_df["subset"].nunique() == 9
    assert "A+B__X_Y" in set(out_df["subset"])
    assert summary["subset_count"] == 9
    assert summary["is_globally_balanced"] is True
    assert summary["labels_used"] == ["A", "B", "C"]
    assert summary["confounders_used"] == ["X", "Y", "Z"]
    assert summary["strata"] == [
        {"label": label, "confounder": confounder, "n_samples": 1}
        for label in ["A", "B", "C"]
        for confounder in ["X", "Y", "Z"]
    ]


def test_prepare_paired_manifest_can_filter_labels(tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    out_path = tmp_path / "toy_filtered.csv"
    _balanced_manifest().to_csv(manifest_path, index=False)

    summary = ppm.prepare_paired_manifest(
        manifest_path=manifest_path,
        confounder_column="scanner_vendor",
        out_path=out_path,
        labels=["A", "C"],
    )

    out_df = pd.read_csv(out_path, dtype=str)
    normalized = load_manifest(str(out_path), confounder_column="scanner_vendor")
    validate_subset_manifest(normalized, str(out_path))

    assert set(out_df["label"]) == {"A", "C"}
    assert out_df["subset"].nunique() == 3
    assert set(out_df["subset"]) == {"X_Y", "X_Z", "Y_Z"}
    assert len(out_df) == 12
    assert summary["subset_count"] == 3
    assert summary["labels_used"] == ["A", "C"]


def test_prepare_paired_manifest_cli_prints_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "toy.csv"
    out_path = tmp_path / "toy_paired.csv"
    _balanced_manifest().to_csv(manifest_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_paired_manifest.py",
            "--manifest",
            str(manifest_path),
            "--confounder-column",
            "scanner_vendor",
            "--labels",
            "A,C",
            "--out",
            str(out_path),
        ],
    )

    assert ppm.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["paired_manifest"] == str(out_path)
    assert payload["subset_count"] == 3


def test_prepare_paired_manifest_rejects_less_than_two_labels(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "toy.csv"
    out_path = tmp_path / "toy_paired.csv"
    _balanced_manifest().to_csv(manifest_path, index=False)

    with pytest.raises(ValueError, match="at least 2 labels"):
        ppm.prepare_paired_manifest(
            manifest_path=manifest_path,
            confounder_column="scanner_vendor",
            out_path=out_path,
            labels=["A"],
        )
