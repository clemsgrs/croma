import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_pathorob as pp


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _is_complete_2x2_subset(df: pd.DataFrame) -> bool:
    labels = sorted(df["label"].astype(str).unique().tolist())
    centers = sorted(df["medical_center"].astype(str).unique().tolist())
    if len(labels) != 2 or len(centers) != 2:
        return False
    for label in labels:
        for center in centers:
            if int(((df["label"] == label) & (df["medical_center"] == center)).sum()) <= 0:
                return False
    return True


def _alignment(output_name: str) -> pp.AlignmentSpec:
    for spec in pp.ALIGNMENTS:
        if spec.output_name == output_name:
            return spec
    raise AssertionError(f"missing alignment {output_name}")


def test_align_dataset_camelyon_reduced_emits_one_complete_2x2_subset(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    manifest_dir = tmp_path / "manifests"
    metadata_dir.mkdir()
    manifest_dir.mkdir()

    _write_csv(
        manifest_dir / "pathorob-camelyon-source.csv",
        [
            {
                "sample_id": f"s{i}",
                "image_path": f"/tmp/{i}.png",
                "label": label,
                "medical_center": center,
                "slide_id": slide,
                "patch_id": patch,
            }
            for i, (label, center, slide, patch) in enumerate(
                [
                    ("normal", "RUMC", "slide_n_r", "p0"),
                    ("normal", "UMCU", "slide_n_u", "p1"),
                    ("tumor", "RUMC", "slide_t_r", "p2"),
                    ("tumor", "UMCU", "slide_t_u", "p3"),
                ]
            )
        ],
    )
    _write_csv(
        metadata_dir / "camelyon_reduced.csv",
        [
            {
                "subset": subset,
                "slide_id": slide,
                "patch_id": patch,
                "biological_class": label,
                "medical_center": center,
            }
            for subset, label, center, slide, patch in [
                ("normal-RUMC", "normal", "RUMC", "slide_n_r", "p0"),
                ("normal-UMCU", "normal", "UMCU", "slide_n_u", "p1"),
                ("tumor-RUMC", "tumor", "RUMC", "slide_t_r", "p2"),
                ("tumor-UMCU", "tumor", "UMCU", "slide_t_u", "p3"),
            ]
        ],
    )

    output_path = pp.align_dataset(
        alignment=_alignment("pathorob-camelyon-reduced"),
        metadata_dir=metadata_dir,
        manifest_dir=manifest_dir,
        progress_on=False,
    )

    out_df = pd.read_csv(output_path, dtype=str)
    assert len(out_df) == 4
    assert out_df["sample_id"].nunique() == 4
    assert out_df["subset"].nunique() == 1
    assert set(out_df["subset"]) == {"RUMC_UMCU"}
    assert _is_complete_2x2_subset(out_df)


def test_align_dataset_tolkach_reduced_expands_cell_buckets_into_quartets(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    manifest_dir = tmp_path / "manifests"
    metadata_dir.mkdir()
    manifest_dir.mkdir()

    labels = ["A", "B", "C"]
    centers = ["X", "Y", "Z"]

    source_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []
    for label in labels:
        for center in centers:
            slide = f"{label}_{center}_slide"
            patch = f"{label}_{center}_patch"
            sample_id = f"{slide}__{patch}"
            source_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"/tmp/{sample_id}.png",
                    "label": label,
                    "medical_center": center,
                    "slide_id": slide,
                    "patch_id": patch,
                }
            )
            metadata_rows.append(
                {
                    "subset": f"{label}-{center}",
                    "slide_id": slide,
                    "patch_id": patch,
                    "biological_class": label,
                    "medical_center": center,
                }
            )

    _write_csv(manifest_dir / "pathorob-tolkach_esca-source.csv", source_rows)
    _write_csv(metadata_dir / "tolkach_esca_reduced.csv", metadata_rows)

    output_path = pp.align_dataset(
        alignment=_alignment("pathorob-tolkach-esca-reduced"),
        metadata_dir=metadata_dir,
        manifest_dir=manifest_dir,
        progress_on=False,
    )

    out_df = pd.read_csv(output_path, dtype=str)
    assert len(out_df) == 36
    assert out_df["sample_id"].nunique() == 9
    assert sorted(out_df.groupby("sample_id").size().unique().tolist()) == [4]
    assert out_df["subset"].nunique() == 9
    assert "A+B__X_Y" in set(out_df["subset"])

    for _subset, subset_df in out_df.groupby("subset", sort=True):
        assert len(subset_df) == 4
        assert _is_complete_2x2_subset(subset_df)


def test_align_dataset_tcga_2x2_keeps_explicit_quartet_memberships(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    manifest_dir = tmp_path / "manifests"
    metadata_dir.mkdir()
    manifest_dir.mkdir()

    rows = []
    for label, center, slide, patch in [
        ("A", "X", "ax", "p0"),
        ("A", "Y", "ay", "p1"),
        ("B", "X", "bx", "p2"),
        ("B", "Y", "by", "p3"),
    ]:
        rows.append(
            {
                "sample_id": f"{slide}__{patch}",
                "image_path": f"/tmp/{slide}__{patch}.png",
                "label": label,
                "medical_center": center,
                "slide_id": slide,
                "patch_id": patch,
            }
        )
    _write_csv(manifest_dir / "pathorob-tcga-source.csv", rows)
    _write_csv(
        metadata_dir / "tcga_2x2.csv",
        [
            {
                "subset": "A_B",
                "slide_id": row["slide_id"],
                "patch_id": row["patch_id"],
                "biological_class": row["label"],
                "medical_center": row["medical_center"],
            }
            for row in rows
        ],
    )

    output_path = pp.align_dataset(
        alignment=_alignment("pathorob-tcga-2x2"),
        metadata_dir=metadata_dir,
        manifest_dir=manifest_dir,
        progress_on=False,
    )

    out_df = pd.read_csv(output_path, dtype=str)
    assert len(out_df) == 4
    assert out_df["subset"].nunique() == 1
    assert set(out_df["subset"]) == {"A_B"}
    assert _is_complete_2x2_subset(out_df)
