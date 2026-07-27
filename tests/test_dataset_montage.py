import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "scripts" / "repro" / "figures"
if str(FIGURES) not in sys.path:
    sys.path.insert(0, str(FIGURES))

import dataset_montage as dm


CLASSES = ("normal", "tumor")
CENTERS = ("RUMC", "UMCU")


def _synthetic_manifest(
    *,
    classes=CLASSES,
    centers=CENTERS,
    per_cell: int = 6,
    seed: int = 1234,
) -> pd.DataFrame:
    """A balanced class x center manifest with unique, on-disk-free image paths.

    Rows are shuffled with a seeded RNG so tests do not accidentally rely on input
    ordering; the image paths are synthetic (never touch the filesystem).
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for klass, center in product(classes, centers):
        for i in range(per_cell):
            sample_id = f"{klass}__{center}__{i}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"/nowhere/{sample_id}.png",
                    "label": klass,
                    "medical_center": center,
                    "slide_id": f"slide_{klass}_{center}",
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def test_select_montage_tiles_covers_full_grid() -> None:
    manifest = _synthetic_manifest()
    selected = dm.select_montage_tiles(
        manifest, classes=CLASSES, centers=CENTERS, seed=0
    )
    assert set(selected.keys()) == set(product(CLASSES, CENTERS))
    # every cell filled with a real candidate path from that exact cell
    for (klass, center), path in selected.items():
        row = manifest[manifest["image_path"] == path]
        assert len(row) == 1
        assert row["label"].iloc[0] == klass
        assert row["medical_center"].iloc[0] == center


def test_select_montage_tiles_is_deterministic() -> None:
    manifest = _synthetic_manifest()
    first = dm.select_montage_tiles(manifest, classes=CLASSES, centers=CENTERS, seed=7)
    second = dm.select_montage_tiles(manifest, classes=CLASSES, centers=CENTERS, seed=7)
    assert first == second


def test_select_montage_tiles_is_row_order_invariant() -> None:
    manifest = _synthetic_manifest(seed=1)
    shuffled = manifest.sample(frac=1.0, random_state=99).reset_index(drop=True)
    a = dm.select_montage_tiles(manifest, classes=CLASSES, centers=CENTERS, seed=3)
    b = dm.select_montage_tiles(shuffled, classes=CLASSES, centers=CENTERS, seed=3)
    assert a == b


def test_select_montage_tiles_seed_changes_selection() -> None:
    # With many candidates per cell, different seeds should give at least one
    # different tile (guards against a constant/ignored seed).
    manifest = _synthetic_manifest(per_cell=25)
    a = dm.select_montage_tiles(manifest, classes=CLASSES, centers=CENTERS, seed=0)
    b = dm.select_montage_tiles(manifest, classes=CLASSES, centers=CENTERS, seed=1)
    assert a != b


def test_select_montage_tiles_restricts_to_requested_pair() -> None:
    # Extra classes/centers in the manifest must not leak into the selection.
    manifest = _synthetic_manifest(
        classes=("normal", "tumor", "other"),
        centers=("RUMC", "UMCU", "CWZ"),
    )
    selected = dm.select_montage_tiles(
        manifest, classes=CLASSES, centers=CENTERS, seed=0
    )
    assert set(selected.keys()) == set(product(CLASSES, CENTERS))
    for (klass, center) in selected:
        assert klass in CLASSES
        assert center in CENTERS


def test_select_montage_tiles_missing_cell_raises() -> None:
    manifest = _synthetic_manifest()
    # Drop one full cell so the grid is incomplete.
    manifest = manifest[
        ~((manifest["label"] == "tumor") & (manifest["medical_center"] == "UMCU"))
    ].reset_index(drop=True)
    with pytest.raises(ValueError, match="no tiles for cell"):
        dm.select_montage_tiles(manifest, classes=CLASSES, centers=CENTERS, seed=0)


def test_select_montage_tiles_requires_two_distinct_values() -> None:
    manifest = _synthetic_manifest()
    with pytest.raises(ValueError, match="2 distinct classes"):
        dm.select_montage_tiles(
            manifest, classes=("normal", "normal"), centers=CENTERS, seed=0
        )
    with pytest.raises(ValueError, match="2 distinct centers"):
        dm.select_montage_tiles(
            manifest, classes=CLASSES, centers=("RUMC", "RUMC"), seed=0
        )


def test_select_montage_tiles_missing_column_raises() -> None:
    manifest = _synthetic_manifest().drop(columns=["medical_center"])
    with pytest.raises(ValueError, match="missing required column"):
        dm.select_montage_tiles(manifest, classes=CLASSES, centers=CENTERS, seed=0)


def test_spec_filtered_applies_subset_and_grid() -> None:
    # A TCGA-like manifest with two subsets; only the requested quartet survives.
    rows: list[dict[str, str]] = []
    for subset, (c1, c2), (ctr1, ctr2) in [
        ("BLCA_BRCA", ("BLCA", "BRCA"), ("MDA", "UPitt")),
        ("BLCA_CESC", ("BLCA", "CESC"), ("MDA", "Duke")),
    ]:
        for klass in (c1, c2):
            for center in (ctr1, ctr2):
                sid = f"{subset}_{klass}_{center}"
                rows.append(
                    {
                        "sample_id": sid,
                        "image_path": f"/nowhere/{sid}.png",
                        "label": klass,
                        "medical_center": center,
                        "slide_id": sid,
                        "subset": subset,
                    }
                )
    manifest = pd.DataFrame(rows)
    spec = dm.MontageSpec(
        name="TCGA-test",
        manifest="unused.csv",
        classes=("BLCA", "BRCA"),
        centers=("MDA", "UPitt"),
        subset="BLCA_BRCA",
    )
    filtered = spec.filtered(manifest)
    assert set(filtered["subset"].unique()) == {"BLCA_BRCA"}
    assert set(filtered["label"].unique()) == {"BLCA", "BRCA"}
    assert set(filtered["medical_center"].unique()) == {"MDA", "UPitt"}
    selected = dm.select_montage_tiles(
        filtered, classes=spec.classes, centers=spec.centers, seed=0
    )
    assert set(selected.keys()) == {
        ("BLCA", "MDA"),
        ("BLCA", "UPitt"),
        ("BRCA", "MDA"),
        ("BRCA", "UPitt"),
    }


def test_collect_blocks_reports_missing_manifest(tmp_path: Path) -> None:
    # No data tree under tmp_path -> every spec is reported as skipped, no crash,
    # and the (image-guarded) render path is never reached.
    blocks, problems = dm.collect_blocks(dm.SPECS, repo=tmp_path, seed=0)
    assert blocks == []
    assert {name for name, _ in problems} == {spec.name for spec in dm.SPECS}


def test_specs_are_valid_grids() -> None:
    # Every committed spec has exactly two distinct biology rows and at least two
    # distinct confounder columns (the column count varies by benchmark).
    for spec in dm.SPECS:
        assert len(spec.classes) == 2 and len(set(spec.classes)) == 2
        assert len(spec.centers) >= 2 and len(set(spec.centers)) == len(spec.centers)
