"""Build the PathoROB manifests: one all-tiles manifest per cohort, plus its RI view.

PathoROB evaluates RI and APD on *different, non-derivable* row sets, so a single
selection column cannot serve both. This script materialises both, reproducibly, from
PathoROB's own metadata CSVs (``data/pathorob/metadata/``) joined against the source
manifests written by ``prepare_pathorob.py``.

Per cohort it writes:

``pathorob-<cohort>.csv``
    Every tile, carrying ``apd_split`` (PathoROB's ``subset`` column: ``ID``/``OOD``).
    This is the tileset source *and* the APD view -- APD evaluates ID+OOD, i.e. the whole
    cohort, so a separate ``-apd.csv`` would be a byte-identical copy.

``pathorob-<cohort>-ri.csv``
    Exactly the rows PathoROB's ``get_meta`` feeds to the Robustness Index. Deliberately
    carries no ``apd_split``: for Tolkach the RI set *straddles* the APD split, so the
    column would actively mislead.

The RI selection is not uniform, which is the whole reason this script exists:

===============  ==========================================  ======
cohort           RI rows come from                           n
===============  ==========================================  ======
camelyon         ``camelyon.csv`` where ``subset == "ID"``    20400
tcga-4x4         ``tcga_4x4.csv`` where ``subset == "ID"``     5760
tcga-2x2         ``tcga_2x2.csv`` in full (no OOD rows)      112800
tolkach-esca     ``tolkach_esca_reduced.csv`` in full          9000
===============  ==========================================  ======

Tolkach is the awkward one. ``tolkach_esca_reduced.csv`` is a *balanced sample* -- 500
tiles per (biological class x centre) over UKK/WNS/CHA_FULL, where WNS and CHA_FULL have
900 available per cell -- so it cannot be reconstructed by filtering, only stored. And
3,000 of its 9,000 rows are ``VALSET1_UKK``, which APD classifies as **OOD**. Hence
``apd_split`` is APD's notion of in/out-of-distribution and nothing more.

Run: python scripts/prep/build_pathorob_views.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = ROOT / "data" / "pathorob" / "metadata"
MANIFEST_DIR = ROOT / "data" / "pathorob" / "manifests"

#: Columns of every emitted manifest, in order. ``apd_split`` is appended for the
#: all-tiles manifests only; ``subset`` for the paired tcga-2x2 view only.
BASE_COLUMNS = ("sample_id", "image_path", "label", "medical_center", "slide_id")


@dataclass(frozen=True)
class CohortSpec:
    """One PathoROB cohort: its all-tiles metadata and how its RI view is selected."""

    cohort: str
    #: Source manifest slug (``pathorob-<slug>-source.csv``) holding sample_id/image_path.
    source_slug: str
    #: PathoROB metadata CSV enumerating every tile of the cohort.
    all_tiles_metadata: str
    #: PathoROB metadata CSV whose rows are exactly the RI set.
    ri_metadata: str
    #: When True, the RI view is ``ri_metadata`` filtered to ``subset == "ID"``.
    #: When False, every row of ``ri_metadata`` is an RI row.
    ri_filters_to_id: bool
    #: Paired designs carry PathoROB's ``subset`` column through to the RI view.
    ri_carries_subset: bool = False
    #: False for tcga-2x2: APD never evaluates it, and its ``subset`` column names the
    #: 2x2 pair rather than ID/OOD, so there is no ``apd_split`` to write and no
    #: all-tiles manifest to emit. Its tileset source stays ``pathorob-tcga-2x2.csv``.
    has_apd: bool = True


COHORTS: tuple[CohortSpec, ...] = (
    CohortSpec("camelyon", "camelyon", "camelyon.csv", "camelyon.csv", True),
    CohortSpec("tcga-4x4", "tcga", "tcga_4x4.csv", "tcga_4x4.csv", True),
    CohortSpec(
        "tcga-2x2",
        "tcga",
        "tcga_2x2.csv",
        "tcga_2x2.csv",
        False,
        ri_carries_subset=True,
        has_apd=False,
    ),
    # The RI set is a stored balanced sample, NOT a filter of the all-tiles file.
    CohortSpec(
        "tolkach-esca",
        "tolkach_esca",
        "tolkach_esca.csv",
        "tolkach_esca_reduced.csv",
        False,
    ),
)


def _load_source(slug: str) -> pd.DataFrame:
    path = MANIFEST_DIR / f"pathorob-{slug}-source.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"source manifest not found: {path}. Run prepare_pathorob.py first."
        )
    df = pd.read_csv(path, dtype=str)
    return df[["slide_id", "patch_id", "sample_id", "image_path"]]


def _join(metadata: str, source: pd.DataFrame) -> pd.DataFrame:
    """Attach sample_id/image_path to a PathoROB metadata CSV, preserving its row order."""
    path = METADATA_DIR / metadata
    if not path.exists():
        raise FileNotFoundError(f"PathoROB metadata not found: {path}")
    meta = pd.read_csv(path, dtype=str)
    for col in ("slide_id", "patch_id"):
        meta[col] = meta[col].str.strip()

    joined = meta.merge(source, on=["slide_id", "patch_id"], how="left")
    missing = int(joined["sample_id"].isna().sum())
    if missing:
        raise ValueError(
            f"{metadata}: {missing}/{len(meta)} rows have no tile in the source manifest"
        )
    if len(joined) != len(meta):
        raise ValueError(
            f"{metadata}: join changed row count {len(meta)} -> {len(joined)}; "
            "the source manifest has duplicate (slide_id, patch_id) keys"
        )
    return joined


def _to_manifest(joined: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": joined["sample_id"],
            "image_path": joined["image_path"],
            "label": joined["biological_class"],
            "medical_center": joined["medical_center"],
            "slide_id": joined["slide_id"],
        }
    )


def build_cohort(spec: CohortSpec) -> list[Path]:
    source = _load_source(spec.source_slug)
    written: list[Path] = []

    if spec.has_apd:
        all_joined = _join(spec.all_tiles_metadata, source)
        all_df = _to_manifest(all_joined)
        # PathoROB's `subset` here is ID/OOD: APD's split, and only APD's.
        all_df["apd_split"] = all_joined["subset"].astype(str)
        if not set(all_df["apd_split"]) <= {"ID", "OOD"}:
            raise ValueError(
                f"{spec.all_tiles_metadata}: subset is not an ID/OOD split "
                f"(saw {sorted(set(all_df['apd_split']))!r})"
            )
        all_path = MANIFEST_DIR / f"pathorob-{spec.cohort}.csv"
        all_df.to_csv(all_path, index=False)
        written.append(all_path)
        split = f"(ID={int((all_df.apd_split == 'ID').sum())}, OOD={int((all_df.apd_split == 'OOD').sum())})"
        n_all: object = len(all_df)
    else:
        split, n_all = "(no APD)", "-"

    ri_joined = _join(spec.ri_metadata, source)
    if spec.ri_filters_to_id:
        ri_joined = ri_joined[ri_joined["subset"] == "ID"].reset_index(drop=True)
    ri_df = _to_manifest(ri_joined)
    if spec.ri_carries_subset:
        ri_df["subset"] = ri_joined["subset"].astype(str)
    ri_path = MANIFEST_DIR / f"pathorob-{spec.cohort}-ri.csv"
    ri_df.to_csv(ri_path, index=False)
    written.append(ri_path)

    print(f"[views] {spec.cohort:<13} all={n_all:>6} {split:<24} ri={len(ri_df):>6}")
    return written


def main() -> int:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for spec in COHORTS:
        build_cohort(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
