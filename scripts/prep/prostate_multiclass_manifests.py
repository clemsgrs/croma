"""Build the two prostate-shift KI+RUMC manifests for the biology-granularity study.

The headline prostate benchmark is a *natural* binary (benign vs tumour at the full
1,850 tiles/cell, the organic Gleason mix). This script derives two further manifests
that share patches so robustness can be read at two biological resolutions on a matched
cell-balanced set, isolating the effect of label granularity:

  4-class   (``prostate-shift-4class-kirumc.csv``):
            biology label in {benign, gleason-3, gleason-4, gleason-5}, 480 tiles per
            (class x centre) -- 480 is capped by the smallest cell (Gleason-3, 481/centre).
            Computed via the paired-2x2 protocol -> C(4,2)=6 grade-pair 2x2s x {KI,RUMC}.

  grade-balanced binary (``prostate-shift-gradebal-binary-kirumc.csv``):
            benign vs tumour, but with the tumour side grade-balanced: 1,440 tumour/centre
            = the SAME 480 Gleason-3 + 480 Gleason-4 + 480 Gleason-5 tiles used by the
            4-class manifest (their union), against 1,440 benign/centre (a superset of the
            4-class's 480 benign). Computed via the paired-2x2 protocol -> the degenerate
            C(2,2)=1 pair. Only the label resolution differs from the 4-class run on the
            tumour side; the tumour patches are identical.

Determinism: a fixed seed and a stable (centre, biology, sample_id) sort make the draw
reproducible. The grade tiles are drawn ONCE and reused in both manifests; the 4-class
benign cell is the first 480 of the binary's 1,440 benign draw, so 4-class benign is a
strict subset. The numbers across granularities are therefore NOT absolute-comparable
(the 4-class folds in grade-vs-grade pairs the binary cannot express, and restricts each
pair's candidate pool to 480 single-grade tiles vs the binary's 1,440 mixed); the study
compares model *rankings/reorderings* and the per-grade-pair breakdown, not raw levels.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
FULL_CSV = REPO / "data/prostate-shift-binary.csv"  # 8,000 rows, the embedded source order
ID_CENTERS = ("KI", "RUMC")
GRADES = ("gleason-3", "gleason-4", "gleason-5")
PER_GRADE = 480          # capped by Gleason-3 (481/centre)
PER_TUMOR = PER_GRADE * len(GRADES)  # 1,440 tumour/centre (grade-balanced)
SEED = 0

OUT_4CLASS = REPO / "data/prostate-shift-4class-kirumc.csv"
OUT_GRADEBAL = REPO / "data/prostate-shift-gradebal-binary-kirumc.csv"
COLUMNS = ["sample_id", "label", "medical_center", "group_id", "biology", "image_path"]


def _draw(df: pd.DataFrame, center: str, biology: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Stable seeded draw of n rows from one (centre, biology) cell."""
    cell = df[(df["medical_center"] == center) & (df["biology"] == biology)]
    cell = cell.sort_values("sample_id").reset_index(drop=True)
    if len(cell) < n:
        raise ValueError(f"cell ({center}, {biology}) has {len(cell)} < {n} tiles")
    pick = rng.permutation(len(cell))[:n]
    return cell.iloc[np.sort(pick)].copy()


def main() -> int:
    df = pd.read_csv(FULL_CSV)
    if list(df.columns) != ["sample_id", "label", "medical_center", "group_id", "biology", "image_path"]:
        raise ValueError(f"unexpected source columns: {df.columns.tolist()}")
    rng = np.random.default_rng(SEED)

    fourclass_rows: list[pd.DataFrame] = []
    gradebal_rows: list[pd.DataFrame] = []
    for center in ID_CENTERS:
        # Tumour grade cells: drawn once, shared by both manifests.
        for grade in GRADES:
            cell = _draw(df, center, grade, PER_GRADE, rng)
            fc = cell.copy()
            fc["label"] = fc["biology"]          # 4-class: label IS the Gleason grade
            fourclass_rows.append(fc)
            gradebal_rows.append(cell.copy())    # binary: keep original benign/tumour label
        # Benign cell: 1,440 for the binary; first 480 of that draw for the 4-class.
        benign = _draw(df, center, "benign", PER_TUMOR, rng)
        gradebal_rows.append(benign.copy())
        fc_benign = benign.iloc[:PER_GRADE].copy()
        fc_benign["label"] = fc_benign["biology"]  # "benign"
        fourclass_rows.append(fc_benign)

    fourclass = pd.concat(fourclass_rows, ignore_index=True).loc[:, COLUMNS]
    gradebal = pd.concat(gradebal_rows, ignore_index=True).loc[:, COLUMNS]

    # Integrity: tumour patches identical across the two manifests; 4-class benign subset.
    fc_tumor = set(fourclass.loc[fourclass["biology"].isin(GRADES), "sample_id"])
    gb_tumor = set(gradebal.loc[gradebal["label"] == "tumor", "sample_id"])
    assert fc_tumor == gb_tumor, "tumour patches must be identical across manifests"
    fc_benign_ids = set(fourclass.loc[fourclass["biology"] == "benign", "sample_id"])
    gb_benign_ids = set(gradebal.loc[gradebal["label"] == "benign", "sample_id"])
    assert fc_benign_ids <= gb_benign_ids, "4-class benign must be a subset of binary benign"

    fourclass.to_csv(OUT_4CLASS, index=False)
    gradebal.to_csv(OUT_GRADEBAL, index=False)

    print(f"[4-class]   {OUT_4CLASS.name}  ({len(fourclass)} rows)")
    print(fourclass.groupby(["medical_center", "label"]).size().to_string())
    print(f"\n[gradebal]  {OUT_GRADEBAL.name}  ({len(gradebal)} rows)")
    print(gradebal.groupby(["medical_center", "label"]).size().to_string())
    print(gradebal.groupby(["medical_center", "biology"]).size().to_string())
    print(f"\nshared tumour tiles: {len(fc_tumor)}  |  4-class benign subset of binary: "
          f"{len(fc_benign_ids)} / {len(gb_benign_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
