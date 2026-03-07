
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "image_path", "label", "medical_center", "slide_id")


@dataclass(frozen=True)
class PairSpec:
    dataset: str
    pair_id: str
    classes: tuple[str, str]
    centers: tuple[str, str]


def _normalize_str(v: object) -> str:
    return str(v).strip()


def normalize_center_values(values: object | None) -> tuple[str, ...]:
    if values is None:
        return ()

    raw_values: list[object]
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, Iterable):
        raw_values = list(values)
    else:
        raise TypeError("exclude_centers must be a string, iterable of strings, or None")

    normalized = sorted({_normalize_str(v) for v in raw_values if _normalize_str(v)})
    return tuple(normalized)


def ensure_required_columns(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_manifest(csv_path: str, dataset_name: str = "dataset") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(path)
    ensure_required_columns(df, f"manifest {csv_path}")

    out = df.copy()
    out["dataset"] = str(dataset_name)
    out["sample_id"] = out["sample_id"].map(_normalize_str)
    out["label"] = out["label"].map(_normalize_str)
    out["medical_center"] = out["medical_center"].map(_normalize_str)
    out["slide_id"] = out["slide_id"].map(_normalize_str)
    out["image_path"] = out["image_path"].map(_normalize_str)
    if "subset" in out.columns:
        out["subset"] = out["subset"].map(_normalize_str)
    return out.reset_index(drop=True)


def infer_2x2_pairs(
    df: pd.DataFrame,
    dataset_name: str,
    max_pairs: int | None = None,
    random_state: int = 0,
) -> list[PairSpec]:
    labels = sorted(df["label"].unique().tolist())
    pairs: list[PairSpec] = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            c1, c2 = labels[i], labels[j]
            sub = df[df["label"].isin([c1, c2])]
            centers = sorted(sub["medical_center"].unique().tolist())
            for a in range(len(centers)):
                for b in range(a + 1, len(centers)):
                    m1, m2 = centers[a], centers[b]
                    ok = True
                    for lbl in (c1, c2):
                        for ctr in (m1, m2):
                            n = int(((sub["label"] == lbl) & (sub["medical_center"] == ctr)).sum())
                            if n <= 0:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        pairs.append(
                            PairSpec(
                                dataset=str(dataset_name),
                                pair_id=f"{dataset_name}::{c1}__{c2}::{m1}__{m2}",
                                classes=(str(c1), str(c2)),
                                centers=(str(m1), str(m2)),
                            )
                        )

    if max_pairs is not None and len(pairs) > int(max_pairs):
        rng = np.random.default_rng(int(random_state))
        idx = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(i)] for i in sorted(idx.tolist())]

    return pairs


def subset_by_pair(df: pd.DataFrame, pair: PairSpec) -> pd.DataFrame:
    return df[df["label"].isin(pair.classes) & df["medical_center"].isin(pair.centers)].copy()
