from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("sample_id", "image_path", "label", "medical_center", "slide_id")


@dataclass(frozen=True)
class EvaluationSubset:
    subset_id: str
    rows: pd.DataFrame


def _normalize_str(v: object) -> str:
    return str(v).strip()


def ensure_required_columns(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def load_manifest(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")

    df = pd.read_csv(path, dtype=str)
    ensure_required_columns(df, f"manifest {csv_path}")

    out = df.copy()
    out["dataset"] = path.stem
    out["sample_id"] = out["sample_id"].map(_normalize_str)
    out["label"] = out["label"].map(_normalize_str)
    out["medical_center"] = out["medical_center"].map(_normalize_str)
    out["slide_id"] = out["slide_id"].map(_normalize_str)
    out["image_path"] = out["image_path"].map(_normalize_str)
    if "subset" in out.columns:
        out["subset"] = out["subset"].map(_normalize_str)
    return out.reset_index(drop=True)


def _subset_indices_from_manifest(df: pd.DataFrame, *, require_subset_metadata: bool = True) -> dict[str, list[int]]:
    subset_to_indices: dict[str, list[int]] = {}
    if "subset" in df.columns:
        for idx, subset_id in enumerate(df["subset"].astype(str).tolist()):
            normalized = _normalize_str(subset_id)
            if not normalized:
                raise ValueError("manifest contains blank subset values")
            subset_to_indices.setdefault(normalized, []).append(int(idx))
        return subset_to_indices

    if require_subset_metadata:
        raise ValueError("manifest must define subset membership via a 'subset' column")
    return subset_to_indices


def _subset_frame(df: pd.DataFrame, indices: list[int], *, subset_id: str) -> pd.DataFrame:
    subset_df = df.iloc[indices].copy()
    subset_df["source_sample_index"] = subset_df.index.astype(int)
    subset_df["subset"] = str(subset_id)
    return subset_df.reset_index(drop=True)


def _is_complete_2x2_subset(df: pd.DataFrame) -> bool:
    labels = sorted(df["label"].astype(str).unique().tolist())
    centers = sorted(df["medical_center"].astype(str).unique().tolist())
    if len(labels) != 2 or len(centers) != 2:
        return False
    for label in labels:
        for center in centers:
            cell_n = int(((df["label"].astype(str) == label) & (df["medical_center"].astype(str) == center)).sum())
            if cell_n <= 0:
                return False
    return True


def validate_subset_manifest(df: pd.DataFrame, source: str) -> None:
    subset_to_indices = _subset_indices_from_manifest(df)
    if not subset_to_indices:
        raise ValueError(f"{source} does not define any subsets")
    invalid: list[str] = []
    for subset_id, indices in subset_to_indices.items():
        subset_df = _subset_frame(df, indices, subset_id=subset_id)
        if not _is_complete_2x2_subset(subset_df):
            invalid.append(str(subset_id))
    if invalid:
        joined = ", ".join(sorted(invalid))
        raise ValueError(f"{source} contains invalid subsets (must be complete 2x2 label-center subsets): {joined}")


def resolve_manifest_subsets(df: pd.DataFrame) -> list[EvaluationSubset]:
    subset_to_indices = _subset_indices_from_manifest(df)
    subsets: list[EvaluationSubset] = []
    for subset_id in sorted(subset_to_indices):
        subset_df = _subset_frame(df, subset_to_indices[subset_id], subset_id=subset_id)
        if not _is_complete_2x2_subset(subset_df):
            continue
        subsets.append(EvaluationSubset(subset_id=str(subset_id), rows=subset_df))
    return subsets


def retain_complete_subset_memberships(df: pd.DataFrame) -> pd.DataFrame:
    subset_to_indices = _subset_indices_from_manifest(df, require_subset_metadata=False)
    if not subset_to_indices:
        return df.reset_index(drop=True)

    valid_subsets = {
        subset_id
        for subset_id, indices in subset_to_indices.items()
        if _is_complete_2x2_subset(_subset_frame(df, indices, subset_id=subset_id))
    }

    kept = df.loc[df["subset"].astype(str).isin(valid_subsets)].copy()
    return kept.reset_index(drop=True)
