from __future__ import annotations

import numpy as np
import pandas as pd

from croma.confounders import CANONICAL_CONFOUNDER_COLUMN

EMBEDDING_SOURCE_COLUMNS = (
    "sample_id",
    "image_path",
    "label",
    CANONICAL_CONFOUNDER_COLUMN,
    "slide_id",
)


def _normalize_key_value(value: object) -> str:
    return str(value).strip()


def ensure_embedding_source_columns(df: pd.DataFrame, source: str) -> None:
    missing = [col for col in EMBEDDING_SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns for embedding alignment: {missing}"
        )


def build_embedding_source_manifest(
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    ensure_embedding_source_columns(manifest_df, "manifest")

    unique_rows: list[tuple[str, ...]] = []
    key_to_index: dict[tuple[str, ...], int] = {}
    row_to_source: list[int] = []
    for row in manifest_df.loc[:, list(EMBEDDING_SOURCE_COLUMNS)].itertuples(
        index=False, name=None
    ):
        key = tuple(_normalize_key_value(value) for value in row)
        idx = key_to_index.get(key)
        if idx is None:
            idx = len(unique_rows)
            key_to_index[key] = idx
            unique_rows.append(key)
        row_to_source.append(int(idx))

    embedding_manifest = pd.DataFrame(unique_rows, columns=EMBEDDING_SOURCE_COLUMNS)
    return embedding_manifest, np.asarray(row_to_source, dtype=int)


def build_manifest_row_to_embedding_index(
    manifest_df: pd.DataFrame,
    embedding_manifest_df: pd.DataFrame,
) -> np.ndarray:
    ensure_embedding_source_columns(manifest_df, "evaluation manifest")
    ensure_embedding_source_columns(embedding_manifest_df, "embedding manifest")

    key_to_index: dict[tuple[str, ...], int] = {}
    for idx, row in enumerate(
        embedding_manifest_df.loc[:, list(EMBEDDING_SOURCE_COLUMNS)].itertuples(
            index=False, name=None
        )
    ):
        key = tuple(_normalize_key_value(value) for value in row)
        if key in key_to_index:
            raise ValueError("embedding manifest contains duplicate source rows")
        key_to_index[key] = int(idx)

    row_to_embedding: list[int] = []
    for row_idx, row in enumerate(
        manifest_df.loc[:, list(EMBEDDING_SOURCE_COLUMNS)].itertuples(
            index=False, name=None
        )
    ):
        key = tuple(_normalize_key_value(value) for value in row)
        idx = key_to_index.get(key)
        if idx is None:
            raise ValueError(
                f"evaluation manifest row {row_idx} is missing from embedding manifest"
            )
        row_to_embedding.append(int(idx))

    return np.asarray(row_to_embedding, dtype=int)


def expand_features_to_manifest(
    *,
    features: np.ndarray,
    manifest: pd.DataFrame,
    embedding_manifest: pd.DataFrame,
) -> np.ndarray:
    feature_array = np.asarray(features)
    if int(feature_array.shape[0]) != int(len(embedding_manifest)):
        raise ValueError("embeddings rows must match embedding manifest rows")
    row_to_embedding = build_manifest_row_to_embedding_index(
        manifest, embedding_manifest
    )
    return feature_array[row_to_embedding]
