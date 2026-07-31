from __future__ import annotations

import numpy as np
import pandas as pd

from croma.confounders import CANONICAL_CONFOUNDER_COLUMN
from croma.metrics.pairs import GROUP_COLUMN

EMBEDDING_SOURCE_COLUMNS = (
    "sample_id",
    "image_path",
    "label",
    CANONICAL_CONFOUNDER_COLUMN,
    GROUP_COLUMN,
)


def _normalize_key_value(value: object) -> str:
    return str(value).strip()


def ensure_embedding_source_columns(df: pd.DataFrame, source: str) -> None:
    missing = [col for col in EMBEDDING_SOURCE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns for embedding alignment: {missing}")


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
        manifest_df.loc[:, list(EMBEDDING_SOURCE_COLUMNS)].itertuples(index=False, name=None)
    ):
        key = tuple(_normalize_key_value(value) for value in row)
        idx = key_to_index.get(key)
        if idx is None:
            raise ValueError(
                f"evaluation manifest row {row_idx} is missing from embedding manifest"
            )
        row_to_embedding.append(int(idx))

    return np.asarray(row_to_embedding, dtype=int)


#: What makes two rows the same tile. Everything else -- notably ``label`` -- is an
#: attribute a *view* attaches to the tile, not part of its identity.
TILE_IDENTITY_COLUMNS = ("sample_id", "image_path")


def build_view_row_index(
    view_df: pd.DataFrame,
    tileset_df: pd.DataFrame,
) -> np.ndarray:
    """Map each row of a benchmark's eval manifest to its tileset embedding row.

    A benchmark is a *view* over a tileset: it selects tiles (and, for paired designs,
    may repeat one tile across subsets) and attaches its own ``label``. Identity is the
    tile, not the row's full attribute tuple -- the same tile is ``tumor`` in the binary
    prostate benchmark and ``gleason-3`` in the four-class one, and the same slide is
    cancer ``1`` in PANDA-cancer and ISUP ``2`` in PANDA-ISUP.

    So the lookup is keyed on ``(sample_id, image_path)`` -- the tile -- and nothing else.
    A view may repeat a tile (one row per paired subset) and may relabel it, but it may
    not point a known ``sample_id`` at pixels the tileset never embedded.
    """
    for frame, source in ((view_df, "evaluation manifest"), (tileset_df, "tileset manifest")):
        missing = [c for c in TILE_IDENTITY_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{source} is missing tile-identity columns: {missing}")

    tile_row: dict[tuple[str, str], int] = {}
    paths_for_sample: dict[str, str] = {}
    for idx, row in enumerate(
        tileset_df.loc[:, list(TILE_IDENTITY_COLUMNS)].itertuples(index=False, name=None)
    ):
        key = tuple(_normalize_key_value(value) for value in row)
        if key in tile_row:
            raise ValueError(f"tileset manifest contains duplicate tile {key!r}")
        tile_row[key] = idx
        paths_for_sample.setdefault(key[0], key[1])

    rows: list[int] = []
    for row_idx, row in enumerate(
        view_df.loc[:, list(TILE_IDENTITY_COLUMNS)].itertuples(index=False, name=None)
    ):
        sample_id, image_path = (_normalize_key_value(value) for value in row)
        idx = tile_row.get((sample_id, image_path))
        if idx is not None:
            rows.append(int(idx))
            continue
        if sample_id in paths_for_sample:
            raise ValueError(
                f"evaluation manifest row {row_idx} (sample_id={sample_id!r}) points at "
                f"{image_path!r} but the tileset embedded {paths_for_sample[sample_id]!r}"
            )
        raise ValueError(
            f"evaluation manifest row {row_idx} (sample_id={sample_id!r}) is not in the tileset"
        )

    return np.asarray(rows, dtype=int)


def expand_features_to_manifest(
    *,
    features: np.ndarray,
    manifest: pd.DataFrame,
    embedding_manifest: pd.DataFrame,
) -> np.ndarray:
    feature_array = np.asarray(features)
    if int(feature_array.shape[0]) != int(len(embedding_manifest)):
        raise ValueError("embeddings rows must match embedding manifest rows")
    row_to_embedding = build_manifest_row_to_embedding_index(manifest, embedding_manifest)
    return feature_array[row_to_embedding]
