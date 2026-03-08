"""Fetch PathoROB datasets from Hugging Face, convert parquet rows to local PNG tiles,
and emit MaRI-compatible manifests aligned with PathoROB experimental configurations.

Two-phase workflow:
  Phase A (extract): download HF parquets → decode images → write source manifests
  Phase B (align):   join source manifests with PathoROB metadata → write config manifests
"""

import argparse
import base64
import io
import json
import re
import shutil
from itertools import combinations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from PIL import Image
from progress_utils import progress_bar, progress_write, resolve_progress_mode


# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    output_slug: str
    sample_candidates: tuple[str, ...] = ("patch_id", "sample_id", "id")
    label_candidates: tuple[str, ...] = ("biological_class", "label", "class_label")
    center_candidates: tuple[str, ...] = ("medical_center", "center")
    slide_candidates: tuple[str, ...] = ("slide_id", "case_id", "wsi_id")
    image_candidates: tuple[str, ...] = ("image", "tile", "patch", "img")


DATASETS: dict[str, DatasetSpec] = {
    "camelyon": DatasetSpec(
        key="camelyon",
        repo_id="bifold-pathomics/PathoROB-camelyon",
        output_slug="camelyon",
    ),
    "tcga": DatasetSpec(
        key="tcga",
        repo_id="bifold-pathomics/PathoROB-tcga",
        output_slug="tcga",
    ),
    "tolkach_esca": DatasetSpec(
        key="tolkach_esca",
        repo_id="bifold-pathomics/PathoROB-tolkach_esca",
        output_slug="tolkach_esca",
    ),
}


@dataclass(frozen=True)
class AlignmentSpec:
    """Maps a PathoROB metadata CSV to the output manifest it produces."""

    metadata_filename: str
    dataset_key: str
    output_name: str
    subset_mode: str  # one of: "id_ood", "paired_passthrough", "paired_expand_grid"


ALIGNMENTS: list[AlignmentSpec] = [
    AlignmentSpec("camelyon.csv", "camelyon", "pathorob-camelyon", subset_mode="id_ood"),
    AlignmentSpec(
        "camelyon_reduced.csv",
        "camelyon",
        "pathorob-camelyon-reduced",
        subset_mode="paired_expand_grid",
    ),
    AlignmentSpec("tcga_4x4.csv", "tcga", "pathorob-tcga-4x4", subset_mode="id_ood"),
    AlignmentSpec("tcga_2x2.csv", "tcga", "pathorob-tcga-2x2", subset_mode="paired_passthrough"),
    AlignmentSpec("tolkach_esca.csv", "tolkach_esca", "pathorob-tolkach-esca", subset_mode="id_ood"),
    AlignmentSpec(
        "tolkach_esca_reduced.csv",
        "tolkach_esca",
        "pathorob-tolkach-esca-reduced",
        subset_mode="paired_expand_grid",
    ),
]


# ---------------------------------------------------------------------------
# Helpers: sanitization, column resolution, image decoding
# ---------------------------------------------------------------------------


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return token.strip("_") or "part"


def _resolve_column(columns: list[str], candidates: tuple[str, ...], kind: str, source: Path) -> str:
    for cand in candidates:
        if cand in columns:
            return cand
    raise ValueError(
        f"{source}: could not resolve {kind} column. "
        f"Tried {list(candidates)}. Available columns: {columns}"
    )


def _column_present_in_batch(batch: Any, column: str | None) -> bool:
    if column is None:
        return False
    return int(batch.schema.get_field_index(str(column))) >= 0


def _sample_non_null_batch_value(batch: Any, column: str, *, max_rows: int = 16) -> Any:
    field_idx = int(batch.schema.get_field_index(str(column)))
    if field_idx < 0:
        return None
    arr = batch.column(field_idx)
    limit = min(int(len(arr)), int(max_rows))
    for i in range(limit):
        value = arr[i]
        if hasattr(value, "as_py"):
            value = value.as_py()
        if value is not None:
            return value
    return None


def _is_bytes_like(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview))


def _is_image_dict_like(value: Any) -> bool:
    return isinstance(value, dict) and ("bytes" in value or "path" in value)


def _looks_like_image_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return text.endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")) or "/" in text


def _infer_image_source_in_batch(
    *,
    batch: Any,
    dataset: DatasetSpec,
    required_columns: set[str],
    preferred_image_col: str | None,
) -> tuple[str | None, str | None, str | None]:
    if _column_present_in_batch(batch, preferred_image_col):
        return preferred_image_col, None, None
    for cand in dataset.image_candidates:
        if _column_present_in_batch(batch, cand):
            return str(cand), None, None

    bytes_col: str | None = None
    path_col: str | None = None
    dict_image_col: str | None = None
    for col in list(batch.schema.names):
        if col in required_columns:
            continue
        sample = _sample_non_null_batch_value(batch, col)
        if sample is None:
            continue
        if _is_image_dict_like(sample):
            dict_image_col = str(col)
            break
        if bytes_col is None and _is_bytes_like(sample):
            bytes_col = str(col)
        if path_col is None and _looks_like_image_path(sample):
            path_col = str(col)

    if dict_image_col is not None:
        return dict_image_col, None, None
    if bytes_col is not None or path_col is not None:
        return None, bytes_col, path_col
    return None, None, None


def _to_bytes(value: Any, parquet_parent: Path) -> bytes:
    if value is None:
        raise ValueError("image value is null")
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, str):
            try:
                return base64.b64decode(raw)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("failed to decode base64 image bytes") from exc
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return bytes(raw)
        path = value.get("path")
        if path:
            path_obj = Path(str(path))
            if not path_obj.is_absolute():
                path_obj = parquet_parent / path_obj
            if path_obj.exists():
                return path_obj.read_bytes()
        raise ValueError(
            f"image payload dict has no usable bytes/path. keys={sorted(value.keys())}"
        )
    if isinstance(value, str):
        path_obj = Path(value)
        if not path_obj.is_absolute():
            path_obj = parquet_parent / path_obj
        if path_obj.exists():
            return path_obj.read_bytes()
    raise TypeError(f"Unsupported image value type: {type(value)}")


def _batch_column_values(
    batch: Any,
    column: str,
    *,
    source: Path,
    required: bool,
) -> list[Any] | None:
    field_idx = int(batch.schema.get_field_index(str(column)))
    if field_idx < 0:
        if required:
            raise ValueError(
                f"{source}: batch is missing required column '{column}'. "
                f"Batch columns: {list(batch.schema.names)}"
            )
        return None
    return batch.column(field_idx).to_pylist()


def _decode_to_rgb_image(value: Any, parquet_parent: Path) -> "Image.Image":
    raw = _to_bytes(value, parquet_parent)
    img = Image.open(io.BytesIO(raw))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _normalize_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _is_complete_2x2_block(df: pd.DataFrame, *, label_col: str, center_col: str) -> bool:
    labels = sorted(df[label_col].astype(str).unique().tolist())
    centers = sorted(df[center_col].astype(str).unique().tolist())
    if len(labels) != 2 or len(centers) != 2:
        return False
    for label in labels:
        for center in centers:
            cell_n = int(((df[label_col].astype(str) == label) & (df[center_col].astype(str) == center)).sum())
            if cell_n <= 0:
                return False
    return True


def _paired_subset_id(
    label_pair: tuple[str, str],
    center_pair: tuple[str, str],
    *,
    total_label_count: int,
) -> str:
    center_token = "_".join(_sanitize_token(part) for part in center_pair)
    if int(total_label_count) == 2:
        return center_token
    label_token = "+".join(_sanitize_token(part) for part in label_pair)
    return f"{label_token}__{center_token}"


def _expand_grid_paired_subsets(df: pd.DataFrame, *, label_col: str, center_col: str) -> pd.DataFrame:
    labels = sorted(df[label_col].astype(str).unique().tolist())
    centers = sorted(df[center_col].astype(str).unique().tolist())
    if len(labels) < 2 or len(centers) < 2:
        raise ValueError("paired_expand_grid requires at least 2 labels and 2 medical centers")

    expanded_frames: list[pd.DataFrame] = []
    for label_pair in combinations(labels, 2):
        for center_pair in combinations(centers, 2):
            subset_df = df.loc[
                df[label_col].astype(str).isin(label_pair) & df[center_col].astype(str).isin(center_pair)
            ].copy()
            if not _is_complete_2x2_block(subset_df, label_col=label_col, center_col=center_col):
                continue
            subset_df["subset"] = _paired_subset_id(
                label_pair,
                center_pair,
                total_label_count=len(labels),
            )
            expanded_frames.append(subset_df)

    if not expanded_frames:
        raise ValueError("paired_expand_grid could not construct any complete 2x2 subsets")

    return pd.concat(expanded_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Phase A: Extract — download parquets and write source manifests
# ---------------------------------------------------------------------------


def _convert_parquet_to_rows(
    *,
    dataset: DatasetSpec,
    parquet_path: Path,
    images_dir: Path,
    batch_size: int,
    progress_on: bool,
    seen_sample_ids: set[str],
    seen_file_tokens: set[str],
    shard_token: str | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    parquet_file = pq.ParquetFile(parquet_path)
    columns = list(parquet_file.schema.names)

    sample_col = _resolve_column(columns, dataset.sample_candidates, "sample_id", parquet_path)
    label_col = _resolve_column(columns, dataset.label_candidates, "label", parquet_path)
    center_col = _resolve_column(columns, dataset.center_candidates, "medical_center", parquet_path)
    slide_col = _resolve_column(columns, dataset.slide_candidates, "slide_id", parquet_path)
    preferred_image_col = None
    for cand in dataset.image_candidates:
        if cand in columns:
            preferred_image_col = str(cand)
            break

    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    total_rows = int(parquet_file.metadata.num_rows) if parquet_file.metadata is not None else 0
    batch_iter = parquet_file.iter_batches(batch_size=int(batch_size))
    with progress_bar(total=total_rows, desc=f"convert:{parquet_path.name}", enabled=progress_on, unit="img") as bar:
        global_idx = 0
        for batch in batch_iter:
            sample_values = _batch_column_values(batch, sample_col, source=parquet_path, required=True)
            label_values = _batch_column_values(batch, label_col, source=parquet_path, required=True)
            center_values = _batch_column_values(batch, center_col, source=parquet_path, required=True)
            slide_values = _batch_column_values(batch, slide_col, source=parquet_path, required=True)
            if sample_values is None or label_values is None or center_values is None or slide_values is None:
                raise RuntimeError(f"{parquet_path}: unexpected missing required batch columns")

            required_cols = {sample_col, label_col, center_col, slide_col}
            batch_image_col, batch_bytes_col, batch_path_col = _infer_image_source_in_batch(
                batch=batch,
                dataset=dataset,
                required_columns=required_cols,
                preferred_image_col=preferred_image_col,
            )

            image_values = (
                _batch_column_values(batch, batch_image_col, source=parquet_path, required=True)
                if batch_image_col is not None
                else None
            )
            image_bytes_values = (
                _batch_column_values(batch, batch_bytes_col, source=parquet_path, required=False)
                if batch_bytes_col is not None
                else None
            )
            image_path_values = (
                _batch_column_values(batch, batch_path_col, source=parquet_path, required=False)
                if batch_path_col is not None
                else None
            )

            if image_values is None and image_bytes_values is None and image_path_values is None:
                raise ValueError(
                    f"{parquet_path}: no image source columns found in batch. "
                    f"Batch columns: {list(batch.schema.names)}"
                )

            batch_len = len(sample_values)
            for i in range(batch_len):
                patch_id_raw = _normalize_string(sample_values[i])
                if not patch_id_raw:
                    patch_id_raw = f"sample_{global_idx:08d}"

                slide = _normalize_string(slide_values[i])
                # Build a unique sample_id from slide_id + patch_id (+ shard if multi-part)
                base_sample_id = f"{slide}__{patch_id_raw}"
                if shard_token is not None and str(shard_token).strip():
                    base_sample_id = f"{base_sample_id}__{str(shard_token)}"

                sample_id = base_sample_id
                dup_idx = 1
                while sample_id in seen_sample_ids:
                    sample_id = f"{base_sample_id}__dup{dup_idx:03d}"
                    dup_idx += 1
                seen_sample_ids.add(sample_id)

                file_token_base = _sanitize_token(sample_id)
                file_token = file_token_base
                file_dup_idx = 1
                while file_token in seen_file_tokens:
                    file_token = f"{file_token_base}_{file_dup_idx:03d}"
                    file_dup_idx += 1
                seen_file_tokens.add(file_token)

                label = _normalize_string(label_values[i])
                center = _normalize_string(center_values[i])
                if not label or not center or not slide:
                    raise ValueError(
                        f"{parquet_path}: empty value at row {global_idx} "
                        f"(label='{label}', center='{center}', slide='{slide}')"
                    )

                if image_values is not None:
                    image_payload = image_values[i]
                else:
                    bytes_value = image_bytes_values[i] if image_bytes_values is not None else None
                    path_value = image_path_values[i] if image_path_values is not None else None
                    image_payload = {
                        "bytes": bytes_value,
                        "path": path_value,
                    }
                image_obj = _decode_to_rgb_image(image_payload, parquet_path.parent)
                abs_image = images_dir / f"{file_token}.png"
                image_obj.save(abs_image, format="PNG")

                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_path": str(abs_image),
                        "label": label,
                        "medical_center": center,
                        "slide_id": slide,
                        "patch_id": patch_id_raw,
                    }
                )
                global_idx += 1
                bar.update(1)

    source_columns = ["sample_id", "image_path", "label", "medical_center", "slide_id", "patch_id"]
    out_df = pd.DataFrame(rows, columns=source_columns)
    if out_df.empty:
        raise ValueError(f"{parquet_path}: conversion produced an empty manifest")

    missing_paths = [p for p in out_df["image_path"].tolist() if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError(f"{parquet_path}: {len(missing_paths)} converted image files are missing")

    return rows, {
        "parquet_path": str(parquet_path),
        "shard_token": str(shard_token) if shard_token is not None else "",
        "images_dir": str(images_dir),
        "rows": int(len(out_df)),
        "labels": sorted(out_df["label"].unique().tolist()),
        "centers": sorted(out_df["medical_center"].unique().tolist()),
    }


def _write_dataset_meta(
    *,
    dataset_root: Path,
    dataset: DatasetSpec,
    requested_revision: str,
    resolved_sha: str,
    manifest_path: Path,
    total_rows: int,
    conversions: list[dict[str, Any]],
) -> Path:
    meta = {
        "dataset_key": dataset.key,
        "repo_id": dataset.repo_id,
        "requested_revision": requested_revision,
        "resolved_sha": resolved_sha,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "total_rows": int(total_rows),
        "conversions": conversions,
    }
    out = dataset_root / "prepared_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def _download_dataset_to_temp(
    *,
    dataset: DatasetSpec,
    revision: str,
    tmp_dir: Path,
    token: str | None,
    max_workers: int,
) -> str:
    api = HfApi(token=token)
    info = api.dataset_info(dataset.repo_id, revision=revision)

    if tmp_dir.exists():
        existing = sorted(tmp_dir.rglob("*.parquet"))
        if existing:
            return str(info.sha)
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=dataset.repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=tmp_dir,
        allow_patterns=["data/*.parquet", "data/**/*.parquet"],
        token=token,
        max_workers=int(max_workers),
    )
    return str(info.sha)


SOURCE_COLUMNS = ["sample_id", "image_path", "label", "medical_center", "slide_id", "patch_id"]


def extract_dataset(
    *,
    spec: DatasetSpec,
    pathorob_root: Path,
    manifest_dir: Path,
    revision: str,
    hf_token: str | None,
    batch_size: int,
    max_workers: int,
    progress_on: bool,
) -> Path:
    """Phase A: download HF parquets, decode images, write source manifest.

    Returns the path to the source manifest CSV.
    """
    dataset_root = pathorob_root / spec.output_slug
    images_root = dataset_root / "images"
    tmp_download = dataset_root / "_tmp_parquet_download"
    dataset_root.mkdir(parents=True, exist_ok=True)
    if images_root.exists():
        shutil.rmtree(images_root)
    images_root.mkdir(parents=True, exist_ok=True)

    progress_write(f"[extract] dataset={spec.key} repo={spec.repo_id}", enabled=progress_on)
    conversions: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()
    seen_file_tokens: set[str] = set()
    resolved_sha = ""
    completed = False
    try:
        resolved_sha = _download_dataset_to_temp(
            dataset=spec,
            revision=revision,
            tmp_dir=tmp_download,
            token=hf_token,
            max_workers=max_workers,
        )
        parquet_paths = sorted(tmp_download.rglob("*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"{spec.repo_id}: no parquet files found under {tmp_download}")

        progress_write(
            f"[extract] {spec.key}: found {len(parquet_paths)} parquet file(s) at revision {resolved_sha}",
            enabled=progress_on,
        )

        multi_part = len(parquet_paths) > 1
        for parquet_path in parquet_paths:
            part_token = _sanitize_token(parquet_path.relative_to(tmp_download).with_suffix("").as_posix())
            progress_write(
                f"[extract] convert parquet={parquet_path.relative_to(tmp_download)}",
                enabled=progress_on,
            )
            rows_part, conversion = _convert_parquet_to_rows(
                dataset=spec,
                parquet_path=parquet_path,
                images_dir=images_root,
                batch_size=batch_size,
                progress_on=progress_on,
                seen_sample_ids=seen_sample_ids,
                seen_file_tokens=seen_file_tokens,
                shard_token=part_token if multi_part else None,
            )
            dataset_rows.extend(rows_part)
            conversions.append(conversion)

        source_manifest = manifest_dir / f"pathorob-{spec.output_slug}-source.csv"
        dataset_df = pd.DataFrame(dataset_rows, columns=SOURCE_COLUMNS)
        if dataset_df.empty:
            raise ValueError(f"{spec.repo_id}: merged dataset manifest would be empty")
        if dataset_df["sample_id"].duplicated().any():
            dup = int(dataset_df["sample_id"].duplicated().sum())
            raise ValueError(
                f"{spec.repo_id}: merged dataset has duplicate sample_id values after conversion ({dup})"
            )
        missing_paths = [p for p in dataset_df["image_path"].tolist() if not Path(p).exists()]
        if missing_paths:
            raise FileNotFoundError(
                f"{spec.repo_id}: {len(missing_paths)} merged image paths are missing on disk"
            )
        source_manifest.parent.mkdir(parents=True, exist_ok=True)
        dataset_df.to_csv(source_manifest, index=False)

        total_rows = int(len(dataset_df))
        meta_path = _write_dataset_meta(
            dataset_root=dataset_root,
            dataset=spec,
            requested_revision=revision,
            resolved_sha=resolved_sha,
            manifest_path=source_manifest,
            total_rows=total_rows,
            conversions=conversions,
        )
        progress_write(
            f"[extract] done dataset={spec.key} manifest={source_manifest} rows={total_rows} meta={meta_path}",
            enabled=progress_on,
        )
        completed = True
    finally:
        if completed and tmp_download.exists():
            shutil.rmtree(tmp_download, ignore_errors=True)
        elif not completed and tmp_download.exists():
            progress_write(
                f"[extract] preserving download cache after failure: {tmp_download}",
                enabled=progress_on,
            )

    return source_manifest


# ---------------------------------------------------------------------------
# Phase B: Align — join source manifests with PathoROB metadata
# ---------------------------------------------------------------------------


def align_dataset(
    *,
    alignment: AlignmentSpec,
    metadata_dir: Path,
    manifest_dir: Path,
    progress_on: bool,
) -> Path:
    """Join a source manifest with a PathoROB metadata CSV and write an aligned manifest.

    Returns the path to the aligned manifest CSV.
    """
    metadata_path = metadata_dir / alignment.metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(f"PathoROB metadata not found: {metadata_path}")

    source_key = DATASETS[alignment.dataset_key].output_slug
    source_manifest = manifest_dir / f"pathorob-{source_key}-source.csv"
    if not source_manifest.exists():
        raise FileNotFoundError(
            f"Source manifest not found: {source_manifest}. Run extraction first (without --skip-extract)."
        )

    meta_df = pd.read_csv(metadata_path, dtype=str)
    source_df = pd.read_csv(source_manifest, dtype=str)

    # Ensure join keys are strings
    for col in ("slide_id", "patch_id"):
        meta_df[col] = meta_df[col].astype(str).str.strip()
        source_df[col] = source_df[col].astype(str).str.strip()

    meta_unique = meta_df

    # Keep only join keys + sample_id/image_path from source (avoid column clashes)
    source_join = source_df[["slide_id", "patch_id", "sample_id", "image_path"]]

    joined = pd.merge(
        meta_unique,
        source_join,
        on=["slide_id", "patch_id"],
        how="inner",
    )

    n_meta = len(meta_unique)
    n_matched = len(joined)
    n_unmatched = n_meta - n_matched

    progress_write(
        f"[align] {alignment.output_name}: {n_matched}/{n_meta} matched"
        + (f", {n_unmatched} unmatched" if n_unmatched > 0 else ""),
        enabled=progress_on,
    )

    if n_unmatched > 0:
        raise ValueError(
            f"Alignment failed: {n_unmatched}/{n_meta} metadata rows have no match in source manifest "
            f"for {alignment.output_name}."
        )

    # Use label/center from metadata (ground truth), image_path/sample_id from source
    label_col = "biological_class" if "biological_class" in joined.columns else "label"
    center_col = "medical_center"

    out_df = pd.DataFrame(
        {
            "sample_id": joined["sample_id"],
            "image_path": joined["image_path"],
            "label": joined[label_col],
            "medical_center": joined[center_col],
            "slide_id": joined["slide_id"],
        }
    )

    if alignment.subset_mode == "paired_passthrough":
        out_df["subset"] = joined["subset"].astype(str)
    elif alignment.subset_mode == "paired_expand_grid":
        out_df = _expand_grid_paired_subsets(out_df, label_col="label", center_col="medical_center")
    elif alignment.subset_mode == "id_ood":
        out_df["id_ood"] = joined["subset"].astype(str)
    else:
        raise ValueError(f"Unknown subset_mode for {alignment.output_name}: {alignment.subset_mode}")

    output_path = manifest_dir / f"{alignment.output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    progress_write(
        f"[align] wrote {output_path} ({len(out_df)} rows)",
        enabled=progress_on,
    )


    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch PathoROB datasets from Hugging Face, convert parquet rows to local PNG tiles, "
            "and emit MaRI-compatible manifests aligned with PathoROB experimental configurations."
        )
    )
    parser.add_argument(
        "--datasets",
        default="camelyon,tcga,tolkach_esca",
        help=(
            "Comma-separated dataset keys to prepare "
            "(supported: camelyon, tcga, tolkach_esca; default: camelyon,tcga,tolkach_esca)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="PathoROB root directory where sibling dataset folders will be created.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/pathorob/manifests"),
        help="Directory where generated manifests are written.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="HF revision to fetch (default: main).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face access token. If omitted, uses local auth.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Parquet conversion batch size (default: 512).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max workers for Hugging Face download (default: 8).",
    )
    parser.add_argument(
        "--progress",
        default="auto",
        choices=["auto", "on", "off"],
        help="Progress display mode (default: auto).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    progress_on = resolve_progress_mode(str(args.progress))

    requested_keys = [part.strip() for part in str(args.datasets).split(",") if part.strip()]
    if not requested_keys:
        raise ValueError("--datasets must include at least one key")

    unknown = [k for k in requested_keys if k not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset key(s): {unknown}. Known: {sorted(DATASETS.keys())}")

    pathorob_root = Path(args.output_dir).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = Path("data/pathorob/metadata")

    progress_write(f"[prepare] pathorob_root={pathorob_root}", enabled=progress_on)
    progress_write(f"[prepare] manifest_dir={manifest_dir}", enabled=progress_on)
    progress_write(f"[prepare] metadata_dir={metadata_dir}", enabled=progress_on)

    # Phase A: Extract
    for key in requested_keys:
        spec = DATASETS[key]
        extract_dataset(
            spec=spec,
            pathorob_root=pathorob_root,
            manifest_dir=manifest_dir,
            revision=str(args.revision),
            hf_token=args.hf_token,
            batch_size=int(args.batch_size),
            max_workers=int(args.max_workers),
            progress_on=progress_on,
        )

    # Phase B: Align
    if not metadata_dir.exists():
        progress_write(
            f"[prepare] PathoROB metadata dir not found: {metadata_dir}. Skipping alignment.",
            enabled=progress_on,
        )
        return 0

    for alignment in ALIGNMENTS:
        if alignment.dataset_key not in requested_keys:
            continue
        metadata_path = metadata_dir / alignment.metadata_filename
        if not metadata_path.exists():
            progress_write(
                f"[align] skipping {alignment.output_name}: metadata file not found ({metadata_path})",
                enabled=progress_on,
            )
            continue
        align_dataset(
            alignment=alignment,
            metadata_dir=metadata_dir,
            manifest_dir=manifest_dir,
            progress_on=progress_on,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
