import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import torch
except ModuleNotFoundError:
    torch = None

import layout
from croma.alignment import build_embedding_source_manifest
from croma.metrics.pairs import load_manifest, retain_complete_subset_memberships
from input_fingerprint import manifest_fingerprint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-image embedding files into benchmark-compatible per-model "
            "stacked `.npy` caches."
        )
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to evaluation manifest CSV."
    )
    parser.add_argument(
        "--confounder-column",
        required=True,
        help="Manifest column to treat as the non-biological confounder.",
    )
    parser.add_argument(
        "--mapping-csv",
        required=True,
        type=Path,
        help="CSV with columns image_path,model,embedding_path.",
    )
    parser.add_argument(
        "--tileset",
        required=True,
        help=(
            "Tileset name; the stacked per-model matrices and their row-order manifest "
            "are written to output/embeddings/<tileset>/ (see scripts/bench/layout.py)."
        ),
    )
    parser.add_argument(
        "--models",
        default="",
        help="Optional comma-separated model subset. Defaults to all models in the mapping CSV.",
    )
    parser.add_argument(
        "--evaluation-design",
        default="dataset_wide",
        choices=["paired_2x2", "dataset_wide"],
        help="Must match the evaluation design you will pass to benchmark.",
    )
    return parser.parse_args()


def _parse_models(raw_models: str) -> list[str]:
    if not str(raw_models).strip():
        return []
    parsed = [str(model).strip() for model in str(raw_models).split(",")]
    if any(not model for model in parsed):
        raise ValueError("models list contains an empty entry")
    deduped = list(dict.fromkeys(parsed))
    if len(deduped) != len(parsed):
        raise ValueError("models list contains duplicate entries")
    return deduped


def _prepare_eval_manifest(
    *,
    manifest_df: pd.DataFrame,
    dataset_name: str,
    evaluation_design: str,
) -> pd.DataFrame:
    eval_manifest = manifest_df.copy()
    if evaluation_design == "paired_2x2":
        if "subset" not in eval_manifest.columns:
            raise ValueError(
                f"manifest for dataset '{dataset_name}' must define a 'subset' column for paired_2x2 evaluation"
            )
        eval_manifest = retain_complete_subset_memberships(eval_manifest)
        if len(eval_manifest) == 0:
            raise ValueError(
                f"No evaluable subset-defined samples remain for dataset '{dataset_name}'"
            )
    elif len(eval_manifest) == 0:
        raise ValueError(f"No evaluable samples remain for dataset '{dataset_name}'")

    return eval_manifest.reset_index(drop=True)


def _load_mapping(mapping_csv: Path) -> pd.DataFrame:
    mapping_df = pd.read_csv(mapping_csv, dtype=str)
    required = ("image_path", "model", "embedding_path")
    missing = [col for col in required if col not in mapping_df.columns]
    if missing:
        raise ValueError(f"mapping CSV is missing required columns: {missing}")
    out = mapping_df.loc[:, list(required)].copy()
    for col in required:
        out[col] = out[col].map(lambda v: str(v).strip())
    if (out == "").any().any():
        raise ValueError("mapping CSV contains blank required values")
    return out.reset_index(drop=True)


def _resolve_embedding_path(raw_path: str, *, mapping_csv: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (mapping_csv.parent / path).resolve()
    return path


def _load_npz_embedding(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        if "embedding" in payload.files:
            return np.asarray(payload["embedding"])
        if len(payload.files) == 1:
            return np.asarray(payload[payload.files[0]])
        raise ValueError(
            f"NPZ embedding file must contain exactly one array or an 'embedding' key: {path}"
        )


def _load_pt_embedding(path: Path) -> np.ndarray:
    if torch is None:
        raise RuntimeError(
            f"torch is required to load .pt embedding files: {path}"
        )
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.detach().cpu().numpy()
    if isinstance(payload, dict):
        if "embedding" in payload:
            value = payload["embedding"]
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)
        if len(payload) == 1:
            value = next(iter(payload.values()))
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
            return np.asarray(value)
    raise ValueError(
        f"Unsupported .pt embedding payload at {path}; expected a tensor or a dict containing one array-like value"
    )


def _load_embedding_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz_embedding(path)
    if suffix == ".npy":
        return np.load(path)
    if suffix in {".pt", ".pth"}:
        return _load_pt_embedding(path)
    raise ValueError(
        f"Unsupported embedding file extension for {path}; expected .npz, .npy, .pt, or .pth"
    )


def _coerce_embedding_vector(arr: np.ndarray, *, path: Path) -> np.ndarray:
    values = np.asarray(arr)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and int(values.shape[0]) == 1:
        return values[0]
    raise ValueError(
        f"Embedding at {path} must have shape (d,) or (1, d); got {tuple(int(v) for v in values.shape)}"
    )


def _mapping_for_model(
    *,
    mapping_df: pd.DataFrame,
    model: str,
    mapping_csv: Path,
) -> dict[str, Path]:
    model_rows = mapping_df.loc[mapping_df["model"] == str(model)].copy()
    if len(model_rows) == 0:
        raise ValueError(f"mapping CSV does not contain any rows for model '{model}'")
    duplicated = model_rows.duplicated(subset=["image_path"], keep=False)
    if bool(duplicated.any()):
        dup_paths = sorted(model_rows.loc[duplicated, "image_path"].astype(str).unique())
        raise ValueError(
            f"mapping CSV contains duplicate image_path rows for model '{model}': {dup_paths}"
        )
    return {
        str(row["image_path"]): _resolve_embedding_path(
            str(row["embedding_path"]), mapping_csv=mapping_csv
        )
        for _, row in model_rows.iterrows()
    }


def prepare_benchmark_embeddings(
    *,
    manifest_path: Path,
    confounder_column: str,
    mapping_csv: Path,
    tileset: str,
    models: list[str] | None = None,
    evaluation_design: str = "dataset_wide",
) -> dict:
    manifest = load_manifest(str(manifest_path), confounder_column=confounder_column)
    eval_manifest = _prepare_eval_manifest(
        manifest_df=manifest,
        dataset_name=str(manifest_path.stem),
        evaluation_design=str(evaluation_design),
    )
    # The unique-tile manifest is the tileset's row-order contract: row i of every
    # <Model>.npy describes row i of manifest.csv (see scripts/bench/layout.py).
    embedding_manifest, _ = build_embedding_source_manifest(eval_manifest)
    mapping_df = _load_mapping(mapping_csv)

    selected_models = list(models) if models else sorted(mapping_df["model"].unique())
    if not selected_models:
        raise ValueError("No models were provided and mapping CSV contains no model rows")

    embeddings_dir = layout.embeddings_dir(str(tileset))
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    tileset_manifest_path = layout.tileset_manifest(str(tileset))
    embedding_manifest.to_csv(tileset_manifest_path, index=False)
    embedding_manifest_fp = manifest_fingerprint(embedding_manifest)

    outputs: dict[str, str] = {}
    source_image_paths = embedding_manifest["image_path"].astype(str).tolist()

    for model in selected_models:
        path_by_image = _mapping_for_model(
            mapping_df=mapping_df,
            model=str(model),
            mapping_csv=mapping_csv,
        )
        missing = [image_path for image_path in source_image_paths if image_path not in path_by_image]
        if missing:
            raise ValueError(
                f"mapping CSV is missing embeddings for model '{model}' and image_path values: {missing}"
            )

        stacked_rows: list[np.ndarray] = []
        embedding_dim: int | None = None
        for image_path in source_image_paths:
            embedding_path = path_by_image[str(image_path)]
            if not embedding_path.exists():
                raise FileNotFoundError(
                    f"Embedding file not found for model '{model}': {embedding_path}"
                )
            vector = _coerce_embedding_vector(
                _load_embedding_array(embedding_path), path=embedding_path
            )
            if embedding_dim is None:
                embedding_dim = int(vector.shape[0])
            elif int(vector.shape[0]) != embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for model '{model}': expected {embedding_dim}, got {int(vector.shape[0])} at {embedding_path}"
                )
            stacked_rows.append(np.asarray(vector))

        stacked = np.stack(stacked_rows, axis=0)
        output_path = layout.embedding_path(str(tileset), str(model))
        np.save(output_path, stacked)
        output_path.with_suffix(output_path.suffix + ".json").write_text(
            json.dumps(
                {
                    "manifest": str(tileset_manifest_path),
                    "manifest_fingerprint": embedding_manifest_fp,
                    "n_samples": int(stacked.shape[0]),
                    "embedding_dim": int(stacked.shape[1]),
                    "model_id": str(model),
                    "extract": "precomputed",
                    "mixed_precision": False,
                    "source": "prepare_benchmark_embeddings",
                    "mapping_csv": str(mapping_csv),
                    "evaluation_design": str(evaluation_design),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs[str(model)] = str(output_path)

    return {
        "manifest": str(manifest_path),
        "mapping_csv": str(mapping_csv),
        "tileset": str(tileset),
        "tileset_dir": str(embeddings_dir),
        "tileset_manifest": str(tileset_manifest_path),
        "embedding_manifest_rows": int(len(embedding_manifest)),
        "evaluation_design": str(evaluation_design),
        "models": [str(model) for model in selected_models],
        "outputs": outputs,
    }


def main() -> int:
    args = _parse_args()
    summary = prepare_benchmark_embeddings(
        manifest_path=Path(args.manifest),
        confounder_column=str(args.confounder_column),
        mapping_csv=Path(args.mapping_csv),
        tileset=str(args.tileset),
        models=_parse_models(str(args.models)),
        evaluation_design=str(args.evaluation_design),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
