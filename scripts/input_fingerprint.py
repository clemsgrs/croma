import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

_MANIFEST_COLUMNS = ("sample_id", "label", "medical_center", "slide_id", "image_path")
_OPTIONAL_MANIFEST_COLUMNS = ("subset",)


def _sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_manifest_value(value: object) -> str:
    return str(value).strip()


def manifest_fingerprint(manifest: pd.DataFrame) -> str:
    missing = [c for c in _MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing required columns for fingerprinting: {missing}")

    columns = list(_MANIFEST_COLUMNS) + [c for c in _OPTIONAL_MANIFEST_COLUMNS if c in manifest.columns]
    rows = [
        [_normalize_manifest_value(v) for v in row]
        for row in manifest.loc[:, columns].itertuples(index=False, name=None)
    ]
    payload = {
        "columns": columns,
        "rows": rows,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _sidecar_payload(sidecar_path: Path) -> dict:
    if not sidecar_path.exists():
        return {}
    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    keys = ("manifest", "manifest_fingerprint", "model_id", "extract", "mixed_precision", "n_samples", "embedding_dim")
    return {key: raw.get(key) for key in keys if key in raw}


def embedding_fingerprint(embedding_path: Path) -> str:
    path = Path(embedding_path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    stat = path.stat()
    arr = np.load(path, mmap_mode="r")
    sidecar_path = path.with_suffix(path.suffix + ".json")

    payload = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "shape": [int(x) for x in arr.shape],
        "dtype": str(arr.dtype),
        "sidecar": _sidecar_payload(sidecar_path),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
