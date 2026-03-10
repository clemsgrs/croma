import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import hashlib
import numpy as np
from metrics_io import safe_model_name

CACHE_SCHEMA_VERSION = 2
CACHE_CODE_FINGERPRINT = "cross-margin-rename-v1"


def _normalize_for_key(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _normalize_for_key(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set)):
        normalized = [_normalize_for_key(v) for v in value]
        by_serialized = {
            json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True): x
            for x in normalized
        }
        return [by_serialized[key] for key in sorted(by_serialized)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json(payload: dict) -> str:
    normalized = _normalize_for_key(payload)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_cache_key(
    *,
    artifact_name: str,
    model: str,
    input_fingerprint: dict,
    params: dict,
    schema_version: int = CACHE_SCHEMA_VERSION,
    code_fingerprint: str = CACHE_CODE_FINGERPRINT,
) -> dict:
    canonical_payload = {
        "schema_version": int(schema_version),
        "artifact_name": str(artifact_name),
        "model": str(model),
        "input_fingerprint": _normalize_for_key(input_fingerprint),
        "params": _normalize_for_key(params),
        "code_fingerprint": str(code_fingerprint),
    }
    key_hash = hashlib.sha256(_canonical_json(canonical_payload).encode("utf-8")).hexdigest()
    key = dict(canonical_payload)
    key["key_hash"] = key_hash
    return key

def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _atomic_write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".npy") as tmp:
        np.save(tmp, np.asarray(values))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


class MetricsArtifactCache:
    def __init__(self, *, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.cache_dir = self.results_dir / "cache"
        self.artifacts_dir = self.cache_dir / "artifacts"
        self.index_path = self.cache_dir / "index.jsonl"

    def _artifact_path(self, *, artifact_name: str, model: str, key_hash: str, suffix: str) -> Path:
        return self.artifacts_dir / str(artifact_name) / safe_model_name(model) / f"{key_hash}{suffix}"

    def _load_index(self) -> dict[str, dict]:
        if not self.index_path.exists():
            return {}
        rows: dict[str, dict] = {}
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(row, dict):
                continue
            key_hash = str(row.get("key_hash", "")).strip()
            if not key_hash:
                continue
            rows[key_hash] = row
        return rows

    def _save_index(self, rows_by_hash: dict[str, dict]) -> None:
        serialized_rows = []
        for key_hash in sorted(rows_by_hash):
            serialized_rows.append(json.dumps(rows_by_hash[key_hash], sort_keys=True))
        payload = ("\n".join(serialized_rows) + "\n") if serialized_rows else ""
        _atomic_write_text(self.index_path, payload)

    def _upsert_index_row(self, *, key: dict, payload_kind: str, payload_path: Path) -> None:
        rows = self._load_index()
        rows[str(key["key_hash"])] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "schema_version": int(key["schema_version"]),
            "artifact_name": str(key["artifact_name"]),
            "model": str(key["model"]),
            "input_fingerprint": key["input_fingerprint"],
            "params": key["params"],
            "code_fingerprint": str(key["code_fingerprint"]),
            "key_hash": str(key["key_hash"]),
            "payload_kind": str(payload_kind),
            "payload_path": str(payload_path),
        }
        self._save_index(rows)

    def get_json(self, *, key: dict) -> dict | None:
        key_hash = str(key["key_hash"])
        path = self._artifact_path(
            artifact_name=str(key["artifact_name"]),
            model=str(key["model"]),
            key_hash=key_hash,
            suffix=".json",
        )
        rows = self._load_index()
        row = rows.get(key_hash)
        if row is None or str(row.get("payload_kind")) != "json":
            return None
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def put_json(self, *, key: dict, payload: dict) -> Path:
        key_hash = str(key["key_hash"])
        path = self._artifact_path(
            artifact_name=str(key["artifact_name"]),
            model=str(key["model"]),
            key_hash=key_hash,
            suffix=".json",
        )
        _atomic_write_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")
        self._upsert_index_row(key=key, payload_kind="json", payload_path=path)
        return path

    def get_npy(self, *, key: dict) -> np.ndarray | None:
        key_hash = str(key["key_hash"])
        path = self._artifact_path(
            artifact_name=str(key["artifact_name"]),
            model=str(key["model"]),
            key_hash=key_hash,
            suffix=".npy",
        )
        rows = self._load_index()
        row = rows.get(key_hash)
        if row is None or str(row.get("payload_kind")) != "npy":
            return None
        if not path.exists():
            return None
        try:
            return np.load(path)
        except Exception:  # noqa: BLE001
            return None

    def put_npy(self, *, key: dict, values: np.ndarray) -> Path:
        key_hash = str(key["key_hash"])
        path = self._artifact_path(
            artifact_name=str(key["artifact_name"]),
            model=str(key["model"]),
            key_hash=key_hash,
            suffix=".npy",
        )
        _atomic_write_npy(path, np.asarray(values))
        self._upsert_index_row(key=key, payload_kind="npy", payload_path=path)
        return path
