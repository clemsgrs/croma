"""Crash-safe publication and compatibility checks for tile embeddings."""

import dataclasses
import json
import os
import tempfile
from pathlib import Path

import numpy as np

PROVENANCE_FIELDS = (
    "checkpoint_revision",
    "extraction_contract",
    "precision",
    "manifest_fingerprint",
    "batch_size",
    "output_dtype",
    "output_shape",
)


class ArtifactCompatibilityError(RuntimeError):
    """An existing embedding artifact cannot be safely reused."""


@dataclasses.dataclass(frozen=True)
class EmbeddingArtifactContract:
    """Provenance and matrix properties that define one reusable artifact."""

    checkpoint_revision: str | None
    extraction_contract: dict
    precision: str
    manifest_fingerprint: str
    batch_size: int
    output_dtype: str
    output_shape: tuple[int, int | None]

    def sidecar_payload(self) -> dict:
        if self.output_shape[1] is None:
            raise ValueError("cannot publish an artifact without a known embedding dimension")
        payload = dataclasses.asdict(self)
        payload["output_shape"] = list(self.output_shape)
        return payload


def sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def artifact_is_reusable(output_path: Path, expected: EmbeddingArtifactContract) -> bool:
    """Return whether a complete existing artifact matches ``expected``.

    A matrix and its sidecar are one artifact. A lone member is an interrupted or
    otherwise partial publication and is rejected rather than treated as a cache miss.
    """

    output_path = Path(output_path)
    metadata_path = sidecar_path(output_path)
    matrix_exists = output_path.exists()
    sidecar_exists = metadata_path.exists()
    if not matrix_exists and not sidecar_exists:
        return False
    if matrix_exists != sidecar_exists:
        raise ArtifactCompatibilityError(
            f"incomplete embedding artifact: expected both {output_path} and {metadata_path}"
        )

    try:
        sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ArtifactCompatibilityError(f"invalid embedding sidecar: {metadata_path}") from exc
    if not isinstance(sidecar, dict):
        raise ArtifactCompatibilityError(
            f"invalid embedding sidecar: {metadata_path} must contain an object"
        )

    expected_payload = dataclasses.asdict(expected)
    expected_shape = expected_payload.pop("output_shape")
    mismatches = [
        key
        for key, value in expected_payload.items()
        if key not in sidecar or sidecar[key] != value
    ]
    stored_shape = sidecar.get("output_shape")
    valid_stored_shape = (
        isinstance(stored_shape, list)
        and len(stored_shape) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in stored_shape)
    )
    if not valid_stored_shape:
        mismatches.append("output_shape")
    elif stored_shape[0] != expected_shape[0] or (
        expected_shape[1] is not None and stored_shape[1] != expected_shape[1]
    ):
        mismatches.append("output_shape")
    if mismatches:
        raise ArtifactCompatibilityError(
            "incompatible embedding artifact provenance: " + ", ".join(mismatches)
        )

    try:
        array = np.load(output_path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001
        raise ArtifactCompatibilityError(f"invalid embedding matrix: {output_path}") from exc
    actual_shape = tuple(int(value) for value in array.shape)
    actual_dtype = str(array.dtype)
    complete_shape = tuple(int(value) for value in stored_shape)
    if actual_shape != complete_shape or actual_dtype != expected.output_dtype:
        raise ArtifactCompatibilityError(
            "incompatible embedding matrix: "
            f"expected shape={complete_shape}, dtype={expected.output_dtype}; "
            f"got shape={actual_shape}, dtype={actual_dtype}"
        )
    return True


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_embedding_artifact(
    output_path: Path,
    embeddings: np.ndarray,
    contract: EmbeddingArtifactContract,
) -> None:
    """Publish a matrix, then its sidecar completion marker, durably and atomically."""

    output_path = Path(output_path)
    metadata_path = sidecar_path(output_path)
    array = np.asarray(embeddings)
    actual_shape = tuple(int(value) for value in array.shape)
    actual_dtype = str(array.dtype)
    if actual_shape != contract.output_shape or actual_dtype != contract.output_dtype:
        raise ValueError(
            "embedding matrix does not satisfy its artifact contract: "
            f"expected shape={contract.output_shape}, dtype={contract.output_dtype}; "
            f"got shape={actual_shape}, dtype={actual_dtype}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_tmp: Path | None = None
    sidecar_tmp: Path | None = None
    sidecar_committed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output_path.name}.", dir=output_path.parent, delete=False
        ) as handle:
            matrix_tmp = Path(handle.name)
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{metadata_path.name}.",
            dir=metadata_path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            sidecar_tmp = Path(handle.name)
            json.dump(contract.sidecar_payload(), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # The sidecar is the completion marker. Its removal must be durable before the
        # matrix changes, and the new matrix must be durable before the marker appears.
        metadata_path.unlink(missing_ok=True)
        _fsync_directory(output_path.parent)
        os.replace(matrix_tmp, output_path)
        matrix_tmp = None
        _fsync_directory(output_path.parent)
        os.replace(sidecar_tmp, metadata_path)
        sidecar_tmp = None
        sidecar_committed = True
        _fsync_directory(output_path.parent)
    except BaseException:
        if sidecar_committed:
            metadata_path.unlink(missing_ok=True)
            try:
                _fsync_directory(output_path.parent)
            except OSError:
                pass
        raise
    finally:
        if matrix_tmp is not None:
            matrix_tmp.unlink(missing_ok=True)
        if sidecar_tmp is not None:
            sidecar_tmp.unlink(missing_ok=True)
