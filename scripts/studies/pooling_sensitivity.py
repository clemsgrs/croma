"""Isolated pooling-sensitivity study orchestration.

This module is deliberately outside the benchmark package. Alternative representations
live under ``output/studies/pooling-sensitivity`` and therefore cannot be discovered as
canonical model embeddings by the normal benchmark driver.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import shlex
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import config_context

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from croma.metrics.neighbors import _select_k_from_balanced_accuracy  # noqa: E402
from croma import CRoMa, MaRI, RI, __version__ as croma_version  # noqa: E402
from croma.metrics.base import _PreparedNeighborSubset  # noqa: E402
from croma.metrics.bootstrap import paired_cluster_bootstrap_delta  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
from croma.metrics.mari import TAU_FALLBACK  # noqa: E402
from croma.metrics.neighbors import (  # noqa: E402
    _balanced_accuracy_by_k_from_prepared_neighbors,
)
from croma.metrics.pairs import (  # noqa: E402
    EvaluationSubset,
    normalize_manifest,
    resolve_manifest_subsets,
)
from embedding_artifacts import (  # noqa: E402
    ArtifactCompatibilityError,
    artifact_is_reusable,
    sidecar_path,
)
import extract_embeddings as extraction  # noqa: E402
import benchmarks as benchmark_registry  # noqa: E402
from model_registry import ModelSpec, _build_model_registry  # noqa: E402
from run_config import resolve_sweep_k_values  # noqa: E402
import views as benchmark_views  # noqa: E402

STUDY_MODELS = (
    "Mascaret",
    "Phaet",
    "RudolfV 2",
    "RudolfV 2-B",
    "RudolfV 2-S",
)

STUDY_ALTERNATIVE_POOLING = {
    "Mascaret": "cls-mean-patch",
    "Phaet": "cls-mean-patch",
    "RudolfV 2": "cls-only",
    "RudolfV 2-B": "cls-only",
    "RudolfV 2-S": "cls-only",
}

PATHOROB_TILESETS = (
    "pathorob-camelyon",
    "pathorob-tolkach-esca",
    "pathorob-tcga-2x2",
    "pathorob-tcga-4x4",
)

PRESERVATION_BASELINE_NAME = "preservation-baseline.json"
TRACER_BENCHMARK = "pathorob-camelyon"
TRACER_TILESET = "pathorob-camelyon"
TRACER_MODEL = "Mascaret"
TRACER_ALTERNATIVE = "cls-mean-patch"
NEIGHBOR_WORKING_MEMORY_MIB = 128


@dataclass(frozen=True)
class StudyModelPlan:
    """One canonical encoder and its quarantined pooling alternative."""

    alternative: str
    alternative_width: int
    batch_size: int


@dataclass(frozen=True)
class StudyModelIdentity:
    """Machine identity retained separately from publication spelling."""

    published_name: str
    variant_role: str
    parent_registry_id: str | None
    canonical_width: int


@dataclass(frozen=True)
class StudyBenchmarkPlan:
    """Publication-frozen operating contract for one PathoROB benchmark."""

    tileset: str
    evaluation_design: str
    fixed_k: int
    biological_k_max: int
    diagnostic_k_max: int | None


@dataclass(frozen=True)
class StudyRunPlan:
    """One model-by-benchmark cell in a declared sensitivity panel."""

    model_id: str
    model_plan: StudyModelPlan
    identity: StudyModelIdentity | None
    benchmark_name: str
    benchmark: StudyBenchmarkPlan


WAIV_STUDY_MODELS = {
    "Mascaret": StudyModelPlan(
        alternative="cls-mean-patch",
        alternative_width=3072,
        batch_size=32,
    ),
    "Phaet": StudyModelPlan(
        alternative="cls-mean-patch",
        alternative_width=2048,
        batch_size=64,
    ),
}

RUDOLFV2_STUDY_MODELS = {
    "RudolfV 2": StudyModelPlan(
        alternative="cls-only",
        alternative_width=1536,
        batch_size=32,
    ),
    "RudolfV 2-B": StudyModelPlan(
        alternative="cls-only",
        alternative_width=768,
        batch_size=32,
    ),
    "RudolfV 2-S": StudyModelPlan(
        alternative="cls-only",
        alternative_width=384,
        batch_size=64,
    ),
}

POOLING_STUDY_MODELS = {**WAIV_STUDY_MODELS, **RUDOLFV2_STUDY_MODELS}


def _load_rudolfv2_model_identities() -> dict[str, StudyModelIdentity]:
    """Load teacher/student and report identities from benchmark metadata."""

    metadata = pd.read_csv(
        REPO / "scripts" / "bench" / "model_metadata.csv",
        keep_default_na=False,
        na_values=[],
    ).set_index("model")
    identities: dict[str, StudyModelIdentity] = {}
    for model_id in RUDOLFV2_STUDY_MODELS:
        if model_id not in metadata.index:
            raise RuntimeError(f"study model is missing from model metadata: {model_id!r}")
        row = metadata.loc[model_id]
        try:
            canonical_width = int(str(row["dim"]).strip().removeprefix("$").removesuffix("$"))
        except ValueError as exc:
            raise RuntimeError(
                f"study model has an invalid published embedding dimension: {model_id!r}"
            ) from exc
        parent = str(row["parent_model"]).strip()
        identities[model_id] = StudyModelIdentity(
            published_name=str(row["published_name"]).strip(),
            variant_role=str(row["variant_role"]).strip(),
            parent_registry_id=parent or None,
            canonical_width=int(canonical_width),
        )
    return identities


RUDOLFV2_MODEL_IDENTITIES = _load_rudolfv2_model_identities()

PATHOROB_STUDY_BENCHMARKS = {
    "pathorob-camelyon": StudyBenchmarkPlan(
        tileset="pathorob-camelyon",
        evaluation_design="all",
        fixed_k=11,
        biological_k_max=600,
        diagnostic_k_max=300,
    ),
    "pathorob-tcga-2x2": StudyBenchmarkPlan(
        tileset="pathorob-tcga-2x2",
        evaluation_design="paired_2x2",
        fixed_k=61,
        biological_k_max=1200,
        diagnostic_k_max=None,
    ),
    "pathorob-tcga-4x4": StudyBenchmarkPlan(
        tileset="pathorob-tcga-4x4",
        evaluation_design="all",
        fixed_k=71,
        biological_k_max=600,
        diagnostic_k_max=None,
    ),
    "pathorob-tolkach-esca": StudyBenchmarkPlan(
        tileset="pathorob-tolkach-esca",
        evaluation_design="all",
        fixed_k=61,
        biological_k_max=1000,
        diagnostic_k_max=None,
    ),
}


def build_study_plan(
    *,
    model_ids: tuple[str, ...],
    model_plans: dict[str, StudyModelPlan] | None = None,
    benchmark_ids: tuple[str, ...] | None = None,
) -> tuple[StudyRunPlan, ...]:
    """Build a deterministic model-major matrix from frozen declarations."""

    plans = POOLING_STUDY_MODELS if model_plans is None else model_plans
    unknown = [model_id for model_id in model_ids if model_id not in plans]
    if unknown:
        raise ValueError(f"models do not have pooling-sensitivity plans: {unknown}")
    selected_benchmarks = (
        tuple(PATHOROB_STUDY_BENCHMARKS) if benchmark_ids is None else tuple(benchmark_ids)
    )
    unknown_benchmarks = [
        name for name in selected_benchmarks if name not in PATHOROB_STUDY_BENCHMARKS
    ]
    if unknown_benchmarks:
        raise ValueError(f"unknown pooling-sensitivity benchmarks: {unknown_benchmarks}")
    return tuple(
        StudyRunPlan(
            model_id=model_id,
            model_plan=plans[model_id],
            identity=RUDOLFV2_MODEL_IDENTITIES.get(model_id),
            benchmark_name=benchmark_name,
            benchmark=benchmark_plan,
        )
        for model_id in model_ids
        for benchmark_name in selected_benchmarks
        for benchmark_plan in (PATHOROB_STUDY_BENCHMARKS[benchmark_name],)
    )


@dataclass(frozen=True)
class RepresentationEvaluation:
    """One representation evaluated under the pooling-sensitivity protocol."""

    representation: str
    fixed_k: int
    biological_knn_bacc: float
    confounder_knn_bacc: float
    biological_kstar: int
    biological_kstar_bacc: float
    diagnostic_kstar_300: int | None
    diagnostic_kstar_300_bacc: float | None
    tau: float
    ri: float
    mari: float
    support: float
    ss_dominated_undefined_frac: float
    oo_dominated_undefined_frac: float
    mixed_undefined_frac: float
    croma: float
    croma_f0: float
    croma_ltm10: float
    croma_result: object


@dataclass(frozen=True)
class StudyRun:
    """One paired canonical/alternative result ready for panel publication."""

    benchmark: str
    tileset: str
    model: str
    canonical: RepresentationEvaluation
    alternative: RepresentationEvaluation
    aligned_manifest: pd.DataFrame
    provenance_inputs: dict
    identity: StudyModelIdentity | None = None


@dataclass(frozen=True)
class StudyCellRequest:
    """Serializable inputs for one isolated model-by-benchmark evaluation."""

    benchmark: str
    model: str
    inventory_entry: tuple[Path, str]
    canonical_root: Path
    study_root: Path
    eval_manifest_root: Path
    device_arg: str
    model_plans: dict[str, StudyModelPlan]
    canonical_access_root: Path | None = None
    study_access_root: Path | None = None
    eval_manifest_access_root: Path | None = None
    eval_manifest_override: Path | None = None


_COMPARISON_MEASURES = (
    "biological_knn_bacc",
    "confounder_knn_bacc",
    "biological_kstar",
    "biological_kstar_bacc",
    "diagnostic_kstar_300",
    "diagnostic_kstar_300_bacc",
    "tau",
    "ri",
    "mari",
    "support",
    "ss_dominated_undefined_frac",
    "oo_dominated_undefined_frac",
    "mixed_undefined_frac",
    "croma",
    "croma_f0",
    "croma_ltm10",
)


def select_biological_kstars(
    scores: dict[int, float],
    *,
    production_k_max: int = 600,
    diagnostic_k_max: int | None = 300,
) -> tuple[tuple[int, float], tuple[int, float] | None]:
    """Select production and optional diagnostic k* with smallest-k ties."""

    production_grid = resolve_sweep_k_values(production_k_max, "sparse")
    missing = [k for k in production_grid if k not in scores]
    if missing:
        raise ValueError(f"biological k* scores are missing sparse k values: {missing}")
    production_k = _select_k_from_balanced_accuracy(k_values=production_grid, scores=scores)
    if diagnostic_k_max is None:
        return (int(production_k), float(scores[production_k])), None
    diagnostic_grid = resolve_sweep_k_values(diagnostic_k_max, "sparse")
    diagnostic_k = _select_k_from_balanced_accuracy(k_values=diagnostic_grid, scores=scores)
    return (
        (int(production_k), float(scores[production_k])),
        (int(diagnostic_k), float(scores[diagnostic_k])),
    )


def _prepare_streamed_paired_evaluation(
    *,
    features: np.ndarray,
    subsets: list[EvaluationSubset],
    production_k_values: list[int],
    fixed_k: int,
    warn_context: str,
) -> tuple[list[_PreparedNeighborSubset], dict[int, float]]:
    """Stream full-k subset caches while retaining only fixed-k columns."""

    fixed_caches: list[_PreparedNeighborSubset] = []
    full_k_values = sorted({int(fixed_k), *production_k_values})

    def full_caches() -> Iterator[_PreparedNeighborSubset]:
        for full_cache in RI._iter_paired_subset_neighbor_cache(
            features=features,
            subsets=subsets,
            k_values=full_k_values,
            assume_normalized=False,
        ):
            width = min(int(fixed_k), int(full_cache.neigh_idx.shape[1]))
            fixed_caches.append(
                replace(
                    full_cache,
                    neigh_idx=full_cache.neigh_idx[:, :width].copy(),
                    neigh_dist=full_cache.neigh_dist[:, :width].copy(),
                    valid_counts=np.minimum(full_cache.valid_counts, int(fixed_k)),
                )
            )
            yield full_cache

    biological_scores = RI._knn_balanced_accuracy_by_k_from_prepared_subsets(
        prepared_subsets=full_caches(),
        target="label",
        k_values=production_k_values,
        warn_context=warn_context,
    )
    return fixed_caches, biological_scores


def evaluate_representation(
    *,
    representation: str,
    features: np.ndarray,
    manifest: pd.DataFrame,
    confounder_column: str,
    evaluation_design: str = "all",
    fixed_k: int = 11,
    production_k_max: int = 600,
    diagnostic_k_max: int | None = 300,
    headline_m: int = CROMA_HEADLINE_M,
    croma_start_k: int = 200,
) -> RepresentationEvaluation:
    """Evaluate one representation with one design-specific neighbour cache."""

    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] != len(manifest):
        raise ValueError("features must be a 2-D matrix aligned one-to-one with manifest")
    if not np.isfinite(features).all():
        raise ValueError(f"representation {representation!r} contains non-finite values")
    normalized_manifest = normalize_manifest(
        manifest,
        confounder_column=confounder_column,
        source=f"{representation} evaluation manifest",
    )
    production_grid = resolve_sweep_k_values(production_k_max, "sparse")
    k_values = sorted({int(fixed_k), *production_grid})
    if evaluation_design == "paired_2x2":
        prepared, biological_scores = _prepare_streamed_paired_evaluation(
            features=features,
            subsets=resolve_manifest_subsets(normalized_manifest),
            production_k_values=production_grid,
            fixed_k=int(fixed_k),
            warn_context=f"{representation} biological k*",
        )
        fixed_biological = RI._knn_balanced_accuracy_by_k_from_prepared_subsets(
            prepared_subsets=prepared,
            target="label",
            k_values=[int(fixed_k)],
            warn_context=f"{representation} fixed-k biological probe",
        )[int(fixed_k)]
        fixed_confounder = RI._knn_balanced_accuracy_by_k_from_prepared_subsets(
            prepared_subsets=prepared,
            target="confounder",
            k_values=[int(fixed_k)],
            warn_context=f"{representation} fixed-k confounder probe",
        )[int(fixed_k)]
    elif evaluation_design == "all":
        prepared = RI._prepare_all_rows_neighbor_cache(
            features=features,
            df=normalized_manifest,
            k_values=k_values,
        )
        biological_scores = _balanced_accuracy_by_k_from_prepared_neighbors(
            labels=prepared.labels,
            neigh_idx=prepared.neigh_idx,
            valid_counts=prepared.valid_counts,
            k_values=production_grid,
        )
        fixed_biological = _balanced_accuracy_by_k_from_prepared_neighbors(
            labels=prepared.labels,
            neigh_idx=prepared.neigh_idx,
            valid_counts=prepared.valid_counts,
            k_values=[int(fixed_k)],
        )[int(fixed_k)]
        fixed_confounder = _balanced_accuracy_by_k_from_prepared_neighbors(
            labels=prepared.centers,
            neigh_idx=prepared.neigh_idx,
            valid_counts=prepared.valid_counts,
            k_values=[int(fixed_k)],
        )[int(fixed_k)]
    else:
        raise ValueError("evaluation_design must be 'all' or 'paired_2x2'")

    production_selection, diagnostic_selection = select_biological_kstars(
        biological_scores,
        production_k_max=production_k_max,
        diagnostic_k_max=diagnostic_k_max,
    )
    production_k, production_bacc = production_selection
    if diagnostic_selection is None:
        diagnostic_k = None
        diagnostic_bacc = None
    else:
        diagnostic_k, diagnostic_bacc = diagnostic_selection

    dataset_name = RI._infer_dataset_name(normalized_manifest)
    if evaluation_design == "paired_2x2":
        ri_artifacts = RI._compute_artifacts_from_prepared_subsets(
            prepared_subsets=prepared,
            dataset_name=dataset_name,
            k_values=[int(fixed_k)],
            evaluation_design=evaluation_design,
            selected_k=int(fixed_k),
        )
        typed_chunks = [
            MaRI._typed_neighbor_distances_from_neighbors(
                labels=subset.labels,
                centers=subset.centers,
                neigh_idx=subset.neigh_idx,
                neigh_dist=subset.neigh_dist,
                valid_counts=subset.valid_counts,
                k=int(fixed_k),
            )
            for subset in prepared
        ]
        typed_distances = np.concatenate(typed_chunks) if typed_chunks else np.empty(0, dtype=float)
    else:
        ri_artifacts = RI._compute_artifacts_from_prepared_all_rows(
            prepared_neighbors=prepared,
            dataset_name=dataset_name,
            k_values=[int(fixed_k)],
            selected_k=int(fixed_k),
        )
        typed_distances = MaRI._typed_neighbor_distances_from_neighbors(
            labels=prepared.labels,
            centers=prepared.centers,
            neigh_idx=prepared.neigh_idx,
            neigh_dist=prepared.neigh_dist,
            valid_counts=prepared.valid_counts,
            k=int(fixed_k),
        )
    recommended_tau = (
        float(np.median(typed_distances)) if int(typed_distances.size) > 0 else float("nan")
    )
    tau = (
        recommended_tau
        if np.isfinite(recommended_tau) and recommended_tau > 0.0
        else float(TAU_FALLBACK)
    )
    if evaluation_design == "paired_2x2":
        mari_artifacts = MaRI._compute_artifacts_from_prepared_subsets(
            prepared_subsets=prepared,
            dataset_name=dataset_name,
            k_values=[int(fixed_k)],
            evaluation_design=evaluation_design,
            selected_k=int(fixed_k),
            tau=float(tau),
        )
    else:
        mari_artifacts = MaRI._compute_artifacts_from_prepared_all_rows(
            prepared_neighbors=prepared,
            dataset_name=dataset_name,
            k_values=[int(fixed_k)],
            selected_k=int(fixed_k),
            tau=float(tau),
        )
    ri_result = ri_artifacts.result
    mari_result = mari_artifacts.result
    if ri_result is None or mari_result is None:
        raise RuntimeError("fixed-k RI/MaRI evaluation did not return a result")
    shared_fields = (
        "support",
        "ss_dominated_undefined_frac",
        "oo_dominated_undefined_frac",
        "mixed_undefined_frac",
    )
    if not np.array_equal(
        ri_result.occurrence_defined_mask, mari_result.occurrence_defined_mask
    ) or any(
        float(getattr(ri_result, field)) != float(getattr(mari_result, field))
        for field in shared_fields
    ):
        raise RuntimeError("fixed-k RI and MaRI must share support and cause fractions")

    ri_value = float(ri_result.value)
    mari_value = float(mari_result.value)
    support = float(ri_result.support)
    ss_dominated_undefined_frac = float(ri_result.ss_dominated_undefined_frac)
    oo_dominated_undefined_frac = float(ri_result.oo_dominated_undefined_frac)
    mixed_undefined_frac = float(ri_result.mixed_undefined_frac)
    del prepared, typed_distances, ri_artifacts, mari_artifacts, ri_result, mari_result
    if evaluation_design == "paired_2x2":
        del typed_chunks

    croma_result = CRoMa.compute(
        features=features,
        manifest=normalized_manifest,
        confounder_column="confounder",
        evaluation_design=evaluation_design,
        m=int(headline_m),
        alpha=0.10,
        start_k=int(croma_start_k),
    )
    return RepresentationEvaluation(
        representation=str(representation),
        fixed_k=int(fixed_k),
        biological_knn_bacc=float(fixed_biological),
        confounder_knn_bacc=float(fixed_confounder),
        biological_kstar=int(production_k),
        biological_kstar_bacc=float(production_bacc),
        diagnostic_kstar_300=(None if diagnostic_k is None else int(diagnostic_k)),
        diagnostic_kstar_300_bacc=(None if diagnostic_bacc is None else float(diagnostic_bacc)),
        tau=float(tau),
        ri=ri_value,
        mari=mari_value,
        support=support,
        ss_dominated_undefined_frac=ss_dominated_undefined_frac,
        oo_dominated_undefined_frac=oo_dominated_undefined_frac,
        mixed_undefined_frac=mixed_undefined_frac,
        croma=float(croma_result.value),
        croma_f0=float(croma_result.f0),
        croma_ltm10=float(croma_result.ltm_alpha),
        croma_result=croma_result,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _resolved_access_root(logical_root: Path, access_root: Path | None) -> Path:
    """Return the explicit read root, defaulting to the logical root."""

    return Path(logical_root if access_root is None else access_root).resolve()


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize arrays as an NPZ with stable member order and ZIP metadata."""

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or "\\" in name:
                raise ValueError(f"invalid NPZ member name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"deterministic NPZ member {name!r} cannot contain objects")
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue())
    return output.getvalue()


def _bundle_targets(study_root: Path, files: dict[Path, bytes]) -> list[tuple[Path, bytes]]:
    if not files:
        raise ValueError("study bundle must contain at least one file")
    root = Path(study_root).resolve()
    targets: list[tuple[Path, bytes]] = []
    seen: set[Path] = set()
    for relative, payload in files.items():
        relative = Path(relative)
        if relative.is_absolute():
            raise ValueError(f"study bundle path must be relative: {relative}")
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"study bundle path escapes the study root: {relative}")
        if target in seen:
            raise ValueError(f"duplicate study bundle target: {relative}")
        seen.add(target)
        targets.append((target, bytes(payload)))
    return targets


def publish_study_bundle(
    study_root: Path,
    files: dict[Path, bytes],
    *,
    check: bool = False,
    force: bool = False,
) -> str:
    """Publish deterministic study files with resume/check/force semantics."""

    if check and force:
        raise ValueError("--check and --force are mutually exclusive")
    targets = _bundle_targets(study_root, files)
    existing = [target.exists() for target, _payload in targets]
    matching = [
        target.is_file() and not target.is_symlink() and target.read_bytes() == payload
        for target, payload in targets
    ]

    if check:
        if not all(matching):
            mismatches = [str(path) for (path, _), match in zip(targets, matching) if not match]
            raise RuntimeError(
                "study check failed; target bytes differ or are missing: " + ", ".join(mismatches)
            )
        return "checked"

    if all(matching):
        return "reused"
    if any(existing) and not force:
        raise RuntimeError(
            "study bundle is partial or incompatible; inspect it or rerun with --force"
        )

    for (target, payload), match in zip(targets, matching):
        if match:
            continue
        if target.is_symlink():
            raise RuntimeError(f"refusing to replace a study symlink: {target}")
        _atomic_write(target, payload)
    return "forced" if force else "written"


_OCCURRENCE_IDENTITY_FIELDS = (
    "occurrence_index",
    "source_sample_index",
    "subset",
    "sample_id",
    "group_id",
)


def build_occurrence_identity(*, result, manifest: pd.DataFrame) -> pd.DataFrame:
    """Resolve a result's complete occurrence identity from its source manifest."""

    missing = [field for field in ("sample_id", "group_id") if field not in manifest]
    if missing:
        raise ValueError(f"occurrence manifest is missing identity columns: {missing}")
    sources = np.asarray(result.occurrence_source_indices, dtype=np.int64)
    subsets = np.asarray(result.occurrence_subsets).astype(str)
    if sources.ndim != 1 or subsets.shape != sources.shape:
        raise ValueError("occurrence source and subset identities must be aligned 1-D arrays")
    if np.any(sources < 0) or np.any(sources >= len(manifest)):
        raise ValueError("occurrence source identity is outside the manifest")
    source_rows = manifest.iloc[sources]
    return pd.DataFrame(
        {
            "occurrence_index": np.arange(len(sources), dtype=np.int64),
            "source_sample_index": sources,
            "subset": subsets,
            "sample_id": source_rows["sample_id"].astype(str).to_numpy(),
            "group_id": source_rows["group_id"].astype(str).to_numpy(),
        }
    )


def validate_occurrence_identity(
    *, canonical_identity: pd.DataFrame, alternative_identity: pd.DataFrame
) -> None:
    """Require exact paired occurrence/source/subset/sample/group identity."""

    for field in _OCCURRENCE_IDENTITY_FIELDS:
        if field not in canonical_identity or field not in alternative_identity:
            raise ValueError(f"occurrence identity mismatch: missing {field}")
        if not np.array_equal(
            canonical_identity[field].to_numpy(),
            alternative_identity[field].to_numpy(),
        ):
            raise ValueError(f"occurrence identity mismatch: {field}")


def align_paired_occurrence_manifest(
    *, canonical, alternative, manifest: pd.DataFrame
) -> pd.DataFrame:
    """Independently derive and validate both representations' occurrence identities."""

    canonical_identity = build_occurrence_identity(result=canonical, manifest=manifest)
    alternative_identity = build_occurrence_identity(result=alternative, manifest=manifest)
    validate_occurrence_identity(
        canonical_identity=canonical_identity,
        alternative_identity=alternative_identity,
    )
    return canonical_identity


def build_occurrence_arrays(
    *, canonical, alternative, aligned_manifest: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Build the paired occurrence artifact after exact identity validation."""

    canonical_values = np.asarray(canonical.sample_values_aligned, dtype=np.float64)
    alternative_values = np.asarray(alternative.sample_values_aligned, dtype=np.float64)
    canonical_sources = np.asarray(canonical.occurrence_source_indices, dtype=np.int64)
    alternative_sources = np.asarray(alternative.occurrence_source_indices, dtype=np.int64)
    canonical_subsets = np.asarray(canonical.occurrence_subsets).astype(str)
    alternative_subsets = np.asarray(alternative.occurrence_subsets).astype(str)
    expected_sources = (
        aligned_manifest["source_sample_index"].to_numpy(dtype=np.int64)
        if "source_sample_index" in aligned_manifest
        else np.arange(len(aligned_manifest), dtype=np.int64)
    )
    expected_occurrences = (
        aligned_manifest["occurrence_index"].to_numpy(dtype=np.int64)
        if "occurrence_index" in aligned_manifest
        else np.arange(len(aligned_manifest), dtype=np.int64)
    )
    expected_subsets = (
        aligned_manifest["subset"].astype(str).to_numpy()
        if "subset" in aligned_manifest
        else np.full(len(aligned_manifest), "dataset", dtype="<U7")
    )
    identity_matches = (
        np.array_equal(
            expected_occurrences,
            np.arange(len(aligned_manifest), dtype=np.int64),
        )
        and np.array_equal(canonical_sources, alternative_sources)
        and np.array_equal(canonical_sources, expected_sources)
        and np.array_equal(canonical_subsets, alternative_subsets)
        and np.array_equal(canonical_subsets, expected_subsets)
    )
    expected_length = len(aligned_manifest)
    if (
        not identity_matches
        or canonical_values.shape != (expected_length,)
        or alternative_values.shape != (expected_length,)
    ):
        raise ValueError(
            "occurrence identity mismatch between canonical, alternative, and manifest order"
        )
    if not np.isfinite(canonical_values).all() or not np.isfinite(alternative_values).all():
        raise ValueError("CRoMa occurrence values must all be finite")

    return {
        "canonical_croma": canonical_values,
        "alternative_croma": alternative_values,
        "occurrence_index": expected_occurrences,
        "source_sample_index": expected_sources,
        "subset": np.asarray(expected_subsets, dtype=str),
        "sample_id": np.asarray(aligned_manifest["sample_id"].astype(str).tolist(), dtype=str),
        "group_id": np.asarray(aligned_manifest["group_id"].astype(str).tolist(), dtype=str),
    }


def build_comparison_frames(
    *,
    benchmark: str,
    tileset: str,
    model: str,
    canonical: RepresentationEvaluation,
    alternative: RepresentationEvaluation,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the paired wide comparison and raw representation ranking tables."""

    if canonical.representation != "canonical":
        raise ValueError("canonical evaluation must use representation='canonical'")
    if canonical.fixed_k != alternative.fixed_k:
        raise ValueError("canonical and alternative must use the same fixed k")
    identity = RUDOLFV2_MODEL_IDENTITIES.get(str(model))

    comparison: dict[str, str | int | float] = {
        "benchmark": str(benchmark),
        "tileset": str(tileset),
        "model": str(model),
        "canonical_representation": canonical.representation,
        "alternative_representation": alternative.representation,
        "fixed_k": int(canonical.fixed_k),
    }
    if identity is not None:
        comparison["variant_role"] = identity.variant_role
        comparison["parent_model"] = identity.parent_registry_id or ""
    for measure in _COMPARISON_MEASURES:
        canonical_value = getattr(canonical, measure)
        alternative_value = getattr(alternative, measure)
        if canonical_value is None or alternative_value is None:
            canonical_value = float("nan")
            alternative_value = float("nan")
        delta = float(alternative_value) - float(canonical_value)
        comparison[f"canonical_{measure}"] = canonical_value
        comparison[f"alternative_{measure}"] = alternative_value
        comparison[f"delta_{measure}"] = delta
        comparison[f"abs_delta_{measure}"] = abs(delta)

    ranking_rows: list[dict[str, str | int | float]] = []
    for evaluation in (canonical, alternative):
        row: dict[str, str | int | float] = {
            "benchmark": str(benchmark),
            "tileset": str(tileset),
            "model": str(model),
            "representation": evaluation.representation,
            "fixed_k": int(evaluation.fixed_k),
        }
        if identity is not None:
            row["variant_role"] = identity.variant_role
            row["parent_model"] = identity.parent_registry_id or ""
        for measure in _COMPARISON_MEASURES:
            value = getattr(evaluation, measure)
            row[measure] = float("nan") if value is None else value
        ranking_rows.append(row)
    rankings = (
        pd.DataFrame(ranking_rows)
        .sort_values(["croma", "representation"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    rank_column = 6 if identity is not None else 4
    rankings.insert(rank_column, "croma_rank", np.arange(1, len(rankings) + 1, dtype=int))
    return pd.DataFrame([comparison]), rankings


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _render_report(
    *,
    benchmark: str,
    model: str,
    canonical: RepresentationEvaluation,
    alternative: RepresentationEvaluation,
    comparison: pd.Series,
) -> bytes:
    benchmark_plan = PATHOROB_STUDY_BENCHMARKS.get(benchmark)
    production_k_max = benchmark_plan.biological_k_max if benchmark_plan is not None else 600
    diagnostic_k_max = benchmark_plan.diagnostic_k_max if benchmark_plan is not None else 300
    diagnostic_sentence = (
        f" The truncated-at-{diagnostic_k_max} diagnostic never changes the fixed-k " "comparison."
        if diagnostic_k_max is not None
        else ""
    )
    identity = RUDOLFV2_MODEL_IDENTITIES.get(str(model))
    report_model = str(model) if identity is None else identity.published_name
    relationship = ""
    if identity is not None:
        if identity.parent_registry_id is None:
            relationship = " Family role: teacher."
        else:
            parent = RUDOLFV2_MODEL_IDENTITIES[identity.parent_registry_id].published_name
            relationship = f" Family role: distilled student of `{parent}`."
    lines = [
        "# Pooling sensitivity tracer",
        "",
        f"Benchmark: `{benchmark}`. Model: `{report_model}`.{relationship} "
        f"All paired comparisons use fixed k={canonical.fixed_k}.",
        "",
        "MaRI uses a separate automatically resolved tau for each representation at the "
        f"fixed comparison k. Biological k* is reported separately from the production "
        f"sparse sweep to k_max={production_k_max}." + diagnostic_sentence,
        "",
        "| measure | canonical | alternative | signed delta | absolute delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for measure in _COMPARISON_MEASURES:
        lines.append(
            f"| {measure} | {float(comparison[f'canonical_{measure}']):.9g} | "
            f"{float(comparison[f'alternative_{measure}']):.9g} | "
            f"{float(comparison[f'delta_{measure}']):+.9g} | "
            f"{float(comparison[f'abs_delta_{measure}']):.9g} |"
        )
    lines.extend(
        [
            "",
            "## Paired uncertainty",
            "",
            f"Alternative-minus-canonical headline CRoMa: {float(comparison['croma_delta_ci_point']):+.9g} "
            f"(two-sided 95% CI [{float(comparison['croma_delta_ci_lo']):+.9g}, "
            f"{float(comparison['croma_delta_ci_hi']):+.9g}]); supported: "
            f"{bool(comparison['croma_delta_supported'])}.",
            f"Median paired per-occurrence CRoMa delta (descriptive): {float(comparison['median_paired_occurrence_croma_delta']):+.9g}.",
            "",
            "## Conclusions",
            "",
            f"The headline CRoMa shift is {float(comparison['delta_croma']):+.9g}; "
            f"F(0) shifts {float(comparison['delta_croma_f0']):+.9g} and LTM10 shifts "
            f"{float(comparison['delta_croma_ltm10']):+.9g}. This single-model tracer "
            "establishes the sign and tail contract; cross-model order and family-level "
            "conclusions belong to the complete five-encoder study.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def render_study_bundle(
    *,
    benchmark: str,
    tileset: str,
    model: str,
    canonical: RepresentationEvaluation,
    alternative: RepresentationEvaluation,
    aligned_manifest: pd.DataFrame,
    provenance_inputs: dict,
    replay_commands: list[str],
    n_boot: int = 2000,
) -> dict[Path, bytes]:
    """Render every deterministic result file in memory before publication."""

    comparisons, rankings = build_comparison_frames(
        benchmark=benchmark,
        tileset=tileset,
        model=model,
        canonical=canonical,
        alternative=alternative,
    )
    occurrence_arrays = build_occurrence_arrays(
        canonical=canonical.croma_result,
        alternative=alternative.croma_result,
        aligned_manifest=aligned_manifest,
    )
    ci = paired_cluster_bootstrap_delta(
        occurrence_arrays["canonical_croma"],
        occurrence_arrays["alternative_croma"],
        occurrence_arrays["group_id"],
        subset_ids=occurrence_arrays["subset"],
        n_boot=int(n_boot),
        level=0.95,
        seed=0,
    )
    headline_delta = float(alternative.croma) - float(canonical.croma)
    if not np.isclose(ci.point, headline_delta, rtol=1e-12, atol=1e-12):
        raise RuntimeError(
            "bootstrap point does not match alternative-minus-canonical headline CRoMa"
        )
    comparisons["croma_delta_ci_point"] = float(ci.point)
    comparisons["croma_delta_ci_lo"] = float(ci.lo)
    comparisons["croma_delta_ci_hi"] = float(ci.hi)
    comparisons["croma_delta_supported"] = bool(ci.lo > 0.0 or ci.hi < 0.0)
    canonical_pooled = float(np.median(occurrence_arrays["canonical_croma"]))
    alternative_pooled = float(np.median(occurrence_arrays["alternative_croma"]))
    comparisons["canonical_pooled_occurrence_croma"] = canonical_pooled
    comparisons["alternative_pooled_occurrence_croma"] = alternative_pooled
    comparisons["delta_pooled_occurrence_croma"] = alternative_pooled - canonical_pooled
    comparisons["median_paired_occurrence_croma_delta"] = float(
        np.median(occurrence_arrays["alternative_croma"] - occurrence_arrays["canonical_croma"])
    )

    files: dict[Path, bytes] = {
        Path("results/comparisons.csv"): _csv_bytes(comparisons),
        Path("results/rankings.csv"): _csv_bytes(rankings),
        Path("per-occurrence")
        / benchmark
        / f"{model}.npz": deterministic_npz_bytes(occurrence_arrays),
        Path("report.md"): _render_report(
            benchmark=benchmark,
            model=model,
            canonical=canonical,
            alternative=alternative,
            comparison=comparisons.iloc[0],
        ),
    }
    benchmark_plan = PATHOROB_STUDY_BENCHMARKS.get(benchmark)
    production_k_max = benchmark_plan.biological_k_max if benchmark_plan is not None else 600
    diagnostic_k_max = benchmark_plan.diagnostic_k_max if benchmark_plan is not None else 300
    provenance = {
        "schema_version": 1,
        "study": "pooling-sensitivity",
        "croma_version": str(croma_version),
        "benchmark": str(benchmark),
        "tileset": str(tileset),
        "model": str(model),
        "representations": [canonical.representation, alternative.representation],
        "fixed_k": int(canonical.fixed_k),
        "biological_kstar": {
            "grid": "pathorob-sparse",
            "production_k_max": int(production_k_max),
            "diagnostic_k_max": (None if diagnostic_k_max is None else int(diagnostic_k_max)),
            "tie_break": "smallest-k",
        },
        "mari_tau": "per-representation-auto-at-fixed-k",
        "bootstrap": {
            "grouping": "shared-group_id",
            "level": 0.95,
            "method": "numpy-linear-percentile",
            "n_boot": int(n_boot),
            "seed": 0,
            "contrast": "alternative-minus-canonical-headline-croma",
        },
        "inputs": dict(provenance_inputs),
        "replay_commands": list(replay_commands),
        "output_artifacts": {
            path.as_posix(): {"sha256": _sha256_bytes(payload), "size": len(payload)}
            for path, payload in sorted(files.items(), key=lambda item: item[0].as_posix())
        },
    }
    identity = RUDOLFV2_MODEL_IDENTITIES.get(str(model))
    if identity is not None:
        provenance["model_identity"] = {
            "model_registry_id": str(model),
            "variant_role": identity.variant_role,
            "parent_registry_id": identity.parent_registry_id,
            "canonical_width": identity.canonical_width,
        }
    files[Path("run-provenance.json")] = (
        json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return files


def render_panel_bundle(
    *,
    runs: list[StudyRun],
    replay_commands: list[str],
    n_boot: int = 2000,
) -> dict[Path, bytes]:
    """Render a deterministic aggregate without depending on caller run order."""

    if not runs:
        raise ValueError("pooling-sensitivity panel requires at least one run")
    ordered = sorted(runs, key=lambda run: (run.benchmark, run.model))
    identities = [(run.benchmark, run.model) for run in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("pooling-sensitivity panel contains a duplicate benchmark/model run")

    comparison_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    per_occurrence: dict[Path, bytes] = {}
    run_provenance: list[dict] = []
    report_sections: list[str] = ["# Pooling sensitivity panel", ""]
    for run in ordered:
        bundle = render_study_bundle(
            benchmark=run.benchmark,
            tileset=run.tileset,
            model=run.model,
            canonical=run.canonical,
            alternative=run.alternative,
            aligned_manifest=run.aligned_manifest,
            provenance_inputs=run.provenance_inputs,
            replay_commands=replay_commands,
            n_boot=int(n_boot),
        )
        comparison_frames.append(pd.read_csv(io.BytesIO(bundle[Path("results/comparisons.csv")])))
        ranking_frames.append(pd.read_csv(io.BytesIO(bundle[Path("results/rankings.csv")])))
        occurrence_path = Path("per-occurrence") / run.benchmark / f"{run.model}.npz"
        per_occurrence[occurrence_path] = bundle[occurrence_path]
        per_run_provenance = json.loads(bundle[Path("run-provenance.json")])
        run_provenance.append(per_run_provenance)
        report = bundle[Path("report.md")].decode("utf-8").strip()
        report_model = run.model if run.identity is None else run.identity.published_name
        report_sections.extend([f"## {run.benchmark} — {report_model}", "", report, ""])

    files: dict[Path, bytes] = {
        Path("results/comparisons.csv"): _csv_bytes(
            pd.concat(comparison_frames, ignore_index=True)
        ),
        Path("results/rankings.csv"): _csv_bytes(pd.concat(ranking_frames, ignore_index=True)),
        Path("report.md"): ("\n".join(report_sections).rstrip() + "\n").encode("utf-8"),
        **per_occurrence,
    }
    provenance = {
        "schema_version": 2,
        "study": "pooling-sensitivity",
        "croma_version": str(croma_version),
        "bootstrap": {
            "grouping": "shared-group_id",
            "level": 0.95,
            "method": "numpy-linear-percentile",
            "n_boot": int(n_boot),
            "seed": 0,
            "contrast": "alternative-minus-canonical-headline-croma",
        },
        "runs": run_provenance,
        "replay_commands": list(replay_commands),
        "output_artifacts": {
            path.as_posix(): {"sha256": _sha256_bytes(payload), "size": len(payload)}
            for path, payload in sorted(files.items(), key=lambda item: item[0].as_posix())
        },
    }
    files[Path("run-provenance.json")] = (
        json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated pooling-sensitivity panel.")
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO / "output" / "embeddings",
        help="Read-only canonical embeddings root.",
    )
    parser.add_argument(
        "--canonical-access-root",
        type=Path,
        default=None,
        help="Optional exact local mirror used only for canonical reads.",
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=REPO / "output" / "studies" / "pooling-sensitivity",
        help="Only root this study may write.",
    )
    parser.add_argument(
        "--study-access-root",
        type=Path,
        default=None,
        help="Optional exact local mirror used only for study-artifact reads.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Legacy single-benchmark manifest override.",
    )
    parser.add_argument(
        "--eval-manifest-root",
        type=Path,
        default=REPO / "data" / "pathorob" / "manifests",
        help="Directory containing the four PathoROB RI-view manifests.",
    )
    parser.add_argument(
        "--eval-manifest-access-root",
        type=Path,
        default=None,
        help="Optional exact local mirror used only for evaluation-manifest reads.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Legacy Mascaret tracer batch override; panel defaults are model-specific.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=tuple(POOLING_STUDY_MODELS))
    parser.add_argument("--benchmarks", nargs="+", choices=tuple(PATHOROB_STUDY_BENCHMARKS))
    workflow = parser.add_mutually_exclusive_group()
    workflow.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract/validate the selected alternative inventory and stop.",
    )
    workflow.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Validate and evaluate existing alternatives without extraction.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Capture/verify the protected 5x4 canonical artifact baseline and stop.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Recompute in temporary/in-memory storage and compare without target writes.",
    )
    mode.add_argument(
        "--force",
        action="store_true",
        help="Replace only incompatible study-owned targets.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    capture_preservation_baseline(
        canonical_root=args.canonical_root,
        study_root=args.study_root,
        canonical_access_root=args.canonical_access_root,
        study_access_root=args.study_access_root,
        check=bool(args.check),
    )
    if args.baseline_only:
        return 0
    if args.eval_manifest is not None and args.models is None and args.benchmarks is None:
        if args.extract_only or args.evaluate_only:
            raise ValueError(
                "--extract-only/--evaluate-only cannot be combined with the legacy tracer CLI"
            )
        run_mascaret_camelyon(
            canonical_root=args.canonical_root,
            study_root=args.study_root,
            eval_manifest_path=args.eval_manifest,
            device_arg=str(args.device),
            batch_size=(
                WAIV_STUDY_MODELS[TRACER_MODEL].batch_size
                if args.batch_size is None
                else int(args.batch_size)
            ),
            num_workers=int(args.num_workers),
            check=bool(args.check),
            force=bool(args.force),
        )
        return 0
    models = tuple(args.models or POOLING_STUDY_MODELS)
    benchmarks = tuple(args.benchmarks or PATHOROB_STUDY_BENCHMARKS)
    if (
        args.canonical_access_root is not None
        or args.study_access_root is not None
        or args.eval_manifest_access_root is not None
    ) and not args.evaluate_only:
        raise ValueError("local access roots require --evaluate-only")
    if args.batch_size is not None and models != (TRACER_MODEL,):
        raise ValueError("--batch-size is only valid with --models Mascaret")
    if args.eval_manifest is not None and benchmarks != (TRACER_BENCHMARK,):
        raise ValueError("--eval-manifest is only valid with --benchmarks pathorob-camelyon")
    if args.evaluate_only:
        inventory = {
            (run.benchmark_name, run.model_id): (
                study_embedding_path(
                    study_root=args.study_root,
                    tileset=run.benchmark.tileset,
                    model=run.model_id,
                    representation=run.model_plan.alternative,
                ),
                "reused",
            )
            for run in build_study_plan(
                model_ids=models,
                model_plans=POOLING_STUDY_MODELS,
                benchmark_ids=benchmarks,
            )
        }
    else:
        inventory = extract_pooling_panel(
            canonical_root=args.canonical_root,
            study_root=args.study_root,
            device_arg=str(args.device),
            num_workers=int(args.num_workers),
            model_plans=POOLING_STUDY_MODELS,
            check=bool(args.check),
            force=bool(args.force),
            models=models,
            benchmarks=benchmarks,
        )
    build_study_inventory(
        canonical_root=args.canonical_root,
        study_root=args.study_root,
        canonical_access_root=args.canonical_access_root,
        study_access_root=args.study_access_root,
        model_ids=models,
        device_arg=str(args.device),
        model_plans=POOLING_STUDY_MODELS,
        benchmark_ids=benchmarks,
    )
    if args.extract_only:
        verify_preservation_baseline(
            canonical_root=args.canonical_root,
            study_root=args.study_root,
        )
        return 0
    runs = evaluate_pooling_panel_isolated(
        canonical_root=args.canonical_root,
        study_root=args.study_root,
        eval_manifest_root=args.eval_manifest_root,
        canonical_access_root=args.canonical_access_root,
        study_access_root=args.study_access_root,
        eval_manifest_access_root=args.eval_manifest_access_root,
        device_arg=str(args.device),
        inventory=inventory,
        model_plans=POOLING_STUDY_MODELS,
        eval_manifest_override=args.eval_manifest,
    )
    replay = shlex.join(
        [
            "python",
            "scripts/studies/pooling_sensitivity.py",
            "--canonical-root",
            str(Path(args.canonical_root).resolve()),
            "--study-root",
            str(Path(args.study_root).resolve()),
            "--eval-manifest-root",
            str(Path(args.eval_manifest_root).resolve()),
            "--device",
            str(args.device),
            "--num-workers",
            str(args.num_workers),
            "--models",
            *models,
            "--benchmarks",
            *benchmarks,
        ]
    )
    bundle = render_panel_bundle(
        runs=runs,
        replay_commands=[replay, replay + " --evaluate-only --check"],
    )
    publish_study_bundle(
        args.study_root,
        bundle,
        check=bool(args.check),
        force=bool(args.force),
    )
    verify_preservation_baseline(
        canonical_root=args.canonical_root,
        study_root=args.study_root,
        canonical_access_root=args.canonical_access_root,
        study_access_root=args.study_access_root,
    )
    return 0


def capture_preservation_baseline(
    *,
    canonical_root: Path,
    study_root: Path,
    canonical_access_root: Path | None = None,
    study_access_root: Path | None = None,
    check: bool = False,
) -> Path:
    """Record hashes and filesystem metadata for the exact protected 5x4 panel."""

    canonical_root = Path(canonical_root).resolve()
    study_root = Path(study_root).resolve()
    canonical_access_root = _resolved_access_root(
        canonical_root,
        canonical_access_root,
    )
    study_access_root = _resolved_access_root(study_root, study_access_root)
    artifacts: list[dict[str, str | int]] = []
    for tileset in PATHOROB_TILESETS:
        for model in STUDY_MODELS:
            matrix = canonical_access_root / tileset / f"{model}.npy"
            for kind, path in (
                ("matrix", matrix),
                ("sidecar", matrix.with_suffix(".npy.json")),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"protected canonical {kind} is missing: {path}")
                stat = path.stat()
                artifacts.append(
                    {
                        "kind": kind,
                        "relative_path": path.relative_to(canonical_access_root).as_posix(),
                        "sha256": _sha256_file(path),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )

    payload = (
        json.dumps(
            {"schema_version": 1, "artifacts": artifacts},
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    output_path = study_root / PRESERVATION_BASELINE_NAME
    access_output_path = study_access_root / PRESERVATION_BASELINE_NAME
    if access_output_path.exists():
        if access_output_path.read_bytes() != payload:
            raise RuntimeError(
                "preservation baseline already exists with different bytes: "
                f"{access_output_path}"
            )
        return output_path
    if check:
        raise RuntimeError(
            f"preservation baseline check failed because the target is missing: {output_path}"
        )
    _atomic_write(output_path, payload)
    return output_path


def verify_preservation_baseline(
    *,
    canonical_root: Path,
    study_root: Path,
    canonical_access_root: Path | None = None,
    study_access_root: Path | None = None,
) -> dict[str, int]:
    """Verify hashes, sizes, and mtimes against the recorded protected baseline."""

    logical_study_root = Path(study_root).resolve()
    baseline_path = (
        _resolved_access_root(logical_study_root, study_access_root) / PRESERVATION_BASELINE_NAME
    )
    if not baseline_path.is_file():
        raise FileNotFoundError(f"preservation baseline is missing: {baseline_path}")
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 40:
        raise RuntimeError("preservation baseline must contain exactly 40 artifacts")
    canonical_root = Path(canonical_root).resolve()
    canonical_access_root = _resolved_access_root(canonical_root, canonical_access_root)
    for expected in artifacts:
        path = canonical_access_root / str(expected["relative_path"])
        if not path.is_file():
            raise RuntimeError(f"protected canonical artifact disappeared: {path}")
        stat = path.stat()
        actual = {
            "sha256": _sha256_file(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        mismatched = [key for key, value in actual.items() if value != expected[key]]
        if mismatched:
            raise RuntimeError(
                f"protected canonical artifact changed ({', '.join(mismatched)}): {path}"
            )
    return {"artifacts": 40, "matrices": 20, "sidecars": 20}


def study_embedding_path(
    *, study_root: Path, tileset: str, model: str, representation: str
) -> Path:
    if representation == "canonical":
        raise ValueError("canonical embeddings remain read-only at their canonical source")
    return Path(study_root) / "embeddings" / str(tileset) / str(model) / f"{representation}.npy"


def _require_isolated_target(*, target: Path, study_root: Path, canonical_path: Path) -> None:
    root = Path(study_root).resolve()
    if target.is_symlink() or sidecar_path(target).is_symlink():
        raise RuntimeError("alternative artifact may not use symlinks")
    resolved_target = Path(target).resolve()
    if not resolved_target.is_relative_to(root):
        raise RuntimeError(f"alternative target escapes study root: {target}")
    if resolved_target == Path(canonical_path).resolve():
        raise RuntimeError("alternative target aliases the canonical embedding")
    if target.exists() and os.path.samefile(target, canonical_path):
        raise RuntimeError("alternative target hard-links the canonical embedding")
    canonical_sidecar = sidecar_path(canonical_path)
    target_sidecar = sidecar_path(target)
    if (
        target_sidecar.exists()
        and canonical_sidecar.exists()
        and os.path.samefile(target_sidecar, canonical_sidecar)
    ):
        raise RuntimeError("alternative sidecar hard-links the canonical sidecar")


def _validate_study_matrix(
    path: Path,
    expected: extraction.EmbeddingArtifactContract,
) -> None:
    """Require the complete numerical contract that sidecar matching cannot prove."""

    matrix = np.load(path, mmap_mode="r")
    if matrix.shape != expected.output_shape:
        raise ArtifactCompatibilityError(
            f"alternative matrix must have shape {expected.output_shape}; got {matrix.shape}"
        )
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ArtifactCompatibilityError("alternative matrix must be finite FP32")


_RUDOLFV2_CANONICAL_POOLING = {
    "method": "concatenate-cls-and-mean-patches",
    "register_tokens_excluded": 8,
    "patch_tokens": 784,
}


def _prepare_rudolf_cls_derivation(
    *,
    canonical_path: Path,
    manifest_path: Path,
    spec: ModelSpec,
    batch_size: int,
    device_arg: str,
    alternative_contract: extraction.EmbeddingArtifactContract,
) -> tuple[extraction.EmbeddingArtifactContract, np.ndarray]:
    """Validate and describe the lossless canonical-CLS prefix derivation."""

    canonical_contract = extraction.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=int(batch_size),
        device_arg=device_arg,
        pooling="canonical",
    )
    if not artifact_is_reusable(canonical_path, canonical_contract):
        raise FileNotFoundError(f"canonical embedding is missing: {canonical_path}")
    _validate_study_matrix(canonical_path, canonical_contract)
    canonical = np.load(canonical_path, mmap_mode="r")
    canonical_width = int(canonical_contract.output_shape[1])
    alternative_width = int(alternative_contract.output_shape[1])
    if canonical_contract.extraction_contract.get("pooling") != _RUDOLFV2_CANONICAL_POOLING:
        raise ArtifactCompatibilityError(
            "Rudolf canonical artifact must use the exact CLS+mean-patches layout"
        )
    if canonical_width != 2 * alternative_width:
        raise ArtifactCompatibilityError(
            "Rudolf CLS-only width must be exactly half the canonical CLS+mean width"
        )
    canonical_sidecar = sidecar_path(canonical_path)
    derivation = {
        "method": "validated-canonical-cls-prefix",
        "source_representation": "canonical-cls-plus-mean-patches",
        "source_matrix_sha256": _sha256_file(canonical_path),
        "source_sidecar_sha256": _sha256_file(canonical_sidecar),
        "source_width": canonical_width,
        "slice_start": 0,
        "slice_stop": alternative_width,
        "output_normalization": "none",
    }
    extraction_contract = dict(alternative_contract.extraction_contract)
    extraction_contract["materialization"] = derivation
    return (
        replace(
            alternative_contract,
            extraction_contract=extraction_contract,
        ),
        canonical,
    )


def _materialize_study_representation(
    *,
    output_path: Path,
    canonical: np.ndarray | None,
    expected: extraction.EmbeddingArtifactContract,
    manifest_path: Path,
    spec: ModelSpec,
    batch_size: int,
    num_workers: int,
    device_arg: str,
    representation: str,
    progress_enabled: bool,
) -> None:
    """Publish one alternative through its declared materialization path."""

    if canonical is not None:
        width = int(expected.output_shape[1])
        extraction.publish_embedding_artifact(
            output_path,
            np.asarray(canonical[:, :width]),
            expected,
        )
        return
    extraction.embed_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        spec=spec,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device_arg=device_arg,
        artifact_contract=expected,
        progress_enabled=progress_enabled,
        pooling=representation,
    )


def extract_study_representation(
    *,
    canonical_root: Path,
    study_root: Path,
    tileset: str,
    model: str,
    representation: str,
    batch_size: int,
    num_workers: int,
    device_arg: str,
    check: bool = False,
    force: bool = False,
) -> tuple[Path, str]:
    """Extract or verify one alternative without exposing it to canonical discovery."""

    canonical_root = Path(canonical_root).resolve()
    study_root = Path(study_root).resolve()
    manifest_path = canonical_root / tileset / "manifest.csv"
    canonical_path = canonical_root / tileset / f"{model}.npy"
    if not canonical_path.is_file():
        raise FileNotFoundError(f"canonical embedding is missing: {canonical_path}")
    target = study_embedding_path(
        study_root=study_root,
        tileset=tileset,
        model=model,
        representation=representation,
    )
    _require_isolated_target(
        target=target,
        study_root=study_root,
        canonical_path=canonical_path,
    )
    try:
        spec = _build_model_registry()[model]
    except KeyError:
        raise ValueError(f"unknown canonical model: {model!r}") from None
    expected_representation = STUDY_ALTERNATIVE_POOLING.get(model)
    if representation != expected_representation:
        raise ValueError(
            f"representation {representation!r} is not mapped for {model!r}; "
            f"expected {expected_representation!r}"
        )
    expected = extraction.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=int(batch_size),
        device_arg=device_arg,
        pooling=representation,
    )
    canonical: np.ndarray | None = None
    if spec.backend == "rudolfv2" and representation == "cls-only":
        expected, canonical = _prepare_rudolf_cls_derivation(
            canonical_path=canonical_path,
            manifest_path=manifest_path,
            spec=spec,
            batch_size=int(batch_size),
            device_arg=device_arg,
            alternative_contract=expected,
        )
    if not check:
        try:
            if artifact_is_reusable(target, expected):
                _validate_study_matrix(target, expected)
                return target, "reused"
        except ArtifactCompatibilityError:
            if not force:
                raise

    if check:
        with tempfile.TemporaryDirectory(prefix="croma-pooling-check-") as directory:
            temporary = Path(directory) / target.name
            _materialize_study_representation(
                output_path=temporary,
                canonical=canonical,
                expected=expected,
                manifest_path=manifest_path,
                spec=spec,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                device_arg=device_arg,
                progress_enabled=False,
                representation=representation,
            )
            _validate_study_matrix(temporary, expected)
            if not target.is_file() or not sidecar_path(target).is_file():
                raise RuntimeError("study extraction check failed: target artifact is missing")
            if (
                target.read_bytes() != temporary.read_bytes()
                or sidecar_path(target).read_bytes() != sidecar_path(temporary).read_bytes()
            ):
                raise RuntimeError("study extraction check failed: target bytes differ")
        return target, "checked"

    _materialize_study_representation(
        output_path=target,
        canonical=canonical,
        expected=expected,
        manifest_path=manifest_path,
        spec=spec,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device_arg=device_arg,
        progress_enabled=True,
        representation=representation,
    )
    _validate_study_matrix(target, expected)
    return target, "forced" if force else "written"


def extract_pooling_panel(
    *,
    canonical_root: Path,
    study_root: Path,
    device_arg: str,
    num_workers: int,
    model_plans: dict[str, StudyModelPlan],
    check: bool = False,
    force: bool = False,
    models: tuple[str, ...] | None = None,
    benchmarks: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], tuple[Path, str]]:
    """Extract or validate one declared model/benchmark cross-product."""

    selected_models = tuple(model_plans) if models is None else tuple(models)
    selected_benchmarks = (
        tuple(PATHOROB_STUDY_BENCHMARKS) if benchmarks is None else tuple(benchmarks)
    )
    unknown_models = [name for name in selected_models if name not in model_plans]
    unknown_benchmarks = [
        name for name in selected_benchmarks if name not in PATHOROB_STUDY_BENCHMARKS
    ]
    if unknown_models or unknown_benchmarks:
        raise ValueError(
            f"unknown pooling panel selection: models={unknown_models}, "
            f"benchmarks={unknown_benchmarks}"
        )

    inventory: dict[tuple[str, str], tuple[Path, str]] = {}
    for benchmark in selected_benchmarks:
        benchmark_plan = PATHOROB_STUDY_BENCHMARKS[benchmark]
        for model in selected_models:
            model_plan = model_plans[model]
            mode = "check" if check else "extract"
            print(f"[study] {mode} {benchmark} / {model}", flush=True)
            artifact = extract_study_representation(
                canonical_root=canonical_root,
                study_root=study_root,
                tileset=benchmark_plan.tileset,
                model=model,
                representation=model_plan.alternative,
                batch_size=model_plan.batch_size,
                num_workers=int(num_workers),
                device_arg=device_arg,
                check=bool(check),
                force=bool(force),
            )
            inventory[(benchmark, model)] = artifact
            print(
                f"[study] completed {mode} {benchmark} / {model}: {artifact[1]}",
                flush=True,
            )
    return inventory


def extract_waiv_panel(
    *,
    canonical_root: Path,
    study_root: Path,
    device_arg: str,
    num_workers: int,
    check: bool = False,
    force: bool = False,
    models: tuple[str, ...] | None = None,
    benchmarks: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], tuple[Path, str]]:
    """Compatibility wrapper for the issue-151 Waiv panel."""

    return extract_pooling_panel(
        canonical_root=canonical_root,
        study_root=study_root,
        device_arg=device_arg,
        num_workers=num_workers,
        model_plans=WAIV_STUDY_MODELS,
        check=check,
        force=force,
        models=models,
        benchmarks=benchmarks,
    )


def build_study_inventory(
    *,
    canonical_root: Path,
    study_root: Path,
    model_ids: tuple[str, ...],
    device_arg: str,
    canonical_access_root: Path | None = None,
    study_access_root: Path | None = None,
    model_plans: dict[str, StudyModelPlan] | None = None,
    benchmark_ids: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Validate every declared alternative and return its deterministic inventory."""

    canonical_root = Path(canonical_root).resolve()
    study_root = Path(study_root).resolve()
    canonical_access_root = _resolved_access_root(
        canonical_root,
        canonical_access_root,
    )
    study_access_root = _resolved_access_root(study_root, study_access_root)
    plans = POOLING_STUDY_MODELS if model_plans is None else model_plans
    registry = _build_model_registry()
    rows: list[dict[str, str | int]] = []
    for run in build_study_plan(
        model_ids=model_ids,
        model_plans=plans,
        benchmark_ids=benchmark_ids,
    ):
        manifest_path = canonical_access_root / run.benchmark.tileset / "manifest.csv"
        logical_canonical_path = canonical_root / run.benchmark.tileset / f"{run.model_id}.npy"
        canonical_path = canonical_access_root / run.benchmark.tileset / f"{run.model_id}.npy"
        if not canonical_path.is_file():
            raise FileNotFoundError(f"canonical embedding is missing: {canonical_path}")
        target = study_embedding_path(
            study_root=study_root,
            tileset=run.benchmark.tileset,
            model=run.model_id,
            representation=run.model_plan.alternative,
        )
        _require_isolated_target(
            target=target,
            study_root=study_root,
            canonical_path=logical_canonical_path,
        )
        access_target = study_embedding_path(
            study_root=study_access_root,
            tileset=run.benchmark.tileset,
            model=run.model_id,
            representation=run.model_plan.alternative,
        )
        expected = extraction.build_embedding_artifact_contract(
            manifest_path=manifest_path,
            spec=registry[run.model_id],
            batch_size=run.model_plan.batch_size,
            device_arg=device_arg,
            pooling=run.model_plan.alternative,
        )
        if registry[run.model_id].backend == "rudolfv2":
            expected, _canonical = _prepare_rudolf_cls_derivation(
                canonical_path=canonical_path,
                manifest_path=manifest_path,
                spec=registry[run.model_id],
                batch_size=run.model_plan.batch_size,
                device_arg=device_arg,
                alternative_contract=expected,
            )
        if not artifact_is_reusable(access_target, expected):
            raise FileNotFoundError(f"alternative embedding is missing: {access_target}")
        _validate_study_matrix(access_target, expected)
        identity = run.identity
        rows.append(
            {
                "benchmark": run.benchmark_name,
                "tileset": run.benchmark.tileset,
                "model_registry_id": run.model_id,
                "variant_role": "" if identity is None else identity.variant_role,
                "parent_registry_id": (
                    "" if identity is None else (identity.parent_registry_id or "")
                ),
                "representation": run.model_plan.alternative,
                "rows": int(expected.output_shape[0]),
                "width": int(expected.output_shape[1]),
                "dtype": "float32",
                "manifest_fingerprint": expected.manifest_fingerprint,
                "matrix_sha256": _sha256_file(access_target),
                "sidecar_sha256": _sha256_file(sidecar_path(access_target)),
            }
        )
    return pd.DataFrame(rows)


def _file_provenance(
    path: Path,
    *,
    access_path: Path | None = None,
) -> dict[str, str | int]:
    source = Path(path if access_path is None else access_path)
    stat = source.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_validated_canonical_matrix(
    *,
    canonical_path: Path,
    manifest_path: Path,
    batch_size: int,
    device_arg: str,
    model: str = TRACER_MODEL,
) -> np.ndarray:
    """Open a canonical matrix only when its complete contract matches."""

    spec = _build_model_registry()[model]
    expected = extraction.build_embedding_artifact_contract(
        manifest_path=manifest_path,
        spec=spec,
        batch_size=int(batch_size),
        device_arg=device_arg,
        pooling="canonical",
    )
    if not artifact_is_reusable(canonical_path, expected):
        raise FileNotFoundError(f"canonical embedding is missing: {canonical_path}")
    matrix = np.load(canonical_path, mmap_mode="r")
    if matrix.shape != expected.output_shape:
        raise RuntimeError(
            f"canonical {model} must have shape {expected.output_shape}; got {matrix.shape}"
        )
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise RuntimeError(f"canonical {model} must be finite FP32")
    return matrix


def _evaluation_manifest_path(root: Path, benchmark: str) -> Path:
    relative = Path(benchmark_registry.get(benchmark).manifest)
    return Path(root).resolve() / relative.name


def evaluate_pooling_panel(
    *,
    canonical_root: Path,
    study_root: Path,
    eval_manifest_root: Path,
    device_arg: str,
    inventory: dict[tuple[str, str], tuple[Path, str]],
    model_plans: dict[str, StudyModelPlan],
    canonical_access_root: Path | None = None,
    study_access_root: Path | None = None,
    eval_manifest_access_root: Path | None = None,
    eval_manifest_override: Path | None = None,
) -> list[StudyRun]:
    """Evaluate an extracted inventory at frozen study operating points."""

    canonical_root = Path(canonical_root).resolve()
    study_root = Path(study_root).resolve()
    canonical_access_root = _resolved_access_root(
        canonical_root,
        canonical_access_root,
    )
    study_access_root = _resolved_access_root(study_root, study_access_root)
    eval_manifest_root = Path(eval_manifest_root).resolve()
    eval_manifest_access_root = _resolved_access_root(
        eval_manifest_root,
        eval_manifest_access_root,
    )
    runs: list[StudyRun] = []
    for benchmark, model in sorted(inventory):
        try:
            benchmark_plan = PATHOROB_STUDY_BENCHMARKS[benchmark]
            model_plan = model_plans[model]
        except KeyError:
            raise ValueError(
                f"inventory contains an unknown study run: {(benchmark, model)}"
            ) from None
        logical_eval_manifest_path = (
            Path(eval_manifest_override).resolve()
            if eval_manifest_override is not None
            else _evaluation_manifest_path(eval_manifest_root, benchmark)
        )
        eval_manifest_path = (
            logical_eval_manifest_path
            if eval_manifest_override is not None
            else _evaluation_manifest_path(eval_manifest_access_root, benchmark)
        )
        view = benchmark_views.load_view(
            benchmark,
            embeddings_root=canonical_access_root,
            eval_manifest_path=eval_manifest_path,
        )
        if view.spec.design != benchmark_plan.evaluation_design:
            raise RuntimeError(
                f"benchmark design drift for {benchmark}: "
                f"expected {benchmark_plan.evaluation_design}, got {view.spec.design}"
            )
        logical_tileset_manifest_path = canonical_root / benchmark_plan.tileset / "manifest.csv"
        tileset_manifest_path = canonical_access_root / benchmark_plan.tileset / "manifest.csv"
        logical_canonical_path = canonical_root / benchmark_plan.tileset / f"{model}.npy"
        canonical_path = canonical_access_root / benchmark_plan.tileset / f"{model}.npy"
        canonical_full = _load_validated_canonical_matrix(
            canonical_path=canonical_path,
            manifest_path=tileset_manifest_path,
            batch_size=model_plan.batch_size,
            device_arg=device_arg,
            model=model,
        )
        logical_alternative_path = inventory[(benchmark, model)][0]
        alternative_path = study_embedding_path(
            study_root=study_access_root,
            tileset=benchmark_plan.tileset,
            model=model,
            representation=model_plan.alternative,
        )
        model_spec = _build_model_registry()[model]
        expected_alternative = extraction.build_embedding_artifact_contract(
            manifest_path=tileset_manifest_path,
            spec=model_spec,
            batch_size=model_plan.batch_size,
            device_arg=device_arg,
            pooling=model_plan.alternative,
        )
        if model_spec.backend == "rudolfv2":
            expected_alternative, _canonical = _prepare_rudolf_cls_derivation(
                canonical_path=canonical_path,
                manifest_path=tileset_manifest_path,
                spec=model_spec,
                batch_size=model_plan.batch_size,
                device_arg=device_arg,
                alternative_contract=expected_alternative,
            )
        if not artifact_is_reusable(alternative_path, expected_alternative):
            raise FileNotFoundError(f"alternative embedding is missing: {alternative_path}")
        alternative_full = np.load(alternative_path, mmap_mode="r")
        expected_shape = (expected_alternative.output_shape[0], model_plan.alternative_width)
        if expected_alternative.output_shape != expected_shape:
            raise RuntimeError(
                f"{model} extraction width drifted from the study plan: "
                f"{expected_alternative.output_shape[1]} != {model_plan.alternative_width}"
            )
        if alternative_full.shape != expected_shape:
            raise RuntimeError(
                f"{model} {model_plan.alternative} must have shape {expected_shape}; "
                f"got {alternative_full.shape}"
            )
        if alternative_full.dtype != np.float32 or not np.isfinite(alternative_full).all():
            raise RuntimeError(f"{model} {model_plan.alternative} must be finite FP32")

        evaluation_kwargs = {
            "manifest": view.eval_manifest,
            "confounder_column": "confounder",
            "evaluation_design": benchmark_plan.evaluation_design,
            "fixed_k": benchmark_plan.fixed_k,
            "production_k_max": benchmark_plan.biological_k_max,
            "diagnostic_k_max": benchmark_plan.diagnostic_k_max,
        }
        print(f"[study] evaluate {benchmark} / {model} / canonical", flush=True)
        canonical_eval = evaluate_representation(
            representation="canonical",
            features=np.asarray(canonical_full[view.rows]),
            **evaluation_kwargs,
        )
        print(f"[study] evaluate {benchmark} / {model} / alternative", flush=True)
        alternative_eval = evaluate_representation(
            representation=model_plan.alternative,
            features=np.asarray(alternative_full[view.rows]),
            **evaluation_kwargs,
        )
        print(f"[study] completed {benchmark} / {model}", flush=True)
        aligned_manifest = align_paired_occurrence_manifest(
            canonical=canonical_eval.croma_result,
            alternative=alternative_eval.croma_result,
            manifest=view.eval_manifest,
        )
        runs.append(
            StudyRun(
                benchmark=benchmark,
                tileset=benchmark_plan.tileset,
                model=model,
                canonical=canonical_eval,
                alternative=alternative_eval,
                aligned_manifest=aligned_manifest,
                provenance_inputs={
                    "canonical_matrix": _file_provenance(
                        logical_canonical_path,
                        access_path=canonical_path,
                    ),
                    "canonical_sidecar": _file_provenance(
                        sidecar_path(logical_canonical_path),
                        access_path=sidecar_path(canonical_path),
                    ),
                    "alternative_matrix": _file_provenance(
                        logical_alternative_path,
                        access_path=alternative_path,
                    ),
                    "alternative_sidecar": _file_provenance(
                        sidecar_path(logical_alternative_path),
                        access_path=sidecar_path(alternative_path),
                    ),
                    "tileset_manifest": _file_provenance(
                        logical_tileset_manifest_path,
                        access_path=tileset_manifest_path,
                    ),
                    "evaluation_manifest": _file_provenance(
                        logical_eval_manifest_path,
                        access_path=eval_manifest_path,
                    ),
                    "preservation_baseline": _file_provenance(
                        study_root / PRESERVATION_BASELINE_NAME,
                        access_path=study_access_root / PRESERVATION_BASELINE_NAME,
                    ),
                },
                identity=RUDOLFV2_MODEL_IDENTITIES.get(model),
            )
        )
    return runs


def _evaluate_pooling_cell(request: StudyCellRequest) -> StudyRun:
    """Evaluate exactly one cell inside a worker process."""

    with config_context(working_memory=NEIGHBOR_WORKING_MEMORY_MIB):
        runs = evaluate_pooling_panel(
            canonical_root=request.canonical_root,
            study_root=request.study_root,
            eval_manifest_root=request.eval_manifest_root,
            canonical_access_root=request.canonical_access_root,
            study_access_root=request.study_access_root,
            eval_manifest_access_root=request.eval_manifest_access_root,
            device_arg=request.device_arg,
            inventory={(request.benchmark, request.model): request.inventory_entry},
            model_plans=request.model_plans,
            eval_manifest_override=request.eval_manifest_override,
        )
    if len(runs) != 1:
        raise RuntimeError(
            f"isolated evaluation returned {len(runs)} cells for "
            f"{(request.benchmark, request.model)}"
        )
    return runs[0]


def _evaluate_pooling_cell_in_fresh_process(request: StudyCellRequest) -> StudyRun:
    """Evaluate one cell and release all worker allocations before returning."""

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        return executor.submit(_evaluate_pooling_cell, request).result()


def evaluate_pooling_panel_isolated(
    *,
    canonical_root: Path,
    study_root: Path,
    eval_manifest_root: Path,
    device_arg: str,
    inventory: dict[tuple[str, str], tuple[Path, str]],
    model_plans: dict[str, StudyModelPlan],
    canonical_access_root: Path | None = None,
    study_access_root: Path | None = None,
    eval_manifest_access_root: Path | None = None,
    eval_manifest_override: Path | None = None,
    cell_runner: Callable[[StudyCellRequest], StudyRun] | None = None,
) -> list[StudyRun]:
    """Evaluate cells sequentially in fresh processes to bound allocator high-water state."""

    run_cell = _evaluate_pooling_cell_in_fresh_process if cell_runner is None else cell_runner
    runs: list[StudyRun] = []
    for benchmark, model in sorted(inventory):
        request = StudyCellRequest(
            benchmark=benchmark,
            model=model,
            inventory_entry=inventory[(benchmark, model)],
            canonical_root=Path(canonical_root),
            study_root=Path(study_root),
            eval_manifest_root=Path(eval_manifest_root),
            canonical_access_root=(
                None if canonical_access_root is None else Path(canonical_access_root)
            ),
            study_access_root=None if study_access_root is None else Path(study_access_root),
            eval_manifest_access_root=(
                None if eval_manifest_access_root is None else Path(eval_manifest_access_root)
            ),
            device_arg=str(device_arg),
            model_plans=model_plans,
            eval_manifest_override=(
                None if eval_manifest_override is None else Path(eval_manifest_override)
            ),
        )
        runs.append(run_cell(request))
    return runs


def evaluate_waiv_panel(
    *,
    canonical_root: Path,
    study_root: Path,
    eval_manifest_root: Path,
    device_arg: str,
    inventory: dict[tuple[str, str], tuple[Path, str]],
    eval_manifest_override: Path | None = None,
) -> list[StudyRun]:
    """Compatibility wrapper for the issue-151 Waiv panel."""

    return evaluate_pooling_panel(
        canonical_root=canonical_root,
        study_root=study_root,
        eval_manifest_root=eval_manifest_root,
        device_arg=device_arg,
        inventory=inventory,
        model_plans=WAIV_STUDY_MODELS,
        eval_manifest_override=eval_manifest_override,
    )


def run_mascaret_camelyon(
    *,
    canonical_root: Path,
    study_root: Path,
    eval_manifest_path: Path,
    device_arg: str,
    batch_size: int,
    num_workers: int,
    check: bool = False,
    force: bool = False,
) -> dict[str, str]:
    """Run the issue-150 end-to-end Mascaret/Camelyon tracer."""

    canonical_root = Path(canonical_root).resolve()
    study_root = Path(study_root).resolve()
    view = benchmark_views.load_view(
        TRACER_BENCHMARK,
        embeddings_root=canonical_root,
        eval_manifest_path=eval_manifest_path,
    )
    tileset_manifest_path = canonical_root / TRACER_TILESET / "manifest.csv"
    canonical_path = canonical_root / TRACER_TILESET / f"{TRACER_MODEL}.npy"
    canonical_full = _load_validated_canonical_matrix(
        canonical_path=canonical_path,
        manifest_path=tileset_manifest_path,
        batch_size=batch_size,
        device_arg=device_arg,
    )
    alternative_path, extraction_status = extract_study_representation(
        canonical_root=canonical_root,
        study_root=study_root,
        tileset=TRACER_TILESET,
        model=TRACER_MODEL,
        representation=TRACER_ALTERNATIVE,
        batch_size=batch_size,
        num_workers=num_workers,
        device_arg=device_arg,
        check=check,
        force=force,
    )
    tileset_manifest = pd.read_csv(tileset_manifest_path)
    eval_manifest = view.eval_manifest
    alternative_full = np.load(alternative_path, mmap_mode="r")
    expected_rows = len(tileset_manifest)
    if canonical_full.shape[0] != expected_rows:
        raise RuntimeError("canonical matrix row count does not match tileset manifest")
    if alternative_full.shape != (expected_rows, 3072):
        raise RuntimeError(
            f"Mascaret cls-mean-patch must have shape ({expected_rows}, 3072); "
            f"got {alternative_full.shape}"
        )
    if alternative_full.dtype != np.float32 or not np.isfinite(alternative_full).all():
        raise RuntimeError("Mascaret cls-mean-patch must be finite FP32")
    canonical_features = np.asarray(canonical_full[view.rows])
    alternative_features = np.asarray(alternative_full[view.rows])

    canonical_eval = evaluate_representation(
        representation="canonical",
        features=canonical_features,
        manifest=eval_manifest,
        confounder_column="confounder",
    )
    alternative_eval = evaluate_representation(
        representation=TRACER_ALTERNATIVE,
        features=alternative_features,
        manifest=eval_manifest,
        confounder_column="confounder",
    )
    aligned_manifest = align_paired_occurrence_manifest(
        canonical=canonical_eval.croma_result,
        alternative=alternative_eval.croma_result,
        manifest=eval_manifest,
    )
    provenance_inputs = {
        "canonical_matrix": _file_provenance(canonical_path),
        "canonical_sidecar": _file_provenance(sidecar_path(canonical_path)),
        "alternative_matrix": _file_provenance(alternative_path),
        "alternative_sidecar": _file_provenance(sidecar_path(alternative_path)),
        "tileset_manifest": _file_provenance(tileset_manifest_path),
        "evaluation_manifest": _file_provenance(Path(eval_manifest_path)),
        "preservation_baseline": _file_provenance(study_root / PRESERVATION_BASELINE_NAME),
    }
    replay = (
        "python scripts/studies/pooling_sensitivity.py "
        f"--canonical-root {canonical_root} --study-root {study_root} "
        f"--eval-manifest {Path(eval_manifest_path).resolve()} "
        f"--device {device_arg} --batch-size {batch_size} --num-workers {num_workers}"
    )
    bundle = render_study_bundle(
        benchmark=TRACER_BENCHMARK,
        tileset=TRACER_TILESET,
        model=TRACER_MODEL,
        canonical=canonical_eval,
        alternative=alternative_eval,
        aligned_manifest=aligned_manifest,
        provenance_inputs=provenance_inputs,
        replay_commands=[replay, replay + " --check"],
    )
    result_status = publish_study_bundle(
        study_root,
        bundle,
        check=check,
        force=force,
    )
    preservation = verify_preservation_baseline(
        canonical_root=canonical_root,
        study_root=study_root,
    )
    return {
        "extraction": extraction_status,
        "results": result_status,
        "preservation": f"verified-{preservation['artifacts']}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
