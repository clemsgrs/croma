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
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from croma.metrics.neighbors import _select_k_from_balanced_accuracy  # noqa: E402
from croma import CRoMa, MaRI, RI, __version__ as croma_version  # noqa: E402
from croma.metrics.bootstrap import paired_cluster_bootstrap_delta  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
from croma.metrics.mari import TAU_FALLBACK  # noqa: E402
from croma.metrics.neighbors import (  # noqa: E402
    _balanced_accuracy_by_k_from_prepared_neighbors,
)
from croma.metrics.pairs import normalize_manifest  # noqa: E402
from embedding_artifacts import (  # noqa: E402
    ArtifactCompatibilityError,
    artifact_is_reusable,
    sidecar_path,
)
import extract_embeddings as extraction  # noqa: E402
from model_registry import _build_model_registry  # noqa: E402
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


@dataclass(frozen=True)
class RepresentationEvaluation:
    """One representation evaluated under the pooling-sensitivity protocol."""

    representation: str
    fixed_k: int
    biological_knn_bacc: float
    confounder_knn_bacc: float
    biological_kstar: int
    biological_kstar_bacc: float
    diagnostic_kstar_300: int
    diagnostic_kstar_300_bacc: float
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
    diagnostic_k_max: int = 300,
) -> tuple[tuple[int, float], tuple[int, float]]:
    """Select production-600 and diagnostic-300 k* with production ties."""

    production_grid = resolve_sweep_k_values(production_k_max, "sparse")
    diagnostic_grid = resolve_sweep_k_values(diagnostic_k_max, "sparse")
    missing = [k for k in production_grid if k not in scores]
    if missing:
        raise ValueError(f"biological k* scores are missing sparse k values: {missing}")
    production_k = _select_k_from_balanced_accuracy(k_values=production_grid, scores=scores)
    diagnostic_k = _select_k_from_balanced_accuracy(k_values=diagnostic_grid, scores=scores)
    return (
        (int(production_k), float(scores[production_k])),
        (int(diagnostic_k), float(scores[diagnostic_k])),
    )


def evaluate_representation(
    *,
    representation: str,
    features: np.ndarray,
    manifest: pd.DataFrame,
    confounder_column: str,
    fixed_k: int = 11,
    production_k_max: int = 600,
    diagnostic_k_max: int = 300,
    headline_m: int = CROMA_HEADLINE_M,
    croma_start_k: int = 200,
) -> RepresentationEvaluation:
    """Evaluate one all-rows representation with one shared production neighbour cache."""

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
    diagnostic_grid = resolve_sweep_k_values(diagnostic_k_max, "sparse")
    k_values = sorted({int(fixed_k), *production_grid})
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
    (production_k, production_bacc), (diagnostic_k, diagnostic_bacc) = select_biological_kstars(
        biological_scores,
        production_k_max=production_k_max,
        diagnostic_k_max=diagnostic_k_max,
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

    dataset_name = RI._infer_dataset_name(normalized_manifest)
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

    croma_result = CRoMa.compute(
        features=features,
        manifest=normalized_manifest,
        confounder_column="confounder",
        evaluation_design="all",
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
        diagnostic_kstar_300=int(diagnostic_k),
        diagnostic_kstar_300_bacc=float(diagnostic_bacc),
        tau=float(tau),
        ri=float(ri_result.value),
        mari=float(mari_result.value),
        support=float(ri_result.support),
        ss_dominated_undefined_frac=float(ri_result.ss_dominated_undefined_frac),
        oo_dominated_undefined_frac=float(ri_result.oo_dominated_undefined_frac),
        mixed_undefined_frac=float(ri_result.mixed_undefined_frac),
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

    for target, payload in targets:
        if target.is_symlink():
            raise RuntimeError(f"refusing to replace a study symlink: {target}")
        _atomic_write(target, payload)
    return "forced" if force else "written"


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
    expected_subsets = (
        aligned_manifest["subset"].astype(str).to_numpy()
        if "subset" in aligned_manifest
        else np.full(len(aligned_manifest), "dataset", dtype="<U7")
    )
    identity_matches = (
        np.array_equal(canonical_sources, alternative_sources)
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
        "occurrence_index": np.arange(expected_length, dtype=np.int64),
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

    comparison: dict[str, str | int | float] = {
        "benchmark": str(benchmark),
        "tileset": str(tileset),
        "model": str(model),
        "canonical_representation": canonical.representation,
        "alternative_representation": alternative.representation,
        "fixed_k": int(canonical.fixed_k),
    }
    for measure in _COMPARISON_MEASURES:
        canonical_value = getattr(canonical, measure)
        alternative_value = getattr(alternative, measure)
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
        for measure in _COMPARISON_MEASURES:
            row[measure] = getattr(evaluation, measure)
        ranking_rows.append(row)
    rankings = (
        pd.DataFrame(ranking_rows)
        .sort_values(["croma", "representation"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    rankings.insert(4, "croma_rank", np.arange(1, len(rankings) + 1, dtype=int))
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
    lines = [
        "# Pooling sensitivity tracer",
        "",
        f"Benchmark: `{benchmark}`. Model: `{model}`. All paired comparisons use fixed k={canonical.fixed_k}.",
        "",
        "MaRI uses a separate automatically resolved tau for each representation at the fixed comparison k. Biological k* is reported separately from the production sparse sweep to k_max=600; the truncated-at-300 diagnostic never changes the fixed-k comparison.",
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
            "production_k_max": 600,
            "diagnostic_k_max": 300,
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
    files[Path("run-provenance.json")] = (
        json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated Mascaret/Camelyon pooling-sensitivity tracer."
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=REPO / "output" / "embeddings",
        help="Read-only canonical embeddings root.",
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=REPO / "output" / "studies" / "pooling-sensitivity",
        help="Only root this study may write.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=REPO / "data" / "pathorob" / "manifests" / "pathorob-camelyon-ri.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
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
        check=bool(args.check),
    )
    if args.baseline_only:
        return 0
    run_mascaret_camelyon(
        canonical_root=args.canonical_root,
        study_root=args.study_root,
        eval_manifest_path=args.eval_manifest,
        device_arg=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        check=bool(args.check),
        force=bool(args.force),
    )
    return 0


def capture_preservation_baseline(
    *, canonical_root: Path, study_root: Path, check: bool = False
) -> Path:
    """Record hashes and filesystem metadata for the exact protected 5x4 panel."""

    canonical_root = Path(canonical_root).resolve()
    study_root = Path(study_root).resolve()
    artifacts: list[dict[str, str | int]] = []
    for tileset in PATHOROB_TILESETS:
        for model in STUDY_MODELS:
            matrix = canonical_root / tileset / f"{model}.npy"
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
                        "relative_path": path.relative_to(canonical_root).as_posix(),
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
    if output_path.exists():
        if output_path.read_bytes() != payload:
            raise RuntimeError(
                f"preservation baseline already exists with different bytes: {output_path}"
            )
        return output_path
    if check:
        raise RuntimeError(
            f"preservation baseline check failed because the target is missing: {output_path}"
        )
    _atomic_write(output_path, payload)
    return output_path


def verify_preservation_baseline(*, canonical_root: Path, study_root: Path) -> dict[str, int]:
    """Verify hashes, sizes, and mtimes against the recorded protected baseline."""

    baseline_path = Path(study_root).resolve() / PRESERVATION_BASELINE_NAME
    if not baseline_path.is_file():
        raise FileNotFoundError(f"preservation baseline is missing: {baseline_path}")
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 40:
        raise RuntimeError("preservation baseline must contain exactly 40 artifacts")
    canonical_root = Path(canonical_root).resolve()
    for expected in artifacts:
        path = canonical_root / str(expected["relative_path"])
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
    if not check and not force:
        try:
            if artifact_is_reusable(target, expected):
                return target, "reused"
        except ArtifactCompatibilityError:
            raise

    if check:
        with tempfile.TemporaryDirectory(prefix="croma-pooling-check-") as directory:
            temporary = Path(directory) / target.name
            extraction.embed_manifest(
                manifest_path=manifest_path,
                output_path=temporary,
                spec=spec,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                device_arg=device_arg,
                artifact_contract=expected,
                progress_enabled=False,
                pooling=representation,
            )
            if not target.is_file() or not sidecar_path(target).is_file():
                raise RuntimeError("study extraction check failed: target artifact is missing")
            if (
                target.read_bytes() != temporary.read_bytes()
                or sidecar_path(target).read_bytes() != sidecar_path(temporary).read_bytes()
            ):
                raise RuntimeError("study extraction check failed: target bytes differ")
        return target, "checked"

    extraction.embed_manifest(
        manifest_path=manifest_path,
        output_path=target,
        spec=spec,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        device_arg=device_arg,
        artifact_contract=expected,
        progress_enabled=True,
        pooling=representation,
    )
    return target, "forced" if force else "written"


def _file_provenance(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_validated_canonical_matrix(
    *,
    canonical_path: Path,
    manifest_path: Path,
    batch_size: int,
    device_arg: str,
) -> np.ndarray:
    """Open a canonical Mascaret matrix only when its full contract matches."""

    spec = _build_model_registry()[TRACER_MODEL]
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
            f"canonical Mascaret must have shape {expected.output_shape}; got {matrix.shape}"
        )
    if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise RuntimeError("canonical Mascaret must be finite FP32")
    return matrix


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
    aligned_manifest = eval_manifest.copy().reset_index(drop=True)
    aligned_manifest["source_sample_index"] = np.arange(len(aligned_manifest), dtype=int)
    aligned_manifest["subset"] = "dataset"
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
