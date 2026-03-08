import argparse
import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extract_embeddings as ee
from model_registry import ModelSpec, _build_model_registry, _parse_models
from common import parse_k_candidates
from input_fingerprint import embedding_fingerprint, manifest_fingerprint
from mari import CCRR, MaRI, RI
from mari.metrics.neighbors import (
    _knn_balanced_accuracy_by_k,
    _normalize_k_values,
    _select_k_from_balanced_accuracy,
)
from mari.metrics.pairs import (
    load_manifest,
    normalize_center_values,
    resolve_manifest_subsets,
    retain_complete_subset_memberships,
)
from mari.types import CCRRResult
from metrics_cache import MetricsArtifactCache, build_cache_key
from metrics_io import (
    ccrr_search_signature,
    excluded_centers_signature,
    k_candidates_signature,
    save_metrics,
)
from progress_utils import model_block, progress_write, resolve_progress_mode
from plotting import (
    plot_benchmark_6panel_summary,
    plot_bio_vs_center_scatter,
    plot_ccrr_ltm_comparison,
    plot_ccrr_m_sweep_with_ltm,
    plot_ccrr_sample_distributions,
    plot_ccrr_trend_quadrants,
    plot_ccrr_vs_mari_scatter,
    plot_knn_bio_k_sweep,
    plot_knn_center_k_sweep,
    plot_mari_k_sweep,
    plot_mari_vs_ri_scatter,
    plot_ri_k_sweep,
)


def _safe_model_name(model: str) -> str:
    return str(model).replace("/", "_").replace(":", "_")


def _sample_distribution_dir(results_dir: Path) -> Path:
    return results_dir / "sample_distributions"


def _distribution_path(results_dir: Path, metric_name: str, model: str) -> Path:
    return _sample_distribution_dir(results_dir) / f"{metric_name}.{_safe_model_name(model)}.npy"


def _distribution_meta_path(results_dir: Path, metric_name: str, model: str) -> Path:
    return _sample_distribution_dir(results_dir) / f"{metric_name}.{_safe_model_name(model)}.json"


def _per_sample_metrics_path(results_dir: Path) -> Path:
    return results_dir / "per_sample_metrics.csv"


def _per_sample_metrics_json_path(results_dir: Path) -> Path:
    return results_dir / "per_sample_metrics.json"


def _per_sample_metrics_by_model_dir(results_dir: Path) -> Path:
    return results_dir / "per_sample_metrics_by_model"


def _per_sample_metrics_by_model_paths(results_dir: Path, model: str) -> tuple[Path, Path]:
    base = _per_sample_metrics_by_model_dir(results_dir) / _safe_model_name(model)
    return base.with_suffix(".csv"), base.with_suffix(".json")


def _npy_matches_shape(values: np.ndarray | None, expected_shape: tuple[int, ...]) -> bool:
    if values is None:
        return False
    return tuple(int(v) for v in values.shape) == tuple(int(v) for v in expected_shape)


def _save_mari_sample_distribution(
    *,
    results_dir: Path,
    model: str,
    dataset: str,
    evaluation_design: str,
    evaluation_unit: str,
    tau: float,
    selected_k: int,
    n_total_units: int,
    n_undefined_units: int,
    values: np.ndarray,
) -> Path:
    out_path = _distribution_path(results_dir, "mari", model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=float)
    np.save(out_path, arr)
    meta_path = _distribution_meta_path(results_dir, "mari", model)
    meta = {
        "dataset": str(dataset),
        "model": str(model),
        "evaluation_design": str(evaluation_design),
        "evaluation_unit": str(evaluation_unit),
        "tau": float(tau),
        "k": int(selected_k),
        "n_total_units": int(n_total_units),
        "n_undefined_units": int(n_undefined_units),
        "n_defined_units": int(arr.shape[0]),
        "distribution_path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_path


def _save_ri_sample_distribution(
    *,
    results_dir: Path,
    model: str,
    dataset: str,
    evaluation_design: str,
    evaluation_unit: str,
    selected_k: int,
    n_total_units: int,
    n_undefined_units: int,
    values: np.ndarray,
) -> Path:
    out_path = _distribution_path(results_dir, "ri", model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=float)
    np.save(out_path, arr)
    meta_path = _distribution_meta_path(results_dir, "ri", model)
    meta = {
        "dataset": str(dataset),
        "model": str(model),
        "evaluation_design": str(evaluation_design),
        "evaluation_unit": str(evaluation_unit),
        "k": int(selected_k),
        "n_total_units": int(n_total_units),
        "n_undefined_units": int(n_undefined_units),
        "n_defined_units": int(arr.shape[0]),
        "distribution_path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified benchmark pipeline: extract embeddings, compute RI/MaRI metrics, and plot results."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest CSV.")
    parser.add_argument("--dataset-name", default="dataset", help="Dataset label for metrics output.")
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names. If omitted, all registered models are evaluated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Benchmark output directory.")
    parser.add_argument(
        "--evaluation-design",
        default="paired_2x2",
        choices=["paired_2x2", "dataset_wide"],
        help="Evaluation design for RI/MaRI/CCRR.",
    )
    parser.add_argument("--k-candidates", default="3,5,7,10,15,20,25", help="Comma-separated k candidates.")
    parser.add_argument(
        "--continuous-k-sweep-max",
        type=int,
        default=0,
        help="If > 0, sweep k continuously from 1..max instead of only --k-candidates.",
    )
    parser.add_argument("--tau", type=float, default=0.2, help="MaRI tau.")
    parser.add_argument(
        "--ccrr-m-max",
        type=int,
        default=20,
        help="Maximum m for CCRR sweep. All integers 1..m_max are evaluated at no extra search cost (default 20).",
    )
    parser.add_argument(
        "--ccrr-start-k",
        type=int,
        default=200,
        help="Initial k for CCRR iterative neighbor search (default 200).",
    )
    parser.add_argument(
        "--ccrr-k-growth-factor",
        type=float,
        default=2.0,
        help="Geometric growth factor for CCRR iterative k search (>1, default 2.0).",
    )
    parser.add_argument(
        "--ccrr-alpha",
        type=float,
        default=0.10,
        help="Tail percentile alpha used for CCRR Q_alpha/LTM_alpha reporting (default 0.10).",
    )
    parser.add_argument(
        "--exclude-center",
        action="append",
        default=[],
        help="Medical center to exclude from computation. Repeat flag to exclude multiple centers.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress display mode: auto=TTY only, on=always, off=never.",
    )
    parser.add_argument("--force-embed", action="store_true", help="Force re-extraction of embeddings.")
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Force recomputation of metrics even if compatible cache exists.",
    )
    return parser.parse_args()


def _resolve_models(raw_models: str, registry: dict[str, ModelSpec]) -> list[str]:
    if str(raw_models).strip():
        models = _parse_models(raw_models)
        unknown = [m for m in models if m not in registry]
        if unknown:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown model(s): {unknown}. Available: {available}")
        return models
    return list(registry.keys())


def _resolve_sweep_k_values(k_candidates: list[int], continuous_k_sweep_max: int) -> list[int]:
    if int(continuous_k_sweep_max) > 0:
        return list(range(1, int(continuous_k_sweep_max) + 1))
    return _normalize_k_values(k_candidates)


def _prepare_eval_manifest(
    *,
    manifest_df: pd.DataFrame,
    dataset_name: str,
    excluded_centers: list[str] | tuple[str, ...],
    evaluation_design: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    center_series = manifest_df["medical_center"].map(str).str.strip()
    keep_mask = ~center_series.isin(excluded_centers)
    if not bool(keep_mask.any()):
        excluded_txt = ", ".join(excluded_centers)
        raise ValueError(
            f"No samples remain after excluding centers [{excluded_txt}] from dataset '{dataset_name}'"
        )

    eval_manifest = manifest_df.loc[keep_mask].copy()
    eval_manifest["_embedding_source_index"] = np.flatnonzero(keep_mask.to_numpy())
    if evaluation_design == "paired_2x2":
        if "subset" not in eval_manifest.columns:
            raise ValueError(
                f"manifest for dataset '{dataset_name}' must define a 'subset' column for paired_2x2 evaluation"
            )
        eval_manifest = retain_complete_subset_memberships(eval_manifest)
        if len(eval_manifest) == 0:
            raise ValueError(f"No evaluable subset-defined samples remain for dataset '{dataset_name}'")
    elif len(eval_manifest) == 0:
        raise ValueError(f"No evaluable samples remain for dataset '{dataset_name}'")

    keep_indices = eval_manifest["_embedding_source_index"].to_numpy(dtype=int)
    eval_manifest = eval_manifest.drop(columns=["_embedding_source_index"]).reset_index(drop=True)
    return eval_manifest, keep_indices


def _build_aligned_manifest(
    *,
    eval_manifest: pd.DataFrame,
    evaluation_design: str,
) -> pd.DataFrame:
    if evaluation_design == "dataset_wide":
        aligned_manifest = eval_manifest.copy().reset_index(drop=True)
        aligned_manifest["source_sample_index"] = np.arange(len(aligned_manifest), dtype=int)
        aligned_manifest["subset"] = "dataset"
        return aligned_manifest

    subsets = resolve_manifest_subsets(eval_manifest)
    if not subsets:
        raise RuntimeError("paired_2x2 evaluation requires at least one complete manifest-defined subset")
    aligned_manifest = pd.concat([subset.rows for subset in subsets], ignore_index=True)
    aligned_manifest["source_sample_index"] = aligned_manifest["source_sample_index"].astype(int)
    aligned_manifest["subset"] = aligned_manifest["subset"].astype(str)
    return aligned_manifest.reset_index(drop=True)


def _build_per_sample_rows(
    *,
    aligned_manifest: pd.DataFrame,
    dataset_name: str,
    model: str,
    evaluation_design: str,
    evaluation_unit: str,
    selected_k: int,
    tau: float,
    ccrr_alpha: float,
    ccrr_search_sig: str,
    excluded_centers_sig: str,
    ri_samples_aligned: np.ndarray,
    mari_samples_aligned: np.ndarray,
    ri_defined_mask: np.ndarray,
    mari_defined_mask: np.ndarray,
    ri_undefined_types: np.ndarray,
    mari_undefined_types: np.ndarray,
    ccrr_samples_aligned_by_m: np.ndarray,
    ccrr_m_values: list[int],
) -> list[dict]:
    rows: list[dict] = []
    if ri_samples_aligned.shape != mari_samples_aligned.shape:
        raise RuntimeError("RI and MaRI aligned arrays must have the same shape")
    if ri_samples_aligned.shape[0] != len(aligned_manifest):
        raise RuntimeError("Aligned metric arrays must match the aligned evaluation manifest row count")
    if ccrr_samples_aligned_by_m.shape != (len(aligned_manifest), len(ccrr_m_values)):
        raise RuntimeError("Aligned CCRR array must match the aligned evaluation manifest row count and m sweep")

    for occurrence_index, sample_row in aligned_manifest.reset_index(drop=True).iterrows():
        source_sample_index = int(sample_row["source_sample_index"])
        record = {
            "dataset": str(dataset_name),
            "model": str(model),
            "evaluation_design": str(evaluation_design),
            "evaluation_unit": str(evaluation_unit),
            "occurrence_index": int(occurrence_index),
            "sample_index": int(source_sample_index),
            "source_sample_index": int(source_sample_index),
            "subset": str(sample_row.get("subset", "dataset")),
            "sample_id": str(sample_row["sample_id"]),
            "slide_id": str(sample_row["slide_id"]),
            "label": str(sample_row["label"]),
            "medical_center": str(sample_row["medical_center"]),
            "k": int(selected_k),
            "tau": float(tau),
            "ccrr_alpha": float(ccrr_alpha),
            "ccrr_search": str(ccrr_search_sig),
            "excluded_centers": str(excluded_centers_sig),
            "ri": float(ri_samples_aligned[occurrence_index]),
            "mari": float(mari_samples_aligned[occurrence_index]),
            "ri_defined": bool(ri_defined_mask[occurrence_index]),
            "mari_defined": bool(mari_defined_mask[occurrence_index]),
            "ri_undefined_type": int(ri_undefined_types[occurrence_index]),
            "mari_undefined_type": int(mari_undefined_types[occurrence_index]),
        }
        for m_pos, m in enumerate(ccrr_m_values):
            record[f"ccrr_m{int(m)}"] = float(ccrr_samples_aligned_by_m[occurrence_index, m_pos])
        rows.append(record)
    return rows


def _knn_balanced_accuracy_by_k_for_design(
    *,
    features: np.ndarray,
    manifest: pd.DataFrame,
    target_column: str,
    k_values: list[int],
    evaluation_design: str,
    warn_context: str,
) -> dict[int, float]:
    if evaluation_design == "dataset_wide":
        labels = pd.factorize(manifest[target_column])[0].astype(int)
        slide_ids = manifest["slide_id"].astype(str).to_numpy()
        return _knn_balanced_accuracy_by_k(
            features=features,
            labels=labels,
            slide_ids=slide_ids,
            k_values=k_values,
            warn_context=warn_context,
        )

    subset_scores: list[dict[int, float]] = []
    grouped = manifest.groupby("subset", sort=True, dropna=False)
    for _subset_id, subset_df in grouped:
        if len(subset_df) <= 1:
            continue
        idx = subset_df.index.to_numpy(dtype=int)
        labels = pd.factorize(subset_df[target_column])[0].astype(int)
        slide_ids = subset_df["slide_id"].astype(str).to_numpy()
        valid_k = [int(k) for k in k_values if int(k) < len(subset_df)]
        if not valid_k:
            continue
        scores = _knn_balanced_accuracy_by_k(
            features=features[idx],
            labels=labels,
            slide_ids=slide_ids,
            k_values=valid_k,
            warn_context=warn_context,
        )
        subset_scores.append(scores)

    if not subset_scores:
        raise RuntimeError(f"{warn_context}: no evaluable paired_2x2 subsets remain for kNN curves")

    out: dict[int, float] = {}
    for k in k_values:
        vals = [scores[int(k)] for scores in subset_scores if int(k) in scores]
        if vals:
            out[int(k)] = float(np.mean(vals))
    return out



def _curve_payload(values: dict[int, float]) -> dict:
    return {"values": {str(int(k)): float(v) for k, v in sorted(values.items(), key=lambda kv: int(kv[0]))}}


def _curve_from_payload(payload: dict, *, expected_k_values: list[int]) -> dict[int, float] | None:
    if not isinstance(payload, dict):
        return None
    values = payload.get("values")
    if not isinstance(values, dict):
        return None
    try:
        parsed = {int(k): float(v) for k, v in values.items()}
    except Exception:  # noqa: BLE001
        return None
    if set(parsed) != {int(k) for k in expected_k_values}:
        return None
    return parsed


def _summary_from_payload(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    required = ("k", "value", "std", "undefined_frac")
    for key in required:
        if key not in payload:
            return None
    try:
        result = {
            "k": int(payload["k"]),
            "value": float(payload["value"]),
            "std": float(payload["std"]),
            "undefined_frac": float(payload["undefined_frac"]),
            "ss_dominated_undefined_frac": float(payload.get("ss_dominated_undefined_frac", 0.0)),
            "oo_dominated_undefined_frac": float(payload.get("oo_dominated_undefined_frac", 0.0)),
            "mixed_undefined_frac": float(payload.get("mixed_undefined_frac", 0.0)),
            "evaluation_design": str(payload.get("evaluation_design", "paired_2x2")),
            "evaluation_unit": str(payload.get("evaluation_unit", "occurrence")),
        }
        return result
    except Exception:  # noqa: BLE001
        return None


def _ccrr_result_to_payload(result: CCRRResult, m: int) -> dict:
    return {
        "m": int(m),
        "ccrr": float(result.value),
        "ccrr_std": float(result.std),
        "ccrr_undefined_frac": float(result.undefined_frac),
        "ccrr_k_start": int(result.k_start),
        "ccrr_k_final": int(result.k_final),
        "ccrr_retries": int(result.retries),
        "ccrr_alpha": float(result.alpha),
        "ccrr_q_alpha": float(result.q_alpha),
        "ccrr_ltm_alpha": float(result.ltm_alpha),
    }


def _ccrr_payload_from_results(results: dict[int, CCRRResult]) -> dict:
    return {
        "by_m": {
            str(int(m)): _ccrr_result_to_payload(result=res, m=int(m))
            for m, res in sorted(results.items(), key=lambda kv: int(kv[0]))
        }
    }


def _ccrr_payload_to_by_m(payload: dict, *, expected_m_values: list[int]) -> dict[int, dict] | None:
    if not isinstance(payload, dict):
        return None
    raw_by_m = payload.get("by_m")
    if not isinstance(raw_by_m, dict):
        return None
    try:
        by_m = {int(k): dict(v) for k, v in raw_by_m.items()}
    except Exception:  # noqa: BLE001
        return None
    if set(by_m) != {int(m) for m in expected_m_values}:
        return None
    return by_m


def _compute_ccrr_by_m(
    *,
    features: np.ndarray,
    manifest: pd.DataFrame,
    evaluation_design: str,
    m_values: list[int],
    ccrr_start_k: int,
    ccrr_k_growth_factor: float,
    ccrr_alpha: float,
) -> dict[int, CCRRResult]:
    return cast(
        dict[int, CCRRResult],
        CCRR.compute(
            features=features,
            manifest=manifest,
            evaluation_design=evaluation_design,
            m=[int(m) for m in m_values],
            alpha=float(ccrr_alpha),
            start_k=int(ccrr_start_k),
            k_growth_factor=float(ccrr_k_growth_factor),
        ),
    )



def main() -> int:
    args = _parse_args()
    progress_enabled = resolve_progress_mode(str(args.progress))
    if int(args.ccrr_start_k) < 1:
        raise ValueError("--ccrr-start-k must be >= 1")
    if float(args.ccrr_k_growth_factor) <= 1.0:
        raise ValueError("--ccrr-k-growth-factor must be > 1")
    if float(args.ccrr_alpha) <= 0.0 or float(args.ccrr_alpha) > 1.0:
        raise ValueError("--ccrr-alpha must be in (0, 1]")
    if int(args.ccrr_m_max) < 1:
        raise ValueError("--ccrr-m-max must be >= 1")

    registry = _build_model_registry()
    models = _resolve_models(args.models, registry)

    output_dir = args.output_dir
    dataset_dir = output_dir / args.manifest.stem
    embeddings_dir = dataset_dir / "embeddings"
    results_dir = dataset_dir / "results"
    plots_dir = dataset_dir / "plots"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = results_dir / "metrics.csv"
    metrics_json = results_dir / "metrics.json"
    k_sweep_csv = results_dir / "k_sweep_metrics.csv"
    k_sweep_json = results_dir / "k_sweep_metrics.json"
    ccrr_m_sweep_csv = results_dir / "ccrr_m_sweep_metrics.csv"
    ccrr_m_sweep_json = results_dir / "ccrr_m_sweep_metrics.json"
    per_sample_csv = _per_sample_metrics_path(results_dir)
    per_sample_json = _per_sample_metrics_json_path(results_dir)

    cache = MetricsArtifactCache(results_dir=results_dir)

    k_candidates = parse_k_candidates(args.k_candidates)
    k_values = _resolve_sweep_k_values(
        k_candidates=k_candidates,
        continuous_k_sweep_max=int(args.continuous_k_sweep_max),
    )
    ccrr_m_values = list(range(1, int(args.ccrr_m_max) + 1))
    k_candidates_sig = k_candidates_signature(k_values)
    excluded_centers = normalize_center_values(args.exclude_center)
    excluded_centers_sig = excluded_centers_signature(excluded_centers)
    ccrr_search_sig = ccrr_search_signature(
        start_k=int(args.ccrr_start_k),
        k_growth_factor=float(args.ccrr_k_growth_factor),
        alpha=float(args.ccrr_alpha),
    )

    extraction_status: dict[str, str] = {}
    metrics_status: dict[str, str] = {}
    failures: list[str] = []
    evaluation_design = str(args.evaluation_design)
    rows: list[dict] = []
    k_sweep_rows: list[dict] = []
    ccrr_m_sweep_rows: list[dict] = []
    per_sample_rows: list[dict] = []
    per_sample_rows_by_model: dict[str, list[dict]] = {}

    progress_write(f"[benchmark] manifest={args.manifest}", enabled=progress_enabled)
    progress_write(f"[benchmark] models={', '.join(models)}", enabled=progress_enabled)
    progress_write(f"[benchmark] output_dir={output_dir}", enabled=progress_enabled)
    progress_write(f"[benchmark] dataset_dir={dataset_dir}", enabled=progress_enabled)
    progress_write(f"[benchmark] evaluation_design={evaluation_design}", enabled=progress_enabled)
    manifest_df = load_manifest(str(args.manifest), dataset_name=str(args.dataset_name))
    base_manifest_fingerprint = manifest_fingerprint(manifest_df)
    eval_manifest, keep_indices = _prepare_eval_manifest(
        manifest_df=manifest_df,
        dataset_name=str(args.dataset_name),
        excluded_centers=excluded_centers,
        evaluation_design=evaluation_design,
    )
    aligned_manifest = _build_aligned_manifest(
        eval_manifest=eval_manifest,
        evaluation_design=evaluation_design,
    )

    for i, model in enumerate(models):
        with model_block(model, i + 1, len(models), enabled=progress_enabled) as ticker:
            output_path = ee._output_path_in_dir(args.manifest, embeddings_dir, model)
            spec = registry[model]
            ticker.start("embed")
            if output_path.exists() and not args.force_embed:
                ticker.log(f"[benchmark] embedding cache hit -> {output_path}")
                extraction_status[model] = "skipped"
                ticker.done("embed", cached=True)
            else:
                try:
                    ee.embed_manifest(
                        manifest_path=args.manifest,
                        output_path=output_path,
                        spec=spec,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        device_arg=str(args.device),
                        progress_enabled=progress_enabled,
                        tile_progress_leave=False,
                    )
                    extraction_status[model] = "ok"
                    ticker.done("embed")
                except Exception as exc:  # noqa: BLE001
                    extraction_status[model] = "failed"
                    metrics_status[model] = "failed"
                    failures.append(f"{model}: extraction failed ({exc})")
                    ticker.log(f"[benchmark] extraction failed: {exc}")
                    continue

            try:
                embedding_fp = embedding_fingerprint(output_path)
                input_fp = {
                    "manifest_fingerprint": base_manifest_fingerprint,
                    "embedding_fingerprint": embedding_fp,
                    "excluded_centers_signature": excluded_centers_sig,
                }
    
                k_values_param = [int(k) for k in k_values]
                tau_value = float(args.tau)

                keys = {
                    "knn_bio_curve": build_cache_key(
                        artifact_name="knn_bio_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param},
                    ),
                    "knn_center_curve": build_cache_key(
                        artifact_name="knn_center_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param},
                    ),
                    "ri_curve": build_cache_key(
                        artifact_name="ri_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param},
                    ),
                    "mari_curve": build_cache_key(
                        artifact_name="mari_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param, "tau": tau_value},
                    ),
                    "ri_summary": build_cache_key(
                        artifact_name="ri_summary",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param},
                    ),
                    "ri_samples": build_cache_key(
                        artifact_name="ri_samples",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param},
                    ),
                    "ri_samples_aligned": build_cache_key(
                        artifact_name="ri_samples_aligned",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param},
                    ),
                    "ri_undefined_types": build_cache_key(
                        artifact_name="ri_undefined_types",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param},
                    ),
                    "mari_summary": build_cache_key(
                        artifact_name="mari_summary",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param, "tau": tau_value},
                    ),
                    "mari_samples": build_cache_key(
                        artifact_name="mari_samples",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param, "tau": tau_value},
                    ),
                    "mari_samples_aligned": build_cache_key(
                        artifact_name="mari_samples_aligned",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param, "tau": tau_value},
                    ),
                    "mari_undefined_types": build_cache_key(
                        artifact_name="mari_undefined_types",
                        model=model,
                        input_fingerprint=input_fp,
                        params={"evaluation_design": evaluation_design, "k_values": k_values_param, "tau": tau_value},
                    ),
                    "ccrr_m_sweep": build_cache_key(
                        artifact_name="ccrr_m_sweep",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "m_max": int(args.ccrr_m_max),
                            "start_k": int(args.ccrr_start_k),
                            "k_growth_factor": float(args.ccrr_k_growth_factor),
                            "alpha": float(args.ccrr_alpha),
                        },
                    ),
                    "ccrr_m1_samples": build_cache_key(
                        artifact_name="ccrr_m1_samples",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "m_max": int(args.ccrr_m_max),
                            "start_k": int(args.ccrr_start_k),
                            "k_growth_factor": float(args.ccrr_k_growth_factor),
                            "alpha": float(args.ccrr_alpha),
                        },
                    ),
                    "ccrr_samples_aligned_by_m": build_cache_key(
                        artifact_name="ccrr_samples_aligned_by_m",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "m_max": int(args.ccrr_m_max),
                            "start_k": int(args.ccrr_start_k),
                            "k_growth_factor": float(args.ccrr_k_growth_factor),
                            "alpha": float(args.ccrr_alpha),
                        },
                    ),
                }
    
                knn_bacc_by_k: dict[int, float] | None = None
                knn_center_bacc_by_k: dict[int, float] | None = None
                ri_curve: dict[int, float] | None = None
                mari_curve: dict[int, float] | None = None
                ri_summary: dict | None = None
                ri_samples: np.ndarray | None = None
                ri_samples_aligned: np.ndarray | None = None
                ri_undefined_types: np.ndarray | None = None
                mari_summary: dict | None = None
                mari_samples: np.ndarray | None = None
                mari_samples_aligned: np.ndarray | None = None
                mari_undefined_types: np.ndarray | None = None
                ccrr_by_m: dict[int, dict] | None = None
                ccrr_samples: np.ndarray | None = None
                ccrr_samples_aligned_by_m: np.ndarray | None = None
    
                all_cache_hit = not bool(args.recompute_metrics)
    
                if not args.recompute_metrics:
                    knn_bacc_by_k = _curve_from_payload(
                        cache.get_json(key=keys["knn_bio_curve"]),
                        expected_k_values=k_values,
                    )
                    if knn_bacc_by_k is None:
                        all_cache_hit = False
    
                    knn_center_bacc_by_k = _curve_from_payload(
                        cache.get_json(key=keys["knn_center_curve"]),
                        expected_k_values=k_values,
                    )
                    if knn_center_bacc_by_k is None:
                        all_cache_hit = False
    
                    ri_curve = _curve_from_payload(
                        cache.get_json(key=keys["ri_curve"]),
                        expected_k_values=k_values,
                    )
                    if ri_curve is None:
                        all_cache_hit = False
    
                    mari_curve = _curve_from_payload(
                        cache.get_json(key=keys["mari_curve"]),
                        expected_k_values=k_values,
                    )
                    if mari_curve is None:
                        all_cache_hit = False
    
                    ri_summary = _summary_from_payload(cache.get_json(key=keys["ri_summary"]))
                    ri_samples = cache.get_npy(key=keys["ri_samples"])
                    ri_samples_aligned = cache.get_npy(key=keys["ri_samples_aligned"])
                    ri_undefined_types = cache.get_npy(key=keys["ri_undefined_types"])
                    if (
                        ri_summary is None
                        or ri_samples is None
                        or not _npy_matches_shape(ri_samples_aligned, (len(aligned_manifest),))
                        or not _npy_matches_shape(ri_undefined_types, (len(aligned_manifest),))
                        or str(ri_summary["evaluation_design"]) != evaluation_design
                        or str(ri_summary["evaluation_unit"]) not in {"sample", "occurrence"}
                    ):
                        all_cache_hit = False

                    mari_summary = _summary_from_payload(cache.get_json(key=keys["mari_summary"]))
                    mari_samples = cache.get_npy(key=keys["mari_samples"])
                    mari_samples_aligned = cache.get_npy(key=keys["mari_samples_aligned"])
                    mari_undefined_types = cache.get_npy(key=keys["mari_undefined_types"])
                    if (
                        mari_summary is None
                        or mari_samples is None
                        or not _npy_matches_shape(mari_samples_aligned, (len(aligned_manifest),))
                        or not _npy_matches_shape(mari_undefined_types, (len(aligned_manifest),))
                        or str(mari_summary["evaluation_design"]) != evaluation_design
                        or str(mari_summary["evaluation_unit"]) not in {"sample", "occurrence"}
                    ):
                        all_cache_hit = False
    
                    ccrr_by_m = _ccrr_payload_to_by_m(
                        cache.get_json(key=keys["ccrr_m_sweep"]),
                        expected_m_values=ccrr_m_values,
                    )
                    ccrr_samples = cache.get_npy(key=keys["ccrr_m1_samples"])
                    ccrr_samples_aligned_by_m = cache.get_npy(key=keys["ccrr_samples_aligned_by_m"])
                    if (
                        ccrr_by_m is None
                        or ccrr_samples is None
                        or not _npy_matches_shape(
                            ccrr_samples_aligned_by_m,
                            (len(aligned_manifest), len(ccrr_m_values)),
                        )
                    ):
                        all_cache_hit = False
                else:
                    all_cache_hit = False
    
                features_full: np.ndarray | None = None
                eval_features: np.ndarray | None = None
    
                def _ensure_eval_features() -> np.ndarray:
                    nonlocal features_full, eval_features
                    if eval_features is None:
                        if features_full is None:
                            features_full = np.load(output_path)
                        eval_features = features_full[keep_indices]
                    return eval_features
    
                knn_was_cached = knn_bacc_by_k is not None and knn_center_bacc_by_k is not None
                ticker.start("knn")
                if knn_bacc_by_k is None:
                    knn_bacc_by_k = _knn_balanced_accuracy_by_k_for_design(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        target_column="label",
                        k_values=k_values,
                        evaluation_design=evaluation_design,
                        warn_context=f"{args.dataset_name} k-curve",
                    )
                    cache.put_json(key=keys["knn_bio_curve"], payload=_curve_payload(knn_bacc_by_k))

                if knn_center_bacc_by_k is None:
                    knn_center_bacc_by_k = _knn_balanced_accuracy_by_k_for_design(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        target_column="medical_center",
                        k_values=k_values,
                        evaluation_design=evaluation_design,
                        warn_context=f"{args.dataset_name} center-k-curve",
                    )
                    cache.put_json(key=keys["knn_center_curve"], payload=_curve_payload(knn_center_bacc_by_k))
    
                selected_k = _select_k_from_balanced_accuracy(
                    k_values=k_values,
                    scores=knn_bacc_by_k,
                )
                selected_k_center = _select_k_from_balanced_accuracy(
                    k_values=k_values,
                    scores=knn_center_bacc_by_k,
                )
                ticker.done("knn", cached=knn_was_cached)

                ri_was_cached = (
                    ri_curve is not None
                    and ri_summary is not None
                    and ri_samples is not None
                    and ri_samples_aligned is not None
                    and ri_undefined_types is not None
                )
                ticker.start("RI")
                if ri_curve is None:
                    ri_curve = RI.compute_curve(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        evaluation_design=evaluation_design,
                        k_values=k_values,
                    )
                    cache.put_json(key=keys["ri_curve"], payload=_curve_payload(ri_curve))

                if ri_summary is None or ri_samples is None or ri_samples_aligned is None or ri_undefined_types is None:
                    ri = RI.compute(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        evaluation_design=evaluation_design,
                        k_candidates=k_values,
                    )
                    if int(ri.k) != int(selected_k):
                        raise RuntimeError(
                            f"Inconsistent selected k: RI returned {ri.k} but kNN balanced accuracy selected {selected_k}"
                        )
                    ri_summary = {
                        "k": int(ri.k),
                        "value": float(ri.value),
                        "std": float(ri.std),
                        "undefined_frac": float(ri.undefined_frac),
                        "ss_dominated_undefined_frac": float(ri.ss_dominated_undefined_frac),
                        "oo_dominated_undefined_frac": float(ri.oo_dominated_undefined_frac),
                        "mixed_undefined_frac": float(ri.mixed_undefined_frac),
                        "evaluation_design": str(ri.evaluation_design),
                        "evaluation_unit": str(ri.evaluation_unit),
                    }
                    ri_samples = np.asarray(ri.sample_values, dtype=float)
                    ri_samples_aligned = np.asarray(ri.sample_values_aligned, dtype=float)
                    ri_undefined_types = np.asarray(ri.sample_undefined_types, dtype=int)
                    cache.put_json(key=keys["ri_summary"], payload=ri_summary)
                    cache.put_npy(key=keys["ri_samples"], values=ri_samples)
                    cache.put_npy(key=keys["ri_samples_aligned"], values=ri_samples_aligned)
                    cache.put_npy(key=keys["ri_undefined_types"], values=ri_undefined_types)
                else:
                    ri_samples = np.asarray(ri_samples, dtype=float)
                    if ri_samples_aligned is not None:
                        ri_samples_aligned = np.asarray(ri_samples_aligned, dtype=float)
                    if ri_undefined_types is not None:
                        ri_undefined_types = np.asarray(ri_undefined_types, dtype=int)
                ticker.done("RI", cached=ri_was_cached)

                mari_was_cached = (
                    mari_curve is not None
                    and mari_summary is not None
                    and mari_samples is not None
                    and mari_samples_aligned is not None
                    and mari_undefined_types is not None
                )
                ticker.start("MaRI")
                if mari_curve is None:
                    mari_curve = MaRI.compute_curve(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        evaluation_design=evaluation_design,
                        k_values=k_values,
                        tau=float(args.tau),
                    )
                    cache.put_json(key=keys["mari_curve"], payload=_curve_payload(mari_curve))

                if mari_summary is None or mari_samples is None or mari_samples_aligned is None or mari_undefined_types is None:
                    mari = MaRI.compute(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        evaluation_design=evaluation_design,
                        k_candidates=k_values,
                        tau=float(args.tau),
                    )
                    if int(mari.k) != int(selected_k):
                        raise RuntimeError(
                            f"Inconsistent selected k: MaRI returned {mari.k} but kNN balanced accuracy selected {selected_k}"
                        )
                    mari_summary = {
                        "k": int(mari.k),
                        "value": float(mari.value),
                        "std": float(mari.std),
                        "undefined_frac": float(mari.undefined_frac),
                        "ss_dominated_undefined_frac": float(mari.ss_dominated_undefined_frac),
                        "oo_dominated_undefined_frac": float(mari.oo_dominated_undefined_frac),
                        "mixed_undefined_frac": float(mari.mixed_undefined_frac),
                        "evaluation_design": str(mari.evaluation_design),
                        "evaluation_unit": str(mari.evaluation_unit),
                    }
                    mari_samples = np.asarray(mari.sample_values, dtype=float)
                    mari_samples_aligned = np.asarray(mari.sample_values_aligned, dtype=float)
                    mari_undefined_types = np.asarray(mari.sample_undefined_types, dtype=int)
                    cache.put_json(key=keys["mari_summary"], payload=mari_summary)
                    cache.put_npy(key=keys["mari_samples"], values=mari_samples)
                    cache.put_npy(key=keys["mari_samples_aligned"], values=mari_samples_aligned)
                    cache.put_npy(key=keys["mari_undefined_types"], values=mari_undefined_types)
                else:
                    mari_samples = np.asarray(mari_samples, dtype=float)
                    if mari_samples_aligned is not None:
                        mari_samples_aligned = np.asarray(mari_samples_aligned, dtype=float)
                    if mari_undefined_types is not None:
                        mari_undefined_types = np.asarray(mari_undefined_types, dtype=int)
                ticker.done("MaRI", cached=mari_was_cached)
    
                ccrr_was_cached = (
                    ccrr_by_m is not None
                    and ccrr_samples is not None
                    and ccrr_samples_aligned_by_m is not None
                )
                ticker.start("CCRR")
                if ccrr_by_m is None or ccrr_samples is None or ccrr_samples_aligned_by_m is None:
                    ccrr_results = _compute_ccrr_by_m(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        evaluation_design=evaluation_design,
                        m_values=ccrr_m_values,
                        ccrr_start_k=int(args.ccrr_start_k),
                        ccrr_k_growth_factor=float(args.ccrr_k_growth_factor),
                        ccrr_alpha=float(args.ccrr_alpha),
                    )
                    ccrr_by_m = _ccrr_payload_to_by_m(
                        _ccrr_payload_from_results(ccrr_results),
                        expected_m_values=ccrr_m_values,
                    )
                    if ccrr_by_m is None:
                        raise RuntimeError("Failed to serialize ccrr m-sweep cache payload")
                    ccrr_samples = np.asarray(ccrr_results[1].sample_values, dtype=float)
                    ccrr_samples_aligned_by_m = np.column_stack(
                        [
                            np.asarray(ccrr_results[int(m)].sample_values_aligned, dtype=float)
                            for m in ccrr_m_values
                        ]
                    )
                    cache.put_json(key=keys["ccrr_m_sweep"], payload={"by_m": {str(k): v for k, v in ccrr_by_m.items()}})
                    cache.put_npy(key=keys["ccrr_m1_samples"], values=ccrr_samples)
                    cache.put_npy(key=keys["ccrr_samples_aligned_by_m"], values=ccrr_samples_aligned_by_m)
                else:
                    ccrr_samples = np.asarray(ccrr_samples, dtype=float)
                    if ccrr_samples_aligned_by_m is not None:
                        ccrr_samples_aligned_by_m = np.asarray(ccrr_samples_aligned_by_m, dtype=float)
                ticker.done("CCRR", cached=ccrr_was_cached)

                evaluation_unit = str(ri_summary["evaluation_unit"])
                ccrr_m_rows_for_model: list[dict] = []
                for m in ccrr_m_values:
                    payload = ccrr_by_m[int(m)]
                    ccrr_m_rows_for_model.append(
                        {
                            "dataset": str(args.dataset_name),
                            "model": str(model),
                            "evaluation_design": evaluation_design,
                            "evaluation_unit": evaluation_unit,
                            "tau": float(args.tau),
                            "k_candidates": str(k_candidates_sig),
                            "excluded_centers": str(excluded_centers_sig),
                            "ccrr_search": str(ccrr_search_sig),
                            "m": int(payload["m"]),
                            "ccrr": float(payload["ccrr"]),
                            "ccrr_std": float(payload["ccrr_std"]),
                            "ccrr_undefined_frac": float(payload["ccrr_undefined_frac"]),
                            "ccrr_k_start": int(payload["ccrr_k_start"]),
                            "ccrr_k_final": int(payload["ccrr_k_final"]),
                            "ccrr_retries": int(payload["ccrr_retries"]),
                            "ccrr_alpha": float(payload["ccrr_alpha"]),
                            "ccrr_q_alpha": float(payload["ccrr_q_alpha"]),
                            "ccrr_ltm_alpha": float(payload["ccrr_ltm_alpha"]),
                            "embedding_path": str(output_path),
                        }
                    )
    
                m_sorted = sorted(ccrr_m_values)
                ccrr_curve = [float(ccrr_by_m[m]["ccrr"]) for m in m_sorted]
                finite_curve = [c for c in ccrr_curve if np.isfinite(c)]
                if len(m_sorted) > 1:
                    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                    ccrr_auc = float(_trapz(ccrr_curve, m_sorted) / (m_sorted[-1] - m_sorted[0]))
                else:
                    ccrr_auc = ccrr_curve[0] if ccrr_curve else float("nan")
                ccrr_min_val = float(min(finite_curve)) if finite_curve else float("nan")
                ccrr_delta = float(ccrr_curve[-1] - ccrr_curve[0]) if len(ccrr_curve) > 1 else 0.0

                total_n = int(len(aligned_manifest))
                ri_undefined_n = int(np.count_nonzero(~np.isfinite(ri_samples_aligned)))
                mari_undefined_n = int(np.count_nonzero(~np.isfinite(mari_samples_aligned)))
                ri_ss_frac = float(ri_summary.get("ss_dominated_undefined_frac", 0.0))
                ri_oo_frac = float(ri_summary.get("oo_dominated_undefined_frac", 0.0))
                ri_mixed_frac = float(ri_summary.get("mixed_undefined_frac", 0.0))
                mari_ss_frac = float(mari_summary.get("ss_dominated_undefined_frac", 0.0))
                mari_oo_frac = float(mari_summary.get("oo_dominated_undefined_frac", 0.0))
                mari_mixed_frac = float(mari_summary.get("mixed_undefined_frac", 0.0))
                saved_dist_path = _save_mari_sample_distribution(
                    results_dir=results_dir,
                    model=model,
                    dataset=str(args.dataset_name),
                    evaluation_design=evaluation_design,
                    evaluation_unit=evaluation_unit,
                    tau=float(args.tau),
                    selected_k=int(selected_k),
                    n_total_units=total_n,
                    n_undefined_units=mari_undefined_n,
                    values=mari_samples,
                )
                saved_ri_dist_path = _save_ri_sample_distribution(
                    results_dir=results_dir,
                    model=model,
                    dataset=str(args.dataset_name),
                    evaluation_design=evaluation_design,
                    evaluation_unit=evaluation_unit,
                    selected_k=int(selected_k),
                    n_total_units=total_n,
                    n_undefined_units=ri_undefined_n,
                    values=ri_samples,
                )
    
                ccrr_result = ccrr_by_m[1]
                ccrr_dist_path = _distribution_path(results_dir, "ccrr", model)
                ccrr_dist_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(ccrr_dist_path, ccrr_samples)
    
                row = {
                    "dataset": str(args.dataset_name),
                    "model": model,
                    "k": int(ri_summary["k"]),
                    "evaluation_design": evaluation_design,
                    "evaluation_unit": evaluation_unit,
                    "tau": float(args.tau),
                    "k_candidates": k_candidates_sig,
                    "excluded_centers": excluded_centers_sig,
                    "ccrr_search": ccrr_search_sig,
                    "bio_knn_bacc": float(knn_bacc_by_k[int(selected_k)]),
                    "center_knn_bacc": float(knn_center_bacc_by_k[int(selected_k_center)]),
                    "selected_k_center": int(selected_k_center),
                    "ri": float(ri_summary["value"]),
                    "ri_std": float(ri_summary["std"]),
                    "mari": float(mari_summary["value"]),
                    "mari_std": float(mari_summary["std"]),
                    "ri_undefined_frac": float(ri_summary["undefined_frac"]),
                    "ri_ss_dominated_undefined_frac": ri_ss_frac,
                    "ri_oo_dominated_undefined_frac": ri_oo_frac,
                    "ri_mixed_undefined_frac": ri_mixed_frac,
                    "mari_undefined_frac": float(mari_summary["undefined_frac"]),
                    "mari_ss_dominated_undefined_frac": mari_ss_frac,
                    "mari_oo_dominated_undefined_frac": mari_oo_frac,
                    "mari_mixed_undefined_frac": mari_mixed_frac,
                    "ri_samples_path": str(saved_ri_dist_path),
                    "mari_samples_path": str(saved_dist_path),
                    "ccrr": float(ccrr_result["ccrr"]),
                    "ccrr_std": float(ccrr_result["ccrr_std"]),
                    "ccrr_m": int(ccrr_result["m"]),
                    "ccrr_undefined_frac": float(ccrr_result["ccrr_undefined_frac"]),
                    "ccrr_k_start": int(ccrr_result["ccrr_k_start"]),
                    "ccrr_k_final": int(ccrr_result["ccrr_k_final"]),
                    "ccrr_retries": int(ccrr_result["ccrr_retries"]),
                    "ccrr_alpha": float(ccrr_result["ccrr_alpha"]),
                    "ccrr_q_alpha": float(ccrr_result["ccrr_q_alpha"]),
                    "ccrr_ltm_alpha": float(ccrr_result["ccrr_ltm_alpha"]),
                    "ccrr_auc": ccrr_auc,
                    "ccrr_min": ccrr_min_val,
                    "ccrr_delta": ccrr_delta,
                    "ccrr_samples_path": str(ccrr_dist_path),
                    "embedding_path": str(output_path),
                }
                rows.append(row)
                model_per_sample_rows = _build_per_sample_rows(
                    aligned_manifest=aligned_manifest,
                    dataset_name=str(args.dataset_name),
                    model=str(model),
                    evaluation_design=evaluation_design,
                    evaluation_unit=evaluation_unit,
                    selected_k=int(ri_summary["k"]),
                    tau=float(args.tau),
                    ccrr_alpha=float(args.ccrr_alpha),
                    ccrr_search_sig=str(ccrr_search_sig),
                    excluded_centers_sig=str(excluded_centers_sig),
                    ri_samples_aligned=np.asarray(ri_samples_aligned, dtype=float),
                    mari_samples_aligned=np.asarray(mari_samples_aligned, dtype=float),
                    ri_defined_mask=np.isfinite(np.asarray(ri_samples_aligned, dtype=float)),
                    mari_defined_mask=np.isfinite(np.asarray(mari_samples_aligned, dtype=float)),
                    ri_undefined_types=np.asarray(ri_undefined_types, dtype=int),
                    mari_undefined_types=np.asarray(mari_undefined_types, dtype=int),
                    ccrr_samples_aligned_by_m=np.asarray(ccrr_samples_aligned_by_m, dtype=float),
                    ccrr_m_values=ccrr_m_values,
                )
                per_sample_rows_by_model[str(model)] = model_per_sample_rows
                per_sample_rows.extend(model_per_sample_rows)
                for k in k_values:
                    k_sweep_rows.append(
                        {
                            "dataset": str(args.dataset_name),
                            "model": model,
                            "evaluation_design": evaluation_design,
                            "evaluation_unit": evaluation_unit,
                            "tau": float(args.tau),
                            "k_candidates": k_candidates_sig,
                            "excluded_centers": excluded_centers_sig,
                            "ccrr_search": ccrr_search_sig,
                            "k": int(k),
                            "knn_bacc": float(knn_bacc_by_k[int(k)]),
                            "knn_center_bacc": float(knn_center_bacc_by_k[int(k)]),
                            "ri": float(ri_curve[int(k)]),
                            "mari": float(mari_curve[int(k)]),
                            "selected_k": int(selected_k),
                            "selected_k_center": int(selected_k_center),
                            "continuous_k_sweep": int(int(args.continuous_k_sweep_max) > 0),
                            "embedding_path": str(output_path),
                        }
                    )
                ccrr_m_sweep_rows.extend(ccrr_m_rows_for_model)

                metrics_status[model] = "cached" if all_cache_hit else "ok"
                if all_cache_hit:
                    ticker.log("[benchmark] metrics cache hit")
                else:
                    ticker.log("[benchmark] metrics cache miss: partial/full recompute")
                ticker.log(
                    f"[benchmark] RI={row['ri']:.4f} MaRI={row['mari']:.4f} CCRR={row['ccrr']:.4f}"
                )
                undef_parts = []
                if row["ri_undefined_frac"] > 0.0:
                    undef_parts.append(f"RI={100*row['ri_undefined_frac']:.1f}%")
                if row["mari_undefined_frac"] > 0.0:
                    undef_parts.append(f"MaRI={100*row['mari_undefined_frac']:.1f}%")
                if row["ccrr_undefined_frac"] > 0.0:
                    undef_parts.append(f"CCRR={100*row['ccrr_undefined_frac']:.1f}%")
                if undef_parts:
                    ticker.log(f"[benchmark] undefined samples: {', '.join(undef_parts)}")
            except Exception as exc:  # noqa: BLE001
                metrics_status[model] = "failed"
                failures.append(f"{model}: metrics failed ({exc})")
                ticker.log(f"[benchmark] metrics failed: {exc}")

    if rows:
        save_metrics(rows=rows, csv_path=metrics_csv, json_path=metrics_json)
        save_metrics(rows=k_sweep_rows, csv_path=k_sweep_csv, json_path=k_sweep_json)
        save_metrics(rows=ccrr_m_sweep_rows, csv_path=ccrr_m_sweep_csv, json_path=ccrr_m_sweep_json)
        if per_sample_rows:
            per_sample_rows_sorted = sorted(
                per_sample_rows,
                key=lambda row: (str(row["model"]), int(row["occurrence_index"])),
            )
            save_metrics(rows=per_sample_rows_sorted, csv_path=per_sample_csv, json_path=per_sample_json)
            for model_name, model_rows in per_sample_rows_by_model.items():
                model_csv, model_json = _per_sample_metrics_by_model_paths(results_dir, model_name)
                model_rows_sorted = sorted(model_rows, key=lambda row: int(row["occurrence_index"]))
                save_metrics(rows=model_rows_sorted, csv_path=model_csv, json_path=model_json)
        plot_knn_bio_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "knn_bio_k_sweep.png")
        plot_knn_center_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "knn_center_k_sweep.png")
        plot_ri_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "ri_k_sweep.png")
        plot_mari_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "mari_k_sweep.png")
        plot_ccrr_m_sweep_with_ltm(rows=ccrr_m_sweep_rows, out_path=plots_dir / "ccrr_m_sweep.png")
        plot_ccrr_trend_quadrants(rows=ccrr_m_sweep_rows, out_path=plots_dir / "ccrr_trend_quadrants.png")
        plot_ccrr_ltm_comparison(rows=rows, out_path=plots_dir / "ccrr_ltm_comparison.png")
        plot_bio_vs_center_scatter(rows=rows, out_path=plots_dir / "bio_vs_center_scatter.png")
        plot_mari_vs_ri_scatter(rows=rows, out_path=plots_dir / "mari_vs_ri_scatter.png")
        plot_ccrr_vs_mari_scatter(rows=rows, out_path=plots_dir / "ccrr_vs_mari_scatter.png")
        plot_ccrr_sample_distributions(rows=rows, out_path=plots_dir / "ccrr_sample_distributions.png")
        plot_benchmark_6panel_summary(
            rows=rows,
            k_sweep_rows=k_sweep_rows,
            out_path=plots_dir / "benchmark_6panel_summary.png",
        )

    progress_write("\n[benchmark] === summary ===", enabled=progress_enabled)
    for model in models:
        e = extraction_status.get(model, "n/a")
        m = metrics_status.get(model, "n/a")
        progress_write(f"[benchmark] {model}: extract={e} metrics={m}", enabled=progress_enabled)
    progress_write(f"[benchmark] metrics_csv={metrics_csv}", enabled=progress_enabled)
    progress_write(f"[benchmark] metrics_json={metrics_json}", enabled=progress_enabled)
    progress_write(f"[benchmark] k_sweep_csv={k_sweep_csv}", enabled=progress_enabled)
    progress_write(f"[benchmark] k_sweep_json={k_sweep_json}", enabled=progress_enabled)
    progress_write(f"[benchmark] ccrr_m_sweep_csv={ccrr_m_sweep_csv}", enabled=progress_enabled)
    progress_write(f"[benchmark] ccrr_m_sweep_json={ccrr_m_sweep_json}", enabled=progress_enabled)
    progress_write(f"[benchmark] plots_dir={plots_dir}", enabled=progress_enabled)

    if failures:
        progress_write("[benchmark] failures:", enabled=progress_enabled)
        for msg in failures:
            progress_write(f"[benchmark] - {msg}", enabled=progress_enabled)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
