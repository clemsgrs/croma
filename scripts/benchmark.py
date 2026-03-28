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
from input_fingerprint import embedding_fingerprint, manifest_fingerprint
from croma.alignment import build_embedding_source_manifest
from croma import CCMR, MaRI, RI
from croma.confounders import infer_confounder_display_name
from croma.metrics.neighbors import (
    _knn_balanced_accuracy_by_k,
    _select_k_from_balanced_accuracy,
)
from croma.metrics.pairs import (
    load_manifest,
    resolve_manifest_subsets,
    retain_complete_subset_memberships,
)
from croma.types import CCMRResult
from metrics_cache import MetricsArtifactCache, build_cache_key
from metrics_io import (
    StreamingMetricsWriter,
    ccmr_search_signature,
    k_values_signature,
    save_metrics,
    safe_model_name,
)
from progress_utils import model_block, progress_write, resolve_progress_mode
from plotting import (
    plot_bio_vs_confounder_scatter,
    plot_ccmr_ltm_comparison,
    plot_ccmr_m_sweep_with_ltm,
    plot_ccmr_sample_distributions,
    plot_ccmr_vs_mari_scatter,
    plot_q_alpha_vs_ccmr_scatter,
    plot_knn_bio_k_sweep,
    plot_knn_confounder_k_sweep,
    plot_mari_k_sweep,
    plot_mari_vs_ri_scatter,
    plot_ri_mari_cumulative_mean_k_sweep,
    plot_ri_mari_sample_distributions,
    plot_ri_mari_support,
    plot_ri_k_sweep,
)


def _sample_distribution_dir(results_dir: Path) -> Path:
    return results_dir / "sample_distributions"


def _distribution_path(results_dir: Path, metric_name: str, model: str) -> Path:
    return (
        _sample_distribution_dir(results_dir)
        / f"{metric_name}.{safe_model_name(model)}.npy"
    )


def _distribution_meta_path(results_dir: Path, metric_name: str, model: str) -> Path:
    return (
        _sample_distribution_dir(results_dir)
        / f"{metric_name}.{safe_model_name(model)}.json"
    )


def _per_sample_metrics_path(results_dir: Path) -> Path:
    return results_dir / "per_sample_metrics.csv"


def _per_sample_metrics_json_path(results_dir: Path) -> Path:
    return results_dir / "per_sample_metrics.json"


def _per_sample_metrics_by_model_dir(results_dir: Path) -> Path:
    return results_dir / "per_sample_metrics_by_model"


def _per_sample_metrics_by_model_paths(
    results_dir: Path, model: str
) -> tuple[Path, Path]:
    base = _per_sample_metrics_by_model_dir(results_dir) / safe_model_name(model)
    return base.with_suffix(".csv"), base.with_suffix(".json")


def _npy_matches_shape(
    values: np.ndarray | None, expected_shape: tuple[int, ...]
) -> bool:
    if values is None:
        return False
    return tuple(int(v) for v in values.shape) == tuple(int(v) for v in expected_shape)


def _embedding_manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "embedding_source_manifest.csv"


def _embedding_cache_matches_expected(
    output_path: Path,
    *,
    expected_n_samples: int,
    expected_manifest_fingerprint: str,
    expected_manifest_path: Path,
) -> bool:
    if not output_path.exists():
        return False
    try:
        arr = np.load(output_path, mmap_mode="r")
    except Exception:  # noqa: BLE001
        return False
    if int(arr.shape[0]) != int(expected_n_samples):
        return False
    sidecar_path = output_path.with_suffix(output_path.suffix + ".json")
    if not sidecar_path.exists():
        return False
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(payload, dict):
        return False
    cached_manifest_fingerprint = str(payload.get("manifest_fingerprint", "")).strip()
    if cached_manifest_fingerprint:
        return cached_manifest_fingerprint == str(expected_manifest_fingerprint)
    return str(payload.get("manifest", "")).strip() == str(expected_manifest_path)


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
    median_value: float = float("nan"),
    q_alpha: float = float("nan"),
    ltm_alpha: float = float("nan"),
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
        "median_value": float(median_value),
        "q_alpha": float(q_alpha),
        "ltm_alpha": float(ltm_alpha),
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
    median_value: float = float("nan"),
    q_alpha: float = float("nan"),
    ltm_alpha: float = float("nan"),
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
        "median_value": float(median_value),
        "q_alpha": float(q_alpha),
        "ltm_alpha": float(ltm_alpha),
        "distribution_path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified benchmark pipeline: extract embeddings, compute RI/MaRI metrics, and plot results."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to manifest CSV."
    )
    parser.add_argument(
        "--confounder-column",
        required=True,
        help="Manifest column to treat as the non-biological confounder.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names. If omitted, all registered models are evaluated.",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Benchmark output directory."
    )
    parser.add_argument(
        "--evaluation-design",
        default="paired_2x2",
        choices=["paired_2x2", "dataset_wide"],
        help="Evaluation design for RI/MaRI/CCMR.",
    )
    parser.add_argument(
        "--k-max",
        type=_positive_int,
        default=25,
        help="Maximum k for dense benchmark sweeps; benchmark evaluates all integer k in 1..k_max (default 25).",
    )
    parser.add_argument("--tau", type=float, default=0.2, help="MaRI tau.")
    parser.add_argument(
        "--ccmr-m-max",
        type=int,
        default=20,
        help="Maximum m for CCMR sweep. All integers 1..m_max are evaluated at no extra search cost (default 20).",
    )
    parser.add_argument(
        "--ccmr-start-k",
        type=int,
        default=200,
        help="Initial k for CCMR iterative neighbor search (default 200).",
    )
    parser.add_argument(
        "--ccmr-k-growth-factor",
        type=float,
        default=2.0,
        help="Geometric growth factor for CCMR iterative k search (>1, default 2.0).",
    )
    parser.add_argument(
        "--ccmr-alpha",
        type=float,
        default=0.10,
        help="Tail percentile alpha used for CCMR Q_alpha/LTM_alpha reporting (default 0.10).",
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
    parser.add_argument(
        "--force-embed", action="store_true", help="Force re-extraction of embeddings."
    )
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Force recomputation of metrics even if compatible cache exists.",
    )
    parser.add_argument(
        "--prune-ss-oo",
        action="store_true",
        help=(
            "Prune SS and OO neighbours before counting k. "
            "Each sample's neighbourhood contains only SO/OS neighbours, "
            "eliminating undefined samples caused by SS/OO dominance."
        ),
    )
    parser.add_argument(
        "--summarize-by-mean",
        action="store_true",
        help=(
            "Summarize RI/MaRI as the mean over the full k-curve instead of "
            "selecting a single k via kNN biological accuracy."
        ),
    )
    parser.add_argument(
        "--use-median-k",
        action="store_true",
        help=(
            "Use the median of per-model optimal k values as a shared k for the dataset, "
            "matching the original RI paper's k-selection procedure. "
            "By default each model is evaluated at its own kNN-optimal k."
        ),
    )
    return parser.parse_args()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return int(parsed)


def _resolve_models(raw_models: str, registry: dict[str, ModelSpec]) -> list[str]:
    if str(raw_models).strip():
        return _parse_models(raw_models)
    return list(registry.keys())


def _resolve_sweep_k_values(k_max: int) -> list[int]:
    if int(k_max) <= 0:
        raise ValueError("k_max must be strictly positive")
    return list(range(1, int(k_max) + 1))


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


def _build_aligned_manifest(
    *,
    eval_manifest: pd.DataFrame,
    evaluation_design: str,
) -> pd.DataFrame:
    if evaluation_design == "dataset_wide":
        aligned_manifest = eval_manifest.copy().reset_index(drop=True)
        aligned_manifest["source_sample_index"] = np.arange(
            len(aligned_manifest), dtype=int
        )
        aligned_manifest["subset"] = "dataset"
        return aligned_manifest

    subsets = resolve_manifest_subsets(eval_manifest)
    if not subsets:
        raise RuntimeError(
            "paired_2x2 evaluation requires at least one complete manifest-defined subset"
        )
    aligned_manifest = pd.concat([subset.rows for subset in subsets], ignore_index=True)
    aligned_manifest["source_sample_index"] = aligned_manifest[
        "source_sample_index"
    ].astype(int)
    aligned_manifest["subset"] = aligned_manifest["subset"].astype(str)
    return aligned_manifest.reset_index(drop=True)


def _build_per_sample_rows(
    *,
    aligned_manifest: pd.DataFrame,
    dataset_name: str,
    model: str,
    confounder_column: str,
    confounder_display_name: str,
    evaluation_design: str,
    evaluation_unit: str,
    selected_k: int,
    tau: float,
    ccmr_alpha: float,
    ccmr_search_sig: str,
    ri_samples_aligned: np.ndarray,
    mari_samples_aligned: np.ndarray,
    ri_defined_mask: np.ndarray,
    mari_defined_mask: np.ndarray,
    ri_undefined_types: np.ndarray,
    mari_undefined_types: np.ndarray,
    ccmr_samples_aligned_by_m: np.ndarray,
    ccmr_m_values: list[int],
) -> list[dict]:
    rows: list[dict] = []
    if ri_samples_aligned.shape != mari_samples_aligned.shape:
        raise RuntimeError("RI and MaRI aligned arrays must have the same shape")
    if ri_samples_aligned.shape[0] != len(aligned_manifest):
        raise RuntimeError(
            "Aligned metric arrays must match the aligned evaluation manifest row count"
        )
    if ccmr_samples_aligned_by_m.shape != (len(aligned_manifest), len(ccmr_m_values)):
        raise RuntimeError(
            "Aligned CCMR array must match the aligned evaluation manifest row count and m sweep"
        )

    for occurrence_index, sample_row in aligned_manifest.reset_index(
        drop=True
    ).iterrows():
        source_sample_index = int(sample_row["source_sample_index"])
        record = {
            "dataset": str(dataset_name),
            "model": str(model),
            "confounder_column": str(confounder_column),
            "confounder_display_name": str(confounder_display_name),
            "evaluation_design": str(evaluation_design),
            "evaluation_unit": str(evaluation_unit),
            "occurrence_index": int(occurrence_index),
            "sample_index": int(source_sample_index),
            "source_sample_index": int(source_sample_index),
            "subset": str(sample_row.get("subset", "dataset")),
            "sample_id": str(sample_row["sample_id"]),
            "slide_id": str(sample_row["slide_id"]),
            "label": str(sample_row["label"]),
            "confounder": str(sample_row["confounder"]),
            "k": int(selected_k),
            "tau": float(tau),
            "ccmr_alpha": float(ccmr_alpha),
            "ccmr_search": str(ccmr_search_sig),
            "ri": float(ri_samples_aligned[occurrence_index]),
            "mari": float(mari_samples_aligned[occurrence_index]),
            "ri_defined": bool(ri_defined_mask[occurrence_index]),
            "mari_defined": bool(mari_defined_mask[occurrence_index]),
            "ri_undefined_type": int(ri_undefined_types[occurrence_index]),
            "mari_undefined_type": int(mari_undefined_types[occurrence_index]),
        }
        for m_pos, m in enumerate(ccmr_m_values):
            record[f"ccmr_m{int(m)}"] = float(
                ccmr_samples_aligned_by_m[occurrence_index, m_pos]
            )
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
    prepared_subsets: list | None = None,
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

    if prepared_subsets is not None:
        return RI._knn_balanced_accuracy_by_k_from_prepared_subsets(
            prepared_subsets=prepared_subsets,
            target=str(target_column),
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
        raise RuntimeError(
            f"{warn_context}: no evaluable paired_2x2 subsets remain for kNN curves"
        )

    out: dict[int, float] = {}
    for k in k_values:
        vals = [scores[int(k)] for scores in subset_scores if int(k) in scores]
        if vals:
            out[int(k)] = float(np.mean(vals))
    return out


def _curve_payload(values: dict[int, float]) -> dict:
    return {
        "values": {
            str(int(k)): float(v)
            for k, v in sorted(values.items(), key=lambda kv: int(kv[0]))
        }
    }


def _curve_from_payload(
    payload: dict, *, expected_k_values: list[int]
) -> dict[int, float] | None:
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
            "ss_dominated_undefined_frac": float(
                payload.get("ss_dominated_undefined_frac", 0.0)
            ),
            "oo_dominated_undefined_frac": float(
                payload.get("oo_dominated_undefined_frac", 0.0)
            ),
            "mixed_undefined_frac": float(payload.get("mixed_undefined_frac", 0.0)),
            "evaluation_design": str(payload.get("evaluation_design", "paired_2x2")),
            "evaluation_unit": str(payload.get("evaluation_unit", "occurrence")),
        }
        return result
    except Exception:  # noqa: BLE001
        return None


def _ccmr_result_to_payload(result: CCMRResult, m: int) -> dict:
    return {
        "m": int(m),
        "ccmr": float(result.value),
        "ccmr_std": float(result.std),
        "ccmr_undefined_frac": float(result.undefined_frac),
        "ccmr_k_start": int(result.k_start),
        "ccmr_k_final": int(result.k_final),
        "ccmr_retries": int(result.retries),
        "ccmr_alpha": float(result.alpha),
        "ccmr_q_alpha": float(result.q_alpha),
        "ccmr_ltm_alpha": float(result.ltm_alpha),
    }


def _ccmr_payload_from_results(results: dict[int, CCMRResult]) -> dict:
    return {
        "by_m": {
            str(int(m)): _ccmr_result_to_payload(result=res, m=int(m))
            for m, res in sorted(results.items(), key=lambda kv: int(kv[0]))
        }
    }


def _ccmr_payload_to_by_m(
    payload: dict, *, expected_m_values: list[int]
) -> dict[int, dict] | None:
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


def _compute_ccmr_by_m(
    *,
    features: np.ndarray,
    manifest: pd.DataFrame,
    confounder_column: str,
    evaluation_design: str,
    m_values: list[int],
    ccmr_start_k: int,
    ccmr_k_growth_factor: float,
    ccmr_alpha: float,
) -> dict[int, CCMRResult]:
    return cast(
        dict[int, CCMRResult],
        CCMR.compute(
            features=features,
            manifest=manifest,
            confounder_column=confounder_column,
            evaluation_design=evaluation_design,
            m=[int(m) for m in m_values],
            alpha=float(ccmr_alpha),
            start_k=int(ccmr_start_k),
            k_growth_factor=float(ccmr_k_growth_factor),
        ),
    )


def main() -> int:
    args = _parse_args()
    progress_enabled = resolve_progress_mode(str(args.progress))
    if int(args.ccmr_start_k) < 1:
        raise ValueError("--ccmr-start-k must be >= 1")
    if float(args.ccmr_k_growth_factor) <= 1.0:
        raise ValueError("--ccmr-k-growth-factor must be > 1")
    if float(args.ccmr_alpha) <= 0.0 or float(args.ccmr_alpha) > 1.0:
        raise ValueError("--ccmr-alpha must be in (0, 1]")
    if int(args.ccmr_m_max) < 1:
        raise ValueError("--ccmr-m-max must be >= 1")

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
    ccmr_m_sweep_csv = results_dir / "ccmr_m_sweep_metrics.csv"
    ccmr_m_sweep_json = results_dir / "ccmr_m_sweep_metrics.json"
    per_sample_csv = _per_sample_metrics_path(results_dir)
    per_sample_json = _per_sample_metrics_json_path(results_dir)

    cache = MetricsArtifactCache(results_dir=results_dir)

    k_max = int(args.k_max)
    k_values = _resolve_sweep_k_values(k_max)
    ccmr_m_values = list(range(1, int(args.ccmr_m_max) + 1))
    k_values_sig = k_values_signature(k_values)
    ccmr_search_sig = ccmr_search_signature(
        start_k=int(args.ccmr_start_k),
        k_growth_factor=float(args.ccmr_k_growth_factor),
        alpha=float(args.ccmr_alpha),
    )
    dataset_name = str(args.manifest.stem)

    extraction_status: dict[str, str] = {}
    metrics_status: dict[str, str] = {}
    failures: list[str] = []
    evaluation_design = str(args.evaluation_design)
    rows: list[dict] = []
    k_sweep_rows: list[dict] = []
    ccmr_m_sweep_rows: list[dict] = []
    per_sample_writer = StreamingMetricsWriter(
        csv_path=per_sample_csv, json_path=per_sample_json
    )

    progress_write(f"[benchmark] manifest={args.manifest}", enabled=progress_enabled)
    progress_write(f"[benchmark] models={', '.join(models)}", enabled=progress_enabled)
    progress_write(f"[benchmark] output_dir={output_dir}", enabled=progress_enabled)
    progress_write(f"[benchmark] dataset_dir={dataset_dir}", enabled=progress_enabled)
    progress_write(
        f"[benchmark] evaluation_design={evaluation_design}", enabled=progress_enabled
    )
    confounder_column = str(args.confounder_column)
    confounder_display_name = infer_confounder_display_name(confounder_column)
    manifest_df = load_manifest(str(args.manifest), confounder_column=confounder_column)
    base_manifest_fingerprint = manifest_fingerprint(manifest_df)
    eval_manifest = _prepare_eval_manifest(
        manifest_df=manifest_df,
        dataset_name=dataset_name,
        evaluation_design=evaluation_design,
    )
    aligned_manifest = _build_aligned_manifest(
        eval_manifest=eval_manifest,
        evaluation_design=evaluation_design,
    )
    embedding_manifest, embedding_keep_indices = build_embedding_source_manifest(
        eval_manifest
    )
    embedding_manifest_path = _embedding_manifest_path(dataset_dir)
    embedding_manifest.to_csv(embedding_manifest_path, index=False)
    embedding_manifest_fingerprint = manifest_fingerprint(embedding_manifest)

    # --- Pre-pass: collect per-model best k to determine dataset-wide median k ---
    dataset_median_k: int | None = None
    if args.use_median_k and not (args.prune_ss_oo or args.summarize_by_mean):
        per_model_best_k: list[int] = []
        for model in models:
            try:
                output_path = ee._output_path_in_dir(args.manifest, embeddings_dir, model)
                if not output_path.exists():
                    continue
                embedding_fp = embedding_fingerprint(output_path)
                input_fp_pre = {
                    "manifest_fingerprint": base_manifest_fingerprint,
                    "embedding_fingerprint": embedding_fp,
                }
                knn_bio_key = build_cache_key(
                    artifact_name="knn_bio_curve",
                    model=model,
                    input_fingerprint=input_fp_pre,
                    params={
                        "evaluation_design": evaluation_design,
                        "k_values": [int(k) for k in k_values],
                        "confounder_column": confounder_column,
                    },
                )
                knn_bacc_pre = _curve_from_payload(
                    cache.get_json(key=knn_bio_key), expected_k_values=k_values
                )
                if knn_bacc_pre is None:
                    features_pre = np.load(output_path)[embedding_keep_indices]
                    features_pre_norm = features_pre / (
                        np.linalg.norm(features_pre, axis=1, keepdims=True) + 1e-12
                    )
                    prepared_subsets_pre = (
                        RI._prepare_paired_subset_neighbor_cache(
                            features=features_pre_norm,
                            subsets=resolve_manifest_subsets(eval_manifest),
                            k_values=k_values,
                            prune_ss_oo=bool(args.prune_ss_oo),
                        )
                        if evaluation_design == "paired_2x2"
                        else None
                    )
                    knn_bacc_pre = _knn_balanced_accuracy_by_k_for_design(
                        features=features_pre_norm,
                        manifest=eval_manifest,
                        target_column="label",
                        k_values=k_values,
                        evaluation_design=evaluation_design,
                        warn_context=f"{dataset_name} k-curve (median-k pre-pass)",
                        prepared_subsets=prepared_subsets_pre,
                    )
                    cache.put_json(key=knn_bio_key, payload=_curve_payload(knn_bacc_pre))
                per_model_best_k.append(
                    _select_k_from_balanced_accuracy(k_values=k_values, scores=knn_bacc_pre)
                )
            except Exception:  # noqa: BLE001
                pass
        if per_model_best_k:
            dataset_median_k = int(np.median(per_model_best_k))
            progress_write(
                f"[benchmark] use-median-k: per-model optimal k = {per_model_best_k}",
                enabled=progress_enabled,
            )
            progress_write(
                f"[benchmark] use-median-k: dataset median k = {dataset_median_k}",
                enabled=progress_enabled,
            )

    for i, model in enumerate(models):
        with model_block(model, i + 1, len(models), enabled=progress_enabled) as ticker:
            output_path = ee._output_path_in_dir(args.manifest, embeddings_dir, model)
            spec = registry.get(model)
            ticker.start("embed")
            if not args.force_embed and _embedding_cache_matches_expected(
                output_path,
                expected_n_samples=len(embedding_manifest),
                expected_manifest_fingerprint=embedding_manifest_fingerprint,
                expected_manifest_path=embedding_manifest_path,
            ):
                ticker.log(f"[benchmark] embedding cache hit -> {output_path}")
                extraction_status[model] = "skipped"
                ticker.done("embed", cached=True)
            else:
                if spec is None:
                    extraction_status[model] = "failed"
                    metrics_status[model] = "failed"
                    if bool(args.force_embed):
                        msg = (
                            f"{model}: --force-embed requested but the model is not "
                            "registered for extraction"
                        )
                    else:
                        msg = (
                            f"{model}: no compatible cached embeddings found at "
                            f"{output_path} and the model is not registered for "
                            "extraction"
                        )
                    failures.append(msg)
                    ticker.log(f"[benchmark] {msg}")
                    continue
                try:
                    ee.embed_manifest(
                        manifest_path=embedding_manifest_path,
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
                }

                k_values_param = [int(k) for k in k_values]
                tau_value = float(args.tau)
                prune_ss_oo_value = bool(args.prune_ss_oo)
                summarize_by_mean_value = bool(args.summarize_by_mean)
                use_median_k_value = bool(args.use_median_k)

                keys = {
                    "knn_bio_curve": build_cache_key(
                        artifact_name="knn_bio_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "confounder_column": confounder_column,
                        },
                    ),
                    "knn_confounder_curve": build_cache_key(
                        artifact_name="knn_confounder_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "confounder_column": confounder_column,
                        },
                    ),
                    "ri_curve": build_cache_key(
                        artifact_name="ri_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                        },
                    ),
                    "mari_curve": build_cache_key(
                        artifact_name="mari_curve",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "tau": tau_value,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                        },
                    ),
                    "ri_summary": build_cache_key(
                        artifact_name="ri_summary",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "ri_samples": build_cache_key(
                        artifact_name="ri_samples",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "ri_samples_aligned": build_cache_key(
                        artifact_name="ri_samples_aligned",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "ri_undefined_types": build_cache_key(
                        artifact_name="ri_undefined_types",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "mari_summary": build_cache_key(
                        artifact_name="mari_summary",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "tau": tau_value,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "mari_samples": build_cache_key(
                        artifact_name="mari_samples",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "tau": tau_value,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "mari_samples_aligned": build_cache_key(
                        artifact_name="mari_samples_aligned",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "tau": tau_value,
                            "confounder_column": confounder_column,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "mari_undefined_types": build_cache_key(
                        artifact_name="mari_undefined_types",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "k_values": k_values_param,
                            "tau": tau_value,
                            "prune_ss_oo": prune_ss_oo_value,
                            "summarize_by_mean": summarize_by_mean_value,
                            "use_median_k": use_median_k_value,
                        },
                    ),
                    "ccmr_m_sweep": build_cache_key(
                        artifact_name="ccmr_m_sweep",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "m_max": int(args.ccmr_m_max),
                            "start_k": int(args.ccmr_start_k),
                            "k_growth_factor": float(args.ccmr_k_growth_factor),
                            "alpha": float(args.ccmr_alpha),
                            "confounder_column": confounder_column,
                        },
                    ),
                    "ccmr_m1_samples": build_cache_key(
                        artifact_name="ccmr_m1_samples",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "m_max": int(args.ccmr_m_max),
                            "start_k": int(args.ccmr_start_k),
                            "k_growth_factor": float(args.ccmr_k_growth_factor),
                            "alpha": float(args.ccmr_alpha),
                            "confounder_column": confounder_column,
                        },
                    ),
                    "ccmr_samples_aligned_by_m": build_cache_key(
                        artifact_name="ccmr_samples_aligned_by_m",
                        model=model,
                        input_fingerprint=input_fp,
                        params={
                            "evaluation_design": evaluation_design,
                            "m_max": int(args.ccmr_m_max),
                            "start_k": int(args.ccmr_start_k),
                            "k_growth_factor": float(args.ccmr_k_growth_factor),
                            "alpha": float(args.ccmr_alpha),
                            "confounder_column": confounder_column,
                        },
                    ),
                }

                knn_bacc_by_k: dict[int, float] | None = None
                knn_confounder_bacc_by_k: dict[int, float] | None = None
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
                ccmr_by_m: dict[int, dict] | None = None
                ccmr_samples: np.ndarray | None = None
                ccmr_samples_aligned_by_m: np.ndarray | None = None

                all_cache_hit = not bool(args.recompute_metrics)

                if not args.recompute_metrics:
                    knn_bacc_by_k = _curve_from_payload(
                        cache.get_json(key=keys["knn_bio_curve"]),
                        expected_k_values=k_values,
                    )
                    if knn_bacc_by_k is None:
                        all_cache_hit = False

                    knn_confounder_bacc_by_k = _curve_from_payload(
                        cache.get_json(key=keys["knn_confounder_curve"]),
                        expected_k_values=k_values,
                    )
                    if knn_confounder_bacc_by_k is None:
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

                    ri_summary = _summary_from_payload(
                        cache.get_json(key=keys["ri_summary"])
                    )
                    ri_samples = cache.get_npy(key=keys["ri_samples"])
                    ri_samples_aligned = cache.get_npy(key=keys["ri_samples_aligned"])
                    ri_undefined_types = cache.get_npy(key=keys["ri_undefined_types"])
                    if (
                        ri_summary is None
                        or ri_samples is None
                        or not _npy_matches_shape(
                            ri_samples_aligned, (len(aligned_manifest),)
                        )
                        or not _npy_matches_shape(
                            ri_undefined_types, (len(aligned_manifest),)
                        )
                        or str(ri_summary["evaluation_design"]) != evaluation_design
                        or str(ri_summary["evaluation_unit"])
                        not in {"sample", "occurrence"}
                    ):
                        all_cache_hit = False

                    mari_summary = _summary_from_payload(
                        cache.get_json(key=keys["mari_summary"])
                    )
                    mari_samples = cache.get_npy(key=keys["mari_samples"])
                    mari_samples_aligned = cache.get_npy(
                        key=keys["mari_samples_aligned"]
                    )
                    mari_undefined_types = cache.get_npy(
                        key=keys["mari_undefined_types"]
                    )
                    if (
                        mari_summary is None
                        or mari_samples is None
                        or not _npy_matches_shape(
                            mari_samples_aligned, (len(aligned_manifest),)
                        )
                        or not _npy_matches_shape(
                            mari_undefined_types, (len(aligned_manifest),)
                        )
                        or str(mari_summary["evaluation_design"]) != evaluation_design
                        or str(mari_summary["evaluation_unit"])
                        not in {"sample", "occurrence"}
                    ):
                        all_cache_hit = False

                    ccmr_by_m = _ccmr_payload_to_by_m(
                        cache.get_json(key=keys["ccmr_m_sweep"]),
                        expected_m_values=ccmr_m_values,
                    )
                    ccmr_samples = cache.get_npy(key=keys["ccmr_m1_samples"])
                    ccmr_samples_aligned_by_m = cache.get_npy(
                        key=keys["ccmr_samples_aligned_by_m"]
                    )
                    if (
                        ccmr_by_m is None
                        or ccmr_samples is None
                        or not _npy_matches_shape(
                            ccmr_samples_aligned_by_m,
                            (len(aligned_manifest), len(ccmr_m_values)),
                        )
                    ):
                        all_cache_hit = False
                else:
                    all_cache_hit = False

                features_full: np.ndarray | None = None
                eval_features: np.ndarray | None = None
                eval_features_norm: np.ndarray | None = None
                paired_subset_cache = None

                def _ensure_eval_features() -> np.ndarray:
                    nonlocal features_full, eval_features
                    if eval_features is None:
                        if features_full is None:
                            features_full = np.load(output_path)
                        eval_features = features_full[embedding_keep_indices]
                    return eval_features

                def _ensure_eval_features_norm() -> np.ndarray:
                    nonlocal eval_features_norm
                    if eval_features_norm is None:
                        arr = _ensure_eval_features()
                        eval_features_norm = arr / (
                            np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                        )
                    return eval_features_norm

                def _ensure_paired_subset_cache():
                    nonlocal paired_subset_cache
                    if evaluation_design != "paired_2x2":
                        raise RuntimeError(
                            "paired subset cache is only available for paired_2x2 evaluation"
                        )
                    if paired_subset_cache is None:
                        paired_subset_cache = RI._prepare_paired_subset_neighbor_cache(
                            features=_ensure_eval_features_norm(),
                            subsets=resolve_manifest_subsets(eval_manifest),
                            k_values=k_values,
                            prune_ss_oo=bool(args.prune_ss_oo),
                        )
                    return paired_subset_cache

                knn_was_cached = (
                    knn_bacc_by_k is not None and knn_confounder_bacc_by_k is not None
                )
                ticker.start("knn")
                if knn_bacc_by_k is None:
                    knn_bacc_by_k = _knn_balanced_accuracy_by_k_for_design(
                        features=_ensure_eval_features_norm(),
                        manifest=eval_manifest,
                        target_column="label",
                        k_values=k_values,
                        evaluation_design=evaluation_design,
                        warn_context=f"{dataset_name} k-curve",
                        prepared_subsets=(
                            _ensure_paired_subset_cache()
                            if evaluation_design == "paired_2x2"
                            else None
                        ),
                    )
                    cache.put_json(
                        key=keys["knn_bio_curve"], payload=_curve_payload(knn_bacc_by_k)
                    )

                if knn_confounder_bacc_by_k is None:
                    knn_confounder_bacc_by_k = _knn_balanced_accuracy_by_k_for_design(
                        features=_ensure_eval_features_norm(),
                        manifest=eval_manifest,
                        target_column="confounder",
                        k_values=k_values,
                        evaluation_design=evaluation_design,
                        warn_context=f"{dataset_name} confounder-k-curve",
                        prepared_subsets=(
                            _ensure_paired_subset_cache()
                            if evaluation_design == "paired_2x2"
                            else None
                        ),
                    )
                    cache.put_json(
                        key=keys["knn_confounder_curve"],
                        payload=_curve_payload(knn_confounder_bacc_by_k),
                    )

                selected_k = (
                    max(k_values)
                    if (args.prune_ss_oo or args.summarize_by_mean)
                    else dataset_median_k
                    if args.use_median_k and dataset_median_k is not None
                    else _select_k_from_balanced_accuracy(
                        k_values=k_values,
                        scores=knn_bacc_by_k,
                    )
                )
                selected_k_confounder = _select_k_from_balanced_accuracy(
                    k_values=k_values,
                    scores=knn_confounder_bacc_by_k,
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
                if (
                    ri_curve is None
                    or ri_summary is None
                    or ri_samples is None
                    or ri_samples_aligned is None
                    or ri_undefined_types is None
                ):
                    if evaluation_design == "paired_2x2":
                        ri_artifacts = RI._compute_artifacts_from_prepared_subsets(
                            prepared_subsets=_ensure_paired_subset_cache(),
                            dataset_name=dataset_name,
                            k_values=k_values,
                            evaluation_design=evaluation_design,
                            selected_k=int(selected_k),
                            include_selected_result=True,
                            warn_selected_result=True,
                        )
                    else:
                        ri_artifacts = RI._compute_artifacts(
                            features=_ensure_eval_features_norm(),
                            manifest=eval_manifest,
                            confounder_column=confounder_column,
                            k_values=k_values,
                            evaluation_design=evaluation_design,
                            selected_k=int(selected_k),
                            include_selected_result=True,
                            warn_selected_result=True,
                            prune_ss_oo=bool(args.prune_ss_oo),
                            summarize_by_mean=bool(args.summarize_by_mean),
                        )
                    ri_curve = dict(ri_artifacts.curve)
                    if ri_artifacts.result is None:
                        raise RuntimeError(
                            "RI shared scoring did not return a selected-k result"
                        )
                    ri = ri_artifacts.result
                    if int(ri.k) != int(selected_k):
                        raise RuntimeError(
                            f"Inconsistent selected k: RI returned {ri.k} but kNN balanced accuracy selected {selected_k}"
                        )
                    ri_summary = {
                        "k": int(ri.k),
                        "value": float(ri.value),
                        "std": float(ri.std),
                        "undefined_frac": float(ri.undefined_frac),
                        "ss_dominated_undefined_frac": float(
                            ri.ss_dominated_undefined_frac
                        ),
                        "oo_dominated_undefined_frac": float(
                            ri.oo_dominated_undefined_frac
                        ),
                        "mixed_undefined_frac": float(ri.mixed_undefined_frac),
                        "evaluation_design": str(ri.evaluation_design),
                        "evaluation_unit": str(ri.evaluation_unit),
                    }
                    ri_samples = np.asarray(ri.sample_values, dtype=float)
                    ri_samples_aligned = np.asarray(
                        ri.sample_values_aligned, dtype=float
                    )
                    ri_undefined_types = np.asarray(
                        ri.sample_undefined_types, dtype=int
                    )
                    cache.put_json(
                        key=keys["ri_curve"], payload=_curve_payload(ri_curve)
                    )
                    cache.put_json(key=keys["ri_summary"], payload=ri_summary)
                    cache.put_npy(key=keys["ri_samples"], values=ri_samples)
                    cache.put_npy(
                        key=keys["ri_samples_aligned"], values=ri_samples_aligned
                    )
                    cache.put_npy(
                        key=keys["ri_undefined_types"], values=ri_undefined_types
                    )
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
                if (
                    mari_curve is None
                    or mari_summary is None
                    or mari_samples is None
                    or mari_samples_aligned is None
                    or mari_undefined_types is None
                ):
                    if evaluation_design == "paired_2x2":
                        mari_artifacts = MaRI._compute_artifacts_from_prepared_subsets(
                            prepared_subsets=_ensure_paired_subset_cache(),
                            dataset_name=dataset_name,
                            k_values=k_values,
                            evaluation_design=evaluation_design,
                            selected_k=int(selected_k),
                            include_selected_result=True,
                            warn_selected_result=True,
                            tau=float(args.tau),
                        )
                    else:
                        mari_artifacts = MaRI._compute_artifacts(
                            features=_ensure_eval_features_norm(),
                            manifest=eval_manifest,
                            confounder_column=confounder_column,
                            k_values=k_values,
                            evaluation_design=evaluation_design,
                            selected_k=int(selected_k),
                            include_selected_result=True,
                            warn_selected_result=True,
                            prune_ss_oo=bool(args.prune_ss_oo),
                            summarize_by_mean=bool(args.summarize_by_mean),
                            tau=float(args.tau),
                        )
                    mari_curve = dict(mari_artifacts.curve)
                    if mari_artifacts.result is None:
                        raise RuntimeError(
                            "MaRI shared scoring did not return a selected-k result"
                        )
                    mari = mari_artifacts.result
                    if int(mari.k) != int(selected_k):
                        raise RuntimeError(
                            f"Inconsistent selected k: MaRI returned {mari.k} but kNN balanced accuracy selected {selected_k}"
                        )
                    mari_summary = {
                        "k": int(mari.k),
                        "value": float(mari.value),
                        "std": float(mari.std),
                        "undefined_frac": float(mari.undefined_frac),
                        "ss_dominated_undefined_frac": float(
                            mari.ss_dominated_undefined_frac
                        ),
                        "oo_dominated_undefined_frac": float(
                            mari.oo_dominated_undefined_frac
                        ),
                        "mixed_undefined_frac": float(mari.mixed_undefined_frac),
                        "evaluation_design": str(mari.evaluation_design),
                        "evaluation_unit": str(mari.evaluation_unit),
                    }
                    mari_samples = np.asarray(mari.sample_values, dtype=float)
                    mari_samples_aligned = np.asarray(
                        mari.sample_values_aligned, dtype=float
                    )
                    mari_undefined_types = np.asarray(
                        mari.sample_undefined_types, dtype=int
                    )
                    cache.put_json(
                        key=keys["mari_curve"], payload=_curve_payload(mari_curve)
                    )
                    cache.put_json(key=keys["mari_summary"], payload=mari_summary)
                    cache.put_npy(key=keys["mari_samples"], values=mari_samples)
                    cache.put_npy(
                        key=keys["mari_samples_aligned"], values=mari_samples_aligned
                    )
                    cache.put_npy(
                        key=keys["mari_undefined_types"], values=mari_undefined_types
                    )
                else:
                    mari_samples = np.asarray(mari_samples, dtype=float)
                    if mari_samples_aligned is not None:
                        mari_samples_aligned = np.asarray(
                            mari_samples_aligned, dtype=float
                        )
                    if mari_undefined_types is not None:
                        mari_undefined_types = np.asarray(
                            mari_undefined_types, dtype=int
                        )
                ticker.done("MaRI", cached=mari_was_cached)

                ccmr_was_cached = (
                    ccmr_by_m is not None
                    and ccmr_samples is not None
                    and ccmr_samples_aligned_by_m is not None
                )
                ticker.start("CCMR")
                if (
                    ccmr_by_m is None
                    or ccmr_samples is None
                    or ccmr_samples_aligned_by_m is None
                ):
                    ccmr_results = _compute_ccmr_by_m(
                        features=_ensure_eval_features(),
                        manifest=eval_manifest,
                        confounder_column=confounder_column,
                        evaluation_design=evaluation_design,
                        m_values=ccmr_m_values,
                        ccmr_start_k=int(args.ccmr_start_k),
                        ccmr_k_growth_factor=float(args.ccmr_k_growth_factor),
                        ccmr_alpha=float(args.ccmr_alpha),
                    )
                    ccmr_by_m = _ccmr_payload_to_by_m(
                        _ccmr_payload_from_results(ccmr_results),
                        expected_m_values=ccmr_m_values,
                    )
                    if ccmr_by_m is None:
                        raise RuntimeError(
                            "Failed to serialize ccmr m-sweep cache payload"
                        )
                    ccmr_samples = np.asarray(
                        ccmr_results[1].sample_values, dtype=float
                    )
                    ccmr_samples_aligned_by_m = np.column_stack(
                        [
                            np.asarray(
                                ccmr_results[int(m)].sample_values_aligned, dtype=float
                            )
                            for m in ccmr_m_values
                        ]
                    )
                    cache.put_json(
                        key=keys["ccmr_m_sweep"],
                        payload={"by_m": {str(k): v for k, v in ccmr_by_m.items()}},
                    )
                    cache.put_npy(key=keys["ccmr_m1_samples"], values=ccmr_samples)
                    cache.put_npy(
                        key=keys["ccmr_samples_aligned_by_m"],
                        values=ccmr_samples_aligned_by_m,
                    )
                else:
                    ccmr_samples = np.asarray(ccmr_samples, dtype=float)
                    if ccmr_samples_aligned_by_m is not None:
                        ccmr_samples_aligned_by_m = np.asarray(
                            ccmr_samples_aligned_by_m, dtype=float
                        )
                ticker.done("CCMR", cached=ccmr_was_cached)

                evaluation_unit = str(ri_summary["evaluation_unit"])
                ccmr_m_rows_for_model: list[dict] = []
                for m in ccmr_m_values:
                    payload = ccmr_by_m[int(m)]
                    ccmr_m_rows_for_model.append(
                        {
                            "dataset": dataset_name,
                            "model": str(model),
                            "confounder_column": confounder_column,
                            "confounder_display_name": confounder_display_name,
                            "evaluation_design": evaluation_design,
                            "evaluation_unit": evaluation_unit,
                            "tau": float(args.tau),
                            "k_max": int(k_max),
                            "k_values": str(k_values_sig),
                            "ccmr_search": str(ccmr_search_sig),
                            "m": int(payload["m"]),
                            "ccmr": float(payload["ccmr"]),
                            "ccmr_std": float(payload["ccmr_std"]),
                            "ccmr_undefined_frac": float(
                                payload["ccmr_undefined_frac"]
                            ),
                            "ccmr_k_start": int(payload["ccmr_k_start"]),
                            "ccmr_k_final": int(payload["ccmr_k_final"]),
                            "ccmr_retries": int(payload["ccmr_retries"]),
                            "ccmr_alpha": float(payload["ccmr_alpha"]),
                            "ccmr_q_alpha": float(payload["ccmr_q_alpha"]),
                            "ccmr_ltm_alpha": float(payload["ccmr_ltm_alpha"]),
                            "embedding_path": str(output_path),
                        }
                    )

                m_sorted = sorted(ccmr_m_values)
                ccmr_curve = [float(ccmr_by_m[m]["ccmr"]) for m in m_sorted]
                finite_curve = [c for c in ccmr_curve if np.isfinite(c)]
                if len(m_sorted) > 1:
                    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
                    ccmr_auc = float(
                        _trapz(ccmr_curve, m_sorted) / (m_sorted[-1] - m_sorted[0])
                    )
                else:
                    ccmr_auc = ccmr_curve[0] if ccmr_curve else float("nan")
                ccmr_min_val = (
                    float(min(finite_curve)) if finite_curve else float("nan")
                )
                ccmr_delta = (
                    float(ccmr_curve[-1] - ccmr_curve[0])
                    if len(ccmr_curve) > 1
                    else 0.0
                )

                total_n = int(len(aligned_manifest))
                ri_undefined_n = int(np.count_nonzero(~np.isfinite(ri_samples_aligned)))
                mari_undefined_n = int(
                    np.count_nonzero(~np.isfinite(mari_samples_aligned))
                )
                ri_ss_frac = float(ri_summary.get("ss_dominated_undefined_frac", 0.0))
                ri_oo_frac = float(ri_summary.get("oo_dominated_undefined_frac", 0.0))
                ri_mixed_frac = float(ri_summary.get("mixed_undefined_frac", 0.0))
                mari_ss_frac = float(
                    mari_summary.get("ss_dominated_undefined_frac", 0.0)
                )
                mari_oo_frac = float(
                    mari_summary.get("oo_dominated_undefined_frac", 0.0)
                )
                mari_mixed_frac = float(mari_summary.get("mixed_undefined_frac", 0.0))
                saved_dist_path = _save_mari_sample_distribution(
                    results_dir=results_dir,
                    model=model,
                    dataset=dataset_name,
                    evaluation_design=evaluation_design,
                    evaluation_unit=evaluation_unit,
                    tau=float(args.tau),
                    selected_k=int(selected_k),
                    n_total_units=total_n,
                    n_undefined_units=mari_undefined_n,
                    values=mari_samples,
                    median_value=float(mari_summary.get("median_value", float("nan"))),
                    q_alpha=float(mari_summary.get("q_alpha", float("nan"))),
                    ltm_alpha=float(mari_summary.get("ltm_alpha", float("nan"))),
                )
                saved_ri_dist_path = _save_ri_sample_distribution(
                    results_dir=results_dir,
                    model=model,
                    dataset=dataset_name,
                    evaluation_design=evaluation_design,
                    evaluation_unit=evaluation_unit,
                    selected_k=int(selected_k),
                    n_total_units=total_n,
                    n_undefined_units=ri_undefined_n,
                    values=ri_samples,
                    median_value=float(ri_summary.get("median_value", float("nan"))),
                    q_alpha=float(ri_summary.get("q_alpha", float("nan"))),
                    ltm_alpha=float(ri_summary.get("ltm_alpha", float("nan"))),
                )

                ccmr_result = ccmr_by_m[1]
                ccmr_dist_path = _distribution_path(results_dir, "ccmr", model)
                ccmr_dist_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(ccmr_dist_path, ccmr_samples)

                row = {
                    "dataset": dataset_name,
                    "model": model,
                    "confounder_column": confounder_column,
                    "confounder_display_name": confounder_display_name,
                    "k": int(ri_summary["k"]),
                    "k_max": int(k_max),
                    "evaluation_design": evaluation_design,
                    "evaluation_unit": evaluation_unit,
                    "tau": float(args.tau),
                    "k_values": k_values_sig,
                    "ccmr_search": ccmr_search_sig,
                    "bio_knn_bacc": float(knn_bacc_by_k[int(selected_k)]),
                    "confounder_knn_bacc": float(
                        knn_confounder_bacc_by_k[int(selected_k_confounder)]
                    ),
                    "selected_k_confounder": int(selected_k_confounder),
                    "ri": float(ri_summary["value"]),
                    "ri_std": float(ri_summary["std"]),
                    "ri_median": float(ri_summary.get("median_value", float("nan"))),
                    "ri_q_alpha": float(ri_summary.get("q_alpha", float("nan"))),
                    "ri_ltm_alpha": float(ri_summary.get("ltm_alpha", float("nan"))),
                    "mari": float(mari_summary["value"]),
                    "mari_std": float(mari_summary["std"]),
                    "mari_median": float(mari_summary.get("median_value", float("nan"))),
                    "mari_q_alpha": float(mari_summary.get("q_alpha", float("nan"))),
                    "mari_ltm_alpha": float(mari_summary.get("ltm_alpha", float("nan"))),
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
                    "ccmr": float(ccmr_result["ccmr"]),
                    "ccmr_std": float(ccmr_result["ccmr_std"]),
                    "ccmr_m": int(ccmr_result["m"]),
                    "ccmr_undefined_frac": float(ccmr_result["ccmr_undefined_frac"]),
                    "ccmr_k_start": int(ccmr_result["ccmr_k_start"]),
                    "ccmr_k_final": int(ccmr_result["ccmr_k_final"]),
                    "ccmr_retries": int(ccmr_result["ccmr_retries"]),
                    "ccmr_alpha": float(ccmr_result["ccmr_alpha"]),
                    "ccmr_q_alpha": float(ccmr_result["ccmr_q_alpha"]),
                    "ccmr_ltm_alpha": float(ccmr_result["ccmr_ltm_alpha"]),
                    "ccmr_auc": ccmr_auc,
                    "ccmr_min": ccmr_min_val,
                    "ccmr_delta": ccmr_delta,
                    "ccmr_samples_path": str(ccmr_dist_path),
                    "embedding_path": str(output_path),
                }
                rows.append(row)
                model_per_sample_rows = _build_per_sample_rows(
                    aligned_manifest=aligned_manifest,
                    dataset_name=dataset_name,
                    model=str(model),
                    confounder_column=confounder_column,
                    confounder_display_name=confounder_display_name,
                    evaluation_design=evaluation_design,
                    evaluation_unit=evaluation_unit,
                    selected_k=int(ri_summary["k"]),
                    tau=float(args.tau),
                    ccmr_alpha=float(args.ccmr_alpha),
                    ccmr_search_sig=str(ccmr_search_sig),
                    ri_samples_aligned=np.asarray(ri_samples_aligned, dtype=float),
                    mari_samples_aligned=np.asarray(mari_samples_aligned, dtype=float),
                    ri_defined_mask=np.isfinite(
                        np.asarray(ri_samples_aligned, dtype=float)
                    ),
                    mari_defined_mask=np.isfinite(
                        np.asarray(mari_samples_aligned, dtype=float)
                    ),
                    ri_undefined_types=np.asarray(ri_undefined_types, dtype=int),
                    mari_undefined_types=np.asarray(mari_undefined_types, dtype=int),
                    ccmr_samples_aligned_by_m=np.asarray(
                        ccmr_samples_aligned_by_m, dtype=float
                    ),
                    ccmr_m_values=ccmr_m_values,
                )
                model_per_sample_rows = sorted(
                    model_per_sample_rows, key=lambda row: int(row["occurrence_index"])
                )
                model_csv, model_json = _per_sample_metrics_by_model_paths(
                    results_dir, str(model)
                )
                save_metrics(
                    rows=model_per_sample_rows, csv_path=model_csv, json_path=model_json
                )
                per_sample_writer.write_rows(model_per_sample_rows)
                for k in k_values:
                    k_sweep_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "confounder_column": confounder_column,
                            "confounder_display_name": confounder_display_name,
                            "evaluation_design": evaluation_design,
                            "evaluation_unit": evaluation_unit,
                            "tau": float(args.tau),
                            "k_max": int(k_max),
                            "k_values": k_values_sig,
                            "ccmr_search": ccmr_search_sig,
                            "k": int(k),
                            "knn_bacc": float(knn_bacc_by_k[int(k)]),
                            "knn_confounder_bacc": float(
                                knn_confounder_bacc_by_k[int(k)]
                            ),
                            "ri": float(ri_curve[int(k)]),
                            "mari": float(mari_curve[int(k)]),
                            "selected_k": int(selected_k),
                            "selected_k_confounder": int(selected_k_confounder),
                            "embedding_path": str(output_path),
                        }
                    )
                ccmr_m_sweep_rows.extend(ccmr_m_rows_for_model)

                metrics_status[model] = "cached" if all_cache_hit else "ok"
                if all_cache_hit:
                    ticker.log("[benchmark] metrics cache hit")
                else:
                    ticker.log("[benchmark] metrics cache miss: partial/full recompute")
                ticker.log(
                    f"[benchmark] RI={row['ri']:.4f} MaRI={row['mari']:.4f} CCMR={row['ccmr']:.4f}"
                )
                undef_parts = []
                if row["ri_undefined_frac"] > 0.0:
                    undef_parts.append(f"RI={100*row['ri_undefined_frac']:.1f}%")
                if row["mari_undefined_frac"] > 0.0:
                    undef_parts.append(f"MaRI={100*row['mari_undefined_frac']:.1f}%")
                if row["ccmr_undefined_frac"] > 0.0:
                    undef_parts.append(f"CCMR={100*row['ccmr_undefined_frac']:.1f}%")
                if undef_parts:
                    ticker.log(
                        f"[benchmark] undefined samples: {', '.join(undef_parts)}"
                    )
            except Exception as exc:  # noqa: BLE001
                metrics_status[model] = "failed"
                failures.append(f"{model}: metrics failed ({exc})")
                ticker.log(f"[benchmark] metrics failed: {exc}")

    per_sample_writer.close()

    if rows:
        save_metrics(rows=rows, csv_path=metrics_csv, json_path=metrics_json)
        save_metrics(rows=k_sweep_rows, csv_path=k_sweep_csv, json_path=k_sweep_json)
        save_metrics(
            rows=ccmr_m_sweep_rows,
            csv_path=ccmr_m_sweep_csv,
            json_path=ccmr_m_sweep_json,
        )
        plot_knn_bio_k_sweep(
            rows=k_sweep_rows, out_path=plots_dir / "knn_bio_k_sweep.png"
        )
        plot_knn_confounder_k_sweep(
            rows=k_sweep_rows, out_path=plots_dir / "knn_confounder_k_sweep.png"
        )
        plot_ri_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "ri_k_sweep.png")
        plot_mari_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "mari_k_sweep.png")
        if args.summarize_by_mean:
            plot_ri_mari_cumulative_mean_k_sweep(
                rows=k_sweep_rows,
                out_path=plots_dir / "ri_mari_cumulative_mean_k_sweep.png",
            )
        plot_ccmr_m_sweep_with_ltm(
            rows=ccmr_m_sweep_rows, out_path=plots_dir / "ccmr_m_sweep.png"
        )
        plot_ccmr_ltm_comparison(
            rows=rows, out_path=plots_dir / "ccmr_ltm_comparison.png"
        )
        plot_bio_vs_confounder_scatter(
            rows=rows, out_path=plots_dir / "bio_vs_confounder_scatter.png"
        )
        plot_mari_vs_ri_scatter(
            rows=rows, out_path=plots_dir / "mari_vs_ri_scatter.png"
        )
        plot_ri_mari_support(rows=rows, out_path=plots_dir / "ri_mari_support.png")
        plot_ccmr_vs_mari_scatter(
            rows=rows, out_path=plots_dir / "ccmr_vs_mari_scatter.png"
        )
        plot_q_alpha_vs_ccmr_scatter(
            rows=rows, out_path=plots_dir / "q_alpha_vs_ccmr_scatter.png"
        )
        plot_ccmr_sample_distributions(
            rows=rows, out_path=plots_dir / "ccmr_sample_distributions.png"
        )
        if args.prune_ss_oo:
            plot_ri_mari_sample_distributions(
                rows=rows,
                metric="ri",
                out_path=plots_dir / "ri_sample_distributions.png",
            )
            plot_ri_mari_sample_distributions(
                rows=rows,
                metric="mari",
                out_path=plots_dir / "mari_sample_distributions.png",
            )

    progress_write("\n[benchmark] === summary ===", enabled=progress_enabled)
    for model in models:
        e = extraction_status.get(model, "n/a")
        m = metrics_status.get(model, "n/a")
        progress_write(
            f"[benchmark] {model}: extract={e} metrics={m}", enabled=progress_enabled
        )
    progress_write(f"[benchmark] metrics_csv={metrics_csv}", enabled=progress_enabled)
    progress_write(f"[benchmark] metrics_json={metrics_json}", enabled=progress_enabled)
    progress_write(f"[benchmark] k_sweep_csv={k_sweep_csv}", enabled=progress_enabled)
    progress_write(f"[benchmark] k_sweep_json={k_sweep_json}", enabled=progress_enabled)
    progress_write(
        f"[benchmark] ccmr_m_sweep_csv={ccmr_m_sweep_csv}", enabled=progress_enabled
    )
    progress_write(
        f"[benchmark] ccmr_m_sweep_json={ccmr_m_sweep_json}", enabled=progress_enabled
    )
    progress_write(f"[benchmark] plots_dir={plots_dir}", enabled=progress_enabled)

    if failures:
        progress_write("[benchmark] failures:", enabled=progress_enabled)
        for msg in failures:
            progress_write(f"[benchmark] - {msg}", enabled=progress_enabled)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
