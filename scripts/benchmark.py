import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extract_embeddings as ee
from common import parse_k_candidates
from mari import CCRR, MaRI, RI
from mari.metrics.neighbors import (
    _knn_balanced_accuracy_by_k,
    _normalize_k_values,
    _select_k_from_balanced_accuracy,
)
from mari.metrics.pairs import load_manifest, normalize_center_values
from metrics_io import (
    ccrr_search_signature,
    excluded_centers_signature,
    k_candidates_signature,
    load_cached_k_sweep_rows,
    load_cached_rows,
    save_k_sweep_metrics,
    save_metrics,
)
from plotting import (
    plot_benchmark_6panel_summary,
    plot_bio_vs_center_scatter,
    plot_ccrr_m_sweep,
    plot_ccrr_sample_distributions,
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


def _save_mari_sample_distribution(
    *,
    results_dir: Path,
    model: str,
    dataset: str,
    mode: str,
    tau: float,
    selected_k: int,
    n_total_samples: int,
    n_undefined_samples: int,
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
        "mode": str(mode),
        "tau": float(tau),
        "k": int(selected_k),
        "n_total_samples": int(n_total_samples),
        "n_undefined_samples": int(n_undefined_samples),
        "n_samples": int(arr.shape[0]),
        "distribution_path": str(out_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_path


def _save_ri_sample_distribution(
    *,
    results_dir: Path,
    model: str,
    dataset: str,
    mode: str,
    selected_k: int,
    n_total_samples: int,
    n_undefined_samples: int,
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
        "mode": str(mode),
        "k": int(selected_k),
        "n_total_samples": int(n_total_samples),
        "n_undefined_samples": int(n_undefined_samples),
        "n_samples": int(arr.shape[0]),
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
    parser.add_argument("--mode", default="global", choices=["paired", "global"], help="RI/MaRI mode.")
    parser.add_argument("--k-candidates", default="3,5,7,10,15,20,25", help="Comma-separated k candidates.")
    parser.add_argument(
        "--continuous-k-sweep-max",
        type=int,
        default=0,
        help="If > 0, sweep k continuously from 1..max instead of only --k-candidates.",
    )
    parser.add_argument("--tau", type=float, default=0.2, help="MaRI tau.")
    parser.add_argument(
        "--ccrr-m-candidates",
        default="1,2,3",
        help="Comma-separated CCRR m candidates for m-sweep plotting (must include 1).",
    )
    parser.add_argument(
        "--ccrr-acceptance-threshold",
        type=float,
        default=0.0,
        help="Stop CCRR search once undefined fraction is <= threshold (default 0.0).",
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
        default=1.5,
        help="Geometric growth factor for CCRR iterative k search (>1, default 1.5).",
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
    parser.add_argument("--force-embed", action="store_true", help="Force re-extraction of embeddings.")
    parser.add_argument(
        "--recompute-metrics",
        action="store_true",
        help="Force recomputation of metrics even if compatible cache exists.",
    )
    return parser.parse_args()


def _resolve_models(raw_models: str, registry: dict[str, ee.ModelSpec]) -> list[str]:
    if str(raw_models).strip():
        models = ee._parse_models(raw_models)
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


def _resolve_ccrr_m_values(raw_m_candidates: str) -> list[int]:
    m_values = _normalize_k_values(parse_k_candidates(raw_m_candidates))
    if 1 not in m_values:
        raise ValueError("--ccrr-m-candidates must include m=1")
    return m_values


_REQUIRED_SUMMARY_CACHE_KEYS = (
    "bio_knn_bacc",
    "center_knn_bacc",
    "selected_k_center",
    "ri_undefined_frac",
    "mari_undefined_frac",
    "ccrr_undefined_frac",
    "ccrr_acceptance_met",
    "ccrr_k_final",
)
_REQUIRED_K_SWEEP_CACHE_KEYS = (
    "knn_center_bacc",
    "selected_k_center",
)


def _has_required_cache_keys(row: dict, keys: tuple[str, ...]) -> bool:
    for key in keys:
        if key not in row:
            return False
        value = row[key]
        if pd.isna(value):
            return False
    return True


def _k_sweep_rows_have_required_cache_keys(rows: list[dict], keys: tuple[str, ...]) -> bool:
    if not rows:
        return False
    return all(_has_required_cache_keys(row=r, keys=keys) for r in rows)


def _compute_ccrr_m_sweep_rows(
    *,
    features: np.ndarray,
    manifest: pd.DataFrame,
    mode: str,
    m_values: list[int],
    dataset_name: str,
    model: str,
    tau: float,
    k_candidates_sig: str,
    excluded_centers_sig: str,
    ccrr_search_sig: str,
    embedding_path: Path,
    ccrr_acceptance_threshold: float,
    ccrr_start_k: int,
    ccrr_k_growth_factor: float,
) -> tuple[dict[int, object], list[dict]]:
    ccrr_by_m: dict[int, object] = {}
    rows: list[dict] = []
    for m in m_values:
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode=mode,
            m=int(m),
            acceptance_threshold=float(ccrr_acceptance_threshold),
            start_k=int(ccrr_start_k),
            k_growth_factor=float(ccrr_k_growth_factor),
        )
        ccrr_by_m[int(m)] = result
        rows.append(
            {
                "dataset": str(dataset_name),
                "model": str(model),
                "mode": str(mode),
                "tau": float(tau),
                "k_candidates": str(k_candidates_sig),
                "excluded_centers": str(excluded_centers_sig),
                "ccrr_search": str(ccrr_search_sig),
                "m": int(m),
                "ccrr": float(result.value),
                "ccrr_std": float(result.std),
                "ccrr_undefined_frac": float(result.undefined_frac),
                "ccrr_acceptance_threshold": float(result.acceptance_threshold),
                "ccrr_acceptance_met": bool(result.acceptance_met),
                "ccrr_k_start": int(result.k_start),
                "ccrr_k_final": int(result.k_final),
                "ccrr_retries": int(result.retries),
                "ccrr_alpha": float(result.alpha),
                "ccrr_q_alpha": float(result.q_alpha),
                "ccrr_ltm_alpha": float(result.ltm_alpha),
                "embedding_path": str(embedding_path),
            }
        )
    return ccrr_by_m, rows


def main() -> int:
    args = _parse_args()
    if float(args.ccrr_acceptance_threshold) < 0.0 or float(args.ccrr_acceptance_threshold) > 1.0:
        raise ValueError("--ccrr-acceptance-threshold must be in [0, 1]")
    if int(args.ccrr_start_k) < 1:
        raise ValueError("--ccrr-start-k must be >= 1")
    if float(args.ccrr_k_growth_factor) <= 1.0:
        raise ValueError("--ccrr-k-growth-factor must be > 1")

    registry = ee._build_model_registry()
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
    k_candidates = parse_k_candidates(args.k_candidates)
    k_values = _resolve_sweep_k_values(
        k_candidates=k_candidates,
        continuous_k_sweep_max=int(args.continuous_k_sweep_max),
    )
    ccrr_m_values = _resolve_ccrr_m_values(args.ccrr_m_candidates)
    k_candidates_sig = k_candidates_signature(k_values)
    excluded_centers = normalize_center_values(args.exclude_center)
    excluded_centers_sig = excluded_centers_signature(excluded_centers)
    ccrr_search_sig = ccrr_search_signature(
        acceptance_threshold=float(args.ccrr_acceptance_threshold),
        start_k=int(args.ccrr_start_k),
        k_growth_factor=float(args.ccrr_k_growth_factor),
    )

    cached_rows = {}
    cached_k_sweep_rows = {}
    if not args.recompute_metrics:
        cached_rows = load_cached_rows(
            metrics_csv=metrics_csv,
            models=models,
            mode=args.mode,
            tau=float(args.tau),
            k_candidates_sig=k_candidates_sig,
            excluded_centers_sig=excluded_centers_sig,
            ccrr_search_sig=ccrr_search_sig,
        )
        cached_k_sweep_rows = load_cached_k_sweep_rows(
            metrics_csv=k_sweep_csv,
            models=models,
            mode=args.mode,
            tau=float(args.tau),
            k_candidates_sig=k_candidates_sig,
            excluded_centers_sig=excluded_centers_sig,
            ccrr_search_sig=ccrr_search_sig,
            expected_k_values=k_values,
        )

    extraction_status: dict[str, str] = {}
    metrics_status: dict[str, str] = {}
    failures: list[str] = []
    rows: list[dict] = []
    k_sweep_rows: list[dict] = []
    ccrr_m_sweep_rows: list[dict] = []

    print(f"[benchmark] manifest={args.manifest}")
    print(f"[benchmark] models={', '.join(models)}")
    print(f"[benchmark] output_dir={output_dir}")
    print(f"[benchmark] dataset_dir={dataset_dir}")
    manifest_df = load_manifest(str(args.manifest), dataset_name=str(args.dataset_name))
    center_series = manifest_df["medical_center"].map(str).str.strip()
    keep_mask = ~center_series.isin(excluded_centers)
    if not bool(keep_mask.any()):
        excluded_txt = ", ".join(excluded_centers)
        raise ValueError(
            f"No samples remain after excluding centers [{excluded_txt}] from dataset '{args.dataset_name}'"
        )
    keep_indices = np.flatnonzero(keep_mask.to_numpy())
    eval_manifest = manifest_df.loc[keep_mask].reset_index(drop=True)
    labels = pd.factorize(eval_manifest["label"])[0].astype(int)
    center_labels = pd.factorize(eval_manifest["medical_center"])[0].astype(int)
    slide_ids = eval_manifest["slide_id"].astype(str).to_numpy()

    for model in models:
        output_path = ee._output_path_in_dir(args.manifest, embeddings_dir, model)
        spec = registry[model]

        print(f"\n[benchmark] === {model} ===")
        if output_path.exists() and not args.force_embed:
            print(f"[benchmark] embedding cache hit -> {output_path}")
            extraction_status[model] = "skipped"
        else:
            try:
                ee.embed_manifest(
                    manifest_path=args.manifest,
                    output_path=output_path,
                    spec=spec,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    device_arg=str(args.device),
                )
                extraction_status[model] = "ok"
            except Exception as exc:  # noqa: BLE001
                extraction_status[model] = "failed"
                metrics_status[model] = "failed"
                failures.append(f"{model}: extraction failed ({exc})")
                print(f"[benchmark] extraction failed: {exc}")
                continue

        try:
            mari_dist_path = _distribution_path(results_dir, "mari", model)
            ri_dist_path = _distribution_path(results_dir, "ri", model)
            features = np.load(output_path)
            eval_features = features[keep_indices]
            if model in cached_rows:
                if model in cached_k_sweep_rows and mari_dist_path.exists() and ri_dist_path.exists():
                    row = dict(cached_rows[model])
                    cached_k_rows = [dict(r) for r in cached_k_sweep_rows[model]]
                    summary_ok = _has_required_cache_keys(
                        row=row,
                        keys=_REQUIRED_SUMMARY_CACHE_KEYS,
                    )
                    k_rows_ok = _k_sweep_rows_have_required_cache_keys(
                        rows=cached_k_rows,
                        keys=_REQUIRED_K_SWEEP_CACHE_KEYS,
                    )
                    if summary_ok and k_rows_ok:
                        rows.append(row)
                        k_sweep_rows.extend(cached_k_rows)
                        _, model_ccrr_m_rows = _compute_ccrr_m_sweep_rows(
                            features=eval_features,
                            manifest=eval_manifest,
                            mode=str(args.mode),
                            m_values=ccrr_m_values,
                            dataset_name=str(args.dataset_name),
                            model=model,
                            tau=float(args.tau),
                            k_candidates_sig=k_candidates_sig,
                            excluded_centers_sig=excluded_centers_sig,
                            ccrr_search_sig=ccrr_search_sig,
                            embedding_path=output_path,
                            ccrr_acceptance_threshold=float(args.ccrr_acceptance_threshold),
                            ccrr_start_k=int(args.ccrr_start_k),
                            ccrr_k_growth_factor=float(args.ccrr_k_growth_factor),
                        )
                        ccrr_m_sweep_rows.extend(model_ccrr_m_rows)
                        metrics_status[model] = "cached"
                        print("[benchmark] metrics cache hit")
                        continue
                    print("[benchmark] cache miss (schema upgrade): recomputing metrics")

            knn_bacc_by_k = _knn_balanced_accuracy_by_k(
                features=eval_features,
                labels=labels,
                slide_ids=slide_ids,
                k_values=k_values,
                warn_context=f"{args.dataset_name} k-curve",
            )
            selected_k = _select_k_from_balanced_accuracy(
                k_values=k_values,
                scores=knn_bacc_by_k,
            )
            knn_center_bacc_by_k = _knn_balanced_accuracy_by_k(
                features=eval_features,
                labels=center_labels,
                slide_ids=slide_ids,
                k_values=k_values,
                warn_context=f"{args.dataset_name} center-k-curve",
            )
            selected_k_center = _select_k_from_balanced_accuracy(
                k_values=k_values,
                scores=knn_center_bacc_by_k,
            )
            ri_curve = RI.compute_curve(
                features=eval_features,
                manifest=eval_manifest,
                mode=args.mode,
                k_values=k_values,
            )
            mari_curve = MaRI.compute_curve(
                features=eval_features,
                manifest=eval_manifest,
                mode=args.mode,
                k_values=k_values,
                tau=float(args.tau),
            )

            ri = RI.compute(
                features=eval_features,
                manifest=eval_manifest,
                mode=args.mode,
                k_candidates=k_values,
            )
            mari = MaRI.compute(
                features=eval_features,
                manifest=eval_manifest,
                mode=args.mode,
                k_candidates=k_values,
                tau=float(args.tau),
            )
            if int(ri.k) != int(selected_k):
                raise RuntimeError(
                    f"Inconsistent selected k: RI returned {ri.k} but kNN balanced accuracy selected {selected_k}"
                )
            if int(mari.k) != int(selected_k):
                raise RuntimeError(
                    f"Inconsistent selected k: MaRI returned {mari.k} but kNN balanced accuracy selected {selected_k}"
                )
            total_n = int(len(eval_manifest))
            ri_informative_n = int(len(ri.sample_values))
            mari_informative_n = int(len(mari.sample_values))
            ri_undefined_n = max(0, total_n - ri_informative_n)
            mari_undefined_n = max(0, total_n - mari_informative_n)
            ri_undefined_frac = float(ri_undefined_n / total_n) if total_n > 0 else 0.0
            mari_undefined_frac = float(mari_undefined_n / total_n) if total_n > 0 else 0.0
            saved_dist_path = _save_mari_sample_distribution(
                results_dir=results_dir,
                model=model,
                dataset=str(args.dataset_name),
                mode=str(args.mode),
                tau=float(args.tau),
                selected_k=int(selected_k),
                n_total_samples=total_n,
                n_undefined_samples=mari_undefined_n,
                values=mari.sample_values,
            )
            saved_ri_dist_path = _save_ri_sample_distribution(
                results_dir=results_dir,
                model=model,
                dataset=str(args.dataset_name),
                mode=str(args.mode),
                selected_k=int(selected_k),
                n_total_samples=total_n,
                n_undefined_samples=ri_undefined_n,
                values=ri.sample_values,
            )
            ccrr_by_m, model_ccrr_m_rows = _compute_ccrr_m_sweep_rows(
                features=eval_features,
                manifest=eval_manifest,
                mode=str(args.mode),
                m_values=ccrr_m_values,
                dataset_name=str(args.dataset_name),
                model=model,
                tau=float(args.tau),
                k_candidates_sig=k_candidates_sig,
                excluded_centers_sig=excluded_centers_sig,
                ccrr_search_sig=ccrr_search_sig,
                embedding_path=output_path,
                ccrr_acceptance_threshold=float(args.ccrr_acceptance_threshold),
                ccrr_start_k=int(args.ccrr_start_k),
                ccrr_k_growth_factor=float(args.ccrr_k_growth_factor),
            )
            ccrr_m_sweep_rows.extend(model_ccrr_m_rows)
            ccrr_result = ccrr_by_m[1]
            ccrr_dist_path = _distribution_path(results_dir, "ccrr", model)
            ccrr_dist_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(ccrr_dist_path, np.asarray(ccrr_result.sample_values, dtype=float))
            row = {
                "dataset": str(args.dataset_name),
                "model": model,
                "k": int(ri.k),
                "mode": str(args.mode),
                "tau": float(args.tau),
                "k_candidates": k_candidates_sig,
                "excluded_centers": excluded_centers_sig,
                "ccrr_search": ccrr_search_sig,
                "bio_knn_bacc": float(knn_bacc_by_k[int(selected_k)]),
                "center_knn_bacc": float(knn_center_bacc_by_k[int(selected_k_center)]),
                "selected_k_center": int(selected_k_center),
                "ri": float(ri.value),
                "ri_std": float(ri.std),
                "mari": float(mari.value),
                "mari_std": float(mari.std),
                "ri_undefined_frac": ri_undefined_frac,
                "mari_undefined_frac": mari_undefined_frac,
                "ri_samples_path": str(saved_ri_dist_path),
                "mari_samples_path": str(saved_dist_path),
                "ccrr": float(ccrr_result.value),
                "ccrr_std": float(ccrr_result.std),
                "ccrr_m": int(ccrr_result.m),
                "ccrr_undefined_frac": float(ccrr_result.undefined_frac),
                "ccrr_acceptance_threshold": float(ccrr_result.acceptance_threshold),
                "ccrr_acceptance_met": bool(ccrr_result.acceptance_met),
                "ccrr_k_start": int(ccrr_result.k_start),
                "ccrr_k_final": int(ccrr_result.k_final),
                "ccrr_retries": int(ccrr_result.retries),
                "ccrr_alpha": float(ccrr_result.alpha),
                "ccrr_q_alpha": float(ccrr_result.q_alpha),
                "ccrr_ltm_alpha": float(ccrr_result.ltm_alpha),
                "ccrr_samples_path": str(ccrr_dist_path),
                "embedding_path": str(output_path),
            }
            rows.append(row)
            for k in k_values:
                k_sweep_rows.append(
                    {
                        "dataset": str(args.dataset_name),
                        "model": model,
                        "mode": str(args.mode),
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
            metrics_status[model] = "ok"
            print(
                f"[benchmark] RI={row['ri']:.4f} MaRI={row['mari']:.4f} CCRR={row['ccrr']:.4f} "
                f"undefined samples: RI={100*row['ri_undefined_frac']:.1f}%, MaRI={100*row['mari_undefined_frac']:.1f}%, "
                f"CCRR={100*row['ccrr_undefined_frac']:.1f}%"
            )
        except Exception as exc:  # noqa: BLE001
            metrics_status[model] = "failed"
            failures.append(f"{model}: metrics failed ({exc})")
            print(f"[benchmark] metrics failed: {exc}")

    if rows:
        save_metrics(rows=rows, csv_path=metrics_csv, json_path=metrics_json)
        save_k_sweep_metrics(rows=k_sweep_rows, csv_path=k_sweep_csv, json_path=k_sweep_json)
        save_k_sweep_metrics(rows=ccrr_m_sweep_rows, csv_path=ccrr_m_sweep_csv, json_path=ccrr_m_sweep_json)
        plot_knn_bio_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "knn_bio_k_sweep.png")
        plot_knn_center_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "knn_center_k_sweep.png")
        plot_ri_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "ri_k_sweep.png")
        plot_mari_k_sweep(rows=k_sweep_rows, out_path=plots_dir / "mari_k_sweep.png")
        plot_ccrr_m_sweep(rows=ccrr_m_sweep_rows, out_path=plots_dir / "ccrr_m_sweep.png")
        plot_bio_vs_center_scatter(rows=rows, out_path=plots_dir / "bio_vs_center_scatter.png")
        plot_mari_vs_ri_scatter(rows=rows, out_path=plots_dir / "mari_vs_ri_scatter.png")
        plot_ccrr_vs_mari_scatter(rows=rows, out_path=plots_dir / "ccrr_vs_mari_scatter.png")
        plot_ccrr_sample_distributions(rows=rows, out_path=plots_dir / "ccrr_sample_distributions.png")
        plot_benchmark_6panel_summary(
            rows=rows,
            k_sweep_rows=k_sweep_rows,
            out_path=plots_dir / "benchmark_6panel_summary.png",
        )

    print("\n[benchmark] === summary ===")
    for model in models:
        e = extraction_status.get(model, "n/a")
        m = metrics_status.get(model, "n/a")
        print(f"[benchmark] {model}: extract={e} metrics={m}")
    print(f"[benchmark] metrics_csv={metrics_csv}")
    print(f"[benchmark] metrics_json={metrics_json}")
    print(f"[benchmark] k_sweep_csv={k_sweep_csv}")
    print(f"[benchmark] k_sweep_json={k_sweep_json}")
    print(f"[benchmark] ccrr_m_sweep_csv={ccrr_m_sweep_csv}")
    print(f"[benchmark] ccrr_m_sweep_json={ccrr_m_sweep_json}")
    print(f"[benchmark] plots_dir={plots_dir}")

    if failures:
        print("[benchmark] failures:")
        for msg in failures:
            print(f"[benchmark] - {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
