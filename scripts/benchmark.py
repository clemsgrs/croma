import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extract_embeddings as ee
from common import parse_k_candidates
from input_fingerprint import embedding_fingerprint, manifest_fingerprint
from mari import CCRR, MaRI, RI
from mari.metrics.neighbors import (
    _knn_balanced_accuracy_by_k,
    _normalize_k_values,
    _select_k_from_balanced_accuracy,
)
from mari.metrics.pairs import load_manifest, normalize_center_values
from metrics_cache import MetricsArtifactCache, build_cache_key
from metrics_io import (
    ccrr_search_signature,
    excluded_centers_signature,
    k_candidates_signature,
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
        default="1,5,10,15,20",
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
        return {
            "k": int(payload["k"]),
            "value": float(payload["value"]),
            "std": float(payload["std"]),
            "undefined_frac": float(payload["undefined_frac"]),
        }
    except Exception:  # noqa: BLE001
        return None


def _ccrr_result_to_payload(result: object, m: int) -> dict:
    return {
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
    }


def _ccrr_payload_from_results(results: dict[int, object]) -> dict:
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
    mode: str,
    m_values: list[int],
    ccrr_acceptance_threshold: float,
    ccrr_start_k: int,
    ccrr_k_growth_factor: float,
) -> dict[int, object]:
    out: dict[int, object] = {}
    for m in m_values:
        out[int(m)] = CCRR.compute(
            features=features,
            manifest=manifest,
            mode=mode,
            m=int(m),
            acceptance_threshold=float(ccrr_acceptance_threshold),
            start_k=int(ccrr_start_k),
            k_growth_factor=float(ccrr_k_growth_factor),
        )
    return out


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

    cache = MetricsArtifactCache(results_dir=results_dir)

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
    base_manifest_fingerprint = manifest_fingerprint(manifest_df)

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
            embedding_fp = embedding_fingerprint(output_path)
            input_fp = {
                "manifest_fingerprint": base_manifest_fingerprint,
                "embedding_fingerprint": embedding_fp,
                "excluded_centers_signature": excluded_centers_sig,
            }

            k_values_param = [int(k) for k in k_values]
            m_values_param = [int(m) for m in ccrr_m_values]
            mode_value = str(args.mode)
            tau_value = float(args.tau)

            keys = {
                "knn_bio_curve": build_cache_key(
                    artifact_name="knn_bio_curve",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"k_values": k_values_param},
                ),
                "knn_center_curve": build_cache_key(
                    artifact_name="knn_center_curve",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"k_values": k_values_param},
                ),
                "ri_curve": build_cache_key(
                    artifact_name="ri_curve",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"mode": mode_value, "k_values": k_values_param},
                ),
                "mari_curve": build_cache_key(
                    artifact_name="mari_curve",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"mode": mode_value, "k_values": k_values_param, "tau": tau_value},
                ),
                "ri_summary": build_cache_key(
                    artifact_name="ri_summary",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"mode": mode_value, "k_values": k_values_param},
                ),
                "ri_samples": build_cache_key(
                    artifact_name="ri_samples",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"mode": mode_value, "k_values": k_values_param},
                ),
                "mari_summary": build_cache_key(
                    artifact_name="mari_summary",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"mode": mode_value, "k_values": k_values_param, "tau": tau_value},
                ),
                "mari_samples": build_cache_key(
                    artifact_name="mari_samples",
                    model=model,
                    input_fingerprint=input_fp,
                    params={"mode": mode_value, "k_values": k_values_param, "tau": tau_value},
                ),
                "ccrr_m_sweep": build_cache_key(
                    artifact_name="ccrr_m_sweep",
                    model=model,
                    input_fingerprint=input_fp,
                    params={
                        "mode": mode_value,
                        "m_values": m_values_param,
                        "acceptance_threshold": float(args.ccrr_acceptance_threshold),
                        "start_k": int(args.ccrr_start_k),
                        "k_growth_factor": float(args.ccrr_k_growth_factor),
                        "alpha": 0.10,
                    },
                ),
                "ccrr_m1_samples": build_cache_key(
                    artifact_name="ccrr_m1_samples",
                    model=model,
                    input_fingerprint=input_fp,
                    params={
                        "mode": mode_value,
                        "m_values": m_values_param,
                        "acceptance_threshold": float(args.ccrr_acceptance_threshold),
                        "start_k": int(args.ccrr_start_k),
                        "k_growth_factor": float(args.ccrr_k_growth_factor),
                        "alpha": 0.10,
                    },
                ),
            }

            knn_bacc_by_k: dict[int, float] | None = None
            knn_center_bacc_by_k: dict[int, float] | None = None
            ri_curve: dict[int, float] | None = None
            mari_curve: dict[int, float] | None = None
            ri_summary: dict | None = None
            ri_samples: np.ndarray | None = None
            mari_summary: dict | None = None
            mari_samples: np.ndarray | None = None
            ccrr_by_m: dict[int, dict] | None = None
            ccrr_samples: np.ndarray | None = None

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
                if ri_summary is None or ri_samples is None:
                    all_cache_hit = False

                mari_summary = _summary_from_payload(cache.get_json(key=keys["mari_summary"]))
                mari_samples = cache.get_npy(key=keys["mari_samples"])
                if mari_summary is None or mari_samples is None:
                    all_cache_hit = False

                ccrr_by_m = _ccrr_payload_to_by_m(
                    cache.get_json(key=keys["ccrr_m_sweep"]),
                    expected_m_values=ccrr_m_values,
                )
                ccrr_samples = cache.get_npy(key=keys["ccrr_m1_samples"])
                if ccrr_by_m is None or ccrr_samples is None:
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

            if knn_bacc_by_k is None:
                knn_bacc_by_k = _knn_balanced_accuracy_by_k(
                    features=_ensure_eval_features(),
                    labels=labels,
                    slide_ids=slide_ids,
                    k_values=k_values,
                    warn_context=f"{args.dataset_name} k-curve",
                )
                cache.put_json(key=keys["knn_bio_curve"], payload=_curve_payload(knn_bacc_by_k))

            if knn_center_bacc_by_k is None:
                knn_center_bacc_by_k = _knn_balanced_accuracy_by_k(
                    features=_ensure_eval_features(),
                    labels=center_labels,
                    slide_ids=slide_ids,
                    k_values=k_values,
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

            if ri_curve is None:
                ri_curve = RI.compute_curve(
                    features=_ensure_eval_features(),
                    manifest=eval_manifest,
                    mode=args.mode,
                    k_values=k_values,
                )
                cache.put_json(key=keys["ri_curve"], payload=_curve_payload(ri_curve))

            if mari_curve is None:
                mari_curve = MaRI.compute_curve(
                    features=_ensure_eval_features(),
                    manifest=eval_manifest,
                    mode=args.mode,
                    k_values=k_values,
                    tau=float(args.tau),
                )
                cache.put_json(key=keys["mari_curve"], payload=_curve_payload(mari_curve))

            if ri_summary is None or ri_samples is None:
                ri = RI.compute(
                    features=_ensure_eval_features(),
                    manifest=eval_manifest,
                    mode=args.mode,
                    k_candidates=k_values,
                )
                if int(ri.k) != int(selected_k):
                    raise RuntimeError(
                        f"Inconsistent selected k: RI returned {ri.k} but kNN balanced accuracy selected {selected_k}"
                    )
                total_n = int(len(eval_manifest))
                ri_informative_n = int(len(ri.sample_values))
                ri_undefined_n = max(0, total_n - ri_informative_n)
                ri_summary = {
                    "k": int(ri.k),
                    "value": float(ri.value),
                    "std": float(ri.std),
                    "undefined_frac": float(ri_undefined_n / total_n) if total_n > 0 else 0.0,
                }
                ri_samples = np.asarray(ri.sample_values, dtype=float)
                cache.put_json(key=keys["ri_summary"], payload=ri_summary)
                cache.put_npy(key=keys["ri_samples"], values=ri_samples)
            else:
                ri_samples = np.asarray(ri_samples, dtype=float)

            if mari_summary is None or mari_samples is None:
                mari = MaRI.compute(
                    features=_ensure_eval_features(),
                    manifest=eval_manifest,
                    mode=args.mode,
                    k_candidates=k_values,
                    tau=float(args.tau),
                )
                if int(mari.k) != int(selected_k):
                    raise RuntimeError(
                        f"Inconsistent selected k: MaRI returned {mari.k} but kNN balanced accuracy selected {selected_k}"
                    )
                total_n = int(len(eval_manifest))
                mari_informative_n = int(len(mari.sample_values))
                mari_undefined_n = max(0, total_n - mari_informative_n)
                mari_summary = {
                    "k": int(mari.k),
                    "value": float(mari.value),
                    "std": float(mari.std),
                    "undefined_frac": float(mari_undefined_n / total_n) if total_n > 0 else 0.0,
                }
                mari_samples = np.asarray(mari.sample_values, dtype=float)
                cache.put_json(key=keys["mari_summary"], payload=mari_summary)
                cache.put_npy(key=keys["mari_samples"], values=mari_samples)
            else:
                mari_samples = np.asarray(mari_samples, dtype=float)

            if ccrr_by_m is None or ccrr_samples is None:
                ccrr_results = _compute_ccrr_by_m(
                    features=_ensure_eval_features(),
                    manifest=eval_manifest,
                    mode=str(args.mode),
                    m_values=ccrr_m_values,
                    ccrr_acceptance_threshold=float(args.ccrr_acceptance_threshold),
                    ccrr_start_k=int(args.ccrr_start_k),
                    ccrr_k_growth_factor=float(args.ccrr_k_growth_factor),
                )
                ccrr_by_m = _ccrr_payload_to_by_m(
                    _ccrr_payload_from_results(ccrr_results),
                    expected_m_values=ccrr_m_values,
                )
                if ccrr_by_m is None:
                    raise RuntimeError("Failed to serialize ccrr m-sweep cache payload")
                ccrr_samples = np.asarray(ccrr_results[1].sample_values, dtype=float)
                cache.put_json(key=keys["ccrr_m_sweep"], payload={"by_m": {str(k): v for k, v in ccrr_by_m.items()}})
                cache.put_npy(key=keys["ccrr_m1_samples"], values=ccrr_samples)
            else:
                ccrr_samples = np.asarray(ccrr_samples, dtype=float)

            ccrr_m_rows_for_model: list[dict] = []
            for m in ccrr_m_values:
                payload = ccrr_by_m[int(m)]
                ccrr_m_rows_for_model.append(
                    {
                        "dataset": str(args.dataset_name),
                        "model": str(model),
                        "mode": str(args.mode),
                        "tau": float(args.tau),
                        "k_candidates": str(k_candidates_sig),
                        "excluded_centers": str(excluded_centers_sig),
                        "ccrr_search": str(ccrr_search_sig),
                        "m": int(payload["m"]),
                        "ccrr": float(payload["ccrr"]),
                        "ccrr_std": float(payload["ccrr_std"]),
                        "ccrr_undefined_frac": float(payload["ccrr_undefined_frac"]),
                        "ccrr_acceptance_threshold": float(payload["ccrr_acceptance_threshold"]),
                        "ccrr_acceptance_met": bool(payload["ccrr_acceptance_met"]),
                        "ccrr_k_start": int(payload["ccrr_k_start"]),
                        "ccrr_k_final": int(payload["ccrr_k_final"]),
                        "ccrr_retries": int(payload["ccrr_retries"]),
                        "ccrr_alpha": float(payload["ccrr_alpha"]),
                        "ccrr_q_alpha": float(payload["ccrr_q_alpha"]),
                        "ccrr_ltm_alpha": float(payload["ccrr_ltm_alpha"]),
                        "embedding_path": str(output_path),
                    }
                )

            total_n = int(len(eval_manifest))
            ri_undefined_n = int(round(float(ri_summary["undefined_frac"]) * total_n))
            mari_undefined_n = int(round(float(mari_summary["undefined_frac"]) * total_n))
            saved_dist_path = _save_mari_sample_distribution(
                results_dir=results_dir,
                model=model,
                dataset=str(args.dataset_name),
                mode=str(args.mode),
                tau=float(args.tau),
                selected_k=int(selected_k),
                n_total_samples=total_n,
                n_undefined_samples=mari_undefined_n,
                values=mari_samples,
            )
            saved_ri_dist_path = _save_ri_sample_distribution(
                results_dir=results_dir,
                model=model,
                dataset=str(args.dataset_name),
                mode=str(args.mode),
                selected_k=int(selected_k),
                n_total_samples=total_n,
                n_undefined_samples=ri_undefined_n,
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
                "mode": str(args.mode),
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
                "mari_undefined_frac": float(mari_summary["undefined_frac"]),
                "ri_samples_path": str(saved_ri_dist_path),
                "mari_samples_path": str(saved_dist_path),
                "ccrr": float(ccrr_result["ccrr"]),
                "ccrr_std": float(ccrr_result["ccrr_std"]),
                "ccrr_m": int(ccrr_result["m"]),
                "ccrr_undefined_frac": float(ccrr_result["ccrr_undefined_frac"]),
                "ccrr_acceptance_threshold": float(ccrr_result["ccrr_acceptance_threshold"]),
                "ccrr_acceptance_met": bool(ccrr_result["ccrr_acceptance_met"]),
                "ccrr_k_start": int(ccrr_result["ccrr_k_start"]),
                "ccrr_k_final": int(ccrr_result["ccrr_k_final"]),
                "ccrr_retries": int(ccrr_result["ccrr_retries"]),
                "ccrr_alpha": float(ccrr_result["ccrr_alpha"]),
                "ccrr_q_alpha": float(ccrr_result["ccrr_q_alpha"]),
                "ccrr_ltm_alpha": float(ccrr_result["ccrr_ltm_alpha"]),
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
            ccrr_m_sweep_rows.extend(ccrr_m_rows_for_model)

            metrics_status[model] = "cached" if all_cache_hit else "ok"
            if all_cache_hit:
                print("[benchmark] metrics cache hit")
            else:
                print("[benchmark] metrics cache miss: partial/full recompute")
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
