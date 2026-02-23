#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mari import MaRI, RI
from mari.metrics.neighbors import _knn_balanced_accuracy_by_k, _normalize_k_values, _select_k_from_balanced_accuracy
from mari.metrics.pairs import load_manifest, normalize_center_values
from metrics_io import excluded_centers_signature, k_candidates_signature


def _parse_k_candidates(raw: str) -> list[int]:
    values = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("k-candidates must include at least one integer")
    return values


def _parse_embedding_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in specs:
        if "=" not in item:
            raise ValueError(f"Invalid --embedding spec '{item}'. Expected format: name=/path/to.npy")
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Embedding file not found for '{name}': {path}")
        parsed.append((name.strip(), path))
    return parsed


def _safe_model_name(model: str) -> str:
    return str(model).replace("/", "_").replace(":", "_")


def _sample_distribution_dir(output_dir: Path) -> Path:
    return output_dir / "sample_distributions"


def _save_mari_sample_distribution(
    *,
    output_dir: Path,
    model: str,
    dataset: str,
    mode: str,
    tau: float,
    selected_k: int,
    n_total_samples: int,
    n_undefined_samples: int,
    values: np.ndarray,
) -> Path:
    import json

    dist_dir = _sample_distribution_dir(output_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    out_path = dist_dir / f"mari.{_safe_model_name(model)}.npy"
    arr = np.asarray(values, dtype=float)
    np.save(out_path, arr)
    meta_path = dist_dir / f"mari.{_safe_model_name(model)}.json"
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
    output_dir: Path,
    model: str,
    dataset: str,
    mode: str,
    selected_k: int,
    n_total_samples: int,
    n_undefined_samples: int,
    values: np.ndarray,
) -> Path:
    import json

    dist_dir = _sample_distribution_dir(output_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    out_path = dist_dir / f"ri.{_safe_model_name(model)}.npy"
    arr = np.asarray(values, dtype=float)
    np.save(out_path, arr)
    meta_path = dist_dir / f"ri.{_safe_model_name(model)}.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Script-only benchmark helper for RI/MaRI on precomputed embeddings. "
            "This is intentionally outside the mari package API."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument(
        "--embedding",
        action="append",
        required=True,
        help="Repeated spec in the form name=/absolute/path/to/embeddings.npy",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["paired", "global"],
        help="Evaluation mode for RI/MaRI.",
    )
    parser.add_argument("--k-candidates", default="5,11,21")
    parser.add_argument(
        "--continuous-k-sweep-max",
        type=int,
        default=0,
        help="If > 0, sweep k continuously from 1..max instead of only --k-candidates.",
    )
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument(
        "--exclude-center",
        action="append",
        default=[],
        help="Medical center to exclude from computation. Repeat flag to exclude multiple centers.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--output-k-sweep-csv",
        required=True,
        help="Path for long-format per-k rows (knn_bacc and RI).",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, dataset_name=args.dataset_name)
    parsed_k_candidates = _parse_k_candidates(args.k_candidates)
    if int(args.continuous_k_sweep_max) > 0:
        k_candidates = list(range(1, int(args.continuous_k_sweep_max) + 1))
    else:
        k_candidates = _normalize_k_values(parsed_k_candidates)
    k_candidates_sig = k_candidates_signature(k_candidates)
    excluded_centers = normalize_center_values(args.exclude_center)
    excluded_centers_sig = excluded_centers_signature(excluded_centers)
    specs = _parse_embedding_specs(args.embedding)

    rows: list[dict] = []
    k_sweep_rows: list[dict] = []
    metrics_out_path = Path(args.output_csv)
    mari_dist_out_dir = metrics_out_path.parent
    center_series = manifest["medical_center"].map(str).str.strip()
    keep_mask = ~center_series.isin(excluded_centers)
    if not bool(keep_mask.any()):
        excluded_txt = ", ".join(excluded_centers)
        raise ValueError(
            f"No samples remain after excluding centers [{excluded_txt}] from dataset '{args.dataset_name}'"
        )
    keep_indices = np.flatnonzero(keep_mask.to_numpy())
    eval_manifest = manifest.loc[keep_mask].reset_index(drop=True)
    labels = pd.factorize(eval_manifest["label"])[0].astype(int)
    center_labels = pd.factorize(eval_manifest["medical_center"])[0].astype(int)
    slide_ids = eval_manifest["slide_id"].astype(str).to_numpy()
    for model_name, embedding_path in specs:
        features = np.load(embedding_path)[keep_indices]
        knn_bacc_by_k = _knn_balanced_accuracy_by_k(
            features=features,
            labels=labels,
            slide_ids=slide_ids,
            k_values=k_candidates,
            warn_context=f"{args.dataset_name} k-curve",
        )
        selected_k = _select_k_from_balanced_accuracy(
            k_values=k_candidates,
            scores=knn_bacc_by_k,
        )
        knn_center_bacc_by_k = _knn_balanced_accuracy_by_k(
            features=features,
            labels=center_labels,
            slide_ids=slide_ids,
            k_values=k_candidates,
            warn_context=f"{args.dataset_name} center-k-curve",
        )
        selected_k_center = _select_k_from_balanced_accuracy(
            k_values=k_candidates,
            scores=knn_center_bacc_by_k,
        )
        ri_curve = RI.compute_curve(
            features=features,
            manifest=eval_manifest,
            mode=args.mode,
            k_values=k_candidates,
        )
        mari_curve = MaRI.compute_curve(
            features=features,
            manifest=eval_manifest,
            mode=args.mode,
            k_values=k_candidates,
            tau=args.tau,
        )

        ri = RI.compute(
            features,
            eval_manifest,
            mode=args.mode,
            k_candidates=k_candidates,
        )
        mari = MaRI.compute(
            features,
            eval_manifest,
            mode=args.mode,
            k_candidates=k_candidates,
            tau=args.tau,
        )
        total_n = int(len(eval_manifest))
        ri_informative_n = int(len(ri.sample_values))
        mari_informative_n = int(len(mari.sample_values))
        ri_undefined_n = max(0, total_n - ri_informative_n)
        mari_undefined_n = max(0, total_n - mari_informative_n)
        ri_undefined_frac = float(ri_undefined_n / total_n) if total_n > 0 else 0.0
        mari_undefined_frac = float(mari_undefined_n / total_n) if total_n > 0 else 0.0
        dist_path = _save_mari_sample_distribution(
            output_dir=mari_dist_out_dir,
            model=model_name,
            dataset=str(args.dataset_name),
            mode=str(args.mode),
            tau=float(args.tau),
            selected_k=int(selected_k),
            n_total_samples=total_n,
            n_undefined_samples=mari_undefined_n,
            values=mari.sample_values,
        )
        ri_dist_path = _save_ri_sample_distribution(
            output_dir=mari_dist_out_dir,
            model=model_name,
            dataset=str(args.dataset_name),
            mode=str(args.mode),
            selected_k=int(selected_k),
            n_total_samples=total_n,
            n_undefined_samples=ri_undefined_n,
            values=ri.sample_values,
        )
        rows.append(
            {
                "model": model_name,
                "k": ri.k,
                "ri": ri.value,
                "ri_std": ri.std,
                "ri_undefined_frac": ri_undefined_frac,
                "mari": mari.value,
                "mari_std": mari.std,
                "bio_knn_bacc": float(knn_bacc_by_k[int(selected_k)]),
                "center_knn_bacc": float(knn_center_bacc_by_k[int(selected_k_center)]),
                "selected_k_center": int(selected_k_center),
                "mari_undefined_frac": mari_undefined_frac,
                "k_candidates": k_candidates_sig,
                "excluded_centers": excluded_centers_sig,
                "ri_samples_path": str(ri_dist_path),
                "mari_samples_path": str(dist_path),
            }
        )
        for k in k_candidates:
            k_sweep_rows.append(
                {
                    "dataset": str(args.dataset_name),
                    "model": model_name,
                    "mode": str(args.mode),
                    "tau": float(args.tau),
                    "k_candidates": k_candidates_sig,
                    "excluded_centers": excluded_centers_sig,
                    "k": int(k),
                    "knn_bacc": float(knn_bacc_by_k[int(k)]),
                    "knn_center_bacc": float(knn_center_bacc_by_k[int(k)]),
                    "ri": float(ri_curve[int(k)]),
                    "mari": float(mari_curve[int(k)]),
                    "selected_k": int(selected_k),
                    "selected_k_center": int(selected_k_center),
                    "continuous_k_sweep": int(int(args.continuous_k_sweep_max) > 0),
                    "embedding_path": str(embedding_path),
                }
            )

    out_path = metrics_out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows -> {out_path}")

    out_k_sweep_path = Path(args.output_k_sweep_csv)
    out_k_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(k_sweep_rows).to_csv(out_k_sweep_path, index=False)
    print(f"Wrote {len(k_sweep_rows)} rows -> {out_k_sweep_path}")


if __name__ == "__main__":
    main()
