#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extract_embeddings as ee
from common import parse_k_candidates
from mari import MaRI, RI, lower_tail_mean, tail_percentile
from mari.metrics.pairs import load_manifest
from metrics_io import load_cached_rows, save_metrics
from plotting import plot_rank, plot_ri_vs_mari, plot_tail_fragility


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified benchmark pipeline: extract embeddings, compute RI/MaRI metrics, and plot results."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest CSV.")
    parser.add_argument("--dataset-name", default="dataset", help="Dataset label for metrics output.")
    parser.add_argument("--models", required=True, help="Comma-separated model names.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Benchmark output directory.")
    parser.add_argument("--mode", default="global", choices=["paired", "global"], help="RI/MaRI mode.")
    parser.add_argument("--k-candidates", default="3,5,7,10,15,20,25", help="Comma-separated k candidates.")
    parser.add_argument("--tau", type=float, default=0.2, help="MaRI tau.")
    parser.add_argument("--alpha", type=float, default=10.0, help="Tail percentile alpha.")
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


def main() -> int:
    args = _parse_args()
    models = ee._parse_models(args.models)
    registry = ee._build_model_registry()

    unknown = [m for m in models if m not in registry]
    if unknown:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown model(s): {unknown}. Available: {available}")

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
    cached_rows = {}
    if not args.recompute_metrics:
        cached_rows = load_cached_rows(
            metrics_csv=metrics_csv,
            models=models,
            mode=args.mode,
            tau=float(args.tau),
            alpha=float(args.alpha),
        )

    extraction_status: dict[str, str] = {}
    metrics_status: dict[str, str] = {}
    failures: list[str] = []
    rows: list[dict] = []

    print(f"[benchmark] manifest={args.manifest}")
    print(f"[benchmark] models={', '.join(models)}")
    print(f"[benchmark] output_dir={output_dir}")
    print(f"[benchmark] dataset_dir={dataset_dir}")

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
            if model in cached_rows:
                row = dict(cached_rows[model])
                rows.append(row)
                metrics_status[model] = "cached"
                print("[benchmark] metrics cache hit")
                continue

            manifest_df = load_manifest(str(args.manifest), dataset_name=str(args.dataset_name))
            features = np.load(output_path)
            k_candidates = parse_k_candidates(args.k_candidates)

            ri = RI.compute(
                features=features,
                manifest=manifest_df,
                mode=args.mode,
                k_candidates=k_candidates,
            )
            mari = MaRI.compute(
                features=features,
                manifest=manifest_df,
                mode=args.mode,
                k_candidates=k_candidates,
                tau=float(args.tau),
            )
            row = {
                "dataset": str(args.dataset_name),
                "model": model,
                "k": int(ri.k),
                "mode": str(args.mode),
                "tau": float(args.tau),
                "alpha": float(args.alpha),
                "ri": float(ri.value),
                "ri_std": float(ri.std),
                "mari": float(mari.value),
                "mari_std": float(mari.std),
                "ri_q_alpha": tail_percentile(ri.sample_values, float(args.alpha)),
                "ri_ltm_alpha": lower_tail_mean(ri.sample_values, float(args.alpha)),
                "mari_q_alpha": tail_percentile(mari.sample_values, float(args.alpha)),
                "mari_ltm_alpha": lower_tail_mean(mari.sample_values, float(args.alpha)),
                "embedding_path": str(output_path),
            }
            rows.append(row)
            metrics_status[model] = "ok"
            print(
                f"[benchmark] ri={row['ri']:.4f} mari={row['mari']:.4f} "
                f"q{args.alpha:g}={row['mari_q_alpha']:.4f} ltm{args.alpha:g}={row['mari_ltm_alpha']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001
            metrics_status[model] = "failed"
            failures.append(f"{model}: metrics failed ({exc})")
            print(f"[benchmark] metrics failed: {exc}")

    if rows:
        save_metrics(rows=rows, csv_path=metrics_csv, json_path=metrics_json)
        plot_rank(rows=rows, out_path=plots_dir / "ri_mari_rank.png")
        plot_ri_vs_mari(rows=rows, out_path=plots_dir / "ri_vs_mari_scatter.png")
        plot_tail_fragility(
            rows=rows,
            out_path=plots_dir / "tail_fragility.png",
            alpha=float(args.alpha),
        )

    print("\n[benchmark] === summary ===")
    for model in models:
        e = extraction_status.get(model, "n/a")
        m = metrics_status.get(model, "n/a")
        print(f"[benchmark] {model}: extract={e} metrics={m}")
    print(f"[benchmark] metrics_csv={metrics_csv}")
    print(f"[benchmark] metrics_json={metrics_json}")
    print(f"[benchmark] plots_dir={plots_dir}")

    if failures:
        print("[benchmark] failures:")
        for msg in failures:
            print(f"[benchmark] - {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
