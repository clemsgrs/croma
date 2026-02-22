#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mari import MaRI, RI, lower_tail_mean, tail_percentile
from mari.metrics.pairs import load_manifest


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
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, dataset_name=args.dataset_name)
    k_candidates = _parse_k_candidates(args.k_candidates)
    specs = _parse_embedding_specs(args.embedding)

    rows: list[dict] = []
    for model_name, embedding_path in specs:
        features = np.load(embedding_path)
        ri = RI.compute(
            features,
            manifest,
            mode=args.mode,
            k_candidates=k_candidates,
        )
        mari = MaRI.compute(
            features,
            manifest,
            mode=args.mode,
            k_candidates=k_candidates,
            tau=args.tau,
        )
        rows.append(
            {
                "model": model_name,
                "k": ri.k,
                "ri": ri.value,
                "ri_std": ri.std,
                "ri_q_alpha": tail_percentile(ri.sample_values, args.alpha),
                "ri_ltm_alpha": lower_tail_mean(ri.sample_values, args.alpha),
                "mari": mari.value,
                "mari_std": mari.std,
                "mari_q_alpha": tail_percentile(mari.sample_values, args.alpha),
                "mari_ltm_alpha": lower_tail_mean(mari.sample_values, args.alpha),
            }
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
