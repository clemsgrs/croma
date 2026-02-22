from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mari import MaRI, RI, lower_tail_mean, tail_percentile
from mari.metrics.pairs import load_manifest


def _parse_k_candidates(s: str) -> list[int]:
    values = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("k-candidates must include at least one integer")
    return values


def _result_payload(result, alpha: float | None) -> dict:
    payload = {
        "dataset": result.dataset,
        "k": result.k,
        "value": result.value,
        "std": result.std,
        "n_pairs": result.n_pairs,
    }
    if alpha is not None:
        payload[f"q{alpha:g}"] = tail_percentile(result.sample_values, alpha)
        payload[f"ltm{alpha:g}"] = lower_tail_mean(result.sample_values, alpha)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RI/MaRI metrics.")
    sub = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    shared.add_argument("--embeddings", required=True, help="Path to NPY embeddings.")
    shared.add_argument("--dataset-name", default="dataset", help="Dataset name for manifest loading.")
    shared.add_argument(
        "--mode",
        required=True,
        choices=["paired", "global"],
        help="Evaluation mode: paired=PathoROB-style 2x2 aggregation, global=single full-dataset evaluation.",
    )
    shared.add_argument("--k-candidates", type=_parse_k_candidates, default=[5, 11, 21])
    shared.add_argument("--alpha", type=float, default=None, help="Tail percentile alpha in [0,100].")

    ri_parser = sub.add_parser("ri", parents=[shared], help="Compute RI.")
    mari_parser = sub.add_parser("mari", parents=[shared], help="Compute MaRI.")
    mari_parser.add_argument("--tau", type=float, default=0.2, help="Distance-decay temperature (>0).")

    args = parser.parse_args()
    manifest = load_manifest(str(args.manifest), dataset_name=str(args.dataset_name))
    features = np.load(Path(args.embeddings))

    if args.command == "ri":
        result = RI.compute(
            features=features,
            manifest=manifest,
            mode=str(args.mode),
            k_candidates=args.k_candidates,
        )
    else:
        result = MaRI.compute(
            features=features,
            manifest=manifest,
            mode=str(args.mode),
            k_candidates=args.k_candidates,
            tau=float(args.tau),
        )

    print(json.dumps(_result_payload(result, args.alpha), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
