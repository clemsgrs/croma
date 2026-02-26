from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mari import CCRR, MaRI, RI
from mari.metrics.pairs import load_manifest, normalize_center_values


def _parse_k_candidates(s: str) -> list[int]:
    values = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("k-candidates must include at least one integer")
    return values


def _result_payload(result) -> dict:
    payload = {
        "dataset": result.dataset,
        "k": result.k,
        "value": result.value,
        "std": result.std,
        "n_pairs": result.n_pairs,
    }
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
    shared.add_argument(
        "--exclude-center",
        action="append",
        default=[],
        help="Medical center to exclude from computation. Repeat flag to exclude multiple centers.",
    )

    ri_parser = sub.add_parser("ri", parents=[shared], help="Compute RI.")
    mari_parser = sub.add_parser("mari", parents=[shared], help="Compute MaRI.")
    mari_parser.add_argument("--tau", type=float, default=0.2, help="Distance-decay temperature (>0).")

    ccrr_shared = argparse.ArgumentParser(add_help=False)
    ccrr_shared.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    ccrr_shared.add_argument("--embeddings", required=True, help="Path to NPY embeddings.")
    ccrr_shared.add_argument("--dataset-name", default="dataset", help="Dataset name for manifest loading.")
    ccrr_shared.add_argument(
        "--mode",
        required=True,
        choices=["paired", "global"],
        help="Evaluation mode: paired=PathoROB-style 2x2 aggregation, global=single full-dataset evaluation.",
    )
    ccrr_shared.add_argument(
        "--exclude-center",
        action="append",
        default=[],
        help="Medical center to exclude from computation. Repeat flag to exclude multiple centers.",
    )
    ccrr_parser = sub.add_parser("ccrr", parents=[ccrr_shared], help="Compute CCRR.")
    ccrr_parser.add_argument("--m", type=int, default=1, help="Number of SO/OS neighbors to average (>=1).")
    ccrr_parser.add_argument("--alpha", type=float, default=0.10, help="Tail percentile for Q_alpha and LTM_alpha (default 0.10).")
    ccrr_parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=0.0,
        help="Stop CCRR search once feasible undefined fraction is <= threshold (default 0.0).",
    )
    ccrr_parser.add_argument(
        "--start-k",
        type=int,
        default=200,
        help="Initial k for iterative CCRR neighbor search (default 200).",
    )
    ccrr_parser.add_argument(
        "--k-growth-factor",
        type=float,
        default=1.5,
        help="Geometric growth factor for CCRR iterative k search (>1, default 1.5).",
    )

    args = parser.parse_args()
    manifest = load_manifest(str(args.manifest), dataset_name=str(args.dataset_name))
    features = np.load(Path(args.embeddings))
    excluded_centers = normalize_center_values(args.exclude_center)

    if args.command == "ccrr":
        result = CCRR.compute(
            features=features,
            manifest=manifest,
            mode=str(args.mode),
            m=int(args.m),
            alpha=float(args.alpha),
            exclude_centers=excluded_centers,
            acceptance_threshold=float(args.acceptance_threshold),
            start_k=int(args.start_k),
            k_growth_factor=float(args.k_growth_factor),
        )
        payload = {
            "dataset": result.dataset,
            "m": result.m,
            "value": result.value,
            "undefined_frac": result.undefined_frac,
            "undefined_frac_all": result.undefined_frac_all,
            "undefined_frac_feasible": result.undefined_frac_feasible,
            "n_feasible": result.n_feasible,
            "acceptance_threshold": result.acceptance_threshold,
            "acceptance_met": result.acceptance_met,
            "k_start": result.k_start,
            "k_final": result.k_final,
            "retries": result.retries,
            "alpha": result.alpha,
            "q_alpha": result.q_alpha,
            "ltm_alpha": result.ltm_alpha,
            "excluded_centers": list(excluded_centers),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "ri":
        result = RI.compute(
            features=features,
            manifest=manifest,
            mode=str(args.mode),
            k_candidates=args.k_candidates,
            exclude_centers=excluded_centers,
        )
    else:
        result = MaRI.compute(
            features=features,
            manifest=manifest,
            mode=str(args.mode),
            k_candidates=args.k_candidates,
            tau=float(args.tau),
            exclude_centers=excluded_centers,
        )
    payload = _result_payload(result)
    payload["excluded_centers"] = list(excluded_centers)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
