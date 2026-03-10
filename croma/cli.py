
import argparse
import json
from pathlib import Path

import numpy as np

from croma import CCMR, MaRI, RI
from croma.metrics.pairs import load_manifest, normalize_center_values


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
        "undefined_frac": result.undefined_frac,
        "evaluation_design": result.evaluation_design,
        "evaluation_unit": result.evaluation_unit,
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
        "--evaluation-design",
        required=True,
        choices=["paired_2x2", "dataset_wide"],
        help="Evaluation design: paired_2x2=explicit manifest-defined 2x2 subsets, dataset_wide=one full-dataset evaluation.",
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

    ccmr_shared = argparse.ArgumentParser(add_help=False)
    ccmr_shared.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    ccmr_shared.add_argument("--embeddings", required=True, help="Path to NPY embeddings.")
    ccmr_shared.add_argument("--dataset-name", default="dataset", help="Dataset name for manifest loading.")
    ccmr_shared.add_argument(
        "--evaluation-design",
        required=True,
        choices=["paired_2x2", "dataset_wide"],
        help="Evaluation design: paired_2x2=explicit manifest-defined 2x2 subsets, dataset_wide=one full-dataset evaluation.",
    )
    ccmr_shared.add_argument(
        "--exclude-center",
        action="append",
        default=[],
        help="Medical center to exclude from computation. Repeat flag to exclude multiple centers.",
    )
    ccmr_parser = sub.add_parser("ccmr", parents=[ccmr_shared], help="Compute CCMR.")
    ccmr_parser.add_argument("--m", type=int, default=1, help="Number of SO/OS neighbors to average (>=1).")
    ccmr_parser.add_argument("--alpha", type=float, default=0.10, help="Tail percentile for Q_alpha and LTM_alpha (default 0.10).")
    ccmr_parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=0.0,
        help="Stop CCMR search once undefined fraction is <= threshold (default 0.0).",
    )
    ccmr_parser.add_argument(
        "--start-k",
        type=int,
        default=200,
        help="Initial k for iterative CCMR neighbor search (default 200).",
    )
    ccmr_parser.add_argument(
        "--k-growth-factor",
        type=float,
        default=2.0,
        help="Geometric growth factor for CCMR iterative k search (>1, default 2.0).",
    )

    args = parser.parse_args()
    manifest = load_manifest(str(args.manifest), dataset_name=str(args.dataset_name))
    features = np.load(Path(args.embeddings))
    excluded_centers = normalize_center_values(args.exclude_center)

    if args.command == "ccmr":
        result = CCMR.compute(
            features=features,
            manifest=manifest,
            evaluation_design=str(args.evaluation_design),
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
            "acceptance_threshold": result.acceptance_threshold,
            "acceptance_met": result.acceptance_met,
            "k_start": result.k_start,
            "k_final": result.k_final,
            "retries": result.retries,
            "alpha": result.alpha,
            "q_alpha": result.q_alpha,
            "ltm_alpha": result.ltm_alpha,
            "evaluation_design": result.evaluation_design,
            "evaluation_unit": result.evaluation_unit,
            "excluded_centers": list(excluded_centers),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "ri":
        result = RI.compute(
            features=features,
            manifest=manifest,
            evaluation_design=str(args.evaluation_design),
            k_candidates=args.k_candidates,
            exclude_centers=excluded_centers,
        )
    else:
        result = MaRI.compute(
            features=features,
            manifest=manifest,
            evaluation_design=str(args.evaluation_design),
            k_candidates=args.k_candidates,
            tau=float(args.tau),
            exclude_centers=excluded_centers,
        )
    payload = _result_payload(result)
    payload["excluded_centers"] = list(excluded_centers)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
