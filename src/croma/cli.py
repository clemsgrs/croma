import argparse
import json
from pathlib import Path

import numpy as np

from croma import CCMR, MaRI, RI
from croma.metrics.ccmr import CCMR_HEADLINE_M
from croma.alignment import (
    build_embedding_source_manifest,
    expand_features_to_manifest,
)
from croma.metrics.pairs import load_manifest


def _parse_k_candidates(s: str) -> list[int]:
    values = [int(v.strip()) for v in s.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "k-candidates must include at least one integer"
        )
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


def _load_eval_features(
    *, manifest_path: Path, embeddings_path: Path, confounder_column: str
) -> tuple[np.ndarray, object]:
    manifest = load_manifest(str(manifest_path), confounder_column=confounder_column)
    features = np.load(embeddings_path)
    if int(features.shape[0]) != int(len(manifest)):
        raise ValueError(
            "embeddings rows must match manifest rows. "
            "Metric commands require manifest-aligned embeddings. "
            "If you started from deduplicated embeddings, run "
            "`croma build-embedding-manifest` and `croma expand-embeddings` first."
        )
    return features, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RI, MaRI, and CCMR metrics, or prepare aligned embedding inputs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--manifest", required=True, help="Path to manifest CSV.")
    shared.add_argument("--embeddings", required=True, help="Path to NPY embeddings.")
    shared.add_argument(
        "--confounder-column",
        required=True,
        help="Manifest column to treat as the non-biological confounder.",
    )
    shared.add_argument(
        "--evaluation-design",
        required=True,
        choices=["paired_2x2", "dataset_wide"],
        help="Evaluation design: paired_2x2=explicit manifest-defined 2x2 subsets, dataset_wide=one full-dataset evaluation.",
    )
    shared.add_argument("--k-candidates", type=_parse_k_candidates, default=[5, 11, 21])

    ri_parser = sub.add_parser("ri", parents=[shared], help="Compute RI.")
    mari_parser = sub.add_parser("mari", parents=[shared], help="Compute MaRI.")
    mari_parser.add_argument(
        "--tau", type=float, default=0.2, help="Distance-decay temperature (>0)."
    )

    build_parser = sub.add_parser(
        "build-embedding-manifest", help="Build a deduplicated embedding manifest."
    )
    build_parser.add_argument(
        "--manifest", required=True, help="Path to evaluation manifest CSV."
    )
    build_parser.add_argument(
        "--confounder-column",
        required=True,
        help="Manifest column to treat as the non-biological confounder.",
    )
    build_parser.add_argument(
        "--out",
        required=True,
        help="Path to output CSV for the deduplicated embedding manifest.",
    )

    expand_parser = sub.add_parser(
        "expand-embeddings",
        help="Expand deduplicated embeddings back to manifest-row order.",
    )
    expand_parser.add_argument(
        "--manifest", required=True, help="Path to evaluation manifest CSV."
    )
    expand_parser.add_argument(
        "--confounder-column",
        required=True,
        help="Manifest column to treat as the non-biological confounder.",
    )
    expand_parser.add_argument(
        "--embedding-manifest",
        required=True,
        help="Path to deduplicated embedding manifest CSV.",
    )
    expand_parser.add_argument(
        "--embeddings", required=True, help="Path to deduplicated NPY embeddings."
    )
    expand_parser.add_argument(
        "--out", required=True, help="Path to output manifest-aligned NPY embeddings."
    )

    ccmr_parser = sub.add_parser("ccmr", parents=[shared], help="Compute CCMR.")
    ccmr_parser.add_argument(
        "--m",
        type=int,
        default=CCMR_HEADLINE_M,
        help="Number of SO/OS neighbors to average (>=1).",
    )
    ccmr_parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Tail percentile for Q_alpha and LTM_alpha (default 0.10).",
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
    if args.command == "build-embedding-manifest":
        manifest = load_manifest(
            str(args.manifest), confounder_column=str(args.confounder_column)
        )
        embedding_manifest, _ = build_embedding_source_manifest(manifest)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_manifest.to_csv(out_path, index=False)
        payload = {
            "manifest": str(Path(args.manifest)),
            "manifest_rows": int(len(manifest)),
            "embedding_manifest": str(out_path),
            "embedding_manifest_rows": int(len(embedding_manifest)),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "expand-embeddings":
        manifest = load_manifest(
            str(args.manifest), confounder_column=str(args.confounder_column)
        )
        embedding_manifest = load_manifest(
            str(args.embedding_manifest), confounder_column="confounder"
        )
        features = np.load(Path(args.embeddings))
        expanded = expand_features_to_manifest(
            features=features,
            manifest=manifest,
            embedding_manifest=embedding_manifest,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, expanded)
        payload = {
            "manifest": str(Path(args.manifest)),
            "manifest_rows": int(len(manifest)),
            "embedding_manifest": str(Path(args.embedding_manifest)),
            "embedding_manifest_rows": int(len(embedding_manifest)),
            "embeddings": str(Path(args.embeddings)),
            "expanded_embeddings": str(out_path),
            "expanded_shape": [int(v) for v in expanded.shape],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    features, manifest = _load_eval_features(
        manifest_path=Path(args.manifest),
        embeddings_path=Path(args.embeddings),
        confounder_column=str(args.confounder_column),
    )

    if args.command == "ccmr":
        result = CCMR.compute(
            features=features,
            manifest=manifest,
            confounder_column=str(args.confounder_column),
            evaluation_design=str(args.evaluation_design),
            m=int(args.m),
            alpha=float(args.alpha),
            start_k=int(args.start_k),
            k_growth_factor=float(args.k_growth_factor),
        )
        payload = {
            "dataset": result.dataset,
            "m": result.m,
            "value": result.value,
            "undefined_frac": result.undefined_frac,
            "k_start": result.k_start,
            "k_final": result.k_final,
            "retries": result.retries,
            "alpha": result.alpha,
            "q_alpha": result.q_alpha,
            "ltm_alpha": result.ltm_alpha,
            "evaluation_design": result.evaluation_design,
            "evaluation_unit": result.evaluation_unit,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "ri":
        result = RI.compute(
            features=features,
            manifest=manifest,
            confounder_column=str(args.confounder_column),
            evaluation_design=str(args.evaluation_design),
            k_candidates=args.k_candidates,
        )
    else:
        result = MaRI.compute(
            features=features,
            manifest=manifest,
            confounder_column=str(args.confounder_column),
            evaluation_design=str(args.evaluation_design),
            k_candidates=args.k_candidates,
            tau=float(args.tau),
        )
    payload = _result_payload(result)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
