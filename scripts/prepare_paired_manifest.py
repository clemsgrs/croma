import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from croma.metrics.pairs import load_manifest, validate_subset_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a balanced manifest into all manifest-defined paired_2x2 "
            "subsets over label and confounder pairs."
        )
    )
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to input manifest CSV."
    )
    parser.add_argument(
        "--confounder-column",
        required=True,
        help="Manifest column to treat as the non-biological confounder.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated subset of labels to use when forming label pairs.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to the output paired-manifest CSV.",
    )
    return parser.parse_args()


def _parse_label_subset(raw_labels: str) -> list[str]:
    if not str(raw_labels).strip():
        return []
    labels = [str(value).strip() for value in str(raw_labels).split(",")]
    if any(not value for value in labels):
        raise ValueError("labels list contains an empty entry")
    deduped = list(dict.fromkeys(labels))
    if len(deduped) != len(labels):
        raise ValueError("labels list contains duplicate entries")
    return deduped


def _sanitize_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "value"


def _paired_subset_id(
    label_pair: tuple[str, str],
    confounder_pair: tuple[str, str],
    *,
    total_label_count: int,
) -> str:
    confounder_token = "_".join(_sanitize_token(part) for part in confounder_pair)
    if int(total_label_count) == 2:
        return confounder_token
    label_token = "+".join(_sanitize_token(part) for part in label_pair)
    return f"{label_token}__{confounder_token}"


def _strata_summary(df: pd.DataFrame) -> list[dict[str, object]]:
    grouped = (
        df.groupby(["label", "confounder"], sort=True, dropna=False)
        .size()
        .reset_index(name="n_samples")
    )
    return [
        {
            "label": str(row["label"]),
            "confounder": str(row["confounder"]),
            "n_samples": int(row["n_samples"]),
        }
        for _, row in grouped.iterrows()
    ]


def _is_globally_balanced(strata: list[dict[str, object]]) -> bool:
    counts = {int(row["n_samples"]) for row in strata}
    return len(counts) <= 1


def prepare_paired_manifest(
    *,
    manifest_path: Path,
    confounder_column: str,
    out_path: Path,
    labels: list[str] | None = None,
) -> dict:
    input_columns = pd.read_csv(manifest_path, nrows=0).columns.tolist()
    manifest = load_manifest(str(manifest_path), confounder_column=confounder_column)

    labels_all = sorted(manifest["label"].astype(str).unique().tolist())
    confounders = sorted(manifest["confounder"].astype(str).unique().tolist())
    selected_labels = list(labels) if labels else labels_all
    selected_labels = [str(value).strip() for value in selected_labels]
    missing_labels = [value for value in selected_labels if value not in labels_all]
    if missing_labels:
        raise ValueError(f"Requested labels are missing from the manifest: {missing_labels}")
    if len(selected_labels) < 2:
        raise ValueError("paired manifest generation requires at least 2 labels")
    if len(confounders) < 2:
        raise ValueError("paired manifest generation requires at least 2 confounder values")

    working = manifest.loc[manifest["label"].astype(str).isin(selected_labels)].copy()
    strata = _strata_summary(working)

    expanded_frames: list[pd.DataFrame] = []
    for label_pair in combinations(sorted(selected_labels), 2):
        for confounder_pair in combinations(confounders, 2):
            subset_df = working.loc[
                working["label"].astype(str).isin(label_pair)
                & working["confounder"].astype(str).isin(confounder_pair)
            ].copy()
            if len(subset_df) == 0:
                continue
            counts = (
                subset_df.groupby(["label", "confounder"], sort=True, dropna=False)
                .size()
                .reset_index(name="n_samples")
            )
            if len(counts) != 4 or int((counts["n_samples"] <= 0).sum()) > 0:
                continue
            subset_df["subset"] = _paired_subset_id(
                label_pair,
                confounder_pair,
                total_label_count=len(selected_labels),
            )
            expanded_frames.append(subset_df)

    if not expanded_frames:
        raise ValueError("could not construct any complete 2x2 paired subsets")

    out_df = pd.concat(expanded_frames, ignore_index=True)
    output_columns = [str(col) for col in input_columns if str(col) != "subset"]
    output_columns.append("subset")
    out_df = out_df.loc[:, output_columns].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    validated = load_manifest(str(out_path), confounder_column=confounder_column)
    validate_subset_manifest(validated, str(out_path))

    subset_sizes = (
        out_df.groupby("subset", sort=True, dropna=False).size().reset_index(name="n_rows")
    )
    summary = {
        "manifest": str(manifest_path),
        "paired_manifest": str(out_path),
        "input_rows": int(len(manifest)),
        "filtered_rows": int(len(working)),
        "paired_rows": int(len(out_df)),
        "unique_samples": int(out_df["sample_id"].nunique()),
        "labels_used": sorted(selected_labels),
        "confounders_used": confounders,
        "label_pair_count": int(len(list(combinations(sorted(selected_labels), 2)))),
        "confounder_pair_count": int(len(list(combinations(confounders, 2)))),
        "subset_count": int(out_df["subset"].nunique()),
        "subset_sizes": [
            {"subset": str(row["subset"]), "n_rows": int(row["n_rows"])}
            for _, row in subset_sizes.iterrows()
        ],
        "strata": strata,
        "is_globally_balanced": _is_globally_balanced(strata),
    }
    return summary


def main() -> int:
    args = _parse_args()
    summary = prepare_paired_manifest(
        manifest_path=Path(args.manifest),
        confounder_column=str(args.confounder_column),
        out_path=Path(args.out),
        labels=_parse_label_subset(str(args.labels)),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
