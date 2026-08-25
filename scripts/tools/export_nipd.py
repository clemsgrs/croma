#!/usr/bin/env python3
"""Export the manuscript nIPD study as one committed publication artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

SCHEMA_VERSION = 1
STUDY_REVISION = "expanded-panel-paired-repeat-v1"
MAX_PAYLOAD_BYTES = 300_000
CONTROL = "DINOv2-B"
TILE_MODELS = (
    "CONCH",
    "CONCHv1.5",
    "DINOv2-B",
    "GPFM",
    "GenBio-PathFM",
    "H-optimus-0",
    "H-optimus-1",
    "H0-mini",
    "Hibou-B",
    "Hibou-L",
    "MUSK",
    "Mascaret",
    "Midnight-12k",
    "Phaet",
    "Phikon",
    "Phikon-v2",
    "Prost40M",
    "Prov-GigaPath",
    "RudolfV 2",
    "RudolfV 2-B",
    "RudolfV 2-S",
    "UNI",
    "UNI2-h",
    "Virchow",
    "Virchow2",
    "mSTAR",
)
SLIDE_MODELS = ("MOOZY", "PRISM", "PRISM2", "Prov-GigaPath", "TITAN")
COHORTS = (
    ("camelyon", "Camelyon", 0.5, tuple(i / 7 for i in range(8)), "tile", TILE_MODELS),
    (
        "tcga-4x4",
        "TCGA-4×4",
        0.25,
        (
            0.0,
            0.2041241452319315,
            0.3535533905932738,
            0.5,
            0.67700320038633,
            0.8416254115301732,
            1.0,
        ),
        "tile",
        TILE_MODELS,
    ),
    ("tolkach-esca", "Tolkach-ESCA", 1 / 6, (0.0, 1 / 3, 2 / 3, 1.0), "tile", TILE_MODELS),
    ("pcabiop", "PCaBiop", 0.5, tuple(i / 10 for i in range(11)), "slide", SLIDE_MODELS),
)
SOURCE_NAMES = {
    "camelyon": "camelyon",
    "tcga-4x4": "tcga_4x4",
    "tolkach-esca": "tolkach",
    "pcabiop": "pcabiop",
}
FLOAT_DIGITS = 10


def _round(value: float) -> float:
    return round(float(value), FLOAT_DIGITS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trajectory(accuracies: np.ndarray, chance: float) -> dict:
    baseline = float(accuracies[0].mean())
    skill = baseline - chance
    if skill <= 0:
        raise ValueError(f"baseline balanced accuracy {baseline} does not exceed chance {chance}")
    paired = (accuracies - accuracies[0]) / skill
    mean = paired.mean(axis=1)
    n = paired.shape[1]
    half = student_t.ppf(0.975, df=n - 1) * paired.std(axis=1, ddof=1) / math.sqrt(n)
    return {
        "baseline_balanced_accuracy": _round(baseline),
        "baseline_skill": _round(skill),
        "mean_normalized_trajectory": [_round(value) for value in mean],
        "ci95_low": [_round(value) for value in mean - half],
        "ci95_high": [_round(value) for value in mean + half],
    }


def build_payload(source: Path) -> dict:
    summary_path = source / "apd.csv"
    joined_path = source / "apd_metrics_joined.csv"
    summary = pd.read_csv(summary_path)
    joined = pd.read_csv(joined_path)
    cohorts = []
    raw_hash = hashlib.sha256()
    for slug, label, chance, grid, panel, expected_models in COHORTS:
        source_name = SOURCE_NAMES[slug]
        rows = joined[joined["dataset"] == source_name].set_index("model")
        if set(rows.index) != set(expected_models):
            raise ValueError(f"{slug} roster differs from the approved publication roster")
        models = []
        for model in expected_models:
            raw_path = source / source_name / f"{model}.json"
            raw_bytes = raw_path.read_bytes()
            raw_hash.update(f"{source_name}/{model}.json\0".encode())
            raw_hash.update(raw_bytes)
            raw = json.loads(raw_bytes)
            regimes = {}
            for regime in ("id", "ood"):
                accuracies = np.asarray(raw[f"{regime}_test_accuracies"], dtype=float)
                result = _trajectory(accuracies, chance)
                result["nipd"] = _round(np.trapz(result["mean_normalized_trajectory"], grid))
                reported = float(rows.loc[model, f"nipd_{regime}"])
                if not math.isclose(result["nipd"], reported, abs_tol=2e-9):
                    raise ValueError(f"{slug}/{model}/{regime} nIPD differs from study summary")
                regimes[regime] = result
            is_control = model == CONTROL
            models.append(
                {
                    "model": model,
                    "panel": panel,
                    "ranked": not is_control,
                    "is_control": is_control,
                    "croma_median_m5": _round(rows.loc[model, "croma"]),
                    "regimes": regimes,
                }
            )
        cohorts.append(
            {
                "slug": slug,
                "label": label,
                "chance": _round(chance),
                "cramers_v": [_round(value) for value in grid],
                "models": models,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "study_revision": STUDY_REVISION,
            "source": "output/studies/apd",
            "apd_summary_sha256": _sha256(summary_path),
            "joined_summary_sha256": _sha256(joined_path),
            "raw_cells_sha256": raw_hash.hexdigest(),
            "repeats": 20,
            "croma_radius_m": 5,
            "interval": "paired-repeat Student-t 95%",
        },
        "cohorts": cohorts,
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: dict) -> None:
    try:
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
    except ValueError as error:
        raise ValueError("payload values must be finite") from error
    if len(encoded) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"oversized payload: {len(encoded)} bytes")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    provenance = payload.get("provenance", {})
    if provenance.get("study_revision") != STUDY_REVISION:
        raise ValueError("stale provenance: study revision does not match")
    for key in ("apd_summary_sha256", "joined_summary_sha256", "raw_cells_sha256"):
        value = provenance.get(key, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"stale provenance: malformed {key}")
    observed_slugs = [cohort.get("slug") for cohort in payload.get("cohorts", [])]
    expected_slugs = [cohort[0] for cohort in COHORTS]
    if observed_slugs != expected_slugs:
        raise ValueError(f"cohorts must be exactly {expected_slugs}, got {observed_slugs}")

    for cohort, config in zip(payload["cohorts"], COHORTS):
        slug, _, chance, grid, panel, expected_models = config
        v = cohort.get("cramers_v", [])
        if len(v) != len(grid) or not np.allclose(v, grid, atol=1e-10, rtol=0):
            raise ValueError(f"{slug} Cramér's-V shape/grid drift")
        models = cohort.get("models", [])
        if [model.get("model") for model in models] != list(expected_models):
            raise ValueError(f"{slug} roster drift")
        for model in models:
            expected_control = model["model"] == CONTROL and panel == "tile"
            if model.get("panel") != panel or model.get("is_control") is not expected_control:
                raise ValueError(f"{slug}/{model['model']} control/panel drift")
            if model.get("ranked") is not (not expected_control):
                raise ValueError(f"{slug}/{model['model']} ranked/control drift")
            if not math.isfinite(float(model.get("croma_median_m5", math.nan))):
                raise ValueError(f"{slug}/{model['model']} values must be finite")
            if set(model.get("regimes", {})) != {"id", "ood"}:
                raise ValueError(f"{slug}/{model['model']} malformed regimes")
            for regime, result in model["regimes"].items():
                scalar_keys = ("nipd", "baseline_balanced_accuracy", "baseline_skill")
                if not all(math.isfinite(float(result.get(key, math.nan))) for key in scalar_keys):
                    raise ValueError(f"{slug}/{model['model']}/{regime} values must be finite")
                baseline = float(result["baseline_balanced_accuracy"])
                skill = float(result["baseline_skill"])
                if baseline <= chance or not math.isclose(skill, baseline - chance, abs_tol=2e-10):
                    raise ValueError(
                        f"{slug}/{model['model']}/{regime} baseline skill/domain drift"
                    )
                arrays = [
                    result.get(key, [])
                    for key in ("mean_normalized_trajectory", "ci95_low", "ci95_high")
                ]
                if any(len(values) != len(v) for values in arrays):
                    raise ValueError(f"{slug}/{model['model']}/{regime} malformed trajectory shape")
                if not all(math.isfinite(float(value)) for values in arrays for value in values):
                    raise ValueError(f"{slug}/{model['model']}/{regime} values must be finite")
                if any(lo > mean or mean > hi for mean, lo, hi in zip(*arrays)):
                    raise ValueError(f"{slug}/{model['model']}/{regime} malformed interval bounds")
                area = float(np.trapz(arrays[0], v))
                if not math.isclose(area, float(result["nipd"]), abs_tol=2e-9):
                    raise ValueError(f"{slug}/{model['model']}/{regime} nIPD/trajectory drift")


def summary_csv(payload: dict) -> str:
    fields = [
        "cohort",
        "regime",
        "model",
        "panel",
        "ranked",
        "is_control",
        "croma_median_m5",
        "chance",
        "nipd",
        "baseline_balanced_accuracy",
        "baseline_skill",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for cohort in payload["cohorts"]:
        for model in cohort["models"]:
            for regime in ("id", "ood"):
                result = model["regimes"][regime]
                writer.writerow(
                    {
                        "cohort": cohort["slug"],
                        "regime": regime,
                        "model": model["model"],
                        "panel": model["panel"],
                        "ranked": model["ranked"],
                        "is_control": model["is_control"],
                        "croma_median_m5": model["croma_median_m5"],
                        "chance": cohort["chance"],
                        "nipd": result["nipd"],
                        "baseline_balanced_accuracy": result["baseline_balanced_accuracy"],
                        "baseline_skill": result["baseline_skill"],
                    }
                )
    return stream.getvalue()


def export(source: Path, destination: Path) -> None:
    payload = build_payload(source)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "nipd.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (destination / "nipd.csv").write_text(summary_csv(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/studies/apd"))
    parser.add_argument("--destination", type=Path, default=Path("results"))
    args = parser.parse_args()
    export(args.source, args.destination)


if __name__ == "__main__":
    main()
