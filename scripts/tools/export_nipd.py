#!/usr/bin/env python3
"""Export the manuscript nIPD study as one committed publication artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
LOCAL_SRC = (ROOT / "src").resolve()
# Test modules may already have imported a globally installed croma. Publication must
# never inherit that mutable interpreter state: evict only a foreign package, then put
# this checkout first. A normal CLI process has nothing to evict.
cached_croma = sys.modules.get("croma")
if cached_croma is not None and not Path(cached_croma.__file__).resolve().is_relative_to(LOCAL_SRC):
    for module_name in [
        name for name in sys.modules if name == "croma" or name.startswith("croma.")
    ]:
        del sys.modules[module_name]
sys.path.insert(0, str(LOCAL_SRC))
from croma import nipd  # noqa: E402


def _project_version() -> str:
    match = re.search(
        r'(?ms)^\[project\]\s*$.*?^version\s*=\s*"([^"]+)"',
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    )
    if match is None:
        raise ValueError("pyproject.toml has no [project] version")
    return match.group(1)


SCHEMA_VERSION = 2
STUDY_REVISION = "expanded-panel-paired-repeat-v1"
PUBLICATION_DATE = "2026-08-25"
CROMA_VERSION = _project_version()
MAX_PAYLOAD_BYTES = 300_000
CONTROL = "DINOv2-B"
EXPECTED_SOURCE_PROVENANCE = {
    "apd_summary_sha256": "76d4cc240ea20f772da1c8b293549e7af5d12140dede33fbb1d55ab033d4a25a",
    "joined_summary_sha256": "aec4dc6b4f4ff3fa2b91441ea7edf85f34564a6f43134360304118c719c965ba",
    "raw_cells_sha256": "5429312ba08f4934b958429ade68d8f6b97025ad23d7a52468627515b2abf0da",
}
FLOAT_DIGITS = 10


@dataclass(frozen=True)
class Cohort:
    slug: str
    source_name: str
    label: str
    chance: float
    cramers_v: tuple[float, ...]
    panel: str
    association_descriptive: bool = False


COHORTS = (
    Cohort("camelyon", "camelyon", "Camelyon", 0.5, tuple(i / 7 for i in range(8)), "tile"),
    Cohort(
        "tcga-4x4",
        "tcga_4x4",
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
    ),
    Cohort("tolkach-esca", "tolkach", "Tolkach-ESCA", 1 / 6, (0.0, 1 / 3, 2 / 3, 1.0), "tile"),
    Cohort(
        "pcabiop",
        "pcabiop",
        "PCaBiop",
        0.5,
        tuple(i / 10 for i in range(11)),
        "slide",
        association_descriptive=True,
    ),
)


def _round(value: float) -> float:
    return round(float(value), FLOAT_DIGITS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_metadata(path: Path = ROOT / "scripts" / "bench" / "model_metadata.csv") -> dict:
    metadata = pd.read_csv(path).fillna("")
    panels = {}
    for panel in ("tile", "slide"):
        rows = metadata[metadata["panel"] == panel].sort_values("panel_order")
        panels[panel] = [
            {
                "registry_model": str(row["model"]),
                "model": str(row["published_name"] or row["model"]),
            }
            for _, row in rows.iterrows()
        ]
    return panels


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
        "mean_normalized_trajectory": [_round(value) for value in mean],
        "ci95_low": [_round(value) for value in mean - half],
        "ci95_high": [_round(value) for value in mean + half],
    }


def _association(models: list[dict], *, descriptive: bool) -> dict:
    """Derive the one ranked-panel association float basis used by every site view."""
    ranked = [model for model in models if model["ranked"]]
    regimes = {}
    for regime in ("id", "ood"):
        xs = np.asarray([model["croma_median_m5"] for model in ranked], dtype=float)
        ys = np.asarray([model["regimes"][regime]["nipd"] for model in ranked], dtype=float)
        rho = float(spearmanr(xs, ys).statistic)
        trend = None
        if not descriptive:
            fit = linregress(xs, ys)
            trend = {
                "slope": _round(fit.slope),
                "intercept": _round(fit.intercept),
                "x_min": _round(xs.min()),
                "x_max": _round(xs.max()),
            }
        regimes[regime] = {
            "n": len(ranked),
            "spearman_rho": _round(rho),
            "trend": trend,
        }
    return {"descriptive": descriptive, "regimes": regimes}


def build_payload(source: Path) -> dict:
    summary_path = source / "apd.csv"
    joined_path = source / "apd_metrics_joined.csv"
    pd.read_csv(summary_path)
    joined = pd.read_csv(joined_path)
    panel_models = _model_metadata()
    cohorts = []
    raw_hash = hashlib.sha256()
    for cohort in COHORTS:
        identities = panel_models[cohort.panel]
        registry_models = [identity["registry_model"] for identity in identities]
        rows = joined[joined["dataset"] == cohort.source_name].set_index("model")
        if set(rows.index) != set(registry_models):
            raise ValueError(f"{cohort.slug} roster differs from canonical model metadata")
        models = []
        for identity in identities:
            registry_model = identity["registry_model"]
            raw_path = source / cohort.source_name / f"{registry_model}.json"
            raw_bytes = raw_path.read_bytes()
            raw_hash.update(f"{cohort.source_name}/{registry_model}.json\0".encode())
            raw_hash.update(raw_bytes)
            raw = json.loads(raw_bytes)
            regimes = {}
            for regime in ("id", "ood"):
                accuracies = np.asarray(raw[f"{regime}_test_accuracies"], dtype=float)
                result = _trajectory(accuracies, cohort.chance)
                result["nipd"] = _round(nipd(accuracies, cohort.cramers_v, cohort.chance))
                reported = float(rows.loc[registry_model, f"nipd_{regime}"])
                if not math.isclose(result["nipd"], reported, abs_tol=2e-9):
                    raise ValueError(
                        f"{cohort.slug}/{identity['model']}/{regime} nIPD differs from study summary"
                    )
                regimes[regime] = result
            is_control = registry_model == CONTROL
            models.append(
                {
                    "model": identity["model"],
                    "registry_model": registry_model,
                    "panel": cohort.panel,
                    "ranked": not is_control,
                    "is_control": is_control,
                    "croma_median_m5": _round(rows.loc[registry_model, "croma"]),
                    "regimes": regimes,
                }
            )
        cohorts.append(
            {
                "slug": cohort.slug,
                "label": cohort.label,
                "chance": _round(cohort.chance),
                "cramers_v": [_round(value) for value in cohort.cramers_v],
                "association": _association(models, descriptive=cohort.association_descriptive),
                "models": models,
            }
        )
    source_provenance = {
        "apd_summary_sha256": _sha256(summary_path),
        "joined_summary_sha256": _sha256(joined_path),
        "raw_cells_sha256": raw_hash.hexdigest(),
    }
    if source_provenance != EXPECTED_SOURCE_PROVENANCE:
        raise ValueError("stale provenance: canonical study inputs changed; review and republish")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "study_revision": STUDY_REVISION,
            "published": PUBLICATION_DATE,
            "croma_version": CROMA_VERSION,
            "source": "output/studies/apd",
            **source_provenance,
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
    if (
        provenance.get("published") != PUBLICATION_DATE
        or provenance.get("croma_version") != CROMA_VERSION
    ):
        raise ValueError("stale provenance: publication identity does not match")
    observed_source_provenance = {key: provenance.get(key) for key in EXPECTED_SOURCE_PROVENANCE}
    if observed_source_provenance != EXPECTED_SOURCE_PROVENANCE:
        raise ValueError("stale provenance: source hashes do not match")
    observed_slugs = [cohort.get("slug") for cohort in payload.get("cohorts", [])]
    expected_slugs = [cohort.slug for cohort in COHORTS]
    if observed_slugs != expected_slugs:
        raise ValueError(f"cohorts must be exactly {expected_slugs}, got {observed_slugs}")

    panel_models = _model_metadata()
    for published_cohort, config in zip(payload["cohorts"], COHORTS):
        v = published_cohort.get("cramers_v", [])
        if len(v) != len(config.cramers_v) or not np.allclose(
            v, config.cramers_v, atol=1e-10, rtol=0
        ):
            raise ValueError(f"{config.slug} Cramér's-V shape/grid drift")
        chance = float(published_cohort.get("chance", math.nan))
        if not math.isclose(chance, config.chance, abs_tol=1e-10):
            raise ValueError(f"{config.slug} chance drift")
        expected_models = panel_models[config.panel]
        models = published_cohort.get("models", [])
        identities = [(model.get("model"), model.get("registry_model")) for model in models]
        expected_identities = [
            (model["model"], model["registry_model"]) for model in expected_models
        ]
        if identities != expected_identities:
            raise ValueError(f"{config.slug} roster drift")
        for model in models:
            expected_control = model["registry_model"] == CONTROL and config.panel == "tile"
            if (
                model.get("panel") != config.panel
                or model.get("is_control") is not expected_control
            ):
                raise ValueError(f"{config.slug}/{model['model']} control/panel drift")
            if model.get("ranked") is not (not expected_control):
                raise ValueError(f"{config.slug}/{model['model']} ranked/control drift")
            if not math.isfinite(float(model.get("croma_median_m5", math.nan))):
                raise ValueError(f"{config.slug}/{model['model']} values must be finite")
            if set(model.get("regimes", {})) != {"id", "ood"}:
                raise ValueError(f"{config.slug}/{model['model']} malformed regimes")
            for regime, result in model["regimes"].items():
                scalar_keys = ("nipd", "baseline_balanced_accuracy")
                if not all(math.isfinite(float(result.get(key, math.nan))) for key in scalar_keys):
                    raise ValueError(
                        f"{config.slug}/{model['model']}/{regime} values must be finite"
                    )
                if float(result["baseline_balanced_accuracy"]) <= chance:
                    raise ValueError(
                        f"{config.slug}/{model['model']}/{regime} baseline/domain drift"
                    )
                arrays = [
                    result.get(key, [])
                    for key in ("mean_normalized_trajectory", "ci95_low", "ci95_high")
                ]
                if any(len(values) != len(v) for values in arrays):
                    raise ValueError(
                        f"{config.slug}/{model['model']}/{regime} malformed trajectory shape"
                    )
                if not all(math.isfinite(float(value)) for values in arrays for value in values):
                    raise ValueError(
                        f"{config.slug}/{model['model']}/{regime} values must be finite"
                    )
                if any(lo > mean or mean > hi for mean, lo, hi in zip(*arrays)):
                    raise ValueError(
                        f"{config.slug}/{model['model']}/{regime} malformed interval bounds"
                    )
                # Spell out the trapezoidal rule used by ``croma.nipd``. NumPy 2
                # removed ``np.trapz``, while ``np.trapezoid`` is absent from the
                # oldest supported NumPy; this expression works across both and keeps
                # validation independent of the reduction call that produced nIPD.
                trajectory = np.asarray(arrays[0], dtype=float)
                coordinates = np.asarray(v, dtype=float)
                area = float(
                    np.sum(np.diff(coordinates) * (trajectory[:-1] + trajectory[1:]) / 2.0)
                )
                if not math.isclose(area, float(result["nipd"]), abs_tol=2e-9):
                    raise ValueError(
                        f"{config.slug}/{model['model']}/{regime} nIPD/trajectory drift"
                    )
        expected_association = _association(models, descriptive=config.association_descriptive)
        if published_cohort.get("association") != expected_association:
            raise ValueError(f"{config.slug} association metadata drift")


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
        "normalized_change_at_max_v",
        "baseline_balanced_accuracy",
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
                        "normalized_change_at_max_v": result["mean_normalized_trajectory"][-1],
                        "baseline_balanced_accuracy": result["baseline_balanced_accuracy"],
                    }
                )
    return stream.getvalue()


def export(source: Path, destination: Path) -> None:
    payload = build_payload(source)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "nipd.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (destination / "nipd.csv").write_text(summary_csv(payload), encoding="utf-8")


def check_committed(destination: Path) -> None:
    payload_path = destination / "nipd.json"
    summary_path = destination / "nipd.csv"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    validate_payload(payload)
    if summary_path.read_text(encoding="utf-8") != summary_csv(payload):
        raise ValueError("stale publication: nipd.csv differs from the committed JSON float basis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("output/studies/apd"))
    parser.add_argument("--destination", type=Path, default=Path("results"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate committed provenance and prove the summary CSV is fresh from the JSON",
    )
    args = parser.parse_args()
    if args.check:
        check_committed(args.destination)
    else:
        export(args.source, args.destination)


if __name__ == "__main__":
    main()
