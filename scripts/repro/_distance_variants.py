"""The distance-variants study as the paper reads it: measures, summary and provenance.

``scripts/studies/distance_variants.py`` computes everything; this module only loads what it
wrote, for ``tab:distance-variants`` and its macros. It never warns and skips. A missing or
incomplete study raises ``FileNotFoundError``, and a study that no longer describes the
files on disk raises :class:`CaptionClaimError`: an output changed after the study wrote it,
or an input it was computed from (a run's ``metrics.csv`` or ``run_config.json``, the APD
study's ``apd.csv`` or ``apd_correlation.csv``) changed after it was summarised.

The embeddings and the runs' per-sample files are fingerprinted too, when each cell is
scored, but are not re-hashed on every build: they run to gigabytes, and a re-scored run
rewrites its ``metrics.csv``, which is re-hashed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (HERE, REPO / "scripts" / "studies"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import distance_variants as study  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402

STUDY_DIR = study.STUDY_DIR
REMEDY = (
    "run scripts/studies/distance_variants.py sweep --workers 4 (resumable; it summarises "
    "when complete), or its summarise command after an input changed"
)
#: Recorded inputs too large to re-hash on every build (see the module docstring).
_NOT_REHASHED = ("embeddings:", "/per_sample_metrics_by_model/")

#: The tile benchmarks in the order the paper presents them.
BENCHMARK_ORDER = (
    "pathorob-camelyon",
    "pathorob-tcga-2x2",
    "pathorob-tcga-4x4",
    "pathorob-tolkach-esca",
)
#: The headline variants compared with cosine, in row order.
ALTERNATIVES = ("normalised-euclidean", "euclidean", "whitened")
#: Camelyon's whitening-threshold sensitivity, reported in the footer.
SENSITIVITY = ("whitened-90", "whitened-99")


@dataclass(frozen=True)
class DistanceVariants:
    measures: pd.DataFrame
    summary: pd.DataFrame
    provenance: dict

    def row(self, benchmark: str, variant: str) -> pd.Series:
        hit = self.measures[
            (self.measures["benchmark"] == benchmark) & (self.measures["variant"] == variant)
        ]
        if len(hit) != 1:
            raise CaptionClaimError(f"measures.csv has {len(hit)} rows for {benchmark}/{variant}")
        return hit.iloc[0]

    def nipd_benchmarks(self) -> list[str]:
        return [b for b in BENCHMARK_ORDER if b in study.NIPD_DATASETS]


def _check_digest(path: Path, digest: str, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} is absent ({what}) -- {REMEDY}")
    if study.sha256(path) != digest:
        raise CaptionClaimError(
            f"{path} no longer matches the sha256 the distance-variants study recorded "
            f"({what}) -- {REMEDY}"
        )


def load(root: Path = STUDY_DIR, *, repo: Path = REPO) -> DistanceVariants:
    """The complete study at ``root``, verified against its own provenance."""
    root, repo = Path(root), Path(repo)
    provenance_path = root / study.PROVENANCE
    if not provenance_path.exists():
        raise FileNotFoundError(f"{provenance_path} is absent -- {REMEDY}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _check_digest(root / study.DESIGN, provenance["design_sha256"], "its design")
    for name, digest in provenance["outputs"].items():
        _check_digest(root / name, digest, "its output")
    for key, digest in provenance["inputs"].items():
        if any(marker in key for marker in _NOT_REHASHED):
            continue
        _check_digest(repo / key, digest, "an input it was computed from")

    design = json.loads((root / study.DESIGN).read_text(encoding="utf-8"))
    measures = pd.read_csv(root / study.MEASURES, float_precision="round_trip")
    summary = pd.read_csv(root / study.SUMMARY, float_precision="round_trip")
    expected = {
        (b, v["name"]) for b, spec in design["benchmarks"].items() for v in spec["variants"]
    }
    missing = sorted(expected - set(zip(measures["benchmark"], measures["variant"])))
    if missing:
        raise CaptionClaimError(f"measures.csv lacks {missing} -- {REMEDY}")
    for benchmark in BENCHMARK_ORDER:
        names = {v["name"] for v in design["benchmarks"].get(benchmark, {}).get("variants", [])}
        if not {"cosine", *ALTERNATIVES} <= names:
            raise CaptionClaimError(f"the study does not cover {benchmark} with every variant")
    if not set(SENSITIVITY) <= {
        v["name"] for v in design["benchmarks"]["pathorob-camelyon"]["variants"]
    }:
        raise CaptionClaimError("the study lacks Camelyon's whitening sensitivity")
    for benchmark, spec in design["benchmarks"].items():
        if "results_dir" in spec:
            _check_cosine_matches_run(summary, benchmark, repo / spec["results_dir"])
    return DistanceVariants(measures=measures, summary=summary, provenance=provenance)


#: The run's pooled columns the cosine cells must equal, exactly.
_RUN_COLUMNS = {"median": "croma", "f0": "croma_f0"}


def _check_cosine_matches_run(summary: pd.DataFrame, benchmark: str, results_dir: Path) -> None:
    """Cosine's pooled median and F(0) per encoder are the run's, or the cells are stale.

    The study scores a cell once and keeps it, so a run re-scored after its cells were
    computed but before the summary would otherwise pass every digest check: the summary
    re-hashes the new ``metrics.csv``. Comparing the numbers themselves closes that gap.
    """
    metrics = pd.read_csv(results_dir / "metrics.csv", float_precision="round_trip")
    run = metrics.set_index("model")
    cosine = summary[(summary["benchmark"] == benchmark) & (summary["variant"] == "cosine")]
    cosine = cosine.set_index("model")
    if sorted(cosine.index) != sorted(run.index):
        raise CaptionClaimError(
            f"{benchmark}: the study scored {sorted(cosine.index)} but the run holds "
            f"{sorted(run.index)} -- {REMEDY}"
        )
    for ours, theirs in _RUN_COLUMNS.items():
        differ = [m for m in run.index if float(cosine.loc[m, ours]) != float(run.loc[m, theirs])]
        if differ:
            raise CaptionClaimError(
                f"{benchmark}: cosine {ours} differs from the run's {theirs} for {differ}; "
                f"the run was re-scored after the study -- {REMEDY}"
            )
