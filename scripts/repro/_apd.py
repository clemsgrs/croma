"""Shared basis for downstream-correlation manuscript artifacts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "studies" / "apd"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _cross_benchmark import short_label  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402,F401
from loaders import DATASETS  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "bench"))

CORRELATION_CSV = REPO / "output/studies/apd/apd_correlation.csv"
JOINED_CSV = REPO / "output/studies/apd/apd_metrics_joined.csv"

# Figures and the correlation table use the three 20-model PathoROB benchmarks.
# PCaBiop's four-model correlations remain available for supplementary prose.
FIGURE_DATASETS = ["camelyon", "tcga_4x4", "tolkach"]
TARGETS = [("nipd_id", "in-domain"), ("nipd_ood", "out-of-domain")]
METRICS = ["croma", "ri", "mari"]


@dataclass(frozen=True)
class Apd:
    """Read-only view over the per-benchmark downstream correlations."""

    corr: pd.DataFrame

    def rho(self, target: str, metric: str, scope: str) -> float:
        row = self.corr[
            (self.corr["target"] == target)
            & (self.corr["metric"] == metric)
            & (self.corr["scope"] == scope)
        ]
        if row.empty:
            raise KeyError(f"no rho for ({target}, {metric}, {scope}) in {CORRELATION_CSV}")
        return float(row["spearman"].iloc[0])

    def n(self, scope: str) -> int:
        rows = self.corr[self.corr["scope"] == scope]
        if rows.empty:
            raise KeyError(f"scope {scope!r} absent from {CORRELATION_CSV}")
        counts = set(rows["n"])
        if len(counts) != 1:
            raise CaptionClaimError(
                f"scope {scope!r} was computed over differing model counts {sorted(counts)}"
            )
        return int(counts.pop())

    @property
    def n_models(self) -> int:
        """Number of ranked encoders in each tile benchmark."""
        counts = {dataset: self.n(dataset) for dataset in FIGURE_DATASETS}
        if len(set(counts.values())) != 1:
            raise CaptionClaimError(f"tile benchmarks use different model rosters: {counts}")
        return next(iter(counts.values()))

    def assert_control_excluded(self) -> None:
        """Prove the natural-image control is present in the join but absent from ranks."""
        from plotting.style import CONTROL_MODEL

        if not JOINED_CSV.exists():
            raise FileNotFoundError(
                f"{JOINED_CSV} is absent; run scripts/studies/apd/apd_croma_correlation.py."
            )
        joined = pd.read_csv(JOINED_CSV, usecols=["dataset", "model"])
        for dataset in FIGURE_DATASETS:
            roster = set(joined.loc[joined["dataset"] == dataset, "model"])
            if CONTROL_MODEL not in roster:
                raise CaptionClaimError(
                    f"{CONTROL_MODEL} is missing from the {dataset} join, so exclusion "
                    "cannot be verified"
                )
            if self.n(dataset) != len(roster) - 1:
                raise CaptionClaimError(
                    f"the caption says the natural-image control is excluded, but {dataset}'s "
                    f"rank correlation ran over {self.n(dataset)} of {len(roster)} joined models"
                )

    def benchmark_labels(self) -> list[str]:
        return [by_benchmark(DATASETS[dataset]["benchmark"]).short_name
                for dataset in FIGURE_DATASETS]

    def operating_points(self) -> list[tuple[str, int]]:
        """Operating point for RI/MaRI on each tile benchmark."""
        out = []
        for dataset in FIGURE_DATASETS:
            entry = by_benchmark(DATASETS[dataset]["benchmark"])
            ks = set(pd.read_csv(entry.metrics_rel)["k"])
            if len(ks) != 1:
                raise CaptionClaimError(
                    f"{entry.benchmark} has no single operating point (k in {sorted(ks)})"
                )
            out.append((short_label(entry.benchmark), int(ks.pop())))
        return out


def load() -> Apd:
    if not CORRELATION_CSV.exists():
        raise FileNotFoundError(
            f"{CORRELATION_CSV} is absent. Run the downstream study and correlation script."
        )
    return Apd(corr=pd.read_csv(CORRELATION_CSV))
