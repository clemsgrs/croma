"""Basis for ``tab:nipd-auroc`` and its macros: nIPD from AUROC beside balanced accuracy.

``scripts/studies/auc/`` scores every probe of the APD study's sweep with AUROC as well as
balanced accuracy. Two consumers read it and must agree:

* ``generate_nipd_auroc_table.py`` renders ``tab:nipd-auroc``;
* ``generate_paper_values.py`` emits the agreement and in-domain CRoMa rhos as macros.

Both go through :func:`load`, which recomputes the reduction and the correlations from the
AUROC cells themselves (``report.reduce`` then ``report.correlate``; a few seconds) rather
than trusting a CSV that could predate a re-run cell. ``report.reduce`` raises on a
missing or stale cell, so neither consumer can warn and skip.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _paper_tables import CaptionClaimError, bare_num

REPO = Path(__file__).resolve().parents[2]
AUC_STUDY = REPO / "scripts" / "studies" / "auc"
if str(AUC_STUDY) not in sys.path:
    sys.path.insert(0, str(AUC_STUDY))

import report  # noqa: E402

#: The cohort the paper names PCaBiop has no manifest short name (its benchmark is PANDA).
SLIDE_COHORT = "pcabiop"
SLIDE_LABEL = "PCaBiop"
#: p and q values below this print as an inequality. A Monte-Carlo p cannot fall below
#: 1/(permutations + 1), so three decimals is the resolution the tile cohorts support.
P_FLOOR = 0.001


def interval(lo: float, hi: float) -> str:
    """``$[lo, hi]$`` with two decimals, as the other correlation tables print a CI."""
    return rf"$[{bare_num(lo)}, {bare_num(hi)}]$"


def significance(symbol: str, value: float) -> str:
    """``$q{<}0.001$`` below :data:`P_FLOOR`, else ``$q{=}0.299$``; likewise for p."""
    if value < P_FLOOR:
        return rf"${symbol}{{<}}{P_FLOOR:g}$"
    return rf"${symbol}{{=}}{value:.3f}$"


@dataclass(frozen=True)
class NipdAuroc:
    #: One row per cohort x encoder x regime (``report.reduce``).
    reduction: pd.DataFrame
    #: One row per cohort x regime x correlation (``report.correlate``).
    corr: pd.DataFrame

    def cell(self, dataset: str, regime: str, x: str, y: str) -> pd.Series:
        rows = self.corr[
            (self.corr["dataset"] == dataset)
            & (self.corr["regime"] == regime)
            & (self.corr["x"] == x)
            & (self.corr["y"] == y)
        ]
        if len(rows) != 1:
            raise KeyError(f"no single correlation for ({dataset}, {regime}, {x} ~ {y})")
        return rows.iloc[0]

    def _single(self, column: str, mask: pd.Series) -> int:
        values = set(self.corr.loc[mask, column])
        if len(values) != 1:
            raise CaptionClaimError(f"the correlations disagree on {column}: {sorted(values)}")
        return int(values.pop())

    def n(self, dataset: str) -> int:
        """Ranked encoders behind every correlation on ``dataset``."""
        return self._single("n", self.corr["dataset"] == dataset)

    @property
    def tile_datasets(self) -> list[str]:
        return [d for d in report.FAMILY_DATASETS if d in set(self.corr["dataset"])]

    @property
    def n_ranked(self) -> int:
        """The ranked tile encoders per cohort, one count the caption can quote."""
        return self._single("n", self.corr["dataset"].isin(self.tile_datasets))

    @property
    def family_size(self) -> int:
        return int(self.corr["in_family"].sum())

    @property
    def n_boot(self) -> int:
        return self._single("n_boot", self.corr["dataset"].notna())

    @property
    def level(self) -> float:
        levels = set(self.corr["level"])
        if len(levels) != 1:
            raise CaptionClaimError(f"the correlations disagree on CI level: {sorted(levels)}")
        return float(levels.pop())

    def n_permutations(self, kind: str) -> int:
        """Orderings behind each ``"monte-carlo"`` or ``"exact"`` permutation p."""
        return self._single("n_permutations", self.corr["permutation"] == kind)

    def operating_k(self, dataset: str) -> int | None:
        """The run's single RI/MaRI k on ``dataset``; ``None`` where each encoder has its own."""
        ks = set(self.reduction.loc[self.reduction["dataset"] == dataset, "k"])
        return int(ks.pop()) if len(ks) == 1 else None


def from_reduction(reduction: pd.DataFrame, **correlate_kwargs) -> NipdAuroc:
    """Correlate ``reduction`` and check the claims the caption makes about its panel."""
    corr = report.correlate(reduction, **correlate_kwargs)
    study = NipdAuroc(reduction=reduction, corr=corr)
    for dataset in study.tile_datasets:
        cohort = reduction[(reduction["dataset"] == dataset) & (reduction["regime"] == "id")]
        if int(cohort["control"].sum()) != 1:
            raise CaptionClaimError(
                f"the caption says the natural-image control is excluded, but {dataset} "
                f"carries {int(cohort['control'].sum())} control rows, not one"
            )
        if study.n(dataset) != len(cohort) - 1:
            raise CaptionClaimError(f"{dataset}'s correlations do not span its ranked encoders")
    counts = {dataset: study.n(dataset) for dataset in study.tile_datasets}
    if len(set(counts.values())) != 1:
        raise CaptionClaimError(f"the tile cohorts rank different panels: {counts}")
    return study


def load(root: Path = REPO) -> NipdAuroc:
    """The study recomputed from the AUROC cells under ``root``; raises rather than skips."""
    return from_reduction(report.reduce(Path(root)))
