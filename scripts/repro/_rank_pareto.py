"""The rank-aggregate Pareto data for the three tile benchmarks.

The per-benchmark Pareto figures (``fig:croma-pareto`` and its two supplementary panels) plot
median \\code{CRoMa} against tail severity ``LTM_10`` one benchmark at a time. This module
collapses the three tile benchmarks into a single overview by *rank*: each pathology encoder is
ranked within every benchmark by median \\code{CRoMa} (rank 1 = highest median) and by ``LTM_10``
(rank 1 = mildest tail), and the two axes are the mean of those ranks across benchmarks.

The panel, the median ranks and the TCGA-exposed (dagger) set all come from
``_cross_benchmark.load()``, so this overview and the rank-trajectory figure (``fig:cross-benchmark``)
cannot disagree about which encoders competed or which are exposed. Mirrors ``_distributions.py``
and ``_cross_benchmark.py``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _cross_benchmark import PANEL, load as load_cross, short_label  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402


def _tail_rank_matrix(models: pd.Index) -> pd.DataFrame:
    """Per-(model, benchmark) rank by ``LTM_10``, 1 = mildest tail (the largest, least-negative LTM).

    Read from the same ``metrics.csv`` the medians come from, restricted to and ordered by the
    ranked roster ``models`` (the control is already dropped there), so this rank matrix aligns
    model-for-model with the median-rank matrix. The run directory is resolved through
    ``paper_manifest`` -- never a spelled-out protocol.
    """
    cols = {}
    for benchmark in PANEL:
        series = (
            pd.read_csv(REPO / by_benchmark(benchmark).metrics_rel)
            .set_index("model")["croma_ltm_alpha"]
        )
        cols[short_label(benchmark)] = series
    ltm = pd.DataFrame(cols).reindex(models)
    if ltm.isna().any().any():
        missing = ltm.index[ltm.isna().any(axis=1)].tolist()
        raise ValueError(
            f"missing LTM_10 for {missing} on one of {list(ltm.columns)}; the tail-rank matrix "
            f"would not align with the median-rank matrix."
        )
    return ltm.rank(ascending=False, method="first").astype(int)


@dataclass(frozen=True)
class RankPareto:
    """The three tile benchmarks aggregated by rank."""

    #: model x benchmark, signed \code{CRoMa} margin at the headline m (control dropped).
    medians: pd.DataFrame
    #: model x benchmark, 1 = highest median.
    median_ranks: pd.DataFrame
    #: model x benchmark, 1 = mildest tail (largest LTM_10).
    tail_ranks: pd.DataFrame
    #: TCGA-exposed encoders, per model_metadata.csv, intersected with the roster.
    exposed: frozenset[str]
    #: The benchmark designated adversarial by the cross-benchmark analysis.
    adversarial: str

    @property
    def models(self) -> list[str]:
        return list(self.medians.index)

    @property
    def labels(self) -> list[str]:
        return list(self.medians.columns)

    @property
    def n_models(self) -> int:
        return len(self.medians)

    @property
    def n_benchmarks(self) -> int:
        return self.medians.shape[1]

    @property
    def mean_median_rank(self) -> pd.Series:
        return self.median_ranks.mean(axis=1)

    @property
    def mean_tail_rank(self) -> pd.Series:
        return self.tail_ranks.mean(axis=1)

    @property
    def frontier(self) -> tuple[str, ...]:
        """Encoders on the mean-rank Pareto frontier: no other encoder has a lower (better) mean
        rank on *both* axes.

        Computed through the plotting library's ``_pareto_frontier_max_max`` on negated ranks
        (fewer is better, so a larger negated rank is better) -- the same primitive the figure
        rings, so the caption and the figure cannot disagree about which encoders are undominated.
        Imported lazily so this module stays free of the matplotlib stack. Returned in the
        frontier-staircase order that primitive produces.
        """
        bench = str(REPO / "scripts" / "bench")
        if bench not in sys.path:
            sys.path.insert(0, bench)
        from plotting import _pareto_frontier_max_max  # noqa: E402 (lazy: keeps import light)

        mmr, mtr = self.mean_median_rank, self.mean_tail_rank
        points = [(m, -float(mmr[m]), -float(mtr[m])) for m in self.models]
        return tuple(_pareto_frontier_max_max(points))

    def rows(self) -> list[dict]:
        """One dict per encoder for the figure: the two mean ranks plus the exposed flag."""
        mmr, mtr = self.mean_median_rank, self.mean_tail_rank
        return [
            {
                "model": m,
                "median_rank": float(mmr[m]),
                "tail_rank": float(mtr[m]),
                "exposed": m in self.exposed,
            }
            for m in self.models
        ]


def load() -> RankPareto:
    """Aggregate the three tile benchmarks by rank.

    Reuses ``_cross_benchmark.load()`` for the median margins, the median ranks and the exposed
    set (so the roster and the dagger set match ``fig:cross-benchmark`` exactly), then adds the
    tail-rank matrix from the same runs.
    """
    cb = load_cross()
    return RankPareto(
        medians=cb.croma,
        median_ranks=cb.ranks,
        tail_ranks=_tail_rank_matrix(cb.croma.index),
        exposed=cb.exposed,
        adversarial=cb.adversarial,
    )


def assert_exposure_marked(rp: RankPareto) -> frozenset[str]:
    """The dagger legend marks the TCGA-exposed encoders, so the set must be non-empty and drawn
    from the ranked roster.
    """
    marked = rp.exposed & set(rp.models)
    if not marked:
        raise CaptionClaimError(
            "no encoder is TCGA-exposed; the dagger legend would mark nothing. Drop it, or the "
            "legend describes a marker the figure never draws."
        )
    return frozenset(marked)
