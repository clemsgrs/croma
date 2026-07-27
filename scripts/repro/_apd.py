"""The one computation behind the APD-validation figure and its correlation table.

``scripts/studies/apd/`` computes APD and its rank correlation with CRoMa/RI/MaRI. This
module reads the resulting artifact and answers, in one place, every question the two
paper floats ask of it: how many models entered, how many (model, benchmark) pairs the
headline pools over, at which operating point RI and MaRI were evaluated, and whether the
in-domain probe really is the cleaner test.

Those numbers used to be typed into the captions. They said "$16$ foundation models" long
after the panel reached 20, and "the shared $k{=}15$" for RI/MaRI at a time when the three
benchmarks ran at k = 11, 71 and 61 and shared nothing. A caption that states a protocol
it does not read is worse than one that omits it.

The scope vocabulary is the study's own (see ``scripts/studies/apd/loaders.py``):

  ``camelyon`` / ``tcga_4x4`` / ``tolkach`` / ``prostate``
      one benchmark each.
  ``headline``
      the three faithful PathoROB benchmarks pooled. This is what the paper calls
      "pooled", and what the ``\\Apd...Pooled`` macros carry.
  ``pooled``
      all four, prostate included. Never the headline: prostate's out-of-domain arm is a
      single small centre and cannot enter a cross-benchmark APD_OOD statistic.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (REPO / "src", HERE, REPO / "scripts" / "studies" / "apd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _cross_benchmark import short_label  # noqa: E402
from _paper_tables import CaptionClaimError  # noqa: E402,F401  (re-exported)
from loaders import DATASETS, HEADLINE_DATASETS  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

CORRELATION_CSV = REPO / "output/studies/apd/apd_correlation.csv"

#: The join keeps every model APD was computed for, control included; the correlations are
#: taken over the ranked panel only. Comparing the two is how ``assert_control_excluded``
#: proves the captions' exclusion sentence.
JOINED_CSV = REPO / "output/studies/apd/apd_metrics_joined.csv"

#: The macros name this scope "Pooled"; the study calls it "headline". Both mean the three
#: faithful benchmarks, and neither means ``pooled``.
HEADLINE_SCOPE = "headline"

#: Row order of the figure and column order of the table.
TARGETS = [("apd_id", "in-domain"), ("apd_ood", "out-of-domain")]
METRICS = ["croma", "ri", "mari"]

#: How far the three metrics' headline rho may spread before "all three track APD
#: comparably" stops being a fair description of the table.
COMPARABLE_SPREAD = 0.10


@dataclass(frozen=True)
class Apd:
    corr: pd.DataFrame

    def rho(self, target: str, metric: str, scope: str) -> float:
        row = self.corr[(self.corr["target"] == target)
                        & (self.corr["metric"] == metric)
                        & (self.corr["scope"] == scope)]
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
                f"scope {scope!r} was computed over differing model counts {sorted(counts)}; "
                "a metric is missing a model, so no single n describes the column."
            )
        return int(counts.pop())

    @property
    def n_models(self) -> int:
        """Models per benchmark. The captions say "each of the N models", so if the three
        benchmarks disagree there is no such N and the sentence must not be written."""
        counts = {ds: self.n(ds) for ds in HEADLINE_DATASETS}
        if len(set(counts.values())) != 1:
            raise CaptionClaimError(
                f"the faithful benchmarks were evaluated over different rosters: {counts}. "
                "Re-run scripts/studies/apd/apd_experiment.py for the missing models."
            )
        return next(iter(counts.values()))

    @property
    def n_pairs(self) -> int:
        return self.n(HEADLINE_SCOPE)

    def assert_control_excluded(self) -> None:
        """Both captions state the natural-image control is excluded. Prove it from the data.

        ``loaders.ranked`` drops the control before any rho is computed, and ``n_models``
        counts whatever survived -- but "the natural-image control is excluded" is a
        *sentence*, and a sentence outlives the helper it describes. Were ``ranked`` ever
        dropped from ``corr_block``, every rho would quietly absorb an encoder that is
        doubly flattered here (its CRoMa is high because its biological neighbourhoods are
        poor, and APD is a *relative* drop, so it is scored leniently for having little
        accuracy to lose), ``n_models`` would read one higher, and the caption would still
        assert the exclusion.

        So: the control must be present in the join (its APD stays on record) and absent
        from the rank correlations, on every benchmark whose roster contains it.
        """
        from croma.plotstyle import CONTROL_MODEL

        if not JOINED_CSV.exists():
            raise FileNotFoundError(
                f"{JOINED_CSV} is absent; run scripts/studies/apd/apd_croma_correlation.py."
            )
        joined = pd.read_csv(JOINED_CSV, usecols=["dataset", "model"])
        for ds in HEADLINE_DATASETS:
            roster = set(joined.loc[joined["dataset"] == ds, "model"])
            if CONTROL_MODEL not in roster:
                raise CaptionClaimError(
                    f"{CONTROL_MODEL} is missing from the {ds} join, so the caption's "
                    "exclusion sentence has nothing to exclude. Was APD run for it?"
                )
            if self.n(ds) != len(roster) - 1:
                raise CaptionClaimError(
                    f"the caption says the natural-image control is excluded, but {ds}'s "
                    f"rank correlation ran over {self.n(ds)} of {len(roster)} joined models. "
                    f"Expected {len(roster) - 1}: is loaders.ranked still applied?"
                )

    def benchmark_labels(self) -> list[str]:
        # The dashed short name that heads a float ("TCGA-4x4"), not the math form
        # ``short_label`` yields ("TCGA ($4\times4$)") -- that one is for operating-point prose.
        return [by_benchmark(DATASETS[ds]["benchmark"]).short_name for ds in HEADLINE_DATASETS]

    def operating_points(self) -> list[tuple[str, int]]:
        """(label, k) for each faithful benchmark, read from the run RI/MaRI came from.

        RI and MaRI are k-dependent; CRoMa is not. Naming the k is the only way a reader
        can tell which half of this table would move under a different protocol.
        """
        out = []
        for ds in HEADLINE_DATASETS:
            entry = by_benchmark(DATASETS[ds]["benchmark"])
            ks = set(pd.read_csv(entry.metrics_rel)["k"])
            if len(ks) != 1:
                raise CaptionClaimError(
                    f"{entry.benchmark} has no single operating point (k in {sorted(ks)}); "
                    "the caption cannot name one. Was this run at k-star?"
                )
            out.append((short_label(entry.benchmark), int(ks.pop())))
        return out


def load() -> Apd:
    if not CORRELATION_CSV.exists():
        raise FileNotFoundError(
            f"{CORRELATION_CSV} is absent. Run scripts/studies/apd/apd_experiment.py, then "
            "scripts/studies/apd/apd_croma_correlation.py."
        )
    return Apd(corr=pd.read_csv(CORRELATION_CSV))
