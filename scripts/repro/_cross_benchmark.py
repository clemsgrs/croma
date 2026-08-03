"""The one computation behind the cross-benchmark rank figure and its caption.

A rank bump chart across the three tile-level PathoROB benchmarks, plus every claim its
caption makes. Both used to be derived independently -- a script drew the PDF, a human
typed the caption beside it -- and they drifted apart in three separate ways at once:

* the caption said "TCGA and Tolkach-ESCA produce none [confounder-dominant markers]",
  which stopped being true when the TCGA-4x4 run was corrected from eight centres to
  PathoROB's four in-domain ones and ``Prost40M`` crossed below zero;
* it said "the five models whose pretraining data overlaps TCGA" while
  ``model_metadata.csv`` records nine;
* it called Tolkach-ESCA "an unexposed benchmark", though one of its four cohorts *is*
  TCGA -- the premise of ``tab:pretraining-overlap``, in the same supplement.

One loader, two renderers. The figure draws what ``load()`` returns; the float generator
writes a caption whose every number comes from the same object, and refuses to ship a
sentence the data contradicts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (REPO / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _paper_tables import CROMA_HEADLINE_M, croma_as_margin  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting.style import CONTROL_MODEL  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

#: The three tile benchmarks the chart spans, in the paper's order. Labels are the
#: manifest's display names with the corpus prefix dropped -- an axis tick has no room
#: for it, and a second hand-written name list is how the last drift started.
_PREFIX = "PathoROB "
PANEL: list[str] = [
    "pathorob-camelyon",
    "pathorob-tcga-4x4",
    "pathorob-tolkach-esca",
]

METADATA = HERE / "model_metadata.csv"

#: Cohort of Tolkach-ESCA drawn from TCGA. Tolkach is therefore *partly* exposed, and a
#: TCGA-pretrained model's rank on it cannot be called leakage-free without checking.
TOLKACH_TCGA_COHORT = "VALSET3_TCGA"
TOLKACH_TCGA_COHORT_TEX = TOLKACH_TCGA_COHORT.replace("_", r"\_")
OVERLAP_PER_SAMPLE = REPO / "output/studies/pretraining-overlap/per_sample_metrics.csv"


def short_label(benchmark: str) -> str:
    return by_benchmark(benchmark).display_name.removeprefix(_PREFIX)


@dataclass(frozen=True)
class CrossBenchmark:
    """Everything the figure draws and the caption asserts, computed once."""

    #: models x benchmarks, signed CRoMa margin at the headline m. Ranked panel only.
    croma: pd.DataFrame
    #: same shape; 1 = most robust within a benchmark.
    ranks: pd.DataFrame
    #: models whose pretraining corpus overlaps TCGA, per model_metadata.csv.
    exposed: frozenset[str]

    @property
    def labels(self) -> list[str]:
        return list(self.croma.columns)

    @property
    def n_models(self) -> int:
        return len(self.croma)

    def hollow(self, label: str) -> list[str]:
        """Confounder-dominant models (CRoMa < 0) on one benchmark, worst first."""
        col = self.croma[label]
        return list(col[col < 0].sort_values().index)

    @property
    def adversarial(self) -> str:
        """The benchmark that pushes the most models below zero. Defined once, because the
        caption both names it and contrasts the others against it."""
        return max(self.labels, key=lambda lab: len(self.hollow(lab)))

    def leader(self, label: str) -> str:
        return self.ranks[label].idxmin()

    def rank_of(self, model: str, label: str) -> int:
        return int(self.ranks.loc[model, label])


def load() -> CrossBenchmark:
    """Signed CRoMa per (model, benchmark) over the ranked panel.

    The natural-image control is dropped: this is a rank chart, and a model that never
    saw tissue holds no rank among pathology encoders (see CONTEXT.md, "ranked panel").
    It is a floor, and a floor does not compete.
    """
    cols = {}
    for benchmark in PANEL:
        series = pd.read_csv(by_benchmark(benchmark).metrics_rel).set_index("model")["croma"]
        cols[short_label(benchmark)] = croma_as_margin(series)

    croma = pd.DataFrame(cols).dropna().drop(index=CONTROL_MODEL, errors="ignore")
    ranks = croma.rank(ascending=False, method="first").astype(int)

    md = pd.read_csv(METADATA)
    exposed = frozenset(md.loc[md["tcga_exposed"], "model"]) & set(croma.index)
    return CrossBenchmark(croma=croma, ranks=ranks, exposed=exposed)


def tolkach_median_without_tcga() -> pd.Series:
    """Median CRoMa on Tolkach-ESCA's three *non*-TCGA cohorts, best first.

    The only way to ask whether a TCGA-pretrained model's Tolkach rank is inflated by the one
    TCGA cohort it contains. Delegates to ``_overlap``, which raises when the study has not
    been run and owns the sole definition of this median; Section 3.4 cites the same numbers.
    """
    from _overlap import rows as overlap_rows

    r = overlap_rows(include_control=False)
    return r.set_index("model")["rest"].sort_values(ascending=False)
