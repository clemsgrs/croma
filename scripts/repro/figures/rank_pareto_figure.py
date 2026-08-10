"""The rank-aggregate Pareto overview across the three tile benchmarks (fig:croma-pareto-rank).

A single panel that collapses the per-benchmark median-vs-tail Pareto figures into one: mean
median-\\code{CRoMa} rank on the x-axis against mean ``LTM_10`` rank on the y-axis, one point per
pathology encoder, with the Pareto frontier drawn. Ranking is scale-free, so -- unlike a mean of
the raw margins -- a wider-margin benchmark cannot dominate the aggregate, and the TCGA
in-distribution boost counts as one rank rather than a large margin. TCGA-exposed encoders carry
a dagger after their name in their point label, because two of the three benchmarks contain a TCGA cohort.

  output/studies/rank-pareto/plots/pdf/rank_pareto.pdf   fig:croma-pareto-rank   (supplement)

It renders into a standalone study directory beside no single benchmark run (it aggregates three),
never into ``paper/``: a figure staged straight into the manuscript tree is a second source of
truth that goes stale silently (ADR-0010). Copy the PDF to ``paper/figures/rank_pareto.pdf`` by
hand if it earns its float; ``scripts/repro/check_paper_figures.py`` reports the ones that have
drifted. The panel, the ranks and the exposed set come from ``_rank_pareto.load()`` (which resolves
every run through ``paper_manifest``), the same loader the float generator reads, so the figure
and its caption cannot disagree.

Run: python scripts/repro/figures/rank_pareto_figure.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plotting as P  # noqa: E402
from plotting import style as plotstyle  # noqa: E402
from _rank_pareto import load  # noqa: E402

#: Beside the study data it is drawn from, under ``plots/{pdf,png}/``. Nothing writes into
#: ``paper/figures/``; copy what the paper needs by hand.
OUT = REPO / "output/studies/rank-pareto/plots/rank_pareto.png"


def main() -> None:
    plotstyle.apply_style()
    rp = load()
    P.plot_rank_pareto(rp.rows(), OUT, n_benchmarks=rp.n_benchmarks)
    print(f"wrote {P._pdf_export_path(OUT)}")
    print(
        f"\n{rp.n_models} models across {rp.n_benchmarks} benchmarks, "
        f"{len(rp.exposed)} TCGA-exposed"
    )
    print("frontier (mean-rank Pareto):", ", ".join(rp.frontier))


if __name__ == "__main__":
    main()
