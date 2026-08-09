"""The median-vs-tail Pareto figures for the decision-procedure argument.

One figure per tile benchmark: median \\code{CRoMa} (central tendency) on the x-axis against the
worst-decile mean ``LTM`` (tail severity) on the y-axis, one point per pathology encoder, with
the Pareto frontier drawn. It answers the "so which model is best?" question the tail argument
otherwise leaves implicit -- larger on both axes is better, so the non-dominated set is the
upper-right frontier and every other encoder is beaten on *both* axes at once. The natural-image
control holds no rank and is dropped (``dist.pathology``), as everywhere else.

  <camelyon>/studies/plots/pdf/croma_pareto.pdf   fig:croma-pareto           (main text)
  <tcga-4x4>/studies/plots/pdf/croma_pareto.pdf   fig:croma-pareto-tcga4x4   (supp)
  <tolkach>/studies/plots/pdf/croma_pareto.pdf    fig:croma-pareto-tolkach   (supp)

Every panel labels its Pareto frontier in bold and flags -- with a dagger after the encoder's name
in the legend -- those whose pretraining overlaps a *scored* cohort of that benchmark:
CAMELYON-trained on the Camelyon panel, TCGA-exposed on TCGA-4x4, and the RudolfV 2 family with
possible institutional/source-domain exposure to Tolkach-ESCA's scored CHA cohort. The latter
does not assert patient- or slide-level leakage. Each set is resolved through
``_distributions.exposed_models`` so the figure and the float caption count one set (mirrors the
rank-aggregate overview's exposure marks).

Each renders into its own run's ``studies/plots/{pdf,png}/`` beside the data it reads, never
into ``paper/``: a figure staged straight into the manuscript tree is a second source of truth
that goes stale silently (see ADR-0010 and the note in ``apd_figure.py``). Copy each PDF into
``paper/figures/results/<benchmark>/pdf/croma_pareto.pdf`` by hand if it earns its float;
``scripts/repro/check_paper_figures.py`` reports the ones that have drifted.

Every run directory is resolved through ``paper_manifest`` (via ``_distributions``), so this
script names no protocol.

Run: python scripts/repro/figures/croma_pareto_figure.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import plotting as P  # noqa: E402
from plotting import style as plotstyle  # noqa: E402
from _distributions import CAMELYON, SUPP_BENCHMARKS, exposed_models, load  # noqa: E402

#: Camelyon in the main text, the two other tile benchmarks in the supplement. Every panel
#: draws the pathology roster only; the control is a floor, not a competitor.
BENCHMARKS = (CAMELYON, *SUPP_BENCHMARKS)


def _ltm_alpha_pct(entry) -> int:
    """The worst-decile fraction the run reports, as a percent (10 for the headline LTM_10).

    Read from the run rather than assumed, so the axis label follows the data if the tail
    fraction ever changes; the paper writes this as ``LTM_10``.
    """
    metrics = pd.read_csv(REPO / entry.metrics_rel)
    alphas = sorted(metrics["croma_alpha"].dropna().unique())
    if len(alphas) != 1:
        raise ValueError(f"expected one CRoMa LTM alpha on {entry.benchmark}, got {alphas}")
    return int(round(float(alphas[0]) * 100))


def main() -> None:
    plotstyle.apply_style()
    for entry in BENCHMARKS:
        dist = load(entry=entry)
        # Margin-scale median and LTM straight from the guarded loader; the control is excluded
        # from the frontier just as it is from every ranking.
        rows = [
            {"model": m.name, "croma": m.median, "croma_ltm_alpha": m.ltm}
            for m in dist.pathology
        ]
        out_png = REPO / entry.studies_rel / "plots" / "croma_pareto.png"
        # Every panel labels its frontier and marks (a dagger in the legend) the encoders exposed
        # to a scored cohort of this benchmark (Camelyon: corpus; TCGA-4x4: corpus; Tolkach:
        # possible Charite institutional/source-domain overlap), from the same metadata the
        # captions read.
        P.plot_croma_pareto(
            rows,
            out_png,
            exposed=exposed_models(entry, dist),
            ltm_alpha_pct=_ltm_alpha_pct(entry),
        )
        print(f"wrote {P._pdf_export_path(out_png)}")


if __name__ == "__main__":
    main()
