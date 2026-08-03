"""Per-sample CRoMa ridgelines for the tail analysis.

One figure per benchmark. Each tile figure draws the full encoder roster plus the natural-image
control; the slide-level PANDA figure draws its four whole-slide encoders and no control:

  <camelyon>/studies/plots/pdf/croma_distribution.pdf   fig:croma-distribution        (main text)
  <tcga-4x4>/studies/plots/pdf/croma_distribution.pdf   fig:croma-distribution-tcga4x4 (supp)
  <tolkach>/studies/plots/pdf/croma_distribution.pdf    fig:croma-distribution-tolkach (supp)
  <panda>/studies/plots/pdf/croma_distribution.pdf      fig:croma-distribution-panda   (supp)

Each renders into its own run's ``studies/plots/{pdf,png}/`` beside the data it reads, never
into ``paper/``: a figure staged straight into the manuscript tree is a second source of truth
that goes stale silently (see the note on ``OUT`` in ``apd_figure.py``). Copy each PDF into
``paper/figures/results/<benchmark>/pdf/`` by hand if it earns its float;
``scripts/repro/check_paper_figures.py`` reports the ones that have drifted.

Every run directory comes from ``_distributions`` (which resolves the protocol via
``paper_manifest``), so this script names no protocol.

Run: python scripts/repro/figures/croma_distribution_figure.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plotting as P  # noqa: E402
from plotting import style as plotstyle  # noqa: E402
from _distributions import CAMELYON, PANDA, SUPP_BENCHMARKS  # noqa: E402

#: Every benchmark the tail analysis draws: Camelyon in the main text, the two other tile
#: benchmarks and the slide-level PANDA panel in the supplement. Each tile figure draws the
#: full roster plus the natural-image control; PANDA has no control, only its four encoders.
BENCHMARKS = (CAMELYON, *SUPP_BENCHMARKS, PANDA)


def _rows(entry) -> list[dict]:
    """The benchmark's per-model metrics rows (each carries a ``croma_samples_path``)."""
    path = REPO / entry.run_rel / "results" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent -- run scripts/repro/run_benchmarks.sh first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    plotstyle.apply_style()
    for entry in BENCHMARKS:
        # The studies plots dir hangs off the run, so a benchmark re-render of <run>/plots never
        # clobbers these curated figures, and check_paper_figures still sources them from here.
        out_png = REPO / entry.studies_rel / "plots" / "croma_distribution.png"
        # The full roster (models=None): every encoder plus the natural-image control.
        P.plot_croma_sample_distributions(rows=_rows(entry), out_path=out_png)
        print(f"wrote {P._pdf_export_path(out_png)}")


if __name__ == "__main__":
    main()
