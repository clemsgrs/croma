"""Re-render the paper's Camelyon benchmark figures from the FAITHFUL run.

The paper's tables and prose numbers use the faithful (n=20,400) run, but the
benchmark figures had been rendered from the reduced-kstar (n=4,000) run, so a
reader cross-referencing figure values against the tables saw small mismatches
(e.g. Virchow2 CRoMa 0.21 in figures vs 0.20 in the table). This script renders
the standard benchmark figures the paper actually includes from the faithful
saved rows -- no recompute -- into the paper figure directory, unifying the whole
paper on the faithful run. The four experiment-script figures (croma_rank,
oo_fraction, support_vs_k, tau_sensitivity) already read faithful and are
re-run separately.

This is a thin wrapper over scripts/bench/render.py: render.py owns the single
plot-call sequence (rendering the faithful run's full figure set into a scratch
dir), and this script only picks the subset the paper \\includegraphics and copies
their PDF/PNG into the paper figure directory.

Usage: python scripts/repro/figures/regen_paper_figs_faithful.py
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))

import render  # noqa: E402

RUN_DIR = ROOT / "output/faithful/pathorob-camelyon-faithful"
PAPER = ROOT / "paper/figures/results/pathorob-camelyon-faithful"
TMP = RUN_DIR / "results/plots_faithful_paper"

# The paper \includegraphics for these standard benchmark figures (the other 4
# referenced figures are rendered by their own experiment scripts). render.py
# emits the full set into TMP; we copy just this subset into the paper dir.
PAPER_FIGS = [
    "croma_ltm_bars",
    "croma_vs_mari_scatter",
    "mari_vs_ri_scatter",
    "ri_mari_support",
    "knn_bio_k_sweep",
    "ri_k_sweep",
    "croma_m_sweep",
]


def main() -> int:
    render.render_run(RUN_DIR, plots_dir=TMP)

    for name in PAPER_FIGS:
        for sub, ext in (("pdf", "pdf"), ("png", "png")):
            src = TMP / sub / f"{name}.{ext}"
            dst = PAPER / sub / f"{name}.{ext}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                print(f"  MISSING {src}")
                continue
            shutil.copyfile(src, dst)
        print(f"  {name} -> {PAPER.name}/{{pdf,png}}/")

    print(f"\nrendered {len(PAPER_FIGS)} standard figures from faithful into {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
