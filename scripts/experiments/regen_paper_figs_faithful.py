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

Usage: python scripts/experiments/regen_paper_figs_faithful.py
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import plotting  # noqa: E402

SRC = ROOT / "output/faithful/pathorob-camelyon-faithful/results"
PAPER = ROOT / "paper/figures/results/pathorob-camelyon-faithful"
TMP = SRC / "plots_faithful_paper"

# figures the paper \includegraphics that come from standard benchmark rows
# (the other 4 referenced figures are rendered by their experiment scripts).
METRICS_FIGS = {
    "croma_ltm_bars": plotting.plot_croma_ltm_bars,
    "croma_vs_mari_scatter": plotting.plot_croma_vs_mari_scatter,
    "mari_vs_ri_scatter": plotting.plot_mari_vs_ri_scatter,
    "ri_mari_support": plotting.plot_ri_mari_support,
}
KSWEEP_FIGS = {
    "knn_bio_k_sweep": plotting.plot_knn_bio_k_sweep,
    "ri_k_sweep": plotting.plot_ri_k_sweep,
}


def main() -> int:
    metrics = json.loads((SRC / "metrics.json").read_text())
    k_sweep = json.loads((SRC / "k_sweep_metrics.json").read_text())
    m_sweep = json.loads((SRC / "croma_m_sweep_metrics.json").read_text())
    TMP.mkdir(parents=True, exist_ok=True)

    rendered = []
    for name, fn in METRICS_FIGS.items():
        fn(rows=metrics, out_path=TMP / f"{name}.png")
        rendered.append(name)
    for name, fn in KSWEEP_FIGS.items():
        fn(rows=k_sweep, out_path=TMP / f"{name}.png")
        rendered.append(name)
    plotting.plot_croma_m_sweep(rows=m_sweep, out_path=TMP / "croma_m_sweep.png")
    rendered.append("croma_m_sweep")

    for name in rendered:
        for sub, ext in (("pdf", "pdf"), ("png", "png")):
            src = TMP / sub / f"{name}.{ext}"
            dst = PAPER / sub / f"{name}.{ext}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                print(f"  MISSING {src}")
                continue
            shutil.copyfile(src, dst)
        print(f"  {name} -> {PAPER.name}/{{pdf,png}}/")

    print(f"\nrendered {len(rendered)} standard figures from faithful into {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
