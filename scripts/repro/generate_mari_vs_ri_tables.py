"""Build the two MaRI-versus-RI supplementary tables (tab:mari-vs-ri, tab:mari-tau-sweep).

Both render from ``output/studies/mari-vs-ri/`` through ``_mari_vs_ri``, the basis the
in-text macros in ``generate_paper_values.py`` also read. A missing study output fails the
build rather than skipping it.

Run: python scripts/repro/generate_mari_vs_ri_tables.py
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _mari_vs_ri  # noqa: E402

OUT_DIR = Path("paper/sections/supp")
SUMMARY = "mari_vs_ri.tex"
TAU_SWEEP = "mari_tau_sweep.tex"


def build(root: Path) -> dict[str, str]:
    """The two tables' LaTeX, keyed by file name."""
    study = _mari_vs_ri.load(root)
    return {
        SUMMARY: _mari_vs_ri.render_summary_table(study),
        TAU_SWEEP: _mari_vs_ri.render_tau_sweep_table(study),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    for name, tex in build(args.root).items():
        path = args.out_dir / name
        path.write_text(tex)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
