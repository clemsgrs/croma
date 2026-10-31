r"""Build the supplementary bootstrap-uncertainty section.

Reads the outputs of ``scripts/studies/bootstrap_uncertainty.py`` and
``scripts/studies/mari_ri_delta.py`` through ``_uncertainty`` and writes
``paper/sections/supp/bootstrap_uncertainty.tex``:

* one table per benchmark (Camelyon, TCGA-2x2, TCGA-4x4, Tolkach-ESCA, PCaBiop): median
  [CI], rank [interval], F(0) [CI], LTM_10 [CI] and P(frontier) per ranked encoder;
* the Delta = MaRI - RI table, per encoder x tile benchmark, with the count of intervals that
  exclude 0;
* the cross-benchmark rank agreement of median CRoMa with encoder-bootstrap CIs.

A missing study output raises (``_uncertainty.load``), so ``build_paper.py`` fails rather than
shipping the section without it.

Usage: python scripts/repro/generate_uncertainty_supp_table.py
"""

import argparse
from pathlib import Path

import _uncertainty

OUT = Path("paper/sections/supp/bootstrap_uncertainty.tex")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=_uncertainty.REPO)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.root / OUT
    out.write_text(_uncertainty.render(_uncertainty.load(args.root)))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
