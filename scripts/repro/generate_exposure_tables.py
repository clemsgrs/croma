"""Build the pretraining-exposure tables.

  paper/sections/supp/exposure_matrix.tex   tab:exposure-matrix   (\\input by supp.tex)
  paper/sections/supp/drop_exposed.tex      tab:drop-exposed      (\\input by supp.tex)

The matrix reads the committed ``scripts/bench/model_metadata.csv``; the sensitivity reads the
Camelyon, TCGA-4x4 and Tolkach-ESCA runs ``paper_manifest`` names and the APD join. Each table
records the sha256 of every input it read in its header. The numbers the prose quotes are the
same build's macros, emitted by ``generate_paper_values.py``. Everything lives in ``_exposure.py``.

Run: python scripts/repro/generate_exposure_tables.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _exposure import DROP_EXPOSED_OUT, MATRIX_OUT, build  # noqa: E402


def main() -> None:
    built = build()
    MATRIX_OUT.write_text(built.matrix_tex)
    print(f"wrote {MATRIX_OUT}")
    DROP_EXPOSED_OUT.write_text(built.drop_exposed_tex)
    print(f"wrote {DROP_EXPOSED_OUT}")


if __name__ == "__main__":
    main()
