"""Recompute the PathoROB-comparable benchmark metrics, then rebuild the paper artifacts.

For each tile benchmark this runs the compute-only ``benchmark.py`` at the protocol the
manifest records for it, then hands off to ``build_paper.py``. ``benchmark.py`` reads the
tileset's pre-extracted embeddings (``output/embeddings/<tileset>/``) and a row-view of them,
so there is nothing to materialise here; embeddings are produced once per tileset by
``scripts/bench/extract_embeddings.py``.

Two constants used to live here and no longer do, because both were second sources of truth
that silently diverged from the runs (ADR-0010):

* a hard-coded 16-model ``MODELS`` roster -- the panel is whatever the benchmark has
  embeddings for, and pinning it here kept the paper at 16 models after it grew to 21;
* a ``PROTOCOL = "k-star"`` pin -- the protocol belongs to the benchmark, and this pin had
  the tables rendered from k-star runs while the prose macros were computed from median-k.

Usage:
    python scripts/repro/reproduce_faithful.py            # every tile benchmark
    python scripts/repro/reproduce_faithful.py Camelyon   # one, by manifest prefix
"""

import subprocess
import sys
from pathlib import Path

from paper_manifest import TABLES

REPO = Path(__file__).resolve().parents[2]

#: The slide panel is recomputed by its own driver; this script owns the tile benchmarks.
TILE_PREFIXES = [t.prefix for t in TABLES if t.model_type == "tile-level"]


def run(prefix: str) -> None:
    entry = next(t for t in TABLES if t.prefix == prefix)
    cmd = [sys.executable, str(REPO / "scripts/bench/benchmark.py"),
           "--benchmark", entry.benchmark,
           "--protocol", entry.protocol,
           "--k-max", "100",
           "--progress", "off"]
    print(f"[{prefix}] {' '.join(cmd[3:])}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)
    print(f"[{prefix}] metrics -> {entry.metrics_rel}", flush=True)


if __name__ == "__main__":
    targets = sys.argv[1:] or TILE_PREFIXES
    unknown = set(targets) - {t.prefix for t in TABLES}
    if unknown:
        raise SystemExit(f"unknown manifest prefix(es): {sorted(unknown)}")
    for prefix in targets:
        run(prefix)
    subprocess.run([sys.executable, str(REPO / "scripts/repro/build_paper.py")], check=True)
