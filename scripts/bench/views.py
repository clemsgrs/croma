"""Load a benchmark as a row-view over its tileset's embeddings.

This is the one place that knows how to turn a benchmark name into *the rows it
evaluates*. Reach for it from any study or figure script that needs a benchmark's
embeddings, manifest, or run directory.

The trap it exists to close: before the tileset/metrics split, a benchmark's directory
held an ``embedding_source_manifest.csv`` that was exactly its evaluated sample set
(Camelyon-faithful: 20,400 rows). Now the embeddings directory holds the *tileset*
manifest, which is a superset (Camelyon: 22,402 rows). Reading that manifest and its
matrices directly would silently evaluate the wrong sample set -- 2,002 extra tiles from
the three out-of-domain centres -- and quietly change every number. ``load_view`` applies
the benchmark's own manifest, so the rows are always the ones the benchmark defines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import benchmarks
import layout
from benchmarks import BenchmarkSpec

from croma.alignment import build_view_row_index
from croma.metrics.pairs import load_manifest


@dataclass(frozen=True)
class BenchmarkView:
    """A benchmark's evaluated rows, resolved against its tileset."""

    spec: BenchmarkSpec
    #: Canonicalised eval manifest (confounder column renamed to ``confounder``).
    eval_manifest: pd.DataFrame
    #: ``rows[i]`` is the tileset embedding row for eval row ``i``.
    rows: np.ndarray
    #: Models embedded for this benchmark's tileset.
    models: tuple[str, ...]

    def features(self, model: str) -> np.ndarray:
        """Embeddings for this benchmark, one row per eval-manifest row."""
        path = layout.embedding_path(self.spec.tileset, model)
        if not path.exists():
            raise FileNotFoundError(
                f"no embeddings for {model!r} in tileset {self.spec.tileset!r}: {path}"
            )
        matrix = np.load(path, mmap_mode="r")
        return np.asarray(matrix[self.rows])

    def metrics_dir(self, protocol: str) -> Path:
        return layout.metrics_dir(protocol, self.spec.name)

    def results_dir(self, protocol: str) -> Path:
        return layout.results_dir(protocol, self.spec.name)

    def studies_dir(self, protocol: str) -> Path:
        return layout.metrics_dir(protocol, self.spec.name) / "studies"


def load_view(benchmark: str) -> BenchmarkView:
    spec = benchmarks.get(benchmark)
    tileset_manifest_path = layout.tileset_manifest(spec.tileset)
    if not tileset_manifest_path.exists():
        raise FileNotFoundError(
            f"tileset {spec.tileset!r} is not embedded: missing {tileset_manifest_path}"
        )
    tileset_manifest = pd.read_csv(tileset_manifest_path)
    eval_manifest = load_manifest(
        str(layout.REPO / spec.manifest), confounder_column=spec.confounder_column
    ).reset_index(drop=True)
    rows = build_view_row_index(eval_manifest, tileset_manifest)
    models = tuple(sorted(p.stem for p in layout.embeddings_dir(spec.tileset).glob("*.npy")))
    return BenchmarkView(spec=spec, eval_manifest=eval_manifest, rows=rows, models=models)
