"""Canonical on-disk layout for benchmark inputs and outputs.

Two roots, split along the precious/cheap line:

``output/embeddings/<tileset>/``
    One embedding matrix per *tileset* -- a physical set of tiles that was pushed
    through the encoders. Written once, by ``scripts/bench/extract_embeddings.py``,
    and never touched again by a downstream choice. ``manifest.csv`` defines the row
    order: row ``i`` of every ``<Model>.npy`` describes ``manifest.csv`` row ``i``.
    That alignment is the only invariant the embeddings tree guarantees.

``output/metrics/<protocol>/<benchmark>/``
    Everything derived: ``results/`` and ``plots/``. Cheap to regenerate, so it is
    scoped by the *protocol* (the operating point -- ``k-star`` or ``median-k``)
    that produced it. Protocol-first, so two protocols over the same benchmark sit
    side by side and diff cleanly.

A *benchmark* is a ``(tileset, eval manifest)`` pair: the eval manifest selects rows
from the tileset (and, for paired designs, may repeat a row across subsets). A benchmark
therefore never owns embeddings -- it borrows a row-view of its tileset's. See
``scripts/bench/benchmarks.py`` for the registry.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repo root, and the output tree beneath it. Both are read at call time by the helpers
#: below, so a test (or a scratch run) can retarget the tree by rebinding them.
REPO = Path(os.environ.get("CROMA_REPO", Path(__file__).resolve().parents[2]))
OUTPUT_ROOT = Path(os.environ.get("CROMA_OUTPUT_ROOT", REPO / "output"))

#: Name of the row-order contract inside every tileset directory.
TILESET_MANIFEST_NAME = "manifest.csv"

#: The operating points a metrics run may be scoped by.
PROTOCOLS = ("k-star", "median-k")


def embeddings_root() -> Path:
    return OUTPUT_ROOT / "embeddings"


def metrics_root() -> Path:
    return OUTPUT_ROOT / "metrics"


def validate_protocol(protocol: str) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"unknown protocol {protocol!r}; expected one of {list(PROTOCOLS)}"
        )
    return protocol


def embeddings_dir(tileset: str) -> Path:
    """Directory holding ``<Model>.npy`` (+ sidecars) and ``manifest.csv``."""
    return embeddings_root() / tileset


def tileset_manifest(tileset: str) -> Path:
    """The CSV whose row order the tileset's embedding rows are aligned to."""
    return embeddings_dir(tileset) / TILESET_MANIFEST_NAME


def embedding_path(tileset: str, model: str) -> Path:
    return embeddings_dir(tileset) / f"{model}.npy"


def metrics_dir(protocol: str, benchmark: str) -> Path:
    """Run directory for one benchmark at one operating point."""
    return metrics_root() / validate_protocol(protocol) / benchmark


def results_dir(protocol: str, benchmark: str) -> Path:
    return metrics_dir(protocol, benchmark) / "results"


def plots_dir(protocol: str, benchmark: str) -> Path:
    return metrics_dir(protocol, benchmark) / "plots"
