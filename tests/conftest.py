"""Shared harness for driving the compute-only benchmark driver in tests.

``benchmark.py`` resolves everything from two places: the benchmark registry and the
output layout. So a test sets up a *tileset* on disk (embeddings + manifest.csv),
registers a *benchmark* that views it, and points ``layout`` at the temp tree.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmarks as benchmarks_module  # noqa: E402
import layout as layout_module  # noqa: E402
from benchmarks import BenchmarkSpec  # noqa: E402


@pytest.fixture
def bench_env(tmp_path, monkeypatch):
    """Retarget layout at a temp tree and give a builder for tilesets/benchmarks."""
    monkeypatch.setattr(layout_module, "REPO", tmp_path)
    monkeypatch.setattr(layout_module, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(
        benchmarks_module, "BENCHMARKS", dict(benchmarks_module.BENCHMARKS)
    )
    return _BenchEnv(tmp_path, monkeypatch)


class _BenchEnv:
    def __init__(self, root: Path, monkeypatch) -> None:
        self.root = root
        self._monkeypatch = monkeypatch

    def write_tileset(
        self,
        tileset: str,
        manifest: pd.DataFrame,
        features: dict[str, np.ndarray],
    ) -> Path:
        """Materialise ``output/embeddings/<tileset>/`` with manifest.csv + <Model>.npy."""
        directory = layout_module.embeddings_dir(tileset)
        directory.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(layout_module.tileset_manifest(tileset), index=False)
        for model, matrix in features.items():
            arr = np.asarray(matrix, dtype=np.float32)
            if arr.shape[0] != len(manifest):
                raise AssertionError(
                    f"{model}: {arr.shape[0]} embedding rows vs {len(manifest)} manifest rows"
                )
            np.save(directory / f"{model}.npy", arr)
        return directory

    def register(
        self,
        name: str,
        *,
        tileset: str,
        manifest: pd.DataFrame,
        design: str = "dataset_wide",
        k_max: int = 5,
        confounder_column: str = "medical_center",
    ) -> BenchmarkSpec:
        """Write a benchmark's eval manifest under data/ and register it."""
        rel = Path("data") / f"{name}.csv"
        target = self.root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(target, index=False)
        spec = BenchmarkSpec(
            name=name,
            tileset=tileset,
            manifest=str(rel),
            design=design,
            k_max=k_max,
            confounder_column=confounder_column,
        )
        benchmarks_module.BENCHMARKS[name] = spec
        return spec

    def respec(self, name: str, **changes) -> BenchmarkSpec:
        spec = replace(benchmarks_module.BENCHMARKS[name], **changes)
        benchmarks_module.BENCHMARKS[name] = spec
        return spec

    def run(self, benchmark: str, protocol: str = "k-star", *extra: str) -> int:
        import benchmark as bm

        argv = ["benchmark.py", "--benchmark", benchmark, "--protocol", protocol, *extra]
        self._monkeypatch.setattr(sys, "argv", argv)
        return bm.main()

    def results_dir(self, benchmark: str, protocol: str = "k-star") -> Path:
        return layout_module.results_dir(protocol, benchmark)
