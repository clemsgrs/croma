"""benchmark.py discovers its model roster from the tileset's ``.npy`` files.

There is no model registry gate any more: a model is evaluated iff its embedding matrix
exists in the tileset. So an encoder absent from ``model_registry`` is evaluated when its
``.npy`` is present, and a requested model with no ``.npy`` is a hard error naming it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm
from model_registry import _build_model_registry


def _toy_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "image_path": [f"/tmp/{i}.png" for i in range(4)],
            "label": ["A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2"],
            "group_id": ["sl0", "sl1", "sl2", "sl3"],
        }
    )


def _features() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def test_benchmark_evaluates_models_present_only_as_npy(bench_env) -> None:
    # Neither name is in the tile model registry; they exist only as embedded matrices.
    registry = _build_model_registry()
    assert "PRISM" not in registry
    assert "TITAN" not in registry

    manifest = _toy_manifest()
    tileset = manifest.copy()
    bench_env.write_tileset(
        "unknown-tiles",
        tileset,
        {"PRISM": _features(), "TITAN": _features()},
    )
    bench_env.register(
        "toy",
        tileset="unknown-tiles",
        manifest=manifest,
        design="all",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    assert bench_env.run("toy", "k-star", "--models", "PRISM,TITAN", "--progress", "off") == 0

    metrics_df = pd.read_csv(bench_env.results_dir("toy") / "metrics.csv")
    assert metrics_df["model"].tolist() == ["PRISM", "TITAN"]


def test_benchmark_errors_when_requested_model_has_no_embedding(bench_env) -> None:
    manifest = _toy_manifest()
    tileset = manifest.copy()
    bench_env.write_tileset("unknown-tiles", tileset, {"PRISM": _features()})
    bench_env.register(
        "toy",
        tileset="unknown-tiles",
        manifest=manifest,
        design="all",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    with pytest.raises(SystemExit) as excinfo:
        bench_env.run("toy", "k-star", "--models", "PRISM,UNI", "--progress", "off")
    # The missing model must be named so the caller knows what to embed first.
    assert "UNI" in str(excinfo.value)


def test_benchmark_errors_when_tileset_has_no_embeddings(bench_env) -> None:
    manifest = _toy_manifest()
    tileset = manifest.copy()
    # Write the tileset manifest but no ``.npy`` matrices at all.
    bench_env.write_tileset("empty-tiles", tileset, {})
    bench_env.register(
        "toy",
        tileset="empty-tiles",
        manifest=manifest,
        design="all",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    with pytest.raises(SystemExit) as excinfo:
        bench_env.run("toy", "k-star", "--progress", "off")
    assert "empty-tiles" in str(excinfo.value)
    assert "extract_embeddings.py" in str(excinfo.value)
