"""The output layout and the benchmark registry are the navigable contract."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmarks  # noqa: E402
import layout  # noqa: E402


def test_embeddings_are_addressed_by_tileset_not_by_benchmark() -> None:
    assert layout.embeddings_dir("pathorob-camelyon") == (
        layout.OUTPUT_ROOT / "embeddings" / "pathorob-camelyon"
    )
    assert layout.tileset_manifest("pathorob-camelyon").name == "manifest.csv"
    assert layout.embedding_path("pathorob-camelyon", "UNI").name == "UNI.npy"


def test_metrics_are_scoped_by_protocol_then_benchmark() -> None:
    assert layout.metrics_dir("median-k", "pathorob-camelyon") == (
        layout.OUTPUT_ROOT / "metrics" / "median-k" / "pathorob-camelyon"
    )
    assert layout.results_dir("k-star", "pathorob-camelyon").parts[-2:] == (
        "pathorob-camelyon",
        "results",
    )


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown protocol"):
        layout.metrics_dir("median_k", "pathorob-camelyon")


def test_every_benchmark_names_a_registered_tileset() -> None:
    for name, spec in benchmarks.BENCHMARKS.items():
        assert spec.name == name
        assert spec.tileset in benchmarks.TILESETS
        assert spec.design in {"all", "paired_2x2"}
        assert spec.k_max >= 1


def test_k_max_is_a_uniform_100_across_every_benchmark() -> None:
    # No per-benchmark override survives: every spec uses the single ceiling constant.
    assert benchmarks.DEFAULT_K_MAX == 100
    for spec in benchmarks.BENCHMARKS.values():
        assert spec.k_max == benchmarks.DEFAULT_K_MAX


def test_benchmarks_share_tilesets_so_embeddings_are_never_duplicated() -> None:
    # The whole point of the split: several views, one embedding matrix.
    assert set(benchmarks.TILESETS["prostate-shift"]) == {
        "prostate",
        "prostate-4class",
        "prostate-gradebal",
    }
    assert set(benchmarks.TILESETS["panda-wsi"]) == {"panda", "panda-isup"}
    assert set(benchmarks.TILESETS["pathorob-camelyon"]) == {"pathorob-camelyon"}


def test_benchmark_manifests_are_repo_relative_csv_paths() -> None:
    for spec in benchmarks.BENCHMARKS.values():
        assert spec.manifest.endswith(".csv")
        assert not spec.manifest.startswith("/")
        assert not spec.manifest.startswith(
            "output/"
        ), f"{spec.name}: eval manifests are inputs and belong under data/"


def test_unknown_benchmark_is_rejected_with_the_registered_names() -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        benchmarks.get("camelyon-faithful")


def test_registry_carries_no_model_roster() -> None:
    # Models are discovered from what is embedded for the tileset, so a newly embedded
    # encoder joins every benchmark over that tileset without touching the registry.
    spec = benchmarks.get("pathorob-camelyon")
    assert not hasattr(spec, "models")
