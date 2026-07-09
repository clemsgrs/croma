import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm
import croma.metrics.neighbors as nb
from croma import MaRI, RI
from croma.metrics.pairs import normalize_manifest, resolve_manifest_subsets


def _paired_manifest() -> pd.DataFrame:
    base_rows = [
        ("s0", "/tmp/s0.png", "A", "V1", "sl0"),
        ("s1", "/tmp/s1.png", "A", "V2", "sl1"),
        ("s2", "/tmp/s2.png", "B", "V1", "sl2"),
        ("s3", "/tmp/s3.png", "B", "V2", "sl3"),
    ]
    rows: list[dict[str, str]] = []
    for subset in ("pair1", "pair2"):
        for sample_id, image_path, label, confounder, slide_id in base_rows:
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "label": label,
                    "scanner_vendor": confounder,
                    "slide_id": slide_id,
                    "subset": subset,
                    "dataset": "toy",
                }
            )
    return pd.DataFrame(rows)


def _paired_features() -> np.ndarray:
    return np.asarray(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.85, 0.15, 0.00, 0.00],
            [0.15, 0.85, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00],
            [0.85, 0.15, 0.00, 0.00],
            [0.15, 0.85, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
        ],
        dtype=float,
    )


def _tileset_from(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in manifest.columns if c not in ("subset", "dataset")]
    return (
        manifest.loc[:, cols]
        .drop_duplicates(subset=["sample_id", "image_path"])
        .reset_index(drop=True)
    )


def test_paired_cached_artifacts_match_uncached_metric_outputs() -> None:
    manifest = _paired_manifest()
    features = _paired_features()
    k_values = [1, 3]
    selected_k = 1
    normalized = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    canonical_manifest = normalize_manifest(
        manifest, confounder_column="scanner_vendor", source="manifest"
    )
    subsets = resolve_manifest_subsets(canonical_manifest)

    for metric_cls, extra_kwargs in ((RI, {}), (MaRI, {"tau": 0.2})):
        uncached = metric_cls._compute_artifacts(
            features=features,
            manifest=manifest,
            confounder_column="scanner_vendor",
            k_values=k_values,
            evaluation_design="paired_2x2",
            selected_k=selected_k,
            include_selected_result=True,
            warn_selected_result=False,
            **extra_kwargs,
        )
        cached_subsets = metric_cls._prepare_paired_subset_neighbor_cache(
            features=normalized,
            subsets=subsets,
            k_values=k_values,
        )
        cached = metric_cls._compute_artifacts_from_prepared_subsets(
            prepared_subsets=cached_subsets,
            dataset_name="toy",
            k_values=k_values,
            evaluation_design="paired_2x2",
            selected_k=selected_k,
            include_selected_result=True,
            warn_selected_result=False,
            **extra_kwargs,
        )

        assert cached.curve == uncached.curve
        assert uncached.result is not None
        assert cached.result is not None
        assert cached.result.k == uncached.result.k
        assert cached.result.value == uncached.result.value
        assert cached.result.std == uncached.result.std
        np.testing.assert_array_equal(
            cached.result.pair_values, uncached.result.pair_values
        )
        np.testing.assert_allclose(
            cached.result.sample_values,
            uncached.result.sample_values,
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            cached.result.sample_values_aligned,
            uncached.result.sample_values_aligned,
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
        )
        np.testing.assert_array_equal(
            cached.result.occurrence_defined_mask,
            uncached.result.occurrence_defined_mask,
        )
        np.testing.assert_array_equal(
            cached.result.sample_undefined_types, uncached.result.sample_undefined_types
        )
        np.testing.assert_array_equal(
            cached.result.occurrence_subsets, uncached.result.occurrence_subsets
        )
        np.testing.assert_array_equal(
            cached.result.occurrence_source_indices,
            uncached.result.occurrence_source_indices,
        )


def test_benchmark_paired_prepares_neighbors_once_per_subset(bench_env) -> None:
    manifest = _paired_manifest()
    # The eval manifest repeats each tile across pair1/pair2, so the tileset holds one
    # embedding per distinct tile (4), and the benchmark gathers a row-view of them.
    tileset = _tileset_from(manifest)
    bench_env.write_tileset(
        "paired-tiles", tileset, {"M1": _paired_features()[: len(tileset)]}
    )
    bench_env.register(
        "toy",
        tileset="paired-tiles",
        manifest=manifest,
        design="paired_2x2",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    neighbor_prepare_calls = {"count": 0}
    original_prepare = nb._prepare_neighbors_with_meta

    def wrapped_prepare(*args, **kwargs):
        neighbor_prepare_calls["count"] += 1
        return original_prepare(*args, **kwargs)

    bench_env._monkeypatch.setattr(nb, "_prepare_neighbors_with_meta", wrapped_prepare)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0
    assert neighbor_prepare_calls["count"] == 2

    results_dir = bench_env.results_dir("toy")
    per_sample_df = pd.read_csv(results_dir / "per_sample_metrics.csv")
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    assert len(per_sample_df) == 8
    assert per_sample_df["occurrence_index"].tolist() == list(range(8))
    assert set(per_sample_df["subset"]) == {"pair1", "pair2"}
    assert set(per_sample_df["source_sample_index"]) == set(range(8))
    assert set(metrics_df["evaluation_design"]) == {"paired_2x2"}


def _dataset_wide_manifest() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append(
            {
                "sample_id": f"s{i}",
                "image_path": f"/tmp/s{i}.png",
                "label": "A" if i % 2 == 0 else "B",
                "scanner_vendor": "V1" if i < 4 else "V2",
                "slide_id": f"sl{i}",
                "dataset": "toy",
            }
        )
    return pd.DataFrame(rows)


def _dataset_wide_features(tileset: pd.DataFrame) -> np.ndarray:
    base = {"A": [1.0, 0.0], "B": [0.0, 1.0]}
    conf_offset = {"V1": 0.0, "V2": 0.25}
    return np.asarray(
        [
            base[row["label"]] + [conf_offset[row["scanner_vendor"]], 0.0]
            for _, row in tileset.iterrows()
        ],
        dtype=float,
    )


def test_benchmark_dataset_wide_shares_one_neighbor_cache_across_ri_mari_tau(
    bench_env,
) -> None:
    """RI, MaRI and the tau-scale check must reuse a single dataset-wide neighbour cache.

    Per model on a cold cache the only neighbour preparations are the two (unpruned) kNN
    curves plus one shared RI/MaRI/tau cache -> 3 calls. Before the shared cache RI, MaRI
    and the tau check each prepared their own (5 calls); this pins the reuse.
    """
    manifest = _dataset_wide_manifest()
    tileset = _tileset_from(manifest)
    bench_env.write_tileset(
        "wide-tiles", tileset, {"M1": _dataset_wide_features(tileset)}
    )
    bench_env.register(
        "toy",
        tileset="wide-tiles",
        manifest=manifest,
        design="dataset_wide",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    neighbor_prepare_calls = {"count": 0}
    original_prepare = nb._prepare_neighbors_with_meta

    def wrapped_prepare(*args, **kwargs):
        neighbor_prepare_calls["count"] += 1
        return original_prepare(*args, **kwargs)

    bench_env._monkeypatch.setattr(nb, "_prepare_neighbors_with_meta", wrapped_prepare)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0
    # 2 unpruned kNN curves (bio + confounder) + 1 shared RI/MaRI/tau cache.
    assert neighbor_prepare_calls["count"] == 3

    metrics_df = pd.read_csv(bench_env.results_dir("toy") / "metrics.csv")
    assert set(metrics_df["evaluation_design"]) == {"dataset_wide"}
