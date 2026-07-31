import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm


def _toy_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3", "s0", "s4", "s5", "s6"],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2", "V1", "V2", "V1", "V2"],
            "group_id": ["sl0", "sl1", "sl2", "sl3", "sl0", "sl4", "sl5", "sl6"],
            "subset": [
                "pair1",
                "pair1",
                "pair1",
                "pair1",
                "pair2",
                "pair2",
                "pair2",
                "pair2",
            ],
            "dataset": ["toy"] * 8,
        }
    )


def _toy_features() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.85, 0.15, 0.00, 0.00],
            [0.15, 0.85, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.98, 0.02, 0.00, 0.00],
            [0.82, 0.18, 0.00, 0.00],
            [0.18, 0.82, 0.00, 0.00],
            [0.02, 0.98, 0.00, 0.00],
        ],
        dtype=float,
    )


TOY_TILESET = "toy-tiles"


def _tileset_manifest() -> pd.DataFrame:
    manifest = _toy_manifest().drop(columns=["subset", "dataset"])
    return manifest.drop_duplicates(subset=["sample_id", "image_path"]).reset_index(drop=True)


def _setup(bench_env, *, name: str = "toy", k_max: int = 3) -> None:
    """Embed a single-model toy tileset and register a dataset-wide benchmark."""
    tileset = _tileset_manifest()
    bench_env.write_tileset(TOY_TILESET, tileset, {"M1": _toy_features()[: len(tileset)]})
    bench_env.register(
        name,
        tileset=TOY_TILESET,
        manifest=_toy_manifest(),
        design="dataset_wide",
        k_max=k_max,
        confounder_column="scanner_vendor",
    )


def test_tau_change_recomputes_only_mari(bench_env) -> None:
    _setup(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    calls = {
        "ri": 0,
        "ri_prepared": 0,
        "mari": 0,
        "mari_prepared": 0,
        "croma": 0,
        "knn": 0,
        "knn_prepared": 0,
    }
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_dataset_wide
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_knn_prepared = bm.RI._knn_balanced_accuracy_by_k_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_dataset_wide
    original_mari_prepared = bm.MaRI._compute_artifacts_from_prepared_subsets
    original_croma_compute = bm.CRoMa.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_artifacts(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_artifacts(*args, **kwargs)

    def wrapped_ri_prepared(*args, **kwargs):
        calls["ri_prepared"] += 1
        return original_ri_prepared(*args, **kwargs)

    def wrapped_knn_prepared(*args, **kwargs):
        calls["knn_prepared"] += 1
        return original_knn_prepared(*args, **kwargs)

    def wrapped_mari_artifacts(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_artifacts(*args, **kwargs)

    def wrapped_mari_prepared(*args, **kwargs):
        calls["mari_prepared"] += 1
        return original_mari_prepared(*args, **kwargs)

    def wrapped_croma_compute(*args, **kwargs):
        calls["croma"] += 1
        return original_croma_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    mp = bench_env._monkeypatch
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.RI, "_knn_balanced_accuracy_by_k_from_prepared_subsets", wrapped_knn_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_mari_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_subsets", wrapped_mari_prepared)
    mp.setattr(bm.CRoMa, "compute", wrapped_croma_compute)
    mp.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert bench_env.run("toy", "k-star", "--tau", "0.3", "--progress", "off") == 0

    assert calls["mari"] > 0
    assert calls["ri"] == 0
    assert calls["croma"] == 0
    assert calls["knn"] == 0


def test_croma_search_change_recomputes_only_croma(bench_env) -> None:
    _setup(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    calls = {
        "ri": 0,
        "ri_prepared": 0,
        "mari": 0,
        "mari_prepared": 0,
        "croma": 0,
        "knn": 0,
        "knn_prepared": 0,
    }
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_dataset_wide
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_knn_prepared = bm.RI._knn_balanced_accuracy_by_k_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_dataset_wide
    original_mari_prepared = bm.MaRI._compute_artifacts_from_prepared_subsets
    original_croma_compute = bm.CRoMa.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_artifacts(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_artifacts(*args, **kwargs)

    def wrapped_ri_prepared(*args, **kwargs):
        calls["ri_prepared"] += 1
        return original_ri_prepared(*args, **kwargs)

    def wrapped_knn_prepared(*args, **kwargs):
        calls["knn_prepared"] += 1
        return original_knn_prepared(*args, **kwargs)

    def wrapped_mari_artifacts(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_artifacts(*args, **kwargs)

    def wrapped_mari_prepared(*args, **kwargs):
        calls["mari_prepared"] += 1
        return original_mari_prepared(*args, **kwargs)

    def wrapped_croma_compute(*args, **kwargs):
        calls["croma"] += 1
        return original_croma_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    mp = bench_env._monkeypatch
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.RI, "_knn_balanced_accuracy_by_k_from_prepared_subsets", wrapped_knn_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_mari_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_subsets", wrapped_mari_prepared)
    mp.setattr(bm.CRoMa, "compute", wrapped_croma_compute)
    mp.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert bench_env.run("toy", "k-star", "--croma-start-k", "300", "--progress", "off") == 0

    assert calls["croma"] > 0
    assert calls["ri"] == 0
    assert calls["mari"] == 0
    assert calls["knn"] == 0


def test_k_values_change_recomputes_knn_ri_mari_not_croma(bench_env) -> None:
    _setup(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    calls = {
        "ri": 0,
        "ri_prepared": 0,
        "mari": 0,
        "mari_prepared": 0,
        "croma": 0,
        "knn": 0,
    }
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_dataset_wide
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_dataset_wide
    original_mari_prepared = bm.MaRI._compute_artifacts_from_prepared_subsets
    original_croma_compute = bm.CRoMa.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_artifacts(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_artifacts(*args, **kwargs)

    def wrapped_ri_prepared(*args, **kwargs):
        calls["ri_prepared"] += 1
        return original_ri_prepared(*args, **kwargs)

    def wrapped_mari_artifacts(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_artifacts(*args, **kwargs)

    def wrapped_mari_prepared(*args, **kwargs):
        calls["mari_prepared"] += 1
        return original_mari_prepared(*args, **kwargs)

    def wrapped_croma_compute(*args, **kwargs):
        calls["croma"] += 1
        return original_croma_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    mp = bench_env._monkeypatch
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_mari_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_subsets", wrapped_mari_prepared)
    mp.setattr(bm.CRoMa, "compute", wrapped_croma_compute)
    mp.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    # A different k ceiling changes the RI/MaRI/kNN cache keys but leaves the CRoMa
    # m-sweep (which never depends on k) untouched.
    assert bench_env.run("toy", "k-star", "--k-max", "5", "--progress", "off") == 0

    assert calls["ri"] > 0
    assert calls["mari"] > 0
    assert calls["knn"] > 0
    assert calls["croma"] == 0


def test_evaluation_design_change_recomputes_all_artifacts(bench_env) -> None:
    # Two benchmarks over the same tileset differing only in evaluation design get
    # independent run directories, so the second (paired) run cannot reuse the first's
    # (dataset-wide) metric cache: every metric recomputes.
    _setup(bench_env, name="toy-wide")
    bench_env.register(
        "toy-paired",
        tileset=TOY_TILESET,
        manifest=_toy_manifest(),
        design="paired_2x2",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    assert bench_env.run("toy-wide", "k-star", "--progress", "off") == 0

    calls = {
        "ri": 0,
        "ri_prepared": 0,
        "mari": 0,
        "mari_prepared": 0,
        "croma": 0,
        "knn": 0,
        "knn_prepared": 0,
    }
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_dataset_wide
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_knn_prepared = bm.RI._knn_balanced_accuracy_by_k_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_dataset_wide
    original_mari_prepared = bm.MaRI._compute_artifacts_from_prepared_subsets
    original_croma_compute = bm.CRoMa.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_artifacts(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_artifacts(*args, **kwargs)

    def wrapped_ri_prepared(*args, **kwargs):
        calls["ri_prepared"] += 1
        return original_ri_prepared(*args, **kwargs)

    def wrapped_knn_prepared(*args, **kwargs):
        calls["knn_prepared"] += 1
        return original_knn_prepared(*args, **kwargs)

    def wrapped_mari_artifacts(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_artifacts(*args, **kwargs)

    def wrapped_mari_prepared(*args, **kwargs):
        calls["mari_prepared"] += 1
        return original_mari_prepared(*args, **kwargs)

    def wrapped_croma_compute(*args, **kwargs):
        calls["croma"] += 1
        return original_croma_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    mp = bench_env._monkeypatch
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.RI, "_knn_balanced_accuracy_by_k_from_prepared_subsets", wrapped_knn_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_mari_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_subsets", wrapped_mari_prepared)
    mp.setattr(bm.CRoMa, "compute", wrapped_croma_compute)
    mp.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert bench_env.run("toy-paired", "k-star", "--progress", "off") == 0

    assert calls["ri"] > 0 or calls["ri_prepared"] > 0
    assert calls["mari"] > 0 or calls["mari_prepared"] > 0
    assert calls["croma"] > 0
    assert calls["knn"] > 0 or calls["knn_prepared"] > 0


def test_recompute_metrics_flag_forces_all(bench_env) -> None:
    _setup(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    calls = {"ri": 0, "mari": 0, "croma": 0, "knn": 0}
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_dataset_wide
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_dataset_wide
    original_croma_compute = bm.CRoMa.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_artifacts(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_artifacts(*args, **kwargs)

    def wrapped_mari_artifacts(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_artifacts(*args, **kwargs)

    def wrapped_croma_compute(*args, **kwargs):
        calls["croma"] += 1
        return original_croma_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    mp = bench_env._monkeypatch
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_ri_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_mari_artifacts)
    mp.setattr(bm.CRoMa, "compute", wrapped_croma_compute)
    mp.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert bench_env.run("toy", "k-star", "--recompute-metrics", "--progress", "off") == 0

    assert calls["ri"] > 0
    assert calls["mari"] > 0
    assert calls["croma"] > 0
    assert calls["knn"] > 0


def test_cold_cache_uses_one_shared_scoring_pass_per_metric(bench_env) -> None:
    _setup(bench_env)

    calls = {"ri": 0, "mari": 0}
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_dataset_wide
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_dataset_wide

    def wrapped_ri_artifacts(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_artifacts(*args, **kwargs)

    def wrapped_mari_artifacts(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_artifacts(*args, **kwargs)

    def fail_public_metric_api(*args, **kwargs):
        raise AssertionError(
            "benchmark should use the shared _compute_artifacts path on cold-cache RI/MaRI misses"
        )

    mp = bench_env._monkeypatch
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_ri_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_dataset_wide", wrapped_mari_artifacts)
    mp.setattr(bm.RI, "compute", fail_public_metric_api)
    mp.setattr(bm.RI, "compute_curve", fail_public_metric_api)
    mp.setattr(bm.MaRI, "compute", fail_public_metric_api)
    mp.setattr(bm.MaRI, "compute_curve", fail_public_metric_api)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    assert calls["ri"] == 1
    assert calls["mari"] == 1
