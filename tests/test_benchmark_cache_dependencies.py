import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm
from support_schema import RETIRED_AGGREGATE_FIELD, RETIRED_FLAGSHIP_AGGREGATE_FIELD


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
    """Embed a single-model toy tileset and register an all-rows benchmark."""
    tileset = _tileset_manifest()
    bench_env.write_tileset(TOY_TILESET, tileset, {"M1": _toy_features()[: len(tileset)]})
    bench_env.register(
        name,
        tileset=TOY_TILESET,
        manifest=_toy_manifest(),
        design="all",
        k_max=k_max,
        confounder_column="scanner_vendor",
    )


def _spy_on_croma_compute(bench_env) -> dict[str, int]:
    calls = {"croma": 0}
    original_croma_compute = bm.CRoMa.compute

    def wrapped_croma_compute(*args, **kwargs):
        calls["croma"] += 1
        return original_croma_compute(*args, **kwargs)

    bench_env._monkeypatch.setattr(bm.CRoMa, "compute", wrapped_croma_compute)
    return calls


def test_median_k_roster_change_rekeys_only_operating_point_artifacts(bench_env) -> None:
    """A changed panel median must not reuse retained models' old-k artifacts."""
    tileset = _tileset_manifest()
    base = _toy_features()[: len(tileset)]
    bench_env.write_tileset(
        TOY_TILESET,
        tileset,
        {
            "M1": base,
            "M2": base[:, [1, 0, 2, 3]],
            "M3": np.column_stack([base, np.full(len(base), 2.0)]),
        },
    )
    bench_env.register(
        "toy",
        tileset=TOY_TILESET,
        manifest=_toy_manifest(),
        design="all",
        k_max=3,
        confounder_column="scanner_vendor",
    )

    biological_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=int)

    def fake_knn_balanced_accuracy_by_k(
        *,
        features: np.ndarray,
        labels: np.ndarray,
        group_ids: np.ndarray,
        k_values: list[int],
        warn_context: str,
    ) -> dict[int, float]:
        del group_ids, warn_context
        if not np.array_equal(labels, biological_labels):
            return {1: 0.55, 2: 0.65, 3: 0.75}
        marker = int(np.argmax(features[0]))
        if marker == 0:  # M1: k*=1
            return {1: 0.90, 2: 0.70, 3: 0.60}
        return {1: 0.60, 2: 0.70, 3: 0.90}  # M2/M3: k*=3

    bench_env._monkeypatch.setattr(
        bm, "_knn_balanced_accuracy_by_k", fake_knn_balanced_accuracy_by_k
    )

    assert bench_env.run("toy", "median-k", "--models", "M1,M2,M3", "--progress", "off") == 0
    results_dir = bench_env.results_dir("toy", "median-k")
    first_metrics = pd.read_csv(results_dir / "metrics.csv")
    assert set(first_metrics["k"]) == {3}

    artifacts = results_dir / "cache" / "artifacts"
    operating_artifacts = {
        "ri_summary",
        "ri_samples",
        "ri_samples_aligned",
        "ri_undefined_types",
        "mari_summary",
        "mari_samples",
        "mari_samples_aligned",
        "mari_undefined_types",
        "tau_assessment",
        "mari_curve",
    }
    operating_before = {
        (artifact, model): set((artifacts / artifact / model).glob("*"))
        for artifact in operating_artifacts
        for model in ("M1", "M2")
    }
    assert all(len(paths) == 1 for paths in operating_before.values())
    unaffected_artifacts = {
        "knn_bio_curve",
        "knn_confounder_curve",
        "ri_curve",
        "croma_m_sweep",
        "croma_headline_samples",
        "croma_samples_aligned_by_m",
    }
    unaffected_before = {
        (artifact, model): {
            path: (path.stat().st_ino, path.stat().st_mtime_ns)
            for path in (artifacts / artifact / model).glob("*")
        }
        for artifact in unaffected_artifacts
        for model in ("M1", "M2")
    }
    assert all(len(files) == 1 for files in unaffected_before.values())

    assert bench_env.run("toy", "median-k", "--models", "M1,M2", "--progress", "off") == 0

    second_metrics = pd.read_csv(results_dir / "metrics.csv")
    assert set(second_metrics["k"]) == {1}
    for (artifact, model), old_paths in operating_before.items():
        new_paths = set((artifacts / artifact / model).glob("*"))
        assert len(new_paths) == 2
        assert len(new_paths - old_paths) == 1, f"{artifact}/{model} was not re-keyed"
    for (artifact, model), old_files in unaffected_before.items():
        new_files = {
            path: (path.stat().st_ino, path.stat().st_mtime_ns)
            for path in (artifacts / artifact / model).glob("*")
        }
        assert new_files == old_files, f"{artifact}/{model} was recomputed"


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
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_all_rows
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_knn_prepared = bm.RI._knn_balanced_accuracy_by_k_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_all_rows
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
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_all_rows", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.RI, "_knn_balanced_accuracy_by_k_from_prepared_subsets", wrapped_knn_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_all_rows", wrapped_mari_artifacts)
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
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_all_rows
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_knn_prepared = bm.RI._knn_balanced_accuracy_by_k_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_all_rows
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
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_all_rows", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.RI, "_knn_balanced_accuracy_by_k_from_prepared_subsets", wrapped_knn_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_all_rows", wrapped_mari_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_subsets", wrapped_mari_prepared)
    mp.setattr(bm.CRoMa, "compute", wrapped_croma_compute)
    mp.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert bench_env.run("toy", "k-star", "--croma-start-k", "300", "--progress", "off") == 0

    assert calls["croma"] > 0
    assert calls["ri"] == 0
    assert calls["mari"] == 0
    assert calls["knn"] == 0


def test_partial_croma_cache_is_rejected_and_recomputed(bench_env) -> None:
    """A pre-total-support cache cannot bypass the public CRoMa contract."""
    _setup(bench_env)
    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    cache_artifacts = bench_env.results_dir("toy") / "cache" / "artifacts"
    aligned_path = next((cache_artifacts / "croma_samples_aligned_by_m" / "M1").glob("*.npy"))
    aligned = np.load(aligned_path)
    aligned[0, 0] = np.nan
    np.save(aligned_path, aligned)

    calls = _spy_on_croma_compute(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0
    assert calls["croma"] == 1

    metrics = pd.read_csv(bench_env.results_dir("toy") / "metrics.csv")
    assert metrics.loc[0, "croma"] == 0.9657262993950306
    assert RETIRED_FLAGSHIP_AGGREGATE_FIELD not in metrics.columns
    per_sample = pd.read_csv(bench_env.results_dir("toy") / "per_sample_metrics.csv")
    np.testing.assert_allclose(
        per_sample["croma_m1"],
        [
            0.9619974087755232,
            0.970965547572644,
            0.970965547572644,
            0.9619974087755232,
            0.9694551900145382,
            0.95136476856489,
            0.95136476856489,
            0.9694551900145382,
        ],
        rtol=0.0,
        atol=1e-15,
    )


def test_truncated_croma_headline_cache_is_rejected_and_recomputed(bench_env) -> None:
    _setup(bench_env)
    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    cache_artifacts = bench_env.results_dir("toy") / "cache" / "artifacts"
    headline_path = next((cache_artifacts / "croma_headline_samples" / "M1").glob("*.npy"))
    np.save(headline_path, np.load(headline_path)[:-1])

    calls = _spy_on_croma_compute(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0
    assert calls["croma"] == 1


def test_legacy_croma_coverage_payload_is_not_a_cache_hit(bench_env) -> None:
    _setup(bench_env)
    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    cache_artifacts = bench_env.results_dir("toy") / "cache" / "artifacts"
    payload_path = next((cache_artifacts / "croma_m_sweep" / "M1").glob("*.json"))
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["by_m"]["1"][RETIRED_FLAGSHIP_AGGREGATE_FIELD] = 0.0
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    calls = _spy_on_croma_compute(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0
    assert calls["croma"] == 1


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
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_all_rows
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_all_rows
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
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_all_rows", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_all_rows", wrapped_mari_artifacts)
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
    # (all-rows) metric cache: every metric recomputes.
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
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_all_rows
    original_ri_prepared = bm.RI._compute_artifacts_from_prepared_subsets
    original_knn_prepared = bm.RI._knn_balanced_accuracy_by_k_from_prepared_subsets
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_all_rows
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
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_all_rows", wrapped_ri_artifacts)
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_subsets", wrapped_ri_prepared)
    mp.setattr(bm.RI, "_knn_balanced_accuracy_by_k_from_prepared_subsets", wrapped_knn_prepared)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_all_rows", wrapped_mari_artifacts)
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
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_all_rows
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_all_rows
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
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_all_rows", wrapped_ri_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_all_rows", wrapped_mari_artifacts)
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
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_all_rows
    original_mari_artifacts = bm.MaRI._compute_artifacts_from_prepared_all_rows

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
    mp.setattr(bm.RI, "_compute_artifacts_from_prepared_all_rows", wrapped_ri_artifacts)
    mp.setattr(bm.MaRI, "_compute_artifacts_from_prepared_all_rows", wrapped_mari_artifacts)
    mp.setattr(bm.RI, "compute", fail_public_metric_api)
    mp.setattr(bm.RI, "compute_curve", fail_public_metric_api)
    mp.setattr(bm.MaRI, "compute", fail_public_metric_api)
    mp.setattr(bm.MaRI, "compute_curve", fail_public_metric_api)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    assert calls["ri"] == 1
    assert calls["mari"] == 1


def test_metric_summary_caches_store_positive_support(bench_env) -> None:
    """New RI/MaRI summaries identify the supported share without complementation."""
    _setup(bench_env)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    cache_artifacts = bench_env.results_dir("toy") / "cache" / "artifacts"
    for artifact_name in ("ri_summary", "mari_summary"):
        payload_path = next((cache_artifacts / artifact_name / "M1").glob("*.json"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["support"] == 0.25
        assert RETIRED_AGGREGATE_FIELD not in payload


def test_metric_summary_cache_rejects_retired_aggregate() -> None:
    payload = {
        "k": 1,
        "value": 0.5,
        "std": 0.0,
        "support": 0.25,
        "ss_dominated_undefined_frac": 0.75,
        "oo_dominated_undefined_frac": 0.0,
        "mixed_undefined_frac": 0.0,
        "median_value": 0.5,
        "q_alpha": 0.5,
        "ltm_alpha": 0.5,
        "evaluation_design": "all",
        "evaluation_unit": "sample",
        RETIRED_AGGREGATE_FIELD: 0.75,
    }

    assert bm._summary_from_payload(payload) is None


def test_legacy_summary_without_support_is_not_a_cache_hit(bench_env) -> None:
    """A pre-positive-schema RI summary is recomputed instead of silently upgraded."""
    _setup(bench_env)
    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    cache_artifacts = bench_env.results_dir("toy") / "cache" / "artifacts"
    payload_path = next((cache_artifacts / "ri_summary" / "M1").glob("*.json"))
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload.pop("support")
    payload["value"] = 99.0
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    calls = {"ri": 0}
    original_ri_artifacts = bm.RI._compute_artifacts_from_prepared_all_rows

    def wrapped_ri_artifacts(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_artifacts(*args, **kwargs)

    bench_env._monkeypatch.setattr(
        bm.RI, "_compute_artifacts_from_prepared_all_rows", wrapped_ri_artifacts
    )

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0
    assert calls["ri"] == 1
    metrics = pd.read_csv(bench_env.results_dir("toy") / "metrics.csv")
    assert metrics.loc[0, "ri"] == 1.0
