import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm
import run_config
from croma.types import CRoMaResult


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


def _toy_features(model_name: str = "M1") -> np.ndarray:
    base = np.array(
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
    if model_name == "M2":
        return base[:, [1, 0, 2, 3]]
    return base


TOY_TILESET = "toy-tiles"


def _tileset_manifest() -> pd.DataFrame:
    """The tileset: one row per distinct tile, in embedding row order."""
    manifest = _toy_manifest().drop(columns=["subset", "dataset"])
    return manifest.drop_duplicates(subset=["sample_id", "image_path"]).reset_index(drop=True)


def _setup(
    bench_env,
    *,
    models: list[str],
    name: str = "toy",
    design: str = "all",
    k_max: int = 3,
) -> None:
    """Embed a toy tileset and register a benchmark that views all of it."""
    tileset = _tileset_manifest()
    bench_env.write_tileset(
        TOY_TILESET, tileset, {m: _toy_features(m)[: len(tileset)] for m in models}
    )
    bench_env.register(
        name,
        tileset=TOY_TILESET,
        manifest=_toy_manifest(),
        design=design,
        k_max=k_max,
        confounder_column="scanner_vendor",
    )


# Flags the compute-only rewrite dropped. Every one must make _parse_args() exit 2.
_REMOVED_FLAGS = [
    ["--manifest", "toy.csv"],
    ["--output-dir", "out"],
    ["--confounder-column", "scanner_vendor"],
    ["--evaluation-design", "all"],
    ["--use-median-k"],
    ["--force-embed"],
    ["--device", "cpu"],
    ["--batch-size", "8"],
    ["--num-workers", "2"],
    ["--dataset-name", "override"],
    ["--exclude-confounder", "C1"],
]


@pytest.mark.parametrize("argv_tail", _REMOVED_FLAGS)
def test_benchmark_rejects_removed_flags(monkeypatch, argv_tail: list[str]) -> None:
    argv = ["benchmark.py", "--benchmark", "toy", "--protocol", "k-star", *argv_tail]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        bm._parse_args()
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv_tail",
    [
        ["--k-candidates", "1,3"],
        ["--continuous-k-sweep-max", "4"],
    ],
)
def test_benchmark_rejects_removed_k_sweep_flags(monkeypatch, argv_tail: list[str]) -> None:
    argv = [
        "benchmark.py",
        "--benchmark",
        "toy",
        "--protocol",
        "k-star",
        "--k-max",
        "3",
        *argv_tail,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        bm._parse_args()
    assert excinfo.value.code == 2


def test_median_k_prepass_exits_when_a_model_npy_is_missing(bench_env, monkeypatch) -> None:
    # Every model must contribute to the shared median; a model whose embeddings vanish
    # before the pre-pass reads them must fail loudly (naming the model), never silently
    # shift the median or fall back to per-model k*.
    import layout

    models = ["M1", "M2"]
    _setup(bench_env, models=models)

    # The availability glob still sees both .npy, so the roster check passes and the run
    # enters the median-k pre-pass. There, embedding_path for M2 resolves to a file that
    # does not exist, exercising the pre-pass's own existence guard.
    real_embedding_path = layout.embedding_path

    def fake_embedding_path(tileset: str, model: str):
        if model == "M2":
            return layout.embeddings_dir(tileset) / "vanished-M2.npy"
        return real_embedding_path(tileset, model)

    monkeypatch.setattr(layout, "embedding_path", fake_embedding_path)

    with pytest.raises(SystemExit) as excinfo:
        bench_env.run("toy", "median-k", "--progress", "off")
    assert "M2" in str(excinfo.value)


def test_benchmark_uses_benchmark_name_for_dataset(bench_env) -> None:
    # The manifest may carry a stale ``dataset`` column; the run must label rows with the
    # registered benchmark name, not the manifest stem or the manifest's own column.
    _setup(bench_env, models=["M1"], name="release-ready")

    assert bench_env.run("release-ready", "k-star", "--progress", "off") == 0

    metrics_df = pd.read_csv(bench_env.results_dir("release-ready") / "metrics.csv")
    assert set(metrics_df["dataset"]) == {"release-ready"}


def test_benchmark_all_rows_outputs_sample_level_rows(bench_env) -> None:
    models = ["M1", "M2"]
    _setup(bench_env, models=models)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    results_dir = bench_env.results_dir("toy")
    dataset_dir = results_dir.parent
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    k_sweep_df = pd.read_csv(results_dir / "k_sweep_metrics.csv")
    per_sample_df = pd.read_csv(results_dir / "per_sample_metrics.csv")
    per_model_dir = results_dir / "per_sample_metrics_by_model"

    assert set(metrics_df["model"]) == set(models)
    assert set(metrics_df["evaluation_design"]) == {"all"}
    assert set(metrics_df["evaluation_unit"]) == {"sample"}
    assert set(metrics_df["k_max"]) == {3}
    assert set(k_sweep_df["evaluation_design"]) == {"all"}
    assert set(k_sweep_df["evaluation_unit"]) == {"sample"}
    assert set(k_sweep_df["k_max"]) == {3}
    assert set(per_sample_df["evaluation_design"]) == {"all"}
    assert set(per_sample_df["evaluation_unit"]) == {"sample"}
    assert set(per_sample_df["subset"]) == {"dataset"}

    expected_rows = len(_toy_manifest()) * len(models)
    assert len(per_sample_df) == expected_rows
    for model in models:
        model_rows = per_sample_df[per_sample_df["model"] == model].sort_values("occurrence_index")
        assert model_rows["occurrence_index"].tolist() == list(range(len(_toy_manifest())))
        assert model_rows["sample_index"].tolist() == list(range(len(_toy_manifest())))
        model_df = pd.read_csv(per_model_dir / f"{model}.csv")
        assert model_df["occurrence_index"].tolist() == list(range(len(_toy_manifest())))
        assert set(model_df["model"]) == {model}

    for path in (
        results_dir / "metrics.csv",
        results_dir / "k_sweep_metrics.csv",
        results_dir / "croma_m_sweep_metrics.csv",
        results_dir / "per_sample_metrics.csv",
        per_model_dir / "M1.csv",
        per_model_dir / "M2.csv",
    ):
        assert path.exists(), f"Missing output: {path}"

    # The compute driver is plot-free: it writes no figures and no plots/ directory.
    # Rendering is scripts/bench/render.py's responsibility (see test_render.py).
    assert not (dataset_dir / "plots").exists()


def test_benchmark_rows_use_shared_support_schema_with_temporary_legacy_columns(
    bench_env,
) -> None:
    _setup(bench_env, models=["M1"])

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    results_dir = bench_env.results_dir("toy")
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    m_sweep_df = pd.read_csv(results_dir / "croma_m_sweep_metrics.csv")
    row = metrics_df.iloc[0]

    assert row[
        [
            "support",
            "ss_dominated_undefined_frac",
            "oo_dominated_undefined_frac",
            "mixed_undefined_frac",
        ]
    ].tolist() == [0.25, 0.75, 0.0, 0.0]
    assert {
        "ri_undefined_frac",
        "ri_ss_dominated_undefined_frac",
        "ri_oo_dominated_undefined_frac",
        "ri_mixed_undefined_frac",
        "mari_undefined_frac",
        "mari_ss_dominated_undefined_frac",
        "mari_oo_dominated_undefined_frac",
        "mari_mixed_undefined_frac",
    }.issubset(metrics_df.columns)
    assert "croma_undefined_frac" not in metrics_df.columns
    assert "croma_undefined_frac" not in m_sweep_df.columns


def test_benchmark_progress_reports_shared_support_and_undefined_causes(bench_env, capsys) -> None:
    _setup(bench_env, models=["M1"])

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    output = capsys.readouterr().out
    assert (
        "[benchmark] RI/MaRI support=25.0%; undefined causes: " "SS=75.0%, OO=0.0%, mixed=0.0%"
    ) in output
    assert "undefined samples:" not in output


def test_benchmark_paired_outputs_occurrence_level_rows(bench_env) -> None:
    manifest = _toy_manifest()
    _setup(bench_env, models=["M1"], design="paired_2x2")

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    results_dir = bench_env.results_dir("toy")
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    per_sample_df = pd.read_csv(results_dir / "per_sample_metrics.csv")

    assert set(metrics_df["evaluation_design"]) == {"paired_2x2"}
    assert set(metrics_df["evaluation_unit"]) == {"occurrence"}
    assert set(per_sample_df["evaluation_design"]) == {"paired_2x2"}
    assert set(per_sample_df["evaluation_unit"]) == {"occurrence"}
    assert set(per_sample_df["subset"]) == {"pair1", "pair2"}
    assert per_sample_df["occurrence_index"].tolist() == list(range(len(manifest)))
    assert int((per_sample_df["sample_id"] == "s0").sum()) == 2


def test_benchmark_writes_per_sample_artifact_with_undefined_rows(bench_env) -> None:
    manifest = _toy_manifest()
    model = "M1"

    class _FakeRobustnessResult:
        def __init__(self, k: int, values: list[float], undef_types: list[int]) -> None:
            aligned = np.asarray(values, dtype=float)
            informative = np.isfinite(aligned)
            self.dataset = "toy"
            self.k = int(k)
            self.value = 0.5
            self.std = 0.0
            self.n_pairs = 1
            self.pair_values = np.asarray([0.5], dtype=float)
            self.sample_values = aligned[informative]
            self.sample_values_aligned = aligned
            self.sample_undefined_types = np.asarray(undef_types, dtype=int)
            self.undefined_frac = float((~informative).mean())
            self.support = float(informative.mean())
            self.ss_dominated_undefined_frac = float(np.mean(self.sample_undefined_types == 1))
            self.oo_dominated_undefined_frac = float(np.mean(self.sample_undefined_types == 2))
            self.mixed_undefined_frac = float(np.mean(self.sample_undefined_types == 3))
            self.median_value = 0.4321
            self.q_alpha = -0.1234
            self.ltm_alpha = -0.2345
            self.evaluation_design = "all"
            self.evaluation_unit = "sample"

    class _FakeCRoMaResult:
        def __init__(self, m: int, values: list[float]) -> None:
            aligned = np.asarray(values, dtype=float)
            informative = np.isfinite(aligned)
            self.dataset = "toy"
            self.m = int(m)
            self.value = 1.1
            self.std = 0.0
            self.n_pairs = 1
            self.pair_values = np.asarray([1.1], dtype=float)
            self.sample_values = aligned[informative]
            self.sample_values_aligned = aligned
            self.occurrence_defined_mask = informative
            self.undefined_frac = float((~informative).mean())
            self.evaluation_design = "all"
            self.evaluation_unit = "sample"
            self.k_start = 200
            self.k_final = 200
            self.retries = 0
            self.alpha = 0.10
            self.q_alpha = 0.5
            self.ltm_alpha = 0.4
            self.f0 = float((self.sample_values <= 0.0).mean())

    def fake_ri_compute_artifacts(
        *,
        prepared_neighbors: object,
        dataset_name: str,
        k_values: list[int],
        selected_k: int,
        include_selected_result: bool,
        warn_selected_result: bool,
    ) -> SimpleNamespace:
        assert include_selected_result is True
        assert warn_selected_result is True
        result = _FakeRobustnessResult(
            k=int(selected_k),
            values=[0.10, np.nan, 0.30, np.nan, 0.50, 0.60, np.nan, 0.80],
            undef_types=[0, 1, 0, 2, 0, 0, 3, 0],
        )
        return SimpleNamespace(curve={int(k): 0.5 for k in k_values}, result=result)

    def fake_mari_compute_artifacts(
        *,
        prepared_neighbors: object,
        dataset_name: str,
        k_values: list[int],
        selected_k: int,
        include_selected_result: bool,
        warn_selected_result: bool,
        tau: float,
    ) -> SimpleNamespace:
        assert include_selected_result is True
        assert warn_selected_result is True
        result = _FakeRobustnessResult(
            k=int(selected_k),
            values=[0.20, np.nan, 0.35, np.nan, 0.55, 0.65, np.nan, 0.85],
            undef_types=[0, 3, 0, 1, 0, 0, 2, 0],
        )
        return SimpleNamespace(curve={int(k): 0.5 for k in k_values}, result=result)

    def fake_croma_compute(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        confounder_column: str,
        evaluation_design: str,
        m: list[int],
        alpha: float,
        start_k: int,
        k_growth_factor: float,
    ) -> dict[int, _FakeCRoMaResult]:
        assert evaluation_design == "all"
        return {
            1: _FakeCRoMaResult(1, [1.1, 0.9, 1.2, 1.3, 1.4, 1.25, 0.8, 1.0]),
            2: _FakeCRoMaResult(2, [1.2, 1.0, 1.1, 1.35, 1.45, 1.25, 0.85, 1.05]),
        }

    _setup(bench_env, models=[model])
    import benchmark as bm_mod

    bench_env._monkeypatch.setattr(
        bm_mod.RI,
        "_compute_artifacts_from_prepared_all_rows",
        fake_ri_compute_artifacts,
    )
    bench_env._monkeypatch.setattr(
        bm_mod.MaRI,
        "_compute_artifacts_from_prepared_all_rows",
        fake_mari_compute_artifacts,
    )
    bench_env._monkeypatch.setattr(bm_mod.CRoMa, "compute", fake_croma_compute)

    assert bench_env.run("toy", "k-star", "--croma-m-max", "2", "--progress", "off") == 0

    metrics_df = pd.read_csv(bench_env.results_dir("toy") / "metrics.csv")
    assert metrics_df.loc[0, ["ri_median", "ri_q_alpha", "ri_ltm_alpha"]].tolist() == [
        0.4321,
        -0.1234,
        -0.2345,
    ]
    assert metrics_df.loc[0, ["mari_median", "mari_q_alpha", "mari_ltm_alpha"]].tolist() == [
        0.4321,
        -0.1234,
        -0.2345,
    ]

    per_sample_df = pd.read_csv(bench_env.results_dir("toy") / "per_sample_metrics.csv")
    assert len(per_sample_df) == len(manifest)
    assert per_sample_df["sample_index"].tolist() == list(range(len(manifest)))
    assert per_sample_df["ri_defined"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert per_sample_df["mari_defined"].tolist() == [
        True,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
    ]
    assert per_sample_df["ri_undefined_type"].tolist() == [0, 1, 0, 2, 0, 0, 3, 0]
    assert per_sample_df["mari_undefined_type"].tolist() == [0, 3, 0, 1, 0, 0, 2, 0]
    assert np.isnan(per_sample_df.loc[1, "ri"])
    assert np.isnan(per_sample_df.loc[3, "mari"])
    assert per_sample_df.loc[2, "croma_m1"] == 1.2
    assert per_sample_df.loc[5, "croma_m2"] == 1.25


def test_benchmark_k_max_uses_full_range(bench_env) -> None:
    models = ["M1", "M2"]
    _setup(bench_env, models=models, k_max=4)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    df = pd.read_csv(bench_env.results_dir("toy") / "k_sweep_metrics.csv")
    assert set(df["model"]) == set(models)
    assert set(df["k"]) == {1, 2, 3, 4}
    assert set(df["k_max"]) == {4}


def test_benchmark_can_select_different_confounder_k(bench_env) -> None:
    def fake_knn_balanced_accuracy_by_k(
        *,
        features: np.ndarray,
        labels: np.ndarray,
        group_ids: np.ndarray,
        k_values: list[int],
        warn_context: str,
    ) -> dict[int, float]:
        # The confounder (scanner vendor) alternates V1/V2 -> factorized [0,1,0,1,...];
        # its balanced accuracy peaks at k=3. The biological label is [0,0,1,1,...] and
        # peaks at k=1. This decouples the two selected-k values.
        if np.array_equal(labels, np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)):
            return {int(k): v for k, v in zip(k_values, [0.60, 0.75, 0.92], strict=False)}
        return {int(k): v for k, v in zip(k_values, [0.90, 0.70, 0.68], strict=False)}

    _setup(bench_env, models=["M1"])
    import benchmark as bm_mod

    bench_env._monkeypatch.setattr(
        bm_mod, "_knn_balanced_accuracy_by_k", fake_knn_balanced_accuracy_by_k
    )

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    results_dir = bench_env.results_dir("toy")
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    k_sweep_df = pd.read_csv(results_dir / "k_sweep_metrics.csv")

    assert int(metrics_df.loc[0, "k"]) == 1
    assert int(metrics_df.loc[0, "selected_k_confounder"]) == 3
    assert int(k_sweep_df.loc[0, "selected_k"]) == 1
    assert int(k_sweep_df.loc[0, "selected_k_confounder"]) == 3


# --- k-grid: dense (default) vs PathoROB's sparse grid -------------------------------


def test_dense_grid_is_every_integer_up_to_k_max() -> None:
    assert bm._resolve_sweep_k_values(10) == list(range(1, 11))
    assert bm._resolve_sweep_k_values(10, "dense") == list(range(1, 11))


@pytest.mark.parametrize("k_max", [100, 600, 1000, 1200])
def test_sparse_grid_reproduces_pathorob_get_k_values(k_max: int) -> None:
    """PathoROB: ``[1,3,5,7,9] + np.arange(11, max_k, 10)`` -- k_max exclusive in the tail."""
    expected = [1, 3, 5, 7, 9] + list(range(11, k_max, 10))
    assert bm._resolve_sweep_k_values(k_max, "sparse") == expected


def test_sparse_grid_excludes_k_max_in_the_tail() -> None:
    # Upstream's arange stops before max_k, so a ceiling of 100 sweeps only up to 91.
    values = bm._resolve_sweep_k_values(100, "sparse")
    assert values[-1] == 91
    assert 100 not in values


# --- median-k: the shared operating point must be a k the sweep scored ----------------


def test_shared_operating_k_matches_np_median_for_an_odd_model_count() -> None:
    assert bm._shared_operating_k({"a": 5, "b": 11, "c": 61}) == 11


def test_shared_operating_k_is_always_some_models_own_kstar() -> None:
    """An even model count must not average two central k* into an unswept value.

    ``np.median({5, 5, 7, 7})`` is ``6``, which PathoROB's sparse grid never scores; the
    run then fails with "selected_k=6 is not available in scored-by-k results".
    """
    per_model = {"a": 5, "b": 5, "c": 7, "d": 7}
    shared = bm._shared_operating_k(per_model)
    assert shared == 5
    assert shared in set(per_model.values())
    assert shared in bm._resolve_sweep_k_values(100, "sparse")


@pytest.mark.parametrize("grid", ["dense", "sparse"])
def test_shared_operating_k_lands_on_the_grid_for_any_model_count(grid: str) -> None:
    k_values = bm._resolve_sweep_k_values(100, grid)
    for n_models in range(1, 12):
        per_model = {f"m{i}": k_values[i % len(k_values)] for i in range(n_models)}
        assert bm._shared_operating_k(per_model) in k_values


def test_shared_operating_k_rejects_an_empty_prepass() -> None:
    with pytest.raises(ValueError, match="no per-model"):
        bm._shared_operating_k({})


def test_sparse_grid_degrades_gracefully_for_small_ceilings() -> None:
    assert bm._resolve_sweep_k_values(5, "sparse") == [1, 3, 5]
    assert bm._resolve_sweep_k_values(1, "sparse") == [1]


def test_unknown_k_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown k grid"):
        bm._resolve_sweep_k_values(10, "bogus")


def test_k_grid_sparse_is_recorded_in_the_swept_k_values(bench_env) -> None:
    """The grid is part of the protocol, so the run must record which k it actually swept."""
    _setup(bench_env, models=["M1"])
    bench_env.respec("toy", k_max=9)

    assert bench_env.run("toy", "k-star", "--k-grid", "sparse", "--progress", "off") == 0

    k_sweep_df = pd.read_csv(bench_env.results_dir("toy") / "k_sweep_metrics.csv")
    assert sorted(k_sweep_df["k"].unique().tolist()) == [1, 3, 5, 7, 9]


def test_k_grid_defaults_to_dense(bench_env) -> None:
    _setup(bench_env, models=["M1"])
    bench_env.respec("toy", k_max=5)

    assert bench_env.run("toy", "k-star", "--progress", "off") == 0

    k_sweep_df = pd.read_csv(bench_env.results_dir("toy") / "k_sweep_metrics.csv")
    assert sorted(k_sweep_df["k"].unique().tolist()) == [1, 2, 3, 4, 5]


def test_benchmark_serializes_the_canonical_croma_f0(bench_env) -> None:
    """``croma_f0`` reaches every serialized layer, and matches the per-sample column.

    The published tables read F(0) off ``metrics.csv``; nothing downstream may have to
    rebuild it from the per-sample artifact, so the stored value has to be the same number
    the distribution gives -- ``<= 0`` over the defined occurrences.
    """
    models = ["M1", "M2"]
    _setup(bench_env, models=models)

    # This toy set is too small to fill the default headline radius; m<=2 keeps every
    # occurrence defined, so the assertions below compare real fractions.
    assert bench_env.run("toy", "k-star", "--croma-m-max", "2", "--progress", "off") == 0

    results_dir = bench_env.results_dir("toy")
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    m_sweep_df = pd.read_csv(results_dir / "croma_m_sweep_metrics.csv")
    per_sample_df = pd.read_csv(results_dir / "per_sample_metrics.csv")

    assert "croma_f0" in metrics_df.columns
    assert "croma_f0" in m_sweep_df.columns

    for _, row in metrics_df.iterrows():
        samples = per_sample_df.loc[
            per_sample_df["model"] == row["model"], f"croma_m{int(row['croma_m'])}"
        ]
        defined = samples[np.isfinite(samples)]
        assert row["croma_f0"] == pytest.approx(float((defined <= 0.0).mean()))

    for _, row in m_sweep_df.iterrows():
        samples = per_sample_df.loc[
            per_sample_df["model"] == row["model"], f"croma_m{int(row['m'])}"
        ]
        defined = samples[np.isfinite(samples)]
        assert row["croma_f0"] == pytest.approx(float((defined <= 0.0).mean()))


def test_cached_payload_keys_are_exactly_what_a_run_writes() -> None:
    """The cache's required-key set and the writer must not drift apart.

    They are two spellings of one payload: if the writer gains a statistic the required
    set does not know about, a stale entry is read back missing it and the run raises on
    the key. This is what keeps the duplication honest.
    """
    result = CRoMaResult(
        dataset="toy",
        m=5,
        value=0.1,
        std=0.0,
        n_pairs=1,
        pair_values=np.asarray([0.1]),
        sample_values=np.asarray([0.1]),
        sample_values_aligned=np.asarray([0.1]),
        occurrence_defined_mask=np.asarray([True]),
        undefined_frac=0.0,
    )

    assert set(bm._croma_result_to_payload(result, m=5)) == bm._CROMA_PAYLOAD_KEYS


def test_shared_support_rejects_ri_mari_mismatch() -> None:
    causes = {
        "ss_dominated_undefined_frac": 0.5,
        "oo_dominated_undefined_frac": 0.0,
        "mixed_undefined_frac": 0.0,
    }

    with pytest.raises(RuntimeError, match="RI and MaRI support must match"):
        bm._shared_support_fields(
            {"support": 0.5, **causes},
            {"support": 0.25, **causes},
        )


def _croma_result_for_serialization(
    *,
    sample_values_aligned: np.ndarray,
    occurrence_defined_mask: np.ndarray,
    undefined_frac: float,
) -> CRoMaResult:
    return CRoMaResult(
        dataset="toy",
        m=1,
        value=0.1,
        std=0.0,
        n_pairs=1,
        pair_values=np.asarray([0.1]),
        sample_values=np.asarray([0.1]),
        sample_values_aligned=sample_values_aligned,
        occurrence_defined_mask=occurrence_defined_mask,
        undefined_frac=undefined_frac,
    )


def test_croma_result_with_nonzero_undefined_fraction_fails_before_serialization() -> None:
    result = _croma_result_for_serialization(
        sample_values_aligned=np.asarray([0.1]),
        occurrence_defined_mask=np.asarray([True]),
        undefined_frac=0.5,
    )

    with pytest.raises(RuntimeError, match="incomplete support"):
        bm._croma_result_to_payload(result, m=1)


def test_croma_result_with_undefined_unit_fails_before_serialization() -> None:
    result = _croma_result_for_serialization(
        sample_values_aligned=np.asarray([0.1]),
        occurrence_defined_mask=np.asarray([False]),
        undefined_frac=0.0,
    )

    with pytest.raises(RuntimeError, match="undefined evaluation units"):
        bm._croma_result_to_payload(result, m=1)


def test_croma_result_with_nonfinite_sample_fails_before_serialization() -> None:
    result = _croma_result_for_serialization(
        sample_values_aligned=np.asarray([np.nan]),
        occurrence_defined_mask=np.asarray([True]),
        undefined_frac=0.0,
    )

    with pytest.raises(RuntimeError, match="non-finite samples"):
        bm._croma_result_to_payload(result, m=1)


def test_run_writes_a_replayable_run_config(bench_env) -> None:
    """A run records the grid it swept, and the legacy backfill recovers the same answer.

    Both halves matter. New runs are self-describing, and runs predating the sidecar are
    reconstructed from their own ``metrics.csv``; if the two ever disagreed, backfilling the
    committed runs would rewrite history rather than record it.
    """
    _setup(bench_env, models=["M1"])
    bench_env.respec("toy", k_max=9)

    assert bench_env.run("toy", "k-star", "--k-grid", "sparse", "--progress", "off") == 0

    results = bench_env.results_dir("toy")
    config = run_config.read_run_config(results)
    assert config is not None
    assert config["replay"]["k_grid"] == "sparse"
    assert config["resolved"]["protocol"] == "k-star"
    assert config["resolved"]["benchmark"] == "toy"

    replayed = run_config.replay_args(config)
    assert replayed[replayed.index("--k-grid") + 1] == "sparse"

    inferred = run_config.infer_replay_from_metrics(results / "metrics.csv")
    assert inferred["k_grid"] == config["replay"]["k_grid"]
    assert inferred["k_max"] == config["replay"]["k_max"]
    assert inferred["tau"] == config["replay"]["tau"]
