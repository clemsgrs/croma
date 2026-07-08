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


def _toy_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3", "s0", "s4", "s5", "s6"],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "scanner_vendor": ["V1", "V2", "V1", "V2", "V1", "V2", "V1", "V2"],
            "slide_id": ["sl0", "sl1", "sl2", "sl3", "sl0", "sl4", "sl5", "sl6"],
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


def _install_fake_registry_and_embed(monkeypatch, models: list[str]) -> None:
    def fake_registry() -> dict:
        return {name: object() for name in models}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        model_name = output_path.stem
        arr = _toy_features(model_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        output_path.with_suffix(".npy.json").write_text(
            '{"manifest":"%s","model_id":"fake","extract":"cls","mixed_precision":false}\n'
            % str(manifest_path),
            encoding="utf-8",
        )
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)


def _install_noop_plots(monkeypatch) -> None:
    def fake_plot(*, out_path: Path, **kwargs: object) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"plot")

    plot_names = [
        "plot_bio_vs_confounder_scatter",
        "plot_croma_ltm_bars",
        "plot_croma_ltm_scatter",
        "plot_croma_m_sweep",
        "plot_croma_sample_distributions",
        "plot_croma_vs_mari_scatter",
        "plot_q_alpha_vs_croma_scatter",
        "plot_knn_bio_k_sweep",
        "plot_knn_confounder_k_sweep",
        "plot_mari_k_sweep",
        "plot_mari_vs_ri_scatter",
        "plot_ri_mari_support",
        "plot_ri_k_sweep",
    ]
    for name in plot_names:
        monkeypatch.setattr(bm, name, fake_plot)


def _run_benchmark(
    monkeypatch,
    *,
    manifest_path: Path,
    output_dir: Path,
    models: list[str] | None,
    evaluation_design: str = "dataset_wide",
    k_max: int = 3,
    extra_args: list[str] | None = None,
) -> int:
    argv = [
        "benchmark.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--confounder-column",
        "scanner_vendor",
        "--evaluation-design",
        evaluation_design,
        "--k-max",
        str(int(k_max)),
        "--progress",
        "off",
    ]
    if models is not None:
        argv.extend(["--models", ",".join(models)])
    if extra_args:
        argv.extend(extra_args)
    monkeypatch.setattr(sys, "argv", argv)
    return bm.main()


@pytest.mark.parametrize(
    "argv_tail",
    [
        ["--dataset-name", "override"],
        ["--exclude-confounder", "C1"],
    ],
)
def test_benchmark_rejects_removed_flags(
    monkeypatch, tmp_path: Path, argv_tail: list[str]
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"

    argv = [
        "benchmark.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
    ]
    argv.extend(argv_tail)
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
def test_benchmark_rejects_removed_k_sweep_flags(
    monkeypatch, tmp_path: Path, argv_tail: list[str]
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"

    argv = [
        "benchmark.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--confounder-column",
        "scanner_vendor",
        "--k-max",
        "3",
    ]
    argv.extend(argv_tail)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        bm._parse_args()
    assert excinfo.value.code == 2


def test_benchmark_uses_manifest_stem_for_dataset_output(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "release-ready.csv"
    manifest = _toy_manifest()
    manifest["dataset"] = "wrong_name"
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, models=["M1"])
    _install_noop_plots(monkeypatch)

    assert (
        _run_benchmark(
            monkeypatch,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models=["M1"],
        )
        == 0
    )

    metrics_df = pd.read_csv(
        output_dir / manifest_path.stem / "results" / "metrics.csv"
    )
    assert set(metrics_df["dataset"]) == {"release-ready"}


def test_benchmark_dataset_wide_outputs_sample_level_rows(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    models = ["M1", "M2"]
    _install_fake_registry_and_embed(monkeypatch, models=models)
    _install_noop_plots(monkeypatch)

    assert (
        _run_benchmark(
            monkeypatch,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models=None,
        )
        == 0
    )

    dataset_dir = output_dir / manifest_path.stem
    results_dir = dataset_dir / "results"
    plots_dir = dataset_dir / "plots"
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    k_sweep_df = pd.read_csv(results_dir / "k_sweep_metrics.csv")
    per_sample_df = pd.read_csv(results_dir / "per_sample_metrics.csv")
    per_model_dir = results_dir / "per_sample_metrics_by_model"

    assert set(metrics_df["model"]) == set(models)
    assert set(metrics_df["evaluation_design"]) == {"dataset_wide"}
    assert set(metrics_df["evaluation_unit"]) == {"sample"}
    assert set(metrics_df["k_max"]) == {3}
    assert set(k_sweep_df["evaluation_design"]) == {"dataset_wide"}
    assert set(k_sweep_df["evaluation_unit"]) == {"sample"}
    assert set(k_sweep_df["k_max"]) == {3}
    assert set(per_sample_df["evaluation_design"]) == {"dataset_wide"}
    assert set(per_sample_df["evaluation_unit"]) == {"sample"}
    assert set(per_sample_df["subset"]) == {"dataset"}

    expected_rows = len(_toy_manifest()) * len(models)
    assert len(per_sample_df) == expected_rows
    for model in models:
        model_rows = per_sample_df[per_sample_df["model"] == model].sort_values(
            "occurrence_index"
        )
        assert model_rows["occurrence_index"].tolist() == list(
            range(len(_toy_manifest()))
        )
        assert model_rows["sample_index"].tolist() == list(range(len(_toy_manifest())))
        model_df = pd.read_csv(per_model_dir / f"{model}.csv")
        assert model_df["occurrence_index"].tolist() == list(
            range(len(_toy_manifest()))
        )
        assert set(model_df["model"]) == {model}

    for path in (
        results_dir / "metrics.csv",
        results_dir / "k_sweep_metrics.csv",
        results_dir / "croma_m_sweep_metrics.csv",
        results_dir / "per_sample_metrics.csv",
        per_model_dir / "M1.csv",
        per_model_dir / "M2.csv",
        plots_dir / "croma_ltm_bars.png",
        plots_dir / "croma_ltm_scatter.png",
        plots_dir / "ri_mari_support.png",
    ):
        assert path.exists(), f"Missing output: {path}"

    for path in (
        plots_dir / "benchmark_6panel_summary.png",
        plots_dir / "croma_trend_quadrants.png",
    ):
        assert not path.exists(), f"Deprecated plot should not be written: {path}"


def test_benchmark_paired_outputs_occurrence_level_rows(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    manifest = _toy_manifest()
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, models=["M1"])
    _install_noop_plots(monkeypatch)

    assert (
        _run_benchmark(
            monkeypatch,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models=["M1"],
            evaluation_design="paired_2x2",
        )
        == 0
    )

    results_dir = output_dir / manifest_path.stem / "results"
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    per_sample_df = pd.read_csv(results_dir / "per_sample_metrics.csv")

    assert set(metrics_df["evaluation_design"]) == {"paired_2x2"}
    assert set(metrics_df["evaluation_unit"]) == {"occurrence"}
    assert set(per_sample_df["evaluation_design"]) == {"paired_2x2"}
    assert set(per_sample_df["evaluation_unit"]) == {"occurrence"}
    assert set(per_sample_df["subset"]) == {"pair1", "pair2"}
    assert per_sample_df["occurrence_index"].tolist() == list(range(len(manifest)))
    assert int((per_sample_df["sample_id"] == "s0").sum()) == 2


def test_benchmark_writes_per_sample_artifact_with_undefined_rows(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    manifest = _toy_manifest()
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
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
            self.ss_dominated_undefined_frac = float(
                np.mean(self.sample_undefined_types == 1)
            )
            self.oo_dominated_undefined_frac = float(
                np.mean(self.sample_undefined_types == 2)
            )
            self.mixed_undefined_frac = float(np.mean(self.sample_undefined_types == 3))
            self.evaluation_design = "dataset_wide"
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
            self.undefined_frac = float((~informative).mean())
            self.evaluation_design = "dataset_wide"
            self.evaluation_unit = "sample"
            self.k_start = 200
            self.k_final = 200
            self.retries = 0
            self.alpha = 0.10
            self.q_alpha = 0.5
            self.ltm_alpha = 0.4

    def fake_ri_compute_artifacts(
        *,
        prepared_neighbors: object,
        dataset_name: str,
        k_values: list[int],
        selected_k: int,
        include_selected_result: bool,
        warn_selected_result: bool,
        summarize_by_mean: bool = False,
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
        summarize_by_mean: bool = False,
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
        assert evaluation_design == "dataset_wide"
        return {
            1: _FakeCRoMaResult(1, [1.1, 0.9, np.nan, 1.3, 1.4, np.nan, 0.8, 1.0]),
            2: _FakeCRoMaResult(2, [1.2, 1.0, np.nan, 1.35, 1.45, np.nan, 0.85, 1.05]),
        }

    _install_fake_registry_and_embed(monkeypatch, models=[model])
    _install_noop_plots(monkeypatch)
    monkeypatch.setattr(
        bm.RI,
        "_compute_artifacts_from_prepared_dataset_wide",
        fake_ri_compute_artifacts,
    )
    monkeypatch.setattr(
        bm.MaRI,
        "_compute_artifacts_from_prepared_dataset_wide",
        fake_mari_compute_artifacts,
    )
    monkeypatch.setattr(bm.CRoMa, "compute", fake_croma_compute)

    assert (
        _run_benchmark(
            monkeypatch,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models=[model],
            extra_args=["--croma-m-max", "2"],
        )
        == 0
    )

    per_sample_df = pd.read_csv(
        output_dir / manifest_path.stem / "results" / "per_sample_metrics.csv"
    )
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
    assert np.isnan(per_sample_df.loc[2, "croma_m1"])
    assert np.isnan(per_sample_df.loc[5, "croma_m2"])


def test_benchmark_k_max_uses_full_range(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    models = ["M1", "M2"]
    _install_fake_registry_and_embed(monkeypatch, models=models)
    _install_noop_plots(monkeypatch)

    assert (
        _run_benchmark(
            monkeypatch,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models=models,
            k_max=4,
        )
        == 0
    )

    df = pd.read_csv(
        output_dir / manifest_path.stem / "results" / "k_sweep_metrics.csv"
    )
    assert set(df["model"]) == set(models)
    assert set(df["k"]) == {1, 2, 3, 4}
    assert set(df["k_max"]) == {4}


def test_benchmark_can_select_different_confounder_k(
    monkeypatch, tmp_path: Path
) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"

    class _CurveResult:
        def __init__(self, k: int) -> None:
            self.k = int(k)
            self.value = 0.5
            self.std = 0.0
            self.n_pairs = 1
            self.sample_values = np.asarray([0.2, 0.6, 0.8], dtype=float)
            self.sample_values_aligned = np.asarray(
                [0.2, np.nan, 0.6, 0.8, np.nan, np.nan, np.nan, np.nan], dtype=float
            )
            self.sample_undefined_types = np.asarray(
                [0, 3, 0, 0, 3, 3, 3, 3], dtype=int
            )
            self.undefined_frac = 0.0
            self.ss_dominated_undefined_frac = 0.0
            self.oo_dominated_undefined_frac = 0.0
            self.mixed_undefined_frac = 0.0
            self.evaluation_design = "dataset_wide"
            self.evaluation_unit = "sample"

    def fake_knn_balanced_accuracy_by_k(
        *,
        features: np.ndarray,
        labels: np.ndarray,
        slide_ids: np.ndarray,
        k_values: list[int],
        warn_context: str,
    ) -> dict[int, float]:
        if np.array_equal(labels, np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)):
            return {
                int(k): v for k, v in zip(k_values, [0.60, 0.75, 0.92], strict=False)
            }
        return {int(k): v for k, v in zip(k_values, [0.90, 0.70, 0.68], strict=False)}

    def fake_compute_artifacts(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        confounder_column: str,
        k_values: list[int],
        evaluation_design: str,
        selected_k: int,
        include_selected_result: bool,
        warn_selected_result: bool,
        tau: float | None = None,
        prune_ss_oo: bool = False,
        summarize_by_mean: bool = False,
    ) -> SimpleNamespace:
        del include_selected_result, warn_selected_result
        return SimpleNamespace(
            curve={int(k): 0.5 for k in k_values},
            result=_CurveResult(k=int(selected_k)),
        )

    _install_fake_registry_and_embed(monkeypatch, models=["M1"])
    _install_noop_plots(monkeypatch)
    monkeypatch.setattr(
        bm, "_knn_balanced_accuracy_by_k", fake_knn_balanced_accuracy_by_k
    )
    monkeypatch.setattr(bm.RI, "_compute_artifacts", fake_compute_artifacts)
    monkeypatch.setattr(bm.MaRI, "_compute_artifacts", fake_compute_artifacts)

    assert (
        _run_benchmark(
            monkeypatch,
            manifest_path=manifest_path,
            output_dir=output_dir,
            models=["M1"],
        )
        == 0
    )

    dataset_dir = output_dir / manifest_path.stem
    metrics_df = pd.read_csv(dataset_dir / "results" / "metrics.csv")
    k_sweep_df = pd.read_csv(dataset_dir / "results" / "k_sweep_metrics.csv")

    assert int(metrics_df.loc[0, "k"]) == 1
    assert int(metrics_df.loc[0, "selected_k_confounder"]) == 3
    assert int(k_sweep_df.loc[0, "selected_k"]) == 1
    assert int(k_sweep_df.loc[0, "selected_k_confounder"]) == 3
