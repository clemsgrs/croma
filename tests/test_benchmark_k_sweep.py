import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm


def _toy_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "medical_center": ["C1", "C1", "C2", "C2", "C1", "C1", "C2", "C2"],
            "slide_id": [f"slide-{i}" for i in range(8)],
            "dataset": ["toy"] * 8,
        }
    )


def _toy_features(model_name: str = "M1") -> np.ndarray:
    base = np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.92, 0.08, 0.00, 0.00],
            [0.90, 0.10, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.05, 0.95, 0.00, 0.00],
            [0.08, 0.92, 0.00, 0.00],
            [0.10, 0.90, 0.00, 0.00],
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
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)


def _run_benchmark(
    monkeypatch,
    *,
    manifest_path: Path,
    output_dir: Path,
    models: list[str] | None,
    extra_args: list[str] | None = None,
) -> int:
    argv = [
        "benchmark.py",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--mode",
        "global",
        "--k-candidates",
        "1,3",
        "--progress",
        "off",
    ]
    if models is not None:
        argv.extend(["--models", ",".join(models)])
    if extra_args:
        argv.extend(extra_args)
    monkeypatch.setattr(sys, "argv", argv)
    return bm.main()


def test_benchmark_smoke_writes_current_outputs_for_all_registry_models(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    models = ["M1", "M2"]
    _install_fake_registry_and_embed(monkeypatch, models=models)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        models=None,
        extra_args=["--exclude-center", "C999"],
    ) == 0

    dataset_dir = output_dir / manifest_path.stem
    results_dir = dataset_dir / "results"
    plots_dir = dataset_dir / "plots"
    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    k_sweep_df = pd.read_csv(results_dir / "k_sweep_metrics.csv")
    per_sample_df = pd.read_csv(results_dir / "per_sample_metrics.csv")

    for path in (
        results_dir / "metrics.csv",
        results_dir / "metrics.json",
        results_dir / "k_sweep_metrics.csv",
        results_dir / "k_sweep_metrics.json",
        results_dir / "ccrr_m_sweep_metrics.csv",
        results_dir / "ccrr_m_sweep_metrics.json",
        results_dir / "per_sample_metrics.csv",
        results_dir / "per_sample_metrics.json",
        results_dir / "per_sample_metrics_by_model",
        results_dir / "sample_distributions",
        plots_dir / "benchmark_6panel_summary.png",
        plots_dir / "ccrr_ltm_comparison.png",
    ):
        assert path.exists(), f"Missing output: {path}"

    assert set(metrics_df["model"]) == set(models)
    assert set(k_sweep_df["model"]) == set(models)
    assert set(k_sweep_df["k"]) == {1, 3}
    assert "selected_k_center" in metrics_df.columns
    assert "selected_k_center" in k_sweep_df.columns
    assert set(metrics_df["excluded_centers"]) == {"C999"}
    assert set(k_sweep_df["excluded_centers"]) == {"C999"}
    assert set(per_sample_df["model"]) == set(models)


def test_benchmark_writes_per_sample_artifact_with_undefined_rows(monkeypatch, tmp_path: Path) -> None:
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
            self.ss_dominated_undefined_frac = float(np.mean(self.sample_undefined_types == 1))
            self.oo_dominated_undefined_frac = float(np.mean(self.sample_undefined_types == 2))
            self.mixed_undefined_frac = float(np.mean(self.sample_undefined_types == 3))

    class _FakeCCRRResult:
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
            self.acceptance_threshold = 0.0
            self.acceptance_met = True
            self.k_start = 200
            self.k_final = 200
            self.retries = 0
            self.alpha = 0.10
            self.q_alpha = 0.5
            self.ltm_alpha = 0.4

    def fake_compute_curve(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        mode: str,
        k_values: list[int],
        tau: float | None = None,
    ) -> dict[int, float]:
        return {int(k): 0.5 for k in k_values}

    def fake_ri_compute(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        mode: str,
        k_candidates: list[int],
    ) -> _FakeRobustnessResult:
        return _FakeRobustnessResult(
            k=int(k_candidates[0]),
            values=[0.10, np.nan, 0.30, np.nan, 0.50, 0.60, np.nan, 0.80],
            undef_types=[0, 1, 0, 2, 0, 0, 3, 0],
        )

    def fake_mari_compute(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        mode: str,
        k_candidates: list[int],
        tau: float,
    ) -> _FakeRobustnessResult:
        return _FakeRobustnessResult(
            k=int(k_candidates[0]),
            values=[0.20, np.nan, 0.35, np.nan, 0.55, 0.65, np.nan, 0.85],
            undef_types=[0, 3, 0, 1, 0, 0, 2, 0],
        )

    def fake_ccrr_compute(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        mode: str,
        m: list[int],
        alpha: float,
        start_k: int,
        k_growth_factor: float,
    ) -> dict[int, _FakeCCRRResult]:
        return {
            1: _FakeCCRRResult(1, [1.1, 0.9, np.nan, 1.3, 1.4, np.nan, 0.8, 1.0]),
            2: _FakeCCRRResult(2, [1.2, 1.0, np.nan, 1.35, 1.45, np.nan, 0.85, 1.05]),
        }

    _install_fake_registry_and_embed(monkeypatch, models=[model])
    monkeypatch.setattr(bm.RI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.MaRI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.RI, "compute", fake_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", fake_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", fake_ccrr_compute)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        models=[model],
        extra_args=["--ccrr-m-max", "2"],
    ) == 0

    per_sample_df = pd.read_csv(output_dir / manifest_path.stem / "results" / "per_sample_metrics.csv")
    assert len(per_sample_df) == len(manifest)
    assert per_sample_df["sample_index"].tolist() == list(range(len(manifest)))
    assert per_sample_df["ri_defined"].tolist() == [True, False, True, False, True, True, False, True]
    assert per_sample_df["mari_defined"].tolist() == [True, False, True, False, True, True, False, True]
    assert per_sample_df["ri_undefined_type"].tolist() == [0, 1, 0, 2, 0, 0, 3, 0]
    assert per_sample_df["mari_undefined_type"].tolist() == [0, 3, 0, 1, 0, 0, 2, 0]
    assert np.isnan(per_sample_df.loc[1, "ri"])
    assert np.isnan(per_sample_df.loc[3, "mari"])
    assert np.isnan(per_sample_df.loc[2, "ccrr_m1"])
    assert np.isnan(per_sample_df.loc[5, "ccrr_m2"])


def test_benchmark_skips_per_sample_artifact_in_paired_mode(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, models=["M1"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            "M1",
            "--output-dir",
            str(output_dir),
            "--mode",
            "paired",
            "--k-candidates",
            "1,3",
            "--progress",
            "off",
        ],
    )

    assert bm.main() == 0
    results_dir = output_dir / manifest_path.stem / "results"
    assert (results_dir / "metrics.csv").exists()
    assert not (results_dir / "per_sample_metrics.csv").exists()
    assert not (results_dir / "per_sample_metrics.json").exists()
    assert not (results_dir / "per_sample_metrics_by_model").exists()


def test_benchmark_continuous_k_sweep_uses_full_range(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    models = ["M1", "M2"]
    _install_fake_registry_and_embed(monkeypatch, models=models)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        models=models,
        extra_args=["--k-candidates", "3,5", "--continuous-k-sweep-max", "4"],
    ) == 0

    df = pd.read_csv(output_dir / manifest_path.stem / "results" / "k_sweep_metrics.csv")
    assert set(df["model"]) == set(models)
    assert set(df["k"]) == {1, 2, 3, 4}


def test_benchmark_can_select_different_center_k(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"

    class _CurveResult:
        def __init__(self, k: int) -> None:
            self.k = int(k)
            self.value = 0.5
            self.std = 0.0
            self.sample_values = np.asarray([0.2, 0.6, 0.8], dtype=float)
            self.sample_values_aligned = np.asarray([0.2, np.nan, 0.6, 0.8, np.nan, np.nan, np.nan, np.nan], dtype=float)
            self.sample_undefined_types = np.asarray([0, 3, 0, 0, 3, 3, 3, 3], dtype=int)
            self.undefined_frac = 0.0
            self.ss_dominated_undefined_frac = 0.0
            self.oo_dominated_undefined_frac = 0.0
            self.mixed_undefined_frac = 0.0

    def fake_knn_balanced_accuracy_by_k(
        *,
        features: np.ndarray,
        labels: np.ndarray,
        slide_ids: np.ndarray,
        k_values: list[int],
        warn_context: str,
    ) -> dict[int, float]:
        if np.array_equal(labels, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)):
            return {int(k): v for k, v in zip(k_values, [0.90, 0.70], strict=False)}
        return {int(k): v for k, v in zip(k_values, [0.60, 0.92], strict=False)}

    def fake_compute_curve(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        mode: str,
        k_values: list[int],
        tau: float | None = None,
    ) -> dict[int, float]:
        return {int(k): 0.5 for k in k_values}

    def fake_compute(
        *,
        features: np.ndarray,
        manifest: pd.DataFrame,
        mode: str,
        k_candidates: list[int],
        tau: float | None = None,
    ) -> _CurveResult:
        return _CurveResult(k=int(k_candidates[0]))

    _install_fake_registry_and_embed(monkeypatch, models=["M1"])
    monkeypatch.setattr(bm, "_knn_balanced_accuracy_by_k", fake_knn_balanced_accuracy_by_k)
    monkeypatch.setattr(bm.RI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.MaRI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.RI, "compute", fake_compute)
    monkeypatch.setattr(bm.MaRI, "compute", fake_compute)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        models=["M1"],
        extra_args=None,
    ) == 0

    dataset_dir = output_dir / manifest_path.stem
    metrics_df = pd.read_csv(dataset_dir / "results" / "metrics.csv")
    k_sweep_df = pd.read_csv(dataset_dir / "results" / "k_sweep_metrics.csv")

    assert int(metrics_df.loc[0, "k"]) == 1
    assert int(metrics_df.loc[0, "selected_k_center"]) == 3
    assert int(k_sweep_df.loc[0, "selected_k"]) == 1
    assert int(k_sweep_df.loc[0, "selected_k_center"]) == 3


def test_benchmark_recomputes_when_cached_schema_is_stale(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "toy.csv"
    _toy_manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, models=["M1"])

    results_dir = output_dir / manifest_path.stem / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset": "dataset",
                "model": "M1",
                "k": 1,
                "mode": "global",
                "tau": 0.2,
                "alpha": 10.0,
                "k_candidates": "1,3",
                "ri": 0.4,
                "ri_std": 0.0,
                "mari": 0.5,
                "mari_std": 0.0,
                "ri_undefined_frac": 0.0,
                "mari_undefined_frac": 0.0,
                "ri_samples_path": str(results_dir / "sample_distributions" / "ri.M1.npy"),
                "mari_samples_path": str(results_dir / "sample_distributions" / "mari.M1.npy"),
                "embedding_path": "placeholder.npy",
            }
        ]
    ).to_csv(results_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "dataset",
                "model": "M1",
                "mode": "global",
                "tau": 0.2,
                "alpha": 10.0,
                "k_candidates": "1,3",
                "k": 1,
                "knn_bacc": 0.70,
                "ri": 0.40,
                "mari": 0.45,
                "selected_k": 1,
                "continuous_k_sweep": 0,
                "embedding_path": "placeholder.npy",
            }
        ]
    ).to_csv(results_dir / "k_sweep_metrics.csv", index=False)

    dist_dir = results_dir / "sample_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    np.save(dist_dir / "ri.M1.npy", np.asarray([0.3, 0.4], dtype=float))
    np.save(dist_dir / "mari.M1.npy", np.asarray([0.3, 0.4], dtype=float))

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        models=["M1"],
        extra_args=None,
    ) == 0

    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    k_sweep_df = pd.read_csv(results_dir / "k_sweep_metrics.csv")
    assert "bio_knn_bacc" in metrics_df.columns
    assert "center_knn_bacc" in metrics_df.columns
    assert "selected_k_center" in metrics_df.columns
    assert "knn_center_bacc" in k_sweep_df.columns
    assert "selected_k_center" in k_sweep_df.columns
