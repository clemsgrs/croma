
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def _toy_features(model_name: str) -> np.ndarray:
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


def test_benchmark_writes_k_sweep_rows_and_plot(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    models = ["M1", "M2"]

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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            ",".join(models),
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
        ],
    )

    code = bm.main()
    assert code == 0

    dataset_dir = output_dir / manifest_path.stem
    k_sweep_csv = dataset_dir / "results" / "k_sweep_metrics.csv"
    k_sweep_json = dataset_dir / "results" / "k_sweep_metrics.json"
    bio_k_plot_path = dataset_dir / "plots" / "knn_bio_k_sweep.png"
    center_k_plot_path = dataset_dir / "plots" / "knn_center_k_sweep.png"
    ri_k_plot_path = dataset_dir / "plots" / "ri_k_sweep.png"
    mari_plot_path = dataset_dir / "plots" / "mari_k_sweep.png"
    ccrr_m_plot_path = dataset_dir / "plots" / "ccrr_m_sweep.png"
    ccrr_ltm_plot_path = dataset_dir / "plots" / "ccrr_ltm_comparison.png"
    bio_center_plot_path = dataset_dir / "plots" / "bio_vs_center_scatter.png"
    mari_ri_plot_path = dataset_dir / "plots" / "mari_vs_ri_scatter.png"
    summary_plot_path = dataset_dir / "plots" / "benchmark_6panel_summary.png"
    ccrr_m_sweep_csv = dataset_dir / "results" / "ccrr_m_sweep_metrics.csv"
    ccrr_m_sweep_json = dataset_dir / "results" / "ccrr_m_sweep_metrics.json"
    per_sample_csv = dataset_dir / "results" / "per_sample_metrics.csv"
    per_sample_json = dataset_dir / "results" / "per_sample_metrics.json"
    per_sample_by_model_dir = dataset_dir / "results" / "per_sample_metrics_by_model"
    legacy_rank_plot = dataset_dir / "plots" / "ri_mari_rank.png"
    legacy_three_panel = dataset_dir / "plots" / "knn_bacc_ri_k_sweep.png"
    legacy_mari_dist_plot = dataset_dir / "plots" / "mari_sample_distributions.png"
    legacy_old_scatter = dataset_dir / "plots" / "biological_vs_center_knn_bacc.png"
    legacy_old_mari_ri = dataset_dir / "plots" / "ri_vs_mari_scatter.png"
    sample_dist_dir = dataset_dir / "results" / "sample_distributions"
    legacy_mari_dist_dir = dataset_dir / "results" / "mari_sample_distributions"
    legacy_ri_dist_dir = dataset_dir / "results" / "ri_sample_distributions"

    assert k_sweep_csv.exists()
    assert k_sweep_json.exists()
    assert bio_k_plot_path.exists()
    assert center_k_plot_path.exists()
    assert ri_k_plot_path.exists()
    assert mari_plot_path.exists()
    assert ccrr_m_plot_path.exists()
    assert ccrr_ltm_plot_path.exists()
    assert bio_center_plot_path.exists()
    assert mari_ri_plot_path.exists()
    assert summary_plot_path.exists()
    assert ccrr_m_sweep_csv.exists()
    assert ccrr_m_sweep_json.exists()
    assert per_sample_csv.exists()
    assert per_sample_json.exists()
    assert per_sample_by_model_dir.exists()
    assert sample_dist_dir.exists()
    assert not legacy_mari_dist_dir.exists()
    assert not legacy_ri_dist_dir.exists()
    assert not legacy_rank_plot.exists()
    assert not legacy_three_panel.exists()
    assert not legacy_mari_dist_plot.exists()
    assert not legacy_old_scatter.exists()
    assert not legacy_old_mari_ri.exists()

    df = pd.read_csv(k_sweep_csv)
    assert len(df) == len(models) * 2
    assert set(df["model"]) == set(models)
    assert set(df["k"]) == {1, 3}
    assert "mari" in df.columns
    assert "knn_center_bacc" in df.columns
    assert "selected_k_center" in df.columns

    ccrr_m_df = pd.read_csv(ccrr_m_sweep_csv)
    assert len(ccrr_m_df) >= len(models)
    assert set(ccrr_m_df["model"]) == set(models)
    assert "m" in ccrr_m_df.columns
    assert "ccrr" in ccrr_m_df.columns
    assert "ccrr_search" in ccrr_m_df.columns
    assert "ccrr_k_start" in ccrr_m_df.columns
    assert "ccrr_k_final" in ccrr_m_df.columns
    assert "ccrr_retries" in ccrr_m_df.columns
    assert set(ccrr_m_df["m"]) >= {1}

    metrics_df = pd.read_csv(dataset_dir / "results" / "metrics.csv")
    assert "ri_undefined_frac" in metrics_df.columns
    assert "mari_undefined_frac" in metrics_df.columns
    assert "ri_samples_path" in metrics_df.columns
    assert "mari_samples_path" in metrics_df.columns
    assert "bio_knn_bacc" in metrics_df.columns
    assert "center_knn_bacc" in metrics_df.columns
    assert "selected_k_center" in metrics_df.columns
    assert "ccrr_search" in metrics_df.columns
    assert "ccrr_k_start" in metrics_df.columns
    assert "ccrr_k_final" in metrics_df.columns
    assert "ccrr_retries" in metrics_df.columns
    assert ((metrics_df["ri_undefined_frac"] >= 0.0) & (metrics_df["ri_undefined_frac"] <= 1.0)).all()
    assert ((metrics_df["mari_undefined_frac"] >= 0.0) & (metrics_df["mari_undefined_frac"] <= 1.0)).all()
    assert ((metrics_df["ccrr_undefined_frac"] >= 0.0) & (metrics_df["ccrr_undefined_frac"] <= 1.0)).all()

    per_sample_df = pd.read_csv(per_sample_csv)
    assert len(per_sample_df) == len(models) * len(manifest)
    assert set(per_sample_df["model"]) == set(models)
    assert {
        "dataset",
        "model",
        "mode",
        "sample_index",
        "sample_id",
        "slide_id",
        "label",
        "medical_center",
        "k",
        "tau",
        "ccrr_alpha",
        "ccrr_search",
        "excluded_centers",
        "ri",
        "mari",
        "ri_defined",
        "mari_defined",
        "ri_undefined_type",
        "mari_undefined_type",
        "ccrr_m1",
    }.issubset(per_sample_df.columns)
    assert set(per_sample_df["sample_index"]) == set(range(len(manifest)))
    for model in models:
        model_rows = per_sample_df[per_sample_df["model"] == model].sort_values("sample_index").reset_index(drop=True)
        assert model_rows["sample_index"].tolist() == list(range(len(manifest)))
        assert model_rows["sample_id"].tolist() == manifest["sample_id"].tolist()
        assert model_rows["slide_id"].tolist() == manifest["slide_id"].tolist()
        assert model_rows["label"].tolist() == manifest["label"].tolist()
        assert model_rows["medical_center"].tolist() == manifest["medical_center"].tolist()

        model_csv = per_sample_by_model_dir / f"{model}.csv"
        model_json = per_sample_by_model_dir / f"{model}.json"
        assert model_csv.exists()
        assert model_json.exists()
        model_df = pd.read_csv(model_csv)
        assert len(model_df) == len(manifest)
        assert model_df.sort_values("sample_index")["sample_id"].tolist() == manifest["sample_id"].tolist()

    for model in models:
        mari_dist_path = sample_dist_dir / f"mari.{model}.npy"
        ri_dist_path = sample_dist_dir / f"ri.{model}.npy"
        embedding_path = dataset_dir / "embeddings" / f"{model}.npy"
        assert embedding_path.exists()
        assert mari_dist_path.exists()
        assert ri_dist_path.exists()
        arr = np.load(mari_dist_path)
        ri_arr = np.load(ri_dist_path)
        assert 0 < arr.shape[0] <= 8
        assert 0 < ri_arr.shape[0] <= 8


def test_benchmark_writes_per_sample_artifact_with_undefined_rows(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    model = "M1"

    def fake_registry() -> dict:
        return {model: object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

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
        tau: float | None = None,
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
        tau: float | None = None,
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

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)
    monkeypatch.setattr(bm.RI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.MaRI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.RI, "compute", fake_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", fake_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", fake_ccrr_compute)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            model,
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
            "--ccrr-m-max",
            "2",
        ],
    )

    code = bm.main()
    assert code == 0

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


def test_benchmark_reuses_cached_per_sample_artifacts(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    model = "M1"

    def fake_registry() -> dict:
        return {model: object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)

    argv = [
        "benchmark.py",
        "--manifest",
        str(manifest_path),
        "--models",
        model,
        "--output-dir",
        str(output_dir),
        "--mode",
        "global",
        "--k-candidates",
        "1,3",
        "--ccrr-m-max",
        "2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert bm.main() == 0

    results_dir = output_dir / manifest_path.stem / "results"
    before_csv = (results_dir / "per_sample_metrics.csv").read_text(encoding="utf-8")
    before_json = (results_dir / "per_sample_metrics.json").read_text(encoding="utf-8")

    ri_calls = {"n": 0}
    mari_calls = {"n": 0}
    ccrr_calls = {"n": 0}
    original_ri_compute = bm.RI.compute
    original_mari_compute = bm.MaRI.compute
    original_ccrr_compute = bm.CCRR.compute

    def wrapped_ri_compute(*args, **kwargs):
        ri_calls["n"] += 1
        return original_ri_compute(*args, **kwargs)

    def wrapped_mari_compute(*args, **kwargs):
        mari_calls["n"] += 1
        return original_mari_compute(*args, **kwargs)

    def wrapped_ccrr_compute(*args, **kwargs):
        ccrr_calls["n"] += 1
        return original_ccrr_compute(*args, **kwargs)

    monkeypatch.setattr(bm.RI, "compute", wrapped_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", wrapped_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", wrapped_ccrr_compute)
    monkeypatch.setattr(sys, "argv", argv)
    assert bm.main() == 0

    after_csv = (results_dir / "per_sample_metrics.csv").read_text(encoding="utf-8")
    after_json = (results_dir / "per_sample_metrics.json").read_text(encoding="utf-8")
    assert before_csv == after_csv
    assert before_json == after_json
    assert ri_calls["n"] == 0
    assert mari_calls["n"] == 0
    assert ccrr_calls["n"] == 0


def test_benchmark_skips_per_sample_artifact_in_paired_mode(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    model = "M1"

    def fake_registry() -> dict:
        return {model: object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            model,
            "--output-dir",
            str(output_dir),
            "--mode",
            "paired",
            "--k-candidates",
            "1,3",
        ],
    )

    assert bm.main() == 0
    results_dir = output_dir / manifest_path.stem / "results"
    assert (results_dir / "metrics.csv").exists()
    assert not (results_dir / "per_sample_metrics.csv").exists()
    assert not (results_dir / "per_sample_metrics.json").exists()
    assert not (results_dir / "per_sample_metrics_by_model").exists()


def test_benchmark_uses_all_registry_models_when_models_arg_missing(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    models = ["M1", "M2", "M3"]

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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
        ],
    )

    code = bm.main()
    assert code == 0

    dataset_dir = output_dir / manifest_path.stem
    df = pd.read_csv(dataset_dir / "results" / "k_sweep_metrics.csv")
    assert set(df["model"]) == set(models)
    assert len(df) == len(models) * 2


def test_benchmark_continuous_k_sweep_uses_full_range(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    models = ["M1", "M2"]

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

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            ",".join(models),
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "3,5",
            "--continuous-k-sweep-max",
            "4",
        ],
    )

    code = bm.main()
    assert code == 0

    dataset_dir = output_dir / manifest_path.stem
    df = pd.read_csv(dataset_dir / "results" / "k_sweep_metrics.csv")
    assert set(df["model"]) == set(models)
    assert set(df["k"]) == {1, 2, 3, 4}
    assert len(df) == len(models) * 4


def test_benchmark_can_select_different_center_k(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    models = ["M1"]

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
        arr = _toy_features("M1")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    def fake_knn_balanced_accuracy_by_k(
        *,
        features: np.ndarray,
        labels: np.ndarray,
        slide_ids: np.ndarray,
        k_values: list[int],
        warn_context: str,
    ) -> dict[int, float]:
        # Biological labels are [0,0,0,0,1,1,1,1]; center labels are interleaved.
        if np.array_equal(labels, np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)):
            return {int(k): v for k, v in zip(k_values, [0.90, 0.70], strict=False)}
        return {int(k): v for k, v in zip(k_values, [0.60, 0.92], strict=False)}

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

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)
    monkeypatch.setattr(bm, "_knn_balanced_accuracy_by_k", fake_knn_balanced_accuracy_by_k)
    monkeypatch.setattr(bm.RI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.MaRI, "compute_curve", fake_compute_curve)
    monkeypatch.setattr(bm.RI, "compute", fake_compute)
    monkeypatch.setattr(bm.MaRI, "compute", fake_compute)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            ",".join(models),
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
        ],
    )

    code = bm.main()
    assert code == 0

    dataset_dir = output_dir / manifest_path.stem
    metrics_df = pd.read_csv(dataset_dir / "results" / "metrics.csv")
    k_sweep_df = pd.read_csv(dataset_dir / "results" / "k_sweep_metrics.csv")

    assert int(metrics_df.loc[0, "k"]) == 1
    assert int(metrics_df.loc[0, "selected_k_center"]) == 3
    assert int(k_sweep_df.loc[0, "selected_k"]) == 1
    assert int(k_sweep_df.loc[0, "selected_k_center"]) == 3


def test_benchmark_recomputes_when_cached_schema_is_stale(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    model = "M1"

    def fake_registry() -> dict:
        return {model: object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)

    dataset_dir = output_dir / manifest_path.stem
    results_dir = dataset_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Write stale cache rows (missing center-related columns).
    pd.DataFrame(
        [
            {
                "dataset": "dataset",
                "model": model,
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
                "ri_samples_path": str(results_dir / "sample_distributions" / f"ri.{model}.npy"),
                "mari_samples_path": str(results_dir / "sample_distributions" / f"mari.{model}.npy"),
                "embedding_path": "placeholder.npy",
            }
        ]
    ).to_csv(results_dir / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "dataset",
                "model": model,
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
            },
            {
                "dataset": "dataset",
                "model": model,
                "mode": "global",
                "tau": 0.2,
                "alpha": 10.0,
                "k_candidates": "1,3",
                "k": 3,
                "knn_bacc": 0.60,
                "ri": 0.50,
                "mari": 0.55,
                "selected_k": 1,
                "continuous_k_sweep": 0,
                "embedding_path": "placeholder.npy",
            },
        ]
    ).to_csv(results_dir / "k_sweep_metrics.csv", index=False)

    # Presence of distributions should not force cache reuse if schema is stale.
    dist_dir = results_dir / "sample_distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    for prefix in ("ri", "mari"):
        np.save(dist_dir / f"{prefix}.{model}.npy", np.asarray([0.3, 0.4], dtype=float))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            model,
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
        ],
    )

    code = bm.main()
    assert code == 0

    metrics_df = pd.read_csv(results_dir / "metrics.csv")
    k_sweep_df = pd.read_csv(results_dir / "k_sweep_metrics.csv")
    assert "bio_knn_bacc" in metrics_df.columns
    assert "center_knn_bacc" in metrics_df.columns
    assert "selected_k_center" in metrics_df.columns
    assert "knn_center_bacc" in k_sweep_df.columns
    assert "selected_k_center" in k_sweep_df.columns


def test_benchmark_records_excluded_centers_signature(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    model = "M1"

    def fake_registry() -> dict:
        return {model: object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            model,
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
            "--exclude-center",
            "C999",
        ],
    )

    code = bm.main()
    assert code == 0

    dataset_dir = output_dir / manifest_path.stem
    metrics_df = pd.read_csv(dataset_dir / "results" / "metrics.csv")
    k_sweep_df = pd.read_csv(dataset_dir / "results" / "k_sweep_metrics.csv")
    assert "excluded_centers" in metrics_df.columns
    assert "excluded_centers" in k_sweep_df.columns
    assert set(metrics_df["excluded_centers"]) == {"C999"}
    assert set(k_sweep_df["excluded_centers"]) == {"C999"}


def test_benchmark_recomputes_when_ccrr_search_settings_change(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    model = "M1"

    def fake_registry() -> dict:
        return {model: object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            model,
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
        ],
    )
    code = bm.main()
    assert code == 0

    ri_calls = {"n": 0}
    ccrr_calls = {"n": 0}
    original_ri_compute = bm.RI.compute
    original_ccrr_compute = bm.CCRR.compute

    def wrapped_ri_compute(*args, **kwargs):
        ri_calls["n"] += 1
        return original_ri_compute(*args, **kwargs)

    def wrapped_ccrr_compute(*args, **kwargs):
        ccrr_calls["n"] += 1
        return original_ccrr_compute(*args, **kwargs)

    monkeypatch.setattr(bm.RI, "compute", wrapped_ri_compute)
    monkeypatch.setattr(bm.CCRR, "compute", wrapped_ccrr_compute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            model,
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
            "--ccrr-start-k",
            "300",
        ],
    )
    code = bm.main()
    assert code == 0
    assert ri_calls["n"] == 0
    assert ccrr_calls["n"] > 0


def test_benchmark_rejects_invalid_ccrr_alpha(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "missing.csv"
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--ccrr-alpha",
            "0",
        ],
    )
    with pytest.raises(ValueError, match=r"--ccrr-alpha must be in \(0, 1\]"):
        bm.main()


def test_benchmark_runs_with_progress_off(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)

    output_dir = tmp_path / "out"
    model = "M1"

    def fake_registry() -> dict:
        return {model: object()}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm, "_build_model_registry", fake_registry)
    monkeypatch.setattr(bm.ee, "embed_manifest", fake_embed_manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            model,
            "--output-dir",
            str(output_dir),
            "--mode",
            "global",
            "--k-candidates",
            "1,3",
            "--progress",
            "off",
        ],
    )

    code = bm.main()
    assert code == 0
    dataset_dir = output_dir / manifest_path.stem
    assert (dataset_dir / "results" / "metrics.csv").exists()
