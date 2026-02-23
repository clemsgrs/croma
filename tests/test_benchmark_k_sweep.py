from __future__ import annotations

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
    ) -> tuple[Path, tuple[int, int]]:
        model_name = output_path.stem.split(".")[-1]
        arr = _toy_features(model_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm.ee, "_build_model_registry", fake_registry)
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
    bio_center_plot_path = dataset_dir / "plots" / "bio_vs_center_scatter.png"
    mari_ri_plot_path = dataset_dir / "plots" / "mari_vs_ri_scatter.png"
    summary_plot_path = dataset_dir / "plots" / "benchmark_6panel_summary.png"
    legacy_rank_plot = dataset_dir / "plots" / "ri_mari_rank.png"
    legacy_three_panel = dataset_dir / "plots" / "knn_bacc_ri_k_sweep.png"
    legacy_mari_dist_plot = dataset_dir / "plots" / "mari_sample_distributions.png"
    legacy_old_scatter = dataset_dir / "plots" / "biological_vs_center_knn_bacc.png"
    legacy_old_mari_ri = dataset_dir / "plots" / "ri_vs_mari_scatter.png"
    mari_dist_dir = dataset_dir / "results" / "mari_sample_distributions"
    ri_dist_dir = dataset_dir / "results" / "ri_sample_distributions"

    assert k_sweep_csv.exists()
    assert k_sweep_json.exists()
    assert bio_k_plot_path.exists()
    assert center_k_plot_path.exists()
    assert ri_k_plot_path.exists()
    assert mari_plot_path.exists()
    assert bio_center_plot_path.exists()
    assert mari_ri_plot_path.exists()
    assert summary_plot_path.exists()
    assert mari_dist_dir.exists()
    assert ri_dist_dir.exists()
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

    metrics_df = pd.read_csv(dataset_dir / "results" / "metrics.csv")
    assert "ri_undefined_frac" in metrics_df.columns
    assert "mari_undefined_frac" in metrics_df.columns
    assert "ri_samples_path" in metrics_df.columns
    assert "mari_samples_path" in metrics_df.columns
    assert "bio_knn_bacc" in metrics_df.columns
    assert "center_knn_bacc" in metrics_df.columns
    assert "selected_k_center" in metrics_df.columns
    assert ((metrics_df["ri_undefined_frac"] >= 0.0) & (metrics_df["ri_undefined_frac"] <= 1.0)).all()
    assert ((metrics_df["mari_undefined_frac"] >= 0.0) & (metrics_df["mari_undefined_frac"] <= 1.0)).all()

    for model in models:
        mari_dist_path = mari_dist_dir / f"mari_samples.{model}.npy"
        ri_dist_path = ri_dist_dir / f"ri_samples.{model}.npy"
        assert mari_dist_path.exists()
        assert ri_dist_path.exists()
        arr = np.load(mari_dist_path)
        ri_arr = np.load(ri_dist_path)
        assert 0 < arr.shape[0] <= 8
        assert 0 < ri_arr.shape[0] <= 8


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
    ) -> tuple[Path, tuple[int, int]]:
        model_name = output_path.stem.split(".")[-1]
        arr = _toy_features(model_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm.ee, "_build_model_registry", fake_registry)
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
    ) -> tuple[Path, tuple[int, int]]:
        model_name = output_path.stem.split(".")[-1]
        arr = _toy_features(model_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm.ee, "_build_model_registry", fake_registry)
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

    monkeypatch.setattr(bm.ee, "_build_model_registry", fake_registry)
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
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm.ee, "_build_model_registry", fake_registry)
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
                "ri_samples_path": str(results_dir / "ri_sample_distributions" / f"ri_samples.{model}.npy"),
                "mari_samples_path": str(results_dir / "mari_sample_distributions" / f"mari_samples.{model}.npy"),
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
    for subdir, prefix in [("ri_sample_distributions", "ri_samples"), ("mari_sample_distributions", "mari_samples")]:
        dist_dir = results_dir / subdir
        dist_dir.mkdir(parents=True, exist_ok=True)
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
    ) -> tuple[Path, tuple[int, int]]:
        arr = _toy_features(model)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, arr)
        return output_path, (int(arr.shape[0]), int(arr.shape[1]))

    monkeypatch.setattr(bm.ee, "_build_model_registry", fake_registry)
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
