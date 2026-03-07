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
            "sample_id": ["s0", "s1", "s2", "s3", "s0", "s4", "s5", "s6"],
            "image_path": [f"/tmp/{i}.png" for i in range(8)],
            "label": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "medical_center": ["C1", "C2", "C1", "C2", "C1", "C2", "C1", "C2"],
            "slide_id": ["sl0", "sl1", "sl2", "sl3", "sl0", "sl4", "sl5", "sl6"],
            "subset": ["pair1", "pair1", "pair1", "pair1", "pair2", "pair2", "pair2", "pair2"],
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


def _install_fake_registry_and_embed(monkeypatch, model: str = "M1") -> None:
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
        arr = _toy_features()
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

    for name in (
        "plot_benchmark_6panel_summary",
        "plot_bio_vs_center_scatter",
        "plot_ccrr_ltm_comparison",
        "plot_ccrr_m_sweep_with_ltm",
        "plot_ccrr_sample_distributions",
        "plot_ccrr_trend_quadrants",
        "plot_ccrr_vs_mari_scatter",
        "plot_knn_bio_k_sweep",
        "plot_knn_center_k_sweep",
        "plot_mari_k_sweep",
        "plot_mari_vs_ri_scatter",
        "plot_ri_k_sweep",
    ):
        monkeypatch.setattr(bm, name, fake_plot)


def _run_benchmark(
    monkeypatch,
    *,
    manifest_path: Path,
    output_dir: Path,
    model: str = "M1",
    evaluation_design: str = "dataset_wide",
    extra_args: list[str] | None = None,
) -> int:
    args = [
        "benchmark.py",
        "--manifest",
        str(manifest_path),
        "--models",
        model,
        "--output-dir",
        str(output_dir),
        "--evaluation-design",
        evaluation_design,
        "--k-candidates",
        "1,3",
        "--progress",
        "off",
    ]
    if extra_args:
        args.extend(extra_args)
    monkeypatch.setattr(sys, "argv", args)
    return bm.main()


def test_tau_change_recomputes_only_mari(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, model="M1")
    _install_noop_plots(monkeypatch)

    assert _run_benchmark(monkeypatch, manifest_path=manifest_path, output_dir=output_dir) == 0

    calls = {"ri": 0, "mari": 0, "ccrr": 0, "knn": 0}
    original_ri_compute = bm.RI.compute
    original_mari_compute = bm.MaRI.compute
    original_ccrr_compute = bm.CCRR.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_compute(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_compute(*args, **kwargs)

    def wrapped_mari_compute(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_compute(*args, **kwargs)

    def wrapped_ccrr_compute(*args, **kwargs):
        calls["ccrr"] += 1
        return original_ccrr_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    monkeypatch.setattr(bm.RI, "compute", wrapped_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", wrapped_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", wrapped_ccrr_compute)
    monkeypatch.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        extra_args=["--tau", "0.3"],
    ) == 0

    assert calls["mari"] > 0
    assert calls["ri"] == 0
    assert calls["ccrr"] == 0
    assert calls["knn"] == 0


def test_ccrr_search_change_recomputes_only_ccrr(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, model="M1")
    _install_noop_plots(monkeypatch)

    assert _run_benchmark(monkeypatch, manifest_path=manifest_path, output_dir=output_dir) == 0

    calls = {"ri": 0, "mari": 0, "ccrr": 0, "knn": 0}
    original_ri_compute = bm.RI.compute
    original_mari_compute = bm.MaRI.compute
    original_ccrr_compute = bm.CCRR.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_compute(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_compute(*args, **kwargs)

    def wrapped_mari_compute(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_compute(*args, **kwargs)

    def wrapped_ccrr_compute(*args, **kwargs):
        calls["ccrr"] += 1
        return original_ccrr_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    monkeypatch.setattr(bm.RI, "compute", wrapped_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", wrapped_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", wrapped_ccrr_compute)
    monkeypatch.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        extra_args=["--ccrr-start-k", "300"],
    ) == 0

    assert calls["ccrr"] > 0
    assert calls["ri"] == 0
    assert calls["mari"] == 0
    assert calls["knn"] == 0


def test_k_values_change_recomputes_knn_ri_mari_not_ccrr(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, model="M1")
    _install_noop_plots(monkeypatch)

    assert _run_benchmark(monkeypatch, manifest_path=manifest_path, output_dir=output_dir) == 0

    calls = {"ri": 0, "mari": 0, "ccrr": 0, "knn": 0}
    original_ri_compute = bm.RI.compute
    original_mari_compute = bm.MaRI.compute
    original_ccrr_compute = bm.CCRR.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_compute(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_compute(*args, **kwargs)

    def wrapped_mari_compute(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_compute(*args, **kwargs)

    def wrapped_ccrr_compute(*args, **kwargs):
        calls["ccrr"] += 1
        return original_ccrr_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    monkeypatch.setattr(bm.RI, "compute", wrapped_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", wrapped_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", wrapped_ccrr_compute)
    monkeypatch.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        extra_args=["--k-candidates", "1,5"],
    ) == 0

    assert calls["ri"] > 0
    assert calls["mari"] > 0
    assert calls["knn"] > 0
    assert calls["ccrr"] == 0


def test_evaluation_design_change_recomputes_all_artifacts(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, model="M1")
    _install_noop_plots(monkeypatch)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        evaluation_design="dataset_wide",
    ) == 0

    calls = {"ri": 0, "mari": 0, "ccrr": 0, "knn": 0}
    original_ri_compute = bm.RI.compute
    original_mari_compute = bm.MaRI.compute
    original_ccrr_compute = bm.CCRR.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_compute(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_compute(*args, **kwargs)

    def wrapped_mari_compute(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_compute(*args, **kwargs)

    def wrapped_ccrr_compute(*args, **kwargs):
        calls["ccrr"] += 1
        return original_ccrr_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    monkeypatch.setattr(bm.RI, "compute", wrapped_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", wrapped_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", wrapped_ccrr_compute)
    monkeypatch.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        evaluation_design="paired_2x2",
    ) == 0

    assert calls["ri"] > 0
    assert calls["mari"] > 0
    assert calls["ccrr"] > 0
    assert calls["knn"] > 0


def test_recompute_metrics_flag_forces_all(monkeypatch, tmp_path: Path) -> None:
    manifest = _toy_manifest()
    manifest_path = tmp_path / "toy.csv"
    manifest.to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch, model="M1")
    _install_noop_plots(monkeypatch)

    assert _run_benchmark(monkeypatch, manifest_path=manifest_path, output_dir=output_dir) == 0

    calls = {"ri": 0, "mari": 0, "ccrr": 0, "knn": 0}
    original_ri_compute = bm.RI.compute
    original_mari_compute = bm.MaRI.compute
    original_ccrr_compute = bm.CCRR.compute
    original_knn = bm._knn_balanced_accuracy_by_k

    def wrapped_ri_compute(*args, **kwargs):
        calls["ri"] += 1
        return original_ri_compute(*args, **kwargs)

    def wrapped_mari_compute(*args, **kwargs):
        calls["mari"] += 1
        return original_mari_compute(*args, **kwargs)

    def wrapped_ccrr_compute(*args, **kwargs):
        calls["ccrr"] += 1
        return original_ccrr_compute(*args, **kwargs)

    def wrapped_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    monkeypatch.setattr(bm.RI, "compute", wrapped_ri_compute)
    monkeypatch.setattr(bm.MaRI, "compute", wrapped_mari_compute)
    monkeypatch.setattr(bm.CCRR, "compute", wrapped_ccrr_compute)
    monkeypatch.setattr(bm, "_knn_balanced_accuracy_by_k", wrapped_knn)

    assert _run_benchmark(
        monkeypatch,
        manifest_path=manifest_path,
        output_dir=output_dir,
        extra_args=["--recompute-metrics"],
    ) == 0

    assert calls["ri"] > 0
    assert calls["mari"] > 0
    assert calls["ccrr"] > 0
    assert calls["knn"] > 0
