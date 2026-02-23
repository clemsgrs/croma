from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plotting import (
    MODEL_COLOR_MAP,
    _color_for_model,
    plot_benchmark_6panel_summary,
    plot_bio_vs_center_scatter,
    plot_knn_bio_k_sweep,
    plot_knn_center_k_sweep,
    plot_mari_k_sweep,
    plot_mari_vs_ri_scatter,
    plot_ri_k_sweep,
)


def _sample_k_rows() -> list[dict]:
    return [
        {
            "model": "Virchow2",
            "k": 1,
            "knn_bacc": 0.60,
            "knn_center_bacc": 0.66,
            "ri": 0.40,
            "mari": 0.45,
            "selected_k": 3,
            "selected_k_center": 1,
        },
        {
            "model": "Virchow2",
            "k": 3,
            "knn_bacc": 0.72,
            "knn_center_bacc": 0.63,
            "ri": 0.55,
            "mari": 0.59,
            "selected_k": 3,
            "selected_k_center": 1,
        },
        {
            "model": "UNI",
            "k": 1,
            "knn_bacc": 0.58,
            "knn_center_bacc": 0.69,
            "ri": 0.30,
            "mari": 0.28,
            "selected_k": 1,
            "selected_k_center": 3,
        },
        {
            "model": "UNI",
            "k": 3,
            "knn_bacc": 0.57,
            "knn_center_bacc": 0.74,
            "ri": 0.35,
            "mari": 0.33,
            "selected_k": 1,
            "selected_k_center": 3,
        },
    ]


def _sample_summary_rows() -> list[dict]:
    return [
        {
            "model": "Virchow2",
            "bio_knn_bacc": 0.72,
            "center_knn_bacc": 0.66,
            "ri": 0.55,
            "mari": 0.59,
        },
        {
            "model": "UNI",
            "bio_knn_bacc": 0.58,
            "center_knn_bacc": 0.74,
            "ri": 0.35,
            "mari": 0.33,
        },
    ]


def test_color_map_integrity_matches_expected_values() -> None:
    expected = {
        "Virchow2": "#ff7f0e",
        "Virchow": "#ffbb78",
        "UNI2-h": "#2ca02c",
        "UNI": "#98df8a",
        "CONCHv1.5": "#d62728",
        "CONCH": "#ff9896",
        "Phikon-v2": "#9467bd",
        "Phikon": "#c5b0d5",
        "H-optimus-1": "#8c564b",
        "H-optimus-0": "#c49c94",
        "H0-mini": "#d7b5b0",
        "Prov-GigaPath": "#1f77b4",
        "Midnight": "#17becf",
        "Hibou-L": "#e377c2",
        "Hibou-B": "#f7b6d2",
        "Prost40M": "#636363",
    }
    assert MODEL_COLOR_MAP == expected


def test_unknown_model_color_fallback_is_gray() -> None:
    assert _color_for_model("UnknownModel") == "#808080"


def test_plot_knn_bio_k_sweep_creates_png(tmp_path: Path) -> None:
    out_path = tmp_path / "knn_bio_k_sweep.png"
    plot_knn_bio_k_sweep(rows=_sample_k_rows(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_knn_center_k_sweep_creates_png(tmp_path: Path) -> None:
    out_path = tmp_path / "knn_center_k_sweep.png"
    plot_knn_center_k_sweep(rows=_sample_k_rows(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_ri_k_sweep_creates_png(tmp_path: Path) -> None:
    out_path = tmp_path / "ri_k_sweep.png"
    plot_ri_k_sweep(rows=_sample_k_rows(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_mari_k_sweep_creates_png(tmp_path: Path) -> None:
    out_path = tmp_path / "mari_k_sweep.png"
    plot_mari_k_sweep(rows=_sample_k_rows(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_bio_vs_center_scatter_creates_png(tmp_path: Path) -> None:
    out_path = tmp_path / "bio_vs_center_scatter.png"
    plot_bio_vs_center_scatter(rows=_sample_summary_rows(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_mari_vs_ri_scatter_creates_png(tmp_path: Path) -> None:
    out_path = tmp_path / "mari_vs_ri_scatter.png"
    plot_mari_vs_ri_scatter(rows=_sample_summary_rows(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_benchmark_6panel_summary_creates_png(tmp_path: Path) -> None:
    out_path = tmp_path / "benchmark_6panel_summary.png"
    plot_benchmark_6panel_summary(rows=_sample_summary_rows(), k_sweep_rows=_sample_k_rows(), out_path=out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_center_curve_uses_cross_and_star_markers(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes

    calls: list[str | None] = []
    original_scatter = matplotlib.axes.Axes.scatter

    def spy_scatter(self, x, y, *args, **kwargs):
        calls.append(kwargs.get("marker"))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)

    rows = [
        {
            "model": "Virchow2",
            "k": 1,
            "knn_bacc": 0.60,
            "knn_center_bacc": 0.66,
            "ri": 0.40,
            "mari": 0.45,
            "selected_k": 3,
            "selected_k_center": 1,
        },
        {
            "model": "Virchow2",
            "k": 3,
            "knn_bacc": 0.72,
            "knn_center_bacc": 0.63,
            "ri": 0.55,
            "mari": 0.59,
            "selected_k": 3,
            "selected_k_center": 1,
        },
    ]
    out_path = tmp_path / "center_markers.png"
    plot_knn_center_k_sweep(rows=rows, out_path=out_path)

    assert "X" in calls
    assert "*" in calls


def test_ri_and_mari_curves_highlight_selected_k_with_star(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes

    star_x_values: list[int] = []
    original_scatter = matplotlib.axes.Axes.scatter

    def spy_scatter(self, x, y, *args, **kwargs):
        marker = kwargs.get("marker")
        if marker == "*":
            xx = np.asarray(x).astype(int)
            if xx.size > 0:
                star_x_values.append(int(xx[0]))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)

    rows = [
        {
            "model": "Virchow2",
            "k": 1,
            "knn_bacc": 0.60,
            "knn_center_bacc": 0.66,
            "ri": 0.40,
            "mari": 0.45,
            "selected_k": 3,
            "selected_k_center": 1,
        },
        {
            "model": "Virchow2",
            "k": 3,
            "knn_bacc": 0.72,
            "knn_center_bacc": 0.63,
            "ri": 0.55,
            "mari": 0.59,
            "selected_k": 3,
            "selected_k_center": 1,
        },
    ]

    plot_ri_k_sweep(rows=rows, out_path=tmp_path / "ri_star.png")
    plot_mari_k_sweep(rows=rows, out_path=tmp_path / "mari_star.png")

    assert 3 in star_x_values
