import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plotting import (
    plot_benchmark_6panel_summary,
    plot_bio_vs_center_scatter,
    plot_ccrr_ltm_comparison,
    plot_ccrr_m_sweep_with_ltm,
    plot_knn_center_k_sweep,
    plot_mari_k_sweep,
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


def _sample_ccrr_m_rows() -> list[dict]:
    return [
        {"model": "Virchow2", "m": 1, "ccrr": 1.40, "ccrr_ltm_alpha": 1.05},
        {"model": "Virchow2", "m": 2, "ccrr": 1.22, "ccrr_ltm_alpha": 0.95},
        {"model": "Virchow2", "m": 3, "ccrr": 1.10, "ccrr_ltm_alpha": 0.88},
        {"model": "UNI", "m": 1, "ccrr": 0.92, "ccrr_ltm_alpha": 0.70},
        {"model": "UNI", "m": 2, "ccrr": 0.97, "ccrr_ltm_alpha": 0.75},
        {"model": "UNI", "m": 3, "ccrr": 1.03, "ccrr_ltm_alpha": 0.80},
    ]


def _sample_ccrr_ltm_rows() -> list[dict]:
    return [
        {"model": "Virchow2", "ccrr": 1.30, "ccrr_ltm_alpha": 1.10, "ccrr_alpha": 0.10},
        {"model": "UNI", "ccrr": 1.05, "ccrr_ltm_alpha": 0.82, "ccrr_alpha": 0.10},
        {"model": "CONCH", "ccrr": 0.96, "ccrr_ltm_alpha": 0.61, "ccrr_alpha": 0.10},
    ]
def test_representative_plotting_entrypoints_write_pngs(tmp_path: Path) -> None:
    cases = [
        (plot_bio_vs_center_scatter, {"rows": _sample_summary_rows()}, "bio_vs_center_scatter.png"),
        (plot_ccrr_m_sweep_with_ltm, {"rows": _sample_ccrr_m_rows()}, "ccrr_m_sweep.png"),
        (
            plot_benchmark_6panel_summary,
            {"rows": _sample_summary_rows(), "k_sweep_rows": _sample_k_rows()},
            "benchmark_6panel_summary.png",
        ),
    ]

    for fn, kwargs, filename in cases:
        out_path = tmp_path / filename
        fn(out_path=out_path, **kwargs)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_plot_ccrr_ltm_comparison_filters_invalid_rows_and_sorts_descending(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes

    points: list[tuple[float, float]] = []
    ltm_heights: list[float] = []
    original_scatter = matplotlib.axes.Axes.scatter
    original_bar = matplotlib.axes.Axes.bar

    def spy_scatter(self, x, y, *args, **kwargs):
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        if xx.size == 1 and yy.size == 1:
            points.append((float(xx[0]), float(yy[0])))
        return original_scatter(self, x, y, *args, **kwargs)

    def spy_bar(self, x, height, *args, **kwargs):
        if "LTM@" in str(kwargs.get("label", "")):
            ltm_heights.extend(float(v) for v in np.asarray(height, dtype=float).tolist())
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy_bar)

    rows = _sample_ccrr_ltm_rows() + [{"model": "Bad", "ccrr": float("nan"), "ccrr_ltm_alpha": 0.5, "ccrr_alpha": 0.1}]
    out_path = tmp_path / "ccrr_ltm_comparison.png"
    plot_ccrr_ltm_comparison(rows=rows, out_path=out_path)

    assert out_path.exists()
    assert {(1.30, 1.10), (1.05, 0.82), (0.96, 0.61)}.issubset(set(points))
    assert ltm_heights == sorted(ltm_heights, reverse=True)


def test_selected_k_markers_are_highlighted(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes

    markers: list[str | None] = []
    star_x_values: list[int] = []
    original_scatter = matplotlib.axes.Axes.scatter

    def spy_scatter(self, x, y, *args, **kwargs):
        marker = kwargs.get("marker")
        markers.append(marker)
        if marker == "*":
            xx = np.asarray(x, dtype=int)
            if xx.size > 0:
                star_x_values.append(int(xx[0]))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)

    rows = _sample_k_rows()
    plot_knn_center_k_sweep(rows=rows, out_path=tmp_path / "center_markers.png")
    plot_ri_k_sweep(rows=rows, out_path=tmp_path / "ri_star.png")
    plot_mari_k_sweep(rows=rows, out_path=tmp_path / "mari_star.png")

    assert "X" in markers
    assert "*" in markers
    assert 3 in star_x_values
