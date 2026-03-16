import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plotting import (
    _support_plot_rows,
    plot_bio_vs_confounder_scatter,
    plot_ccmr_ltm_comparison,
    plot_ccmr_m_sweep_with_ltm,
    plot_knn_confounder_k_sweep,
    plot_mari_k_sweep,
    plot_ri_mari_support,
    plot_ri_k_sweep,
)


def _sample_k_rows() -> list[dict]:
    return [
        {
            "model": "Virchow2",
            "k": 1,
            "knn_bacc": 0.60,
            "knn_confounder_bacc": 0.66,
            "ri": 0.40,
            "mari": 0.45,
            "selected_k": 3,
            "selected_k_confounder": 1,
            "confounder_display_name": "Medical Center",
        },
        {
            "model": "Virchow2",
            "k": 3,
            "knn_bacc": 0.72,
            "knn_confounder_bacc": 0.63,
            "ri": 0.55,
            "mari": 0.59,
            "selected_k": 3,
            "selected_k_confounder": 1,
            "confounder_display_name": "Medical Center",
        },
        {
            "model": "UNI",
            "k": 1,
            "knn_bacc": 0.58,
            "knn_confounder_bacc": 0.69,
            "ri": 0.30,
            "mari": 0.28,
            "selected_k": 1,
            "selected_k_confounder": 3,
            "confounder_display_name": "Medical Center",
        },
        {
            "model": "UNI",
            "k": 3,
            "knn_bacc": 0.57,
            "knn_confounder_bacc": 0.74,
            "ri": 0.35,
            "mari": 0.33,
            "selected_k": 1,
            "selected_k_confounder": 3,
            "confounder_display_name": "Medical Center",
        },
    ]


def _sample_summary_rows() -> list[dict]:
    return [
        {
            "model": "Virchow2",
            "bio_knn_bacc": 0.72,
            "confounder_knn_bacc": 0.66,
            "ri": 0.55,
            "mari": 0.59,
            "confounder_display_name": "Medical Center",
        },
        {
            "model": "UNI",
            "bio_knn_bacc": 0.58,
            "confounder_knn_bacc": 0.74,
            "ri": 0.35,
            "mari": 0.33,
            "confounder_display_name": "Medical Center",
        },
    ]


def _sample_support_rows() -> list[dict]:
    return [
        {
            "model": "Virchow2",
            "ri_undefined_frac": 0.12,
            "mari_undefined_frac": 0.18,
        },
        {
            "model": "UNI",
            "ri_undefined_frac": 0.35,
            "mari_undefined_frac": 0.52,
        },
    ]


def _sample_ccmr_m_rows() -> list[dict]:
    return [
        {"model": "Virchow2", "m": 1, "ccmr": 1.40, "ccmr_ltm_alpha": 1.05},
        {"model": "Virchow2", "m": 2, "ccmr": 1.22, "ccmr_ltm_alpha": 0.95},
        {"model": "Virchow2", "m": 3, "ccmr": 1.10, "ccmr_ltm_alpha": 0.88},
        {"model": "UNI", "m": 1, "ccmr": 0.92, "ccmr_ltm_alpha": 0.70},
        {"model": "UNI", "m": 2, "ccmr": 0.97, "ccmr_ltm_alpha": 0.75},
        {"model": "UNI", "m": 3, "ccmr": 1.03, "ccmr_ltm_alpha": 0.80},
    ]


def _sample_ccmr_ltm_rows() -> list[dict]:
    return [
        {"model": "Virchow2", "ccmr": 1.30, "ccmr_ltm_alpha": 1.10, "ccmr_alpha": 0.10},
        {"model": "UNI", "ccmr": 1.05, "ccmr_ltm_alpha": 0.82, "ccmr_alpha": 0.10},
        {"model": "CONCH", "ccmr": 0.96, "ccmr_ltm_alpha": 0.61, "ccmr_alpha": 0.10},
    ]


def test_representative_plotting_entrypoints_write_pngs(tmp_path: Path) -> None:
    cases = [
        (
            plot_bio_vs_confounder_scatter,
            {"rows": _sample_summary_rows()},
            "bio_vs_confounder_scatter.png",
        ),
        (
            plot_ccmr_m_sweep_with_ltm,
            {"rows": _sample_ccmr_m_rows()},
            "ccmr_m_sweep.png",
        ),
        (
            plot_ri_k_sweep,
            {"rows": _sample_k_rows()},
            "ri_k_sweep.png",
        ),
        (
            plot_ri_mari_support,
            {"rows": _sample_support_rows()},
            "ri_mari_support.png",
        ),
    ]

    for fn, kwargs, filename in cases:
        out_path = tmp_path / filename
        fn(out_path=out_path, **kwargs)
        assert out_path.exists()
        assert out_path.stat().st_size > 0


def test_support_plot_rows_use_defined_fraction_thresholds_and_worst_first_order() -> None:
    rows = _support_plot_rows(_sample_support_rows())

    assert [(row["model"], row["metric"]) for row in rows] == [
        ("UNI", "RI"),
        ("UNI", "MaRI"),
        ("Virchow2", "RI"),
        ("Virchow2", "MaRI"),
    ]

    indexed = {(row["model"], row["metric"]): row for row in rows}
    assert indexed[("Virchow2", "RI")]["defined_frac"] == pytest.approx(0.88)
    assert indexed[("Virchow2", "RI")]["status"] == "good"
    assert indexed[("UNI", "RI")]["defined_frac"] == pytest.approx(0.65)
    assert indexed[("UNI", "RI")]["status"] == "warning"
    assert indexed[("UNI", "MaRI")]["defined_frac"] == pytest.approx(0.48)
    assert indexed[("UNI", "MaRI")]["status"] == "critical"


def test_plot_ccmr_ltm_comparison_filters_invalid_rows_and_sorts_descending(
    monkeypatch, tmp_path: Path
) -> None:
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
            ltm_heights.extend(
                float(v) for v in np.asarray(height, dtype=float).tolist()
            )
        return original_bar(self, x, height, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy_bar)

    rows = _sample_ccmr_ltm_rows() + [
        {"model": "Bad", "ccmr": float("nan"), "ccmr_ltm_alpha": 0.5, "ccmr_alpha": 0.1}
    ]
    out_path = tmp_path / "ccmr_ltm_comparison.png"
    plot_ccmr_ltm_comparison(rows=rows, out_path=out_path)

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
    plot_knn_confounder_k_sweep(rows=rows, out_path=tmp_path / "confounder_markers.png")
    plot_ri_k_sweep(rows=rows, out_path=tmp_path / "ri_star.png")
    plot_mari_k_sweep(rows=rows, out_path=tmp_path / "mari_star.png")

    assert "X" in markers
    assert "*" in markers
    assert 3 in star_x_values


def test_confounder_plot_uses_display_name(tmp_path: Path) -> None:
    rows = [
        {
            "model": "Virchow2",
            "bio_knn_bacc": 0.72,
            "confounder_knn_bacc": 0.66,
            "ri": 0.55,
            "mari": 0.59,
            "confounder_display_name": "Scanner Vendor",
        }
    ]

    out_path = tmp_path / "bio_vs_confounder_scatter.png"
    plot_bio_vs_confounder_scatter(rows=rows, out_path=out_path)

    assert out_path.exists()
