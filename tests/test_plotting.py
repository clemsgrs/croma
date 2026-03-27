import sys
from pathlib import Path

import numpy as np
import matplotlib.figure
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plotting import (
    _color_for_model,
    _pdf_export_path,
    _png_export_path,
    _support_plot_rows,
    plot_bio_vs_confounder_scatter,
    plot_ccmr_ltm_comparison,
    plot_ccmr_m_sweep_with_ltm,
    plot_ccmr_sample_distributions,
    plot_knn_confounder_k_sweep,
    plot_mari_k_sweep,
    plot_q_alpha_vs_ccmr_scatter,
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


def _sample_dense_k_rows() -> list[dict]:
    rows: list[dict] = []
    for k in range(1, 26):
        rows.append(
            {
                "model": "Virchow2",
                "k": k,
                "knn_bacc": 0.50 + 0.01 * k,
                "knn_confounder_bacc": 0.70 - 0.005 * k,
                "ri": 0.30 + 0.01 * k,
                "mari": 0.35 + 0.008 * k,
                "selected_k": 11,
                "selected_k_confounder": 7,
                "confounder_display_name": "Medical Center",
            }
        )
    return rows


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
            "mari_undefined_frac": 0.12,
        },
        {
            "model": "UNI",
            "ri_undefined_frac": 0.35,
            "mari_undefined_frac": 0.35,
        },
        {
            "model": "CONCH",
            "ri_undefined_frac": 0.58,
            "mari_undefined_frac": 0.58,
        },
        {
            "model": "Phikon",
            "ri_undefined_frac": 0.80,
            "mari_undefined_frac": 0.80,
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
        {"model": "CONCH", "m": 1, "ccmr": 1.15, "ccmr_ltm_alpha": 0.90},
        {"model": "CONCH", "m": 2, "ccmr": 1.08, "ccmr_ltm_alpha": 0.85},
        {"model": "CONCH", "m": 3, "ccmr": 1.02, "ccmr_ltm_alpha": 0.78},
        {"model": "Phikon", "m": 1, "ccmr": 0.88, "ccmr_ltm_alpha": 0.65},
        {"model": "Phikon", "m": 2, "ccmr": 0.93, "ccmr_ltm_alpha": 0.70},
        {"model": "Phikon", "m": 3, "ccmr": 0.99, "ccmr_ltm_alpha": 0.74},
    ]


def _sample_ccmr_ltm_rows() -> list[dict]:
    return [
        {"model": "Virchow2", "ccmr": 1.30, "ccmr_ltm_alpha": 1.10, "ccmr_alpha": 0.10},
        {"model": "UNI", "ccmr": 1.05, "ccmr_ltm_alpha": 0.82, "ccmr_alpha": 0.10},
        {"model": "CONCH", "ccmr": 0.96, "ccmr_ltm_alpha": 0.61, "ccmr_alpha": 0.10},
    ]


def _sample_ccmr_distribution_rows(tmp_path: Path) -> list[dict]:
    by_model = {
        "Virchow2": np.asarray([0.82, 1.10, 1.32, 1.45], dtype=float),
        "UNI": np.asarray([0.60, 0.84, 0.93, 1.20], dtype=float),
        "CONCH": np.asarray([1.02, 1.18, 1.28, 1.35], dtype=float),
    }
    rows: list[dict] = []
    for model, values in by_model.items():
        path = tmp_path / f"{model}.ccmr.npy"
        np.save(path, values)
        rows.append(
            {
                "model": model,
                "ccmr": {"Virchow2": 1.55, "UNI": 0.97, "CONCH": 1.21}[model],
                "ccmr_q_alpha": {"Virchow2": 0.82, "UNI": 0.60, "CONCH": 1.02}[model],
                "ccmr_alpha": 0.10,
                "ccmr_samples_path": str(path),
            }
        )
    return rows


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
        png_path = _png_export_path(out_path)
        assert png_path.exists()
        assert png_path.stat().st_size > 0


def _sample_ccmr_distribution_rows(tmp_path: Path) -> list[dict]:
    samples_path = tmp_path / "ccmr_samples_virchow2.npy"
    np.save(samples_path, np.array([0.8, 0.9, 1.1, 1.2, 1.05], dtype=float))
    return [
        {
            "model": "Virchow2",
            "ccmr_samples_path": str(samples_path),
            "ccmr_q_alpha": 0.85,
            "ccmr_alpha": 0.10,
            "ccmr": 1.05,
        }
    ]


def test_plot_writes_matching_pdf_export(tmp_path: Path) -> None:
    out_path = tmp_path / "bio_vs_confounder_scatter.png"

    plot_bio_vs_confounder_scatter(rows=_sample_summary_rows(), out_path=out_path)

    assert _png_export_path(out_path).exists()
    pdf_path = _pdf_export_path(out_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_ccmr_distribution_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    rows = _sample_ccmr_distribution_rows(tmp_path)
    out_path = tmp_path / "ccmr_sample_distributions.png"

    plot_ccmr_sample_distributions(rows=rows, out_path=out_path)

    assert _png_export_path(out_path).exists()
    pdf_path = _pdf_export_path(out_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_precomputed_model_aliases_reuse_family_colors() -> None:
    assert _color_for_model("PRISM") == _color_for_model("Virchow")
    assert _color_for_model("TITAN") == _color_for_model("CONCHv1.5")


def test_scatter_uses_figure_level_bottom_legend(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes
    import matplotlib.figure

    figure_legend_calls: list[dict] = []
    axes_legend_calls: list[dict] = []
    original_figure_legend = matplotlib.figure.Figure.legend
    original_axes_legend = matplotlib.axes.Axes.legend

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(dict(kwargs))
        return original_figure_legend(self, *args, **kwargs)

    def spy_axes_legend(self, *args, **kwargs):
        axes_legend_calls.append(dict(kwargs))
        return original_axes_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_axes_legend)

    plot_bio_vs_confounder_scatter(
        rows=_sample_summary_rows(), out_path=tmp_path / "scatter.png"
    )

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) >= 2
    bbox = legend_kwargs.get("bbox_to_anchor")
    assert bbox is not None
    assert float(bbox[1]) >= 0.0


def test_multi_panel_plot_uses_single_figure_level_legend(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes
    import matplotlib.figure

    figure_legend_calls: list[dict] = []
    axes_legend_calls: list[dict] = []
    original_figure_legend = matplotlib.figure.Figure.legend
    original_axes_legend = matplotlib.axes.Axes.legend

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(dict(kwargs))
        return original_figure_legend(self, *args, **kwargs)

    def spy_axes_legend(self, *args, **kwargs):
        axes_legend_calls.append(dict(kwargs))
        return original_axes_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_axes_legend)

    plot_ccmr_m_sweep_with_ltm(
        rows=_sample_ccmr_m_rows(), out_path=tmp_path / "ccmr_m_sweep.png"
    )

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) == 4


def test_support_plot_rows_use_one_row_per_model_defined_share_thresholds_and_worst_first_order() -> None:
    rows = _support_plot_rows(_sample_support_rows())

    assert [row["model"] for row in rows] == [
        "Phikon",
        "CONCH",
        "UNI",
        "Virchow2",
    ]

    indexed = {row["model"]: row for row in rows}
    assert indexed["Virchow2"]["defined_frac"] == pytest.approx(0.88)
    assert indexed["Virchow2"]["status"] == "good"
    assert indexed["UNI"]["defined_frac"] == pytest.approx(0.65)
    assert indexed["UNI"]["status"] == "good"
    assert indexed["CONCH"]["defined_frac"] == pytest.approx(0.42)
    assert indexed["CONCH"]["status"] == "warning"
    assert indexed["Phikon"]["defined_frac"] == pytest.approx(0.20)
    assert indexed["Phikon"]["status"] == "critical"
    assert indexed["UNI"]["label"] == "65%"


def test_support_plot_uses_bottom_legend_for_thresholds(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.figure

    legend_calls: list[dict[str, object]] = []
    original_legend = matplotlib.figure.Figure.legend

    def spy_legend(self, *args, **kwargs):
        handles = list(args[0]) if args else list(kwargs.get("handles", []))
        labels = [str(getattr(handle, "get_label", lambda: "")()) for handle in handles]
        legend_calls.append(
            {
                "labels": labels,
                "loc": kwargs.get("loc"),
                "ncol": kwargs.get("ncol"),
            }
        )
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_legend)

    out_path = tmp_path / "ri_mari_support.png"
    plot_ri_mari_support(rows=_sample_support_rows(), out_path=out_path)

    assert _png_export_path(out_path).exists()
    assert legend_calls
    labels = legend_calls[0]["labels"]
    assert "Defined <25%" in labels
    assert "Defined <50%" in labels
    assert "Defined >=50%" in labels
    assert legend_calls[0]["loc"] == "lower center"
    assert legend_calls[0]["ncol"] == 3


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

    assert _png_export_path(out_path).exists()
    assert {(1.30, 1.10), (1.05, 0.82), (0.96, 0.61)}.issubset(set(points))
    assert ltm_heights == sorted(ltm_heights, reverse=True)


def test_ccmr_ltm_comparison_uses_local_bar_legend_only(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    figure_legend_calls: list[dict] = []
    axes_legend_calls: list[dict] = []
    original_figure_legend = matplotlib.figure.Figure.legend
    original_axes_legend = matplotlib.axes.Axes.legend

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(dict(kwargs))
        return original_figure_legend(self, *args, **kwargs)

    def spy_axes_legend(self, *args, **kwargs):
        axes_legend_calls.append(dict(kwargs))
        return original_axes_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_axes_legend)

    plot_ccmr_ltm_comparison(
        rows=_sample_ccmr_ltm_rows(),
        out_path=tmp_path / "ccmr_ltm_comparison.png",
    )

    assert not figure_legend_calls
    assert len(axes_legend_calls) == 1
    legend_kwargs = axes_legend_calls[0]
    assert legend_kwargs.get("loc") in {"upper right", "upper left"}


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

    assert "*" in markers
    assert "X" not in markers
    assert 3 in star_x_values


def test_dense_k_sweeps_use_human_friendly_integer_ticks(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    xtick_calls: list[list[int]] = []
    original_set_xticks = matplotlib.axes.Axes.set_xticks

    def spy_set_xticks(self, ticks, *args, **kwargs):
        xtick_calls.append([int(v) for v in np.asarray(ticks, dtype=int).tolist()])
        return original_set_xticks(self, ticks, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", spy_set_xticks)

    plot_ri_k_sweep(rows=_sample_dense_k_rows(), out_path=tmp_path / "ri_dense.png")

    assert [1, 5, 10, 15, 20, 25] in xtick_calls


def test_plot_q_alpha_vs_ccmr_scatter_writes_png(tmp_path: Path) -> None:
    rows = [
        {"model": "Virchow2", "ccmr": 1.05, "ccmr_q_alpha": 0.85, "ccmr_alpha": 0.10},
        {"model": "UNI", "ccmr": 0.92, "ccmr_q_alpha": 0.72, "ccmr_alpha": 0.10},
    ]
    out_path = tmp_path / "q_alpha_vs_ccmr_scatter.png"
    plot_q_alpha_vs_ccmr_scatter(rows=rows, out_path=out_path)
    assert _png_export_path(out_path).exists()
    assert _png_export_path(out_path).stat().st_size > 0


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

    assert _png_export_path(out_path).exists()


def test_precomputed_model_aliases_reuse_family_colors() -> None:
    assert _color_for_model("PRISM") == _color_for_model("Virchow2")
    assert _color_for_model("TITAN") == _color_for_model("CONCHv1.5")


def test_plot_writes_matching_pdf_export(tmp_path: Path) -> None:
    out_path = tmp_path / "bio_vs_confounder_scatter.png"

    plot_bio_vs_confounder_scatter(rows=_sample_summary_rows(), out_path=out_path)

    assert _png_export_path(out_path).exists()
    pdf_path = _pdf_export_path(out_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_scatter_uses_figure_level_bottom_legend(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    figure_legend_calls: list[dict] = []
    axes_legend_calls: list[dict] = []
    original_figure_legend = matplotlib.figure.Figure.legend
    original_axes_legend = matplotlib.axes.Axes.legend

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(dict(kwargs))
        return original_figure_legend(self, *args, **kwargs)

    def spy_axes_legend(self, *args, **kwargs):
        axes_legend_calls.append(dict(kwargs))
        return original_axes_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_axes_legend)

    plot_bio_vs_confounder_scatter(
        rows=_sample_summary_rows(), out_path=tmp_path / "scatter.png"
    )

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) >= 2
    bbox = legend_kwargs.get("bbox_to_anchor")
    assert bbox is not None
    assert float(bbox[1]) >= 0.0


def test_multi_panel_plot_uses_single_figure_level_legend(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    figure_legend_calls: list[dict] = []
    axes_legend_calls: list[dict] = []
    original_figure_legend = matplotlib.figure.Figure.legend
    original_axes_legend = matplotlib.axes.Axes.legend

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(dict(kwargs))
        return original_figure_legend(self, *args, **kwargs)

    def spy_axes_legend(self, *args, **kwargs):
        axes_legend_calls.append(dict(kwargs))
        return original_axes_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_axes_legend)

    plot_ccmr_m_sweep_with_ltm(
        rows=_sample_ccmr_m_rows(), out_path=tmp_path / "ccmr_m_sweep.png"
    )

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) == 4


def test_ccmr_m_sweep_uses_human_friendly_m_ticks(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    xtick_calls: list[list[int]] = []
    original_set_xticks = matplotlib.axes.Axes.set_xticks

    def spy_set_xticks(self, ticks, *args, **kwargs):
        xtick_calls.append([int(v) for v in np.asarray(ticks, dtype=int).tolist()])
        return original_set_xticks(self, ticks, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", spy_set_xticks)

    plot_ccmr_m_sweep_with_ltm(
        rows=_sample_ccmr_m_rows() + [
            {"model": "Virchow2", "m": 20, "ccmr": 1.05, "ccmr_ltm_alpha": 0.82},
            {"model": "UNI", "m": 20, "ccmr": 1.00, "ccmr_ltm_alpha": 0.78},
        ],
        out_path=tmp_path / "ccmr_m_sweep.png",
    )

    assert [1, 5, 10, 15, 20] in xtick_calls


def test_ccmr_distribution_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    rows = _sample_ccmr_distribution_rows(tmp_path)
    out_path = tmp_path / "ccmr_sample_distributions.png"

    plot_ccmr_sample_distributions(rows=rows, out_path=out_path)

    assert _png_export_path(out_path).exists()
    pdf_path = _pdf_export_path(out_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_ccmr_distribution_plot_uses_no_legend_and_ranks_rows(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    figure_legend_calls: list[dict] = []
    axes_legend_calls: list[dict] = []
    model_label_order: list[str] = []
    original_figure_legend = matplotlib.figure.Figure.legend
    original_axes_legend = matplotlib.axes.Axes.legend
    original_text = matplotlib.axes.Axes.text

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(dict(kwargs))
        return original_figure_legend(self, *args, **kwargs)

    def spy_axes_legend(self, *args, **kwargs):
        axes_legend_calls.append(dict(kwargs))
        return original_axes_legend(self, *args, **kwargs)

    def spy_text(self, x, y, s, *args, **kwargs):
        text = str(s)
        if text in {"Virchow2", "CONCH", "UNI"}:
            model_label_order.append(text)
        return original_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_axes_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)

    plot_ccmr_sample_distributions(
        rows=_sample_ccmr_distribution_rows(tmp_path),
        out_path=tmp_path / "ccmr_sample_distributions.png",
    )

    assert not figure_legend_calls
    assert not axes_legend_calls
    assert model_label_order == ["Virchow2", "CONCH", "UNI"]


def test_ccmr_distribution_plot_emits_summary_annotations(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.text

    annotation_texts: list[str] = []
    titles: list[str] = []
    figure_texts: list[str] = []
    title_y_positions: list[float] = []
    subtitle_y_positions: list[float] = []
    original_text = matplotlib.axes.Axes.text
    original_suptitle = matplotlib.figure.Figure.suptitle
    original_figure_text = matplotlib.figure.Figure.text

    def spy_text(self, x, y, s, *args, **kwargs):
        annotation_texts.append(str(s))
        return original_text(self, x, y, s, *args, **kwargs)

    def spy_suptitle(self, t, *args, **kwargs):
        titles.append(str(t))
        title_y_positions.append(float(kwargs.get("y", 0.98)))
        return original_suptitle(self, t, *args, **kwargs)

    def spy_figure_text(self, x, y, s, *args, **kwargs):
        figure_texts.append(str(s))
        if "sorted by CCMR" in str(s):
            subtitle_y_positions.append(float(y))
        return original_figure_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)
    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", spy_suptitle)
    monkeypatch.setattr(matplotlib.figure.Figure, "text", spy_figure_text)

    plot_ccmr_sample_distributions(
        rows=_sample_ccmr_distribution_rows(tmp_path),
        out_path=tmp_path / "ccmr_sample_distributions.png",
    )

    joined = "\n".join(annotation_texts)
    assert "CCMR" in joined
    assert "Q10" in joined
    assert "%<1" in joined
    assert "1.550" in joined
    assert "0.820" in joined
    assert "25.0%" in joined
    assert "0.970" in joined
    assert "75.0%" in joined
    assert "Per-sample CCMR distributions" in titles
    assert any("sorted by CCMR" in text for text in figure_texts)
    assert len(title_y_positions) == 1
    assert len(subtitle_y_positions) == 1
    assert title_y_positions[0] - subtitle_y_positions[0] >= 0.045
