import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plotting import (
    CROMA_DOMAIN,
    _color_for_model,
    _draw_croma_sample_distributions,
    _draw_trend_line,
    _load_croma_sample_rows,
    _pareto_frontier_max_max,
    _pdf_export_path,
    _png_export_path,
    _ridgeline_bands,
    _ridgeline_figure_height,
    _support_plot_rows,
    plot_bio_vs_confounder_scatter,
    plot_croma_ltm_bars,
    plot_croma_ltm_scatter,
    plot_croma_pareto,
    plot_croma_m_sweep,
    plot_croma_sample_distributions,
    plot_knn_confounder_k_sweep,
    plot_mari_k_sweep,
    plot_q_alpha_vs_croma_scatter,
    plot_rank_pareto,
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
            "k": 3,
            "ri_undefined_frac": 0.12,
            "mari_undefined_frac": 0.12,
        },
        {
            "model": "UNI",
            "k": 6,
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


def _sample_croma_m_rows() -> list[dict]:
    return [
        {"model": "Virchow2", "m": 1, "croma": 1.40, "croma_ltm_alpha": 1.05},
        {"model": "Virchow2", "m": 2, "croma": 1.22, "croma_ltm_alpha": 0.95},
        {"model": "Virchow2", "m": 3, "croma": 1.10, "croma_ltm_alpha": 0.88},
        {"model": "UNI", "m": 1, "croma": 0.92, "croma_ltm_alpha": 0.70},
        {"model": "UNI", "m": 2, "croma": 0.97, "croma_ltm_alpha": 0.75},
        {"model": "UNI", "m": 3, "croma": 1.03, "croma_ltm_alpha": 0.80},
        {"model": "CONCH", "m": 1, "croma": 1.15, "croma_ltm_alpha": 0.90},
        {"model": "CONCH", "m": 2, "croma": 1.08, "croma_ltm_alpha": 0.85},
        {"model": "CONCH", "m": 3, "croma": 1.02, "croma_ltm_alpha": 0.78},
        {"model": "Phikon", "m": 1, "croma": 0.88, "croma_ltm_alpha": 0.65},
        {"model": "Phikon", "m": 2, "croma": 0.93, "croma_ltm_alpha": 0.70},
        {"model": "Phikon", "m": 3, "croma": 0.99, "croma_ltm_alpha": 0.74},
    ]


def _sample_croma_ltm_rows() -> list[dict]:
    return [
        {"model": "Virchow2", "croma": 1.30, "croma_ltm_alpha": 1.10, "croma_alpha": 0.10},
        {"model": "UNI", "croma": 1.05, "croma_ltm_alpha": 0.82, "croma_alpha": 0.10},
        {"model": "CONCH", "croma": 0.96, "croma_ltm_alpha": 0.61, "croma_alpha": 0.10},
    ]


def _sample_croma_distribution_rows(tmp_path: Path) -> list[dict]:
    # CRoMa is the signed margin in (-1, 1); per-sample arrays straddle 0, and each
    # row's pooled ``croma`` is the median of its own samples so the ranking the plot
    # sorts on is the one the arrays imply (Virchow2 > CONCH > UNI).
    by_model = {
        "Virchow2": np.asarray([-0.10, 0.20, 0.42, 0.55], dtype=float),
        "UNI": np.asarray([-0.30, -0.15, -0.05, 0.20], dtype=float),
        "CONCH": np.asarray([0.02, 0.18, 0.28, 0.35], dtype=float),
    }
    rows: list[dict] = []
    for model, values in by_model.items():
        path = tmp_path / f"{model}.croma.npy"
        np.save(path, values)
        rows.append(
            {
                "model": model,
                "croma": float(np.median(values)),
                "croma_q_alpha": float(np.percentile(values, 10)),
                "croma_alpha": 0.10,
                "croma_samples_path": str(path),
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
            plot_croma_m_sweep,
            {"rows": _sample_croma_m_rows()},
            "croma_m_sweep.png",
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


def test_plot_writes_matching_pdf_export(tmp_path: Path) -> None:
    out_path = tmp_path / "bio_vs_confounder_scatter.png"

    plot_bio_vs_confounder_scatter(rows=_sample_summary_rows(), out_path=out_path)

    assert _png_export_path(out_path).exists()
    pdf_path = _pdf_export_path(out_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_precomputed_model_aliases_reuse_family_colors() -> None:
    assert _color_for_model("PRISM") == _color_for_model("Virchow")
    assert _color_for_model("TITAN") == _color_for_model("CONCHv1.5")


def test_scatter_uses_figure_level_bottom_legend(monkeypatch, tmp_path: Path) -> None:
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

    plot_bio_vs_confounder_scatter(rows=_sample_summary_rows(), out_path=tmp_path / "scatter.png")

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) >= 2
    bbox = legend_kwargs.get("bbox_to_anchor")
    assert bbox is not None
    assert float(bbox[1]) >= 0.0


def test_multi_panel_plot_uses_single_figure_level_legend(monkeypatch, tmp_path: Path) -> None:
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

    plot_croma_m_sweep(rows=_sample_croma_m_rows(), out_path=tmp_path / "croma_m_sweep.png")

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) == 6


def test_support_plot_rows_use_one_row_per_model_defined_share_thresholds_and_worst_first_order() -> (
    None
):
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
    assert indexed["Virchow2"]["kstar"] == 3
    assert indexed["UNI"]["kstar"] == 6
    assert indexed["CONCH"]["kstar"] is None  # no k provided -> annotation omitted


def test_support_plot_uses_single_colour_without_threshold_legend(
    monkeypatch, tmp_path: Path
) -> None:
    """The support plot uses a single neutral colour and no status legend.

    Severity is conveyed by bar length, the inline percentage labels, and the
    worst-first ordering, so the green/amber/red threshold legend was removed.
    """
    import matplotlib.figure

    legend_labels: list[str] = []
    original_legend = matplotlib.figure.Figure.legend

    def spy_legend(self, *args, **kwargs):
        handles = list(args[0]) if args else list(kwargs.get("handles", []))
        legend_labels.extend(str(getattr(handle, "get_label", lambda: "")()) for handle in handles)
        return original_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_legend)

    out_path = tmp_path / "ri_mari_support.png"
    plot_ri_mari_support(rows=_sample_support_rows(), out_path=out_path)

    assert _png_export_path(out_path).exists()
    # No threshold legend is emitted any more.
    assert "Defined <25%" not in legend_labels
    assert "Defined <50%" not in legend_labels
    assert "Defined >=50%" not in legend_labels


def test_plot_croma_ltm_scatter_filters_invalid_rows_and_uses_threshold_line(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    points: list[tuple[float, float]] = []
    hlines: list[float] = []
    original_scatter = matplotlib.axes.Axes.scatter
    original_axhline = matplotlib.axes.Axes.axhline

    def spy_scatter(self, x, y, *args, **kwargs):
        xx = np.asarray(x, dtype=float)
        yy = np.asarray(y, dtype=float)
        if xx.size == 1 and yy.size == 1:
            points.append((float(xx[0]), float(yy[0])))
        return original_scatter(self, x, y, *args, **kwargs)

    def spy_axhline(self, y=0, *args, **kwargs):
        hlines.append(float(y))
        return original_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", spy_axhline)

    rows = _sample_croma_ltm_rows() + [
        {"model": "Bad", "croma": float("nan"), "croma_ltm_alpha": 0.5, "croma_alpha": 0.1}
    ]
    out_path = tmp_path / "croma_ltm_scatter.png"
    plot_croma_ltm_scatter(rows=rows, out_path=out_path)

    assert _png_export_path(out_path).exists()
    # The NaN-CRoMa "Bad" row is filtered; the three valid points are plotted.
    assert {(1.30, 1.10), (1.05, 0.82), (0.96, 0.61)} == set(points)
    # A horizontal CRoMa=0 robustness threshold is drawn (not a y=x diagonal).
    assert hlines == [0.0]


def test_pareto_frontier_keeps_only_the_undominated_under_max_max() -> None:
    # A dominates C on both axes; B trades a lower median for a milder tail, so A and B are
    # both non-dominated. D ties A's x but has a worse y, so it is dominated off the tie.
    points = [
        ("A", 0.20, -0.10),
        ("B", 0.15, -0.05),
        ("C", 0.10, -0.30),
        ("D", 0.20, -0.25),
    ]
    # Returned x-ascending: B (milder tail, lower median) then A (higher median, deeper tail).
    assert _pareto_frontier_max_max(points) == ["B", "A"]


def test_pareto_frontier_collapses_to_one_when_a_point_wins_both_axes() -> None:
    points = [("win", 0.30, -0.05), ("lose1", 0.10, -0.20), ("lose2", 0.29, -0.06)]
    assert _pareto_frontier_max_max(points) == ["win"]


def test_plot_croma_pareto_rings_labels_the_frontier_and_marks_the_exposed(
    tmp_path: Path,
) -> None:
    """Every panel rings the frontier, labels every point directly -- bold for frontier
    members, muted for the dominated -- and flags exposure with a dagger in the point label.
    There is no legend: 25 swatches were unmatchable to 25 dots. No contrast callout, no
    direction cue."""
    import matplotlib.axes
    import matplotlib.figure

    ring_calls: list[int] = []
    scatter_labels: list[str] = []
    annotate_texts: list[str] = []
    figure_legend_calls: list[dict] = []
    original_scatter = matplotlib.axes.Axes.scatter
    original_annotate = matplotlib.axes.Axes.annotate
    original_figure_legend = matplotlib.figure.Figure.legend

    def spy_scatter(self, x, y, *args, **kwargs):
        # The frontier halo is the one scatter drawn with hollow markers; capture how many
        # points it rings so the test pins the frontier size the drawer computed. Exposure is not
        # drawn on the point -- it is a dagger appended to the exposed models' legend labels, so
        # capture every per-model scatter label and count the daggered ones.
        if kwargs.get("facecolors") == "none":
            ring_calls.append(int(np.asarray(x, dtype=float).size))
        label = kwargs.get("label")
        if isinstance(label, str):
            scatter_labels.append(label)
        return original_scatter(self, x, y, *args, **kwargs)

    def spy_annotate(self, text, *args, **kwargs):
        annotate_texts.append(text)
        return original_annotate(self, text, *args, **kwargs)

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(kwargs)
        return original_figure_legend(self, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", spy_annotate)
    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    try:
        # Virchow2 (highest median) and GenBio-PathFM (mildest tail) are the two undominated
        # encoders; CONCH ties Virchow2's median with a deeper tail, so it is dominated, and
        # Phikon is dominated outright. GenBio-PathFM is exposed, so its point label carries
        # the dagger; Virchow2 is a labelled frontier member with no dagger.
        rows = [
            {"model": "Virchow2", "croma": 0.20, "croma_ltm_alpha": -0.11, "croma_alpha": 0.1},
            {"model": "GenBio-PathFM", "croma": 0.19, "croma_ltm_alpha": -0.07, "croma_alpha": 0.1},
            {"model": "CONCH", "croma": 0.20, "croma_ltm_alpha": -0.20, "croma_alpha": 0.1},
            {"model": "Phikon", "croma": -0.20, "croma_ltm_alpha": -0.50, "croma_alpha": 0.1},
        ]
        out_path = tmp_path / "croma_pareto.png"
        plot_croma_pareto(rows=rows, out_path=out_path, exposed={"GenBio-PathFM"})
    finally:
        monkeypatch.undo()

    assert _png_export_path(out_path).exists()
    assert _pdf_export_path(out_path).exists()
    # Exactly the two undominated encoders are ringed; CONCH (median tie, deeper tail) is not.
    assert ring_calls == [2]
    # Every point is labelled: the frontier in bold, the dominated in muted text, and the
    # single exposed encoder carries the dagger in its point label.
    assert any(t == "Virchow2" for t in annotate_texts)
    assert any(t == r"GenBio-PathFM $\dagger$" for t in annotate_texts)
    assert any(t == "CONCH" for t in annotate_texts)
    assert any(t == "Phikon" for t in annotate_texts)
    # No legend, and no daggered legend entries: identity lives on the points now.
    assert figure_legend_calls == []
    assert not any(r"$\dagger$" in l for l in scatter_labels)
    assert not any(t == r"$\dagger$" for t in annotate_texts)
    assert not any("dominated" in t for t in annotate_texts)
    assert not any("more robust" in t for t in annotate_texts)


def test_plot_croma_pareto_labels_every_frontier_member_and_marks_off_frontier(
    tmp_path: Path,
) -> None:
    """What the supplementary panels used to suppress is now always drawn: a larger, clustered
    frontier still gets a bold label on every ringed member, and an exposed encoder carries its
    point-label dagger whether or not it sits on the frontier, so the ring count and the
    exposure count are independent."""
    import matplotlib.axes

    ring_calls: list[int] = []
    scatter_labels: list[str] = []
    annotate_texts: list[str] = []
    original_scatter = matplotlib.axes.Axes.scatter
    original_annotate = matplotlib.axes.Axes.annotate

    def spy_scatter(self, x, y, *args, **kwargs):
        if kwargs.get("facecolors") == "none":
            ring_calls.append(int(np.asarray(x, dtype=float).size))
        label = kwargs.get("label")
        if isinstance(label, str):
            scatter_labels.append(label)
        return original_scatter(self, x, y, *args, **kwargs)

    def spy_annotate(self, text, *args, **kwargs):
        annotate_texts.append(text)
        return original_annotate(self, text, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", spy_annotate)
    try:
        # Midnight-12k (highest median), CONCHv1.5 and H-optimus-1 (mildest tail) form the
        # frontier; Phikon is dominated. Midnight-12k (frontier) and Phikon (dominated) are
        # exposed, so two legend daggers are drawn while three points are ringed.
        rows = [
            {"model": "Midnight-12k", "croma": 0.40, "croma_ltm_alpha": -0.21, "croma_alpha": 0.1},
            {"model": "H-optimus-1", "croma": 0.09, "croma_ltm_alpha": -0.10, "croma_alpha": 0.1},
            {"model": "CONCHv1.5", "croma": 0.15, "croma_ltm_alpha": -0.13, "croma_alpha": 0.1},
            {"model": "Phikon", "croma": 0.02, "croma_ltm_alpha": -0.20, "croma_alpha": 0.1},
        ]
        out_path = tmp_path / "croma_pareto_supp.png"
        plot_croma_pareto(rows=rows, out_path=out_path, exposed={"Midnight-12k", "Phikon"})
    finally:
        monkeypatch.undo()

    assert _png_export_path(out_path).exists()
    # The three-member frontier is ringed (Midnight, CONCHv1.5, H-optimus-1)...
    assert ring_calls == [3]
    # ...and every ringed member is labelled, with the exposed one daggered in its label.
    for name in (r"Midnight-12k $\dagger$", "CONCHv1.5", "H-optimus-1"):
        assert any(t == name for t in annotate_texts)
    # The exposed *dominated* encoder is daggered too: exposure and frontier are independent.
    assert any(t == r"Phikon $\dagger$" for t in annotate_texts)
    assert not any(r"$\dagger$" in l for l in scatter_labels)
    assert not any(t == r"$\dagger$" for t in annotate_texts)
    assert not any("more robust" in t for t in annotate_texts)


def test_plot_rank_pareto_rings_labels_the_frontier_and_marks_the_exposed(
    tmp_path: Path,
) -> None:
    """The mean-rank overview rings the min-min frontier, labels every point directly (as the
    per-benchmark panels do), and daggers every TCGA-exposed encoder in its point label. One
    exposed encoder is on the frontier, one is dominated, so the ring count and the exposure
    count are genuinely independent."""
    import matplotlib.axes

    ring_calls: list[int] = []
    scatter_labels: list[str] = []
    annotate_texts: list[str] = []
    original_scatter = matplotlib.axes.Axes.scatter
    original_annotate = matplotlib.axes.Axes.annotate

    def spy_scatter(self, x, y, *args, **kwargs):
        if kwargs.get("facecolors") == "none":
            ring_calls.append(int(np.asarray(x, dtype=float).size))
        label = kwargs.get("label")
        if isinstance(label, str):
            scatter_labels.append(label)
        return original_scatter(self, x, y, *args, **kwargs)

    def spy_annotate(self, text, *args, **kwargs):
        annotate_texts.append(text)
        return original_annotate(self, text, *args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", spy_scatter)
    monkeypatch.setattr(matplotlib.axes.Axes, "annotate", spy_annotate)
    try:
        # CONCH (best median), CONCHv1.5, GenBio-PathFM and H-optimus-1 (best tail) form the
        # frontier; Midnight-12k (good median, deep tail) is dominated. GenBio-PathFM is an
        # exposed frontier member, Midnight-12k an exposed dominated one, so the ring count (4)
        # and the exposure count (2) are genuinely independent, overlapping only at GenBio-PathFM.
        rows = [
            {"model": "CONCH", "median_rank": 2.5, "tail_rank": 7.0, "exposed": False},
            {"model": "CONCHv1.5", "median_rank": 3.5, "tail_rank": 4.0, "exposed": False},
            {"model": "H-optimus-1", "median_rank": 8.0, "tail_rank": 3.0, "exposed": False},
            {"model": "GenBio-PathFM", "median_rank": 3.0, "tail_rank": 6.0, "exposed": True},
            {"model": "Midnight-12k", "median_rank": 3.0, "tail_rank": 15.0, "exposed": True},
        ]
        out_path = tmp_path / "rank_pareto.png"
        plot_rank_pareto(rows=rows, out_path=out_path, n_benchmarks=3)
    finally:
        monkeypatch.undo()

    assert _png_export_path(out_path).exists()
    assert _pdf_export_path(out_path).exists()
    # The four undominated encoders are ringed; Midnight-12k (good median, deep tail) is not.
    assert ring_calls == [4]
    # ...and every point is labelled, with the two exposed encoders daggered in their labels --
    # the frontier one and the dominated one alike.
    for name in ("CONCH", "CONCHv1.5", "H-optimus-1", r"GenBio-PathFM $\dagger$"):
        assert any(t == name for t in annotate_texts)
    assert any(t == r"Midnight-12k $\dagger$" for t in annotate_texts)
    assert not any(r"$\dagger$" in l for l in scatter_labels)
    assert not any(t == r"$\dagger$" for t in annotate_texts)
    assert not any("more robust" in t for t in annotate_texts)


def test_plot_croma_ltm_bars_sorts_descending_with_threshold_and_local_legend(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.axes

    ltm_heights: list[float] = []
    hlines: list[float] = []
    figure_legend_calls: list[dict] = []
    axes_legend_calls: list[dict] = []
    original_bar = matplotlib.axes.Axes.bar
    original_axhline = matplotlib.axes.Axes.axhline
    original_figure_legend = matplotlib.figure.Figure.legend
    original_axes_legend = matplotlib.axes.Axes.legend

    def spy_bar(self, x, height, *args, **kwargs):
        if "LTM@" in str(kwargs.get("label", "")):
            ltm_heights.extend(float(v) for v in np.asarray(height, dtype=float).tolist())
        return original_bar(self, x, height, *args, **kwargs)

    def spy_axhline(self, y=0, *args, **kwargs):
        hlines.append(float(y))
        return original_axhline(self, y, *args, **kwargs)

    def spy_figure_legend(self, *args, **kwargs):
        figure_legend_calls.append(dict(kwargs))
        return original_figure_legend(self, *args, **kwargs)

    def spy_axes_legend(self, *args, **kwargs):
        axes_legend_calls.append(dict(kwargs))
        return original_axes_legend(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", spy_bar)
    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", spy_axhline)
    monkeypatch.setattr(matplotlib.figure.Figure, "legend", spy_figure_legend)
    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy_axes_legend)

    out_path = tmp_path / "croma_ltm_bars.png"
    plot_croma_ltm_bars(rows=_sample_croma_ltm_rows(), out_path=out_path)

    assert _png_export_path(out_path).exists()
    assert ltm_heights == sorted(ltm_heights, reverse=True)
    assert hlines == [0.0]
    # A single local (axes-level) legend, no figure-level legend.
    assert not figure_legend_calls
    assert len(axes_legend_calls) == 1
    assert axes_legend_calls[0].get("loc") in {"upper right", "upper left"}


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


def test_dense_k_sweeps_use_human_friendly_integer_ticks(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes

    xtick_calls: list[list[int]] = []
    original_set_xticks = matplotlib.axes.Axes.set_xticks

    def spy_set_xticks(self, ticks, *args, **kwargs):
        xtick_calls.append([int(v) for v in np.asarray(ticks, dtype=int).tolist()])
        return original_set_xticks(self, ticks, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", spy_set_xticks)

    plot_ri_k_sweep(rows=_sample_dense_k_rows(), out_path=tmp_path / "ri_dense.png")

    assert [1, 5, 10, 15, 20, 25] in xtick_calls


def test_plot_q_alpha_vs_croma_scatter_writes_png(tmp_path: Path) -> None:
    rows = [
        {"model": "Virchow2", "croma": 1.05, "croma_q_alpha": 0.85, "croma_alpha": 0.10},
        {"model": "UNI", "croma": 0.92, "croma_q_alpha": 0.72, "croma_alpha": 0.10},
    ]
    out_path = tmp_path / "q_alpha_vs_croma_scatter.png"
    plot_q_alpha_vs_croma_scatter(rows=rows, out_path=out_path)
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


def test_scatter_uses_figure_level_bottom_legend(monkeypatch, tmp_path: Path) -> None:
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

    plot_bio_vs_confounder_scatter(rows=_sample_summary_rows(), out_path=tmp_path / "scatter.png")

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) >= 2
    bbox = legend_kwargs.get("bbox_to_anchor")
    assert bbox is not None
    assert float(bbox[1]) >= 0.0


def test_multi_panel_plot_uses_single_figure_level_legend(monkeypatch, tmp_path: Path) -> None:
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

    plot_croma_m_sweep(rows=_sample_croma_m_rows(), out_path=tmp_path / "croma_m_sweep.png")

    assert not axes_legend_calls
    assert len(figure_legend_calls) == 1
    legend_kwargs = figure_legend_calls[0]
    assert legend_kwargs.get("loc") == "lower center"
    assert int(legend_kwargs.get("ncol", 0)) == 6


def test_croma_m_sweep_uses_human_friendly_m_ticks(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.axes

    xtick_calls: list[list[int]] = []
    original_set_xticks = matplotlib.axes.Axes.set_xticks

    def spy_set_xticks(self, ticks, *args, **kwargs):
        xtick_calls.append([int(v) for v in np.asarray(ticks, dtype=int).tolist()])
        return original_set_xticks(self, ticks, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xticks", spy_set_xticks)

    plot_croma_m_sweep(
        rows=_sample_croma_m_rows()
        + [
            {"model": "Virchow2", "m": 20, "croma": 1.05, "croma_ltm_alpha": 0.82},
            {"model": "UNI", "m": 20, "croma": 1.00, "croma_ltm_alpha": 0.78},
        ],
        out_path=tmp_path / "croma_m_sweep.png",
    )

    assert [1, 5, 10, 15, 20] in xtick_calls


def test_croma_distribution_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    rows = _sample_croma_distribution_rows(tmp_path)
    out_path = tmp_path / "croma_sample_distributions.png"

    plot_croma_sample_distributions(rows=rows, out_path=out_path)

    assert _png_export_path(out_path).exists()
    pdf_path = _pdf_export_path(out_path)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_croma_distribution_plot_uses_no_legend_and_ranks_rows(monkeypatch, tmp_path: Path) -> None:
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

    plot_croma_sample_distributions(
        rows=_sample_croma_distribution_rows(tmp_path),
        out_path=tmp_path / "croma_sample_distributions.png",
    )

    assert not figure_legend_calls
    assert not axes_legend_calls
    assert model_label_order == ["Virchow2", "CONCH", "UNI"]


def test_croma_distribution_plot_emits_summary_annotations(monkeypatch, tmp_path: Path) -> None:
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
        if "sorted by CRoMa" in str(s):
            subtitle_y_positions.append(float(y))
        return original_figure_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", spy_text)
    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", spy_suptitle)
    monkeypatch.setattr(matplotlib.figure.Figure, "text", spy_figure_text)

    plot_croma_sample_distributions(
        rows=_sample_croma_distribution_rows(tmp_path),
        out_path=tmp_path / "croma_sample_distributions.png",
    )

    # The pooled CRoMa / Q10 / %<0 side panel was removed: CRoMa and F(0) are table
    # columns, and the numbers only added clutter beside the ridgelines.
    joined = "\n".join(annotation_texts)
    assert "Q10" not in joined
    assert "%<0" not in joined
    assert not any(text.endswith("%") for text in annotation_texts)
    # Only the model labels are drawn onto the axes.
    assert set(annotation_texts) == {"Virchow2", "CONCH", "UNI"}
    assert "Per-sample CRoMa distributions" in titles
    assert any("sorted by CRoMa" in text for text in figure_texts)
    assert len(title_y_positions) == 1
    assert len(subtitle_y_positions) == 1
    assert title_y_positions[0] - subtitle_y_positions[0] >= 0.045


def test_croma_distribution_plot_frames_the_full_margin_domain(tmp_path: Path) -> None:
    """The x-axis spans CRoMa's whole codomain, not the sampled range.

    The samples here reach only [-0.30, 0.55], but the frame stays at (-1, 1) so the
    zero-margin boundary sits at the centre and the figure is comparable across
    benchmarks. Data-driven limits would also clip the KDE tails mid-slope.
    """
    import matplotlib.pyplot as plt

    model_data = _load_croma_sample_rows(_sample_croma_distribution_rows(tmp_path))
    fig, ax = plt.subplots()
    try:
        _draw_croma_sample_distributions(ax, model_data)
        assert CROMA_DOMAIN == (-1.0, 1.0)
        assert ax.get_xlim() == CROMA_DOMAIN
        assert [d["model"] for d in model_data] == ["Virchow2", "CONCH", "UNI"]
    finally:
        plt.close(fig)


def test_croma_distribution_bands_are_physically_constant_across_rosters() -> None:
    """The title and x-label bands keep a fixed size in inches, not in figure fractions.

    The ridgeline grows one row per model, so the figure is 3.5in tall for the 4 slide
    encoders and ~10.3in for the 21 tile encoders. A fixed *fractional* bottom margin
    that clears the x-label on the tall figure leaves only ~0.26in on the short one,
    which clipped the "Per-sample CRoMa" label off the slide-level figures entirely.
    """
    short = _ridgeline_figure_height(4)
    tall = _ridgeline_figure_height(21)
    assert short == 3.5
    assert tall > 10.0

    for height in (short, tall):
        bands = _ridgeline_bands(height)
        # Reserved bands, converted back to inches, stay constant.
        assert bands["bottom"] * height == pytest.approx(0.52)
        assert (1.0 - bands["top"]) * height == pytest.approx(0.60)
        # Subtitle sits below the title and above the axes, at both heights.
        assert bands["title_y"] > bands["subtitle_y"] > bands["top"]


def test_croma_distribution_plot_renders_without_q_alpha(tmp_path: Path) -> None:
    """Rows lacking ``croma_q_alpha`` still render now that the Q10 markers are gone.

    The drawer used to require the key, so a run that never wrote it silently dropped
    the whole figure rather than the one annotation that needed it.
    """
    rows = _sample_croma_distribution_rows(tmp_path)
    for row in rows:
        row.pop("croma_q_alpha")
        row.pop("croma_alpha")
    out_path = tmp_path / "croma_sample_distributions.png"

    plot_croma_sample_distributions(rows=rows, out_path=out_path)

    assert _png_export_path(out_path).exists()
    assert _png_export_path(out_path).stat().st_size > 0


def _trend_line_segment(ax) -> tuple[np.ndarray, np.ndarray]:
    """The one dotted line the trend drawer adds (the only ':' line on a bare axes)."""
    dotted = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
    assert len(dotted) == 1
    return dotted[0].get_xdata(), dotted[0].get_ydata()


def test_trend_line_spans_only_the_data_range_and_reports_its_slope() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    xs = np.array([1.0, 2.0, 3.0, 10.0])
    slope = _draw_trend_line(ax, xs, 2.0 * xs + 1.0)
    xdata, ydata = _trend_line_segment(ax)
    plt.close(fig)

    assert slope == pytest.approx(2.0)
    # Drawn across the observed x-range only: never extrapolated past the outermost model.
    assert xdata == pytest.approx([1.0, 10.0])
    assert ydata == pytest.approx([3.0, 21.0])


def test_trend_line_slope_sign_tracks_a_negative_relationship() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    slope = _draw_trend_line(ax, [0.0, 1.0, 2.0, 3.0], [5.0, 3.0, 2.0, -1.0])
    plt.close(fig)
    assert slope is not None and slope < 0


def test_trend_line_ignores_non_finite_points() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    xs = [1.0, 2.0, 3.0, float("nan"), 4.0]
    ys = [2.0, 4.0, 6.0, 100.0, float("inf")]
    slope = _draw_trend_line(ax, xs, ys)
    xdata, _ = _trend_line_segment(ax)
    plt.close(fig)

    # The NaN-x and inf-y rows are dropped, so the fit is the exact y = 2x of the rest.
    assert slope == pytest.approx(2.0)
    assert xdata == pytest.approx([1.0, 3.0])


@pytest.mark.parametrize(
    "xs, ys",
    [
        ([1.0, 2.0], [1.0, 2.0]),  # two points make a line, not a trend
        ([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]),  # constant x has no slope to estimate
        ([], []),
    ],
)
def test_trend_line_draws_nothing_when_there_is_no_trend_to_fit(xs, ys) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    slope = _draw_trend_line(ax, xs, ys)
    dotted = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
    plt.close(fig)

    assert slope is None
    assert dotted == []
