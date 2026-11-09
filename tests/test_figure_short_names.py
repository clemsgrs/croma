"""Text drawn inside a figure prints a model's short published name.

Every published frame carries the full name ("Mettle-1 (Pre-release)"), which is what the
tables, captions and results/ print. Beside a point, on a ridgeline row or in a legend it
does not fit, so the figure call sites ask the style layer for the short form
("Mettle-1"). These tests render each figure and read back every string it drew.
"""

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts" / "bench",):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import layout  # noqa: E402
from plotting import (  # noqa: E402
    plot_bio_vs_confounder_scatter,
    plot_croma_pareto,
    plot_croma_sample_distributions,
    plot_knn_bio_k_sweep,
    plot_knn_confounder_k_sweep,
    plot_mari_k_sweep,
    plot_rank_pareto,
    plot_ri_k_sweep,
)

FULL = "Mettle-1 (Pre-release)"
SHORT = "Mettle-1"


@contextmanager
def drawn_text(monkeypatch):
    """Every string drawn on a figure, collected as the figure is closed."""
    texts: list[str] = []
    close = plt.close

    def capture(fig=None):
        if isinstance(fig, matplotlib.figure.Figure):
            texts.extend(t.get_text() for t in fig.findobj(Text))
        return close(fig)

    monkeypatch.setattr(plt, "close", capture)
    yield texts


def _assert_short(texts: list[str]) -> None:
    assert any(t.startswith(SHORT) for t in texts), texts
    assert not any("Pre-release" in t for t in texts), texts


def test_pareto_frontier_tags_and_point_labels_print_the_short_name(monkeypatch, tmp_path):
    # Mettle-1 holds the mildest tail (a bold frontier tag, daggered as exposed); UNI is
    # dominated, so the muted point labels are exercised too.
    rows = [
        {"model": "Virchow2", "croma": 0.20, "croma_ltm_alpha": -0.11, "bio": 0.95},
        {"model": FULL, "croma": 0.19, "croma_ltm_alpha": -0.02, "bio": 0.94},
        {"model": "UNI", "croma": 0.10, "croma_ltm_alpha": -0.30, "bio": 0.90},
    ]
    with drawn_text(monkeypatch) as texts:
        plot_croma_pareto(rows=rows, out_path=tmp_path / "pareto.png", exposed={FULL})

    _assert_short(texts)
    assert rf"{SHORT} $\dagger$" in texts


def test_a_dominated_point_on_the_rank_pareto_prints_the_short_name(monkeypatch, tmp_path):
    rows = [
        {"model": "CONCH", "median_rank": 1.0, "tail_rank": 1.0, "exposed": False},
        {"model": FULL, "median_rank": 5.0, "tail_rank": 4.0, "exposed": False},
        {"model": "UNI", "median_rank": 3.0, "tail_rank": 6.0, "exposed": False},
    ]
    rows = [{**row, "bio_rank": 2.0} for row in rows]
    with drawn_text(monkeypatch) as texts:
        plot_rank_pareto(rows=rows, out_path=tmp_path / "rank.png", n_benchmarks=3)

    _assert_short(texts)
    assert SHORT in texts


def test_the_shared_model_legend_prints_the_short_name(monkeypatch, tmp_path):
    rows = [
        {
            "model": model,
            "bio_knn_bacc": bio,
            "confounder_knn_bacc": confounder,
            "ri": 0.5,
            "mari": 0.5,
            "confounder_display_name": "Medical Center",
        }
        for model, bio, confounder in ((FULL, 0.72, 0.66), ("UNI", 0.58, 0.74))
    ]
    with drawn_text(monkeypatch) as texts:
        plot_bio_vs_confounder_scatter(rows=rows, out_path=tmp_path / "scatter.png")

    _assert_short(texts)
    assert SHORT in texts


#: Six encoders: the focused APD view samples six points of the CRoMa ranking.
APD_MODELS = ["Mettle", "UNI", "CONCH", "Virchow2", "GPFM", "Phikon"]


@pytest.fixture
def apd_study(monkeypatch, tmp_path):
    """A synthetic APD study tree the two APD figures read and render into.

    The study stores registry identities, as ``output/studies/apd`` does; the figures
    publish them. Every directory points into ``tmp_path``.
    """
    pd = pytest.importorskip("pandas")
    figures = ROOT / "scripts" / "repro" / "figures"
    if str(figures) not in sys.path:
        sys.path.insert(0, str(figures))
    import apd_figure
    import apd_per_v_figure

    raw = tmp_path / "apd"
    rows = []
    for dataset, _ in apd_per_v_figure.BENCHMARKS:
        n_splits = len(apd_per_v_figure.training_correlations(dataset))
        (raw / dataset).mkdir(parents=True)
        for i, model in enumerate(APD_MODELS):
            accuracies = np.linspace(0.95 - 0.01 * i, 0.80, n_splits)[:, None] + [0, 0.01, 0.02]
            payload = {f"{r}_test_accuracies": accuracies.tolist() for r in ("id", "ood")}
            (raw / dataset / f"{model}.json").write_text(json.dumps(payload))
            rows.append({"dataset": dataset, "model": model, "croma": 0.3 - 0.05 * i})
    frame = pd.DataFrame(rows)
    frame.to_csv(raw / "apd.csv", index=False)
    frame["nipd_id"] = frame["nipd_ood"] = (frame["croma"] - 0.4) / 10

    monkeypatch.setattr(apd_per_v_figure, "RAW", raw)
    monkeypatch.setattr(apd_per_v_figure, "OUT", tmp_path / "plots" / "per_v.png")
    monkeypatch.setattr(apd_figure, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(apd_figure, "read_joined", lambda: frame)
    for cached in ("_roster", "_accuracy_matrix", "_mean_and_ci", "_normalized_mean_and_ci"):
        getattr(apd_per_v_figure, cached).cache_clear()
    yield apd_figure, apd_per_v_figure
    for cached in ("_roster", "_accuracy_matrix", "_mean_and_ci", "_normalized_mean_and_ci"):
        getattr(apd_per_v_figure, cached).cache_clear()


@pytest.mark.parametrize("focused", [False, True], ids=["composite", "focused"])
def test_the_apd_composite_legend_prints_the_short_name(apd_study, monkeypatch, focused):
    apd_figure, _ = apd_study
    with drawn_text(monkeypatch) as texts:
        apd_figure.main(focused=focused)

    _assert_short(texts)
    assert SHORT in texts


def test_the_apd_per_correlation_legend_prints_the_short_name(apd_study, monkeypatch):
    _, apd_per_v_figure = apd_study
    with drawn_text(monkeypatch) as texts:
        apd_per_v_figure.main()

    _assert_short(texts)
    assert SHORT in texts


def test_scale_scatter_point_labels_print_the_short_name(monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")
    figures = ROOT / "scripts" / "repro" / "figures"
    if str(figures) not in sys.path:
        sys.path.insert(0, str(figures))
    import scale_scatter

    frame = pd.DataFrame(
        {
            "model": [FULL, "UNI"],
            "croma_mean": [0.2, 0.1],
            "regime": ["vision-only", "vision-only"],
            "params": [float("nan"), 3.07e8],
            "n_wsis": [1.0e5, 1.0e5],
            "n_datasets": [3, 3],
        }
    )
    with drawn_text(monkeypatch) as texts:
        scale_scatter.render_scale_scatter(frame, tmp_path / "scale.pdf")

    _assert_short(texts)
    assert SHORT in texts


def test_cross_benchmark_end_labels_print_the_short_name(monkeypatch, tmp_path):
    pd = pytest.importorskip("pandas")
    for path in (ROOT / "scripts" / "repro", ROOT / "scripts" / "repro" / "figures"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import cross_benchmark_figure
    from _cross_benchmark import CrossBenchmark

    croma = pd.DataFrame(
        {"Camelyon": [0.20, 0.10], "TCGA-4x4": [0.11, 0.15], "Tolkach-ESCA": [0.37, 0.30]},
        index=[FULL, "UNI"],
    )
    ranks = croma.rank(ascending=False, method="first").astype(int)
    bench = CrossBenchmark(croma=croma, ranks=ranks, exposed=frozenset({FULL}))
    monkeypatch.setattr(cross_benchmark_figure, "load", lambda: bench)
    monkeypatch.setattr(cross_benchmark_figure, "OUT", tmp_path / "cross_benchmark.pdf")
    with drawn_text(monkeypatch) as texts:
        cross_benchmark_figure.main()

    _assert_short(texts)
    assert rf"{SHORT}$^\dagger$  0.20" in texts


def test_ridgeline_rows_print_the_short_name(monkeypatch, tmp_path):
    # The row carries the published name; the run stored the array under the registry name.
    rows = []
    for model, registry, values in [
        (FULL, "Mettle", [-0.10, 0.20, 0.42, 0.55]),
        ("UNI", "UNI", [-0.30, -0.15, -0.05, 0.20]),
    ]:
        path = layout.sample_distribution_path(tmp_path, "croma", registry)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(values))
        rows.append(
            {
                "model": model,
                "croma": float(np.median(values)),
                "croma_q_alpha": float(np.percentile(values, 10)),
                "croma_alpha": 0.10,
                "croma_samples_path": str(path),
            }
        )
    with drawn_text(monkeypatch) as texts:
        plot_croma_sample_distributions(
            rows=rows, out_path=tmp_path / "ridge.png", results_dir=tmp_path
        )

    _assert_short(texts)
    assert SHORT in texts


@pytest.mark.parametrize(
    "plot",
    [plot_knn_bio_k_sweep, plot_knn_confounder_k_sweep, plot_ri_k_sweep, plot_mari_k_sweep],
    ids=["bio", "confounder", "ri", "mari"],
)
def test_the_k_sweep_legend_prints_the_short_name(plot, monkeypatch, tmp_path):
    # A run directory's metrics rows carry the registry identity ("Mettle"), as the
    # benchmark writes them; the legend publishes it.
    rows = [
        {
            "model": model,
            "k": k,
            "selected_k": 11,
            "knn_bacc": 0.9 - 0.001 * k,
            "knn_confounder_bacc": 0.6,
            "ri": 0.5,
            "mari": 0.5,
            "confounder_display_name": "Medical Center",
        }
        for model in ("Mettle", "UNI")
        for k in (1, 11, 21)
    ]
    with drawn_text(monkeypatch) as texts:
        plot(rows=rows, out_path=tmp_path / "k_sweep.png")

    _assert_short(texts)
    assert SHORT in texts
