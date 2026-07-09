"""Render a benchmark run's figure set from its written metrics artifacts.

benchmark.py is the compute-only driver: it writes the metrics JSON/CSV, the k-sweep
and CRoMa m-sweep rows, and the per-sample sample-distribution ``.npy`` files into a run
directory, but it imports no matplotlib and makes no plot calls. This module is the single
definition of *which* plots a run emits and *how* they are sequenced. Given a completed
run directory it loads the written artifacts and emits the full figure set into
``<run-dir>/plots/{png,pdf}/`` (each plot function writes both a PNG and a PDF).

The figure sequence here is the exact sequence that used to be inlined at the end of
benchmark.py's ``main``; the two conditional groups (cumulative-mean k-sweeps and RI/MaRI
sample distributions) are gated on the flags benchmark.py records in
``results/render_manifest.json``.

Usage:
    python scripts/bench/render.py <run-dir>
    # e.g. python scripts/bench/render.py output/metrics/median-k/camelyon
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plotting  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    """Load a metrics JSON file written by benchmark.save_metrics (a list of row dicts)."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return json.loads(text)


def _load_render_flags(results_dir: Path) -> dict[str, bool]:
    """Read the render flags benchmark.py persisted; default to the base figure set.

    Runs produced before render_manifest.json existed simply render the unconditional
    figure set (both flags default False), matching a plain ``benchmark.py`` invocation.
    """
    manifest_path = results_dir / "render_manifest.json"
    flags = {"summarize_by_mean": False, "prune_ss_oo": False}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in flags:
            flags[key] = bool(payload.get(key, flags[key]))
    return flags


def render_figure_set(
    *,
    rows: list[dict],
    k_sweep_rows: list[dict],
    croma_m_sweep_rows: list[dict],
    plots_dir: Path,
    summarize_by_mean: bool = False,
    prune_ss_oo: bool = False,
) -> list[Path]:
    """Emit the full benchmark figure set into ``plots_dir``.

    This is the single, canonical plot sequence -- byte-for-byte the ordering that
    benchmark.py used to inline. Returns the list of nominal ``.png`` out_paths that were
    requested (each also produces a sibling ``pdf/<name>.pdf`` and ``png/<name>.png``).
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []

    def _render(fn, *, out_name: str, **kwargs) -> None:
        out_path = plots_dir / out_name
        fn(out_path=out_path, **kwargs)
        rendered.append(out_path)

    _render(
        plotting.plot_knn_bio_k_sweep,
        rows=k_sweep_rows,
        out_name="knn_bio_k_sweep.png",
    )
    _render(
        plotting.plot_knn_confounder_k_sweep,
        rows=k_sweep_rows,
        out_name="knn_confounder_k_sweep.png",
    )
    _render(plotting.plot_ri_k_sweep, rows=k_sweep_rows, out_name="ri_k_sweep.png")
    _render(plotting.plot_mari_k_sweep, rows=k_sweep_rows, out_name="mari_k_sweep.png")
    if summarize_by_mean:
        _render(
            plotting.plot_ri_cumulative_mean_k_sweep,
            rows=k_sweep_rows,
            out_name="ri_cumulative_mean_k_sweep.png",
        )
        _render(
            plotting.plot_mari_cumulative_mean_k_sweep,
            rows=k_sweep_rows,
            out_name="mari_cumulative_mean_k_sweep.png",
        )
    _render(
        plotting.plot_croma_m_sweep,
        rows=croma_m_sweep_rows,
        out_name="croma_m_sweep.png",
    )
    _render(plotting.plot_croma_ltm_scatter, rows=rows, out_name="croma_ltm_scatter.png")
    _render(plotting.plot_croma_ltm_bars, rows=rows, out_name="croma_ltm_bars.png")
    _render(
        plotting.plot_bio_vs_confounder_scatter,
        rows=rows,
        out_name="bio_vs_confounder_scatter.png",
    )
    _render(plotting.plot_mari_vs_ri_scatter, rows=rows, out_name="mari_vs_ri_scatter.png")
    _render(plotting.plot_ri_mari_support, rows=rows, out_name="ri_mari_support.png")
    _render(
        plotting.plot_croma_vs_mari_scatter,
        rows=rows,
        out_name="croma_vs_mari_scatter.png",
    )
    _render(
        plotting.plot_q_alpha_vs_croma_scatter,
        rows=rows,
        out_name="q_alpha_vs_croma_scatter.png",
    )
    _render(
        plotting.plot_croma_sample_distributions,
        rows=rows,
        out_name="croma_sample_distributions.png",
    )
    if prune_ss_oo:
        _render(
            plotting.plot_ri_mari_sample_distributions,
            rows=rows,
            metric="ri",
            out_name="ri_sample_distributions.png",
        )
        _render(
            plotting.plot_ri_mari_sample_distributions,
            rows=rows,
            metric="mari",
            out_name="mari_sample_distributions.png",
        )
    return rendered


def render_run(
    run_dir: Path,
    *,
    plots_dir: Path | None = None,
    summarize_by_mean: bool | None = None,
    prune_ss_oo: bool | None = None,
) -> list[Path]:
    """Render the figure set for a completed benchmark run directory.

    ``run_dir`` is a benchmark run directory (``output/metrics/<protocol>/<benchmark>``):
    it holds ``results/metrics.json`` and the sweep JSONs. Figures land in ``run_dir/plots``
    unless ``plots_dir`` is given. When the render flags are not passed explicitly they are
    read from ``results/render_manifest.json``.
    """
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    if plots_dir is None:
        plots_dir = run_dir / "plots"

    rows = _load_rows(results_dir / "metrics.json")
    if not rows:
        raise FileNotFoundError(
            f"no metrics rows found under {results_dir} -- run benchmark.py first"
        )
    k_sweep_rows = _load_rows(results_dir / "k_sweep_metrics.json")
    croma_m_sweep_rows = _load_rows(results_dir / "croma_m_sweep_metrics.json")

    flags = _load_render_flags(results_dir)
    if summarize_by_mean is None:
        summarize_by_mean = flags["summarize_by_mean"]
    if prune_ss_oo is None:
        prune_ss_oo = flags["prune_ss_oo"]

    return render_figure_set(
        rows=rows,
        k_sweep_rows=k_sweep_rows,
        croma_m_sweep_rows=croma_m_sweep_rows,
        plots_dir=plots_dir,
        summarize_by_mean=bool(summarize_by_mean),
        prune_ss_oo=bool(prune_ss_oo),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a benchmark run's figure set from its written metrics artifacts "
            "(run scripts/bench/benchmark.py first to produce them)."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Benchmark run directory (output/metrics/<protocol>/<benchmark>).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rendered = render_run(args.run_dir)
    plots_dir = Path(args.run_dir) / "plots"
    print(f"[render] rendered {len(rendered)} figures into {plots_dir}/{{png,pdf}}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
