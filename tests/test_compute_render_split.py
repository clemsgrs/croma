"""Characterization tests for the benchmark compute/render split (issue #57).

These pin the seam between the two halves of the benchmark pipeline:

* ``benchmark.py`` is compute-only -- it writes the metrics JSON/CSV and per-sample
  artifacts and imports no matplotlib. ``test_compute_metrics_match_golden`` asserts the
  compute output on a fixed synthetic manifest is byte-stable against a committed golden.
* ``render.py`` turns a completed run directory into the figure set.
  ``test_render_emits_expected_figures`` asserts it emits every base-set figure (PNG+PDF)
  without error.

The point is the *seam*, not matplotlib internals: we check the metric artifacts do not
drift and that render consumes them into the expected files.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "bench"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark as bm  # noqa: E402
import render  # noqa: E402

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "compute_golden_metrics.json"

MODELS = ["MA", "MB", "MC"]
# 2 biological labels x 2 confounder vendors x _N_PER samples each.
_N_PER = 8
_LABELS: list[str] = []
_VENDORS: list[str] = []
for _lbl in ("A", "B"):
    for _ven in ("V1", "V2"):
        _LABELS += [_lbl] * _N_PER
        _VENDORS += [_ven] * _N_PER
# Per-model confounder-signal strength: MA is the most confounded (lowest robustness),
# MC the least, giving three distinct, defined metric profiles.
_VENDOR_SCALE = {"MA": 0.6, "MB": 0.3, "MC": 0.05}
# Columns holding per-run absolute tmp paths -- excluded from the golden comparison.
_PATH_COLUMNS = (
    "embedding_path",
    "ri_samples_path",
    "mari_samples_path",
    "croma_samples_path",
)

# The base figure set render.py emits for a plain (no summarize-by-mean / no prune) run.
_BASE_FIGURE_NAMES = (
    "knn_bio_k_sweep",
    "knn_confounder_k_sweep",
    "ri_k_sweep",
    "mari_k_sweep",
    "croma_m_sweep",
    "croma_ltm_scatter",
    "croma_ltm_bars",
    "bio_vs_confounder_scatter",
    "mari_vs_ri_scatter",
    "ri_mari_support",
    "croma_vs_mari_scatter",
    "q_alpha_vs_croma_scatter",
    "croma_sample_distributions",
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(len(_LABELS))],
            "image_path": [f"/tmp/{i}.png" for i in range(len(_LABELS))],
            "label": list(_LABELS),
            "scanner_vendor": list(_VENDORS),
            "slide_id": [f"sl{i}" for i in range(len(_LABELS))],
        }
    )


def _features(model_name: str) -> np.ndarray:
    """Deterministic, seeded per-model embeddings with label signal + vendor confound."""
    seed = {"MA": 0, "MB": 1, "MC": 2}[model_name]
    vendor_scale = _VENDOR_SCALE[model_name]
    rng = np.random.default_rng(seed)
    dim = 16
    label_sep = 2.0
    noise = 0.5
    rows = []
    for lbl, ven in zip(_LABELS, _VENDORS):
        v = np.zeros(dim)
        v[0 if lbl == "A" else 1] = label_sep
        v[2] += vendor_scale * (1.0 if ven == "V1" else -1.0)
        rows.append(v + rng.normal(0.0, noise, size=dim))
    return np.asarray(rows, dtype=float)


def _install_fake_registry_and_embed(monkeypatch) -> None:
    def fake_registry() -> dict:
        return {name: object() for name in MODELS}

    def fake_embed_manifest(
        manifest_path: Path,
        output_path: Path,
        spec: object,
        batch_size: int,
        num_workers: int,
        device_arg: str,
        **kwargs: object,
    ) -> tuple[Path, tuple[int, int]]:
        arr = _features(output_path.stem)
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


def run_compute(monkeypatch, tmp_path: Path) -> Path:
    """Run the compute-only driver on the fixed synthetic manifest; return the run dir."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "toy.csv"
    _manifest().to_csv(manifest_path, index=False)
    output_dir = tmp_path / "out"
    _install_fake_registry_and_embed(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--manifest",
            str(manifest_path),
            "--models",
            ",".join(MODELS),
            "--output-dir",
            str(output_dir),
            "--confounder-column",
            "scanner_vendor",
            "--evaluation-design",
            "dataset_wide",
            "--k-max",
            "5",
            "--progress",
            "off",
        ],
    )
    assert bm.main() == 0
    return output_dir / manifest_path.stem


def _stable_rows(run_dir: Path) -> list[dict]:
    rows = json.loads((run_dir / "results" / "metrics.json").read_text())
    for row in rows:
        for col in _PATH_COLUMNS:
            row.pop(col, None)
    return rows


def _assert_rows_match(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.keys() == want.keys()
        for key in want:
            gv, wv = got[key], want[key]
            if isinstance(wv, float):
                if wv != wv:  # NaN
                    assert gv != gv, f"{key}: expected NaN, got {gv}"
                else:
                    assert gv == pytest.approx(wv, rel=1e-6, abs=1e-9), key
            else:
                assert gv == wv, key


def test_compute_metrics_match_golden(monkeypatch, tmp_path: Path) -> None:
    run_dir = run_compute(monkeypatch, tmp_path)
    rows = _stable_rows(run_dir)
    golden = json.loads(GOLDEN_PATH.read_text())
    _assert_rows_match(rows, golden)


def test_compute_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    run_a = run_compute(monkeypatch, tmp_path / "a")
    run_b = run_compute(monkeypatch, tmp_path / "b")
    _assert_rows_match(_stable_rows(run_a), _stable_rows(run_b))


def test_compute_writes_no_matplotlib_and_no_plots(monkeypatch, tmp_path: Path) -> None:
    run_dir = run_compute(monkeypatch, tmp_path)
    # Compute writes the metric artifacts and a render manifest, but no figures.
    assert (run_dir / "results" / "metrics.json").exists()
    assert (run_dir / "results" / "render_manifest.json").exists()
    assert not (run_dir / "plots").exists()


def test_benchmark_module_imports_no_matplotlib() -> None:
    """Importing the compute driver must not pull in matplotlib or the plotting module."""
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(SCRIPTS)!r}); "
        "import benchmark; "
        "assert 'matplotlib' not in sys.modules, 'benchmark imported matplotlib'; "
        "assert 'plotting' not in sys.modules, 'benchmark imported plotting'; "
        "print('ok')"
    )
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{SCRIPTS}"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def _figure_paths(plots_dir: Path, name: str) -> tuple[Path, Path]:
    return plots_dir / "png" / f"{name}.png", plots_dir / "pdf" / f"{name}.pdf"


def test_render_emits_expected_figures(monkeypatch, tmp_path: Path) -> None:
    run_dir = run_compute(monkeypatch, tmp_path)
    rendered = render.render_run(run_dir)

    plots_dir = run_dir / "plots"
    assert plots_dir.is_dir()
    # Every base-set figure produced a PNG and a PDF.
    assert {p.stem for p in rendered} == set(_BASE_FIGURE_NAMES)
    for name in _BASE_FIGURE_NAMES:
        png_path, pdf_path = _figure_paths(plots_dir, name)
        assert png_path.exists(), f"missing PNG for {name}"
        assert pdf_path.exists(), f"missing PDF for {name}"
        assert png_path.stat().st_size > 0
        assert pdf_path.stat().st_size > 0


def test_render_cli_runs(monkeypatch, tmp_path: Path) -> None:
    run_dir = run_compute(monkeypatch, tmp_path)
    assert render.main([str(run_dir)]) == 0
    assert (run_dir / "plots" / "png" / "ri_mari_support.png").exists()


def test_render_into_custom_plots_dir(monkeypatch, tmp_path: Path) -> None:
    """The plots_dir override (used by regen_paper_figs_faithful.py) redirects output."""
    run_dir = run_compute(monkeypatch, tmp_path)
    scratch = run_dir / "results" / "plots_scratch"
    render.render_run(run_dir, plots_dir=scratch)
    assert (scratch / "png" / "ri_mari_support.png").exists()
    assert (scratch / "pdf" / "ri_mari_support.pdf").exists()
    # The default plots dir stays untouched when an override is given.
    assert not (run_dir / "plots").exists()


def test_render_respects_manifest_flags(monkeypatch, tmp_path: Path) -> None:
    """A prune-ss-oo / summarize-by-mean run must additionally emit the gated figures."""
    run_dir = run_compute(monkeypatch, tmp_path)
    # Simulate a run whose flags request the conditional figures.
    manifest_path = run_dir / "results" / "render_manifest.json"
    manifest_path.write_text(
        json.dumps({"summarize_by_mean": True, "prune_ss_oo": True}) + "\n"
    )
    rendered = render.render_run(run_dir)
    names = {p.stem for p in rendered}
    assert "ri_cumulative_mean_k_sweep" in names
    assert "mari_cumulative_mean_k_sweep" in names
    assert "ri_sample_distributions" in names
    assert "mari_sample_distributions" in names
