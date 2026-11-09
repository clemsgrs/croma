"""A run's per-sample distributions are found through the layout, not a recorded path.

``benchmark.py`` used to store absolute paths that began at the checkout which ran it. Once
that checkout was gone, every reader that opened a recorded path broke: the paper's F(0)
macros crashed, and the ridgeline silently dropped every model it could not find. The
readers now derive a model's distribution from the run's results directory, and new runs
record their paths relative to the repo root.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts" / "bench", ROOT / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import layout  # noqa: E402
from plotting import _load_croma_sample_rows  # noqa: E402

#: The checkout that ran the benchmark, long since deleted.
GONE = Path("/var/tmp/no-such-checkout")

#: Per-sample CRoMa per model, keyed by registry name. F(0) is the closed-boundary fraction
#: at or below zero: 0, 1/4, 1/2 and 3/4, so it rises with the confounder probe while pooled
#: CRoMa falls. ``Mettle`` is published as "Mettle-1 (Pre-release)", so a reader holding the
#: published frame must map the name back to find the file.
DISTRIBUTIONS = {
    "Alpha": [0.10, 0.20, 0.30, 0.40],
    "Beta": [-0.10, 0.20, 0.30, 0.40],
    "Gamma": [-0.10, 0.00, 0.30, 0.40],
    "Mettle": [-0.30, -0.20, -0.10, 0.40],
}


def _stale_run(results_dir: Path) -> list[dict]:
    """A run whose distributions sit in the layout but whose rows name a deleted checkout."""
    rows = []
    for rank, (model, values) in enumerate(DISTRIBUTIONS.items()):
        np.save(_ensure(layout.sample_distribution_path(results_dir, "croma", model)), values)
        stale = GONE / "output" / "metrics" / "median-k" / "toy" / "results"
        rows.append(
            {
                "model": model,
                "croma": 0.4 - 0.1 * rank,
                "ri": 0.3 - 0.1 * rank,
                "mari": 0.35 - 0.1 * rank,
                "croma_ltm_alpha": -0.1 * rank,
                "confounder_knn_bacc": 0.6 + 0.1 * rank,
                "bio_knn_bacc": 0.9 - 0.05 * rank,
                "croma_samples_path": str(layout.sample_distribution_path(stale, "croma", model)),
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "metrics.csv", index=False)
    (results_dir / "metrics.json").write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _ensure(path: Path) -> Path:
    """``path``, once the directory that holds it exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_a_distribution_lives_beside_its_run_under_the_metric_and_model_name(tmp_path):
    results = tmp_path / "results"

    assert layout.sample_distribution_path(results, "croma", "CONCH") == (
        results / "sample_distributions" / "croma.CONCH.npy"
    )
    # Registry names may carry path separators; the file name must not.
    assert layout.sample_distribution_path(results, "mari", "hf:org/Model").name == (
        "mari.hf_org_Model.npy"
    )


def test_a_recorded_path_is_repo_relative_even_through_a_symlinked_output(tmp_path, monkeypatch):
    # The live tree: ``output`` is a symlink to shared storage. Resolving it would record the
    # storage path; the record must stay ``output/...``.
    storage = _directory(tmp_path / "share" / "output")
    repo = _directory(tmp_path / "checkout")
    (repo / "output").symlink_to(storage, target_is_directory=True)
    monkeypatch.setattr(layout, "REPO", repo)
    monkeypatch.setattr(layout, "OUTPUT_ROOT", repo / "output")

    run = layout.results_dir("median-k", "pathorob-camelyon")
    recorded = layout.repo_relative(layout.sample_distribution_path(run, "croma", "CONCH"))

    assert recorded == (
        "output/metrics/median-k/pathorob-camelyon/results/sample_distributions/croma.CONCH.npy"
    )


def test_a_path_outside_the_repo_is_recorded_unchanged(tmp_path, monkeypatch):
    # A study may drive the benchmark into a scratch output tree outside the checkout; that
    # path has no repo-relative form, and the run must not fail for it.
    monkeypatch.setattr(layout, "REPO", tmp_path / "checkout")
    scratch = tmp_path / "scratch" / "output" / "embeddings" / "toy" / "M1.npy"

    assert layout.repo_relative(scratch) == str(scratch)


def test_the_ridgeline_finds_every_distribution_when_the_recorded_prefix_is_gone(tmp_path):
    results = _directory(tmp_path / "run" / "results")
    rows = _stale_run(results)
    assert not GONE.exists()

    loaded = _load_croma_sample_rows(rows, results_dir=results)

    assert [d["model"] for d in loaded] == ["Alpha", "Beta", "Gamma", "Mettle"]
    for d in loaded:
        np.testing.assert_array_equal(d["values"], DISTRIBUTIONS[d["model"]])


def test_a_scored_model_without_its_distribution_fails_the_ridgeline_by_name(tmp_path):
    results = _directory(tmp_path / "run" / "results")
    rows = _stale_run(results)
    missing = layout.sample_distribution_path(results, "croma", "Gamma")
    missing.unlink()

    with pytest.raises(FileNotFoundError) as excinfo:
        _load_croma_sample_rows(rows, results_dir=results)

    assert "Gamma" in str(excinfo.value)
    assert str(missing) in str(excinfo.value)


def test_unscored_and_unrequested_models_are_still_skipped(tmp_path):
    results = _directory(tmp_path / "run" / "results")
    rows = _stale_run(results)
    # Neither row is drawn, so neither needs a distribution on disk.
    rows[0]["croma"] = float("nan")
    layout.sample_distribution_path(results, "croma", "Alpha").unlink()
    layout.sample_distribution_path(results, "croma", "Mettle").unlink()

    loaded = _load_croma_sample_rows(rows, results_dir=results, models=["Alpha", "Beta", "Gamma"])

    assert [d["model"] for d in loaded] == ["Beta", "Gamma"]


def test_the_paper_f0_macros_read_each_distribution_from_the_run_not_the_recorded_path(
    tmp_path,
):
    import generate_paper_values as gpv
    from plotting.style import published_models

    results = _directory(tmp_path / "run" / "results")
    _stale_run(results)
    df = published_models(pd.read_csv(results / "metrics.csv"))
    assert "Mettle-1 (Pre-release)" in set(df["model"])

    lines, _ = gpv._probe_macros("Tolkach", df, results)

    # F(0) climbs with the probe (0, 1/4, 1/2, 3/4) while pooled CRoMa falls, so the two
    # rank correlations have opposite signs: only the distributions can give F(0) its +1.
    assert r"\newcommand{\TolkachProbeFZeroRho}{$1.00$}" in lines
    assert r"\newcommand{\TolkachProbeCromaRho}{$-1.00$}" in lines


# --- what a new run records -----------------------------------------------------------------


def _toy_benchmark(bench_env) -> None:
    """A two-label, two-vendor tileset of 16 tiles, embedded by one model, viewed whole."""
    n = 16
    labels = ["A", "B"] * (n // 2)
    vendors = ["V1", "V1", "V2", "V2"] * (n // 4)
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(n)],
            "image_path": [f"/tiles/{i}.png" for i in range(n)],
            "label": labels,
            "scanner_vendor": vendors,
            "group_id": [f"g{i}" for i in range(n)],
        }
    )
    rng = np.random.default_rng(0)
    features = np.zeros((n, 4))
    features[:, 0] = [1.0 if lbl == "A" else -1.0 for lbl in labels]
    features[:, 1] = [0.3 if v == "V1" else -0.3 for v in vendors]
    features += rng.normal(0.0, 0.2, size=features.shape)
    bench_env.write_tileset("toy-tiles", manifest, {"M1": features})
    bench_env.register(
        "toy", tileset="toy-tiles", manifest=manifest, k_max=3, confounder_column="scanner_vendor"
    )
    assert bench_env.run("toy", "k-star", "--progress", "off") == 0


RUN = "output/metrics/k-star/toy/results"


def test_a_result_row_records_its_artifacts_relative_to_the_repo_root(bench_env):
    _toy_benchmark(bench_env)

    row = pd.read_csv(bench_env.results_dir("toy") / "metrics.csv").iloc[0]

    assert row["model"] == "M1"
    assert row["ri_samples_path"] == f"{RUN}/sample_distributions/ri.M1.npy"
    assert row["mari_samples_path"] == f"{RUN}/sample_distributions/mari.M1.npy"
    assert row["croma_samples_path"] == f"{RUN}/sample_distributions/croma.M1.npy"
    assert row["embedding_path"] == "output/embeddings/toy-tiles/M1.npy"
    # Read against the repo root, each one names the artifact the run wrote.
    for column in ("ri_samples_path", "mari_samples_path", "croma_samples_path", "embedding_path"):
        assert (bench_env.root / row[column]).is_file(), column


def test_the_sweep_rows_record_the_embedding_relative_to_the_repo_root(bench_env):
    _toy_benchmark(bench_env)
    results = bench_env.results_dir("toy")

    for name in ("k_sweep_metrics", "croma_m_sweep_metrics"):
        sweep = pd.read_csv(results / f"{name}.csv")
        assert set(sweep["embedding_path"]) == {"output/embeddings/toy-tiles/M1.npy"}, name
        rows = json.loads((results / f"{name}.json").read_text(encoding="utf-8"))
        assert {r["embedding_path"] for r in rows} == {"output/embeddings/toy-tiles/M1.npy"}


def test_a_distribution_sidecar_records_its_array_relative_to_the_repo_root(bench_env):
    _toy_benchmark(bench_env)
    sidecars = bench_env.results_dir("toy") / "sample_distributions"

    for metric in ("ri", "mari"):
        meta = json.loads((sidecars / f"{metric}.M1.json").read_text(encoding="utf-8"))
        assert meta["distribution_path"] == f"{RUN}/sample_distributions/{metric}.M1.npy"


def test_the_cache_index_records_each_payload_relative_to_the_repo_root(bench_env):
    _toy_benchmark(bench_env)
    index = bench_env.results_dir("toy") / "cache" / "index.jsonl"

    entries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]

    assert entries
    for entry in entries:
        payload = entry["payload_path"]
        assert payload.startswith(f"{RUN}/cache/artifacts/{entry['artifact_name']}/M1/")
        assert (bench_env.root / payload).is_file()
