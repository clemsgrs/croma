"""Slide encoders versus mean-pooling of their own tile encoders on PCaBiop.

The study re-runs each slide encoder with its tile embeddings saved, so the slide arm and
the mean-pooled arm come from one tiling of the same slides. These tests cover what the
study owns: the re-extraction config, the reduction to two arms, the reproduction check
that decides which slide arm is used, the paired differences, and the fence that keeps the
study arms out of the published slide panel. None of them needs a GPU, torch or the real
cohort.

The rendered table, its macros and the full reproduction's study commands read the paper's
tooling, so their tests live in ``test_slide_vs_meanpool_paper.py``, which the sdist leaves
out together with ``scripts/repro``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts" / "studies" / "apd", ROOT / "scripts" / "studies"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SLIDE_PANEL = ["PRISM", "PRISM2", "TITAN", "Prov-GigaPath", "MOOZY"]


def _write_pcabiop(root: Path, models: list[str], n_rows: int = 4) -> Path:
    directory = root / "output" / "embeddings" / "pcabiop"
    directory.mkdir(parents=True, exist_ok=True)
    for model in models:
        np.save(directory / f"{model}.npy", np.zeros((n_rows, 2), dtype=np.float32))
    return directory


def test_the_pcabiop_apd_roster_is_the_published_slide_panel(tmp_path, monkeypatch):
    # Study arms live in the study's outputs, but one can still reach the published
    # directory (an ``--embeddings-dir`` override, or a reduction from before the study
    # kept its arms to itself). A default APD run that took every .npy in that directory
    # would then fold it into the published panel.
    import apd_experiment
    import layout

    monkeypatch.setattr(layout, "OUTPUT_ROOT", tmp_path / "output")
    _write_pcabiop(tmp_path, SLIDE_PANEL + ["TITAN-meanpool", "TITAN-rerun"])

    assert apd_experiment.model_list("pcabiop") == SLIDE_PANEL


#: A source run config in the layout slide2vec used before hs2p retired three tiling keys
#: (the published MOOZY PANDA run is written this way).
LEGACY_SOURCE_CONFIG = {
    "csv": "/elsewhere/panda.csv",
    "output_dir": "/elsewhere/features/abc123",
    "resume": False,
    "resume_dirname": None,
    "model": {"name": "moozy-slide", "save_tile_embeddings": False, "batch_size": 32},
    "tiling": {
        "backend": "openslide",
        "independent_sampling": True,
        "params": {"requested_tile_size_px": 224, "tissue_threshold": 0.1},
        "seg_params": {"method": None, "downsample": 64},
    },
    "speed": {"num_dataloader_workers": 16, "num_preprocessing_workers": 16},
    "wandb": {"enable": True, "resume_id": "abc123"},
}


def test_the_extraction_config_saves_tiles_and_runs_on_the_installed_tiler(tmp_path):
    import slide_vs_meanpool as study

    cfg = study.extraction_config(
        LEGACY_SOURCE_CONFIG,
        cohort="panda",
        slide_csv=tmp_path / "panda.csv",
        run_root=tmp_path / "runs" / "MOOZY",
    )

    tiling = cfg["tiling"]
    # Retired keys are mapped onto their replacements, never dropped silently.
    assert "independent_sampling" not in tiling
    assert "tissue_threshold" not in tiling["params"]
    assert tiling["masks"]["strategy"] == "independent"
    assert tiling["masks"]["min_coverage"]["tissue"] == 0.1
    assert tiling["seg_params"]["method"] == "hsv"
    # PANDA slides and masks are read with ASAP, as the published PANDA runs did.
    assert tiling["backend"] == "asap" and tiling["mask_backend"] == "asap"
    # One run yields both arms: slide embeddings and the tile embeddings behind them.
    assert cfg["model"]["save_tile_embeddings"] is True
    assert cfg["model"]["name"] == "moozy-slide"
    assert cfg["csv"] == str(tmp_path / "panda.csv")
    assert cfg["output_dir"] == str(tmp_path / "runs" / "MOOZY")
    assert cfg["resume"] is True and cfg["resume_dirname"] == "panda"
    assert cfg["wandb"]["enable"] is False
    # The source config is the record of the published run; it must not be mutated.
    assert LEGACY_SOURCE_CONFIG["tiling"]["independent_sampling"] is True


def test_par_slides_keep_the_published_par_reader(tmp_path):
    import slide_vs_meanpool as study

    cfg = study.extraction_config(
        LEGACY_SOURCE_CONFIG,
        cohort="par",
        slide_csv=tmp_path / "par.csv",
        run_root=tmp_path / "runs" / "MOOZY",
    )

    assert cfg["tiling"]["backend"] == "openslide"
    assert cfg["tiling"]["mask_backend"] == "openslide"
    assert cfg["resume_dirname"] == "par"


#: Three PCaBiop rows: two in-domain PANDA slides then one out-of-domain PAR slide, the
#: order every pcabiop matrix follows.
ROWS = pd.DataFrame(
    {
        "sample_id": ["p1", "p2", "q1"],
        "label": ["benign", "cancer", "cancer"],
        "confounder": ["radboud", "karolinska", "PAR"],
        "group_id": ["p1", "p2", "q1"],
        "subset": ["ID", "ID", "OOD"],
    }
)

#: Per-slide tile embeddings (2-d) and the slide embedding the re-run produced.
TILES = {
    "p1": [[1.0, 0.0], [3.0, 2.0]],
    "p2": [[0.0, 4.0]],
    "q1": [[2.0, 2.0], [4.0, 0.0], [0.0, 1.0]],
}
SLIDES = {"p1": [1.0, 2.0], "p2": [3.0, 1.0], "q1": [0.5, 0.5]}


def _write_run(run_root: Path) -> None:
    """A finished slide2vec re-run: tile and slide embeddings per slide, per cohort."""
    for sid, cohort in (("p1", "panda"), ("p2", "panda"), ("q1", "par")):
        for kind, payload in (("tile_embeddings", TILES), ("slide_embeddings", SLIDES)):
            directory = run_root / cohort / kind
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / f"{sid}.npy", np.asarray(payload[sid], dtype=np.float16))


def _published(tmp_path: Path, published) -> Path:
    """A stand-in for the published ``output/embeddings/pcabiop/``, holding ``TITAN.npy``."""
    directory = tmp_path / "published" / "pcabiop"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "TITAN.npy", np.asarray(published, dtype=np.float32))
    return directory


def _study_arms(tmp_path: Path) -> Path:
    """Where the study arms land by default: inside the study's own outputs."""
    return tmp_path / "study" / "embeddings" / "pcabiop"


def _reduce(tmp_path: Path, published: np.ndarray, embeddings_dir: Path | None = None):
    import slide_vs_meanpool as study

    published_dir = _published(tmp_path, published)
    run_root = tmp_path / "study" / "runs" / "TITAN"
    _write_run(run_root)
    override = {} if embeddings_dir is None else {"embeddings_dir": embeddings_dir}
    record = study.reduce_pair(
        study.PAIRS_BY_MODEL["TITAN"],
        rows=ROWS,
        run_root=run_root,
        study_dir=tmp_path / "study",
        published_dir=published_dir,
        **override,
    )
    return record, embeddings_dir or _study_arms(tmp_path)


def test_the_mean_pooled_arm_is_the_plain_mean_of_each_slides_tiles(tmp_path):
    import json

    record, embeddings = _reduce(tmp_path, published=[SLIDES[s] for s in ROWS["sample_id"]])

    meanpool = np.load(embeddings / "TITAN-meanpool.npy")
    np.testing.assert_allclose(meanpool, [[2.0, 1.0], [0.0, 4.0], [2.0, 1.0]])
    assert meanpool.dtype == np.float32
    sidecar = json.loads((embeddings / "TITAN-meanpool.npy.json").read_text())
    assert sidecar["n_tiles"] == [2, 1, 3]
    assert sidecar["tile_encoder"] == "conchv15"
    assert sidecar["tile_encoder_output_variant"] == "default"
    assert sidecar["pooling"] == "mean"
    assert set(sidecar["source_runs"]) == {"panda", "par"}
    assert (sidecar["n_id"], sidecar["n_ood"]) == (2, 1)
    assert record["meanpool_arm"] == "TITAN-meanpool"


def test_a_reproducing_rerun_keeps_the_published_slide_arm(tmp_path):
    # Scaling leaves the cosine at 1: what is compared is the direction of each embedding.
    published = [[2.0 * v for v in SLIDES[s]] for s in ROWS["sample_id"]]

    record, embeddings = _reduce(tmp_path, published=published)

    assert record["reproduced"] is True
    assert record["slide_arm"] == "TITAN"
    assert (record["n_slides"], record["n_reproduced"]) == (3, 3)
    assert not (embeddings / "TITAN-rerun.npy").exists()


def test_a_rerun_that_does_not_reproduce_becomes_the_slide_arm(tmp_path):
    published = [SLIDES["p1"], [-1.0, 3.0], SLIDES["q1"]]  # p2 is orthogonal to its re-run

    record, embeddings = _reduce(tmp_path, published=published)

    assert record["reproduced"] is False
    assert record["slide_arm"] == "TITAN-rerun"
    assert record["n_reproduced"] == 2
    assert record["min_cosine"] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(
        np.load(embeddings / "TITAN-rerun.npy"), [SLIDES[s] for s in ROWS["sample_id"]]
    )
    per_slide = pd.read_csv(tmp_path / "study" / "reproduction" / "TITAN.csv")
    assert per_slide["sample_id"].tolist() == ["p1", "p2", "q1"]


def test_reduction_never_overwrites_an_existing_different_embedding(tmp_path):
    import slide_vs_meanpool as study

    embeddings = _study_arms(tmp_path)
    embeddings.mkdir(parents=True)
    np.save(embeddings / "TITAN-meanpool.npy", np.ones((3, 2), dtype=np.float32))

    with pytest.raises(FileExistsError, match="TITAN-meanpool"):
        _reduce(tmp_path, published=[SLIDES[s] for s in ROWS["sample_id"]])
    np.testing.assert_array_equal(np.load(embeddings / "TITAN-meanpool.npy"), np.ones((3, 2)))
    assert not (embeddings / "TITAN-meanpool.npy.json").exists()

    # Re-running a finished reduction is a no-op rather than an error.
    (embeddings / "TITAN-meanpool.npy").unlink()
    first, _ = _reduce(tmp_path, published=[SLIDES[s] for s in ROWS["sample_id"]])
    second, _ = _reduce(tmp_path, published=[SLIDES[s] for s in ROWS["sample_id"]])
    assert first == second
    assert study.PAIRS_BY_MODEL["TITAN"].slide_model == "TITAN"


def test_reducing_an_unfinished_extraction_names_the_extraction_step(tmp_path):
    # The one-command reproduction runs reduce; it cannot run the GPU re-extraction, so
    # an absent or partial run must say what is missing and how to produce it.
    import slide_vs_meanpool as study

    published = _published(tmp_path, np.zeros((3, 2)))
    run_root = tmp_path / "study" / "runs" / "TITAN"
    _write_run(run_root)
    (run_root / "par" / "slide_embeddings" / "q1.npy").unlink()

    with pytest.raises(RuntimeError, match=r"TITAN/par: 0/1 .* extract"):
        study.reduce_pair(
            study.PAIRS_BY_MODEL["TITAN"],
            rows=ROWS,
            run_root=run_root,
            study_dir=tmp_path / "study",
            published_dir=published,
        )
    assert not (_study_arms(tmp_path) / "TITAN-meanpool.npy").exists()


# --- where the study arms live ---------------------------------------------------------


def test_the_default_arm_directory_is_in_the_study_outputs_not_the_published_tileset():
    import slide_vs_meanpool as study

    default = study.study_embeddings_dir(study.STUDY_DIR)

    assert default == study.STUDY_DIR / "embeddings" / "pcabiop"
    assert not default.is_relative_to(study.PCABIOP_DIR)


def test_reduce_writes_the_study_arms_into_the_study_outputs_by_default(tmp_path, monkeypatch):
    # Other branches' PCaBiop APD sweeps glob the published directory for models, so the
    # study arms never land there unless an override asks for it.
    import slide_vs_meanpool as study

    published = _published(tmp_path, [SLIDES["p1"], [-1.0, 3.0], SLIDES["q1"]])  # p2 differs
    ROWS.to_csv(published / "manifest.csv", index=False)
    before = {p.name: p.read_bytes() for p in published.iterdir()}
    study_dir = tmp_path / "study"
    _write_run(study_dir / "runs" / "TITAN")
    monkeypatch.setattr(study, "PCABIOP_DIR", published)

    assert study.main(["reduce", "--study-dir", str(study_dir), "--pairs", "TITAN"]) == 0

    assert sorted(p.name for p in _study_arms(tmp_path).iterdir()) == [
        "TITAN-meanpool.npy",
        "TITAN-meanpool.npy.json",
        "TITAN-rerun.npy",
        "TITAN-rerun.npy.json",
    ]
    # The published directory is read (row order, the published TITAN.npy), never written.
    assert {p.name: p.read_bytes() for p in published.iterdir()} == before


def test_an_override_into_the_published_directory_still_never_touches_a_published_file(
    tmp_path,
):
    published = [SLIDES["p1"], [-1.0, 3.0], SLIDES["q1"]]
    published_dir = tmp_path / "published" / "pcabiop"

    record, embeddings = _reduce(tmp_path, published=published, embeddings_dir=published_dir)

    assert record["slide_arm"] == "TITAN-rerun"
    assert (embeddings / "TITAN-meanpool.npy").exists()
    np.testing.assert_array_equal(np.load(published_dir / "TITAN.npy"), published)


def test_evaluate_reads_study_arms_from_the_study_outputs_and_published_arms_read_only(
    tmp_path,
):
    import slide_vs_meanpool as study

    published = _published(tmp_path, np.zeros((3, 2)))
    arms = _study_arms(tmp_path)
    arms.mkdir(parents=True)
    for name in ("TITAN-meanpool", "MOOZY-rerun", "MOOZY-meanpool"):
        np.save(arms / f"{name}.npy", np.ones((3, 2), dtype=np.float32))
    models = ["TITAN", "TITAN-meanpool", "MOOZY-rerun", "MOOZY-meanpool"]

    sources = study.arm_sources(models, embeddings_dir=arms, published_dir=published)

    assert sources == {
        "TITAN": published / "TITAN.npy",
        "TITAN-meanpool": arms / "TITAN-meanpool.npy",
        "MOOZY-rerun": arms / "MOOZY-rerun.npy",
        "MOOZY-meanpool": arms / "MOOZY-meanpool.npy",
    }
    # A study arm is never looked up beside the published embeddings, even if one is there.
    np.save(published / "PRISM2-meanpool.npy", np.ones((3, 2), dtype=np.float32))
    with pytest.raises(FileNotFoundError, match=r"PRISM2-meanpool.*reduce"):
        study.arm_sources(["PRISM2-meanpool"], embeddings_dir=arms, published_dir=published)


def test_the_downstream_driver_scores_the_arms_from_their_sources(tmp_path, monkeypatch):
    # The APD loader reads ``layout.embeddings_dir("pcabiop")``; the study stages its arms
    # in a study-local root for it, as it does for the benchmark runner.
    import apd_experiment
    import layout
    import slide_vs_meanpool as study

    published = _published(tmp_path, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    arms = _study_arms(tmp_path)
    arms.mkdir(parents=True)
    np.save(arms / "TITAN-meanpool.npy", np.full((3, 2), 7.0, dtype=np.float32))
    models = ["TITAN", "TITAN-meanpool"]
    sources = study.arm_sources(models, embeddings_dir=arms, published_dir=published)
    seen = {}

    def fake_run(datasets, models, *, iterations, out_dir, overwrite, jobs):
        import json

        for model in models:
            seen[model] = np.load(layout.embeddings_dir("pcabiop") / f"{model}.npy")
            cell = Path(out_dir) / "pcabiop" / f"{model}.json"
            cell.parent.mkdir(parents=True, exist_ok=True)
            cell.write_text(json.dumps({"model": model}))
        return pd.DataFrame({"dataset": "pcabiop", "model": list(models)})

    monkeypatch.setattr(apd_experiment, "run", fake_run)
    output_root = layout.OUTPUT_ROOT

    apd, cells, _ = study.run_downstream(
        models, sources=sources, study_dir=tmp_path / "study", jobs=1
    )

    np.testing.assert_array_equal(seen["TITAN"], np.load(published / "TITAN.npy"))
    np.testing.assert_array_equal(seen["TITAN-meanpool"], np.full((3, 2), 7.0))
    assert set(cells) == set(models) and apd["model"].tolist() == models
    assert layout.OUTPUT_ROOT == output_root


def test_an_apd_cell_is_recomputed_when_its_arms_source_matrix_changes(tmp_path, monkeypatch):
    # The APD driver resumes a cell by model name alone. A re-reduced arm, an
    # ``--embeddings-dir`` override or a changed published ``<M>.npy`` keeps its name, so
    # without a record of the content behind each cell its nIPD would silently go stale
    # while the (fingerprinted) representation metrics recompute.
    import hashlib
    import json

    import apd_experiment
    import layout
    import slide_vs_meanpool as study

    published = _published(tmp_path, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    arms = _study_arms(tmp_path)
    arms.mkdir(parents=True)
    np.save(arms / "TITAN-meanpool.npy", np.full((3, 2), 7.0, dtype=np.float32))
    models = ["TITAN", "TITAN-meanpool"]
    sources = study.arm_sources(models, embeddings_dir=arms, published_dir=published)
    study_dir = tmp_path / "study"
    computed = []

    def fake_run(datasets, models, *, iterations, out_dir, overwrite, jobs):
        # apd_experiment's resume rule: a present cell is loaded unless ``overwrite``.
        for model in models:
            cell = Path(out_dir) / "pcabiop" / f"{model}.json"
            if cell.exists() and not overwrite:
                continue
            computed.append(model)
            matrix = np.load(layout.embeddings_dir("pcabiop") / f"{model}.npy")
            cell.parent.mkdir(parents=True, exist_ok=True)
            cell.write_text(json.dumps({"model": model, "sum": float(matrix.sum())}))
        return pd.DataFrame({"dataset": "pcabiop", "model": list(models)})

    monkeypatch.setattr(apd_experiment, "run", fake_run)

    def downstream():
        computed.clear()
        return study.run_downstream(models, sources=sources, study_dir=study_dir, jobs=1)

    # A cell left by a run that recorded no source content is not trusted.
    stale = study_dir / "apd" / "pcabiop" / "TITAN.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"model": "TITAN", "sum": -1.0}))
    first = downstream()
    assert sorted(computed) == sorted(models)
    assert first[1]["TITAN"]["sum"] == 4.0

    # Same content: both cells are reused.
    downstream()
    assert computed == []

    # Re-reduced: same arm name, other content. Only that arm's cell is recomputed.
    np.save(arms / "TITAN-meanpool.npy", np.full((3, 2), 9.0, dtype=np.float32))
    apd, cells, sha256 = downstream()
    assert computed == ["TITAN-meanpool"]
    assert cells["TITAN-meanpool"]["sum"] == 54.0
    assert set(apd["model"]) == set(models)
    assert sha256 == {m: hashlib.sha256(sources[m].read_bytes()).hexdigest() for m in models}
    assert sha256["TITAN"] == first[2]["TITAN"]
    assert sha256["TITAN-meanpool"] != first[2]["TITAN-meanpool"]

    # ``recompute`` recomputes every cell whatever its record says.
    computed.clear()
    study.run_downstream(models, sources=sources, study_dir=study_dir, jobs=1, recompute=True)
    assert sorted(computed) == sorted(models)


#: Source hashes the stubbed downstream driver reports for the two TITAN arms.
ARM_SHA256 = {"TITAN-rerun": "a" * 64, "TITAN-meanpool": "b" * 64}


def _evaluate(tmp_path: Path, monkeypatch, extraction: str | None, on_score=None) -> Path:
    """``evaluate`` on one reduced pair, with both drivers stubbed out.

    ``extraction`` is the text of ``extraction.json`` (``None``: no record). ``on_score``,
    if given, is called with ``"representation"`` or ``"downstream"`` from inside the
    stubbed driver of that name, so a test can see the state the scoring runs in.
    """
    import json
    from types import SimpleNamespace

    import loaders
    import slide_vs_meanpool as study
    from croma.metrics.croma import CROMA_HEADLINE_M

    study_dir = tmp_path / "study"
    study_dir.mkdir(parents=True, exist_ok=True)
    reduction = {"TITAN": {"slide_arm": "TITAN-rerun", "meanpool_arm": "TITAN-meanpool"}}
    (study_dir / study.REPRODUCTION_JSON).write_text(json.dumps(reduction))
    if extraction is not None:
        (study_dir / "extraction.json").write_text(extraction)
    models = list(ARM_SHA256)
    per_sample = _per_sample(
        {"TITAN-rerun": np.linspace(-0.5, 0.5, 40), "TITAN-meanpool": np.zeros(40)}
    ).rename(columns={"croma": f"croma_m{int(CROMA_HEADLINE_M)}"})
    metrics = pd.DataFrame({"model": models, **{c: [0.5, 0.4] for c in study.METRIC_COLUMNS}})
    apd = pd.DataFrame(
        {"dataset": "pcabiop", "model": models, "nipd_id": [0.0, -0.1], "nipd_ood": [0.0, -0.2]}
    )
    cells = {m: _cell([FLAT], [FLAT]) for m in models}

    def run_representation(models, **_):
        if on_score is not None:
            on_score("representation")
        return metrics, per_sample, ["--x"]

    def run_downstream(models, **_):
        if on_score is not None:
            on_score("downstream")
        return apd, cells, ARM_SHA256

    monkeypatch.setattr(study, "_pcabiop_rows", lambda: ROWS)
    monkeypatch.setattr(
        study, "arm_sources", lambda models, **_: {m: tmp_path / f"{m}.npy" for m in models}
    )
    monkeypatch.setattr(study, "run_representation", run_representation)
    monkeypatch.setattr(study, "check_published_arms", lambda *_: None)
    monkeypatch.setattr(study, "run_downstream", run_downstream)
    monkeypatch.setattr(study, "_published_run", lambda: SimpleNamespace(protocol="k-star"))
    monkeypatch.setattr(loaders, "training_correlations", lambda dataset: TOY_V)

    study.evaluate(["TITAN"], study_dir=study_dir, jobs=1)
    return study_dir


def test_evaluate_needs_no_slide2vec_and_records_the_extracting_version(tmp_path, monkeypatch):
    # Scoring reduced arms needs no encoder package, and the version worth recording is
    # the one that extracted the pair, which ``extract`` writes into its record. A record
    # without the field (the running extraction's predates it) gives null.
    import importlib.metadata
    import json

    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    record = {"TITAN": {"status": "done", "started": "2026-10-30 21:24:28"}}

    study_dir = _evaluate(tmp_path, monkeypatch, extraction=json.dumps(record))

    provenance = json.loads((study_dir / "provenance.json").read_text())
    assert provenance["pairs"]["TITAN"]["slide2vec_version"] is None
    assert provenance["extraction"] == record
    assert provenance["arm_source_sha256"] == ARM_SHA256
    assert (study_dir / "summary.csv").exists() and (study_dir / "differences.csv").exists()

    record["TITAN"]["slide2vec_version"] = "7.1.4"
    study_dir = _evaluate(tmp_path, monkeypatch, extraction=json.dumps(record))
    provenance = json.loads((study_dir / "provenance.json").read_text())
    assert provenance["pairs"]["TITAN"]["slide2vec_version"] == "7.1.4"


def test_evaluate_writes_no_artifact_when_its_provenance_cannot_be_built(tmp_path, monkeypatch):
    # The three artifacts describe one evaluation: a failure while assembling the
    # provenance must not leave fresh CSVs beside a missing or older provenance.json.
    import json

    with pytest.raises(json.JSONDecodeError):
        _evaluate(tmp_path, monkeypatch, extraction="{not json")

    for name in ("summary.csv", "differences.csv", "provenance.json"):
        assert not (tmp_path / "study" / name).exists()


def _numpy_blas_threads() -> list[int]:
    """The thread counts of the BLAS pools NumPy loaded (OpenBLAS in the NumPy wheels)."""
    from threadpoolctl import threadpool_info

    return [
        info["num_threads"]
        for info in threadpool_info()
        if info.get("user_api") == "blas" and "numpy" in str(info.get("filepath"))
    ]


def _forked_numpy_blas_threads() -> list[int]:
    """:func:`_numpy_blas_threads` in a worker forked the way the APD driver forks its own."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("fork")) as pool:
        return pool.submit(_numpy_blas_threads).result()


def test_evaluate_scores_with_one_blas_thread_per_process(tmp_path, monkeypatch):
    # The study imports NumPy before ``apd_experiment`` sets its one-thread defaults, so
    # those come too late for NumPy's BLAS pool. Without a runtime limit, the scoring and
    # every APD worker forked from it run with as many BLAS threads as the host gives.
    import multiprocessing

    from threadpoolctl import threadpool_limits

    if not _numpy_blas_threads():
        pytest.skip("NumPy reports no BLAS thread pool")
    can_fork = "fork" in multiprocessing.get_all_start_methods()
    seen = {}

    def record(step):
        seen[step] = _numpy_blas_threads()
        if can_fork and step == "downstream":
            seen["apd worker"] = _forked_numpy_blas_threads()

    with threadpool_limits(4, user_api="blas"):
        assert set(_numpy_blas_threads()) == {4}  # the pool the study starts from
        _evaluate(tmp_path, monkeypatch, extraction=None, on_score=record)
        assert set(_numpy_blas_threads()) == {4}  # and gives back to its caller

    assert set(seen) == {"representation", "downstream"} | ({"apd worker"} if can_fork else set())
    assert all(set(threads) == {1} for threads in seen.values()), seen


def test_evaluate_defaults_to_the_study_arm_directory(tmp_path, monkeypatch):
    import slide_vs_meanpool as study

    captured = {}
    monkeypatch.setattr(study, "evaluate", lambda pairs, **kwargs: captured.update(kwargs))

    study.main(["evaluate", "--study-dir", str(tmp_path / "study")])
    assert captured["embeddings_dir"] == _study_arms(tmp_path)
    assert captured["published_dir"] == study.PCABIOP_DIR

    study.main(["evaluate", "--study-dir", str(tmp_path / "study"), "--embeddings-dir", "x"])
    assert captured["embeddings_dir"] == Path("x")


def test_each_cohort_rerun_embeds_exactly_its_pcabiop_slides_in_row_order():
    import slide_vs_meanpool as study

    source = pd.DataFrame(
        {
            "sample_id": ["zz", "q1", "p2", "p1"],  # a superset, in another order
            "image_path": ["/s/zz", "/s/q1", "/s/p2", "/s/p1"],
            "mask_path": ["/m/zz", "/m/q1", "/m/p2", "/m/p1"],
        }
    )

    panda = study.slide_list(ROWS, "panda", source)
    par = study.slide_list(ROWS, "par", source)

    assert panda["sample_id"].tolist() == ["p1", "p2"]
    assert par["sample_id"].tolist() == ["q1"]
    assert panda.columns.tolist() == ["sample_id", "image_path", "mask_path"]
    with pytest.raises(ValueError, match="q1"):
        study.slide_list(ROWS, "par", source[source["sample_id"] != "q1"])


def test_a_cohort_is_complete_only_when_every_slide_has_both_embeddings(tmp_path):
    import slide_vs_meanpool as study

    run_root = tmp_path / "runs" / "TITAN"
    _write_run(run_root)
    assert study.cohort_progress(run_root / "panda", ["p1", "p2"]) == (2, 2)

    (run_root / "panda" / "tile_embeddings" / "p2.npy").unlink()
    assert study.cohort_progress(run_root / "panda", ["p1", "p2"]) == (1, 2)
    assert study.cohort_progress(tmp_path / "not-started", ["p1"]) == (0, 1)


def _fake_slide2vec(tmp_path: Path, monkeypatch) -> list[dict]:
    """``extract`` on ROWS with the GPU probe and slide2vec stubbed out.

    The stub reads its config and slide list as the real child would, relative to the
    ``cwd`` it is launched with, and embeds every listed slide where the config says.
    Returns the recorded launches.
    """
    import subprocess

    import slide_vs_meanpool as study
    import yaml

    features = tmp_path / "features"
    pair = study.PAIRS_BY_MODEL["TITAN"]
    for run in (pair.panda_run, pair.par_run):
        (features / run).mkdir(parents=True, exist_ok=True)
        (features / run / "config.yaml").write_text(yaml.safe_dump(LEGACY_SOURCE_CONFIG))
    slide_sources = {}
    for cohort, ids in (("panda", ["p1", "p2"]), ("par", ["q1"])):
        slide_sources[cohort] = tmp_path / f"{cohort}-slides.csv"
        pd.DataFrame(
            {
                "sample_id": ids,
                "image_path": [f"/s/{i}" for i in ids],
                "mask_path": [f"/m/{i}" for i in ids],
            }
        ).to_csv(slide_sources[cohort], index=False)
    monkeypatch.setattr(study, "FEATURES_ROOT", features)
    monkeypatch.setattr(study, "SLIDE_SOURCES", slide_sources)
    monkeypatch.setattr(study, "_pcabiop_rows", lambda: ROWS)
    monkeypatch.setattr(study, "_gpu_free_mib", lambda: float("inf"))
    monkeypatch.setenv("HF_MODULES_CACHE", str(tmp_path / "hf-modules"))
    launches = []

    def fake_run(cmd, *, env, cwd, check):
        cfg = yaml.safe_load((Path(cwd) / cmd[-1]).read_text())
        launches.append({"cmd": cmd, "cwd": cwd, "cfg": cfg})
        out = Path(cwd) / cfg["output_dir"] / cfg["resume_dirname"]
        for sid in pd.read_csv(Path(cwd) / cfg["csv"])["sample_id"]:
            for kind in ("tile_embeddings", "slide_embeddings"):
                (out / kind).mkdir(parents=True, exist_ok=True)
                np.save(out / kind / f"{sid}.npy", np.zeros((1, 2), dtype=np.float32))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return launches


def test_extract_hands_slide2vec_absolute_paths_from_a_relative_study_dir(tmp_path, monkeypatch):
    # slide2vec runs with the pair's run directory as its working directory, so a path
    # handed to it relative to the caller's directory would be read under the run dir.
    # The study dir is made absolute, not resolved: through the ``output/`` symlink the
    # recorded paths keep the spelling they were given.
    import slide_vs_meanpool as study

    launches = _fake_slide2vec(tmp_path, monkeypatch)
    (tmp_path / "real").mkdir()
    (tmp_path / "linked").symlink_to(tmp_path / "real")
    monkeypatch.chdir(tmp_path)

    assert study.main(["extract", "--study-dir", "linked/study", "--pairs", "TITAN"]) == 0

    run_root = Path.cwd() / "linked" / "study" / "runs" / "TITAN"
    assert [launch["cfg"]["resume_dirname"] for launch in launches] == ["panda", "par"]
    for launch in launches:
        assert Path(launch["cmd"][-1]).is_absolute()
        assert Path(launch["cwd"]) == run_root
        assert Path(launch["cfg"]["csv"]).is_absolute()
        assert Path(launch["cfg"]["output_dir"]) == run_root
    assert (run_root.parents[1] / "extraction.json").exists()


def test_extract_records_the_slide2vec_version_behind_each_pair(tmp_path, monkeypatch):
    # ``evaluate`` reports this version, so it must come from the extracting environment.
    import importlib.metadata
    import json

    import slide_vs_meanpool as study

    _fake_slide2vec(tmp_path, monkeypatch)
    installed = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "7.1.4" if name == "slide2vec" else installed(name),
    )
    study_dir = tmp_path / "study"

    assert study.main(["extract", "--study-dir", str(study_dir), "--pairs", "TITAN"]) == 0

    record = json.loads((study_dir / "extraction.json").read_text())
    assert record["TITAN"]["status"] == "done"
    assert record["TITAN"]["slide2vec_version"] == "7.1.4"


#: The record ``extract`` wrote when it finished a pair.
DONE_RECORD = {
    "status": "done",
    "started": "2026-10-20 09:00:00",
    "finished": "2026-10-20 13:00:00",
    "slide2vec_version": "7.1.4",
}


def _complete_pair_on_a_cpu_host(tmp_path: Path, monkeypatch, model: str) -> Path:
    """A study dir whose ``model`` re-run is finished, seen from a host with no free GPU.

    ``extraction.json`` holds the record of the finished run. Waiting for the GPU,
    launching slide2vec and a slide2vec other than the recorded one are all made visible,
    so a pass with nothing to do can be checked to touch none of them.
    """
    import json
    import subprocess
    import time

    import slide_vs_meanpool as study

    study_dir = tmp_path / "study"
    _write_run(study_dir / "runs" / model)
    (study_dir / "extraction.json").write_text(json.dumps({model: DONE_RECORD}, indent=2) + "\n")
    monkeypatch.setattr(study, "_pcabiop_rows", lambda: ROWS)
    monkeypatch.setattr(study, "_gpu_free_mib", lambda: 0.0)  # what a CPU host reports
    monkeypatch.setattr(study, "_installed_version", lambda name: "9.9.9")

    def no_sleep(seconds):
        raise AssertionError(f"waited {seconds}s for a GPU that a finished pair does not need")

    def no_launch(*args, **kwargs):
        raise AssertionError("launched slide2vec for a finished pair")

    monkeypatch.setattr(time, "sleep", no_sleep)
    monkeypatch.setattr(subprocess, "run", no_launch)
    return study_dir


def test_a_finished_optional_pair_keeps_its_record_when_the_gpu_is_busy(tmp_path, monkeypatch):
    import json

    import slide_vs_meanpool as study

    assert not study.PAIRS_BY_MODEL["PRISM"].required
    study_dir = _complete_pair_on_a_cpu_host(tmp_path, monkeypatch, "PRISM")

    assert study.main(["extract", "--study-dir", str(study_dir), "--pairs", "PRISM"]) == 0

    # Not "skipped (GPU busy)": the pair ran, and its record says when and with what.
    assert json.loads((study_dir / "extraction.json").read_text()) == {"PRISM": DONE_RECORD}


def test_a_finished_required_pair_neither_waits_for_the_gpu_nor_restamps_its_record(
    tmp_path, monkeypatch
):
    import json

    import slide_vs_meanpool as study

    assert study.PAIRS_BY_MODEL["TITAN"].required
    study_dir = _complete_pair_on_a_cpu_host(tmp_path, monkeypatch, "TITAN")

    assert study.main(["extract", "--study-dir", str(study_dir), "--pairs", "TITAN"]) == 0

    # The times and the slide2vec version stay those of the run that embedded the slides.
    assert json.loads((study_dir / "extraction.json").read_text()) == {"TITAN": DONE_RECORD}


def test_a_pair_is_complete_only_when_both_its_cohorts_are(tmp_path):
    import slide_vs_meanpool as study

    pair, study_dir = study.PAIRS_BY_MODEL["TITAN"], tmp_path / "study"
    assert not study.pair_complete(pair, ROWS, study_dir=study_dir)

    run_root = study.pair_run_root(study_dir, pair)
    _write_run(run_root)
    assert study.pair_complete(pair, ROWS, study_dir=study_dir)

    (run_root / "panda" / "slide_embeddings" / "p2.npy").unlink()
    assert not study.pair_complete(pair, ROWS, study_dir=study_dir)
    # A smoke run's limit counts only the first slides of each cohort.
    assert study.pair_complete(pair, ROWS, study_dir=study_dir, limit=1)


# --- paired differences ---------------------------------------------------------------

#: Cramer's V schedule and chance level of a three-split toy probe sweep.
TOY_V = [0.0, 0.5, 1.0]
CHANCE = 0.5
FLAT = [0.9, 0.9, 0.9]  # no degradation: nIPD 0
FALL = [0.9, 0.7, 0.5]  # linear fall from baseline to chance at V=1: nIPD -0.5


def _per_sample(models: dict[str, np.ndarray]) -> pd.DataFrame:
    """In-domain per-sample CRoMa for each arm on a balanced 2x2 of 40 slides."""
    n = len(next(iter(models.values())))
    label = np.tile(["benign", "cancer"], n // 2)
    confounder = np.repeat(["radboud", "karolinska"], n // 2)
    ids = [f"s{i:02d}" for i in range(n)]
    frames = [
        pd.DataFrame(
            {
                "model": model,
                "sample_id": ids,
                "group_id": ids,
                "label": label,
                "confounder": confounder,
                "croma": values,
            }
        )
        for model, values in models.items()
    ]
    return pd.concat(frames, ignore_index=True)


def _cell(id_columns, ood_columns) -> dict:
    """An APD cell: (n_splits, n_replicates) accuracy matrices, one column per replicate."""
    return {
        "id_test_accuracies": np.asarray(id_columns, dtype=float).T.tolist(),
        "ood_test_accuracies": np.asarray(ood_columns, dtype=float).T.tolist(),
    }


def _differences(per_sample, cells, arms, seed=0):
    import slide_vs_meanpool as study

    return study.paired_differences(
        arms, per_sample, cells, cramers_v=TOY_V, chance=CHANCE, n_boot=200, seed=seed
    ).set_index(["pair", "statistic"])


def test_paired_croma_differences_share_one_bootstrap_draw():
    slide = np.tile([0.05, 0.15, 0.25, 0.35], 10)
    shifted = slide - 0.10  # every slide's margin shrinks by exactly 0.10 when mean-pooled
    per_sample = _per_sample({"A": slide, "A-meanpool": shifted, "B": slide, "B-meanpool": slide})
    cells = {arm: _cell([FLAT], [FLAT]) for arm in ("A", "A-meanpool", "B", "B-meanpool")}

    diff = _differences(per_sample, cells, [("A", "A", "A-meanpool"), ("B", "B", "B-meanpool")])

    # A constant shift is a constant shift on every shared draw: a zero-width interval.
    for statistic in ("median_croma", "ltm"):
        row = diff.loc[("A", statistic)]
        assert row["point"] == pytest.approx(0.10)
        assert (row["lo"], row["hi"]) == (pytest.approx(0.10), pytest.approx(0.10))
    # F(0): no slide arm margin is <= 0, a quarter of the mean-pooled ones are.
    assert diff.loc[("A", "f0"), "point"] == pytest.approx(-0.25)
    assert diff.loc[("A", "f0"), "lo"] <= -0.25 <= diff.loc[("A", "f0"), "hi"]
    # Identical arms differ by nothing on any draw.
    for statistic in ("median_croma", "f0", "ltm"):
        row = diff.loc[("B", statistic)]
        assert (row["point"], row["lo"], row["hi"]) == (0.0, 0.0, 0.0)


def test_nipd_differences_carry_their_spread_over_paired_probe_replicates():
    per_sample = _per_sample({"A": np.linspace(-0.5, 0.5, 40), "A-meanpool": np.zeros(40)})
    cells = {
        "A": _cell(id_columns=[FLAT, FLAT], ood_columns=[FLAT, FLAT]),
        # Replicate 1 falls to chance in domain; out of domain both replicates fall.
        "A-meanpool": _cell(id_columns=[FLAT, FALL], ood_columns=[FALL, FALL]),
    }

    diff = _differences(per_sample, cells, [("A", "A", "A-meanpool")])

    # In domain the mean-pooled curve is the average of a flat and a falling replicate,
    # g(V) = -V/2, whose area is -0.25; the per-replicate differences are 0 and +0.5.
    nipd_id = diff.loc[("A", "nipd_id")]
    assert nipd_id["point"] == pytest.approx(0.25)
    assert nipd_id["spread"] == pytest.approx(np.std([0.0, 0.5], ddof=1))
    assert nipd_id["n"] == 2
    nipd_ood = diff.loc[("A", "nipd_ood")]
    assert (nipd_ood["point"], nipd_ood["spread"]) == (pytest.approx(0.5), pytest.approx(0.0))


#: Two probe replicates whose balanced baselines straddle chance: the first has no nIPD of
#: its own, but their mean baseline (0.60) is above chance, so the cell's nIPD exists.
BELOW_CHANCE = [0.49, 0.49, 0.49]
ABOVE_CHANCE = [0.71, 0.71, 0.71]


def test_a_replicate_at_or_below_chance_leaves_the_nipd_comparison_standing():
    per_sample = _per_sample({"A": np.linspace(-0.5, 0.5, 40), "A-meanpool": np.zeros(40)})
    cell = _cell(id_columns=[BELOW_CHANCE, ABOVE_CHANCE], ood_columns=[FLAT, FLAT])
    cells = {"A": cell, "A-meanpool": cell}

    diff = _differences(per_sample, cells, [("A", "A", "A-meanpool")])

    # Identical arms: the aggregate difference is defined and zero. One replicate is left
    # to measure the spread over, and one difference has no sample standard deviation.
    nipd_id = diff.loc[("A", "nipd_id")]
    assert nipd_id["point"] == 0.0
    assert nipd_id["n"] == 1
    assert np.isnan(nipd_id["spread"])
    nipd_ood = diff.loc[("A", "nipd_ood")]
    assert (nipd_ood["point"], nipd_ood["spread"], nipd_ood["n"]) == (0.0, 0.0, 2)


def test_the_nipd_spread_is_over_the_replicates_both_arms_define():
    per_sample = _per_sample({"A": np.linspace(-0.5, 0.5, 40), "A-meanpool": np.zeros(40)})
    cells = {
        "A": _cell(id_columns=[FLAT, FLAT, FALL], ood_columns=[FLAT, FLAT, FLAT]),
        # The mean-pooled arm's first replicate is below chance, so that replicate drops
        # out of the pair's spread, although the slide arm's first replicate is defined.
        "A-meanpool": _cell(
            id_columns=[BELOW_CHANCE, ABOVE_CHANCE, FLAT], ood_columns=[FLAT, FLAT, FLAT]
        ),
    }

    diff = _differences(per_sample, cells, [("A", "A", "A-meanpool")])

    # The point estimate still averages all three replicates in each arm: the slide arm's
    # mean curve is 0.9, 0.8333, 0.7667 (g = 0, -1/6, -1/3, area -1/6) and the mean-pooled
    # arm's is flat at 0.7 (area 0). The spread is over replicates 2 and 3 only.
    nipd_id = diff.loc[("A", "nipd_id")]
    assert nipd_id["point"] == pytest.approx(-1.0 / 6.0)
    assert nipd_id["spread"] == pytest.approx(np.std([0.0, -0.5], ddof=1))
    assert nipd_id["n"] == 2


def test_paired_differences_are_reproducible_from_their_seed():
    per_sample = _per_sample(
        {"A": np.linspace(-0.5, 0.5, 40), "A-meanpool": np.linspace(-0.6, 0.6, 40)}
    )
    cells = {arm: _cell([FLAT], [FLAT]) for arm in ("A", "A-meanpool")}
    arms = [("A", "A", "A-meanpool")]

    first = _differences(per_sample, cells, arms, seed=7)
    pd.testing.assert_frame_equal(first, _differences(per_sample, cells, arms, seed=7))


# --- the published slide panel is untouched ------------------------------------------


def test_the_published_slide_payloads_carry_exactly_the_slide_panel():
    """Study arms live in the study directory; the results payload and the site built
    from it must never list one."""
    import slide_vs_meanpool as study
    from loaders import slide_panel

    panel = set(slide_panel())
    assert set(pd.read_csv(ROOT / "results" / "pcabiop.csv")["model"]) == panel
    nipd = pd.read_csv(ROOT / "results" / "nipd.csv")
    assert set(nipd.loc[nipd["cohort"] == "pcabiop", "model"]) == panel

    suffixes = (study.MEANPOOL_SUFFIX, study.RERUN_SUFFIX)
    for path in sorted((ROOT / "results").iterdir()):
        text = path.read_text(encoding="utf-8")
        for pair in study.PAIRS:
            for suffix in suffixes:
                assert pair.slide_model + suffix not in text, f"{path.name} lists a study arm"
