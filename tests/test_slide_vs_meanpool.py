"""Slide encoders versus mean-pooling of their own tile encoders on PCaBiop.

The study re-runs each slide encoder with its tile embeddings saved, so the slide arm and
the mean-pooled arm come from one tiling of the same slides. These tests cover what the
study and its paper table own: the re-extraction config, the reduction to two arms, the
reproduction check that decides which slide arm is used, the paired differences, the
rendered table, and the fence that keeps the study arms out of the published slide panel.
None of them needs a GPU, torch or the real cohort.
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

    apd, cells = study.run_downstream(models, sources=sources, study_dir=tmp_path / "study", jobs=1)

    np.testing.assert_array_equal(seen["TITAN"], np.load(published / "TITAN.npy"))
    np.testing.assert_array_equal(seen["TITAN-meanpool"], np.full((3, 2), 7.0))
    assert set(cells) == set(models) and apd["model"].tolist() == models
    assert layout.OUTPUT_ROOT == output_root


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


def test_paired_differences_are_reproducible_from_their_seed():
    per_sample = _per_sample(
        {"A": np.linspace(-0.5, 0.5, 40), "A-meanpool": np.linspace(-0.6, 0.6, 40)}
    )
    cells = {arm: _cell([FLAT], [FLAT]) for arm in ("A", "A-meanpool")}
    arms = [("A", "A", "A-meanpool")]

    first = _differences(per_sample, cells, arms, seed=7)
    pd.testing.assert_frame_equal(first, _differences(per_sample, cells, arms, seed=7))


# --- the paper table ----------------------------------------------------------------

REPRO = ROOT / "scripts" / "repro"


def _repro():
    if str(REPRO) not in sys.path:
        sys.path.insert(0, str(REPRO))
    import _slide_vs_meanpool as basis
    import generate_slide_vs_meanpool_table as table

    return basis, table


def _write_study(study_dir: Path, pairs=("TITAN", "MOOZY", "PRISM2")) -> Path:
    """Study artifacts as ``evaluate`` writes them, with hand-picked values per pair."""
    import json

    study_dir.mkdir(parents=True, exist_ok=True)
    summary, differences, reproduction = [], [], {}
    for i, pair in enumerate(pairs):
        reproduced = pair != "TITAN"
        slide_arm = pair if reproduced else f"{pair}-rerun"
        for role, model, values in (
            ("slide", slide_arm, (-0.304, 0.895, -0.6017, -0.1886, -0.1258)),
            ("meanpool", f"{pair}-meanpool", (-0.204, 0.795, -0.5017, -0.0886, -0.0258)),
        ):
            croma, f0, ltm, nipd_id, nipd_ood = values
            summary.append(
                dict(
                    pair=pair,
                    arm=role,
                    model=model,
                    bio_knn_bacc=[0.915, 0.971, 0.975][i],
                    croma=croma,
                    croma_f0=f0,
                    croma_ltm_alpha=ltm,
                    nipd_id=nipd_id,
                    nipd_ood=nipd_ood,
                )
            )
        for statistic, point in (("median_croma", -0.1), ("f0", 0.2), ("ltm", -0.1)):
            differences.append(
                dict(
                    pair=pair,
                    statistic=statistic,
                    point=point,
                    lo=point - 0.05,
                    hi=point + 0.04,
                    spread=np.nan,
                    n=2000,
                )
            )
        for statistic in ("nipd_id", "nipd_ood"):
            differences.append(
                dict(
                    pair=pair,
                    statistic=statistic,
                    point=-0.1,
                    lo=np.nan,
                    hi=np.nan,
                    spread=0.012,
                    n=20,
                )
            )
        reproduction[pair] = dict(
            slide_arm=slide_arm,
            reproduced=reproduced,
            n_slides=1324,
            n_reproduced=1324 if reproduced else 17,
            min_cosine=0.99991 if reproduced else 0.4123,
        )
    pd.DataFrame(summary).to_csv(study_dir / "summary.csv", index=False)
    pd.DataFrame(differences).to_csv(study_dir / "differences.csv", index=False)
    provenance = {
        "pairs": {p: {"reproduction": reproduction[p]} for p in pairs},
        "bootstrap": {"n_boot": 2000, "level": 0.95},
        "downstream": {"iterations": 20},
        "representation": {"protocol": "k-star", "croma_headline_m": 5},
    }
    (study_dir / "provenance.json").write_text(json.dumps(provenance))
    return study_dir


def test_the_table_renders_every_declared_pair_with_both_arms_and_paired_cis(tmp_path):
    basis, table = _repro()

    tex = table.build(basis.load(_write_study(tmp_path / "study")))

    assert r"\label{tab:slide-vs-meanpool}" in tex
    for pair in basis.RENDERED_PAIRS:
        assert rf"\multirow{{3}}{{*}}{{{pair}}}" in tex
    # The TITAN block: its re-run slide arm, the mean-pooled arm, then slide - mean-pool.
    assert r"& slide & $0.915$ & $-0.30$ & $0.895$ & $-0.60$ & $-18.9$ & $-12.6$ \\" in tex
    assert r"& mean-pool & $0.915$ & $-0.20$ & $0.795$ & $-0.50$ & $-8.9$ & $-2.6$ \\" in tex
    assert (
        r"& $\Delta$ & & $-0.10$\,{\scriptsize$[-0.15, -0.06]$} "
        r"& $0.200$\,{\scriptsize$[0.150, 0.240]$} & $-0.10$\,{\scriptsize$[-0.15, -0.06]$} "
        r"& $-10.0$\,{\scriptsize$\pm1.2$} & $-10.0$\,{\scriptsize$\pm1.2$} \\"
    ) in tex


def test_the_caption_names_rendered_and_missing_pairs_and_the_reproduction_result(tmp_path):
    basis, table = _repro()
    import slide_vs_meanpool as study

    tex = table.build(basis.load(_write_study(tmp_path / "study")))
    caption = tex[tex.index(r"\caption{") :]

    declared = [pair.slide_model for pair in study.PAIRS]
    missing = [p for p in declared if p not in basis.RENDERED_PAIRS]
    assert missing, "every declared pair is rendered, so the missing-pair clause asserts nothing"
    for pair in [*basis.RENDERED_PAIRS, *missing]:
        assert pair in caption
    assert "mean of the same" in caption  # how the mean-pooled arm is built
    # TITAN's re-run did not reproduce, so its re-run slide embeddings are the slide arm.
    assert "TITAN: $17$ of $1{,}324$ slides" in caption
    assert "minimum cosine $0.412$" in caption


def test_a_declared_pair_without_study_output_fails_the_build(tmp_path):
    basis, _ = _repro()
    study_dir = _write_study(tmp_path / "study", pairs=("TITAN", "MOOZY"))

    with pytest.raises(basis.CaptionClaimError, match="PRISM2"):
        basis.load(study_dir)


def test_the_table_and_macros_rebuild_byte_for_byte(tmp_path):
    basis, table = _repro()
    study_dir = _write_study(tmp_path / "study")
    out = tmp_path / "slide_vs_meanpool.tex"

    table.main(["--study-dir", str(study_dir), "--out", str(out)])
    first = out.read_bytes()
    table.main(["--study-dir", str(study_dir), "--out", str(out)])
    assert out.read_bytes() == first

    macros = basis.macros(basis.load(study_dir))
    assert macros == basis.macros(basis.load(study_dir))
    assert r"\newcommand{\SlideMeanpoolTitanCromaDiff}{$-0.10$}" in macros
    assert r"\newcommand{\SlideMeanpoolPrismTwoNipdOodDiffSd}{$1.2$}" in macros
    assert r"\newcommand{\SlideMeanpoolMoozyFzeroDiffCI}{$[0.150, 0.240]$}" in macros


def test_the_paper_values_carry_the_paired_differences_or_fail_on_a_missing_pair(tmp_path):
    basis, _ = _repro()
    import generate_paper_values as gpv

    # Not run at all: the study is skipped by name, like every other study block.
    assert gpv._slide_vs_meanpool_macros(tmp_path) == []

    study_dir = _write_study(tmp_path / basis.STUDY_REL)
    lines = gpv._slide_vs_meanpool_macros(tmp_path)
    assert lines == basis.macros(basis.load(study_dir))

    _write_study(tmp_path / basis.STUDY_REL, pairs=("TITAN", "MOOZY"))
    with pytest.raises(basis.CaptionClaimError, match="PRISM2"):
        gpv._slide_vs_meanpool_macros(tmp_path)


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


# --- the one-command reproduction ------------------------------------------------------


def test_the_full_reproduction_reduces_then_evaluates_the_study():
    # The studies run in list order, and build_paper.py runs after all of them.
    _repro()
    import reproduce_faithful

    commands = [" ".join([script, *extra]) for _, script, extra in reproduce_faithful.STUDIES]

    def position(needle: str) -> int:
        return next(i for i, command in enumerate(commands) if needle in command)

    assert position("slide_vs_meanpool.py reduce") < position(
        "slide_vs_meanpool.py evaluate --recompute"
    )
    # Both steps take the study-root default for the arms, so evaluate reads what reduce
    # wrote and neither writes beside the published embeddings.
    study_commands = [c for c in commands if "slide_vs_meanpool.py" in c]
    assert len(study_commands) == 2
    assert not any("--embeddings-dir" in c for c in study_commands)
