"""The probe-record study driver: per-sample predictions of the shortcut sweep, on disk.

``scripts/studies/apd/probe_records.py`` re-runs the APD study's sweep with
``croma.downstream.probe_sweep_with_predictions`` and writes one compressed record per
cohort x encoder x training-tile level. What it owns is the cell list, the record format,
the proof that a full-tile record is the published APD cell, and the tile keys that let a
record join a run's ``per_sample_metrics.csv``. The tests below run it on a toy cohort laid
out like Camelyon -- PathoROB's own schedule, three rows per slide -- so a sweep takes
seconds; one more reads the real PCaBiop records when a checkout has them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "scripts" / "studies" / "apd"
if str(STUDY) not in sys.path:
    sys.path.insert(0, str(STUDY))

TOY_ROWS_PER_SLIDE = 3
TOY_SLIDES_PER_CELL = 17
TOY = dict(
    src="toy-tileset",
    benchmark="toy",
    metadata="data/toy.csv",
    split_key="camelyon",
    centers_id=["RUMC", "UMCU"],
    centers_ood=["CWZ"],
    biological_classes=["normal", "tumor"],
    num_splits=8,
    num_slides_per_category=TOY_SLIDES_PER_CELL,
    num_patches_per_slide=TOY_ROWS_PER_SLIDE,
)


def _toy_cohort(seed: int = 0):
    """A Camelyon-shaped cohort and its tile keys, ID rows in cell order, slide by slide."""
    from loaders import Cohort

    rng = np.random.default_rng(seed)
    rows, confounders, labels, keys = [], [], [], []
    for confounder, centre in enumerate(TOY["centers_id"]):
        for label, name in enumerate(TOY["biological_classes"]):
            for slide in range(TOY_SLIDES_PER_CELL):
                group = f"{centre}-{name}-{slide:02d}"
                for tile in range(TOY_ROWS_PER_SLIDE):
                    rows.append(
                        rng.normal(scale=0.9, size=4) + [1.5 * label, 1.5 * confounder, 0, 0]
                    )
                    confounders.append(confounder)
                    labels.append(label)
                    keys.append((f"{group}__{tile}", f"/tiles/{group}__{tile}.png", group))
    ood_x = rng.normal(scale=0.9, size=(12, 4)) + np.repeat(
        [[0, 3.0, 0, 0], [1.5, 3.0, 0, 0]], 6, 0
    )
    ood_y = np.repeat([0, 1], 6)

    cohort = Cohort(
        embeddings=np.array(rows),
        confounders=np.array(confounders),
        labels=np.array(labels),
        ood_embeddings=ood_x,
        ood_labels=ood_y,
        group_ids=[group for *_, group in keys],
    )
    id_keys = pd.DataFrame(keys, columns=["sample_id", "image_path", "group_id"])
    id_keys["label"] = [TOY["biological_classes"][label] for label in labels]
    id_keys["confounder"] = [TOY["centers_id"][confounder] for confounder in confounders]
    ood_keys = pd.DataFrame(
        {
            "sample_id": [f"ood-{row}" for row in range(12)],
            "image_path": [f"/tiles/ood-{row}.png" for row in range(12)],
            "group_id": [f"ood-{row // 3}" for row in range(12)],
            "label": [TOY["biological_classes"][label] for label in ood_y],
            "confounder": "CWZ",
        }
    )
    return cohort, id_keys, ood_keys


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Register the toy cohort, point both studies at it, and publish its APD cell."""
    import apd_experiment
    import loaders
    import probe_records

    monkeypatch.setitem(loaders.DATASETS, "toy", TOY)
    cohort, id_keys, ood_keys = _toy_cohort()
    monkeypatch.setattr(apd_experiment, "load_data", lambda model, dataset: cohort)
    monkeypatch.setattr(
        probe_records, "load_cohort", lambda dataset, model: (cohort, id_keys, ood_keys)
    )
    apd_dir = tmp_path / "apd"
    cell = apd_dir / "toy" / "Toy-FM.json"
    cell.parent.mkdir(parents=True)
    cell.write_text(json.dumps(apd_experiment.compute("Toy-FM", "toy", iterations=2)))
    return dict(
        apd_dir=apd_dir,
        out_dir=tmp_path / "records",
        cohort=cohort,
        id_keys=id_keys,
        ood_keys=ood_keys,
    )


def test_the_runs_table_is_the_one_the_study_specifies() -> None:
    # Which tile levels each cohort is recorded at is the study's design, so it is pinned
    # here against that design rather than read back from the code. Tolkach-ESCA has no
    # full-tile level: its representation cohort and probe pool are different rows, so a
    # per-sample join is impossible there and its full-tile accuracies are the APD cells.
    import probe_records

    assert probe_records.RUNS == {
        "camelyon": (300, 30, 3),
        "tcga_4x4": (30, 3, 1),
        "pcabiop": (1,),
        "tolkach": (10, 1),
    }


def test_the_cell_list_is_the_apd_studys_roster_at_every_level_slowest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The roster is never written down here: it is whatever the APD study computes for each
    # cohort, so an encoder joining the panel joins the records too. Full-tile Camelyon
    # dominates the compute, so it is queued first.
    import apd_experiment
    import probe_records

    rosters = {
        "camelyon": ["A-FM", "B-FM"],
        "tcga_4x4": ["A-FM", "B-FM"],
        "tolkach": ["A-FM", "B-FM"],
        "pcabiop": ["Slide-FM"],
    }
    monkeypatch.setattr(apd_experiment, "model_list", lambda dataset: rosters[dataset])

    cells = probe_records.cells()

    expected = {
        (cohort, model, tiles)
        for cohort, levels in probe_records.RUNS.items()
        for tiles in levels
        for model in rosters[cohort]
    }
    assert {(cell.cohort, cell.model, cell.tiles) for cell in cells} == expected
    assert len(cells) == len(expected)
    assert [(cell.cohort, cell.tiles) for cell in cells[:2]] == [("camelyon", 300)] * 2
    assert {(cell.cohort, cell.tiles) for cell in cells[-2:]} <= {
        ("tolkach", 1),
        ("tcga_4x4", 1),
    }


def test_a_full_tile_record_is_the_published_apd_cell_with_its_predictions(toy) -> None:
    # At full tiles the record re-runs the APD study's own sweep, so its accuracies are the
    # APD cell's, and the record says how close they came.
    import probe_records

    cell = probe_records.Cell("toy", "Toy-FM", TOY_ROWS_PER_SLIDE)

    tag = probe_records.run_cell(cell, out_dir=toy["out_dir"], iterations=2, apd_dir=toy["apd_dir"])

    assert tag == "done"
    record = probe_records.read_record(cell.path(toy["out_dir"]))
    published = json.loads((toy["apd_dir"] / "toy" / "Toy-FM.json").read_text())
    assert np.array_equal(record["accuracies_in_domain"], published["id_test_accuracies"])
    assert np.array_equal(record["accuracies_out_of_domain"], published["ood_test_accuracies"])
    assert record["meta"]["full_tiles"] is True
    assert record["meta"]["apd_reproduction_gap"] == {"in_domain": 0.0, "out_of_domain": 0.0}
    assert record["margins_in_domain"].shape[:3] == (8, 2, record["held_out"].shape[1])
    assert record["margins_out_of_domain"].shape[:3] == (8, 2, 12)
    assert list(record["id_sample_id"]) == list(toy["id_keys"]["sample_id"])
    assert list(record["ood_image_path"]) == list(toy["ood_keys"]["image_path"])


def test_a_full_tile_sweep_that_misses_its_apd_cell_writes_no_record(toy) -> None:
    # The record's claim is "these are the predictions behind the published cell". A sweep
    # that lands 1e-6 away is some other sweep, so the cell fails instead of being stored.
    import probe_records

    path = toy["apd_dir"] / "toy" / "Toy-FM.json"
    published = json.loads(path.read_text())
    published["ood_test_accuracies"][3][1] += 1e-6
    path.write_text(json.dumps(published))
    cell = probe_records.Cell("toy", "Toy-FM", TOY_ROWS_PER_SLIDE)

    with pytest.raises(probe_records.ReproductionError, match="out_of_domain.*1e-09"):
        probe_records.run_cell(cell, out_dir=toy["out_dir"], iterations=2, apd_dir=toy["apd_dir"])

    assert not cell.path(toy["out_dir"]).exists()
    assert not list(toy["out_dir"].rglob("*.tmp"))


def test_a_reduced_tile_record_holds_out_the_full_tile_records_rows(toy) -> None:
    # Fewer tiles per training slide, same held-out rows replicate by replicate: the two
    # records are graded on the same tiles, so their margins compare row for row.
    import probe_records

    full = probe_records.Cell("toy", "Toy-FM", TOY_ROWS_PER_SLIDE)
    thin = probe_records.Cell("toy", "Toy-FM", 1)
    for cell in (full, thin):
        probe_records.run_cell(cell, out_dir=toy["out_dir"], iterations=2, apd_dir=toy["apd_dir"])

    full_record = probe_records.read_record(full.path(toy["out_dir"]))
    thin_record = probe_records.read_record(thin.path(toy["out_dir"]))
    assert thin.path(toy["out_dir"]).parent.name == "tiles-1"
    assert thin_record["meta"]["full_tiles"] is False
    assert thin_record["meta"]["apd_reproduction_gap"] is None
    assert np.array_equal(thin_record["held_out"], full_record["held_out"])
    assert not np.array_equal(thin_record["margins_in_domain"], full_record["margins_in_domain"])


def test_a_valid_record_on_disk_is_skipped_and_an_invalid_one_is_redone(
    toy, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resumable: a re-launch recomputes nothing it already has. "Has" means a record that
    # still checks out -- one whose margins no longer re-score to its accuracies is redone.
    import probe_records

    cell = probe_records.Cell("toy", "Toy-FM", TOY_ROWS_PER_SLIDE)
    common = dict(out_dir=toy["out_dir"], iterations=2, apd_dir=toy["apd_dir"])
    probe_records.run_cell(cell, **common)
    path = cell.path(toy["out_dir"])
    record = probe_records.read_record(path)
    margins = record["margins_in_domain"]

    compute = probe_records.compute
    monkeypatch.setattr(
        probe_records, "compute", lambda *a, **k: pytest.fail("a valid record was recomputed")
    )
    assert probe_records.run_cell(cell, **common) == "skip"

    record["margins_in_domain"] = -margins
    record["meta"] = np.array(json.dumps(record["meta"]))
    probe_records.write_record(path, record)
    monkeypatch.setattr(probe_records, "compute", compute)

    assert probe_records.run_cell(cell, **common) == "done"
    assert np.array_equal(probe_records.read_record(path)["margins_in_domain"], margins)


def test_a_record_is_published_atomically(toy, monkeypatch: pytest.MonkeyPatch) -> None:
    # Interrupted between write and rename, a cell leaves neither a record nor a temporary
    # behind, so the resume logic cannot mistake half a record for a finished one.
    import probe_records

    cell = probe_records.Cell("toy", "Toy-FM", TOY_ROWS_PER_SLIDE)

    def interrupted_replace(source, destination):
        raise OSError("simulated preemption")

    monkeypatch.setattr(probe_records.os, "replace", interrupted_replace)

    with pytest.raises(OSError, match="preemption"):
        probe_records.run_cell(cell, out_dir=toy["out_dir"], iterations=2, apd_dir=toy["apd_dir"])

    assert not cell.path(toy["out_dir"]).exists()
    assert not list(toy["out_dir"].rglob("*.tmp"))


def _per_sample_for(keys: pd.DataFrame, model: str, seed: int = 0) -> pd.DataFrame:
    """A run's per-sample table over ``keys``: shuffled, other encoders mixed in, its own
    class names -- the shape a real ``per_sample_metrics.csv`` has relative to a record."""
    frame = keys[["sample_id", "group_id", "confounder"]].copy()
    frame["label"] = keys["label"].map({"normal": "0", "tumor": "1"})
    frame["croma_m5"] = np.linspace(-1, 1, len(frame))
    other = frame.assign(model="Other-FM", croma_m5=0.0)
    frame = pd.concat([frame.assign(model=model), other], ignore_index=True)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_held_out_rows_join_the_runs_per_sample_metrics_by_tile_key(toy) -> None:
    # Nothing about a per-sample table's row order matches a record's: it is another view
    # of the tileset, written by another program, holding every encoder. The join must
    # therefore go by tile key, and find every held-out row exactly once.
    import probe_records

    cell = probe_records.Cell("toy", "Toy-FM", TOY_ROWS_PER_SLIDE)
    probe_records.run_cell(cell, out_dir=toy["out_dir"], iterations=2, apd_dir=toy["apd_dir"])
    record = probe_records.read_record(cell.path(toy["out_dir"]))
    per_sample = _per_sample_for(toy["id_keys"], "Toy-FM")

    joined = probe_records.join_per_sample(record, per_sample)

    held = np.unique(record["held_out"])
    assert sorted(joined["row"]) == held.tolist()
    expected = per_sample[per_sample["model"] == "Toy-FM"].set_index("sample_id")["croma_m5"]
    for row, sample_id, croma in zip(joined["row"], joined["sample_id"], joined["croma_m5"]):
        assert record["id_sample_id"][row] == sample_id
        assert croma == expected[sample_id]


def test_a_held_out_row_missing_from_the_per_sample_metrics_is_an_error(toy) -> None:
    import probe_records

    cell = probe_records.Cell("toy", "Toy-FM", TOY_ROWS_PER_SLIDE)
    probe_records.run_cell(cell, out_dir=toy["out_dir"], iterations=2, apd_dir=toy["apd_dir"])
    record = probe_records.read_record(cell.path(toy["out_dir"]))
    per_sample = _per_sample_for(toy["id_keys"], "Toy-FM")
    dropped = record["id_sample_id"][record["held_out"][1, 0]]

    with pytest.raises(ValueError, match="1 held-out row"):
        probe_records.join_per_sample(record, per_sample[per_sample["sample_id"] != dropped])


def _toy_tileset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_edit=None):
    """Write the toy cohort's PathoROB-style metadata and its tileset manifest to disk."""
    import layout
    import loaders

    metadata = pd.DataFrame(
        {
            "subset": ["ID", "OOD", "ID", "ID", "OOD"],
            "slide_id": ["s1", "s9", "s1", "s2", "s9"],
            "patch_id": ["p1", "p1", "p2", "p1", "p2"],
            "biological_class": ["normal", "tumor", "normal", "tumor", "normal"],
            "medical_center": ["RUMC", "CWZ", "RUMC", "UMCU", "CWZ"],
        }
    )
    manifest = pd.DataFrame(
        {
            "sample_id": [f"{s}__{p}" for s, p in zip(metadata.slide_id, metadata.patch_id)],
            "image_path": [
                f"/t/{s}__{p}.png" for s, p in zip(metadata.slide_id, metadata.patch_id)
            ],
            "label": metadata.biological_class,
            "confounder": metadata.medical_center,
            "group_id": metadata.slide_id,
        }
    )
    if manifest_edit:
        manifest_edit(manifest)
    metadata.to_csv(tmp_path / "toy.csv", index=False)
    monkeypatch.setattr(layout, "OUTPUT_ROOT", tmp_path / "output")
    layout.tileset_manifest("toy-tileset").parent.mkdir(parents=True)
    manifest.to_csv(layout.tileset_manifest("toy-tileset"), index=False)
    monkeypatch.setitem(loaders.DATASETS, "toy", {**TOY, "metadata": str(tmp_path / "toy.csv")})
    return manifest


def test_tile_keys_come_off_the_tileset_manifest_in_the_cohorts_row_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import loaders

    manifest = _toy_tileset(tmp_path, monkeypatch)

    id_keys, ood_keys = loaders.cohort_tile_keys("toy")

    assert list(id_keys.columns) == list(loaders.TILE_KEY_COLUMNS)
    assert id_keys["sample_id"].tolist() == manifest["sample_id"][[0, 2, 3]].tolist()
    assert ood_keys["sample_id"].tolist() == manifest["sample_id"][[1, 4]].tolist()


def test_a_tileset_manifest_out_of_step_with_the_metadata_yields_no_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Row count is all `load_data` checks. Two tiles of one slide swapped between the
    # manifest and the metadata pass it, and would key every margin to its neighbour.
    import loaders

    def swap_two_tiles(manifest):
        manifest.loc[[0, 2]] = manifest.loc[[2, 0]].to_numpy()

    _toy_tileset(tmp_path, monkeypatch, swap_two_tiles)

    with pytest.raises(ValueError, match="row 0 is not metadata row 0"):
        loaders.cohort_tile_keys("toy")


PCABIOP_RECORDS = ROOT / "output" / "studies" / "probe-records" / "pcabiop" / "tiles-1"


@pytest.mark.skipif(
    not any(PCABIOP_RECORDS.glob("*.npz")),
    reason="the PCaBiop probe records are ignored study output, absent from this checkout",
)
def test_every_pcabiop_record_joins_the_panda_runs_per_sample_metrics() -> None:
    # The one real cohort the join is exercised on: PCaBiop's ID rows are the `panda`
    # run's evaluated slides, so every held-out slide of every replicate must find its row.
    import probe_records

    per_sample = pd.read_csv(
        probe_records.per_sample_path("pcabiop"), dtype=str, keep_default_na=False
    )
    for path in sorted(PCABIOP_RECORDS.glob("*.npz")):
        record = probe_records.read_record(path)

        joined = probe_records.join_per_sample(record, per_sample)

        assert len(joined) == len(np.unique(record["held_out"])), path.name
        assert (joined["model"] == record["meta"]["model"]).all()
