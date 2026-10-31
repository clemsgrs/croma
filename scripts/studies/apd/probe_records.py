"""Per-sample predictions of the confounder-biased probe sweep, at full and fewer tiles.

The APD study (``apd_experiment.py``) keeps one accuracy matrix per probe sweep and
discards the probes. Two questions need more than that: whether the held-out rows a
shortcut-trained probe gets wrong are the ones with a low CRoMa margin, and whether the
association between CRoMa and shortcut susceptibility survives a low-data regime. This
driver re-runs the APD study's own sweep -- its loaders, schedules, seed, validation rule
and slide arrangement, through ``apd_experiment.sweep_arguments`` -- with
``croma.downstream.probe_sweep_with_predictions``, and writes one compressed record per
cohort x encoder x training-tile level:

    output/studies/probe-records/<cohort>/tiles-<n>/<encoder>.npz

where ``n`` is how many rows each training slide contributes (``RUNS``). Reducing and
rendering the records is not done here.

Every record carries its own proof. It holds the sweep's accuracy matrices, and a full-tile
record is refused unless they reproduce the published APD cell
(``output/studies/apd/<cohort>/<encoder>.json``) to within ``REPRODUCTION_TOLERANCE``. A
record's rows are keyed by tile identity (``loaders.TILE_KEY_COLUMNS``), not position, so it
joins a run's ``per_sample_metrics.csv`` by key (``join_per_sample``).

Resumable: a record is written to a temporary file and renamed into place, and a valid one
already on disk is skipped, so a re-launch picks up where the last one stopped. Cells fan
out across processes, one BLAS thread each, slowest first.

    python scripts/studies/apd/probe_records.py --jobs 4      # every cell
    python scripts/studies/apd/probe_records.py --status      # finished vs expected
"""

import os

# Pin BLAS to one thread per process: the probe grid-search parallelises poorly inside one
# process, so the driver fans out across cells instead. Must precede the numpy import.
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
import zipfile  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

import apd_experiment  # noqa: E402
from loaders import (  # noqa: E402
    DATASETS,
    REPO,
    STUDY_DIR,
    TILE_KEY_COLUMNS,
    cohort_tile_keys,
    load_data,
    split_schedule,
    training_correlations,
)

import croma  # noqa: E402
from croma.downstream import IN_DOMAIN, probe_sweep_with_predictions  # noqa: E402

#: Training tiles per slide each cohort is recorded at. The first Camelyon, TCGA-4x4 and
#: PCaBiop level is the cohort's own ``num_patches_per_slide``, the full-tile sweep the APD
#: cells hold; the others are the low-data sweeps. Tolkach-ESCA has no full-tile level: its
#: representation cohort and probe pool are different rows, so a per-sample join is
#: impossible there, and its full-tile accuracies are the APD cells themselves.
RUNS = {
    "camelyon": (300, 30, 3),
    "tcga_4x4": (30, 3, 1),
    "pcabiop": (1,),
    "tolkach": (10, 1),
}

#: Where records go: beside the APD study's cells, not inside them.
RECORDS_DIR = STUDY_DIR.parent / "probe-records"

#: How far a full-tile record's accuracies may sit from its APD cell. The sweep is
#: deterministic, so the honest gap is zero; the tolerance only absorbs the JSON round trip.
REPRODUCTION_TOLERANCE = 1e-9

#: The record layout's version, bumped whenever a reader would have to change.
RECORD_FORMAT = 1

#: The two test sets every record holds, under the names the sweep returns them by. The
#: APD cells store the same matrices as ``id_test_accuracies`` / ``ood_test_accuracies``.
TEST_SETS = (IN_DOMAIN, apd_experiment.OUT_OF_DOMAIN)
APD_CELL_KEYS = {IN_DOMAIN: "id_test_accuracies", apd_experiment.OUT_OF_DOMAIN: "ood_test_accuracies"}

#: The columns a held-out row is matched to a run's per-sample metrics on: tile identity.
#: Not the class, which a benchmark view attaches to a tile rather than owning (PCaBiop's
#: classes are ``benign``/``cancer`` here and ``0``/``1`` in the ``panda`` run).
PER_SAMPLE_JOIN_KEY = ("sample_id", "group_id")


class ReproductionError(ValueError):
    """A full-tile sweep that did not reproduce its published APD cell."""


@dataclass(frozen=True)
class Cell:
    """One record: a cohort, an encoder, and how many rows each training slide gives."""

    cohort: str
    model: str
    tiles: int

    @property
    def rows_per_slide(self) -> int:
        return DATASETS[self.cohort]["num_patches_per_slide"]

    @property
    def full_tiles(self) -> bool:
        return self.tiles == self.rows_per_slide

    def path(self, out_dir=RECORDS_DIR) -> Path:
        return Path(out_dir) / self.cohort / f"tiles-{self.tiles}" / f"{self.model}.npz"

    def __str__(self) -> str:
        return f"{self.cohort}/tiles-{self.tiles}/{self.model}"


def _cost(cell: Cell) -> int:
    """A sweep's rough price: probes fitted x classes x training rows per probe."""
    cfg = DATASETS[cell.cohort]
    schedule = split_schedule(cell.cohort, cell.rows_per_slide)
    train_rows = sum(take for _, _, take in schedule[0][0]) * cell.tiles // cell.rows_per_slide
    return len(schedule) * len(cfg["biological_classes"]) * train_rows


def cells(runs=None) -> list[Cell]:
    """Every record the runs table asks for, slowest sweep first.

    The encoders are the APD study's own roster for each cohort (``model_list``) -- the
    ranked panel plus the control on the PathoROB cohorts, every slide encoder on PCaBiop --
    so nothing here lists a model.
    """
    runs = RUNS if runs is None else runs
    found = []
    for cohort, levels in runs.items():
        rows_per_slide = DATASETS[cohort]["num_patches_per_slide"]
        for tiles in levels:
            if not 1 <= tiles <= rows_per_slide:
                raise ValueError(
                    f"{cohort}: {tiles} training tiles per slide is outside "
                    f"[1, {rows_per_slide}]"
                )
            found.extend(Cell(cohort, model, tiles) for model in apd_experiment.model_list(cohort))
    return sorted(found, key=lambda cell: (-_cost(cell), cell.cohort, -cell.tiles, cell.model))


def load_cohort(dataset, model):
    """One cohort as the APD study loads it, and the tile key of every row it holds."""
    id_keys, ood_keys = cohort_tile_keys(dataset)
    return load_data(model, dataset), id_keys, ood_keys


def _check_keys(cell, cohort, id_keys, ood_keys):
    """Refuse keys that do not describe the rows the sweep is about to be run on."""
    cfg = DATASETS[cell.cohort]
    for what, keys, labels, n_rows in (
        ("ID", id_keys, cohort.labels, len(cohort.embeddings)),
        ("OOD", ood_keys, cohort.ood_labels, len(cohort.ood_embeddings)),
    ):
        if len(keys) != n_rows:
            raise ValueError(f"{cell}: {len(keys)} {what} tile keys for {n_rows} rows")
        named = keys["label"].map(cfg["biological_classes"].index).to_numpy()
        if not np.array_equal(named, np.asarray(labels)):
            raise ValueError(f"{cell}: the {what} tile keys' classes are not the cohort's")
    centres = id_keys["confounder"].map(cfg["centers_id"].index).to_numpy()
    if not np.array_equal(centres, np.asarray(cohort.confounders)):
        raise ValueError(f"{cell}: the ID tile keys' centres are not the cohort's")


def apd_cell_path(cell: Cell, apd_dir=STUDY_DIR) -> Path:
    return Path(apd_dir) / cell.cohort / f"{cell.model}.json"


def _relative(path: Path) -> str:
    """``path`` relative to the checkout when it is inside it, so a record names no clone."""
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


def reproduction_gap(cell: Cell, accuracies, apd_dir=STUDY_DIR) -> dict:
    """How far a full-tile sweep's matrices sit from its APD cell; raises past tolerance."""
    path = apd_cell_path(cell, apd_dir)
    if not path.exists():
        raise ReproductionError(f"{cell}: no APD cell at {path} to reproduce")
    published = json.loads(path.read_text(encoding="utf-8"))
    gaps = {}
    for name, key in APD_CELL_KEYS.items():
        expected = np.asarray(published[key], dtype=float)
        observed = np.asarray(accuracies[name], dtype=float)
        if expected.shape != observed.shape:
            raise ReproductionError(
                f"{cell}: {name} accuracies have shape {observed.shape}, the APD cell's "
                f"{key} {expected.shape}"
            )
        gaps[name] = float(np.abs(observed - expected).max())
        if not gaps[name] <= REPRODUCTION_TOLERANCE:
            raise ReproductionError(
                f"{cell}: {name} accuracies differ from the APD cell by {gaps[name]:.3e}, "
                f"more than {REPRODUCTION_TOLERANCE:g}"
            )
    return gaps


def compute(cell: Cell, *, iterations=apd_experiment.CANONICAL_ITERATIONS, apd_dir=STUDY_DIR):
    """Run one cell's sweep and return its record, as the arrays ``write_record`` stores.

    A full-tile cell runs the APD study's sweep unchanged and must reproduce its APD cell;
    a reduced cell runs the same sweep with ``train_rows_per_slide`` set.
    """
    cfg = DATASETS[cell.cohort]
    cohort, id_keys, ood_keys = load_cohort(cell.cohort, cell.model)
    _check_keys(cell, cohort, id_keys, ood_keys)
    arguments = apd_experiment.sweep_arguments(cell.cohort, cohort)

    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        # lbfgs does not converge at the grid's weakest regularisation on every cohort;
        # the APD cells were computed through the same warnings. They are counted into the
        # record rather than printed, once per probe, into the log.
        warnings.simplefilter("always", ConvergenceWarning)
        record = probe_sweep_with_predictions(
            cohort.embeddings,
            cohort.confounders,
            cohort.labels,
            iterations=iterations,
            train_rows_per_slide=None if cell.full_tiles else cell.tiles,
            **arguments,
        )
    seconds = time.time() - started
    convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    for other in {str(w.message) for w in caught if not issubclass(w.category, ConvergenceWarning)}:
        print(f"[warn] {cell}: {other}", flush=True)

    gaps = reproduction_gap(cell, record.accuracies, apd_dir) if cell.full_tiles else None
    meta = dict(
        format=RECORD_FORMAT,
        cohort=cell.cohort,
        model=cell.model,
        tiles_per_slide=cell.tiles,
        rows_per_slide=cell.rows_per_slide,
        full_tiles=cell.full_tiles,
        iterations=iterations,
        seed=arguments["seed"],
        validation_fraction=arguments["validation_fraction"],
        n_splits=len(arguments["schedule"]),
        cramers_v=list(training_correlations(cell.cohort)),
        biological_classes=list(cfg["biological_classes"]),
        centers_id=list(cfg["centers_id"]),
        test_sets=list(TEST_SETS),
        apd_cell=None if gaps is None else _relative(apd_cell_path(cell, apd_dir)),
        apd_reproduction_gap=gaps,
        convergence_warnings=int(convergence),
        seconds=round(seconds, 1),
        croma_version=croma.__version__,
    )
    arrays = {
        "meta": np.array(json.dumps(meta)),
        "held_out": record.held_out,
        "best_c": record.best_c,
        "classes": record.classes,
    }
    for name in TEST_SETS:
        arrays[f"accuracies_{name}"] = record.accuracies[name]
        arrays[f"margins_{name}"] = record.margins[name]
    for prefix, keys in (("id", id_keys), ("ood", ood_keys)):
        for column in TILE_KEY_COLUMNS:
            arrays[f"{prefix}_{column}"] = keys[column].to_numpy(dtype=str)
    return arrays


def write_record(path, arrays) -> None:
    """Publish a record atomically: a flushed same-directory temporary, then a rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_record(path) -> dict:
    """A record's arrays, with ``meta`` decoded. Nothing in a record needs pickle."""
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["meta"] = json.loads(str(arrays["meta"]))
    return arrays


def _decoded(arrays) -> dict:
    return {**arrays, "meta": json.loads(str(arrays["meta"]))}


def _predicted(margins, classes):
    """What sklearn's ``predict`` makes of ``decision_function``: sign if binary, else argmax."""
    if margins.shape[-1] == 1:
        return classes[(margins[..., 0] > 0).astype(int)]
    return classes[margins.argmax(axis=-1)]


def check_record(record, cell: Cell, *, iterations, apd_dir=STUDY_DIR) -> None:
    """Raise ``ValueError`` unless ``record`` is a complete, self-consistent record of ``cell``.

    Complete: the right cell, every array, every shape. Self-consistent: re-scoring each
    probe's stored margins against its rows' classes gives back the stored accuracy, which
    is what ties every margin to the row it is keyed by. A full-tile record must still
    reproduce its APD cell, so a re-run of the APD study invalidates the records built on
    the old one.
    """
    meta = record["meta"]
    expected = dict(
        format=RECORD_FORMAT,
        cohort=cell.cohort,
        model=cell.model,
        tiles_per_slide=cell.tiles,
        iterations=iterations,
        seed=apd_experiment.PROBE_SEED,
    )
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"{cell}: record has {key}={meta.get(key)!r}, expected {value!r}")

    classes_by_name = {name: index for index, name in enumerate(meta["biological_classes"])}
    n_splits = meta["n_splits"]
    held_out = record["held_out"]
    n_id = len(record["id_sample_id"])
    if held_out.ndim != 2 or held_out.shape[0] != iterations:
        raise ValueError(f"{cell}: held_out has shape {held_out.shape}")
    if held_out.size and not (0 <= held_out.min() and held_out.max() < n_id):
        raise ValueError(f"{cell}: held_out indexes rows outside the record's {n_id} ID rows")
    if record["best_c"].shape != (n_splits, iterations):
        raise ValueError(f"{cell}: best_c has shape {record['best_c'].shape}")

    id_labels = np.array([classes_by_name[label] for label in record["id_label"]])
    ood_labels = np.array([classes_by_name[label] for label in record["ood_label"]])
    for name in TEST_SETS:
        accuracies = record[f"accuracies_{name}"]
        margins = record[f"margins_{name}"]
        rows = held_out.shape[1] if name == IN_DOMAIN else len(ood_labels)
        if accuracies.shape != (n_splits, iterations):
            raise ValueError(f"{cell}: accuracies_{name} has shape {accuracies.shape}")
        if margins.shape[:3] != (n_splits, iterations, rows):
            raise ValueError(f"{cell}: margins_{name} has shape {margins.shape}")
        if not np.isfinite(margins).all():
            raise ValueError(f"{cell}: margins_{name} must be finite")
        for split in range(n_splits):
            for replicate in range(iterations):
                truth = id_labels[held_out[replicate]] if name == IN_DOMAIN else ood_labels
                predicted = _predicted(margins[split, replicate], record["classes"])
                rescored = balanced_accuracy_score(truth, predicted)
                if not np.isclose(rescored, accuracies[split, replicate], rtol=0, atol=1e-12):
                    raise ValueError(
                        f"{cell}: {name} margins at split {split}, replicate {replicate} "
                        f"score {rescored}, not the stored {accuracies[split, replicate]}"
                    )
    if cell.full_tiles:
        reproduction_gap(cell, {name: record[f"accuracies_{name}"] for name in TEST_SETS}, apd_dir)


def run_cell(cell: Cell, *, out_dir=RECORDS_DIR, iterations, apd_dir=STUDY_DIR) -> str:
    """Make sure ``cell``'s record exists and is valid: ``"skip"`` if it was, else ``"done"``."""
    path = cell.path(out_dir)
    if path.exists():
        try:
            check_record(read_record(path), cell, iterations=iterations, apd_dir=apd_dir)
            return "skip"
        except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
            print(f"[redo] {cell}: the record on disk is not valid ({error})", flush=True)
    arrays = compute(cell, iterations=iterations, apd_dir=apd_dir)
    check_record(_decoded(arrays), cell, iterations=iterations, apd_dir=apd_dir)
    write_record(path, arrays)
    return "done"


def _job(task):
    cell, out_dir, iterations, apd_dir = task
    started = time.time()
    try:
        tag = run_cell(cell, out_dir=out_dir, iterations=iterations, apd_dir=apd_dir)
    except Exception as error:  # noqa: BLE001 -- one failed cell must not stop the others
        print(f"[fail] {cell}: {type(error).__name__}: {error}", flush=True)
        return cell, "fail"
    print(f"[{tag}] {cell} ({time.time() - started:.0f}s)", flush=True)
    return cell, tag


def run(selected, *, jobs=1, out_dir=RECORDS_DIR, iterations, apd_dir=STUDY_DIR):
    """Bring every selected cell's record up to date; returns the cells that failed."""
    tasks = [(cell, Path(out_dir), iterations, Path(apd_dir)) for cell in selected]
    print(f"{len(tasks)} cell(s), {jobs} worker(s)", flush=True)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outcomes = [future.result() for future in as_completed(pool.submit(_job, t) for t in tasks)]
    else:
        outcomes = [_job(task) for task in tasks]
    failed = [cell for cell, tag in outcomes if tag == "fail"]
    print(f"finished: {len(tasks) - len(failed)} ok, {len(failed)} failed", flush=True)
    for cell in failed:
        print(f"  failed: {cell}", flush=True)
    return failed


def status(selected, *, out_dir=RECORDS_DIR) -> tuple[int, int]:
    """How many of the selected records are on disk, out of how many expected."""
    present = sum(cell.path(out_dir).exists() for cell in selected)
    return present, len(selected)


def per_sample_path(cohort: str) -> Path:
    """The ``per_sample_metrics.csv`` of the run the paper reports for ``cohort``."""
    from paper_manifest import by_benchmark

    return REPO / by_benchmark(DATASETS[cohort]["benchmark"]).per_sample_rel


def join_per_sample(record, per_sample: pd.DataFrame) -> pd.DataFrame:
    """Every held-out ID row of a record, matched by tile key to its per-sample metrics row.

    Returns one row per distinct held-out row, with ``row`` -- its position in the record's
    ID rows, which is what ``held_out`` holds -- beside the matched per-sample columns. A
    ``model`` column in ``per_sample`` is filtered to the record's encoder first.

    Raises:
        ValueError: If a held-out row matches no per-sample row or more than one, or if a
            matched row names a different centre.
    """
    meta = record["meta"]
    if "model" in per_sample:
        per_sample = per_sample[per_sample["model"] == meta["model"]]
    rows = np.unique(record["held_out"])
    held = pd.DataFrame(
        {column: record[f"id_{column}"][rows] for column in TILE_KEY_COLUMNS}
    ).assign(row=rows)
    key = list(PER_SAMPLE_JOIN_KEY)
    if per_sample.duplicated(key).any():
        raise ValueError(f"{meta['model']}: per-sample metrics repeat a tile key")
    joined = held.merge(
        per_sample.astype({column: str for column in key}),
        on=key,
        how="left",
        suffixes=("_record", ""),
        indicator=True,
    )
    unmatched = joined["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"{meta['model']}: {int(unmatched.sum())} held-out row(s) match no per-sample "
            f"row, e.g. {joined.loc[unmatched, key].iloc[0].to_dict()}"
        )
    if "confounder" in per_sample and not (
        joined["confounder"].astype(str) == joined["confounder_record"]
    ).all():
        raise ValueError(f"{meta['model']}: a held-out row's centre disagrees with its match")
    return joined.drop(columns="_merge")


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cohorts", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--tiles", nargs="+", type=int, default=None, help="only these tile levels")
    p.add_argument("--models", nargs="+", default=None, help="only these encoders")
    p.add_argument("--iterations", type=int, default=apd_experiment.CANONICAL_ITERATIONS)
    p.add_argument("--jobs", type=int, default=1, help="parallel cell processes")
    p.add_argument("--out-dir", type=Path, default=RECORDS_DIR)
    p.add_argument("--apd-dir", type=Path, default=STUDY_DIR)
    p.add_argument("--status", action="store_true", help="count finished records and exit")
    p.add_argument("--list", action="store_true", help="print the cells in run order and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = get_args(argv)
    selected = [
        cell
        for cell in cells({cohort: RUNS[cohort] for cohort in args.cohorts})
        if (args.tiles is None or cell.tiles in args.tiles)
        and (args.models is None or cell.model in args.models)
    ]
    if args.status:
        present, expected = status(selected, out_dir=args.out_dir)
        print(f"{present}/{expected} records present", flush=True)
        return 0 if present == expected else 1
    if args.list:
        for cell in selected:
            print(cell)
        return 0
    failed = run(
        selected,
        jobs=args.jobs,
        out_dir=args.out_dir,
        iterations=args.iterations,
        apd_dir=args.apd_dir,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
