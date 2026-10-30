"""Score and ranking stability of RI, MaRI and CRoMa under cohort subsampling.

Every published score is computed once, on the full cohort, which leaves two questions
unmeasured: whether the scores and the encoder rankings drift as the cohort shrinks,
and whether one metric's ranking is more volatile than another's under the *same*
perturbation of the cohort.

Design
------
Each of the three tile cohorts (Camelyon, TCGA-4x4, Tolkach-ESCA) is subsampled *without
replacement* within every class x centre cell, at three fractions, 20 replicates each.

* **The unit is a block**: the tiles one independence group (``group_id``) contributes to one
  cell. On Camelyon and TCGA-4x4 each slide sits in exactly one cell, so a block is a slide.
  On Tolkach-ESCA a case contributes a 100-tile block to every class it spans, so a block is
  a case's block within the cell. Sampling blocks keeps every cell's count fixed; sampling
  whole cases would let it vary across replicates. A case's surviving blocks in other classes
  stay excluded from each other's neighbourhoods, because the rows keep their ``group_id``.
* **The counts per cell are the design table, not a rounding rule** (``UNITS_PER_CELL``). No
  single rule produces all nine of them -- Camelyon's 50% rounds 8.5 down while
  Tolkach-ESCA's rounds 2.5 up -- so they are declared as stated, every one at least 2.
* **Paired draws**: a replicate is one subsample, seeded from ``(SEED, cohort, fraction,
  replicate)`` alone and shared by every encoder and every metric, so the metrics are
  compared on identical subsamples.

Recompute everything
--------------------
A replicate is the production driver, ``scripts/bench/benchmark.py``, run on the subsampled
manifest; nothing is reimplemented. It is scored under two protocols:

* **re-selected k** (primary): the production ``median-k`` protocol as it stands -- every
  encoder's k* from the driver's biological kNN pre-pass, the shared k from the driver's own
  ``_shared_operating_k`` (lower median) over the production roster, automatic tau, the
  production run's own k grid (sparse);
* **fixed k** (secondary): the same, with the shared k held at the full-cohort run's own
  operating k (read from that run, never typed), which separates the volatility of
  re-selecting k from the volatility at a fixed k.

CRoMa does not depend on k, so both protocols score it identically from one shared cache.

The 100% "subsample" is the production manifest itself, and both protocols must reproduce
the production ``metrics.csv`` exactly -- every stored value, as text -- before any
subsampled replicate of that cohort is run. The published runs were computed under two
OpenBLAS GEMM kernels (see ``CANDIDATE_KERNELS``); the check recovers which one each encoder
used, and every replicate of the cohort reuses it.

Outputs go under ``output/studies/subsampling-stability/``; nothing here writes into a
production ``output/metrics/`` run. The control is computed (it belongs to the production
roster and so to the shared k) but is excluded from every ranking and summary downstream.

Run (detached, resumable; each worker runs one BLAS thread):
  python scripts/studies/subsampling_stability.py sweep --workers 4 --scratch-root /var/tmp/x
  python scripts/studies/subsampling_stability.py status
Once every record is on disk the sweep writes ``complete.json``; the paper build waits for it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark as driver  # noqa: E402
import benchmarks  # noqa: E402
import layout  # noqa: E402
import run_config  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

#: The three PathoROB tile cohorts this study subsamples, in the paper's order.
COHORTS = ("pathorob-camelyon", "pathorob-tcga-4x4", "pathorob-tolkach-esca")

#: Subsampled fractions, in percent. ``FULL`` is the reproduction check, not a replicate.
FRACTIONS = (25, 50, 75)
FULL = 100

N_REPLICATES = 20

#: The study's single seed. A replicate's draw is a function of this and its own coordinates.
SEED = 1729

#: Units per class x centre cell: the full cohort, then the stated count at each fraction.
#: Declared rather than derived -- see the module docstring.
UNITS_PER_CELL: dict[str, tuple[int, dict[int, int]]] = {
    "pathorob-camelyon": (17, {25: 4, 50: 8, 75: 13}),
    "pathorob-tcga-4x4": (12, {25: 3, 50: 6, 75: 9}),
    "pathorob-tolkach-esca": (5, {25: 2, 50: 3, 75: 4}),
}

#: The manifest columns that define a cell and a unit.
LABEL, CONFOUNDER, GROUP = "label", "medical_center", "group_id"
CELL = (LABEL, CONFOUNDER)
UNIT = (LABEL, CONFOUNDER, GROUP)

STUDY = "subsampling-stability"
DESIGN = "design.json"


def study_root() -> Path:
    """``output/studies/subsampling-stability/``: the study's tree, beside -- never inside --
    the production ``output/metrics/`` runs."""
    return layout.OUTPUT_ROOT / "studies" / STUDY


def design(
    *,
    cohorts: tuple[str, ...] = COHORTS,
    fractions: tuple[int, ...] = FRACTIONS,
    n_replicates: int = N_REPLICATES,
) -> dict:
    """The design a study tree was drawn under. The paper is built only from this default."""
    return {
        "cohorts": list(cohorts),
        "fractions": [int(f) for f in fractions],
        "n_replicates": int(n_replicates),
        "seed": SEED,
        "units_per_cell": {
            c: {"full": UNITS_PER_CELL[c][0], **{str(f): units_per_cell(c, f) for f in fractions}}
            for c in cohorts
        },
    }


def write_design(root: Path, record: dict) -> None:
    """Record the design at the study root; a tree drawn under another design is refused."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / DESIGN
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != record:
            raise ValueError(
                f"{path} records another design; move that study aside before drawing this one"
            )
        return
    _atomic_write(path, json.dumps(record, indent=2) + "\n")


def replicate_dir(root: Path, cohort: str, fraction: int, replicate: int) -> Path:
    """``<cohort>/full`` for the 100% run, else ``<cohort>/f<fraction>/r<replicate>``."""
    if int(fraction) == FULL:
        return Path(root) / cohort / "full"
    return Path(root) / cohort / f"f{int(fraction):02d}" / f"r{int(replicate):02d}"


def replicate_seed(cohort: str, fraction: int, replicate: int) -> np.random.SeedSequence:
    """The seed of one replicate's draw: fixed by its coordinates, never by run order."""
    return np.random.SeedSequence(
        [SEED, zlib.crc32(cohort.encode("utf-8")), int(fraction), int(replicate)]
    )


def units_per_cell(cohort: str, fraction: int) -> int:
    """The stated number of units every cell keeps at ``fraction`` percent."""
    full, stated = UNITS_PER_CELL[cohort]
    if int(fraction) == FULL:
        return full
    if int(fraction) not in stated:
        raise ValueError(f"{cohort}: no stated unit count at {fraction}%")
    return stated[int(fraction)]


def draw_subsample(
    manifest: pd.DataFrame, cohort: str, fraction: int, replicate: int
) -> pd.DataFrame:
    """One replicate's subsample of ``manifest``: whole blocks, a fixed count per cell.

    The rows keep the manifest's columns and order, so the 100% draw *is* the manifest.
    """
    full, _ = UNITS_PER_CELL[cohort]
    units = manifest.loc[:, list(UNIT)].drop_duplicates()
    per_cell = units.groupby(list(CELL))[GROUP].nunique()
    if set(per_cell) != {full}:
        raise ValueError(
            f"{cohort}: the design was stated for {full} units in every class x centre "
            f"cell, but the manifest's cells hold {sorted(set(per_cell))}"
        )
    keep = units_per_cell(cohort, fraction)
    rng = np.random.default_rng(replicate_seed(cohort, fraction, replicate))
    chosen = []
    for _cell, cell_units in units.sort_values(list(UNIT)).groupby(list(CELL)):
        picks = rng.choice(len(cell_units), size=keep, replace=False)
        chosen.append(cell_units.iloc[np.sort(picks)])
    selected = pd.MultiIndex.from_frame(pd.concat(chosen))
    mask = pd.MultiIndex.from_frame(manifest.loc[:, list(UNIT)]).isin(selected)
    return manifest.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------- the production run


@dataclass(frozen=True)
class ProductionRun:
    """The full-cohort run a cohort's replicates are re-runs of.

    Everything a replicate needs from it is read from the run itself -- the roster from its
    ``metrics.csv``, the driver flags from its ``run_config.json``, the operating k from both
    -- so a replicate replays the published protocol rather than a typed copy of it.
    """

    benchmark: str
    protocol: str
    results_dir: Path
    #: ``metrics.csv``'s models, in its row order. The shared k is a median over this roster.
    roster: tuple[str, ...]
    #: ``benchmark.py`` flags that replay the run (``run_config.replay_args``).
    replay: tuple[str, ...]
    #: The swept k values; the largest must stay below every subsample's usable neighbours.
    k_values: tuple[int, ...]
    #: The run's shared operating k: the fixed k of the secondary protocol.
    shared_k: int
    metrics: pd.DataFrame


def production_run(benchmark: str) -> ProductionRun:
    """Resolve ``benchmark``'s published run through ``paper_manifest`` and read its protocol."""
    protocol = by_benchmark(benchmark).protocol
    results_dir = layout.results_dir(protocol, benchmark)
    config = run_config.read_run_config(results_dir)
    if config is None:
        raise FileNotFoundError(f"{results_dir} has no run_config.json to replay")
    run_config.check_resolved(config, benchmark=benchmark, protocol=protocol)
    replay = dict(config["replay"])
    if protocol != "median-k":
        raise ValueError(f"{benchmark}: the study re-runs median-k, not {protocol}")
    if replay.get("tau") != "auto" or replay.get("k_grid") != "sparse":
        raise ValueError(
            f"{benchmark}: expected an automatic-tau, sparse-grid run, found "
            f"tau={replay.get('tau')!r}, k_grid={replay.get('k_grid')!r}"
        )
    metrics = pd.read_csv(results_dir / "metrics.csv")
    ks = set(metrics["k"].astype(int))
    operating_k = config["resolved"].get("operating_k")
    if operating_k is None or ks != {int(operating_k)}:
        raise ValueError(
            f"{benchmark}: metrics.csv k={sorted(ks)} disagrees with the recorded "
            f"operating_k={operating_k}"
        )
    k_values = tuple(int(k) for k in str(config["resolved"]["k_values"]).split(","))
    return ProductionRun(
        benchmark=benchmark,
        protocol=protocol,
        results_dir=results_dir,
        roster=tuple(str(m) for m in metrics["model"]),
        replay=tuple(run_config.replay_args(config)),
        k_values=k_values,
        shared_k=int(operating_k),
        metrics=metrics,
    )


# --------------------------------------------------------------------------- one replicate

#: The two protocols of every replicate, by output subdirectory.
RESELECTED, FIXED_K = "reselected", "fixed-k"
VARIANTS = (RESELECTED, FIXED_K)

#: What a driver run leaves behind that the study keeps. The per-sample artifacts (up to a
#: gigabyte per Camelyon run) stay in the scratch tree and are deleted with it.
KEPT_ARTIFACTS = ("metrics.csv", "k_sweep_metrics.csv")

REPLICATE_META = "replicate.json"
SCORES = "scores.csv"
UNITS = "units.csv"
REPRODUCTION = "reproduction.json"
K_STAR = "k_star.json"

#: GEMM kernels (numpy's OpenBLAS core type) a published row may have been computed with.
#: This host auto-selects Cooperlake; part of every published run was computed under
#: SkylakeX, and the two round float32 distances differently enough to flip near-tied
#: neighbours. Which kernel each (cohort, encoder) used is recovered by the 100% reproduction
#: check, never assumed, and every replicate of the cohort reuses it.
CANDIDATE_KERNELS = ("Cooperlake", "SkylakeX")
#: Run in this process, on whatever kernel it already has (the tests, and single-kernel use).
IN_PROCESS = ""

SINGLE_THREAD = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def min_usable_neighbours(manifest: pd.DataFrame) -> int:
    """The fewest neighbours any row has once its own independence group is excluded."""
    return int(len(manifest) - manifest[GROUP].value_counts().max())


@contextmanager
def _driver_on(benchmark: str, protocol: str, manifest_path: Path, scratch: Path):
    """Point the production driver at ``manifest_path`` and a scratch output tree.

    The same two rebindings the benchmark tests use (``tests/conftest.py``): the registry
    entry's manifest, and ``layout.OUTPUT_ROOT``. The scratch root's ``embeddings`` links to
    the real tilesets, so the driver reads the published embeddings and writes nowhere else.
    Both are restored on exit.
    """
    spec = benchmarks.get(benchmark)
    saved_root = layout.OUTPUT_ROOT
    embeddings = layout.embeddings_root().resolve()
    root = Path(scratch) / "output"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "embeddings"
    if not link.is_symlink():
        link.symlink_to(embeddings, target_is_directory=True)
    benchmarks.BENCHMARKS[benchmark] = replace(
        spec, manifest=str(Path(manifest_path).resolve())
    )
    layout.OUTPUT_ROOT = root
    try:
        yield layout.results_dir(protocol, benchmark)
    finally:
        layout.OUTPUT_ROOT = saved_root
        benchmarks.BENCHMARKS[benchmark] = spec


@contextmanager
def _shared_k_pinned(k: int, seen: dict[str, int]):
    """Hold the driver's shared operating k at ``k``, recording the k* it was handed.

    Every other step runs as published; the k* are the driver's own selection, read off the
    call rather than re-derived from a rounded CSV.
    """
    original = driver._shared_operating_k

    def pinned(per_model_best_k: dict[str, int]) -> int:
        original(per_model_best_k)  # keeps the driver's own empty-roster guard
        seen.update({str(m): int(v) for m, v in per_model_best_k.items()})
        return int(k)

    driver._shared_operating_k = pinned
    try:
        yield
    finally:
        driver._shared_operating_k = original


def drive(
    run: ProductionRun,
    models: list[str],
    *,
    pin_k: int,
    manifest_path: Path,
    scratch: Path,
    dest: Path,
) -> dict[str, int]:
    """One production-driver run of ``models`` on ``manifest_path``, shared k held at ``pin_k``.

    Keeps ``metrics.csv`` and ``k_sweep_metrics.csv`` in ``dest`` and returns (and writes to
    ``dest/k_star.json``) each model's k* as the driver selected it. Runs sharing a
    ``scratch`` share the driver's cache, so a second run reuses the kNN curves and CRoMa.
    """
    argv = [
        "benchmark.py",
        "--benchmark",
        run.benchmark,
        "--protocol",
        run.protocol,
        "--models",
        ",".join(models),
        "--progress",
        "off",
        *run.replay,
    ]
    seen: dict[str, int] = {}
    with _driver_on(run.benchmark, run.protocol, manifest_path, scratch) as results_dir:
        saved = sys.argv
        sys.argv = argv
        try:
            with _shared_k_pinned(pin_k, seen):
                code = driver.main()
        finally:
            sys.argv = saved
        if code != 0:
            raise RuntimeError(f"{run.benchmark}: benchmark.py exited {code}")
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name in KEPT_ARTIFACTS:
            shutil.copyfile(results_dir / name, dest / name)
    if set(seen) != set(models):
        raise RuntimeError(f"{run.benchmark}: the driver selected k* for {sorted(seen)}")
    _atomic_write(dest / K_STAR, json.dumps(seen, indent=2, sort_keys=True) + "\n")
    return seen


def numpy_kernel() -> str | None:
    """The core type numpy's OpenBLAS runs, which computes the embedding distances.

    A kernel label always names this library. SciPy bundles an OpenBLAS of its own, which
    picks its core type independently (SkylakeX on this host either way).
    """
    from threadpoolctl import threadpool_info

    for info in threadpool_info():
        if info.get("internal_api") == "openblas" and info.get("prefix") == "libopenblas":
            return str(info.get("architecture"))
    return None


def _stage(
    run: ProductionRun,
    models: list[str],
    *,
    kernel: str,
    pin_k: int,
    manifest_path: Path,
    scratch: Path,
    dest: Path,
) -> dict[str, int]:
    """``drive`` under ``kernel``: here, or in a child process started on that kernel.

    A core type is fixed when OpenBLAS loads, so a kernel other than this process's own needs
    a process of its own.
    """
    if kernel == IN_PROCESS:
        return drive(
            run, models, pin_k=pin_k, manifest_path=manifest_path, scratch=scratch, dest=dest
        )
    import subprocess

    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "drive",
        "--benchmark",
        run.benchmark,
        "--models",
        ",".join(models),
        "--pin-k",
        str(pin_k),
        "--manifest",
        str(manifest_path),
        "--scratch",
        str(scratch),
        "--dest",
        str(dest),
    ]
    env = {**os.environ, **SINGLE_THREAD, "OPENBLAS_CORETYPE": kernel}
    code = subprocess.run(argv, env=env, cwd=REPO).returncode
    if code != 0:
        raise RuntimeError(f"{run.benchmark}: the {kernel} driver run exited {code}")
    return {str(m): int(k) for m, k in json.loads((Path(dest) / K_STAR).read_text()).items()}


def _merge(sources: list[Path], roster: tuple[str, ...], dest: Path) -> None:
    """One ``metrics.csv`` / ``k_sweep_metrics.csv`` per protocol, rows in roster order.

    The values are carried as their stored text, so merging never re-rounds a float.
    """
    dest.mkdir(parents=True, exist_ok=True)
    position = {m: i for i, m in enumerate(roster)}
    for name in KEPT_ARTIFACTS:
        frame = pd.concat([_as_written(s / name) for s in sources], ignore_index=True)
        order = frame["model"].map(position)
        if order.isna().any():
            raise RuntimeError(f"{name}: rows for models outside the roster")
        frame = frame.iloc[np.argsort(order.to_numpy(), kind="stable")]
        frame.to_csv(dest / name, index=False)


def run_replicate(
    run: ProductionRun,
    manifest: pd.DataFrame,
    *,
    out_dir: Path,
    scratch: Path,
    kernels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Re-run the production protocol on ``manifest`` under both protocols, and record it.

    1. **fixed k**: every encoder at the full-cohort operating k. The driver's own kNN
       pre-pass selects each encoder's k* along the way.
    2. **re-selected k**: the shared k is the driver's ``_shared_operating_k`` over the whole
       production roster's k* -- exactly what an unpinned production run computes -- and
       every encoder is re-run at it, reusing the kNN curves and CRoMa from the cache.

    ``kernels`` maps each encoder to the GEMM kernel it runs under (default: all in this
    process). Writes ``out_dir/{reselected,fixed-k}/`` (``metrics.csv`` and
    ``k_sweep_metrics.csv``), ``units.csv`` (the drawn blocks), ``scores.csv`` (one row per
    encoder) and, last, ``replicate.json`` -- its presence marks the replicate complete.
    """
    kernels = {m: IN_PROCESS for m in run.roster} if kernels is None else dict(kernels)
    if set(kernels) != set(run.roster):
        raise ValueError(f"{run.benchmark}: kernels must cover the roster exactly")
    usable = min_usable_neighbours(manifest)
    if max(run.k_values) >= usable:
        raise ValueError(
            f"{run.benchmark}: the k grid reaches k={max(run.k_values)} but a row of this "
            f"subsample has only {usable} usable neighbours; cap the grid"
        )
    groups: dict[str, list[str]] = {}
    for model in run.roster:
        groups.setdefault(kernels[model], []).append(model)

    out_dir = Path(out_dir)
    scratch = Path(scratch)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        manifest_path = scratch / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        stages = scratch / "stages"
        common = dict(manifest_path=manifest_path, scratch=scratch / "driver")

        def where(variant: str, kernel: str) -> Path:
            return stages / variant / (kernel or "in-process")

        k_star: dict[str, int] = {}
        for kernel, models in groups.items():
            k_star.update(
                _stage(
                    run,
                    models,
                    kernel=kernel,
                    pin_k=run.shared_k,
                    dest=where(FIXED_K, kernel),
                    **common,
                )
            )
        shared_k = int(driver._shared_operating_k({m: k_star[m] for m in run.roster}))
        for kernel, models in groups.items():
            _stage(
                run,
                models,
                kernel=kernel,
                pin_k=shared_k,
                dest=where(RESELECTED, kernel),
                **common,
            )
        for variant in VARIANTS:
            _merge([where(variant, k) for k in groups], run.roster, out_dir / variant)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    scores = _scores(run, out_dir, k_star=k_star, shared_k=shared_k)
    units = manifest.loc[:, list(UNIT)].drop_duplicates()
    units.to_csv(out_dir / UNITS, index=False)
    meta = {
        "benchmark": run.benchmark,
        "n_rows": int(len(manifest)),
        "n_units": int(len(units)),
        "shared_k": shared_k,
        "fixed_k": int(run.shared_k),
        "k_grid_max": int(max(run.k_values)),
        "min_usable_neighbours": int(usable),
        "roster": list(run.roster),
        "kernel_by_model": kernels,
    }
    record_replicate(out_dir, scores, meta)
    return scores


def record_replicate(out_dir: Path, scores: pd.DataFrame, meta: dict) -> None:
    """Write a replicate's ``scores.csv``, then its ``replicate.json`` -- the completion mark."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (SCORES + ".tmp")
    scores.to_csv(tmp, index=False)
    os.replace(tmp, out_dir / SCORES)
    _atomic_write(out_dir / REPLICATE_META, json.dumps(meta, indent=2) + "\n")


def _read_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _scores(
    run: ProductionRun, out_dir: Path, *, k_star: dict[str, int], shared_k: int
) -> pd.DataFrame:
    """One row per encoder: both protocols' RI/MaRI/support, CRoMa, and the chosen k."""
    reselected = _read_metrics(out_dir / RESELECTED / "metrics.csv")
    fixed = _read_metrics(out_dir / FIXED_K / "metrics.csv")
    for frame in (reselected, fixed):
        if tuple(frame["model"]) != run.roster:
            raise RuntimeError(f"{run.benchmark}: a driver run lost part of the roster")
    for column in ("croma", "croma_f0", "croma_ltm_alpha"):
        if not reselected[column].equals(fixed[column]):
            raise RuntimeError(
                f"{run.benchmark}: {column} differs between the two runs of one replicate; "
                "CRoMa is k-free, so they did not score the same rows"
            )
    if set(fixed["k"].astype(int)) != {run.shared_k}:
        raise RuntimeError(f"{run.benchmark}: the fixed-k run did not hold k={run.shared_k}")
    if set(reselected["k"].astype(int)) != {shared_k}:
        raise RuntimeError(f"{run.benchmark}: the re-selected run did not hold k={shared_k}")
    return pd.DataFrame(
        {
            "model": reselected["model"],
            "ri": reselected["ri"],
            "mari": reselected["mari"],
            "support": reselected["support"],
            "k": reselected["k"].astype(int),
            "tau": reselected["tau"],
            "k_star": [k_star[str(m)] for m in reselected["model"]],
            "ri_fixed_k": fixed["ri"],
            "mari_fixed_k": fixed["mari"],
            "support_fixed_k": fixed["support"],
            "k_fixed_k": fixed["k"].astype(int),
            "tau_fixed_k": fixed["tau"],
            "croma": reselected["croma"],
            "croma_f0": reselected["croma_f0"],
            "croma_ltm_alpha": reselected["croma_ltm_alpha"],
        }
    )


def read_replicate(out_dir: Path) -> dict:
    """A completed replicate's record, or ``FileNotFoundError`` if it never finished."""
    path = Path(out_dir) / REPLICATE_META
    if not path.exists():
        raise FileNotFoundError(f"replicate not complete: {path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- reproduction


class ReproductionError(AssertionError):
    """The 100% subsample did not reproduce the production run."""


def _as_written(path: Path) -> pd.DataFrame:
    """A ``metrics.csv`` exactly as stored: every value as its text, nothing parsed.

    pandas' default float parser is not round-trip exact, so two different stored values
    can parse to one float; comparing the text is the stricter test.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _differing_columns(new: pd.DataFrame, published: pd.DataFrame) -> list[str]:
    """Columns whose values differ at all. A ``*_path`` column names where an artifact
    lives, so it is compared by file name: the re-run writes under its scratch tree."""
    if list(new.columns) != list(published.columns):
        return ["<columns>"]
    if len(new) != len(published):
        return ["<rows>"]
    new, published = new.reset_index(drop=True), published.reset_index(drop=True)
    differing = []
    for column in published.columns:
        if column.endswith("_path"):
            left = new[column].map(lambda p: Path(str(p)).name)
            right = published[column].map(lambda p: Path(str(p)).name)
        else:
            left, right = new[column], published[column]
        if not left.equals(right):
            differing.append(column)
    return differing


def assign_kernels(
    trials: dict[str, dict[str, pd.DataFrame]], published: pd.DataFrame
) -> dict[str, str]:
    """Each model's kernel: the first candidate under which its fixed-k row reproduces the
    published row exactly. A model no candidate reproduces fails the check.

    Only the fixed-k rows sit at the published operating point. A trial runs the whole
    roster under one kernel, so its re-selected shared k is the median of single-kernel k*,
    not the published mixed-kernel one; at equal k its re-selected rows are the fixed-k rows.
    The record run under the assignment is then checked end to end, on both protocols.
    """
    assignment: dict[str, str] = {}
    unmatched: list[str] = []
    for model in published["model"]:
        row = published[published["model"] == model]
        for kernel, frames in trials.items():
            fixed = frames[FIXED_K]
            if not _differing_columns(fixed[fixed["model"] == model], row):
                assignment[str(model)] = kernel
                break
        else:
            unmatched.append(str(model))
    if unmatched:
        raise ReproductionError(
            f"{unmatched} reproduce their published row under none of {list(trials)}"
        )
    return assignment


def check_reproduction(run: ProductionRun, out_dir: Path) -> dict:
    """Assert both protocols of the 100% record equal the production ``metrics.csv``.

    Every value must match exactly -- there is no tolerance. A pass is recorded as
    ``reproduction.json`` beside the record; no subsampled replicate of the cohort runs
    without one.
    """
    report: dict = {"benchmark": run.benchmark, "variants": {}}
    published = _as_written(run.results_dir / "metrics.csv")
    failures = []
    for variant in VARIANTS:
        new = _as_written(Path(out_dir) / variant / "metrics.csv")
        differing = _differing_columns(new, published)
        report["variants"][variant] = {"differing_columns": differing}
        if differing:
            failures.append(f"{variant}: {differing}")
    if failures:
        raise ReproductionError(
            f"{run.benchmark}: the 100% subsample does not reproduce "
            f"{run.results_dir / 'metrics.csv'} -- " + "; ".join(failures)
        )
    report["identical"] = True
    report["columns_compared"] = list(published.columns)
    report["kernel_by_model"] = read_replicate(out_dir)["kernel_by_model"]
    _atomic_write(Path(out_dir) / REPRODUCTION, json.dumps(report, indent=2) + "\n")
    return report


def reproduce(
    run: ProductionRun,
    manifest: pd.DataFrame,
    *,
    out_dir: Path,
    scratch: Path,
    candidates: tuple[str, ...] = CANDIDATE_KERNELS,
) -> dict:
    """The 100% record, and the proof that it is the production run.

    With several candidate kernels, the full cohort is first run under each
    (``out_dir/kernel-trials/<kernel>/``), every encoder is assigned the kernel that
    reproduces its published row, and the record itself is then run under that assignment
    and checked end to end.
    """
    out_dir = Path(out_dir)
    if len(candidates) == 1:
        kernels = {m: candidates[0] for m in run.roster}
    else:
        published = _as_written(run.results_dir / "metrics.csv")
        trials = {}
        for kernel in candidates:
            where = out_dir / "kernel-trials" / kernel
            if not (where / REPLICATE_META).exists():
                run_replicate(
                    run,
                    manifest,
                    out_dir=where,
                    scratch=Path(scratch) / kernel,
                    kernels={m: kernel for m in run.roster},
                )
            trials[kernel] = {v: _as_written(where / v / "metrics.csv") for v in VARIANTS}
        kernels = assign_kernels(trials, published)
    run_replicate(run, manifest, out_dir=out_dir, scratch=scratch, kernels=kernels)
    return check_reproduction(run, out_dir)


def reproduced(root: Path, cohort: str) -> bool:
    """Whether ``cohort``'s 100% run is on disk and passed its reproduction check."""
    path = replicate_dir(root, cohort, FULL, 0) / REPRODUCTION
    if not path.exists():
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("identical") is True


# --------------------------------------------------------------------------- the sweep

Task = tuple[str, int, int]  # (cohort, fraction, replicate)


def task_name(task: Task) -> str:
    cohort, fraction, replicate = task
    return f"{cohort}-{'full' if fraction == FULL else f'f{fraction:02d}-r{replicate:02d}'}"


def finished(root: Path, task: Task) -> bool:
    """A replicate is finished once its ``replicate.json`` -- written last -- exists."""
    return (replicate_dir(root, *task) / REPLICATE_META).exists()


def plan(root: Path, record: dict) -> tuple[list[Task], list[Task]]:
    """What is left: the 100% runs still unchecked, and the subsampled replicates now runnable.

    A cohort's subsampled replicates are runnable only once its 100% run has reproduced the
    production ``metrics.csv``; finished replicates are never planned again, so a killed
    sweep resumes where it stopped.
    """
    full = [(c, FULL, 0) for c in record["cohorts"] if not reproduced(root, c)]
    ready = [
        (cohort, int(fraction), replicate)
        for replicate in range(int(record["n_replicates"]))
        for cohort in record["cohorts"]
        for fraction in record["fractions"]
        if reproduced(root, cohort) and not finished(root, (cohort, int(fraction), replicate))
    ]
    return full, ready


def status(root: Path, record: dict) -> pd.DataFrame:
    """Finished against expected replicates, per cohort x fraction (100 = the full run)."""
    rows = []
    for cohort in record["cohorts"]:
        rows.append(
            {
                "cohort": cohort,
                "fraction": FULL,
                "finished": int(reproduced(root, cohort)),
                "expected": 1,
            }
        )
        for fraction in record["fractions"]:
            n = int(record["n_replicates"])
            done = sum(finished(root, (cohort, int(fraction), r)) for r in range(n))
            rows.append(
                {"cohort": cohort, "fraction": int(fraction), "finished": done, "expected": n}
            )
    return pd.DataFrame(rows)


#: Written once every planned record is on disk; the paper build waits for it.
COMPLETE = "complete.json"


def mark_complete(root: Path, record: dict) -> bool:
    """Write ``complete.json`` when every replicate has finished; return whether it did."""
    table = status(root, record)
    if not bool((table["finished"] == table["expected"]).all()):
        return False
    stamp = {"finished": int(table["finished"].sum()), "design": record}
    _atomic_write(Path(root) / COMPLETE, json.dumps(stamp, indent=2, sort_keys=True) + "\n")
    return True


def run_task(root: Path, task: Task, scratch_root: Path) -> None:
    """One replicate, end to end: draw, run both protocols, record (and prove, at 100%)."""
    cohort, fraction, replicate = task
    out_dir = replicate_dir(root, *task)
    scratch = Path(scratch_root) / "runs" / task_name(task)
    manifest = pd.read_csv(layout.REPO / benchmarks.get(cohort).manifest)
    run = production_run(cohort)
    if fraction == FULL:
        if not reproduced(root, cohort):
            reproduce(run, manifest, out_dir=out_dir, scratch=scratch)
        return
    if finished(root, task):
        return
    if not reproduced(root, cohort):
        raise RuntimeError(f"{cohort}: the 100% run has not reproduced production yet")
    full = replicate_dir(root, cohort, FULL, 0) / REPRODUCTION
    kernels = json.loads(full.read_text(encoding="utf-8"))["kernel_by_model"]
    drawn = draw_subsample(manifest, cohort, fraction, replicate)
    run_replicate(run, drawn, out_dir=out_dir, scratch=scratch, kernels=kernels)


def _log(message: str) -> None:
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _spawn(root: Path, task: Task, scratch_root: Path) -> int:
    """Run one task in a worker process of its own; its output goes to a per-task log."""
    import subprocess
    import time

    cohort, fraction, replicate = task
    name = task_name(task)
    log_path = Path(scratch_root) / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "replicate",
        "--root",
        str(root),
        "--scratch-root",
        str(scratch_root),
        "--cohort",
        cohort,
        "--fraction",
        str(fraction),
        "--replicate",
        str(replicate),
    ]
    _log(f"start  {name}")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        code = subprocess.run(
            argv,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **SINGLE_THREAD},
            cwd=REPO,
        ).returncode
    elapsed = time.time() - start
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        _log(f"FAILED {name} (exit {code}, {elapsed:.0f}s); log {log_path}:")
        for line in tail:
            print(f"    {line}", flush=True)
    else:
        _log(f"done   {name} ({elapsed:.0f}s)")
    return code


def sweep(root: Path, record: dict, *, workers: int, scratch_root: Path) -> int:
    """Run every unfinished replicate on ``workers`` processes, each cohort's 100% check first.

    A cohort's subsampled replicates are queued the moment its reproduction passes, so the
    cheap cohorts run while the expensive one is still being proven.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    write_design(root, record)
    _log(f"design: {json.dumps(record)}")
    full, ready = plan(root, record)
    _log(f"reproduction checks to run: {len(full)}; replicates ready: {len(ready)}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_spawn, root, t, scratch_root): t for t in full + ready}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                cohort, fraction, _ = pending.pop(future)
                if fraction != FULL:
                    continue
                if not reproduced(root, cohort):
                    _log(f"{cohort}: the 100% run did not reproduce production; held")
                    continue
                queued = [t for t in plan(root, record)[1] if t[0] == cohort]
                _log(f"{cohort}: reproduced; queueing {len(queued)} replicates")
                for task in queued:
                    pending[pool.submit(_spawn, root, task, scratch_root)] = task
    _log("status:\n" + status(root, record).to_string(index=False))
    complete = mark_complete(root, record)
    _log("sweep complete" if complete else "sweep incomplete; re-run to resume")
    return 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_sweep = sub.add_parser("sweep", help="run (or resume) the whole study")
    p_status = sub.add_parser("status", help="count finished against expected replicates")
    p_one = sub.add_parser("replicate", help="one replicate (a sweep worker)")
    for p in (p_sweep, p_status, p_one):
        p.add_argument("--root", type=Path, default=None)
    for p in (p_sweep, p_one):
        p.add_argument("--scratch-root", type=Path, required=True)
    p_sweep.add_argument("--workers", type=int, default=4)
    p_one.add_argument("--cohort", required=True, choices=COHORTS)
    p_one.add_argument("--fraction", type=int, required=True)
    p_one.add_argument("--replicate", type=int, required=True)
    p_drive = sub.add_parser("drive", help="one driver run under this process's kernel")
    p_drive.add_argument("--benchmark", required=True, choices=COHORTS)
    p_drive.add_argument("--models", required=True)
    p_drive.add_argument("--pin-k", type=int, required=True)
    p_drive.add_argument("--manifest", type=Path, required=True)
    p_drive.add_argument("--scratch", type=Path, required=True)
    p_drive.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "drive":
        wanted = os.environ.get("OPENBLAS_CORETYPE", "")
        if wanted and numpy_kernel() != wanted:
            raise RuntimeError(f"asked for {wanted}, numpy's OpenBLAS runs {numpy_kernel()}")
        drive(
            production_run(args.benchmark),
            args.models.split(","),
            pin_k=args.pin_k,
            manifest_path=args.manifest,
            scratch=args.scratch,
            dest=args.dest,
        )
        return 0
    # Workers run from the repository, so a relative path is fixed to the caller's directory
    # first -- made absolute, not resolved, so a symlinked output/ keeps its own name.
    root = (args.root if args.root is not None else study_root()).absolute()
    record = design()
    if args.command == "status":
        table = status(root, record)
        print(table.to_string(index=False))
        print(f"finished {int(table['finished'].sum())} / {int(table['expected'].sum())}")
        return 0
    if args.command == "replicate":
        run_task(root, (args.cohort, args.fraction, args.replicate), args.scratch_root.absolute())
        return 0
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    return sweep(root, record, workers=args.workers, scratch_root=args.scratch_root.absolute())


if __name__ == "__main__":
    raise SystemExit(main())
