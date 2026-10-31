"""The confounder-biased probe schedules, reduced to what ``tab:split-schedule`` reports.

The APD/nIPD sweep trains a biology probe on a training set whose centre-biology
correlation is walked from zero to total, one split at a time. Which rows each split
trains on, validates on and holds out is decided by three pieces of code the study already
runs: the schedule (``loaders.split_schedule``), the validation rule
(``apd_experiment.validation_fraction``) and the library sweep's own slicing
(``croma.downstream.probe``). This module calls those three on the cohort's cell sizes and
counts what comes out, so the table cannot describe a protocol the sweep does not run.

Cell sizes are read from the same APD metadata ``loaders.load_data`` reads, mapped onto the
schedule's ``(centre, class)`` indices the same way. A "slide" is the sweep's slide: a
block of ``num_patches_per_slide`` consecutive rows of one cell (a case on Tolkach-ESCA,
one slide embedding on PCaBiop).

``tiles_per_slide`` overrides a benchmark's rows per slide. Reduced-tile variants of the
sweep (``train_rows_per_slide``) keep the full sweep's slides, splits and held-out tail, and
draw fewer tiles from each training and validation slide; the override runs the library's
own thinning over the full-tile slices, so training and validation sizes are the reduced
sweep's and the slide counts are left alone. Both pools are reported as the metadata has
them: the reduced sweep scores the ID pool's held-out tail at full tiles, and the held-out
OOD pool is a flat test set the sweep never slices.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "bench", REPO / "scripts" / "studies" / "apd"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import apd_experiment  # noqa: E402
from croma.downstream.probe import (  # noqa: E402
    _Thinning,
    _validate_thinning,
    _validated_schedule,
)
from loaders import CROMA_SPLIT_MAPS, DATASETS, split_schedule, training_correlations  # noqa: E402
from paper_manifest import by_benchmark  # noqa: E402

#: The schedules the table documents, in the order it lists them. Prostate-shift has a
#: schedule too, but it is outside the manuscript's benchmark panel.
TABLE_DATASETS = ("camelyon", "tcga_4x4", "tolkach", "pcabiop")

#: PCaBiop is the slide-level APD cohort (PANDA in domain, PAR held out), not the ``panda``
#: benchmark run it is correlated against, so it has no manifest short name of its own.
_NAMES = {"pcabiop": "PCaBiop"}


@dataclass(frozen=True)
class Pool:
    """A pool of centres the sweep draws on: in-domain (trained on) or held out (OOD)."""

    centres: tuple[str, ...]
    slides: int
    tiles: int


@dataclass(frozen=True)
class Split:
    """One split of a schedule, in the sweep's own units.

    ``favourable`` holds the training slides per favourable cell -- one value per cell
    type, since a 4x4 schedule shrinks its anti-diagonal and its other off-diagonal cells
    on different trajectories. ``train`` and ``validation`` count rows (tiles; slides on a
    slide-level cohort). ``test_slides`` is the (min, max) held-out slides per cell.
    """

    index: int
    cramers_v: float
    favourable: tuple[int, ...]
    unfavourable: int
    train: int
    validation: int
    test_slides: tuple[int, int]


@dataclass(frozen=True)
class Schedule:
    key: str
    name: str
    tiles_per_slide: int
    #: The cohort's own rows per slide; ``tiles_per_slide`` is below it only under an override.
    rows_per_slide: int
    n_classes: int
    id_pool: Pool
    ood_pool: Pool
    replicates: int
    seed: int
    croma_authored: bool
    #: True where a case contributes tiles to several classes (Tolkach-ESCA), so the
    #: pool's ``slides`` are cases and a sweep slide is one case's block within a class.
    case_based: bool
    validation_fraction: float | None
    favourable_types: tuple[str, ...]
    splits: tuple[Split, ...]


def _name(key: str) -> str:
    return _NAMES.get(key) or by_benchmark(DATASETS[key]["benchmark"]).short_name


def _pool(md: pd.DataFrame, centres: list[str]) -> Pool:
    missing = sorted(set(centres) - set(md["medical_center"]))
    if missing:
        raise ValueError(f"the metadata holds no rows for centre(s) {missing}")
    return Pool(tuple(centres), slides=int(md["slide_id"].nunique()), tiles=len(md))


def _cell_slides(md_id: pd.DataFrame, cfg: dict) -> list[list[int]]:
    """Slides per ``(centre, class)`` cell, indexed as the schedule addresses cells."""
    npps = cfg["num_patches_per_slide"]
    counts = md_id.groupby(["medical_center", "biological_class"]).size()
    grid = []
    for centre in cfg["centers_id"]:
        row = []
        for label in cfg["biological_classes"]:
            rows = int(counts.get((centre, label), 0))
            if rows % npps:
                raise ValueError(
                    f"cell ({centre}, {label}) holds {rows} rows, not a whole number of "
                    f"{npps}-row slides"
                )
            row.append(rows // npps)
        grid.append(row)
    return grid


def _cell_types(split_maps: list[list[tuple[int, int, int]]], n_centres: int):
    """Split the cells into the unfavourable type and one or more favourable types.

    A cell is unfavourable when the schedule grows it -- it carries more training rows at
    the fully confounded split than at the balanced one -- because it is the cell the
    shortcut aligns centre with class in. Every other cell is favourable; cells sharing a
    trajectory across the splits form one type, listed in the order their first cell is.
    """
    trajectories: dict[tuple[int, int], tuple[int, ...]] = {}
    for split_map in split_maps:
        for i, j, take in split_map:
            trajectories.setdefault((i, j), ())
            trajectories[(i, j)] += (take,)
    unfavourable = {c for c, t in trajectories.items() if t[-1] > t[0]}
    types: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for cell in sorted(trajectories):
        if cell not in unfavourable:
            types.setdefault(trajectories[cell], []).append(cell)
    if not unfavourable or not types:
        raise ValueError("a schedule must both grow and shrink some cells")
    grown = {trajectories[c] for c in unfavourable}
    if len(grown) != 1:
        raise ValueError(f"the unfavourable cells follow {len(grown)} trajectories, not one")

    def kind(cells: list[tuple[int, int]]) -> str:
        return "anti-diagonal" if all(i + j == n_centres - 1 for i, j in cells) else "off-diagonal"

    return next(iter(unfavourable)), [cells[0] for cells in types.values()], tuple(
        kind(cells) for cells in types.values()
    )


def schedule(key: str, metadata: pd.DataFrame, *, tiles_per_slide: int | None = None) -> Schedule:
    """Reduce one benchmark's schedule over its APD metadata to the table's numbers."""
    cfg = DATASETS[key]
    npps = int(cfg["num_patches_per_slide"])
    tiles = int(tiles_per_slide or npps)
    md_id = metadata[metadata["subset"] == "ID"]
    md_ood = metadata[metadata["subset"] == "OOD"]
    slides = _cell_slides(md_id, cfg)

    # The cells as the sweep sees them: row indices, ``npps`` per slide, in schedule order.
    cells, owner, start = [], {}, 0
    for i, row in enumerate(slides):
        cells.append([])
        for j, n in enumerate(row):
            cells[i].append(list(range(start, start + n * npps)))
            owner.update({r: (i, j) for r in cells[i][j]})
            start += n * npps

    plan_specs = split_schedule(key, npps)
    fraction = apd_experiment.validation_fraction(npps)
    plans = _validated_schedule(
        plan_specs,
        cells=cells,
        rows_per_slide=npps,
        validation_fraction=fraction,
        n_classes=len(cfg["biological_classes"]),
    )
    unfavourable, favourable, kinds = _cell_types(
        [plan.split_map for plan in plans], len(cfg["centers_id"])
    )
    # The sizes the thinning keeps depend only on the cell sizes, never on which rows the
    # seed draws, so any replicate seed gives the reduced sweep's counts.
    if tiles != npps:
        _validate_thinning(plans, npps, tiles)
    thinning = (
        None
        if tiles == npps
        else _Thinning(
            cells, rows_per_slide=npps, train_rows_per_slide=tiles, n_rows=start, replicate_seed=0
        )
    )

    splits = []
    for index, (plan, v) in enumerate(zip(plans, training_correlations(key))):
        takes = {(i, j): take for i, j, take in plan.split_map}
        train, validation, held_out = plan.slice(cells, thinning)
        per_cell: dict[tuple[int, int], int] = {}
        for r in held_out:
            per_cell[owner[r]] = per_cell.get(owner[r], 0) + 1
        test = [per_cell.get((i, j), 0) // npps for i in range(len(cells)) for j in range(len(cells[i]))]
        splits.append(
            Split(
                index=index,
                cramers_v=v,
                favourable=tuple(takes[c] // npps for c in favourable),
                unfavourable=takes[unfavourable] // npps,
                train=len(train),
                validation=len(validation),
                test_slides=(min(test), max(test)),
            )
        )

    return Schedule(
        key=key,
        name=_name(key),
        tiles_per_slide=tiles,
        rows_per_slide=npps,
        n_classes=len(cfg["biological_classes"]),
        id_pool=_pool(md_id, cfg["centers_id"]),
        ood_pool=_pool(md_ood, cfg["centers_ood"]),
        replicates=apd_experiment.CANONICAL_ITERATIONS,
        seed=apd_experiment.PROBE_SEED,
        croma_authored=cfg["split_key"] in CROMA_SPLIT_MAPS,
        case_based="case_splits" in cfg,
        validation_fraction=fraction,
        favourable_types=kinds,
        splits=tuple(splits),
    )


def schedules(root: Path = REPO, tiles_per_slide: Mapping[str, int] | None = None) -> list[Schedule]:
    """Every documented schedule, over the APD metadata under ``root``."""
    overrides = dict(tiles_per_slide or {})
    unknown = sorted(set(overrides) - set(TABLE_DATASETS))
    if unknown:
        raise ValueError(f"no schedule to override for {unknown}; known: {TABLE_DATASETS}")
    out = []
    for key in TABLE_DATASETS:
        path = root / DATASETS[key]["metadata"]
        if not path.exists():
            raise FileNotFoundError(f"{key}: APD metadata {path} is missing")
        out.append(schedule(key, pd.read_csv(path), tiles_per_slide=overrides.get(key)))
    return out
