"""Which AUROC cells exist, and whether each still reproduces the APD cell it rides on.

An AUROC cell is the APD study's probe sweep for one (cohort, encoder), re-run with every
probe additionally scored by AUROC. So the cell list is the APD study's: one AUROC cell per
``output/studies/apd/<cohort>/<encoder>.json`` on the cohorts the downstream correlation
covers. Nothing here declares a roster.

A cell is only believed while its balanced accuracies are the APD cell's, so the AUROC beside
them comes from the very probes the published nIPD was computed from. That is checked against
the APD cell as it is *now*: when an encoder's APD cell is re-run (a new preprocessing recipe,
a protocol fix), every AUROC cell riding on the old one turns stale by itself.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "apd") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "apd"))

from loaders import DATASETS, JOIN_KEYS, REPO, STUDY_DIR as APD_DIR  # noqa: E402

#: Where the AUROC cells live, one ``<cohort>/<encoder>.json`` per APD cell, and where the
#: APD cells they ride on live, both relative to a checkout root.
AUC_REL = Path("output/studies/auc")
APD_REL = APD_DIR.relative_to(REPO)
AUC_DIR = REPO / AUC_REL

#: Largest balanced-accuracy difference from the APD cell a current cell may carry. The
#: sweep is deterministic, so a reproduced cell differs by float noise at most (0.0 in
#: practice); a re-preprocessed encoder moves its accuracies by 1e-3 or more.
STALE_TOLERANCE = 1e-9

#: Which AUROC-cell column holds each APD-cell matrix.
_BALANCED_ACCURACY_COLUMNS = {
    "id_test_accuracies": "balacc_id",
    "ood_test_accuracies": "balacc_ood",
}

CellState = Literal["missing", "stale", "current"]


@dataclass(frozen=True)
class CellStatus:
    """Whether one AUROC cell exists and still reproduces its APD cell.

    ``gap`` is the largest absolute balanced-accuracy difference from the APD cell over
    both arms, every split and every replicate: ``nan`` for a missing cell and ``inf`` for
    one whose matrices no longer have the APD cell's shape.
    """

    dataset: str
    model: str
    state: CellState
    gap: float


def cell_path(directory: Path, dataset: str, model: str) -> Path:
    return Path(directory) / dataset / f"{model}.json"


def balanced_accuracy_gap(auc_record: dict, apd_record: dict) -> float:
    """Largest |balanced accuracy| difference between an AUROC cell and an APD cell."""
    rows = np.asarray(auc_record["rows"], dtype=float)
    columns = list(auc_record["columns"])
    gap = 0.0
    for apd_key, column in _BALANCED_ACCURACY_COLUMNS.items():
        mine = rows[..., columns.index(column)]
        theirs = np.asarray(apd_record[apd_key], dtype=float)
        if mine.shape != theirs.shape:
            return float("inf")
        gap = max(gap, float(np.abs(mine - theirs).max()))
    return gap


def cell_status(
    dataset: str, model: str, *, auc_dir: Path = AUC_DIR, apd_dir: Path = APD_DIR
) -> CellStatus:
    """Compare the stored AUROC cell for ``(dataset, model)`` with the current APD cell."""
    stored = cell_path(auc_dir, dataset, model)
    if not stored.exists():
        return CellStatus(dataset, model, "missing", float("nan"))
    gap = balanced_accuracy_gap(
        json.loads(stored.read_text(encoding="utf-8")),
        json.loads(cell_path(apd_dir, dataset, model).read_text(encoding="utf-8")),
    )
    return CellStatus(dataset, model, "current" if gap <= STALE_TOLERANCE else "stale", gap)


def expected_cells(apd_dir: Path = APD_DIR) -> list[tuple[str, str]]:
    """Every (cohort, encoder) the APD study has a cell for, on the correlated cohorts."""
    return [
        (dataset, path.stem)
        for dataset in JOIN_KEYS
        for path in sorted(Path(apd_dir, dataset).glob("*.json"))
    ]


def sweep_size(dataset: str) -> int:
    """Training rows one sweep fits over, summed across its splits: its relative cost.

    Every split trains on up to ``num_slides_per_category`` slides per (centre, class) cell.
    It orders the cohorts the way their cells' wall-clock times do (Camelyon in hours,
    PCaBiop in minutes); it is not a time estimate.
    """
    config = DATASETS[dataset]
    cells = len(config["centers_id"]) * len(config["biological_classes"])
    slides = config["num_slides_per_category"] * config["num_patches_per_slide"]
    return config["num_splits"] * cells * slides


def all_statuses(*, auc_dir: Path = AUC_DIR, apd_dir: Path = APD_DIR) -> list[CellStatus]:
    """The status of every expected cell, in :func:`expected_cells` order."""
    return [
        cell_status(dataset, model, auc_dir=auc_dir, apd_dir=apd_dir)
        for dataset, model in expected_cells(apd_dir)
    ]


def pending_cells(*, auc_dir: Path = AUC_DIR, apd_dir: Path = APD_DIR) -> list[tuple[str, str]]:
    """Every missing or stale cell, the cohort with the largest sweep first."""
    pending = [
        (status.dataset, status.model)
        for status in all_statuses(auc_dir=auc_dir, apd_dir=apd_dir)
        if status.state != "current"
    ]
    return sorted(pending, key=lambda cell: -sweep_size(cell[0]))


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="List the AUROC cells still to run, or check that every cell is current."
    )
    parser.add_argument(
        "command",
        choices=["pending", "check"],
        help="pending: print 'cohort<TAB>encoder' per missing or stale cell, slowest first; "
        "check: print every cell's status and exit 1 unless all are current",
    )
    parser.add_argument("--auc-dir", type=Path, default=AUC_DIR)
    parser.add_argument("--apd-dir", type=Path, default=APD_DIR)
    args = parser.parse_args(argv)

    if args.command == "pending":
        for dataset, model in pending_cells(auc_dir=args.auc_dir, apd_dir=args.apd_dir):
            print(f"{dataset}\t{model}")
        return 0

    statuses = all_statuses(auc_dir=args.auc_dir, apd_dir=args.apd_dir)
    for status in statuses:
        print(f"{status.state:8s} {status.dataset}/{status.model}  gap={status.gap:.1e}")
    count = {
        state: sum(s.state == state for s in statuses) for state in ("current", "missing", "stale")
    }
    print(
        f"{count['current']} of {len(statuses)} expected cells current "
        f"({count['missing']} missing, {count['stale']} stale)"
    )
    return 0 if count["current"] == len(statuses) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
