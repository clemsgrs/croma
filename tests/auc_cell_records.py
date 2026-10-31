"""The two records an AUROC-study test tree is built from: an APD cell, and its AUROC cell.

Shared by ``test_auc_study.py`` (the study, which ships in the sdist) and
``test_nipd_auroc_paper.py`` (its paper-dependent half, which does not).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def apd_cell(path: Path, id_acc: np.ndarray, ood_acc: np.ndarray) -> None:
    """The two matrices of an APD study cell, which is all an AUROC cell is checked against."""
    write_json(
        path,
        {"id_test_accuracies": id_acc.tolist(), "ood_test_accuracies": ood_acc.tolist()},
    )


def auc_cell(
    path: Path,
    id_acc: np.ndarray,
    ood_acc: np.ndarray,
    *,
    id_auroc: np.ndarray | None = None,
    ood_auroc: np.ndarray | None = None,
) -> None:
    """A stored AUROC cell: per (split, replicate) AUROC and balanced accuracy, both arms."""
    id_auroc = id_acc + 0.05 if id_auroc is None else id_auroc
    ood_auroc = ood_acc + 0.05 if ood_auroc is None else ood_auroc
    rows = np.stack([id_auroc, id_acc, ood_auroc, ood_acc, np.ones_like(id_acc)], axis=-1)
    write_json(
        path,
        {
            "columns": ["auroc_id", "balacc_id", "auroc_ood", "balacc_ood", "best_c"],
            "iterations": int(id_acc.shape[1]),
            "rows": rows.tolist(),
        },
    )
