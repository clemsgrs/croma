"""Record AUROC beside balanced accuracy for one (cohort, encoder) probe sweep.

The sweep is the APD study's own: :func:`run_cell` calls ``apd_experiment.compute`` and
observes every probe it trains, scoring each with a threshold-free metric as well. Nothing
about the protocol (schedule, seed, replicate count, validation rule) is restated here, so
the cell cannot drift from the APD cell it rides on. A pure decision-boundary shift moves
balanced accuracy and leaves AUROC alone, which is what this study exists to separate.

Multiclass cohorts get macro one-vs-rest AUROC on the raw class margins
(``decision_function``), not on softmax probabilities: a per-class intercept shift is
exactly the drift under investigation, and the softmax couples every class's score to
every other one's, so probabilities would smuggle the drift back in.

A cell is written only once its balanced accuracies reproduce the current APD cell (see
``auc_cells``), so its AUROC is believed only for probes the published nIPD was computed
from. Usage::

    python scripts/studies/auc/probe_auc_cell.py --dataset camelyon --model UNI
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
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# auc_cells puts the APD study (and through its loaders, src/) on sys.path.
from auc_cells import (  # noqa: E402
    APD_DIR,
    AUC_DIR,
    STALE_TOLERANCE,
    balanced_accuracy_gap,
    cell_path,
    cell_status,
)

import apd_experiment  # noqa: E402
from apd_experiment import CANONICAL_ITERATIONS, PROBE_SEED, _atomic_write_text  # noqa: E402
from loaders import DATASETS  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

#: Per (split, replicate): each arm's AUROC and balanced accuracy, and the probe's C.
COLUMNS = ["auroc_id", "balacc_id", "auroc_ood", "balacc_ood", "best_c"]


def auroc(model, features, labels) -> float:
    """Macro one-vs-rest AUROC on raw class margins; plain AUROC when binary."""
    labels = np.asarray(labels)
    margins = model.decision_function(features)
    if margins.ndim == 1:
        return float(roc_auc_score(labels, margins))
    per_class = [
        roc_auc_score((labels == label).astype(int), margins[:, column])
        for column, label in enumerate(model.classes_)
        if (labels == label).any() and not (labels == label).all()
    ]
    return float(np.mean(per_class))


def _observed_sweep(model: str, dataset: str, iterations: int) -> tuple[dict, np.ndarray]:
    """``apd_experiment.compute`` with every probe it trains also scored by AUROC.

    Returns the APD study's result unchanged and the per-probe record, shaped
    ``(n_splits, iterations, len(COLUMNS))``.
    """
    # Patch the trainer in the module the APD sweep resolves it from, whichever copy of
    # croma that is, so the observer cannot miss the probes.
    sweep = apd_experiment.probe_sweep_over_test_sets
    probe_module = sys.modules[sweep.__module__]
    original = probe_module.train_logistic_regression
    record = []

    def observe(train_x, train_y, val_x, val_y, test_xs, test_ys, **kwargs):
        fitted, best_c, scores = original(
            train_x, train_y, val_x, val_y, test_xs, test_ys, **kwargs
        )
        # test_xs[0] is the held-out in-domain arm and [1] the unseen-confounder arm: the
        # sweep builds them in that order, and this observes the very same probes.
        record.append(
            [
                auroc(fitted, test_xs[0], test_ys[0]),
                float(scores[0]),
                auroc(fitted, test_xs[1], test_ys[1]),
                float(scores[1]),
                float(best_c),
            ]
        )
        return fitted, best_c, scores

    probe_module.train_logistic_regression = observe
    try:
        result = apd_experiment.compute(model, dataset, iterations=iterations)
    finally:
        probe_module.train_logistic_regression = original

    # The sweep loops replicate-major (outer loop replicate, inner loop split), so the flat
    # record reshapes (iterations, splits) and transposes to the study's orientation.
    splits = len(result["id_test_accuracies"])
    rows = np.asarray(record, dtype=float).reshape(iterations, splits, len(COLUMNS))
    return result, rows.transpose(1, 0, 2)


def run_cell(
    dataset: str,
    model: str,
    *,
    iterations: int = CANONICAL_ITERATIONS,
    auc_dir: Path = AUC_DIR,
    apd_dir: Path = APD_DIR,
) -> str:
    """Write the AUROC cell for ``(dataset, model)`` unless a current one exists.

    Returns ``"skip"`` for a cell that exists and is current, ``"done"`` for one written
    now. A stale cell is re-run and replaced. Raises, writing nothing, when there is no
    APD cell to ride on or when the fresh sweep does not reproduce it.
    """
    apd_path = cell_path(apd_dir, dataset, model)
    if not apd_path.exists():
        raise FileNotFoundError(f"{dataset}/{model}: no APD cell at {apd_path} to ride on")
    if cell_status(dataset, model, auc_dir=auc_dir, apd_dir=apd_dir).state == "current":
        print(f"[skip] {dataset}/{model}", flush=True)
        return "skip"

    started = time.time()
    result, rows = _observed_sweep(model, dataset, iterations)
    payload = {
        "dataset": dataset,
        "model": model,
        "iterations": iterations,
        "probe_seed": PROBE_SEED,
        "columns": COLUMNS,
        "rows": rows.tolist(),
        "seconds": time.time() - started,
    }
    # The observer must not have changed the sweep, and the sweep must still be the one the
    # APD cell holds; either failing means the AUROC would describe other probes.
    if balanced_accuracy_gap(payload, result) != 0.0:
        raise RuntimeError(f"{dataset}/{model}: the observed probes are not the sweep's")
    gap = balanced_accuracy_gap(payload, json.loads(apd_path.read_text(encoding="utf-8")))
    if gap > STALE_TOLERANCE:
        raise ValueError(
            f"{dataset}/{model}: the sweep does not reproduce the APD cell at {apd_path} "
            f"(largest balanced-accuracy gap {gap:.1e}); re-run that APD cell first. "
            "Nothing was written."
        )
    _atomic_write_text(cell_path(auc_dir, dataset, model), json.dumps(payload) + "\n")

    means = rows.mean(axis=1)
    print(
        f"[done] {dataset}/{model}: {payload['seconds']:.0f}s  apd_gap={gap:.1e}\n"
        f"  AUROC  ID {means[0, 0]:.3f} -> {means[-1, 0]:.3f}   "
        f"OOD {means[0, 2]:.3f} -> {means[-1, 2]:.3f}\n"
        f"  balacc ID {means[0, 1]:.3f} -> {means[-1, 1]:.3f}   "
        f"OOD {means[0, 3]:.3f} -> {means[-1, 3]:.3f}",
        flush=True,
    )
    return "done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--iterations", type=int, default=CANONICAL_ITERATIONS)
    args = parser.parse_args()
    # lbfgs hits its iteration cap on the smallest C values of the grid search; that is the
    # protocol's behaviour, not news, and it would bury the per-cell summary in the log.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    run_cell(args.dataset, args.model, iterations=args.iterations)
