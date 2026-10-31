"""Run every missing or stale AUROC cell, ``--jobs`` at a time, then check them all.

The work list is ``auc_cells.pending_cells``: one cell per APD-study cell, less those that
exist and still reproduce their APD cell, the slowest cohort first. Each cell runs in its
own ``probe_auc_cell.py`` process with BLAS pinned to one thread and publishes its JSON
atomically, so the sweep is resumable: re-running it after a preemption, or after an APD
re-run made cells stale, does only the work left. It exits 0 only once every expected
cell is current.

    python scripts/studies/auc/sweep.py --jobs 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from auc_cells import APD_DIR, AUC_DIR, all_statuses, pending_cells  # noqa: E402

#: One BLAS thread per cell process: the probe grid search parallelises poorly inside a
#: process, so the sweep fans out across cells instead.
_BLAS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

Runner = Callable[[str, str], int]


def run_cell_process(dataset: str, model: str) -> int:
    """One cell in its own interpreter; returns its exit code."""
    env = {**os.environ, **{name: "1" for name in _BLAS}}
    command = [sys.executable, str(HERE / "probe_auc_cell.py"), "--dataset", dataset]
    return subprocess.run([*command, "--model", model], env=env).returncode


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(
    jobs: int = 1,
    *,
    auc_dir: Path = AUC_DIR,
    apd_dir: Path = APD_DIR,
    runner: Runner = run_cell_process,
) -> int:
    """Run every pending cell, ``jobs`` at a time; 0 once every expected cell is current."""
    cells = pending_cells(auc_dir=auc_dir, apd_dir=apd_dir)
    print(f"{_stamp()} start: {len(cells)} cell(s), {jobs} worker(s)", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        codes = list(pool.map(lambda cell: runner(*cell), cells))
    for (dataset, model), code in zip(cells, codes):
        if code:
            print(f"[failed] {dataset}/{model} (exit {code})", flush=True)

    statuses = all_statuses(auc_dir=auc_dir, apd_dir=apd_dir)
    current = sum(status.state == "current" for status in statuses)
    print(f"{_stamp()} {current} of {len(statuses)} expected cells current", flush=True)
    return 0 if current == len(statuses) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=1, help="cells run at once (default 1)")
    raise SystemExit(run(parser.parse_args().jobs))
