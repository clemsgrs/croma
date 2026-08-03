"""Recompute every number the paper reports, then rebuild the paper artifacts.

One command. For each entry in ``paper_manifest`` it re-runs the compute-only
``benchmark.py`` *with the knobs that run was originally invoked with*, then runs the
studies whose outputs ``build_paper.py`` consumes, then rebuilds ``paper/``.

Two things this script used to get wrong, both of which made a "reproduction" prove less
than it appeared to:

* **It passed no ``--k-grid``.** The paper's runs swept PathoROB's *sparse* grid; the flag
  defaults to *dense*. The operating point is chosen from the swept values, so re-running
  moved it -- on TCGA-4x4 from k=71 to k=69, shifting bio bacc by up to 0.0031. Nothing
  recorded the grid, so nothing could notice. Runs now carry a ``run_config.json`` sidecar
  and this script replays it (``scripts/bench/run_config.py``); runs predating the sidecar
  are backfilled from their own ``metrics.csv``, which records the swept ``k_values``
  verbatim. The grid is deliberately *not* pinned here -- a constant in this file is exactly
  the second source of truth the two notes below record removing.
* **It did not force recomputation.** Without ``--recompute-metrics`` the metrics artifact
  cache short-circuits the run, so the script replayed cached numbers and would have
  "reproduced" the paper even if the code that computed them had changed underneath. It now
  recomputes by default; ``--reuse-cache`` is available for a fast smoke run and says in its
  own name that it proves less.

Two constants also used to live here and no longer do, because both were second sources of
truth that silently diverged from the runs (ADR-0010):

* a hard-coded 16-model ``MODELS`` roster -- the panel is whatever the benchmark has
  embeddings for, and pinning it here kept the paper at 16 models after it grew to 21;
* a ``PROTOCOL = "k-star"`` pin -- the protocol belongs to the benchmark, and this pin had
  the tables rendered from k-star runs while the prose macros were computed from median-k.

Scope: the *numbers*. Figures are rendered beside their data (``scripts/bench/render.py``,
``scripts/repro/figures/``) and curated into ``paper/figures/`` by hand; ADR-0010 keeps that
step manual, and ``scripts/repro/check_paper_figures.py`` audits it.

Usage:
    python scripts/repro/reproduce_faithful.py               # every run, studies, paper
    python scripts/repro/reproduce_faithful.py Camelyon      # one run, by manifest prefix
    python scripts/repro/reproduce_faithful.py --dry-run     # print the commands, run nothing
    python scripts/repro/reproduce_faithful.py --runs-only   # skip studies and build_paper
    python scripts/repro/reproduce_faithful.py --jobs 8      # parallelism for the APD sweep
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from paper_manifest import TABLES

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "bench"))  # noqa: E402  (run_config lives with the driver it configures)
from run_config import (  # noqa: E402
    check_resolved,
    config_path,
    infer_replay_from_metrics,
    read_run_config,
    replay_args,
    write_run_config,
)

#: The studies ``build_paper.py``'s STUDIES generators read. Invoked bare on purpose: each
#: already defaults to the settings the paper used (``n_boot=2000`` with a fixed seed;
#: ``--iterations 20``), and re-declaring those here would be one more list to drift.
#: ``--jobs`` is the exception -- it is a property of the machine, not of the protocol.
STUDIES: list[tuple[str, str, list[str]]] = [
    ("pretraining-overlap", "scripts/studies/pretraining_overlap.py", []),
    ("typed-neighbour-ranks", "scripts/repro/figures/typed_neighbor_rank_experiment.py", []),
    ("bootstrap-uncertainty", "scripts/studies/bootstrap_uncertainty.py", []),
    # Last because it is far and away the longest: a full probe sweep over every
    # model-dataset pair, hours rather than minutes.
    ("apd-sweep", "scripts/studies/apd/apd_experiment.py", ["--overwrite"]),
    ("apd-correlation", "scripts/studies/apd/apd_croma_correlation.py", []),
]


def _run(cmd: list[str], *, label: str, dry_run: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[{label}] {printable}", flush=True)
        return
    print(f"[{label}] {printable}", flush=True)
    started = time.monotonic()
    subprocess.run(cmd, check=True, cwd=REPO)
    print(f"[{label}] done in {time.monotonic() - started:.0f}s", flush=True)


def _replay_for(entry, *, dry_run: bool) -> list[str]:
    """The recorded invocation for this run, backfilling the sidecar if it predates one."""
    run_dir = REPO / entry.run_rel
    config = read_run_config(run_dir)
    if config is None:
        metrics_csv = REPO / entry.metrics_rel
        if not metrics_csv.exists():
            raise SystemExit(
                f"[{entry.prefix}] no run_config.json and no {entry.metrics_rel} to infer "
                "one from. This benchmark has never been run; there is nothing to reproduce "
                "-- run scripts/bench/benchmark.py once to establish it."
            )
        replay = infer_replay_from_metrics(metrics_csv)
        config = {"replay": replay, "resolved": {"benchmark": entry.benchmark,
                                                 "protocol": entry.protocol}}
        print(
            f"[{entry.prefix}] no run_config.json; inferred k_grid={replay['k_grid']} "
            f"k_max={replay['k_max']} from {entry.metrics_rel}",
            flush=True,
        )
        if not dry_run:
            write_run_config(
                results_dir=run_dir / "results",
                replay=replay,
                resolved=dict(config["resolved"]),
            )
            print(f"[{entry.prefix}] backfilled {config_path(run_dir)}", flush=True)
    else:
        check_resolved(config, benchmark=entry.benchmark, protocol=entry.protocol)
    return replay_args(config)


def run_benchmark(entry, *, recompute: bool, dry_run: bool) -> None:
    cmd = [
        sys.executable, str(REPO / "scripts/bench/benchmark.py"),
        "--benchmark", entry.benchmark,
        "--protocol", entry.protocol,
        *_replay_for(entry, dry_run=dry_run),
        "--progress", "off",
    ]
    if recompute:
        cmd.append("--recompute-metrics")
    _run(cmd, label=entry.prefix, dry_run=dry_run)
    if not dry_run:
        print(f"[{entry.prefix}] metrics -> {entry.metrics_rel}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prefixes", nargs="*",
                   help="Manifest prefixes to re-run; default every entry.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every command and run nothing.")
    p.add_argument("--runs-only", action="store_true",
                   help="Re-run the benchmarks; skip the studies and build_paper.py.")
    p.add_argument("--skip-studies", action="store_true",
                   help="Re-run the benchmarks and build_paper.py; skip the studies.")
    p.add_argument("--reuse-cache", action="store_true",
                   help="Allow the metrics artifact cache. Fast, but proves only that the "
                        "cached numbers are still on disk -- not that they recompute.")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel worker processes for the APD sweep (default 1).")
    args = p.parse_args()

    entries = list(TABLES)
    if args.prefixes:
        known = {t.prefix for t in TABLES}
        unknown = sorted(set(args.prefixes) - known)
        if unknown:
            raise SystemExit(
                f"unknown manifest prefix(es): {unknown}; known: {sorted(known)}"
            )
        entries = [t for t in TABLES if t.prefix in set(args.prefixes)]

    for entry in entries:
        run_benchmark(entry, recompute=not args.reuse_cache, dry_run=args.dry_run)

    if args.runs_only:
        print("[repro] --runs-only: stopping before studies and build_paper.py", flush=True)
        return 0

    if not args.skip_studies:
        for label, script, extra in STUDIES:
            cmd = [sys.executable, str(REPO / script), *extra]
            if label == "apd-sweep" and args.jobs > 1:
                cmd += ["--jobs", str(int(args.jobs))]
            _run(cmd, label=label, dry_run=args.dry_run)

    _run([sys.executable, str(REPO / "scripts/repro/build_paper.py")],
         label="build-paper", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
