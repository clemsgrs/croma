"""Rebuild every generated artifact under ``paper/sections/`` from the current runs.

This is the single entrypoint. Run it after any benchmark re-run; the freshness test
(``tests/test_paper_artifacts.py``) fails if you forget. See ADR-0010.

It does not compute anything. Metrics come from ``output/metrics/<protocol>/<benchmark>/``,
produced by ``scripts/repro/run_benchmarks.sh``; model metadata comes from
``scripts/repro/model_metadata.csv``. This script only re-renders LaTeX from them.

Usage:
  python scripts/repro/build_paper.py            # rebuild everything it can
  python scripts/repro/build_paper.py --check    # report drift, write nothing, exit 1 if stale
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from paper_manifest import by_prefix  # noqa: E402

_CAMELYON = by_prefix("Camelyon")

#: (script, required-input, why-it-may-be-absent). Every core generator reads only a
#: benchmark ``metrics.csv`` or the committed metadata CSV, so all of them always run.
CORE = [
    ("generate_results_table.py", None, ""),
    ("generate_panda_table.py", None, ""),
    ("generate_model_tables.py", None, ""),
    ("generate_paper_values.py", None, ""),
    # Reads the three tile runs' metrics.csv + per_sample_metrics.csv (Camelyon for the main
    # float, TCGA-4x4 + Tolkach-ESCA for the two supp floats); the distribution PDFs it
    # captions are drawn separately by figures/croma_distribution_figure.py.
    ("generate_distribution_floats.py", None, ""),
    # Reads the three tile runs' metrics.csv + per_sample_metrics.csv (Camelyon for the main
    # median-vs-tail Pareto float, TCGA-4x4 + Tolkach-ESCA for the two supp floats); the PDFs
    # it captions are drawn by figures/croma_pareto_figure.py.
    ("generate_pareto_float.py", None, ""),
    # Reads the three tile runs' metrics.csv + the committed model_metadata.csv (via
    # _cross_benchmark) to build the rank-aggregate Pareto overview float; the PDF it captions
    # is drawn by figures/rank_pareto_figure.py.
    ("generate_rank_pareto_float.py", None, ""),
]

#: Generators whose input is a *study* artifact rather than a benchmark run. A study that
#: has not been run leaves no artifact, and its table cannot be rebuilt. These skips are
#: reported by name -- a silent skip is how a cited macro once vanished from the paper.
STUDIES = [
    (
        "generate_uncertainty_supp_table.py",
        f"{_CAMELYON.run_rel}/results/bootstrap_uncertainty.json",
        "run scripts/studies/bootstrap_uncertainty.py",
    ),
    (
        "generate_supp_rank_table.py",
        f"{_CAMELYON.studies_rel}/typed_neighbor_rank_summary.csv",
        "run scripts/repro/figures/typed_neighbor_rank_experiment.py",
    ),
    (
        "generate_pretraining_overlap_table.py",
        "output/studies/pretraining-overlap/per_sample_metrics.csv",
        "run scripts/studies/pretraining_overlap.py",
    ),
    # NOTE: generate_cross_benchmark_float.py (the rank bump chart, fig:cross-benchmark) is
    # deliberately not built: the manuscript retired that float in favour of the rank-aggregate
    # Pareto (fig:croma-pareto-rank), and nothing in the reachable tree references it. The
    # generator is kept for its caption-provenance unit tests but produces no paper artifact.
    (
        "generate_apd_floats.py",
        "output/studies/apd/apd_correlation.csv",
        "run scripts/studies/apd/apd_experiment.py, then apd_croma_correlation.py",
    ),
    # Reads each tile run's croma_m_sweep_metrics.csv (Camelyon guarded as representative;
    # the three are produced together). Absent only when a run was benchmarked without the
    # CRoMa averaging-radius sweep, in which case tab:m-sweep cannot be rebuilt.
    (
        "generate_m_sweep_table.py",
        f"{_CAMELYON.run_rel}/results/croma_m_sweep_metrics.csv",
        "re-run the benchmark with the CRoMa m-sweep enabled",
    ),
]


def _run(script: str, extra: list[str]) -> int:
    proc = subprocess.run([sys.executable, str(HERE / script), *extra], cwd=REPO)
    return proc.returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Report drift against what is on disk; write nothing.")
    args = p.parse_args()

    if args.check:
        # Only the results tables render deterministically from a pure function today, so
        # they are what --check compares. The remaining generators are checked by rebuilding.
        return _run("generate_results_table.py", ["--check"])

    failed: list[str] = []
    for script, _, _ in CORE:
        print(f"\n=== {script} ===", flush=True)
        if _run(script, []) != 0:
            failed.append(script)

    skipped: list[tuple[str, str]] = []
    for script, required, remedy in STUDIES:
        if required and not (REPO / required).exists():
            skipped.append((script, f"missing {required} -- {remedy}"))
            continue
        print(f"\n=== {script} ===", flush=True)
        if _run(script, []) != 0:
            failed.append(script)

    print("\n" + "=" * 70)
    for script, why in skipped:
        print(f"SKIPPED  {script}: {why}")
    if failed:
        print(f"FAILED   {', '.join(failed)}")
        return 1
    print(f"OK       {len(CORE) + len(STUDIES) - len(skipped)} generator(s) ran, "
          f"{len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
