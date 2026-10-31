"""Rebuild every generated artifact under ``paper/sections/`` from the current runs.

This is the single entrypoint. Run it after any benchmark re-run; the freshness test
(``tests/test_paper_artifacts.py``) fails if you forget. See ADR-0010.

It does not compute anything. Metrics come from ``output/metrics/<protocol>/<benchmark>/``,
produced by ``scripts/repro/run_benchmarks.sh``; model metadata comes from
``scripts/bench/model_metadata.csv``. This script only re-renders LaTeX from them.

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
    # Reads the APD metadata under data/ through the study's own schedule and the library
    # sweep's slicing; it fails, rather than skips, when that metadata is missing.
    ("generate_split_schedule_table.py", None, ""),
    # Reads every listed benchmark's view, APD metadata and tileset manifest, one tile image
    # per slide, and the committed acquisition.csv of curated scanner facts.
    ("generate_acquisition_table.py", None, ""),
    # Reads output/studies/mari-vs-ri/ (scripts/studies/mari_vs_ri.py) for tab:mari-vs-ri and
    # tab:mari-tau-sweep. A study input, but listed here rather than under STUDIES on purpose:
    # the tables and their macros are cited, so a missing study must fail the build, not skip.
    ("generate_mari_vs_ri_tables.py", None, ""),
    # Reads the robustness fine-tune vs parent study (scripts/studies/finetune_pairs.py) and
    # the committed metadata. Core rather than a study: generate_paper_values.py cites the
    # same CSV, and a missing or stale study must fail the build, never skip it.
    ("generate_finetune_pairs_table.py", None, ""),
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
    # Reads the live Camelyon run plus the two neighbourhood studies written beside it; the
    # contested-fraction summary stands in for both (they are run together). The macros the
    # text quotes from this table are emitted by generate_paper_values.py, which fails when
    # either study is missing.
    (
        "generate_supp_rank_table.py",
        f"{_CAMELYON.studies_rel}/contested_fraction_summary.csv",
        "run scripts/repro/figures/typed_neighbor_rank_experiment.py and "
        "scripts/studies/contested_fraction_experiment.py",
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
    # Reads the three per-cohort studies under output/studies/sample-shortcut/ (PCaBiop, the
    # last one the study writes, stands in for all three), each verified against the sha256
    # of every input it recorded. The macros the text quotes are emitted by
    # generate_paper_values.py, which fails when the study is missing or stale. The montage
    # PDF it captions is drawn by figures/margin_montage_figure.py.
    (
        "generate_sample_shortcut_floats.py",
        "output/studies/sample-shortcut/pcabiop/cohort.json",
        "run scripts/studies/sample_shortcut.py",
    ),
    # Reads each tile run's croma_m_sweep_metrics.csv (Camelyon guarded as representative;
    # the three are produced together). Absent only when a run was benchmarked without the
    # CRoMa averaging-radius sweep, in which case tab:m-sweep cannot be rebuilt.
    (
        "generate_m_sweep_table.py",
        f"{_CAMELYON.run_rel}/results/croma_m_sweep_metrics.csv",
        "re-run the benchmark with the CRoMa m-sweep enabled",
    ),
    # tab:grade-matched (grade-matched SO pools on PCaBiop). No required input: it is never
    # skipped, so a missing or stale study fails the build -- _grade_matched.load raises,
    # and generate_paper_values.py fails on the same study for its macros.
    ("generate_grade_matched_table.py", None, "run scripts/studies/grade_matched_pools.py"),
    # tab:nipd-auroc (nIPD from AUROC beside balanced accuracy). No required input: it is
    # never skipped, so a missing or stale AUROC cell fails the build -- _nipd_auroc.load
    # raises, and generate_paper_values.py fails on the same cells for its macros.
    (
        "generate_nipd_auroc_table.py",
        None,
        "run scripts/studies/auc/sweep.py, then scripts/studies/auc/report.py",
    ),
]


#: Generators whose study artifact is *required*: when it is absent the build fails, naming
#: the study to run, instead of skipping. Their macros feed generated_values.tex, which
#: raises without them too.
REQUIRED_STUDIES = [
    (
        "generate_silhouette_baselines_table.py",
        "output/studies/silhouette-baselines/silhouette_baselines_correlation.csv",
        "run scripts/studies/silhouette_baselines.py (needs croma[repro] for scib-metrics)",
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

    missing: list[tuple[str, str]] = []
    for script, required, remedy in REQUIRED_STUDIES:
        if not (REPO / required).exists():
            missing.append((script, f"missing {required} -- {remedy}"))
            failed.append(script)
            continue
        print(f"\n=== {script} ===", flush=True)
        if _run(script, []) != 0:
            failed.append(script)

    print("\n" + "=" * 70)
    for script, why in skipped:
        print(f"SKIPPED  {script}: {why}")
    for script, why in missing:
        print(f"MISSING  {script}: {why}")
    if failed:
        print(f"FAILED   {', '.join(failed)}")
        return 1
    print(f"OK       {len(CORE) + len(STUDIES) - len(skipped)} generator(s) ran, "
          f"{len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
