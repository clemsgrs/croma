"""Single-command reproduction of the paper's FAITHFUL (PathoROB-comparable) results.

For each faithful benchmark it (1) runs the canonical benchmark.py at the ``k-star``
protocol with the paper's reporting defaults (k_max=100; per-model auto tau; no SS/OO
pruning), then (2) regenerates the LaTeX results table. benchmark.py is now compute-only:
it reads the tileset's pre-extracted embeddings (output/embeddings/<tileset>/) and a
row-view of them for the benchmark, so there is nothing to materialise here -- the
embeddings are produced once per tileset by scripts/bench/extract_embeddings.py.

Usage:
    python scripts/repro/reproduce_faithful.py            # all benchmarks
    python scripts/repro/reproduce_faithful.py camelyon   # one benchmark

Faithful benchmarks (exactly PathoROB's RI sets):
    camelyon  : RUMC+UMCU, 20,400 patches, dataset_wide
    tolkach   : 3 cohorts (TCGA excluded), 9,000 patches, dataset_wide
    tcga_2x2  : 112,800 paired occurrences, paired_2x2
    tcga_4x4  : 8,160 patches, dataset_wide (supplementary)
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "bench"))
sys.path.insert(0, str(REPO))
import layout  # noqa: E402

MODELS = ("CONCH,CONCHv1.5,H-optimus-0,H-optimus-1,H0-mini,Hibou-B,Hibou-L,"
          "Midnight-12k,Phikon,Phikon-v2,Prost40M,Prov-GigaPath,UNI,UNI2-h,Virchow,Virchow2")

PROTOCOL = "k-star"

CFG = {
    # camelyon is the primary table (tab:main-results, the column-defining one): we
    # reproduce its metrics.csv here but regenerate its .tex separately so its
    # self-contained caption is preserved. Its body carries in-table bootstrap CRoMa
    # CIs: regenerate with generate_results_table.py --with-ci (needs the sibling
    # bootstrap_uncertainty.csv from bootstrap_uncertainty.py), then restore the caption.
    "camelyon": dict(benchmark="pathorob-camelyon",
                     name=r"PathoROB Camelyon", label="tab:main-results",
                     out_tex=None),
    "tolkach": dict(benchmark="pathorob-tolkach-esca",
                    name=r"PathoROB Tolkach-ESCA", label="tab:main-results-tolkach",
                    out_tex="paper/sections/results_table_tolkach.tex"),
    "tcga_2x2": dict(benchmark="pathorob-tcga-2x2",
                     name=r"PathoROB TCGA (2$\times$2)", label="tab:main-results-tcga",
                     out_tex="paper/sections/results_table_tcga.tex"),
    "tcga_4x4": dict(benchmark="pathorob-tcga-4x4",
                     name=r"PathoROB TCGA (4$\times$4)", label="tab:main-results-tcga4x4",
                     out_tex="paper/sections/results_table_tcga4x4.tex"),
}


def run(ds):
    cfg = CFG[ds]
    cmd = [sys.executable, str(REPO / "scripts/bench/benchmark.py"),
           "--benchmark", cfg["benchmark"],
           "--protocol", PROTOCOL,
           "--models", MODELS,
           "--k-max", "100",
           "--progress", "off"]
    print(f"[{ds}] running benchmark: {' '.join(cmd[-9:])}", flush=True)
    subprocess.run(cmd, check=True)
    metrics = layout.results_dir(PROTOCOL, cfg["benchmark"]) / "metrics.csv"
    print(f"[{ds}] metrics -> {metrics}", flush=True)
    if cfg["out_tex"]:
        gen = [sys.executable, str(REPO / "scripts/repro/generate_results_table.py"),
               "--metrics", str(metrics), "--name", cfg["name"],
               "--label", cfg["label"], "--out", str(REPO / cfg["out_tex"])]
        subprocess.run(gen, check=True)
        print(f"[{ds}] regenerated {cfg['out_tex']}", flush=True)


if __name__ == "__main__":
    targets = [sys.argv[1]] if len(sys.argv) > 1 else list(CFG)
    for ds in targets:
        run(ds)
    # Refresh the inline derived scalars (\<Bench>CromaSpan, ...) so prose stays in
    # sync with the regenerated tables instead of being hand-typed.
    subprocess.run([sys.executable, str(REPO / "scripts/repro/generate_paper_values.py"),
                    "--root", str(REPO),
                    "--out", str(REPO / "paper/sections/generated_values.tex")],
                   check=True)
    print("regenerated paper/sections/generated_values.tex", flush=True)
