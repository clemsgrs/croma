"""Build the supplementary bootstrap-uncertainty section (feedback points 3 & 6).

Two parts, both read from the per-benchmark ``bootstrap_uncertainty.{json,csv}``
written by ``bootstrap_uncertainty.py``:

  1. a cross-benchmark Spearman summary (Table~\ref{tab:bootstrap-uncertainty}):
     CRoMa vs RI, CRoMa vs MaRI, and RI vs MaRI with bootstrap CIs, showing that
     CRoMa is the least rank-redundant of the three on every benchmark (point 6);
  2. per-model pooled-median CRoMa with 95% CIs and bootstrap rank intervals for the
     benchmarks whose main table does *not* carry in-table CIs -- TCGA-2x2/4x4,
     Tolkach, PANDA (point 3, scope C: Camelyon is in-table, the rest are here).

Usage: python scripts/repro/generate_uncertainty_supp_table.py
"""

import json
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M, scriptsize_ci

ROOT = Path(__file__).resolve().parents[2]

# (display name, benchmark dir) -- Camelyon included for the correlation summary only.
SUMMARY_ORDER = [
    ("PathoROB Camelyon", "output/metrics/k-star/pathorob-camelyon"),
    (r"PathoROB TCGA ($2{\times}2$)", "output/metrics/k-star/pathorob-tcga-2x2"),
    (r"PathoROB TCGA ($4{\times}4$)", "output/metrics/k-star/pathorob-tcga-4x4"),
    ("PathoROB Tolkach-ESCA", "output/metrics/k-star/pathorob-tolkach-esca"),
    ("PANDA", "output/metrics/k-star/panda-isup"),
]
# benchmarks needing a per-model CI table (Camelyon excluded -- it is in the main table)
PER_MODEL_ORDER = SUMMARY_ORDER[1:]
OUT = ROOT / "paper/sections/supp_bootstrap_uncertainty.tex"


def _load(rel_dir: str) -> tuple[dict | None, pd.DataFrame | None]:
    results = ROOT / rel_dir / "results"
    jpath, cpath = results / "bootstrap_uncertainty.json", results / "bootstrap_uncertainty.csv"
    summary = json.loads(jpath.read_text()) if jpath.exists() else None
    df = pd.read_csv(cpath) if cpath.exists() else None
    return summary, df


def _corr_cell(corr: dict, key: str, with_ci: bool = True) -> str:
    c = corr.get(key)
    if c is None:
        return "---"
    if with_ci:
        return f"${c['rho']:.2f}$ " + scriptsize_ci(c["lo"], c["hi"])
    return f"${c['rho']:.2f}$"


def correlation_table() -> list[str]:
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Benchmark & $n$ & \code{CRoMa}--\code{RI} & \code{CRoMa}--\code{MaRI} & \code{RI}--\code{MaRI} \\",
        r"\hline",
    ]
    for name, rel in SUMMARY_ORDER:
        summary, _ = _load(rel)
        if summary is None:
            continue
        corr = summary.get("correlations", {})
        n = corr.get("n_models", summary.get("n_models", "?"))
        if corr.get("skipped"):
            lines.append(f"{name} & {n} & --- & --- & --- \\\\")
        else:
            lines.append(
                f"{name} & {n} & {_corr_cell(corr, 'croma_vs_ri')} & "
                f"{_corr_cell(corr, 'croma_vs_mari')} & {_corr_cell(corr, 'ri_vs_mari')} \\\\"
            )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{\textbf{Cross-model rank correlation between the three metrics, per benchmark.} "
        r"Spearman $\rho$ over the $n$ models of each benchmark, with $95\%$ bootstrap confidence "
        r"intervals over models (resampling the model set). All three metrics rank models similarly, "
        r"but \code{RI} and \code{MaRI} agree with \emph{each other} more strongly than either agrees "
        r"with \code{CRoMa} on every benchmark: \code{CRoMa} is the least rank-redundant of the three. "
        r"Rank agreement is expected and welcome---\code{CRoMa}'s contribution is to rank models on a "
        r"complete, model-comparable population rather than on \code{RI}/\code{MaRI}'s model-dependent "
        r"support (Sec.~\ref{sec:complementarity}). PANDA ($n{=}3$) is too small for a meaningful "
        r"correlation.}",
        r"\label{tab:bootstrap-uncertainty}",
        r"\end{table}",
    ]
    return lines


def per_model_table(name: str, rel: str) -> list[str]:
    summary, df = _load(rel)
    if df is None or summary is None:
        return []
    df = df.sort_values("croma", ascending=False)
    suffix = rel.rstrip("/").split("/")[-1].replace("pathorob-", "").replace("-", "")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Model & \code{CRoMa} (95\% CI) & rank (95\% CI) \\",
        r"\hline",
    ]
    for _, r in df.iterrows():
        croma_ci = scriptsize_ci(r["croma_lo"], r["croma_hi"])
        rank_ci = rf"{{\scriptsize$[{int(r['rank_lo'])}, {int(r['rank_hi'])}]$}}"
        lines.append(
            f"{r['model']} & ${r['croma']:.2f}$ {croma_ci} "
            f"& {int(r['point_rank'])} {rank_ci} \\\\"
        )
    n_boot = summary.get("n_boot", "?")
    n_slides = summary.get("n_slides", "?")
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{\textbf{{Bootstrap uncertainty on {name}.}} Per-model pooled-median "
        rf"\code{{CRoMa}} at $m{{=}}{int(CROMA_HEADLINE_M)}$ with $95\%$ slide-level cluster-bootstrap "
        rf"confidence intervals (${n_boot}$ resamples over ${n_slides}$ slides), and the paired-bootstrap "
        r"rank interval. Sorted by point \code{CRoMa}. As on Camelyon, the sign is stable while the "
        r"closest models form rank ties.}",
        rf"\label{{tab:bootstrap-uncertainty-{suffix}}}",
        r"\end{table}",
    ]
    return lines


def main() -> int:
    out = [
        "% AUTO-GENERATED by scripts/repro/generate_uncertainty_supp_table.py -- do not edit by hand.",
        r"\subsection*{Bootstrap uncertainty and metric redundancy}",
        "",
    ]
    out += correlation_table()
    out.append("")
    for name, rel in PER_MODEL_ORDER:
        rows = per_model_table(name, rel)
        if rows:
            out += rows
            out.append("")
    OUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
