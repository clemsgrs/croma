"""Build the supplementary bootstrap-uncertainty section (feedback points 3 & 6).

Two parts, both read from the per-benchmark ``bootstrap_uncertainty.{json,csv}``
written by ``bootstrap_uncertainty.py``:

  1. a cross-benchmark Spearman summary (Table~\ref{tab:bootstrap-uncertainty}):
     CRoMa vs RI, CRoMa vs MaRI, and RI vs MaRI with bootstrap CIs, showing that
     CRoMa is the least rank-redundant of the three on every benchmark (point 6);
  2. per-model pooled-median CRoMa with 95% CIs and bootstrap rank intervals -- TCGA-2x2/4x4,
     Tolkach, PANDA-ISUP (point 3). Camelyon appears in the correlation summary only.

No results table currently renders in-table CIs (``ResultsTable.with_ci`` is off for every
entry), and this section is not ``\input`` by ``supp.tex``. Both are deliberate: the CIs were
retired from the rendered paper, and the machinery kept for the rebuttal. Turning ``with_ci``
on makes the results captions cite ``tab:bootstrap-uncertainty``, so re-enable that ``\input``
in the same change or the reference dangles.

All cross-model statistics here are computed on the ranked panel -- the natural-image control
is a floor, not a competitor, so it holds no rank and enters no Spearman.

Usage: python scripts/repro/generate_uncertainty_supp_table.py
"""

import json
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M, CaptionClaimError, scriptsize_ci
from paper_manifest import by_prefix

ROOT = Path(__file__).resolve().parents[2]

#: (display name, run dir), in the paper's order. Both the names and the protocol come from
#: the manifest, so a re-run at a different protocol cannot leave this table behind.
_SUMMARY_PREFIXES = ["Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach", "PandaIsup"]
SUMMARY_ORDER = [(by_prefix(p).display_name, by_prefix(p).run_rel) for p in _SUMMARY_PREFIXES]
# benchmarks needing a per-model CI table (Camelyon excluded -- it is in the main table)
PER_MODEL_ORDER = SUMMARY_ORDER[1:]
OUT = ROOT / "paper/sections/supp/bootstrap_uncertainty.tex"


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


def _redundancy_claim(rows: list[tuple[str, dict]]) -> str:
    """"RI and MaRI agree with each other more than either does with CRoMa" -- checked.

    The sentence is the whole point of the table; if a benchmark ever breaks it, say so
    rather than shipping the generalisation.
    """
    offenders = [
        name
        for name, corr in rows
        if not (
            corr["ri_vs_mari"]["rho"] > corr["croma_vs_ri"]["rho"]
            and corr["ri_vs_mari"]["rho"] > corr["croma_vs_mari"]["rho"]
        )
    ]
    if offenders:
        verb = "disagrees" if len(offenders) == 1 else "disagree"
        raise CaptionClaimError(
            "caption claims RI--MaRI is the strongest pair on every benchmark, but "
            f"{', '.join(offenders)} {verb}. Rewrite the sentence, do not silence it."
        )
    return (
        r"All three metrics rank models similarly, but \code{RI} and \code{MaRI} agree with "
        r"\emph{each other} more strongly than either agrees with \code{CRoMa} on every "
        r"benchmark: \code{CRoMa} is the least rank-redundant of the three."
    )


def _skipped_clause(skipped: list[tuple[str, int, int]]) -> str:
    """Name the benchmarks whose row is blank, and why. A dash with no explanation reads
    as a missing number rather than a deliberate floor. The floor is read off the artifact,
    not restated here."""
    if not skipped:
        return ""
    # Several display names already end in a parenthetical, so the count is appended with
    # "with", not wrapped in a second set of brackets.
    parts = [f"{name} with $n{{=}}{n}$" for name, n, _ in skipped]
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    verb, pronoun = ("falls", "it is") if len(parts) == 1 else ("fall", "they are")
    floors = {floor for _, _, floor in skipped}
    if len(floors) != 1:
        raise CaptionClaimError(f"benchmarks were skipped under differing floors: {floors}")
    return (
        f" {joined} {verb} below the ${floors.pop()}$-model floor this table applies, "
        f"so {pronoun} left blank: a Spearman $\\rho$ over so few models carries no usable "
        "confidence interval."
    )


def correlation_table() -> list[str]:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Benchmark & $n$ & \code{CRoMa}--\code{RI} & \code{CRoMa}--\code{MaRI} & \code{RI}--\code{MaRI} \\",
        r"\hline",
    ]
    reported: list[tuple[str, dict]] = []
    skipped: list[tuple[str, int, int]] = []
    for name, rel in SUMMARY_ORDER:
        summary, _ = _load(rel)
        if summary is None:
            continue
        corr = summary.get("correlations", {})
        n = corr.get("n_models", summary.get("n_models", "?"))
        if corr.get("skipped"):
            skipped.append((name, int(n), int(corr["min_models"])))
            lines.append(f"{name} & {n} & --- & --- & --- \\\\")
        else:
            reported.append((name, corr))
            lines.append(
                f"{name} & {n} & {_corr_cell(corr, 'croma_vs_ri')} & "
                f"{_corr_cell(corr, 'croma_vs_mari')} & {_corr_cell(corr, 'ri_vs_mari')} \\\\"
            )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{\textbf{Cross-model rank correlation between the three metrics, per benchmark.} "
        r"Spearman $\rho$ over the $n$ models of each benchmark, with $95\%$ bootstrap confidence "
        r"intervals over models (resampling the model set). "
        + _redundancy_claim(reported)
        + r" Rank agreement is expected and welcome---\code{CRoMa}'s contribution is to rank models "
        r"on a complete, model-comparable population rather than on \code{RI}/\code{MaRI}'s "
        r"model-dependent support (Sec.~\ref{sec:complementarity})."
        + _skipped_clause(skipped)
        + "}",
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
        r"\begin{table}[!htbp]",
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
