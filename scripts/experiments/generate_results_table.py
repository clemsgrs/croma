"""Build a per-dataset main-results (Table 3) LaTeX table from a benchmark metrics.csv.

Mirrors the hand-authored Camelyon table (``paper/sections/results_table.tex``,
``tab:main-results``) so every benchmark gets an identically-structured table with
zero manual entry. Sorted by pooled CRoMa (headline m); per-column best in bold.

Usage:
  python scripts/experiments/generate_results_table.py \
      --metrics output/pathorob-tcga-4x4/results/metrics.csv \
      --name "PathoROB TCGA (4x4)" \
      --label tab:main-results-tcga4x4 \
      --out paper/sections/results_table_tcga4x4.tex
"""

import argparse
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M, scriptsize_ci

# column -> (header, decimals, percent)
COLS = [
    ("k", r"$k^\star$", 0, False),
    ("bio_knn_bacc", "bio bacc", 3, False),
    ("ri", r"\code{RI}", 3, False),
    ("mari", r"\code{MaRI}", 3, False),
    ("croma", r"\code{CRoMa}", 2, False),
    ("croma_ltm_alpha", r"$\mcode{LTM}_{10\%}$", 2, False),
    ("support", "support", 1, True),
]


def _fmt(value: float, decimals: int, percent: bool, bold: bool) -> str:
    if decimals == 0:
        body = f"{int(round(value))}"
    elif percent:
        body = f"{value:.{decimals}f}\\%"
    else:
        body = f"{value:.{decimals}f}"
    return f"\\textbf{{{body}}}" if bold else body


def _load_croma_ci(metrics_csv: Path) -> dict[str, tuple[float, float]] | None:
    """Per-model (lo, hi) CRoMa CI from the sibling bootstrap_uncertainty.csv, if any."""
    ci_path = metrics_csv.parent / "bootstrap_uncertainty.csv"
    if not ci_path.exists():
        return None
    ci = pd.read_csv(ci_path).set_index("model")
    return {m: (float(r["croma_lo"]), float(r["croma_hi"])) for m, r in ci.iterrows()}


def build_table(
    metrics_csv: Path, name: str, label: str, model_type: str, with_ci: bool = False
) -> str:
    df = pd.read_csv(metrics_csv)
    df["support"] = (1.0 - df["ri_undefined_frac"]) * 100.0
    df = df.sort_values("croma", ascending=False).reset_index(drop=True)

    croma_ci = _load_croma_ci(metrics_csv) if with_ci else None
    confounder = df["confounder_display_name"].iloc[0]
    n_models = len(df)
    # per-column best (highest is best for every reported column, incl. support)
    best = {col: df[col].max() for col, _, _, _ in COLS if col != "k"}

    header = " & ".join(["Model"] + [h for _, h, _, _ in COLS]) + r" \\"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{l" + "c" * len(COLS) + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in df.iterrows():
        cells = [row["model"]]
        for col, _, dec, pct in COLS:
            is_best = col != "k" and abs(row[col] - best[col]) < 1e-9
            cell = _fmt(row[col], dec, pct, is_best)
            if col == "croma" and croma_ci is not None and row["model"] in croma_ci:
                lo, hi = croma_ci[row["model"]]
                cell += r"\," + scriptsize_ci(lo, hi)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{\textbf{{Main quantitative results on {name}.}} The {n_models} "
        rf"{model_type} foundation models, sorted by pooled \code{{CRoMa}} ($m{{=}}{int(CROMA_HEADLINE_M)}$). "
        r"Columns are as defined in Table~\ref{tab:main-results} (operating point "
        r"$k^\star$; biological $k$-NN balanced accuracy at $k^\star$; pooled \code{RI} and "
        rf"\code{{MaRI}} at $k^\star$; pooled \code{{CRoMa}} at $m{{=}}{int(CROMA_HEADLINE_M)}$; lower-tail mean "
        r"$\mcode{LTM}_{10\%}$ of \code{CRoMa}; and \emph{support}, the fraction of samples "
        rf"on which \code{{RI}}/\code{{MaRI}} are defined). Confounder: {confounder}. "
        r"Per-column best is in bold."
        + (
            r" \code{CRoMa} brackets are 95\% slide-level cluster-bootstrap confidence "
            r"intervals on the pooled median; overlapping intervals near the top indicate "
            r"a statistical tie (Supplementary Table~\ref{tab:bootstrap-uncertainty})."
            if with_ci
            else ""
        )
        + "}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", required=True, type=Path)
    p.add_argument("--name", required=True, help="Display name for the caption.")
    p.add_argument("--label", required=True, help="LaTeX label, e.g. tab:main-results-tcga4x4.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model-type", default="tile-level",
                   help='Model-modality phrase for the caption (e.g. "slide-level").')
    p.add_argument("--with-ci", action="store_true",
                   help="Render bootstrap CRoMa CIs in-table from sibling bootstrap_uncertainty.csv.")
    args = p.parse_args()

    tex = build_table(args.metrics, args.name, args.label, args.model_type, args.with_ci)
    args.out.write_text(tex)
    print(f"wrote {args.out}  ({tex.count(chr(92) + chr(92))} data rows incl. header)")


if __name__ == "__main__":
    main()
