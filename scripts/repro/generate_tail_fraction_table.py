"""Regenerate the inline tail-fraction table in the supplement from saved m=5 margins.

Run with --check to check the local paper copy or --print to emit the LaTeX without a copy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _distributions import CONTROL_MODEL
from _paper_tables import CROMA_HEADLINE_M, croma_as_margin
from croma.metrics.tail import compute_tail_metrics
from paper_manifest import by_prefix


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach")
LABEL = r"\label{tab:alpha-stability}"


def rank_correlations(margins: dict[str, np.ndarray]) -> tuple[float, float, float]:
    scores = pd.DataFrame({
        name: [compute_tail_metrics(values, alpha).ltm_alpha for alpha in (0.05, 0.10, 0.20)]
        for name, values in margins.items()
    }, index=(0.05, 0.10, 0.20)).T
    if scores.isna().any().any():
        raise ValueError("Every ranked encoder must have finite per-sample margins")
    corr = scores.corr(method="spearman")
    values = (corr.at[0.05, 0.10], corr.at[0.10, 0.20], corr.at[0.05, 0.20])
    if not np.isfinite(values).all():
        raise ValueError("Undefined rank correlation in tail-fraction table")
    return values


def build(root: Path) -> str:
    rows = []
    roster = None
    for prefix in BENCHMARKS:
        entry = by_prefix(prefix)
        metrics = root / entry.metrics_rel
        models = pd.read_csv(metrics).query("model != @CONTROL_MODEL")
        if not models["croma_m"].eq(CROMA_HEADLINE_M).all():
            raise ValueError(f"Unexpected CRoMa radius in {metrics}")
        if models.model.duplicated().any() or (roster is not None and set(models.model) != roster):
            raise ValueError(f"Inconsistent tile encoder roster in {metrics}")
        roster = set(models.model)
        margins = {}
        for model in models.itertuples():
            path = metrics.parent / "sample_distributions" / Path(model.croma_samples_path).name
            margins[model.model] = croma_as_margin(pd.Series(np.load(path))).to_numpy()
        values = " & ".join(f"${value:.3f}$" for value in rank_correlations(margins))
        rows.append(f"{entry.short_name:<12} & {values} " + r"\\")
    slide_count = pd.read_csv(root / by_prefix("Panda").metrics_rel)["model"].nunique()
    slide_count_text = "five" if slide_count == 5 else str(slide_count)
    return "\n".join([
        r"\begin{table}[!htbp]", r"\centering", r"\small", r"\begin{tabular}{lccc}",
        r"\hline", r"Benchmark & $\rho(0.05,0.10)$ & $\rho(0.10,0.20)$ & $\rho(0.05,0.20)$ \\",
        r"\hline", *rows, r"\hline", r"\end{tabular}",
        r"\caption{\textbf{Rank correlations across lower-tail fractions.} Entries are Spearman "
        r"correlations between per-model $\mcode{LTM}_{\alpha}$ rankings at $\alpha=0.05$, "
        r"$0.10$ and $0.20$, computed from \code{CRoMa}$(m{=}5)$ within each benchmark. "
        r"The four tile benchmarks each include the same \TileRankedNModels{} pathology encoders. "
        rf"We omitted PCaBiop because only {slide_count_text} slide encoders were evaluated.}}",
        LABEL, r"\end{table}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--print", dest="print_table", action="store_true")
    args = parser.parse_args()
    table = build(ROOT)
    if args.print_table:
        print(table)
        return
    path = ROOT / "paper/sections/supp/supp.tex"
    source = path.read_text()
    if source.count(LABEL) != 1:
        raise ValueError(f"Expected one {LABEL} in {path}")
    label = source.index(LABEL)
    start = source.rindex(r"\begin{table}", 0, label)
    end = source.index(r"\end{table}", label) + len(r"\end{table}")
    if source[start:end] != table:
        if args.check:
            raise SystemExit(f"Stale {LABEL} in {path}")
        path.write_text(source[:start] + table + source[end:])


if __name__ == "__main__":
    main()
