"""Build the supplementary per-model typed-neighbour-rank LaTeX table.

Fuses the two stories into one table, sorted by the headline pooled CRoMa:
  - depth + asymmetry: median SO-rank and OS-rank (both deep; asymmetry mirrors CRoMa)
  - tail cleanliness: nearest-OS rank for the CRoMa bottom decile vs the rest
"""

import sys
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M
from paper_manifest import by_benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from croma.plotstyle import CONTROL_MODEL  # noqa: E402

ENTRY = by_benchmark("pathorob-camelyon")
rank = pd.read_csv(Path(ENTRY.studies_rel) / "typed_neighbor_rank_summary.csv").set_index("model")
croma = pd.read_csv(ENTRY.metrics_rel).set_index("model")["croma"]

# The study caches a `croma` column for its own scatter, copied from this same metrics.csv.
# Drop it and re-join from the source, so the table cannot show a stale cache.
df = rank.drop(columns=["croma"], errors="ignore").join(croma, how="inner")
if len(df) != len(rank):
    missing = sorted(set(rank.index) - set(df.index))
    raise SystemExit(f"models in the rank summary but not in {ENTRY.metrics_rel}: {missing}")
df = df.sort_values("croma", ascending=False)
# The natural-image control is a floor, not a competitor: it is excluded from the sort and
# rendered in its own band below the rule, exactly as in the main results tables.
ranked = df.drop(index=CONTROL_MODEL, errors="ignore")
control = df.loc[[CONTROL_MODEL]] if CONTROL_MODEL in df.index else df.iloc[:0]


def _row(model: str, r: pd.Series) -> str:
    return (
        f"{model} & {r['croma']:.2f} & {int(round(r['so_med']))} & "
        f"{int(round(r['os_med']))} & {int(round(r['tail_os_rank_med']))} & "
        f"{int(round(r['rest_os_rank_med']))} \\\\"
    )


def _control_clause() -> str:
    if control.empty:
        return ""
    return (
        rf" The natural-image control \code{{{CONTROL_MODEL}}} is shown below the rule and "
        rf"excluded from the ordering."
    )


lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{lccccc}",
    r"\hline",
    r"& & \multicolumn{2}{c}{median rank} & \multicolumn{2}{c}{nearest-\code{OS} rank} \\",
    r"Model & \code{CRoMa} & \code{SO} & \code{OS} & tail & rest \\",
    r"\hline",
]
lines += [_row(model, r) for model, r in ranked.iterrows()]
if not control.empty:
    lines.append(r"\hline")
    lines += [_row(model, r) for model, r in control.iterrows()]

lines += [
    r"\hline",
    r"\end{tabular}",
    r"\caption{\textbf{Typed-neighbour ranks on PathoROB Camelyon.} "
    rf"Models are ordered by pooled $\mcode{{CRoMa}}(m{{=}}{int(CROMA_HEADLINE_M)})$. Median "
    r"\code{SO} and \code{OS} ranks locate the nearest cross-confounder biological match and "
    r"same-confounder biological distractor, respectively, after excluding self and same-slide "
    r"samples. The final columns report nearest-\code{OS} rank in the lowest \code{CRoMa} decile "
    r"and in the remaining samples." + _control_clause() + "}",
    r"\label{tab:typed-neighbour-ranks}",
    r"\end{table}",
]

out = Path("paper/sections/supp/typed_neighbor_ranks.tex")
out.write_text("\n".join(lines) + "\n")
print(df[["croma", "so_med", "os_med", "tail_os_rank_med", "rest_os_rank_med"]].round(2).to_string())
print(f"\nwrote {out}")
