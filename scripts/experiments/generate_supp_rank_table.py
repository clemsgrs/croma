"""Build the supplementary per-model typed-neighbour-rank LaTeX table.

Fuses the two stories into one table, sorted by the headline pooled CRoMa:
  - depth + asymmetry: median SO-rank and OS-rank (both deep; asymmetry mirrors CRoMa)
  - tail cleanliness: nearest-OS rank for the CRoMa bottom decile vs the rest
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402

ROOT = Path("output/faithful/pathorob-camelyon-faithful")
rank = pd.read_csv(ROOT / "typed_neighbor_rank_summary.csv").set_index("model")
croma = pd.read_csv(ROOT / "results" / "metrics.csv").set_index("model")["croma"]

df = rank.join(croma).sort_values("croma", ascending=False)

lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{lccccc}",
    r"\hline",
    r"& & \multicolumn{2}{c}{median rank} & \multicolumn{2}{c}{nearest-\code{OS} rank} \\",
    r"Model & \code{CRoMa} & \code{SO} & \code{OS} & tail & rest \\",
    r"\hline",
]
for model, r in df.iterrows():
    lines.append(
        f"{model} & {r['croma']:.2f} & {int(round(r['so_med']))} & "
        f"{int(round(r['os_med']))} & {int(round(r['tail_os_rank_med']))} & "
        f"{int(round(r['rest_os_rank_med']))} \\\\"
    )
lines += [
    r"\hline",
    r"\end{tabular}",
    r"\caption{\textbf{Per-model typed-neighbour ranks underpinning the global-ordering and "
    rf"tail-cleanliness claims.}} Models are sorted by pooled $\mcode{{CRoMa}}(m{{=}}{int(CROMA_HEADLINE_M)})$. "
    r"\emph{Median rank} columns give the rank (among non-self neighbours, ordered by "
    r"increasing cosine distance, same-slide neighbours excluded as in \code{CRoMa}) at which "
    r"the nearest \code{SO} (same-biology, other-confounder) and nearest \code{OS} "
    r"(other-biology, same-confounder) neighbour first appear. Both are reached far from the "
    r"immediate neighbourhood, and the \code{SO}/\code{OS} asymmetry tracks \code{CRoMa}: robust "
    r"models reach a cross-confounder biological match early and the same-confounder impostor "
    r"late, while fragile models reverse this. \emph{Nearest-\code{OS} rank} columns compare "
    r"the \code{CRoMa} bottom decile (the \code{LTM} tail) against the rest: for every model the "
    r"tail's same-confounder impostor is much nearer, confirming the lower tail flags real "
    r"local shortcuts rather than ratio-compression artefacts.}",
    r"\label{tab:typed-neighbour-ranks}",
    r"\end{table}",
]

out = Path("paper/sections/supp_typed_neighbor_ranks.tex")
out.write_text("\n".join(lines) + "\n")
print(df[["croma", "so_med", "os_med", "tail_os_rank_med", "rest_os_rank_med"]].round(2).to_string())
print(f"\nwrote {out}")
