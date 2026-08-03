"""Build the supplementary per-model typed-neighbour-rank LaTeX table.

Fuses the two stories into one table, sorted by the headline pooled CRoMa:
  - depth + asymmetry: median SO-rank and OS-rank (both deep; asymmetry mirrors CRoMa)
  - tail cleanliness: nearest-OS rank for the CRoMa bottom decile vs the rest
"""

import json
import sys
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M
from paper_manifest import by_benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting.style import CONTROL_MODEL  # noqa: E402

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
        f"{int(round(r['os_med']))} \\\\"
    )


def _thousands(value: int) -> str:
    """LaTeX thousands separator: 20400 -> ``20{,}400``."""
    return f"{int(value):,}".replace(",", "{,}")


#: The caption's headline rank is genuinely *pooled*: the median over every defined sample of
#: every ranked encoder of min(SO rank, OS rank). That is not recoverable from the per-model
#: CSV -- a median of per-model medians answers a different question and lands at 251 rather
#: than 149 -- so it is read from the study's JSON, which is where the study records it.
_summary_json = json.loads(
    (Path(ENTRY.studies_rel) / "typed_neighbor_rank_summary.json").read_text()
)
_first_typed_rank = int(round(float(_summary_json["pooled_first_percentiles_ranked"]["50"])))
_n_candidates = int(round(float(ranked["n_def"].median())))
_operating_k = int(pd.read_csv(ENTRY.metrics_rel)["k"].iloc[0])


def _control_clause() -> str:
    if control.empty:
        return ""
    return (
        rf" \code{{{CONTROL_MODEL}}}, the natural-image control, is reported separately "
        rf"for reference."
    )


# The nearest-OS tail/rest columns are computed by the study and deliberately not shown: the
# supplement's argument is about how deep the first typed neighbour sits, and the two extra
# columns crowd the table without carrying it. They remain in
# typed_neighbor_rank_summary.csv for anyone who wants them.
lines = [
    r"\begin{table}[!htbp]",
    r"\centering",
    r"\small",
    r"\begin{tabular}{lccc}",
    r"\hline",
    r" & & \multicolumn{2}{c}{median rank} \\",
    r"\cline{3-4}",
    r"Model & \code{CRoMa} & \code{SO} & \code{OS} \\",
    r"\hline",
]
lines += [_row(model, r) for model, r in ranked.iterrows()]
if not control.empty:
    lines.append(r"\hline")
    lines += [_row(model, r) for model, r in control.iterrows()]

lines += [
    r"\hline",
    r"\end{tabular}",
    rf"\caption{{\textbf{{Typed-neighbour ranks on {ENTRY.short_name}.}} "
    rf"Models are ordered by pooled $\mcode{{CRoMa}}(m{{=}}{int(CROMA_HEADLINE_M)})$. Median "
    r"\code{SO} and \code{OS} ranks summarize, across models, the rank of the nearest "
    r"cross-confounder biological match and same-confounder biological distractor, "
    r"respectively. Pooled across the \CamelyonRankedNModels{} encoders, the first \code{SO} "
    rf"or \code{{OS}} neighbour occurs at a median rank of $\approx {_first_typed_rank}$ among "
    rf"${_thousands(_n_candidates)}$ candidates, sitting far beyond the $k{{=}}{_operating_k}$ "
    rf"operating point selected by the biological $k$-NN criterion on {ENTRY.short_name}. This "
    r"neighbourhood size is therefore too narrow for the fixed-$k$ metrics \code{RI} and "
    r"\code{MaRI} to capture the typed \code{SO}/\code{OS} contrast they need."
    + _control_clause() + "}",
    r"\label{tab:typed-neighbour-ranks}",
    r"\end{table}",
]

out = Path("paper/sections/supp/typed_neighbor_ranks.tex")
out.write_text("\n".join(lines) + "\n")
print(df[["croma", "so_med", "os_med", "tail_os_rank_med", "rest_os_rank_med"]].round(2).to_string())
print(f"\nwrote {out}")
