r"""Render the typed-neighbour coverage table (``tab:typed-neighbour-coverage``) from the live run.

The table reports, per Camelyon encoder, how deep the nearest typed neighbours sit (median
rank of the first SO and first OS neighbour) and how the fixed-$k$ neighbourhoods that RI and
MaRI score split into pure-SO, pure-OS and contested (both types) ones. It reads three files
from the run the paper reports (``paper_manifest``):

* ``results/metrics.csv`` -- the roster, median CRoMa, support and the operating ``k``;
* ``studies/typed_neighbor_rank_summary.{csv,json}`` --
  ``scripts/repro/figures/typed_neighbor_rank_experiment.py``;
* ``studies/contested_fraction_summary.csv`` -- ``scripts/studies/contested_fraction_experiment.py``.

It used to render a sealed 20-encoder snapshot while the paper's own copy of the table was
hand-typed beside it. Both studies now run on the live run, so the table, its caption and the
three numbers the text quotes from it (``macros``) all come from one place. Every input must
be present and describe the same roster at the same ``k``; anything else fails loudly rather
than rendering a hybrid.

Usage:
  python scripts/repro/generate_supp_rank_table.py
  python scripts/repro/generate_supp_rank_table.py --out /tmp/typed_neighbour_coverage.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M, croma_as_margin
from paper_manifest import by_benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
from plotting.style import CONTROL_MODEL, published_models  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ENTRY = by_benchmark("pathorob-camelyon")
OUT = Path("paper/sections/supp/typed_neighbour_coverage.tex")
RANK_CSV = "typed_neighbor_rank_summary.csv"
RANK_JSON = "typed_neighbor_rank_summary.json"
CONTESTED_CSV = "contested_fraction_summary.csv"
#: The macro family; the table is Camelyon's, so are the numbers the text quotes from it.
PREFIX = ENTRY.prefix

_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


@dataclass(frozen=True)
class Coverage:
    """The joined per-model frame plus the run-level facts the caption and macros state."""

    ranked: pd.DataFrame
    control: pd.DataFrame
    k: int
    n_samples: int
    first_typed_rank: int


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"missing typed-neighbour input: {path}")
    return published_models(pd.read_csv(path))


def _single_k(frame: pd.DataFrame, what: str) -> int:
    ks = sorted(frame["k"].astype(int).unique())
    if len(ks) != 1:
        raise SystemExit(f"{what} is not at one shared k: {ks}")
    return int(ks[0])


def load(root: Path = REPO) -> Coverage:
    """Read and cross-check the run's metrics and its two neighbourhood studies."""
    studies = root / ENTRY.studies_rel
    metrics = _read_csv(root / ENTRY.metrics_rel)
    ranks = _read_csv(studies / RANK_CSV)
    contested = _read_csv(studies / CONTESTED_CSV)
    summary_path = studies / RANK_JSON
    if not summary_path.is_file():
        raise SystemExit(f"missing typed-neighbour input: {summary_path}")
    summary = json.loads(summary_path.read_text())

    roster = set(metrics["model"])
    for name, frame in ((RANK_CSV, ranks), (CONTESTED_CSV, contested)):
        if set(frame["model"]) != roster:
            raise SystemExit(
                f"{name} and {ENTRY.metrics_rel} describe different rosters: "
                f"study-only {sorted(set(frame['model']) - roster)}, "
                f"run-only {sorted(roster - set(frame['model']))}"
            )
    k = _single_k(metrics, ENTRY.metrics_rel)
    contested_k = _single_k(contested, CONTESTED_CSV)
    if contested_k != k:
        raise SystemExit(f"{CONTESTED_CSV} was computed at k={contested_k}, the run is at k={k}")
    ranked_n = len(roster - {CONTROL_MODEL})
    if summary.get("ranked_n_models") != ranked_n:
        raise SystemExit(
            f"{RANK_JSON} pools {summary.get('ranked_n_models')} ranked encoders, the run "
            f"ranks {ranked_n}"
        )

    frame = (
        metrics[["model", "croma", "support"]]
        .assign(croma=lambda d: croma_as_margin(d["croma"]))
        .merge(ranks[["model", "so_med", "os_med"]], on="model")
        .merge(
            contested[
                [
                    "model",
                    "pure_so_frac_of_defined",
                    "pure_os_frac_of_defined",
                    "contested_frac_of_defined",
                ]
            ],
            on="model",
        )
    )
    is_control = frame["model"] == CONTROL_MODEL
    return Coverage(
        ranked=frame[~is_control].sort_values("croma", ascending=False).reset_index(drop=True),
        control=frame[is_control].reset_index(drop=True),
        k=k,
        n_samples=int(summary["n_samples"]),
        first_typed_rank=int(round(float(summary["pooled_first_percentiles_ranked"]["50"]))),
    )


def _pct(fraction: float) -> str:
    return f"{100.0 * float(fraction):.1f}"


def _row(record: pd.Series) -> str:
    cells = [
        str(record["model"]),
        f"{float(record['croma']):.2f}",
        f"{float(record['so_med']):.0f}",
        f"{float(record['os_med']):.0f}",
        _pct(record["support"]),
        _pct(record["pure_so_frac_of_defined"]),
        _pct(record["pure_os_frac_of_defined"]),
        _pct(record["contested_frac_of_defined"]),
    ]
    return " & ".join(cells) + r" \\"


def _thousands(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _count(n: int) -> str:
    return _WORDS[n] if n < len(_WORDS) else str(n)


def _caption(cov: Coverage) -> str:
    m = int(CROMA_HEADLINE_M)
    control = (
        rf" The natural-image control \code{{{CONTROL_MODEL}}} is shown separately."
        if not cov.control.empty
        else ""
    )
    return (
        rf"\textbf{{Typed-neighbour ranks and fixed-$k$ neighbourhood composition on "
        rf"{ENTRY.short_name}.}} Pathology encoders are ordered by median \code{{CRoMa}} "
        rf"($m{{=}}{m}$). Positive values indicate biology-dominant geometry. For each encoder, "
        r"the \textbf{SO} and \textbf{OS} columns report the median rank across samples of the "
        r"nearest cross-confounder biological match and same-confounder biological distractor, "
        rf"respectively. At the shared operating point $k{{=}}{cov.k}$, support is the "
        r"percentage of all samples whose neighbourhood contains at least one \textbf{SO} or "
        r"\textbf{OS} neighbour. Pure-\textbf{SO}, pure-\textbf{OS} and contested neighbourhoods "
        r"partition these supported samples and are reported as percentages of support. Only "
        r"contested neighbourhoods contain both evidence types and can yield different \code{RI} "
        rf"and \code{{MaRI}} values. Across the {_count(len(cov.ranked))} pathology encoders, "
        rf"the nearest typed neighbour occurs at a median rank of approximately "
        rf"${cov.first_typed_rank}$ in a cohort of ${_thousands(cov.n_samples)}$ tiles, well "
        rf"beyond the fixed $k{{=}}{cov.k}$ window.{control}"
    )


def build(root: Path = REPO) -> str:
    cov = load(root)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"& & \multicolumn{2}{c}{median rank}",
        r"& & \multicolumn{3}{c}{among supported samples (\%)} \\",
        r"\cline{3-4}\cline{6-8}",
        r"Model & \code{CRoMa} & \textbf{SO} & \textbf{OS} & support (\%) & pure-\textbf{SO} "
        r"& pure-\textbf{OS} & contested \\",
        r"\hline",
    ]
    lines += [_row(r) for _, r in cov.ranked.iterrows()]
    if not cov.control.empty:
        lines += [r"\hline", *[_row(r) for _, r in cov.control.iterrows()]]
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{_caption(cov)}}}",
        r"\label{tab:typed-neighbour-coverage}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def macros(root: Path = REPO) -> list[str]:
    """The numbers the text quotes from this table, over the ranked encoders only."""
    cov = load(root)
    contested = cov.ranked["contested_frac_of_defined"].astype(float)
    return [
        rf"\newcommand{{\{PREFIX}FirstTypedRankMedian}}{{{cov.first_typed_rank}}}",
        rf"\newcommand{{\{PREFIX}ContestedPctMedian}}{{{_pct(contested.median())}\%}}",
        rf"\newcommand{{\{PREFIX}ContestedPctMax}}{{{_pct(contested.max())}\%}}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rendered = build()
    if not args.out.parent.exists():
        print(f"skip   {args.out} (paper/ absent)", file=sys.stderr)
        return 0
    args.out.write_text(rendered)
    print(f"wrote  {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
