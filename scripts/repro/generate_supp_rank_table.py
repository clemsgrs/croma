"""Render the sealed historical typed-neighbour-rank analysis.

The study predates the five encoders added in issues #129--#132.  Its immutable snapshot
contains 20 pathology encoders plus the DINOv2-B control and must never be joined to the
live 25-encoder metrics.  This renderer therefore validates and reads the co-located
summary and metrics from that sealed snapshot only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from _paper_tables import CROMA_HEADLINE_M
from paper_manifest import HISTORICAL_TYPED_NEIGHBOUR_RUN, by_benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench"))
from plotting.style import CONTROL_MODEL  # noqa: E402

ENTRY = by_benchmark("pathorob-camelyon")
SNAPSHOT = Path(HISTORICAL_TYPED_NEIGHBOUR_RUN.run_rel)
RANK_CSV = SNAPSHOT / "studies/typed_neighbor_rank_summary.csv"
SUMMARY_JSON = SNAPSHOT / "studies/typed_neighbor_rank_summary.json"
METRICS_CSV = SNAPSHOT / "results/metrics.csv"
EXPECTED_SHA256 = {
    RANK_CSV: "2facf7719b38ae7890d01dafe5b7b90c61b53a4409bba400c37a525d05fabc26",
    SUMMARY_JSON: "6f6c981bbad86383af05c4cd3741114423650de94149fbda45942e6225d17239",
    METRICS_CSV: "fa87db10dfc767e2b5b217a2f1cea89c95985bbf87853c139e48900040dc2aa6",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_inputs() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise SystemExit(f"missing sealed typed-neighbour input: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise SystemExit(
                f"sealed typed-neighbour input drifted: {path} "
                f"(expected {expected}, got {actual})"
            )

    rank = pd.read_csv(RANK_CSV).set_index("model")
    summary = json.loads(SUMMARY_JSON.read_text())
    metrics = pd.read_csv(METRICS_CSV).set_index("model")
    if set(rank.index) != set(metrics.index):
        raise SystemExit(
            "sealed typed-neighbour summary and co-located metrics rosters differ: "
            f"summary={sorted(rank.index)}, metrics={sorted(metrics.index)}"
        )
    if CONTROL_MODEL not in rank.index:
        raise SystemExit(f"sealed typed-neighbour roster lacks control {CONTROL_MODEL}")
    ranked_n = len(rank.index) - 1
    if summary.get("ranked_n_models") != ranked_n:
        raise SystemExit(
            "sealed typed-neighbour JSON count disagrees with its metrics roster: "
            f"expected {ranked_n}, got {summary.get('ranked_n_models')}"
        )
    return rank, summary, metrics


def _row(model: str, row: pd.Series) -> str:
    return (
        f"{model} & {row['croma']:.2f} & {int(round(row['so_med']))} & "
        f"{int(round(row['os_med']))} \\\\"
    )


def _thousands(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def build() -> str:
    rank, summary, metrics = _validated_inputs()
    # Use the snapshot's own CRoMa values.  Joining live metrics would turn this into an
    # incoherent hybrid if a pre-existing model ever moves in a later benchmark run.
    frame = rank.drop(columns=["croma"], errors="ignore").join(metrics[["croma"]], how="inner")
    ranked = frame.drop(index=CONTROL_MODEL).sort_values("croma", ascending=False)
    control = frame.loc[[CONTROL_MODEL]]
    ranked_n = int(summary["ranked_n_models"])
    first_typed_rank = int(round(float(summary["pooled_first_percentiles_ranked"]["50"])))
    n_candidates = int(round(float(ranked["n_def"].median())))
    operating_ks = metrics["k"].astype(int).unique().tolist()
    if len(operating_ks) != 1:
        raise SystemExit(f"historical metrics do not have one shared k: {operating_ks}")
    operating_k = operating_ks[0]

    lines = [
        f"% Sealed snapshot artifact SHA-256: {EXPECTED_SHA256[RANK_CSV]}",
        f"% Sealed snapshot summary SHA-256: {EXPECTED_SHA256[SUMMARY_JSON]}",
        f"% Co-located snapshot metrics SHA-256: {EXPECTED_SHA256[METRICS_CSV]}",
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
    lines.extend(_row(model, row) for model, row in ranked.iterrows())
    lines.extend([r"\hline", _row(CONTROL_MODEL, control.iloc[0]), r"\hline", r"\end{tabular}"])
    lines.extend(
        [
            rf"\caption{{\textbf{{Typed-neighbour ranks on {ENTRY.short_name}.}} This is a "
            rf"historical fixed {ranked_n}-pathology-encoder analysis; it is not part of the "
            r"live 25-encoder comparisons. Models are ordered by the snapshot's pooled "
            rf"$\mcode{{CRoMa}}(m{{=}}{int(CROMA_HEADLINE_M)})$. Median \code{{SO}} and "
            r"\code{OS} ranks give the nearest cross-confounder biological match and "
            r"same-confounder biological distractor. Pooled across that fixed historical "
            rf"roster, the first \code{{SO}} or \code{{OS}} neighbour occurs at a median rank "
            rf"of $\approx {first_typed_rank}$ among ${_thousands(n_candidates)}$ candidates, "
            rf"far beyond its $k{{=}}{operating_k}$ operating point. \code{{{CONTROL_MODEL}}} "
            r"is reported separately as the natural-image control.}",
            r"\label{tab:typed-neighbour-ranks}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("paper/sections/supp/typed_neighbor_ranks.tex")
    )
    args = parser.parse_args()
    rendered = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
