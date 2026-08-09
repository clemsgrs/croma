r"""Render a benchmark's main-results LaTeX table -- body **and** caption -- from its metrics.csv.

Every table in the paper's results family (``tab:main-results`` and its per-benchmark
siblings) is produced here from the manifest in ``paper_manifest.py``. Nothing is typed by
hand: the model roster comes from the run, the operating point comes from the run's ``k``
column, while captions are limited to scope, operating points, columns and formatting.
See ADR-0010.

Captions used to be hand-maintained, which is where stale protocol details accumulated:
``tab:main-results`` announced a shared ``k=15`` for a run at ``k=11``. Structural details are
therefore generated, while observed-result interpretation remains in the main text.

Usage:
  python scripts/repro/generate_results_table.py                 # every manifest entry
  python scripts/repro/generate_results_table.py --only Camelyon # one, by macro prefix
  python scripts/repro/generate_results_table.py --check         # render, diff, don't write
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paper_tables import CROMA_HEADLINE_M, CaptionClaimError, croma_as_margin, scriptsize_ci
from paper_manifest import ResultsTable, rendered

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts" / "bench")
)  # noqa: E402  (plotting.style lives with the benchmark plot library)
from plotting.style import CONTROL_MODEL  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# column -> (header, decimals, percent). ``k`` is deliberately absent: it is a single shared
# value under the median-k protocol, so it belongs in the caption rather than in a column of
# identical cells. ``delta`` is present because Section 3 reads its *sign* per model.
COLS = [
    ("bio_knn_bacc", "bio bacc", 3, False),
    ("confounder_knn_bacc", "conf bacc", 3, False),
    ("ri", r"\code{RI}", 3, False),
    ("mari", r"\code{MaRI}", 3, False),
    ("delta", r"$\Delta$", 3, False),
    ("croma", r"\code{CRoMa}", 2, False),
    ("croma_frac_neg", r"$F(0)$", 3, False),
    ("croma_ltm_alpha", r"$\mcode{LTM}_{10}$", 2, False),
    ("support", "support", 1, True),
]

# Diagnostics, not scores: never bolded, and excluded from the per-column "best". conf bacc's
# max marks the *least* robust models, and delta is sign-informative but not a quality ordering.
NO_BOLD = ("confounder_knn_bacc", "delta")

# F(0) is a prevalence for which lower is better, so its "best" is the minimum, not the maximum.
# It is a scored column (it takes a bold), unlike the diagnostics above.
MINIMIZE = ("croma_frac_neg",)

# delta carries an explicit sign: its sign is the point, its magnitude is secondary.
SIGNED = ("delta",)


def _fmt(value: float, decimals: int, percent: bool, bold: bool, signed: bool = False) -> str:
    if percent:
        body = f"{value:.{decimals}f}\\%"
    elif signed:
        body = f"${value:+.{decimals}f}$"
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


def _load_frac_neg(metrics_csv: Path, headline_m: int) -> dict[str, float]:
    """Per-model F(0) at the headline m: the confounder-dominant fraction.

    The boundary is closed -- an exact zero is confounder-dominant -- matching
    ``CRoMaResult.f0``, the canonical definition the library computes. Undefined
    (non-finite) occurrences leave the denominator, as they do before the tail
    statistics.
    """
    ps = pd.read_csv(
        metrics_csv.parent / "per_sample_metrics.csv",
        usecols=["model", f"croma_m{int(headline_m)}"],
    )
    col = f"croma_m{int(headline_m)}"

    def _f0(s: pd.Series) -> float:
        defined = s[np.isfinite(s)]
        return float((defined <= 0.0).mean())

    return ps.groupby("model")[col].apply(_f0).to_dict()


def load_frame(metrics_csv: Path) -> pd.DataFrame:
    """The run's frame with the paper's derived columns, on the canonical margin scale."""
    df = pd.read_csv(metrics_csv)
    df["croma"] = croma_as_margin(df["croma"])
    df["support"] = (1.0 - df["ri_undefined_frac"]) * 100.0
    df["delta"] = df["mari"].astype(float) - df["ri"].astype(float)
    df["croma_frac_neg"] = df["model"].map(_load_frac_neg(metrics_csv, CROMA_HEADLINE_M))
    return df


def split_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into the ranked panel (pathology encoders) and the natural-image control.

    The control is not a competitor: it is a floor. It never enters the sort or the
    per-column best, and it is rendered in its own band beneath the rule. Some benchmarks
    were never embedded for it, so the control frame may be empty.
    """
    is_control = df["model"] == CONTROL_MODEL
    ranked = df[~is_control].sort_values("croma", ascending=False).reset_index(drop=True)
    return ranked, df[is_control].reset_index(drop=True)


def benchmark_exposed(entry: ResultsTable, ranked: pd.DataFrame) -> set[str]:
    """The panel's marked encoders, read from model_metadata.csv.

    The two TCGA tables mark corpus exposure; Tolkach-ESCA marks possible institutional exposure
    to its scored CHA cohort. Both are resolved from the same metadata as the Pareto figure.
    """
    if not entry.mark_exposure_in_table:
        return set()
    from _distributions import metadata_exposed_models

    return set(metadata_exposed_models(entry, set(ranked["model"])))


# --------------------------------------------------------------------------------------
# Caption. Structural details are derived from the run; observed results are not narrated.
# --------------------------------------------------------------------------------------


def _operating_point(df: pd.DataFrame, protocol: str) -> str:
    ks = sorted(df["k"].unique())
    if protocol == "median-k":
        if len(ks) != 1:
            raise CaptionClaimError(
                f"median-k promises one shared k, run has {ks}. The caption cannot call this "
                f"a shared operating point."
            )
        return (
            rf"All models are evaluated at the shared operating point $k{{=}}{int(ks[0])}$, "
            r"the dataset median of the per-model biological $k^\star$."
        )
    lo, hi = int(min(ks)), int(max(ks))
    span = rf"$k^\star{{=}}{lo}$" if lo == hi else rf"$k^\star$ ranging from ${lo}$ to ${hi}$"
    return rf"Each model is evaluated at its own biological $k^\star$ ({span})."


def _control_clause(control: pd.DataFrame) -> str:
    """The control is rendered in its own band below the rule; the caption is the only thing
    that tells a reader the split is deliberate.

    Three of the four manuscript captions end this clause at "separately"; TCGA-2x2 alone adds
    "beneath". One generator cannot emit both, so it emits the majority form -- see
    build_caption for why the majority is the tie-break rather than the more precise variant.
    """
    if control.empty:
        return ""
    return rf", with the natural-image control \code{{{CONTROL_MODEL}}} shown separately"


def _dagger_clause(entry: ResultsTable, exposed: set[str]) -> str:
    """The domain-appropriate dagger explanation; empty when no model is marked."""
    if not exposed:
        return ""
    if entry.exposure_domain == "charite":
        return (
            rf" $\dagger$ marks the ${len(exposed)}$ RudolfV 2 encoders with possible "
            r"institutional/source-domain overlap: their disclosed training corpus includes "
            r"Charit\'e, and the scored CHA cohort is from Charit\'e. Exact patient or slide "
            r"overlap is unknown; the marker does not establish leakage."
        )
    return (
        rf" $\dagger$ marks the ${len(exposed)}$ TCGA-exposed encoders "
        r"(Table~\ref{tab:model-summary})."
    )


def build_caption(entry: ResultsTable, df: pd.DataFrame, exposed: set[str], with_ci: bool) -> str:
    ranked, control = split_panel(df)
    m = int(CROMA_HEADLINE_M)
    confounder = str(df["confounder_display_name"].iloc[0]).lower().replace("center", "centre")
    # The four manuscript captions drifted apart under hand editing and disagree on three
    # independent wordings: the article before the model count ("for 20" vs "for the 20",
    # split 2-2), "ordered by median CRoMa" vs "ordered by CRoMa" (3-1), and whether the
    # control clause ends "separately" or "separately beneath" (3-1). No single rule emits
    # all four, so this one takes the majority on each axis. That is deliberately not the
    # most precise wording -- "beneath" says something true that "separately" leaves out --
    # but it is the wording that reproduces the most of the paper unchanged: TCGA-4x4 and
    # Tolkach come out byte-identical, Camelyon differs by one article, TCGA-2x2 by three
    # words. Prefer a variant? Change it here and all four tables follow.
    head = (
        rf"\textbf{{Representation robustness on {entry.short_name}.}} Pooled results for "
        rf"the {len(ranked)} {entry.model_type} pathology foundation models, ordered by "
        rf"median \code{{CRoMa}} ($m{{=}}{m}$){_control_clause(control)}. "
        rf"{_operating_point(df, entry.protocol)}"
    )
    cols = (
        r" Columns: biological and confounder $k$-NN balanced accuracy (bio bacc and conf "
        rf"bacc; confounder: {confounder}); pooled \code{{RI}} "
        r"and \code{MaRI}; $\Delta{=}\code{MaRI}-\code{RI}$; median \code{CRoMa}; $F(0)$, the "
        r"fraction with $\mcode{CRoMa}\le0$; $\mcode{LTM}_{10}$, the mean of the lowest decile; "
        r"and support, the fraction of samples effectively contributing to \code{RI}/\code{MaRI}. "
        r"Bold denotes the best value in each score column (conf bacc and $\Delta$ are diagnostics)."
    )
    ci = (
        r" \code{CRoMa} brackets are 95\% group-level cluster-bootstrap confidence intervals on "
        r"the pooled median (Supplementary Table~\ref{tab:bootstrap-uncertainty})."
        if with_ci
        else ""
    )
    return (head + cols + _dagger_clause(entry, exposed) + ci).strip()


# --------------------------------------------------------------------------------------


def build_table(entry: ResultsTable, metrics_csv: Path) -> str:
    df = load_frame(metrics_csv)
    ranked, control = split_panel(df)
    # Whether to show CIs is the manifest's decision, not an accident of which study
    # artifacts happen to be on disk. Asking for them without the artifact is an error;
    # silently rendering a table without the CIs its caption promises is worse.
    croma_ci = _load_croma_ci(metrics_csv) if entry.with_ci else None
    if entry.with_ci and croma_ci is None:
        raise FileNotFoundError(
            f"{entry.prefix}: with_ci is set but {metrics_csv.parent}/bootstrap_uncertainty.csv "
            "is absent. Run scripts/studies/bootstrap_uncertainty.py, or clear with_ci."
        )
    best = {
        col: (ranked[col].min() if col in MINIMIZE else ranked[col].max())
        for col, _, _, _ in COLS
        if col not in NO_BOLD
    }
    # TCGA tables mark corpus exposure; Tolkach marks conservative institutional exposure.
    exposed = benchmark_exposed(entry, ranked)

    def row(record: pd.Series, *, bold: bool) -> str:
        name = record["model"]
        cells = [name + (r"$^{\dagger}$" if name in exposed else "")]
        for col, _, dec, pct in COLS:
            is_best = bold and col not in NO_BOLD and abs(record[col] - best[col]) < 1e-9
            cell = _fmt(record[col], dec, pct, is_best, signed=col in SIGNED)
            if col == "croma" and croma_ci and record["model"] in croma_ci:
                cell += r"\," + scriptsize_ci(*croma_ci[record["model"]])
            cells.append(cell)
        return " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(COLS) + "}",
        r"\hline",
        " & ".join(["Model"] + [h for _, h, _, _ in COLS]) + r" \\",
        r"\hline",
    ]
    lines += [row(r, bold=True) for _, r in ranked.iterrows()]
    if not control.empty:
        # The control sits in its own band: a rule separates it from the encoders it is a
        # floor for, and none of its cells can win a bold.
        lines.append(r"\hline")
        lines += [row(r, bold=False) for _, r in control.iterrows()]
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{build_caption(entry, df, exposed, with_ci=croma_ci is not None)}}}",
        rf"\label{{{entry.label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render(entry: ResultsTable, root: Path = REPO) -> str:
    return build_table(entry, root / entry.metrics_rel)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Render just this manifest entry, by macro prefix.")
    p.add_argument(
        "--check",
        action="store_true",
        help="Render and report drift against disk; write nothing. Exit 1 if stale.",
    )
    p.add_argument("--root", type=Path, default=REPO)
    args = p.parse_args()

    entries = rendered()
    if args.only:
        entries = [e for e in entries if e.prefix == args.only]
        if not entries:
            print(f"no rendered manifest entry named {args.only!r}", file=sys.stderr)
            return 2

    stale = 0
    for entry in entries:
        tex = render(entry, args.root)
        out = args.root / entry.out_tex
        if args.check:
            current = out.read_text() if out.exists() else ""
            if current != tex:
                stale += 1
                print(f"STALE  {entry.out_tex}", file=sys.stderr)
            else:
                print(f"ok     {entry.out_tex}", file=sys.stderr)
            continue
        if not out.parent.exists():
            print(f"skip   {entry.out_tex} (paper/ absent)", file=sys.stderr)
            continue
        out.write_text(tex)
        n = len(load_frame(args.root / entry.metrics_rel))
        print(f"wrote  {entry.out_tex}  ({n} models)", file=sys.stderr)

    if args.check and stale:
        print(f"\n{stale} table(s) stale; run scripts/repro/build_paper.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
