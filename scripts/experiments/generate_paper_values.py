"""Emit a LaTeX macro file of derived per-benchmark scalars cited inline in the paper.

Inline prose numbers (the pooled-CCMR span, counts of confounder-dominant models,
the best lower-tail mean, ...) were historically hand-typed, which let them drift
from the auto-generated result tables (e.g. a span upper bound truncated 0.195->0.19
while ``generate_results_table.py`` rounded the same value to 0.20). This script
derives those scalars from the same ``metrics.csv`` the tables are built from and
writes them as ``\\newcommand`` macros, so the prose can cite ``\\CamelyonCcmrSpan``
instead of a literal. Run it whenever the faithful metrics are regenerated.

Scale safety (see the ratio-vs-margin provenance hazard): CCMR is stored as the
signed margin in ``(-1, 1)`` in the canonical dirs, but some older dirs hold the raw
ratio ``d_OS / d_SO`` in ``(0, inf)``. Each benchmark's scale is auto-detected and a
ratio column is transformed to the margin via ``(r - 1) / (r + 1)`` before any
statistic is computed, so the emitted values are always on the paper's margin scale.

Usage:
  python scripts/experiments/generate_paper_values.py
  python scripts/experiments/generate_paper_values.py --out paper/sections/generated_values.tex
  python scripts/experiments/generate_paper_values.py --scale ratio   # force-transform inputs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# macro prefix (LaTeX commands take letters only -> spell digits out) -> metrics.csv
BENCHMARKS: list[tuple[str, str]] = [
    ("Camelyon", "output/faithful/pathorob-camelyon-faithful/results/metrics.csv"),
    ("TcgaTwoByTwo", "output/faithful/pathorob-tcga-2x2/results/metrics.csv"),
    ("TcgaFourByFour", "output/faithful/pathorob-tcga-4x4/results/metrics.csv"),
    ("Tolkach", "output/faithful/pathorob-tolkach-esca-faithful/results/metrics.csv"),
    ("Panda", "output/panda-wsi-cancer/results/metrics.csv"),
    ("PandaIsup", "output/panda-wsi-isup-paired-2x2/results/metrics.csv"),
    ("Prostate", "output/prostate-shift-binary-kirumc/results/metrics.csv"),
]

# SS-shell (local-entanglement) scalars for concern 6. Sourced from the typed-neighbour
# -rank experiment summary, which carries per-model CCMR, SS-shell exit depth, and the
# fixed-k SS-pocket prevalence (fraction with no typed neighbour among the k nearest).
SS_SHELL_SUMMARY = (
    "Camelyon",
    "output/faithful/pathorob-camelyon-faithful/typed_neighbor_rank_summary.csv",
)
SS_POCKET_K = 10  # reference neighbourhood for the prevalence quoted in prose

# Uncertainty scalars for concerns 3 (UQ) and 6 (redundancy). Sourced from the
# bootstrap_uncertainty experiment: per-model pooled-median CCMR CIs + rank stability
# (CSV) and the cross-model Spearman correlations with bootstrap CIs (JSON).
UNCERTAINTY_SUMMARY = (
    "Camelyon",
    "output/faithful/pathorob-camelyon-faithful/results/bootstrap_uncertainty.json",
    "output/faithful/pathorob-camelyon-faithful/results/bootstrap_uncertainty.csv",
)

# Downstream-validation scalars: the APD<->metric rank correlations that fill
# tab:apd-correlation and the prostate second-organ paragraph in apd_validation.tex.
# Sourced from apd_ccmr_correlation.py's output so the table cells and inline rhos cannot
# drift from the recomputed correlations. `headline` (the three faithful benchmarks, 48
# pairs) is the table's "pooled" column; the all-four `pooled` scope is deliberately NOT
# exported -- prostate's single-centre OOD must not enter a pooled APD_OOD statistic.
APD_CORRELATION_CSV = "output/apd/apd_correlation.csv"
APD_TARGET_MACRO = {"apd_id": "Id", "apd_ood": "Ood"}
APD_METRIC_MACRO = {"ccmr": "Ccmr", "ri": "Ri", "mari": "Mari"}
APD_SCOPE_MACRO = {
    "camelyon": "Camelyon",
    "tcga_4x4": "Tcga",
    "tolkach": "Tolkach",
    "headline": "Pooled",
    "prostate": "Prostate",
}
APD_RANGE_BENCHMARKS = ["camelyon", "tcga_4x4", "tolkach"]


def _detect_scale(ccmr: pd.Series) -> str:
    """Return ``"margin"`` or ``"ratio"`` from the value range.

    Margin lives in ``(-1, 1)`` (neutral at 0); ratio lives in ``(0, inf)`` (neutral
    at 1). A negative value is decisive for margin; a value above 1 is decisive for
    ratio; an all-``[0, 1]`` column (every model biology-dominant) is ambiguous and
    defaults to margin, the paper's canonical scale.
    """
    if (ccmr < 0.0).any():
        return "margin"
    if (ccmr > 1.0 + 1e-9).any():
        return "ratio"
    return "margin"


def _to_margin(ccmr: pd.Series, scale: str) -> pd.Series:
    if scale == "ratio":
        return (ccmr - 1.0) / (ccmr + 1.0)
    return ccmr


def _num(value: float, decimals: int = 2) -> str:
    """Math-mode literal, e.g. ``$-0.46$``; ``-0.00`` is normalised to ``$0.00$``."""
    body = f"{value:.{decimals}f}"
    if float(body) == 0.0:
        body = f"{0.0:.{decimals}f}"  # strip a spurious leading minus
    return f"${body}$"


def _macros_for(prefix: str, df: pd.DataFrame, scale_override: str) -> tuple[list[str], str]:
    raw = df["ccmr"].astype(float)
    scale = scale_override if scale_override != "auto" else _detect_scale(raw)
    ccmr = _to_margin(raw, scale)
    ltm = df["ccmr_ltm_alpha"].astype(float) if "ccmr_ltm_alpha" in df else None

    lo, hi = float(ccmr.min()), float(ccmr.max())
    # Each scalar: (macro-suffix, body). Add a line here to expose a new value.
    specs = [
        ("CcmrSpan", f"{_num(lo)}--{_num(hi)}"),
        ("CcmrMin", _num(lo)),
        ("CcmrMax", _num(hi)),
        ("CcmrConfounderDominant", str(int((ccmr < 0.0).sum()))),
        ("NModels", str(int(len(ccmr)))),
    ]
    if ltm is not None:
        specs.append(("CcmrLtmMax", _num(float(ltm.max()))))  # best (least-negative) tail

    # Leader bundle: the model with the highest (least-negative) CCMR. Lets prose cite the
    # leader's own scalars (name, CCMR, biological k-NN accuracy, support) without drift.
    leader = df.loc[ccmr.idxmax()]
    specs.append(("BestModel", str(leader["model"])))
    specs.append(("BestCcmr", _num(hi)))  # == CcmrMax, named for prose readability
    if "bio_knn_bacc" in df:
        specs.append(("BestBioBacc", _num(float(leader["bio_knn_bacc"]), decimals=3)))
    if "ri_undefined_frac" in df:
        support = 1.0 - float(leader["ri_undefined_frac"])
        specs.append(("BestSupport", rf"{support * 100:.1f}\%"))

    lines = [rf"\newcommand{{\{prefix}{suffix}}}{{{body}}}" for suffix, body in specs]
    return lines, scale


def _pct(value: float) -> str:
    r"""Percent literal, e.g. ``70\%`` (rounded to a whole percent)."""
    return rf"{int(round(value * 100))}\%"


def _ss_shell_macros(prefix: str, df: pd.DataFrame) -> list[str]:
    """SS-shell local-entanglement scalars (concern 6) from the typed-rank summary.

    Quotes the prevalence at the highest-CCMR (leader) and least-entangled models to
    make the "even CCMR leaders are locally SS-saturated" point, plus the rank
    correlations that establish "related, not redundant" rather than independence.
    """
    from scipy.stats import spearmanr

    pocket_col = f"ss_pocket_frac_k{SS_POCKET_K}"
    sub = df[df["ccmr"].notna()].copy()
    leader = sub.loc[sub["ccmr"].idxmax()]
    least = sub.loc[sub[pocket_col].idxmin()]
    depth_rho, _ = spearmanr(sub["ccmr"], sub["ss_depth_med"])
    pocket_rho, _ = spearmanr(sub["ccmr"], sub[pocket_col])
    specs = [
        ("SsPocketK", str(SS_POCKET_K)),
        ("SsLeaderModel", str(leader["model"])),
        ("SsLeaderPocketFrac", _pct(float(leader[pocket_col]))),
        ("SsLeastPocketFrac", _pct(float(least[pocket_col]))),
        ("SsDepthCcmrRho", _num(float(depth_rho))),
        ("SsPocketCcmrRho", _num(float(pocket_rho))),
    ]
    return [rf"\newcommand{{\{prefix}{suffix}}}{{{body}}}" for suffix, body in specs]


def _ci(lo: float, hi: float) -> str:
    """Bracketed CI literal in math mode, e.g. ``$[0.17, 0.23]$``."""
    return rf"$[{lo:.2f}, {hi:.2f}]$"


def _uncertainty_macros(prefix: str, summary: dict, df: pd.DataFrame) -> list[str]:
    """Bootstrap UQ + redundancy scalars (concerns 3 and 6).

    Exposes the cross-model rank correlations with CIs (CCMR vs RI/MaRI, and RI vs
    MaRI to make the ``CCMR is the least rank-redundant of the three'' point), the
    headline-leader CCMR with its CI, the size of the top rank-tie cluster (models
    whose 95% rank interval includes rank 1), and the closest adjacent-pair win
    probability (the coin-flip that shows nearby point estimates are statistical
    ties). All sourced from the bootstrap experiment so prose cannot drift.
    """
    corr = summary["correlations"]
    leader = df.sort_values("ccmr", ascending=False).iloc[0]
    top_tie_n = int((df["rank_lo"] == 1).sum())
    adj = summary.get("adjacent_pair_win", [])
    closest = min(adj, key=lambda d: abs(d["p_higher_beats_lower"] - 0.5)) if adj else None
    specs = [
        ("CcmrVsRiRho", _num(float(corr["ccmr_vs_ri"]["rho"]))),
        ("CcmrVsRiCi", _ci(float(corr["ccmr_vs_ri"]["lo"]), float(corr["ccmr_vs_ri"]["hi"]))),
        ("CcmrVsMariRho", _num(float(corr["ccmr_vs_mari"]["rho"]))),
        ("CcmrVsMariCi", _ci(float(corr["ccmr_vs_mari"]["lo"]), float(corr["ccmr_vs_mari"]["hi"]))),
        ("RiVsMariRho", _num(float(corr["ri_vs_mari"]["rho"]))),
        ("CcmrLeaderModel", str(leader["model"])),
        ("CcmrLeaderVal", _num(float(leader["ccmr"]))),
        ("CcmrLeaderCi", _ci(float(leader["ccmr_lo"]), float(leader["ccmr_hi"]))),
        ("CcmrTopTieN", str(top_tie_n)),
        ("CcmrNBoot", str(int(summary["n_boot"]))),
    ]
    if closest is not None:
        specs.append(("CcmrClosestPairWinProb", _num(float(closest["p_higher_beats_lower"]))))
    return [rf"\newcommand{{\{prefix}{suffix}}}{{{body}}}" for suffix, body in specs]


def _apd_bare(value: float) -> str:
    """Bare 2-dp number (no math delimiters) so a macro nests inside ``$\\rho=...$``.

    Normalises a spurious ``-0.00`` to ``0.00``; the math minus is supplied by the
    surrounding ``$...$`` in prose (e.g. prostate's ``$\\rho=\\ApdOodCcmrProstate$``).
    """
    body = f"{value:.2f}"
    if float(body) == 0.0:
        body = f"{0.0:.2f}"
    return body


def _apd_macros(df: pd.DataFrame) -> list[str]:
    """APD<->metric Spearman cells for tab:apd-correlation + the prostate paragraph.

    One ``\\Apd{Id,Ood}{Ccmr,Ri,Mari}{Camelyon,Tcga,Tolkach,Pooled,Prostate}`` macro per
    cell, plus a ``\\Apd{...}Range`` (``$lo$--$hi$`` over the three faithful benchmarks)
    for the inline ``per benchmark'' ranges.
    """
    lines = []
    for target, t_suffix in APD_TARGET_MACRO.items():
        for metric, m_suffix in APD_METRIC_MACRO.items():
            scoped = (df[(df["target"] == target) & (df["metric"] == metric)]
                      .set_index("scope")["spearman"])
            for scope, s_suffix in APD_SCOPE_MACRO.items():
                if scope not in scoped.index:
                    continue
                macro = f"Apd{t_suffix}{m_suffix}{s_suffix}"
                lines.append(rf"\newcommand{{\{macro}}}{{{_apd_bare(float(scoped[scope]))}}}")
            bench = [float(scoped[b]) for b in APD_RANGE_BENCHMARKS if b in scoped.index]
            if bench:
                lo, hi = _apd_bare(min(bench)), _apd_bare(max(bench))
                lines.append(rf"\newcommand{{\Apd{t_suffix}{m_suffix}Range}}{{${lo}$--${hi}$}}")
    return lines


def build(benchmarks: list[tuple[str, str]], root: Path, scale_override: str) -> str:
    out = [
        "% AUTO-GENERATED by scripts/experiments/generate_paper_values.py -- do not edit by hand.",
        "% Re-run after regenerating the faithful metrics; cite e.g. \\CamelyonCcmrSpan in prose.",
    ]
    for prefix, rel in benchmarks:
        path = root / rel
        if not path.exists():
            print(f"warning: missing {path}, skipping {prefix}", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        lines, scale = _macros_for(prefix, df, scale_override)
        out.append(f"% {prefix}: {rel} (scale={scale})")
        out.extend(lines)
        print(f"{prefix:16s} scale={scale:6s} -> {len(lines)} macros", file=sys.stderr)

    ss_prefix, ss_rel = SS_SHELL_SUMMARY
    ss_path = root / ss_rel
    if ss_path.exists():
        ss_lines = _ss_shell_macros(ss_prefix, pd.read_csv(ss_path))
        out.append(f"% {ss_prefix} SS-shell (concern 6): {ss_rel}")
        out.extend(ss_lines)
        print(f"{ss_prefix + ' SS-shell':16s}        -> {len(ss_lines)} macros", file=sys.stderr)
    else:
        print(f"warning: missing {ss_path}, skipping SS-shell macros", file=sys.stderr)

    unc_prefix, unc_json, unc_csv = UNCERTAINTY_SUMMARY
    json_path, csv_path = root / unc_json, root / unc_csv
    if json_path.exists() and csv_path.exists():
        import json as _json

        summary = _json.loads(json_path.read_text())
        unc_lines = _uncertainty_macros(unc_prefix, summary, pd.read_csv(csv_path))
        out.append(f"% {unc_prefix} uncertainty (concerns 3 & 6): {unc_json}")
        out.extend(unc_lines)
        print(f"{unc_prefix + ' uncertainty':16s}     -> {len(unc_lines)} macros", file=sys.stderr)
    else:
        print(f"warning: missing {json_path} or {csv_path}, skipping UQ macros", file=sys.stderr)

    apd_path = root / APD_CORRELATION_CSV
    if apd_path.exists():
        apd_lines = _apd_macros(pd.read_csv(apd_path))
        out.append(f"% APD downstream validation: {APD_CORRELATION_CSV}")
        out.extend(apd_lines)
        print(f"{'APD validation':16s}        -> {len(apd_lines)} macros", file=sys.stderr)
    else:
        print(f"warning: missing {apd_path}, skipping APD macros", file=sys.stderr)
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Repo root for resolving metrics paths.")
    p.add_argument("--out", type=Path, default=Path("paper/sections/generated_values.tex"))
    p.add_argument("--scale", choices=["auto", "margin", "ratio"], default="auto",
                   help="Input CCMR scale; 'auto' detects per benchmark (default).")
    args = p.parse_args()

    tex = build(BENCHMARKS, args.root, args.scale)
    args.out.write_text(tex)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
