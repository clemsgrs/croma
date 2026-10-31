"""Shared basis for the bootstrap-uncertainty supplement and its in-text macros.

Two studies feed it:

* ``scripts/studies/bootstrap_uncertainty.py`` writes, under each benchmark run's
  ``studies/`` directory, ``bootstrap_uncertainty.{csv,json}`` (per-encoder median, F(0) and
  LTM_10 with CIs, rank intervals, frontier probability, separated adjacent pairs), and under
  ``output/studies/bootstrap-uncertainty/`` the cross-benchmark rank agreement and the
  mean-rank frontier probability;
* ``scripts/studies/mari_ri_delta.py`` writes ``mari_ri_delta.{csv,json}`` under each tile
  run's ``studies/`` directory (Delta = MaRI - RI with a paired cluster-bootstrap CI).

Two consumers read them and must agree: ``generate_uncertainty_supp_table.py`` renders the
section, ``generate_paper_values.py`` emits the scalars the text quotes. A missing study output
raises: these floats and macros are cited, so a warn-and-skip would silently delete them.

Every statistic is over the ranked encoders; the natural-image control holds no rank.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "src", HERE, REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _grade_matched import macro_suffix  # noqa: E402
from _paper_tables import CROMA_HEADLINE_M, CaptionClaimError, bare_num, num_math  # noqa: E402
from paper_manifest import by_prefix  # noqa: E402
from plotting.style import CONTROL_MODEL, published_models  # noqa: E402

GENERATOR = "scripts/repro/generate_uncertainty_supp_table.py"
CROSS_DIR = Path("output/studies/bootstrap-uncertainty")
REMEDY = "run scripts/studies/bootstrap_uncertainty.py"
DELTA_REMEDY = "run scripts/studies/mari_ri_delta.py"

#: Benchmarks with a per-encoder table, in the order the section renders them.
PREFIXES = ("Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach", "Panda")
#: The tile benchmarks the Delta table covers.
DELTA_PREFIXES = ("Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach")
#: The slide benchmark's name in the paper; the tile benchmarks use their manifest short name.
_LABELS = {"Panda": "PCaBiop"}
_TABLE_SUFFIX = {
    "Camelyon": "camelyon",
    "TcgaTwoByTwo": "tcga2x2",
    "TcgaFourByFour": "tcga4x4",
    "Tolkach": "tolkach",
    "Panda": "pcabiop",
}
#: What a resampled cluster is: a slide, except Tolkach-ESCA's cases (one case, many classes).
_CLUSTER_NOUN = {"Tolkach": "cases"}
_CONFOUNDER = {"Panda": "data provider"}
#: The benchmarks the main text's cross-cohort agreement is over (TCGA-2x2 shares TCGA-4x4's
#: cohort, so a pair of the two is not cross-cohort).
CROSS_COHORT = ("Camelyon", "TcgaFourByFour", "Tolkach")
DELTA_LABEL = "tab:mari-ri-delta"
CROSS_LABEL = "tab:cross-benchmark-rho"


def label(prefix: str) -> str:
    return _LABELS.get(prefix) or by_prefix(prefix).short_name


def table_label(prefix: str) -> str:
    """The per-encoder uncertainty table of one benchmark; results captions cite it."""
    return f"tab:bootstrap-uncertainty-{_TABLE_SUFFIX[prefix]}"


@dataclass(frozen=True)
class Benchmark:
    """One benchmark's per-encoder uncertainty: ranked encoders, by point rank."""

    prefix: str
    table: pd.DataFrame
    meta: dict
    #: Whether the run holds the natural-image control (the slide panel has none).
    has_control: bool = True


@dataclass(frozen=True)
class Delta:
    """One tile benchmark's Delta = MaRI - RI intervals; the control is flagged."""

    prefix: str
    table: pd.DataFrame
    meta: dict

    @property
    def ranked(self) -> pd.DataFrame:
        return self.table[~self.table["is_control"].astype(bool)]


@dataclass(frozen=True)
class Study:
    benchmarks: tuple[Benchmark, ...]
    deltas: tuple[Delta, ...]
    #: Every pair of tile benchmarks: a, b, n, rho, ci_lo, ci_hi, p_value, headline.
    cross: pd.DataFrame
    #: The mean over the headline pairs with its CI, and the draw's settings.
    cross_meta: dict
    #: model, p_frontier on the cross-benchmark mean-rank frontier.
    mean_rank: pd.DataFrame

    @property
    def n_boot(self) -> int:
        counts = {int(b.meta["n_boot"]) for b in self.benchmarks}
        counts |= {int(d.meta["n_boot"]) for d in self.deltas} | {int(self.cross_meta["n_boot"])}
        if len(counts) != 1:
            raise CaptionClaimError(f"the uncertainty studies disagree on n_boot: {counts}")
        return counts.pop()

    def benchmark(self, prefix: str) -> Benchmark:
        (out,) = [b for b in self.benchmarks if b.prefix == prefix]
        return out


def _read(path: Path, remedy: str) -> pd.DataFrame | dict:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; {remedy}")
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path, float_precision="round_trip")


def load(root: Path = REPO) -> Study:
    root = Path(root)
    benchmarks = []
    for prefix in PREFIXES:
        base = root / by_prefix(prefix).studies_rel
        table = published_models(_read(base / "bootstrap_uncertainty.csv", REMEDY))
        meta = _read(base / "bootstrap_uncertainty.json", REMEDY)
        if len(table) != int(meta["n_models"]):
            raise CaptionClaimError(f"{prefix}: {len(table)} rows for {meta['n_models']} models")
        roster = pd.read_csv(root / by_prefix(prefix).metrics_rel, usecols=["model"])["model"]
        benchmarks.append(
            Benchmark(prefix, table.sort_values("point_rank"), meta, CONTROL_MODEL in set(roster))
        )
    deltas = []
    for prefix in DELTA_PREFIXES:
        base = root / by_prefix(prefix).studies_rel
        table = published_models(_read(base / "mari_ri_delta.csv", DELTA_REMEDY))
        deltas.append(Delta(prefix, table, _read(base / "mari_ri_delta.json", DELTA_REMEDY)))
    cross_dir = root / CROSS_DIR
    return Study(
        benchmarks=tuple(benchmarks),
        deltas=tuple(deltas),
        cross=_read(cross_dir / "cross_benchmark_rho.csv", REMEDY),
        cross_meta=_read(cross_dir / "cross_benchmark_rho.json", REMEDY),
        mean_rank=published_models(_read(cross_dir / "mean_rank_frontier.csv", REMEDY)),
    )


# -- rendering --------------------------------------------------------------------------


def _ci(lo: float, hi: float, decimals: int, signed: bool = False) -> str:
    if signed:
        body = f"{lo:+.{decimals}f}, {hi:+.{decimals}f}"
    else:
        body = f"{bare_num(lo, decimals)}, {bare_num(hi, decimals)}"
    return rf"{{\scriptsize$[{body}]$}}"


def _cell(value: float, lo: float, hi: float, decimals: int, signed: bool = False) -> str:
    signed_point = f"{value:+.{decimals}f}"
    if signed and float(signed_point) != 0.0:
        point = f"${signed_point}$"
    else:  # a point that rounds to zero carries no sign
        point = num_math(value, decimals)
    return rf"{point}\,{_ci(lo, hi, decimals, signed)}"


def _rank_cell(row: pd.Series) -> str:
    return rf"${int(row['point_rank'])}$\,{{\scriptsize$[{int(row['rank_lo'])}, {int(row['rank_hi'])}]$}}"


def _float(columns: str, header: list[str], body: list[str], caption: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            rf"\begin{{tabular}}{{{columns}}}",
            r"\hline",
            *header,
            r"\hline",
            *body,
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )


def _design_clause(b: Benchmark, n_boot: int) -> str:
    strata = r"class $\times$ " if b.meta["strata"] == "class x centre" else ""
    strata += _CONFOUNDER.get(b.prefix, "centre")
    noun = _CLUSTER_NOUN.get(b.prefix, "slides")
    return (
        rf"${n_boot}$ stratified cluster-bootstrap replicates, each redrawing the "
        rf"${int(b.meta['n_clusters'])}$ {noun} with replacement within their "
        rf"${int(b.meta['n_strata'])}$ {strata} strata"
    )


def render_benchmark(b: Benchmark, n_boot: int) -> str:
    """Median [CI], rank [interval], F(0) [CI], LTM_10 [CI] and P(frontier) per encoder."""
    body = []
    for _, r in b.table.iterrows():
        name = rf"\textbf{{{r['model']}}}" if bool(r["on_frontier"]) else str(r["model"])
        body.append(
            " & ".join(
                [
                    name,
                    _cell(r["median"], r["median_lo"], r["median_hi"], 2),
                    _rank_cell(r),
                    _cell(r["f0"], r["f0_lo"], r["f0_hi"], 3),
                    _cell(r["ltm"], r["ltm_lo"], r["ltm_hi"], 2),
                    num_math(float(r["p_frontier"]), 3),
                ]
            )
            + r" \\"
        )
    ltm = rf"$\mcode{{LTM}}_{{{round(100 * float(b.meta['alpha']))}}}$"
    header = [rf"Model & Median & Rank & $F(0)$ & {ltm} & $P(\text{{frontier}})$ \\"]
    kind = by_prefix(b.prefix).model_type
    subsets = (
        rf" As the benchmark reports it, the median is the median of the "
        rf"${int(b.meta['n_subsets'])}$ paired subsets' medians, each subset's median recomputed "
        rf"over the slides a replicate draws; $F(0)$ and {ltm} pool the subsets."
        if int(b.meta["n_subsets"]) > 1
        else ""
    )
    caption = (
        rf"\textbf{{Bootstrap uncertainty of per-sample \code{{CRoMa}} on {label(b.prefix)}.}} "
        rf"For each of the ${len(b.table)}$ ranked {kind} encoders"
        rf"{' (the natural-image control is excluded)' if b.has_control else ''}, the median, "
        rf"$F(0)$ and {ltm} of per-sample "
        rf"\code{{CRoMa}} ($m{{=}}{int(CROMA_HEADLINE_M)}$) with 95\% percentile intervals from "
        rf"{_design_clause(b, n_boot)}. Per-sample margins are held fixed; neighbours are not "
        r"recomputed. Rank is the point rank of the median with its 95\% bootstrap interval. "
        r"$P(\text{frontier})$ is the fraction of replicates in which the encoder lies on the "
        rf"median--{ltm} Pareto frontier; bold names mark the point-estimate "
        rf"frontier. ${int(b.meta['n_separated_adjacent'])}$ of the "
        rf"${int(b.meta['n_adjacent'])}$ adjacent pairs in the point ranking are separated: the "
        r"higher-ranked encoder has the larger median in at least "
        rf"{100 * float(b.meta['separation_threshold']):g}\% of replicates.{subsets}"
    )
    return _float("lccccc", header, body, caption, table_label(b.prefix))


def render_delta(study: Study) -> str:
    """Delta [CI] per ranked encoder x tile benchmark, and the count whose CI excludes 0."""
    deltas = _deltas(study)
    frames = {p: d.ranked.set_index("model") for p, d in deltas}
    models = sorted(set.union(*(set(f.index) for f in frames.values())), key=str.casefold)
    body = []
    for model in models:
        cells = [model]
        for prefix, _ in deltas:
            if model not in frames[prefix].index:
                cells.append("---")
                continue
            r = frames[prefix].loc[model]
            cell = _cell(r["delta"], r["delta_lo"], r["delta_hi"], 3, signed=True)
            # \textbf does not reach into math mode; \boldmath does.
            cells.append(rf"{{\boldmath{cell}}}" if bool(r["excludes_zero"]) else cell)
        body.append(" & ".join(cells) + r" \\")
    counts = [rf"${int(d.meta['n_excluding_zero'])}/{int(d.meta['n_ranked'])}$" for _, d in deltas]
    body += [r"\hline", r"CI excludes $0$ & " + " & ".join(counts) + r" \\"]
    header = [
        "Model & "
        + " & ".join(rf"{label(p)} ($k{{=}}{int(d.meta['k'])}$)" for p, d in deltas)
        + r" \\"
    ]
    caption = (
        r"\textbf{Paired cluster-bootstrap intervals for $\Delta=\mcode{MaRI}-\mcode{RI}$ on "
        r"the tile benchmarks.} $\Delta$ per ranked encoder (the natural-image control is "
        r"excluded) at each benchmark's operating $k$, with its 95\% percentile interval from "
        rf"${study.n_boot}$ replicates. Neighbours are computed once on the full cohort, as the "
        r"benchmark does, and held fixed; the point \code{RI} and \code{MaRI} reproduce the "
        r"published values exactly. Each replicate redraws the anchors by the stratified "
        r"cluster draw of the per-encoder uncertainty tables, one draw shared by \code{RI}, "
        r"\code{MaRI} and every encoder; it recomputes both as pooled ratios over the drawn "
        r"anchors' typed neighbours and re-resolves the automatic $\tau$ as the median "
        r"typed-neighbour distance over those anchors. Bold marks an interval that excludes "
        r"$0$; the last row counts them."
    )
    return _float("l" + "c" * len(DELTA_PREFIXES), header, body, caption, DELTA_LABEL)


def _deltas(study: Study) -> list[tuple[str, Delta]]:
    by = {d.prefix: d for d in study.deltas}
    return [(p, by[p]) for p in DELTA_PREFIXES]


def render_cross(study: Study) -> str:
    """Spearman rho between benchmarks' median CRoMa, every tile pair, and the headline mean."""
    body = []
    for _, r in study.cross.iterrows():
        mark = "$^{*}$" if bool(r["headline"]) else ""
        body.append(
            f"{label(r['a'])}--{label(r['b'])}{mark} & ${int(r['n'])}$ & "
            f"{_cell(r['rho'], r['ci_lo'], r['ci_hi'], 2)}" + r" \\"
        )
    meta = study.cross_meta
    body += [
        r"\hline",
        rf"Mean$^{{*}}$ & ${int(meta['n_models'])}$ & "
        f"{_cell(meta['rho'], meta['ci_lo'], meta['ci_hi'], 2)}" + r" \\",
    ]
    header = [r"Benchmarks & $n$ & Spearman $\rho$ \\"]
    prefixes = {p for pair in meta["pairs"] for p in pair}
    if prefixes != set(CROSS_COHORT):
        raise CaptionClaimError(
            f"the caption says the mean leaves TCGA-2x2 out, over {sorted(CROSS_COHORT)}, but the "
            f"study's mean is over {sorted(prefixes)}"
        )
    headline = sorted(label(p) for p in prefixes)
    caption = (
        r"\textbf{Cross-benchmark rank agreement of median \code{CRoMa}.} Spearman $\rho$ "
        r"between the ranked encoders' median \code{CRoMa} on each pair of tile benchmarks, "
        rf"with 95\% encoder-bootstrap intervals from ${int(meta['n_boot'])}$ replicates; one "
        r"draw of encoders is shared by every pair and by the mean, so the intervals are over "
        r"this panel of encoders. The mean is over the "
        rf"${int(meta['n_pairs'])}$ pairs marked $^{{*}}$, among {', '.join(headline[:-1])} "
        rf"and {headline[-1]}: the cross-cohort agreement the main text quotes. TCGA-2x2 is "
        r"left out of the mean because it shares the TCGA cohort with TCGA-4x4."
    )
    return _float("lcc", header, body, caption, CROSS_LABEL)


def render(study: Study) -> str:
    n_boot = study.n_boot
    parts = [f"% AUTO-GENERATED by {GENERATOR} -- do not edit by hand."]
    parts += [render_benchmark(b, n_boot) for b in study.benchmarks]
    parts += [render_delta(study), render_cross(study)]
    return "\n\n".join(parts) + "\n"


# -- macros -----------------------------------------------------------------------------


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def macros(study: Study, rank_frontier: tuple[str, ...]) -> list[str]:
    """Frontier probabilities, separated adjacent pairs and the Delta counts.

    ``rank_frontier`` is the point mean-rank frontier (``_rank_pareto.RankPareto.frontier``,
    the one ``fig:croma-pareto-rank`` rings); its members' probabilities come from the study.
    """
    lines = [
        f"% Bootstrap uncertainty ({CROSS_DIR} and each run's studies/)",
        _macro("UncertaintyNBoot", str(study.n_boot)),
    ]
    for b in study.benchmarks:
        frontier = b.table[b.table["on_frontier"].astype(bool)]
        if frontier.empty:
            raise CaptionClaimError(f"{b.prefix}: the point median-LTM frontier is empty")
        lines.append(f"% {b.prefix}: frontier probability and separated adjacent pairs")
        for _, r in frontier.sort_values("model").iterrows():
            lines.append(
                _macro(
                    f"{b.prefix}FrontierProb{macro_suffix(str(r['model']))}",
                    num_math(float(r["p_frontier"]), 2),
                )
            )
        lines += [
            _macro(f"{b.prefix}SeparatedAdjacentPairs", str(int(b.meta["n_separated_adjacent"]))),
            _macro(f"{b.prefix}AdjacentPairs", str(int(b.meta["n_adjacent"]))),
        ]
    p = study.mean_rank.set_index("model")["p_frontier"]
    missing = sorted(set(rank_frontier) - set(p.index))
    if missing or not rank_frontier:
        raise CaptionClaimError(f"mean-rank frontier members absent from the study: {missing}")
    lines.append("% Cross-benchmark mean-rank frontier: membership probability")
    lines += [
        _macro(f"TileRankFrontierProb{macro_suffix(m)}", num_math(float(p[m]), 2))
        for m in sorted(rank_frontier)
    ]
    lines.append("% Delta = MaRI - RI: ranked encoders whose paired-bootstrap CI excludes 0")
    for d in study.deltas:
        lines += [
            _macro(f"{d.prefix}DeltaCiExcludesZero", str(int(d.meta["n_excluding_zero"]))),
            _macro(f"{d.prefix}DeltaNRanked", str(int(d.meta["n_ranked"]))),
        ]
    return lines


def cross_cohort_ci_macros(study: Study, rhos: dict[frozenset, float]) -> list[str]:
    """CIs for the cross-cohort agreement, after checking the study scored the same pairs.

    ``rhos`` maps each pair of benchmark prefixes to the rho ``\\CrossCohortRhoMean`` and
    ``\\CrossCohortRhoRange`` are computed from; the study's point rho must match each.
    """
    headline = study.cross[study.cross["headline"].astype(bool)]
    by_pair = {frozenset((r["a"], r["b"])): r for _, r in headline.iterrows()}
    if set(by_pair) != set(rhos):
        raise CaptionClaimError(
            f"the uncertainty study's headline pairs {sorted(map(sorted, by_pair))} are not "
            f"the cross-cohort pairs {sorted(map(sorted, rhos))}; {REMEDY}"
        )
    for pair, rho in rhos.items():
        if abs(float(by_pair[pair]["rho"]) - rho) > 1e-12:
            raise CaptionClaimError(f"{sorted(pair)}: study rho differs from metrics.csv")
    meta = study.cross_meta
    lines = [
        "% Cross-cohort rank agreement: encoder-bootstrap CIs (one draw shared by every pair)",
        _macro(
            "CrossCohortRhoMeanCi",
            rf"$[{bare_num(meta['ci_lo'])}, {bare_num(meta['ci_hi'])}]$",
        ),
    ]
    for _, r in headline.iterrows():
        lines.append(
            _macro(
                f"CrossCohortRho{r['a']}{r['b']}Ci",
                rf"$[{bare_num(r['ci_lo'])}, {bare_num(r['ci_hi'])}]$",
            )
        )
    return lines
