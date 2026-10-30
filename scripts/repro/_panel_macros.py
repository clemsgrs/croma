r"""Roster-dependent macros for ``generated_values.tex``: rankings, frontiers, operating points.

The text quotes a family of numbers that move whenever an encoder joins the tile panel: who
sits on each benchmark's median-vs-tail Pareto frontier, how many encoders are confounder-
dominant or have a negative tail, the top three, the operating ``k``, where the natural-image
control lands, each cohort's composition and per-cell counts (the dataset-cardinality
captions), TCGA-2x2's rank agreement, and how closely APD and nIPD rank the encoders. They were hand-typed and went stale at 26 encoders. Every one is derived here
from the runs ``paper_manifest`` names.

Unlike the optional study blocks of ``generate_paper_values.build``, nothing here warns and
skips: a missing run or study raises, because a skipped ``\newcommand`` is a sentence that no
longer compiles. Every cross-model statistic is over the ranked encoders -- the natural-image
control is a floor, not a competitor -- and is reported separately where the text needs it.

The builders are pure (frame in, lines out) so committed tests can drive them without the
git-ignored ``output/`` tree; ``panel_macros`` is the one loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (HERE, REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "studies" / "apd"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _cohort_counts import CELL, QUARTET_CELL, balance, cells  # noqa: E402
from _paper_tables import CaptionClaimError, bare_num, croma_as_margin, num_math  # noqa: E402
from paper_manifest import by_benchmark, by_prefix  # noqa: E402
from plotting.base import _pareto_frontier_max_max  # noqa: E402
from plotting.style import CONTROL_MODEL, published_models  # noqa: E402

#: The tile benchmarks whose median-vs-tail Pareto figure the paper shows.
FRONTIER_PREFIXES = ("Camelyon", "TcgaFourByFour", "Tolkach")
#: Every tile benchmark the text states an operating point and a control position for.
OPERATING_PREFIXES = ("Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach")
#: TCGA-2x2 is the second view of TCGA: its agreement with TCGA-4x4 is reported on its own,
#: and with the two genuinely different cohorts as a range. It stays out of the cross-cohort
#: mean (``generate_paper_values.CROSS_COHORT_BENCHMARKS``), which this does not touch.
TCGA2X2 = "TcgaTwoByTwo"
TCGA2X2_TWIN = "TcgaFourByFour"
TCGA2X2_OTHERS = ("Camelyon", "Tolkach")
#: The join the nIPD floats read (``scripts/studies/apd/apd_croma_correlation.py``).
APD_JOINED_CSV = "output/studies/apd/apd_metrics_joined.csv"
APD_REGIMES = (("id", "Id"), ("ood", "Ood"))
#: Every cohort the dataset-cardinality figures show, by macro prefix.
CARDINALITY_PREFIXES = ("Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach", "Panda", "PandaIsup")
#: A Tolkach-ESCA ``group_id`` is a case, which contributes tiles to several classes.
CARDINALITY_GROUP = {"Tolkach": "Cases"}
#: Evaluated per slide: their cells state slides only.
SLIDE_LEVEL_PREFIXES = ("Panda", "PandaIsup")

_ORDINALS = ("One", "Two", "Three")


def _macro(prefix: str, suffix: str, body: str) -> str:
    return rf"\newcommand{{\{prefix}{suffix}}}{{{body}}}"


def _ranked(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["model"] != CONTROL_MODEL].copy()


def _names(frame: pd.DataFrame) -> str:
    return ", ".join(sorted(frame["model"].astype(str)))


def ranking_macros(prefix: str, df: pd.DataFrame) -> list[str]:
    """Frontier, sign/tail counts, F(0) extremes and top three over the ranked encoders."""
    ranked = _ranked(df)
    ranked["croma"] = croma_as_margin(ranked["croma"]).astype(float)
    ranked["croma_ltm_alpha"] = ranked["croma_ltm_alpha"].astype(float)
    ranked["croma_f0"] = ranked["croma_f0"].astype(float)

    # The same frontier the Pareto figure rings: median CRoMa and LTM_10, both maximised,
    # named in the staircase order (median ascending) the figure draws it in.
    frontier = _pareto_frontier_max_max(
        [
            (str(r["model"]), float(r["croma"]), float(r["croma_ltm_alpha"]))
            for _, r in ranked.iterrows()
        ]
    )
    if not frontier:
        raise CaptionClaimError(f"{prefix}: the median-vs-LTM Pareto frontier is empty.")
    if len(ranked) < len(_ORDINALS):
        raise CaptionClaimError(f"{prefix}: fewer than three ranked encoders to name a top three.")

    f0 = ranked["croma_f0"]
    lo, hi = float(f0.min()), float(f0.max())
    specs = [
        ("ParetoFrontierModels", ", ".join(frontier)),
        ("ParetoFrontierNModels", str(len(frontier))),
        ("NCromaNonPositive", str(int((ranked["croma"] <= 0.0).sum()))),
        ("NLtmNegative", str(int((ranked["croma_ltm_alpha"] < 0.0).sum()))),
        ("NFZeroPositive", str(int((f0 > 0.0).sum()))),
        ("FZeroMin", num_math(lo, 3)),
        ("FZeroMinModels", _names(ranked[(f0 - lo).abs() < 1e-12])),
        ("FZeroMax", num_math(hi, 3)),
        ("FZeroMaxModels", _names(ranked[(f0 - hi).abs() < 1e-12])),
    ]
    top = ranked.sort_values(["croma", "model"], ascending=[False, True]).head(len(_ORDINALS))
    for ordinal, (_, row) in zip(_ORDINALS, top.iterrows()):
        specs.append((f"Top{ordinal}Model", str(row["model"])))
        specs.append((f"Top{ordinal}Croma", num_math(float(row["croma"]))))
    return [_macro(prefix, suffix, body) for suffix, body in specs]


def operating_point_macros(prefix: str, df: pd.DataFrame) -> list[str]:
    """The run's shared ``k`` and where the natural-image control sits against the panel."""
    ks = sorted(int(k) for k in df["k"].unique())
    if len(ks) != 1:
        raise CaptionClaimError(f"{prefix}: the run is not at one shared k across encoders: {ks}")
    control = df[df["model"] == CONTROL_MODEL]
    if len(control) != 1:
        raise CaptionClaimError(f"{prefix}: the run does not carry the control {CONTROL_MODEL}.")
    control_croma = float(croma_as_margin(control["croma"]).iloc[0])
    ranked_croma = croma_as_margin(_ranked(df)["croma"]).astype(float)
    specs = [
        ("OperatingK", str(ks[0])),
        ("ControlCroma", num_math(control_croma)),
        ("ControlBioBacc", num_math(float(control["bio_knn_bacc"].iloc[0]), 3)),
        ("NBelowControl", str(int((ranked_croma < control_croma).sum()))),
    ]
    return [_macro(prefix, suffix, body) for suffix, body in specs]


def _count(n: int) -> str:
    """A count for text or math mode: ``5{,}100``."""
    return f"{int(n):,}".replace(",", "{,}")


def cardinality_macros(manifests: dict[str, pd.DataFrame]) -> list[str]:
    """Composition and per-cell counts of each cohort, from its view's evaluation rows.

    These are the numbers the dataset-cardinality captions quote, so every cell of a cohort
    must hold the same count (``_cohort_counts.balance``); an unbalanced cohort fails the
    build instead of printing a per-cell count no cell has. A paired cohort's cell is a
    quartet cell; its tiles recur across quartets, so it also states its occurrences and
    the range of distinct slides a class x centre pools over its quartets.
    """
    lines = []
    for prefix, manifest in manifests.items():
        paired = "subset" in manifest
        group = CARDINALITY_GROUP.get(prefix, "Slides")
        try:
            balanced = balance(manifest, QUARTET_CELL if paired else CELL)
        except ValueError as error:
            raise CaptionClaimError(f"{prefix}: {error}") from error
        composition = [
            ("NClasses", manifest["label"].nunique()),
            ("NCentres", manifest["confounder"].nunique()),
        ]
        specs = [(f"Cell{group}", balanced.cell.slides), (f"N{group}", balanced.slides)]
        if prefix not in SLIDE_LEVEL_PREFIXES:
            specs += [
                ("CellTiles", balanced.cell.tiles),
                (f"TilesPer{group[:-1]}", balanced.tiles_per_slide),
                ("NTiles", manifest["sample_id"].nunique()),
            ]
        if paired:
            pooled = [cell.slides for cell in cells(manifest).values()]
            composition.append(("NQuartets", manifest["subset"].nunique()))
            specs += [
                ("NOccurrences", balanced.tiles),
                (f"PooledCell{group}Min", min(pooled)),
                (f"PooledCell{group}Max", max(pooled)),
            ]
        lines += [_macro(prefix, suffix, _count(n)) for suffix, n in composition + specs]
    return lines


def _rho(left: pd.Series, right: pd.Series) -> float:
    from scipy.stats import spearmanr

    shared = sorted((set(left.index) & set(right.index)) - {CONTROL_MODEL})
    if len(shared) < 3:
        raise CaptionClaimError(f"too few shared ranked encoders for a rank correlation: {shared}")
    return float(spearmanr(left[shared], right[shared]).statistic)


def tcga2x2_agreement_macros(croma: dict[str, pd.Series]) -> list[str]:
    """Spearman rho of TCGA-2x2's median CRoMa with each other tile benchmark's."""
    missing = [p for p in (TCGA2X2, TCGA2X2_TWIN, *TCGA2X2_OTHERS) if p not in croma]
    if missing:
        raise CaptionClaimError(f"TCGA-2x2 rank agreement is missing benchmarks: {missing}")
    others = [_rho(croma[TCGA2X2], croma[p]) for p in TCGA2X2_OTHERS]
    return [
        _macro(TCGA2X2, f"Rho{TCGA2X2_TWIN}", num_math(_rho(croma[TCGA2X2], croma[TCGA2X2_TWIN]))),
        _macro(TCGA2X2, "RhoOtherRange", rf"$[{bare_num(min(others))}, {bare_num(max(others))}]$"),
    ]


def apd_nipd_macros(joined: pd.DataFrame, datasets: tuple[str, ...] | None = None) -> list[str]:
    """Per benchmark and regime, how closely APD and nIPD rank the encoders (Spearman rho)."""
    from loaders import DATASETS
    from scipy.stats import spearmanr

    if datasets is None:
        from _apd import FIGURE_DATASETS

        datasets = tuple(FIGURE_DATASETS)
    lines = []
    for dataset in datasets:
        prefix = by_benchmark(DATASETS[dataset]["benchmark"]).prefix
        rows = _ranked(joined[joined["dataset"] == dataset])
        if len(rows) < 3:
            raise CaptionClaimError(f"{dataset}: too few ranked encoders in the APD join.")
        for regime, suffix in APD_REGIMES:
            rho = float(spearmanr(rows[f"apd_{regime}"], rows[f"nipd_{regime}"]).statistic)
            lines.append(_macro(prefix, f"ApdNipdRho{suffix}", num_math(rho)))
    return lines


def _metrics(root: Path, prefix: str) -> pd.DataFrame:
    path = Path(root) / by_prefix(prefix).metrics_rel
    if not path.is_file():
        raise FileNotFoundError(f"{prefix}: missing {path}; re-run the benchmark.")
    return published_models(pd.read_csv(path))


def _view_manifest(prefix: str) -> pd.DataFrame:
    import views

    return views.load_view(by_prefix(prefix).benchmark).eval_manifest


def panel_macros(root: Path) -> list[str]:
    """Every macro above, loaded from the runs under ``root``; any missing input raises."""
    frames = {prefix: _metrics(root, prefix) for prefix in OPERATING_PREFIXES}
    joined_path = Path(root) / APD_JOINED_CSV
    if not joined_path.is_file():
        raise FileNotFoundError(
            f"missing {joined_path}; run scripts/studies/apd/apd_croma_correlation.py."
        )

    lines = ["% Ranking, frontier and operating-point macros over the ranked tile panel"]
    for prefix in FRONTIER_PREFIXES:
        lines += ranking_macros(prefix, frames[prefix])
    for prefix in OPERATING_PREFIXES:
        lines += operating_point_macros(prefix, frames[prefix])
    lines.append("% Cohort composition and per-cell counts (each benchmark's view)")
    lines += cardinality_macros({p: _view_manifest(p) for p in CARDINALITY_PREFIXES})
    from _acquisition import acquisition_macros

    lines += acquisition_macros()
    lines.append("% TCGA-2x2 rank agreement with the other cohorts")
    croma = {
        prefix: pd.Series(croma_as_margin(df["croma"]).to_numpy(float), index=df["model"])
        for prefix, df in frames.items()
    }
    lines += tcga2x2_agreement_macros(croma)
    lines.append(f"% APD vs nIPD ranking agreement: {APD_JOINED_CSV}")
    lines += apd_nipd_macros(published_models(pd.read_csv(joined_path)))
    return lines
