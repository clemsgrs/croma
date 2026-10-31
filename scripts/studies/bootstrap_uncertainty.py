"""Per-encoder uncertainty of the headline CRoMa statistics, ranks and frontiers.

Every benchmark's per-sample CRoMa at the headline ``m`` is read from the run's
``per_sample_metrics.csv``. Margins are taken as given: no neighbour is recomputed, so the
interval describes which slides happened to be sampled, not the encoder's geometry. Each
replicate is one stratified cluster draw (``croma.metrics.bootstrap.cluster_bootstrap_draws``):
slides (the manifest's ``group_id``) are the clusters, resampled within class x centre strata
so every replicate keeps the cohort's design. Where a case spans several classes (Tolkach-ESCA)
the strata are the centres alone. One draw is shared by every encoder, so replicate ``b`` is
comparable across encoders.

Per benchmark the study writes, under the run's ``studies/`` directory:

``bootstrap_uncertainty.csv``
    One row per ranked encoder (the natural-image control is a floor, not a competitor, so it
    is excluded): median, ``F(0)`` and ``LTM_10`` with 95% percentile CIs; median point rank,
    mean rank and 95% rank interval; ``p_frontier``, the share of replicates the encoder is on
    the median-versus-``LTM_10`` Pareto frontier.
``bootstrap_pairwise.csv``
    ``P(A > B)`` on the median for every ordered pair of encoders.
``bootstrap_uncertainty.json``
    The design (strata, cluster and sample counts, ``n_boot``, seed), the adjacent pairs along
    the point ordering with their win probabilities, and how many of them are separated
    (``P >= 0.975``).
``bootstrap_replicates.npz``
    The replicate medians, ``F(0)`` and ``LTM_10`` (``n_boot x encoders``), for studies that
    combine them (the downstream-correlation two-level bootstrap).

On TCGA-2x2 the published median is balanced: the median of the per-subset medians, every
pair of cancer types weighted equally. The study bootstraps that statistic, not the pooled
median. ``F(0)`` and ``LTM_10`` pool every occurrence, as the published columns do. Every
point value is checked against the run's ``metrics.csv`` before anything is written.

Across the three tile benchmarks of the cross-benchmark panel the study also writes, to
``output/studies/bootstrap-uncertainty/``, the membership probability of the mean-rank
frontier (replicate ``b`` of every benchmark combined) and Spearman's rho of median CRoMa for
every pair of tile benchmarks with an encoder-bootstrap CI on one shared encoder draw.

Usage:
  python scripts/studies/bootstrap_uncertainty.py [--n-boot 2000] [--prefix Camelyon ...]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from croma.metrics.bootstrap import (  # noqa: E402
    _percentile_ci,
    _rank_summary,
    cluster_bootstrap_draws,
    encoder_bootstrap_draws,
    shared_draw_tail_statistics,
    spearman_inference,
)
from plotting.base import _pareto_frontier_max_max  # noqa: E402

SEED = 12345
N_BOOT = 2000
LEVEL = 0.95
ALPHA = 0.10
#: ``P(A > B)`` at or above this separates an adjacent pair: the one-sided 97.5% bound
#: matching the two-sided 95% intervals reported beside it.
SEPARATION = 0.975

CLASS_BY_CENTRE = "class x centre"
CENTRE = "centre"


def stratum_labels(
    labels: np.ndarray, confounders: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, str]:
    """Strata for the slide draw, and which design they are.

    Class x centre when every slide sits in one cell; the centre alone when a cluster spans
    classes (a Tolkach-ESCA case holds several tissue classes). A cluster spanning centres
    has no stratum, which is an error.
    """
    frame = pd.DataFrame(
        {"label": np.asarray(labels).astype(str), "centre": np.asarray(confounders).astype(str)}
    )
    frame["cluster"] = np.asarray(clusters)
    spans = frame.groupby("cluster")[["label", "centre"]].nunique()
    if (spans["centre"] > 1).any():
        raise ValueError("a cluster spans several centres, so no stratification contains it")
    if (spans["label"] == 1).all():
        return (frame["label"] + "|" + frame["centre"]).to_numpy(), CLASS_BY_CENTRE
    return frame["centre"].to_numpy(), CENTRE


def frontier_probability(
    models: list[str], medians: np.ndarray, tails: np.ndarray
) -> dict[str, float]:
    """Share of replicates each encoder is on the median-versus-tail Pareto frontier.

    ``medians`` and ``tails`` are ``(n_boot, len(models))``; row ``b`` is one replicate.
    Larger is better on both axes (a milder tail is a larger ``LTM``), through the same
    ``_pareto_frontier_max_max`` the Pareto figures ring.
    """
    hits = dict.fromkeys(models, 0)
    for med_row, tail_row in zip(np.asarray(medians), np.asarray(tails)):
        points = [(m, float(x), float(y)) for m, x, y in zip(models, med_row, tail_row)]
        for name in _pareto_frontier_max_max(points):
            hits[name] += 1
    n = len(medians)
    return {m: hits[m] / n for m in models}


def separated_adjacent_pairs(
    order: list[str], pairwise_win: dict[tuple[str, str], float], threshold: float = SEPARATION
) -> int:
    """Adjacent pairs along ``order`` whose higher encoder wins at least ``threshold``."""
    return sum(pairwise_win[(a, b)] >= threshold for a, b in zip(order, order[1:]))


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    """Row-wise ranks, 1 = largest; ties keep column order (pandas' ``method="first"``)."""
    return rankdata(-np.asarray(values, dtype=float), method="ordinal", axis=-1)


def mean_rank_frontier_probability(
    models: list[str], benchmarks: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    """Share of replicates each encoder is on the cross-benchmark mean-rank frontier.

    ``benchmarks`` maps each benchmark to its ``(medians, tails)`` replicates, each
    ``(n_boot, len(models))`` in ``models`` order. Replicate ``b`` combines replicate ``b``
    of every benchmark (the cohorts are independent, so any pairing is a valid joint draw):
    every encoder is ranked within each benchmark by median and by tail, the ranks are
    averaged over benchmarks, and the frontier is taken on the two mean ranks, fewer being
    better, as ``_rank_pareto.RankPareto.frontier`` takes it on the point values.
    """
    pairs = list(benchmarks.values())
    median_rank = np.mean([_descending_ranks(med) for med, _ in pairs], axis=0)
    tail_rank = np.mean([_descending_ranks(tail) for _, tail in pairs], axis=0)
    return frontier_probability(models, -median_rank, -tail_rank)


def cross_benchmark_rho(
    medians: pd.DataFrame,
    *,
    headline: list[tuple[str, str]],
    n_boot: int = N_BOOT,
    seed: int = SEED,
    level: float = LEVEL,
) -> tuple[pd.DataFrame, dict]:
    """Spearman's rho of median CRoMa for every pair of benchmarks (columns of ``medians``).

    Every pair is resampled on one shared encoder draw, so the replicate mean over the
    ``headline`` pairs has a percentile CI as well. Rows are encoders: pass the ranked panel.
    """
    draws = encoder_bootstrap_draws(len(medians), n_boot=n_boot, seed=seed)
    rows, replicates = [], {}
    for a, b in itertools.combinations(medians.columns, 2):
        inference = spearman_inference(
            medians[a].to_numpy(float), medians[b].to_numpy(float), level=level, draws=draws
        )
        replicates[(a, b)] = inference.replicates
        rows.append(
            {
                "a": a,
                "b": b,
                "n": inference.n,
                "rho": inference.rho,
                "ci_lo": inference.ci.lo,
                "ci_hi": inference.ci.hi,
                "p_value": inference.p_value,
                "headline": (a, b) in headline or (b, a) in headline,
            }
        )
    table = pd.DataFrame(rows)
    chosen = table[table["headline"]]
    pair_keys = list(zip(chosen["a"], chosen["b"]))
    mean_ci = _percentile_ci(
        np.mean([replicates[key] for key in pair_keys], axis=0),
        float(chosen["rho"].mean()),
        level,
        int(n_boot),
    )
    mean = {
        "rho": mean_ci.point,
        "ci_lo": mean_ci.lo,
        "ci_hi": mean_ci.hi,
        "n_pairs": len(pair_keys),
        "n_models": len(medians),
    }
    return table, mean


def _balanced_median(values: np.ndarray, codes: np.ndarray, n_subsets: int) -> float:
    """Median of the per-subset medians of the finite values (each subset weighted once)."""
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(1, n_subsets))
    medians = []
    for chunk in np.split(values[order], bounds):
        chunk = chunk[np.isfinite(chunk)]
        if chunk.size:
            medians.append(np.median(chunk))
    return float(np.median(medians)) if medians else float("nan")


@dataclass(frozen=True, eq=False)
class EncoderUncertainty:
    """One benchmark's bootstrap, ready to write."""

    models: list[str]
    #: one row per encoder in point-rank order; see the module docstring for the columns.
    table: pd.DataFrame
    #: ``median`` / ``f0`` / ``ltm`` replicate values, ``(n_boot, len(models))``.
    replicates: dict[str, np.ndarray]
    #: replicate median ranks, ``(n_boot, len(models))``, rank 1 the highest.
    rank_replicates: np.ndarray
    pairwise_win: dict[tuple[str, str], float]
    #: adjacent pairs along the point ordering: higher, lower, ``P(higher > lower)``.
    adjacent: pd.DataFrame
    n_separated: int
    n_boot: int
    level: float
    alpha: float


def encoder_uncertainty(
    values: dict[str, np.ndarray],
    clusters: np.ndarray,
    *,
    strata: np.ndarray | None,
    subsets: np.ndarray | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    level: float = LEVEL,
    alpha: float = ALPHA,
) -> EncoderUncertainty:
    """Bootstrap every encoder's median, ``F(0)`` and ``LTM_alpha`` on shared slide draws.

    ``values`` maps each ranked encoder to its per-sample margins, aligned to ``clusters``,
    ``strata`` and ``subsets``. With more than one subset the median is balanced over them
    (each replicate re-takes every subset's median of its drawn rows); the tail statistics
    always pool.
    """
    models = list(values)
    tail = shared_draw_tail_statistics(
        values, clusters, strata=strata, alpha=alpha, n_boot=n_boot, level=level, seed=seed
    )
    medians = tail.replicates["median"]
    median_point = {m: tail.ci["median"][m].point for m in models}
    stability, rank_replicates = tail.median_rank, tail.rank_replicates
    if subsets is not None and len(np.unique(subsets)) > 1:
        codes, uniques = pd.factorize(np.asarray(subsets))
        arrs = {m: np.asarray(v, dtype=float) for m, v in values.items()}
        median_point = {m: _balanced_median(arrs[m], codes, len(uniques)) for m in models}
        medians = np.empty((int(n_boot), len(models)), dtype=float)
        draws = cluster_bootstrap_draws(clusters, strata=strata, n_boot=n_boot, seed=seed)
        for b, rows in enumerate(draws):
            for j, m in enumerate(models):
                medians[b, j] = _balanced_median(arrs[m][rows], codes[rows], len(uniques))
        stability, rank_replicates = _rank_summary(
            models, median_point, medians, level=level, n_boot=int(n_boot)
        )
    p_frontier = frontier_probability(models, medians, tail.replicates["ltm"])
    point_frontier = set(
        _pareto_frontier_max_max(
            [(m, median_point[m], tail.ci["ltm"][m].point) for m in models]
        )
    )

    rows = []
    for m in models:
        ci = stability.value_ci[m]
        row = {"model": m, "median": ci.point, "median_lo": ci.lo, "median_hi": ci.hi}
        for stat in ("f0", "ltm"):
            sci = tail.ci[stat][m]
            row.update({stat: sci.point, f"{stat}_lo": sci.lo, f"{stat}_hi": sci.hi})
        row.update(
            {
                "point_rank": stability.point_rank[m],
                "mean_rank": stability.mean_rank[m],
                "rank_lo": stability.rank_lo[m],
                "rank_hi": stability.rank_hi[m],
                "on_frontier": m in point_frontier,
                "p_frontier": p_frontier[m],
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("point_rank", kind="stable").reset_index(drop=True)
    order = table["model"].tolist()
    adjacent = pd.DataFrame(
        [
            {"higher": a, "lower": b, "p_higher_beats_lower": stability.pairwise_win[(a, b)]}
            for a, b in zip(order, order[1:])
        ],
        columns=["higher", "lower", "p_higher_beats_lower"],
    )
    return EncoderUncertainty(
        models=models,
        table=table,
        replicates={"median": medians, "f0": tail.replicates["f0"], "ltm": tail.replicates["ltm"]},
        rank_replicates=rank_replicates,
        pairwise_win=stability.pairwise_win,
        adjacent=adjacent,
        n_separated=separated_adjacent_pairs(order, stability.pairwise_win),
        n_boot=int(n_boot),
        level=float(level),
        alpha=float(alpha),
    )


# --- running the study over the paper's runs --------------------------------------------

#: The per-encoder tables, in the paper's order: four tile benchmarks, then PCaBiop.
TILE_PREFIXES = ("Camelyon", "TcgaTwoByTwo", "TcgaFourByFour", "Tolkach")
SLIDE_PREFIXES = ("Panda",)
PREFIXES = TILE_PREFIXES + SLIDE_PREFIXES
#: The cross-benchmark panel (``_cross_benchmark.PANEL``): its mean-rank frontier, and the
#: pairs whose mean is ``\CrossCohortRhoMean``. TCGA-2x2 shares TCGA-4x4's cohort, so it is a
#: pair of its own but not part of the cross-cohort summary.
FRONTIER_PANEL = ("Camelyon", "TcgaFourByFour", "Tolkach")
HEADLINE_PAIRS = list(itertools.combinations(FRONTIER_PANEL, 2))
CROSS_DIR_REL = "output/studies/bootstrap-uncertainty"
#: Point values must match the published columns to within float noise.
POINT_ATOL = 1e-12


class ReproductionError(AssertionError):
    """The study's point values are not the run's published statistics."""


def _manifest():
    repro = REPO / "scripts" / "repro"
    if str(repro) not in sys.path:
        sys.path.insert(0, str(repro))
    from paper_manifest import by_prefix
    from plotting.style import CONTROL_MODEL

    return by_prefix, CONTROL_MODEL


@dataclass(frozen=True, eq=False)
class Cohort:
    """One run's per-sample margins at the headline ``m``, aligned across ranked encoders."""

    prefix: str
    values: dict[str, np.ndarray]
    clusters: np.ndarray
    strata: np.ndarray
    design: str
    subsets: np.ndarray
    published: pd.DataFrame


def load_cohort(prefix: str, root: Path = REPO) -> Cohort:
    """Read a run's per-sample CRoMa(m) for every ranked encoder, in one shared row order."""
    from croma.metrics.croma import CROMA_HEADLINE_M

    by_prefix, control = _manifest()
    entry = by_prefix(prefix)
    column = f"croma_m{int(CROMA_HEADLINE_M)}"
    cols = ["model", "occurrence_index", "sample_index", "subset", "group_id"]
    cols += ["label", "confounder", column]
    per_sample = pd.read_csv(
        root / entry.per_sample_rel, usecols=cols, float_precision="round_trip"
    )
    published = pd.read_csv(root / entry.metrics_rel, float_precision="round_trip")
    models = [m for m in published["model"] if m != control]
    reference = None
    values: dict[str, np.ndarray] = {}
    for model, rows in per_sample.groupby("model", sort=False):
        if model not in models:
            continue
        rows = rows.sort_values(["occurrence_index", "sample_index"], kind="stable")
        design = rows[["group_id", "subset", "label", "confounder"]].astype(str)
        design = design.reset_index(drop=True)
        if reference is None:
            reference = design
        elif not reference.equals(design):
            raise ReproductionError(f"{prefix}: {model}'s rows do not align with the others")
        values[model] = rows[column].to_numpy(dtype=float)
    missing = sorted(set(models) - set(values))
    if missing or reference is None:
        raise ReproductionError(f"{prefix}: no per-sample rows for {missing}")
    clusters = reference["group_id"].to_numpy()
    strata, kind = stratum_labels(reference["label"], reference["confounder"], clusters)
    return Cohort(
        prefix=prefix,
        values={m: values[m] for m in models},
        clusters=clusters,
        strata=strata,
        design=kind,
        subsets=reference["subset"].to_numpy(),
        published=published.set_index("model").loc[models],
    )


def check_points(result: EncoderUncertainty, cohort: Cohort) -> None:
    """Fail unless every point value is the run's published median, ``F(0)`` and LTM."""
    table = result.table.set_index("model")
    for ours, theirs in (("median", "croma"), ("f0", "croma_f0"), ("ltm", "croma_ltm_alpha")):
        gap = (table[ours] - cohort.published.loc[table.index, theirs]).abs()
        if not (gap <= POINT_ATOL).all():
            worst = gap.idxmax()
            raise ReproductionError(
                f"{cohort.prefix}: {worst}'s {ours} is {table.loc[worst, ours]!r}, but the run "
                f"published {theirs} = {cohort.published.loc[worst, theirs]!r}"
            )


def run_benchmark(
    prefix: str,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    root: Path = REPO,
    out_root: Path | None = None,
) -> EncoderUncertainty:
    """Bootstrap one run (read under ``root``) and write its four outputs beside it (under
    ``out_root``, by default ``root``)."""
    by_prefix, _ = _manifest()
    cohort = load_cohort(prefix, root)
    result = encoder_uncertainty(
        cohort.values,
        cohort.clusters,
        strata=cohort.strata,
        subsets=cohort.subsets,
        n_boot=n_boot,
        seed=seed,
    )
    check_points(result, cohort)
    out = (out_root or root) / by_prefix(prefix).studies_rel
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "bootstrap_uncertainty.csv", index=False)
    pd.DataFrame(
        [{"a": a, "b": b, "p_a_beats_b": p} for (a, b), p in result.pairwise_win.items()]
    ).to_csv(out / "bootstrap_pairwise.csv", index=False)
    np.savez(
        out / "bootstrap_replicates.npz",
        models=np.array(result.models),
        rank=result.rank_replicates,
        **result.replicates,
    )
    n_subsets = int(len(np.unique(cohort.subsets)))
    summary = {
        "benchmark": by_prefix(prefix).benchmark,
        "prefix": prefix,
        "n_boot": result.n_boot,
        "seed": int(seed),
        "level": result.level,
        "alpha": result.alpha,
        "strata": cohort.design,
        "n_strata": int(len(np.unique(cohort.strata))),
        "n_clusters": int(len(np.unique(cohort.clusters))),
        "n_samples": int(len(cohort.clusters)),
        "n_subsets": n_subsets,
        "median": "balanced over subsets" if n_subsets > 1 else "pooled",
        "n_models": len(result.models),
        "separation_threshold": SEPARATION,
        "n_separated_adjacent": int(result.n_separated),
        "n_adjacent": int(len(result.adjacent)),
        "adjacent": result.adjacent.to_dict(orient="records"),
    }
    (out / "bootstrap_uncertainty.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[uncertainty] {prefix}: {len(result.models)} encoders, {summary['n_clusters']} "
        f"clusters in {summary['n_strata']} {cohort.design} strata; "
        f"{result.n_separated}/{len(result.adjacent)} adjacent pairs separated -> {out}",
        flush=True,
    )
    return result


def run_cross_benchmark(
    results: dict[str, EncoderUncertainty],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    out_root: Path = REPO,
) -> None:
    """The mean-rank frontier probabilities and the cross-benchmark rank agreement."""
    by_prefix, _ = _manifest()
    from _cross_benchmark import PANEL

    if tuple(by_prefix(p).benchmark for p in FRONTIER_PANEL) != tuple(PANEL):
        raise ReproductionError(f"FRONTIER_PANEL {FRONTIER_PANEL} is not _cross_benchmark.PANEL")
    models = results[FRONTIER_PANEL[0]].models
    if any(results[p].models != models for p in TILE_PREFIXES):
        raise ReproductionError("the tile benchmarks do not share one ranked roster")
    out = out_root / CROSS_DIR_REL
    out.mkdir(parents=True, exist_ok=True)
    p_frontier = mean_rank_frontier_probability(
        models,
        {
            p: (results[p].replicates["median"], results[p].replicates["ltm"])
            for p in FRONTIER_PANEL
        },
    )
    pd.DataFrame({"model": models, "p_frontier": [p_frontier[m] for m in models]}).to_csv(
        out / "mean_rank_frontier.csv", index=False
    )
    medians = pd.DataFrame(
        {p: results[p].table.set_index("model").loc[models, "median"] for p in TILE_PREFIXES}
    )
    rho, mean = cross_benchmark_rho(medians, headline=HEADLINE_PAIRS, n_boot=n_boot, seed=seed)
    rho.to_csv(out / "cross_benchmark_rho.csv", index=False)
    meta = {**mean, "n_boot": int(n_boot), "seed": int(seed), "pairs": HEADLINE_PAIRS}
    (out / "cross_benchmark_rho.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[uncertainty] cross-benchmark: mean rho {mean['rho']:.3f} "
        f"[{mean['ci_lo']:.3f}, {mean['ci_hi']:.3f}] -> {out}",
        flush=True,
    )


def run(
    prefixes=PREFIXES,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    root: Path = REPO,
    out_root: Path | None = None,
) -> None:
    out_root = out_root or root
    results = {
        p: run_benchmark(p, n_boot=n_boot, seed=seed, root=root, out_root=out_root)
        for p in prefixes
    }
    if set(TILE_PREFIXES) <= set(results):
        run_cross_benchmark(results, n_boot=n_boot, seed=seed, out_root=out_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prefix", action="append", choices=PREFIXES)
    parser.add_argument("--root", type=Path, default=REPO, help="where output/ is read")
    parser.add_argument("--out-root", type=Path, help="where outputs are written (default: --root)")
    args = parser.parse_args()
    run(
        tuple(args.prefix or PREFIXES),
        n_boot=args.n_boot,
        seed=args.seed,
        root=args.root,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
