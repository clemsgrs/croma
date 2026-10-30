"""Does grade heterogeneity explain the negative cancer-anchor margins on PCaBiop?

A negative CRoMa says an anchor's nearest different-label, same-centre (OS) neighbours sit
closer than its nearest same-label, other-centre (SO) ones. PCaBiop's label is coarse --
cancer versus benign -- so a cancer anchor's SO neighbours may be far because they are a
*different ISUP grade*, not because the centres differ technically. This study tests that.

The cancer slides are balanced, 50 per ISUP grade 1-5 per provider. For the cancer anchors
of one provider, ``CRoMa.compute`` runs on a sub-cohort that keeps every slide of that
provider and the other provider's benign slides, but only a chosen *pool* of the other
provider's cancer slides -- so only the SO side changes. Three pools, both directions:

* **full** -- all 250 cancer slides at the other provider: the run itself, asserted to
  reproduce the run's per-sample ``croma_m5`` bit for bit;
* **grade-matched** -- the 50 of the anchor's own grade;
* **random** -- 50 drawn uniformly from the 250, in 200 seeded replicates, one pool per
  (provider, grade) group of anchors per replicate.

If grade heterogeneity drives the negative margins, the matched pool should raise them
above the size-matched random control.

The bitwise full-pool check depends on the BLAS: the run's float32 distances came from
OpenBLAS's SkylakeX sgemm kernel, and TITAN's float16 features (promoted to float64) from
its dgemm on four threads. A different kernel or thread count moves the last bits and the
check fails, so pin both:

Run: OPENBLAS_CORETYPE=SKYLAKEX OPENBLAS_NUM_THREADS=4 \\
     python scripts/studies/grade_matched_pools.py

The full pool runs in the main process; matched and random pools run in ``--workers``
spawned processes with one BLAS thread each, so the study never uses more than four cores.
Outputs go to ``output/studies/grade-matched/`` (``meta.json`` names the run read).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from croma import CRoMa  # noqa: E402

CANCER_LABEL = "1"
GRADES = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Search:
    """How CRoMa is computed: the run's own typed-neighbour search, read at one ``m``.

    ``m_values`` is what the search is asked for (the run asks for ``1..m_max``, so its
    search radius is set by ``m_max``); ``headline_m`` is the ``m`` the study reads.
    """

    m_values: tuple[int, ...]
    headline_m: int
    start_k: int
    k_growth_factor: float
    alpha: float


def cancer_grades(manifest: pd.DataFrame, isup: pd.DataFrame, *, per_grade: int) -> pd.Series:
    """The ISUP grade of every cancer slide in ``manifest``, joined on ``sample_id``.

    ``isup`` is the grading view of the same slides (its ``label`` is the ISUP grade). The
    result is indexed like the cancer rows of ``manifest``.
    """
    grade_of = isup.drop_duplicates("sample_id").set_index("sample_id")["label"]
    cancer = manifest[manifest["label"] == CANCER_LABEL]
    grades = cancer["sample_id"].map(grade_of)
    ungraded = cancer.loc[grades.isna(), "sample_id"].tolist()
    if ungraded:
        raise ValueError(f"{len(ungraded)} cancer slide(s) have no ISUP grade: {ungraded[:5]}")
    grades = grades.astype(int).rename("grade")
    off_scale = cancer.loc[~grades.isin(GRADES), "sample_id"].tolist()
    if off_scale:
        raise ValueError(
            f"{len(off_scale)} cancer slide(s) carry an ISUP grade outside {GRADES}: "
            f"{off_scale[:5]}"
        )
    counts = pd.crosstab(cancer["confounder"], grades).reindex(columns=list(GRADES), fill_value=0)
    if not (counts == per_grade).all().all():
        raise ValueError(
            f"the cancer slides are not {per_grade} per ISUP grade per provider:\n{counts}"
        )
    return grades


def sub_cohort(
    manifest: pd.DataFrame, *, anchor_provider: str, pool: Sequence[int]
) -> np.ndarray:
    """Row positions of the cohort that gives ``anchor_provider``'s cancers ``pool`` as SO.

    CRoMa types a cancer anchor's SO candidates as the other provider's cancer slides and
    its OS candidates as its own provider's benign slides. Keeping every row of the anchor
    provider, the other provider's benign slides, and only ``pool`` of the other provider's
    cancer slides therefore restricts the SO set to ``pool`` and leaves the OS set as it
    was. The other provider's benign slides change nothing for the anchors; they are kept
    because the pooled slides need them as their own OS set, and the library scores every
    row it is given. Rows keep their manifest order, so the full pool is the whole cohort.
    """
    at_anchor = (manifest["confounder"] == anchor_provider).to_numpy()
    cancer = (manifest["label"] == CANCER_LABEL).to_numpy()
    pool = np.unique(np.asarray(pool, dtype=int))
    candidates = np.flatnonzero(~at_anchor & cancer)
    stray = np.setdiff1d(pool, candidates)
    if stray.size:
        raise ValueError(
            f"SO pool rows {stray.tolist()[:5]} are not cancer slides at the provider "
            f"opposite {anchor_provider!r}"
        )
    return np.union1d(np.flatnonzero(at_anchor | ~cancer), pool)


@dataclass(frozen=True)
class AnchorMargins:
    """One encoder's per-anchor CRoMa over its cancer anchors, under the three SO pools."""

    #: ISUP grade of each anchor, ``(n_anchors,)``.
    grade: np.ndarray
    #: Every cancer slide at the other provider (the run's own SO pool), ``(n_anchors,)``.
    full: np.ndarray
    #: The other provider's cancer slides of the anchor's grade, ``(n_anchors,)``.
    matched: np.ndarray
    #: Size-matched random pools, one row per replicate, ``(n_replicates, n_anchors)``.
    random: np.ndarray


def _f0(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """F(0) with the closed boundary: an exactly contested anchor is confounder-dominant."""
    return np.mean(np.asarray(values) <= 0.0, axis=axis)


def _measures(full: np.ndarray, matched: np.ndarray, random: np.ndarray) -> dict:
    replicate_medians = np.median(random, axis=1)
    replicate_f0 = _f0(random, axis=1)
    matched_median = float(np.median(matched))
    random_median = float(np.median(replicate_medians))
    random_mean = random.mean(axis=0)
    nonpositive = random_mean <= 0.0
    n_sign_change = int((nonpositive & (matched > 0.0)).sum())
    n_nonpositive = int(nonpositive.sum())
    lo, hi = np.percentile(replicate_medians, [2.5, 97.5])
    f0_lo, f0_hi = np.percentile(replicate_f0, [2.5, 97.5])
    return {
        "n_anchors": int(len(matched)),
        "full_median": float(np.median(full)),
        "full_f0": float(_f0(full)),
        "matched_median": matched_median,
        "matched_f0": float(_f0(matched)),
        "random_median": random_median,
        "random_median_lo": float(lo),
        "random_median_hi": float(hi),
        "random_f0": float(np.median(replicate_f0)),
        "random_f0_lo": float(f0_lo),
        "random_f0_hi": float(f0_hi),
        "delta_grade": matched_median - random_median,
        "p_one_sided": float(np.mean(replicate_medians >= matched_median)),
        "paired_shift_median": float(np.median(matched - random_mean)),
        "n_random_nonpositive": n_nonpositive,
        "n_sign_change": n_sign_change,
        "sign_change_share": n_sign_change / n_nonpositive if n_nonpositive else float("nan"),
    }


def summarize(margins: AnchorMargins) -> pd.DataFrame:
    """The study's measures over all anchors (``grade == "all"``) and per ISUP grade.

    * pool summaries: median CRoMa and F(0) under the full, matched and random pools; the
      random pool reports the median of its replicate medians (and F(0)s) with their
      2.5--97.5 percentile range;
    * ``delta_grade``: the matched median minus the median of the replicate medians;
    * ``p_one_sided``: the share of replicates whose median is at least the matched median
      (grade matching should *raise* the margin if grade heterogeneity matters);
    * the per-anchor paired shift, matched minus the replicate-mean random margin, and the
      share of anchors at or below zero under the (replicate-mean) random pool that the
      matched pool lifts above zero.
    """
    scopes = [("all", np.ones(len(margins.grade), dtype=bool))]
    scopes += [(str(g), margins.grade == g) for g in np.unique(margins.grade)]
    return pd.DataFrame(
        [
            {
                "grade": scope,
                **_measures(margins.full[mask], margins.matched[mask], margins.random[:, mask]),
            }
            for scope, mask in scopes
        ]
    )


def require_full_pool_reproduces_run(model: str, *, run: np.ndarray, full_pool: np.ndarray) -> None:
    """Fail unless the full pool gives every cancer anchor exactly the run's CRoMa.

    The full pool's sub-cohort is the whole cohort, so this is the run's own computation
    repeated through the study's path; any difference means the two are not the same
    computation. It is bitwise because the cosine distances are float32 for most encoders:
    a different BLAS kernel moves them by ~1e-7, and with Prov-GigaPath's ~1e-3 distances
    that moves a margin by ~1e-5 -- enough to hide a wrong pool behind a tolerance.
    """
    run = np.asarray(run, dtype=float)
    full_pool = np.asarray(full_pool, dtype=float)
    differ = run != full_pool
    if differ.any():
        raise AssertionError(
            f"{model}: the full SO pool reproduces the run's CRoMa on only "
            f"{int((~differ).sum())}/{len(run)} cancer anchors -- {int(differ.sum())}/{len(run)} "
            f"differ, by up to {np.abs(run - full_pool).max():.3g}. If every sub-cohort row "
            "is right, the BLAS differs from the one the run was computed with; see the "
            "module docstring for the environment that reproduces it."
        )


def draw_random_pools(
    candidates: dict[str, np.ndarray],
    groups: Sequence[tuple[str, int]],
    *,
    size: int,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """Seeded size-matched random SO pools: ``pools[replicate, group]`` holds ``size`` rows.

    One pool per (provider, grade) group of anchors per replicate, so anchors that share a
    grade-matched pool also share a random one. Each pool is drawn uniformly, without
    replacement, from all of ``candidates[provider]`` -- every cancer slide at the other
    provider, *not* stratified by grade. That is the control: it shrinks the SO pool to the
    matched pool's size without matching its grade.
    """
    rng = np.random.default_rng(seed)
    pools = np.empty((n_replicates, len(groups), size), dtype=int)
    for r in range(n_replicates):
        for g, (provider, _) in enumerate(groups):
            pools[r, g] = rng.choice(candidates[provider], size=size, replace=False)
    return pools


def anchor_croma(
    features: np.ndarray,
    manifest: pd.DataFrame,
    *,
    rows: np.ndarray,
    anchors: Sequence[int],
    search: Search,
) -> np.ndarray:
    """Per-sample CRoMa(``headline_m``) of ``anchors`` within the sub-cohort ``rows``.

    The margin is the library's: ``CRoMa.compute`` on the sub-cohort, read back per anchor.
    ``rows`` and ``anchors`` are row positions in ``manifest``; values come back in the
    order of ``anchors``.
    """
    rows = np.asarray(rows, dtype=int)
    anchors = np.asarray(anchors, dtype=int)
    position = np.searchsorted(rows, anchors)
    position = np.minimum(position, len(rows) - 1)
    missing = anchors[rows[position] != anchors]
    if missing.size:
        raise ValueError(f"anchor rows {missing.tolist()[:5]} are not in the sub-cohort")
    by_m = CRoMa.compute(
        features=np.asarray(features)[rows],
        manifest=manifest.iloc[rows].reset_index(drop=True),
        confounder_column="confounder",
        m=list(search.m_values),
        alpha=search.alpha,
        start_k=search.start_k,
        k_growth_factor=search.k_growth_factor,
    )
    result = by_m[search.headline_m]
    values = np.full(len(rows), np.nan)
    values[np.asarray(result.occurrence_source_indices, dtype=int)] = result.sample_values_aligned
    return values[position]


@dataclass(frozen=True)
class _GroupTask:
    """One (provider, grade) group of anchors and every SO pool it is scored against."""

    features: np.ndarray
    manifest: pd.DataFrame
    anchor_provider: str
    anchors: np.ndarray
    pools: list[np.ndarray]
    search: Search


def _score_group(task: _GroupTask) -> np.ndarray:
    """``(n_pools, n_anchors)`` CRoMa of the group's anchors, one row per pool."""
    return np.stack(
        [
            anchor_croma(
                task.features,
                task.manifest,
                rows=sub_cohort(task.manifest, anchor_provider=task.anchor_provider, pool=pool),
                anchors=task.anchors,
                search=task.search,
            )
            for pool in task.pools
        ]
    )


def encoder_margins(
    features: np.ndarray,
    manifest: pd.DataFrame,
    grades: pd.Series,
    *,
    random_pools: np.ndarray,
    groups: Sequence[tuple[str, int]],
    search: Search,
    map_fn=map,
    pools_per_task: int | None = None,
) -> AnchorMargins:
    """One encoder's CRoMa on every graded cancer anchor under the full, matched and random pools.

    ``grades`` (from :func:`cancer_grades`) names the anchors, by row position, in the order
    the result reports them. ``random_pools[r, g]`` is replicate ``r``'s pool for
    ``groups[g]``. ``map_fn`` runs the matched and random pools and may be a process pool's
    ``map``; ``pools_per_task`` splits each group's pools over several tasks to balance
    the workers (default: one task per group). The full pool always runs here, in the
    calling process and at its BLAS settings, because
    :func:`require_full_pool_reproduces_run` compares it bitwise.
    """
    anchors = grades.index.to_numpy(dtype=int)
    anchor_grade = grades.to_numpy()
    provider = manifest["confounder"].to_numpy()[anchors]
    cancer = (manifest["label"] == CANCER_LABEL).to_numpy()
    candidates = {
        p: np.flatnonzero(cancer & (manifest["confounder"] != p).to_numpy()) for p in set(provider)
    }
    grade_of_row = np.zeros(len(manifest), dtype=int)
    grade_of_row[anchors] = anchor_grade

    full = np.full(len(anchors), np.nan)
    for p in sorted(candidates):
        mine = provider == p
        task = _GroupTask(features, manifest, p, anchors[mine], [candidates[p]], search)
        full[mine] = _score_group(task)[0]

    tasks, owner, masks = [], [], []
    for g, (p, grade) in enumerate(groups):
        mine = (provider == p) & (anchor_grade == grade)
        matched = candidates[p][grade_of_row[candidates[p]] == grade]
        pools = [matched, *random_pools[:, g]]
        step = pools_per_task or len(pools)
        for start in range(0, len(pools), step):
            tasks.append(
                _GroupTask(features, manifest, p, anchors[mine], pools[start : start + step], search)
            )
            owner.append(g)
        masks.append(mine)

    scored_by_group: list[list[np.ndarray]] = [[] for _ in groups]
    for g, scored in zip(owner, map_fn(_score_group, tasks)):
        scored_by_group[g].append(scored)
    matched_values = np.full(len(anchors), np.nan)
    random_values = np.full((random_pools.shape[0], len(anchors)), np.nan)
    for mine, parts in zip(masks, scored_by_group):
        scored = np.vstack(parts)
        matched_values[mine] = scored[0]
        random_values[:, mine] = scored[1:]
    return AnchorMargins(
        grade=grades.to_numpy(), full=full, matched=matched_values, random=random_values
    )


# --------------------------------------------------------------------------------------
# The real run.
# --------------------------------------------------------------------------------------

BENCHMARK = "panda"
#: The cancer slides are balanced 50 per ISUP grade per provider; a grade-matched pool is
#: therefore 50 slides, and the random control draws pools of the same size.
SLIDES_PER_GRADE = 50
N_REPLICATES = 200
SEED = 0
#: Where the study writes; ``meta.json`` records which run it read.
OUT_DIR = REPO / "output" / "studies" / "grade-matched"


def replicate_table(margins: AnchorMargins) -> pd.DataFrame:
    """Median CRoMa and F(0) of every random replicate, over all anchors and per grade."""
    scopes = [("all", np.ones(len(margins.grade), dtype=bool))]
    scopes += [(str(g), margins.grade == g) for g in np.unique(margins.grade)]
    frames = []
    for scope, mask in scopes:
        values = margins.random[:, mask]
        frames.append(
            pd.DataFrame(
                {
                    "grade": scope,
                    "replicate": np.arange(len(values)),
                    "median": np.median(values, axis=1),
                    "f0": _f0(values, axis=1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _one_blas_thread() -> None:
    import threadpoolctl

    global _BLAS_LIMIT
    _BLAS_LIMIT = threadpoolctl.threadpool_limits(1)


def _search_from_run(run_config: dict, headline_m: int) -> Search:
    replay = run_config["replay"]
    return Search(
        m_values=tuple(range(1, int(replay["croma_m_max"]) + 1)),
        headline_m=headline_m,
        start_k=int(replay["croma_start_k"]),
        k_growth_factor=float(replay["croma_k_growth_factor"]),
        alpha=float(replay["croma_alpha"]),
    )


def _model_first(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["model", *[c for c in frame.columns if c != "model"]]]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import hashlib
    import json
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    import benchmarks
    import views
    from croma.metrics.croma import CROMA_HEADLINE_M
    from croma.metrics.pairs import load_manifest
    from paper_manifest import by_benchmark

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pools-per-task", type=int, default=26)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-replicates", type=int, default=N_REPLICATES, help="for smoke runs")
    args = parser.parse_args(argv)

    entry = by_benchmark(BENCHMARK)
    results = REPO / entry.run_rel / "results"
    run_config = json.loads((results / "run_config.json").read_text())
    headline_m = int(run_config["resolved"]["croma_headline_m"])
    if headline_m != CROMA_HEADLINE_M:
        raise ValueError(f"the run reads CRoMa at m={headline_m}, the paper at {CROMA_HEADLINE_M}")
    search = _search_from_run(run_config, headline_m)

    view = views.load_view(BENCHMARK)
    manifest = view.eval_manifest.reset_index(drop=True)
    manifest["label"] = manifest["label"].astype(str)
    isup_spec = benchmarks.get("panda-isup")
    isup = load_manifest(
        str(REPO / isup_spec.manifest), confounder_column=isup_spec.confounder_column
    )
    isup["label"] = isup["label"].astype(str)
    grades = cancer_grades(manifest, isup, per_grade=SLIDES_PER_GRADE)
    anchors = grades.index.to_numpy(dtype=int)
    providers = sorted(manifest["confounder"].unique())
    groups = [(p, g) for p in providers for g in GRADES]

    per_sample_path = REPO / entry.per_sample_rel
    per_sample_sha256 = hashlib.sha256(per_sample_path.read_bytes()).hexdigest()
    per_sample = pd.read_csv(
        per_sample_path,
        float_precision="round_trip",
        dtype={"label": str, "sample_id": str},
    )
    roster = sorted(pd.read_csv(results / "metrics.csv")["model"].unique())
    if not set(roster) <= set(view.models):
        raise ValueError(f"run models {sorted(set(roster) - set(view.models))} are not embedded")

    cancer = (manifest["label"] == CANCER_LABEL).to_numpy()
    candidates = {
        p: np.flatnonzero(cancer & (manifest["confounder"] != p).to_numpy()) for p in providers
    }
    random_pools = draw_random_pools(
        candidates, groups, size=SLIDES_PER_GRADE, n_replicates=args.n_replicates, seed=SEED
    )

    per_anchor, replicates, summaries = [], [], []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"), initializer=_one_blas_thread
    ) as pool:
        for model in roster:
            run = per_sample[per_sample["model"] == model].sort_values("sample_index")
            if not np.array_equal(run["sample_id"].to_numpy(), manifest["sample_id"].to_numpy()):
                raise ValueError(f"{model}: the run's per-sample rows are not in manifest order")
            margins = encoder_margins(
                view.features(model),
                manifest,
                grades,
                random_pools=random_pools,
                groups=groups,
                search=search,
                map_fn=pool.map,
                pools_per_task=args.pools_per_task,
            )
            require_full_pool_reproduces_run(
                model,
                run=run[f"croma_m{headline_m}"].to_numpy()[anchors],
                full_pool=margins.full,
            )
            per_anchor.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "sample_id": manifest["sample_id"].to_numpy()[anchors],
                        "provider": manifest["confounder"].to_numpy()[anchors],
                        "grade": margins.grade,
                        "croma_full": margins.full,
                        "croma_matched": margins.matched,
                        "croma_random_mean": margins.random.mean(axis=0),
                    }
                )
            )
            replicates.append(replicate_table(margins).assign(model=model))
            summaries.append(summarize(margins).assign(model=model))
            s = summaries[-1].set_index("grade").loc["all"]
            print(
                f"{model}: full {s.full_median:+.3f}  random {s.random_median:+.3f}  "
                f"matched {s.matched_median:+.3f}  delta {s.delta_grade:+.3f}  "
                f"p {s.p_one_sided:.3f}",
                flush=True,
            )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _model_first(pd.concat(per_anchor, ignore_index=True)).to_csv(
        out / "per_anchor.csv", index=False
    )
    _model_first(pd.concat(replicates, ignore_index=True)).to_csv(
        out / "replicates.csv", index=False
    )
    _model_first(pd.concat(summaries, ignore_index=True)).to_csv(out / "summary.csv", index=False)
    counts = pd.crosstab(manifest["confounder"].to_numpy()[anchors], grades.to_numpy())
    meta = {
        "benchmark": BENCHMARK,
        "run_rel": entry.run_rel,
        # The run's per-sample file, read by the full-pool check: an in-place re-run of
        # the benchmark keeps run_rel but changes this, so the paper basis checks it.
        "per_sample_sha256": per_sample_sha256,
        "roster": roster,
        "headline_m": headline_m,
        "search": {
            "m_max": max(search.m_values),
            "start_k": search.start_k,
            "k_growth_factor": search.k_growth_factor,
            "alpha": search.alpha,
        },
        "seed": SEED,
        "n_replicates": args.n_replicates,
        "pool_size": SLIDES_PER_GRADE,
        "cancer_slides_per_grade": {
            p: {str(g): int(counts.loc[p, g]) for g in GRADES} for p in providers
        },
        "n_cancer_anchors": int(len(anchors)),
        "full_pool_reproduces_run": True,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
