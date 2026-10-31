"""How far MaRI departs from RI per encoder, with a paired cluster-bootstrap interval.

For every encoder on the four PathoROB tile benchmarks (Camelyon, TCGA-2x2, TCGA-4x4,
Tolkach-ESCA) the study rebuilds the run's own neighbours once, at the run's operating k
(``_prepare_neighbors`` with same-group exclusion; on TCGA-2x2 one neighbour search per
paired subset, as the benchmark does), and checks they give back the run's published RI,
MaRI, tau and support bit for bit (``mari_vs_ri.verify_reproduction``). Those neighbours are
then held fixed. Each replicate redraws the anchors by a stratified cluster draw -- slides as
clusters, class x centre strata (centre alone where a case spans classes), the design and
seed of ``bootstrap_uncertainty.py`` -- and one draw is shared by RI, MaRI and every encoder.
A drawn anchor contributes its typed (SO/OS) neighbour slots once per draw. Per replicate:

* RI is the pooled SO share of the drawn anchors' typed slots;
* tau is re-resolved as the median typed-slot distance over the drawn anchors' slots (the
  automatic tau, on the replicate's anchors);
* MaRI is the pooled SO share of the drawn slots' weights ``exp(-d / tau)``.

``Delta = MaRI - RI`` and the logit gap ``logit(MaRI) - logit(RI)`` get percentile CIs. Per
benchmark the study writes, under the run's ``studies/`` directory:

``mari_ri_delta.csv``
    One row per encoder (control flagged): k, tau, RI, MaRI, Delta and the logit gap with
    95% CIs, whether Delta's CI excludes 0, and the OpenBLAS kernel the run's rounding was
    reproduced under.
``mari_ri_delta.json``
    The design, ``n_boot``, seed and how many ranked encoders' Delta CI excludes 0.

Run with one BLAS thread per worker (``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "studies"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import mari_vs_ri  # noqa: E402
from bootstrap_uncertainty import SEED, stratum_labels  # noqa: E402
from croma import MaRI, RI  # noqa: E402
from croma.metrics.base import _EvidenceTotals  # noqa: E402
from croma.metrics.bootstrap import _percentile_ci, cluster_bootstrap_draws  # noqa: E402

BENCHMARKS = (
    "pathorob-camelyon",
    "pathorob-tcga-2x2",
    "pathorob-tcga-4x4",
    "pathorob-tolkach-esca",
)
N_BOOT = 2000
LEVEL = 0.95
#: The replicate formula, evaluated with every anchor drawn once, must give the library's
#: pooled MaRI back to this relative tolerance (it sums in a different order); RI and tau
#: are compared exactly.
POINT_RTOL = 1e-12


@dataclass(frozen=True)
class Slots:
    """Every typed (SO/OS) neighbour slot at the operating k, ascending by distance."""

    anchor: np.ndarray
    dist: np.ndarray
    is_so: np.ndarray

    @classmethod
    def concat(cls, parts: list[Slots]) -> Slots:
        anchor = np.concatenate([p.anchor for p in parts])
        dist = np.concatenate([p.dist for p in parts])
        is_so = np.concatenate([p.is_so for p in parts])
        order = np.argsort(dist, kind="stable")
        return cls(anchor[order], dist[order], is_so[order])


def typed_slots(neighbours: mari_vs_ri.Neighbours, k: int, *, offset: int = 0) -> Slots:
    """The SO and OS slots within each anchor's top ``k``, as the library masks them.

    ``offset`` shifts the anchor index, so the slots of several neighbour searches (the
    paired subsets) can share one anchor numbering.
    """
    cols = min(int(k), neighbours.neigh_idx.shape[1])
    col = np.arange(cols)[np.newaxis, :]
    idx = neighbours.neigh_idx[:, :cols]
    valid = (col < np.minimum(neighbours.valid_counts, int(k))[:, np.newaxis]) & (idx >= 0)
    safe = np.where(valid, idx, 0)
    same_label = neighbours.labels[safe] == neighbours.labels[:, np.newaxis]
    same_center = neighbours.centers[safe] == neighbours.centers[:, np.newaxis]
    typed = valid & (same_label != same_center)
    anchor = np.broadcast_to(np.arange(len(idx))[:, np.newaxis], idx.shape)[typed]
    dist = np.asarray(neighbours.neigh_dist[:, :cols][typed], dtype=float)
    is_so = same_label[typed]
    finite = np.isfinite(dist)
    return Slots.concat([Slots(anchor[finite] + int(offset), dist[finite], is_so[finite])])


def _weighted_median(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    """``np.median`` of ``sorted_values`` with value ``i`` repeated ``weights[i]`` times."""
    cumulative = np.cumsum(weights)
    total = int(cumulative[-1])
    upper = sorted_values[np.searchsorted(cumulative, total // 2, side="right")]
    if total % 2:
        return float(upper)
    lower = sorted_values[np.searchsorted(cumulative, total // 2 - 1, side="right")]
    return float((lower + upper) / 2)


def replicate_scores(slots: Slots, counts: np.ndarray) -> tuple[float, float, float]:
    """Pooled RI, MaRI and the re-resolved tau over anchors drawn ``counts`` times each."""
    weights = np.asarray(counts, dtype=float)[slots.anchor]
    ri = float(weights[slots.is_so].sum() / weights.sum())
    tau = _weighted_median(slots.dist, weights)
    weighted = weights * np.exp(-slots.dist / tau)
    mari = float(weighted[slots.is_so].sum() / weighted.sum())
    return ri, mari, tau


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    return np.log(p) - np.log1p(-np.asarray(p))


def summarise(
    ri: float, mari: float, ri_reps: np.ndarray, mari_reps: np.ndarray, *, level: float = LEVEL
) -> dict:
    """Delta and the logit gap with percentile CIs; does Delta's interval exclude 0?"""
    ri_reps, mari_reps = np.asarray(ri_reps, dtype=float), np.asarray(mari_reps, dtype=float)
    n_boot = len(ri_reps)
    delta = _percentile_ci(mari_reps - ri_reps, mari - ri, level, n_boot)
    gap = _percentile_ci(
        _logit(mari_reps) - _logit(ri_reps), float(_logit(mari) - _logit(ri)), level, n_boot
    )
    return {
        "delta": delta.point,
        "delta_lo": delta.lo,
        "delta_hi": delta.hi,
        "logit_gap": gap.point,
        "logit_gap_lo": gap.lo,
        "logit_gap_hi": gap.hi,
        "excludes_zero": bool(delta.lo > 0 or delta.hi < 0),
    }


# --- the paper's runs -------------------------------------------------------------------


@dataclass(frozen=True)
class Anchors:
    """Every scored anchor in slot-numbering order: its slide and its stratum."""

    clusters: np.ndarray
    strata: np.ndarray


def _paired_neighbours(benchmark: str, model: str, kmax: int):
    """The benchmark's paired-subset neighbour searches, one per manifest subset, in order."""
    from croma.metrics.pairs import resolve_manifest_subsets

    view = mari_vs_ri._view(benchmark)
    features = view.features(model)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    caches = RI._prepare_paired_subset_neighbor_cache(
        features=features,
        subsets=resolve_manifest_subsets(view.eval_manifest),
        k_values=[int(kmax)],
    )
    return [
        (
            mari_vs_ri.Neighbours(c.labels, c.centers, c.neigh_idx, c.neigh_dist, c.valid_counts),
            c.source_indices,
            c.group_ids,
        )
        for c in caches
    ]


def _paired_point(parts: list[mari_vs_ri.Neighbours], k: int, tau: float) -> dict:
    """RI, MaRI, tau and support pooled over the paired subsets, as the benchmark pools them."""
    ri_total, mari_total, informative = _EvidenceTotals(), _EvidenceTotals(), []
    typed = []
    for nb in parts:
        ri = mari_vs_ri._score(RI, nb, k)
        mari = mari_vs_ri._score(MaRI, nb, k, tau=tau)
        ri_total.add(ri.so_evidence, ri.os_evidence, ri.log_so_total, ri.log_os_total)
        mari_total.add(mari.so_evidence, mari.os_evidence, mari.log_so_total, mari.log_os_total)
        informative.append(ri.informative_mask)
        typed.append(
            MaRI._typed_neighbor_distances_from_neighbors(
                labels=nb.labels,
                centers=nb.centers,
                neigh_idx=nb.neigh_idx,
                neigh_dist=nb.neigh_dist,
                valid_counts=nb.valid_counts,
                k=int(k),
            )
        )
    distances = np.concatenate(typed)
    w_so, w_os = mari_total.so / ri_total.so, mari_total.os / ri_total.os
    ri_pooled, mari_pooled = float(ri_total.ratio()), float(mari_total.ratio())
    return {
        "k": int(k),
        "tau": float(tau),
        "support": float(np.mean(np.concatenate(informative))),
        "ri": ri_pooled,
        "mari": mari_pooled,
        "logit_gap": float(_logit(mari_pooled) - _logit(ri_pooled)),
        "log_w_ratio": float(np.log(w_so) - np.log(w_os)),
        "recomputed_tau": float(np.median(distances[np.isfinite(distances)])),
    }


def _neighbour_parts(benchmark: str, model: str, kmax: int):
    """``(neighbours, slide ids, labels, confounders)`` per neighbour search, in order."""
    view = mari_vs_ri._view(benchmark)
    manifest = view.eval_manifest
    if view.spec.design == "paired_2x2":
        return [
            (
                nb,
                np.asarray(groups).astype(str),
                manifest["label"].to_numpy()[source],
                manifest["confounder"].to_numpy()[source],
            )
            for nb, source, groups in _paired_neighbours(benchmark, model, kmax)
        ]
    df = RI._normalize_manifest_inputs(manifest, confounder_column=view.spec.confounder_column)
    nb = mari_vs_ri.benchmark_neighbours(benchmark, model, kmax)
    return [
        (
            nb,
            df["group_id"].astype(str).to_numpy(),
            df["label"].to_numpy(),
            df["confounder"].to_numpy(),
        )
    ]


def _delta_task(benchmark: str, model: str, published: dict, kmax: int, n_boot: int, seed: int):
    """Score one encoder and bootstrap its Delta; a non-reproduction is returned, not raised."""
    architecture = mari_vs_ri._numpy_blas_architecture()
    k, tau = int(published["k"]), float(published["tau"])
    context = f"{benchmark}/{model}"
    parts = _neighbour_parts(benchmark, model, kmax)
    if len(parts) == 1:
        nb = parts[0][0]
        row = mari_vs_ri.decompose(nb, k=k, tau=tau)
        recomputed = mari_vs_ri.automatic_tau(nb, k)
    else:
        row = _paired_point([p[0] for p in parts], k, tau)
        recomputed = row.pop("recomputed_tau")
    try:
        mari_vs_ri.verify_reproduction(row, published, recomputed_tau=recomputed, context=context)
    except mari_vs_ri.ReproductionError as exc:
        return benchmark, model, architecture, str(exc), None

    offsets = np.cumsum([0] + [len(p[0].labels) for p in parts])
    slots = Slots.concat([typed_slots(p[0], k, offset=o) for p, o in zip(parts, offsets)])
    clusters = np.concatenate([p[1] for p in parts])
    strata, design = stratum_labels(
        np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]), clusters
    )
    n_anchors = int(offsets[-1])
    ri_unit, mari_unit, tau_unit = replicate_scores(slots, np.ones(n_anchors))
    if ri_unit != row["ri"] or tau_unit != tau or abs(mari_unit / row["mari"] - 1) > POINT_RTOL:
        raise mari_vs_ri.ReproductionError(
            f"{context}: the replicate formula with every anchor once gives RI {ri_unit!r}, "
            f"MaRI {mari_unit!r}, tau {tau_unit!r}, not the run's {row['ri']!r}, "
            f"{row['mari']!r}, {tau!r}"
        )
    reps = np.empty((int(n_boot), 3), dtype=float)
    draws = cluster_bootstrap_draws(clusters, strata=strata, n_boot=n_boot, seed=seed)
    for b, rows in enumerate(draws):
        reps[b] = replicate_scores(slots, np.bincount(rows, minlength=n_anchors))
    summary = summarise(row["ri"], row["mari"], reps[:, 0], reps[:, 1])
    out = {
        "k": k,
        "tau": tau,
        "tau_lo": _percentile_ci(reps[:, 2], tau, LEVEL, int(n_boot)).lo,
        "tau_hi": _percentile_ci(reps[:, 2], tau, LEVEL, int(n_boot)).hi,
        "ri": row["ri"],
        "mari": row["mari"],
        **summary,
        "strata": design,
        "n_clusters": int(len(np.unique(clusters))),
        "n_anchors": n_anchors,
        "blas_coretype": architecture,
    }
    print(
        f"[mari-ri-delta] {context}: Delta {summary['delta']:+.4f} "
        f"[{summary['delta_lo']:+.4f}, {summary['delta_hi']:+.4f}] ({architecture})",
        flush=True,
    )
    return benchmark, model, architecture, out, True


def run(
    *,
    benchmarks=BENCHMARKS,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    workers: int = 4,
    out_root: Path = REPO,
) -> None:
    _, _, by_benchmark, control = mari_vs_ri._repro_imports()
    tasks, rosters = [], {}
    for benchmark in benchmarks:
        metrics = mari_vs_ri._read(REPO / by_benchmark(benchmark).metrics_rel)
        if metrics["k"].nunique() != 1:
            raise mari_vs_ri.ReproductionError(f"{benchmark}: metrics.csv holds more than one k")
        kmax = max(int(k) for k in str(metrics["k_values"].iloc[0]).split(","))
        rosters[benchmark] = metrics["model"].tolist()
        for _, published in metrics.iterrows():
            values = {c: published[c] for c in mari_vs_ri.PUBLISHED_COLUMNS}
            tasks.append((benchmark, published["model"], values, kmax, int(n_boot), int(seed)))

    done = mari_vs_ri._score_encoders(tasks, workers=workers, task=_delta_task)
    for benchmark, models in rosters.items():
        table = pd.DataFrame(
            [
                {"model": m, "is_control": m == control, **done[(benchmark, m)][0]}
                for m in models
            ]
        )
        ranked = table[~table["is_control"]]
        out = out_root / by_benchmark(benchmark).studies_rel
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "mari_ri_delta.csv", index=False)
        meta = {
            "benchmark": benchmark,
            "n_boot": int(n_boot),
            "seed": int(seed),
            "level": LEVEL,
            "k": int(table["k"].iloc[0]),
            "strata": str(table["strata"].iloc[0]),
            "n_clusters": int(table["n_clusters"].iloc[0]),
            "n_anchors": int(table["n_anchors"].iloc[0]),
            "n_ranked": int(len(ranked)),
            "n_excluding_zero": int(ranked["excludes_zero"].sum()),
        }
        (out / "mari_ri_delta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(
            f"[mari-ri-delta] {benchmark}: {meta['n_excluding_zero']}/{meta['n_ranked']} ranked "
            f"encoders' Delta CI excludes 0 -> {out}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument("--out-root", type=Path, default=REPO, help="where outputs are written")
    args = parser.parse_args()
    run(
        benchmarks=tuple(args.benchmark or BENCHMARKS),
        n_boot=args.n_boot,
        seed=args.seed,
        workers=args.workers,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
