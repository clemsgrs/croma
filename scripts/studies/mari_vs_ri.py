"""Why MaRI barely departs from RI on the tile benchmarks, and how much the shared k floats.

For every encoder on the three PathoROB tile benchmarks, the study rebuilds the run's own
neighbours (``_prepare_neighbors``, same-group exclusion, the run's operating k and automatic
tau) and checks they give back the run's published RI, MaRI, tau and support bit for bit.
On those neighbours it then writes, to ``output/studies/mari-vs-ri/``:

``per_encoder.csv``
    One row per benchmark x encoder. ``w_so`` / ``w_os`` are the mean weights over all SO and
    OS neighbour slots pooled across anchors; ``logit_gap`` equals ``log_w_ratio`` exactly
    (``logit(MaRI) - logit(RI) = log(w_so / w_os)``, asserted). ``weight_ratio`` is the median
    over anchors of the nearest neighbour's weight relative to the k-th. ``delta_within`` and
    ``delta_between`` split ``delta = MaRI - RI`` into re-weighting inside each anchor at its
    fixed typed-slot count, and re-weighting across anchors. ``effective_anchor_share`` is
    Kish's effective number of anchors under RI's per-anchor slot counts, over all anchors.
    ``blas_coretype`` is the OpenBLAS kernel the run's rounding was reproduced under.
``tau_sweep.csv``
    MaRI, delta and the effective anchor share under MaRI's anchor weights, rescored at
    ``f * tau`` for each ``f`` in ``TAU_FACTORS`` on the same neighbours.
``tau_sweep_summary.csv``
    Per benchmark x f over the ranked encoders (control excluded): max |delta| and Spearman
    rho of MaRI_f with RI, nIPD_ID and nIPD_OOD (nIPD from the APD study's joined table).
``shared_k_loo.csv``
    Each encoder's k* by the production rule, and the shared operating k (the lower median,
    control included) with each encoder removed in turn. Where a removal moves k, the ranked
    encoders' max |delta RI|, max |delta MaRI|, swapped RI pairs and Spearman rho between the
    two RI orderings, all read from the run's k sweep. That sweep scores MaRI at every k with
    the operating k's tau, so the MaRI column does not re-resolve tau at the moved k.

Every run is resolved through ``paper_manifest.by_benchmark``. Run with one BLAS thread per
worker (``OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1``).
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from benchmark import _shared_operating_k  # noqa: E402
from croma import MaRI, RI  # noqa: E402
from croma.metrics.neighbors import (  # noqa: E402
    _prepare_neighbors,
    _select_k_from_balanced_accuracy,
)


@dataclass(frozen=True)
class Neighbours:
    """One encoder's prepared top-``kmax`` neighbours on one benchmark, as scoring reads them."""

    labels: np.ndarray
    centers: np.ndarray
    neigh_idx: np.ndarray
    neigh_dist: np.ndarray
    valid_counts: np.ndarray


def _score(metric: type, neighbours: Neighbours, k: int, **kwargs: float):
    """The library's own scoring at ``k``: the path the benchmark's pooled values come from."""
    scored = metric._score_all_k_from_neighbors(
        neighbours.labels,
        neighbours.centers,
        neighbours.neigh_idx,
        neighbours.neigh_dist,
        neighbours.valid_counts,
        [int(k)],
        **kwargs,
    )
    return scored[int(k)]


def _logit(p: float) -> float:
    return float(np.log(p) - np.log1p(-p))


def _weight_ratio(neighbours: Neighbours, k: int, tau: float) -> float:
    """Median over anchors of ``exp((d_last - d_1) / tau)`` within each anchor's top-``k``.

    ``d_1`` and ``d_last`` are the nearest and farthest *valid* neighbours, of any type: the
    weight the nearest receives relative to the k-th. Valid neighbours fill each row from the
    left (same-group candidates are skipped, not blanked), so the last one sits at column
    ``min(valid_count, k) - 1``. An anchor with no valid neighbour has no ratio.
    """
    eff_k = np.minimum(neighbours.valid_counts, int(k))
    rows = np.flatnonzero(eff_k > 0)
    d_first = neighbours.neigh_dist[rows, 0]
    d_last = neighbours.neigh_dist[rows, eff_k[rows] - 1]
    return float(np.median(np.exp((d_last - d_first) / float(tau))))


def _effective_anchor_share(anchor_weights: np.ndarray) -> float:
    """Kish's effective number of anchors, ``(sum w)^2 / sum w^2``, over all ``N`` anchors.

    Scale-free in ``w``, so a constant factor on every weight (such as the shift the tau sweep
    applies before exponentiating) cancels.
    """
    w = np.asarray(anchor_weights, dtype=float)
    return float(w.sum() ** 2 / np.square(w).sum() / w.size)


def decompose(neighbours: Neighbours, *, k: int, tau: float) -> dict[str, float]:
    """Pooled RI and MaRI at ``(k, tau)`` and the decomposition of their difference."""
    ri = _score(RI, neighbours, k)
    mari = _score(MaRI, neighbours, k, tau=tau)
    # RI's per-anchor evidence is the SO/OS slot count, MaRI's the summed slot weight.
    w_so = float(mari.so_evidence.sum() / ri.so_evidence.sum())
    w_os = float(mari.os_evidence.sum() / ri.os_evidence.sum())

    # Delta = within + between. Each supported anchor i holds n_i = SO_i + OS_i typed slots and
    # a weighted SO share r_w(i). RI is the n-weighted mean of the *count* shares, so moving
    # from counts to weights inside every anchor, at fixed n_i, is the within-anchor part.
    # What remains is MaRI re-weighting the anchors themselves, by W_i = SO_w(i) + OS_w(i)
    # instead of n_i: a comparison across different anchors.
    n = ri.so_evidence + ri.os_evidence
    supported = n > 0
    weighted = mari.so_evidence[supported] + mari.os_evidence[supported]
    r_w = mari.so_evidence[supported] / weighted
    count_weighted = float((n[supported] * r_w).sum() / n.sum())
    return {
        "k": int(k),
        "tau": float(tau),
        "support": float(np.mean(ri.informative_mask)),
        "ri": float(ri.pooled),
        "mari": float(mari.pooled),
        "delta": float(mari.pooled) - float(ri.pooled),
        "w_so": w_so,
        "w_os": w_os,
        "logit_gap": _logit(float(mari.pooled)) - _logit(float(ri.pooled)),
        "log_w_ratio": float(np.log(w_so) - np.log(w_os)),
        "weight_ratio": _weight_ratio(neighbours, k, tau),
        "delta_within": count_weighted - float(ri.pooled),
        "delta_between": float(mari.pooled) - count_weighted,
        "effective_anchor_share": _effective_anchor_share(n),
    }


class ReproductionError(AssertionError):
    """The study's neighbours do not give back what the run published."""


#: Float tolerance on the identity; both sides are a difference of logs of O(1) numbers.
IDENTITY_ATOL = 1e-9


def verify_reproduction(
    row: dict[str, float],
    published: dict[str, float],
    *,
    recomputed_tau: float,
    context: str,
) -> None:
    """Fail unless ``row`` reproduces the run's ``metrics.csv`` row bit for bit.

    The exact comparison is deliberate. A neighbour search that broke near-ties at the k-th
    slot differently would still agree to a few decimals, and is exactly what this has to
    catch. Exact agreement needs the same floating-point path as the run, BLAS kernel
    included: with OpenBLAS, ``OPENBLAS_CORETYPE`` picks the kernel.
    """
    for column in ("k", "support", "ri", "mari", "tau"):
        if row[column] != published[column]:
            raise ReproductionError(
                f"{context}: {column} = {row[column]!r} but the run published "
                f"{published[column]!r} (check OPENBLAS_CORETYPE matches the run's BLAS kernel)"
            )
    if recomputed_tau != published["tau"]:
        raise ReproductionError(
            f"{context}: the median typed distance at k={row['k']} is {recomputed_tau!r}, "
            f"not the run's automatic tau {published['tau']!r}"
        )
    if abs(row["logit_gap"] - row["log_w_ratio"]) > IDENTITY_ATOL:
        raise ReproductionError(
            f"{context}: identity logit(MaRI) - logit(RI) = log(w_SO / w_OS) fails: "
            f"{row['logit_gap']!r} vs {row['log_w_ratio']!r}"
        )


#: Multiples of each encoder's automatic tau the sweep rescores at, sharp to flat.
TAU_FACTORS = (1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0)


def tau_sweep(
    neighbours: Neighbours,
    *,
    k: int,
    tau: float,
    factors: tuple[float, ...] = TAU_FACTORS,
) -> list[dict[str, float]]:
    """MaRI rescored at ``f * tau`` for each ``f``, on the same top-``k`` neighbours.

    Every distance is shifted down by the smallest valid one before weighting. ``exp(-c / t)``
    then factors out of every weight, so it cancels in MaRI's ratio and in the anchor shares,
    but the raw weights no longer underflow at the sharpest temperatures.
    """
    ri = float(_score(RI, neighbours, k).pooled)
    top_k = neighbours.neigh_dist[:, : int(k)]
    shift = float(np.min(top_k[np.isfinite(top_k)]))
    shifted = Neighbours(
        labels=neighbours.labels,
        centers=neighbours.centers,
        neigh_idx=neighbours.neigh_idx,
        neigh_dist=neighbours.neigh_dist - shift,
        valid_counts=neighbours.valid_counts,
    )
    rows = []
    for f in factors:
        tau_f = float(f) * float(tau)
        mari = _score(MaRI, shifted, k, tau=tau_f)
        rows.append(
            {
                "f": float(f),
                "tau_f": tau_f,
                "mari": float(mari.pooled),
                "delta": float(mari.pooled) - ri,
                "effective_anchor_share": _effective_anchor_share(
                    mari.so_evidence + mari.os_evidence
                ),
            }
        )
    return rows


def summarise_tau_sweep(sweep: pd.DataFrame, *, ranked: list[str]) -> pd.DataFrame:
    """Per benchmark x f, over the ranked encoders: max |Delta| and MaRI_f's Spearman rho.

    ``sweep`` carries one row per benchmark x encoder x f with that encoder's ``ri``,
    ``mari`` (at ``f * tau``), ``delta`` and ``nipd_id`` / ``nipd_ood``.
    """
    rows = []
    kept = sweep[sweep["model"].isin(ranked)]
    for (benchmark, f), group in kept.groupby(["benchmark", "f"], sort=False):
        rows.append(
            {
                "benchmark": benchmark,
                "f": float(f),
                "n": int(len(group)),
                "max_abs_delta": float(group["delta"].abs().max()),
                **{
                    f"rho_{target}": float(spearmanr(group["mari"], group[target]).statistic)
                    for target in ("ri", "nipd_id", "nipd_ood")
                },
            }
        )
    return pd.DataFrame(rows)


EFFECT_COLUMNS = ("max_abs_delta_ri", "max_abs_delta_mari", "swapped_pairs", "rho_ri")


def shared_k_loo(sweep: pd.DataFrame, *, run_k: int, ranked: list[str]) -> pd.DataFrame:
    """One row per removed encoder: its k*, the shared k without it, and, where that k
    differs from the run's, how far the ranked encoders' RI and MaRI move at it.

    ``sweep`` is the run's whole k sweep, control included: production takes the median over
    every evaluated encoder, so the reconstruction must too.
    """
    k_star = k_star_from_sweep(sweep)
    reconstructed = _shared_operating_k(k_star)
    if reconstructed != int(run_k):
        raise ReproductionError(
            f"the production rule gives shared k {reconstructed} over {len(k_star)} "
            f"encoders, but the run used k = {run_k}"
        )
    effects = {}
    rows = []
    for row in leave_one_out(k_star):
        if row["moves_k"] and row["loo_k"] not in effects:
            effects[row["loo_k"]] = alternative_k_effect(
                sweep, run_k=run_k, alt_k=row["loo_k"], models=ranked
            )
        effect = effects.get(row["loo_k"]) if row["moves_k"] else None
        rows.append({**row, **{c: (effect[c] if effect else np.nan) for c in EFFECT_COLUMNS}})
    return pd.DataFrame(rows)


def k_star_from_sweep(sweep: pd.DataFrame) -> dict[str, int]:
    """Each encoder's own k*, by the production rule, from a run's ``k_sweep_metrics.csv``."""
    k_star = {}
    for model, rows in sweep.groupby("model", sort=False):
        k_values = [int(k) for k in str(rows["k_values"].iloc[0]).split(",")]
        scores = {int(k): float(b) for k, b in zip(rows["k"], rows["knn_bacc"])}
        k_star[str(model)] = _select_k_from_balanced_accuracy(k_values=k_values, scores=scores)
    return k_star


def leave_one_out(k_star: dict[str, int]) -> list[dict]:
    """The shared operating k with each encoder removed in turn, the control included.

    Uses the benchmark's own rule (``_shared_operating_k``, the lower median), so the panel
    dependence measured here is the one production has.
    """
    run_k = _shared_operating_k(k_star)
    rows = []
    for removed in k_star:
        loo_k = _shared_operating_k({m: k for m, k in k_star.items() if m != removed})
        rows.append(
            {
                "removed": removed,
                "k_star": int(k_star[removed]),
                "run_k": int(run_k),
                "loo_k": int(loo_k),
                "moves_k": bool(loo_k != run_k),
            }
        )
    return rows


def alternative_k_effect(
    sweep: pd.DataFrame, *, run_k: int, alt_k: int, models: list[str]
) -> dict[str, float]:
    """How far ``models``' RI and MaRI move if the shared k were ``alt_k`` instead of ``run_k``.

    Both are read from the run's k sweep, which scores MaRI at every k with the operating
    k's tau; a run whose shared k had moved would also re-resolve tau at the new k.
    """
    at = sweep.set_index(["k", "model"])
    run = at.loc[int(run_k)].loc[models]
    alt = at.loc[int(alt_k)].loc[models]
    ri_run, ri_alt = run["ri"].to_numpy(float), alt["ri"].to_numpy(float)
    i, j = np.triu_indices(len(models), k=1)
    discordant = (ri_run[i] - ri_run[j]) * (ri_alt[i] - ri_alt[j]) < 0
    return {
        "max_abs_delta_ri": float(np.max(np.abs(ri_alt - ri_run))),
        "max_abs_delta_mari": float(np.max(np.abs(alt["mari"].to_numpy(float) - run["mari"]))),
        "swapped_pairs": int(discordant.sum()),
        "rho_ri": float(spearmanr(ri_run, ri_alt).statistic),
    }


# --- Reading the runs ------------------------------------------------------------------------

BENCHMARKS = ("pathorob-camelyon", "pathorob-tcga-4x4", "pathorob-tolkach-esca")
OUT_DIR = REPO / "output/studies/mari-vs-ri"
APD_JOINED = REPO / "output/studies/apd/apd_metrics_joined.csv"
APD_CORRELATION = REPO / "output/studies/apd/apd_correlation.csv"
PUBLISHED_COLUMNS = ("k", "tau", "support", "ri", "mari")


def _read(path: Path) -> pd.DataFrame:
    # pandas' default float parser can land an ulp off the written value; the exact
    # comparison against metrics.csv needs the round-trip one.
    return pd.read_csv(path, float_precision="round_trip")


def _repro_imports():
    for path in (REPO / "scripts" / "repro", REPO / "scripts" / "studies" / "apd"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import views
    from loaders import DATASETS
    from paper_manifest import by_benchmark
    from plotting.style import CONTROL_MODEL

    return views, DATASETS, by_benchmark, CONTROL_MODEL


@lru_cache(maxsize=None)
def _view(benchmark: str):
    views, *_ = _repro_imports()
    return views.load_view(benchmark)


def benchmark_neighbours(benchmark: str, model: str, kmax: int) -> Neighbours:
    """The benchmark's own all-rows neighbour preparation, step for step."""
    view = _view(benchmark)
    features = view.features(model)
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
    df = RI._normalize_manifest_inputs(
        view.eval_manifest, confounder_column=view.spec.confounder_column
    )
    prepared = RI._prepare_all_rows_inputs(features=features, df=df, assume_normalized=True)
    neigh_idx, neigh_dist, valid_counts = _prepare_neighbors(
        prepared.features, prepared.group_ids, int(kmax)
    )
    return Neighbours(prepared.labels, prepared.centers, neigh_idx, neigh_dist, valid_counts)


def automatic_tau(neighbours: Neighbours, k: int) -> float:
    """The benchmark's automatic tau: the median finite typed-neighbour distance at ``k``."""
    typed = MaRI._typed_neighbor_distances_from_neighbors(
        labels=neighbours.labels,
        centers=neighbours.centers,
        neigh_idx=neighbours.neigh_idx,
        neigh_dist=neighbours.neigh_dist,
        valid_counts=neighbours.valid_counts,
        k=int(k),
    )
    return float(np.median(typed[np.isfinite(typed)]))


#: OpenBLAS kernels the neighbour search is tried under, in turn; ``None`` is the host's
#: default. The published runs were not all scored under one kernel -- encoders whose
#: metrics were cached from an earlier host keep that host's rounding -- and a different
#: kernel rounds the cosine distances differently, which reorders near-ties at the k-th
#: slot. Each encoder keeps the first kernel that gives its published row back exactly.
BLAS_CORETYPES = (None, "SkylakeX", "Cooperlake")


def _numpy_blas_architecture() -> str | None:
    from threadpoolctl import threadpool_info

    for info in threadpool_info():
        if info.get("internal_api") == "openblas" and "numpy" in str(info.get("filepath")):
            return info.get("architecture")
    return None


def _encoder_task(benchmark: str, model: str, published: dict, kmax: int):
    """Score one encoder; a mismatch is returned, not raised, so it can be retried."""
    architecture = _numpy_blas_architecture()
    neighbours = benchmark_neighbours(benchmark, model, kmax)
    k, tau = int(published["k"]), float(published["tau"])
    context = f"{benchmark}/{model}"
    row = decompose(neighbours, k=k, tau=tau)
    try:
        verify_reproduction(
            row, published, recomputed_tau=automatic_tau(neighbours, k), context=context
        )
    except ReproductionError as exc:
        return benchmark, model, architecture, str(exc), None
    sweep = tau_sweep(neighbours, k=k, tau=tau)
    at_tau = next(r for r in sweep if r["f"] == 1.0)
    if abs(at_tau["mari"] - row["mari"]) > IDENTITY_ATOL:
        raise ReproductionError(f"{context}: the shifted sweep at f = 1 is not MaRI")
    return benchmark, model, architecture, {**row, "blas_coretype": architecture}, sweep


def _score_encoders(tasks: list, *, workers: int, coretypes=BLAS_CORETYPES) -> dict:
    """Run every task, retrying the ones that do not reproduce under the next kernel."""
    pending = {(task[0], task[1]): task for task in tasks}
    done, failures, tried = {}, {}, set()
    for coretype in coretypes:
        if not pending or coretype in tried:
            continue
        saved = os.environ.get("OPENBLAS_CORETYPE")
        if coretype is None:
            os.environ.pop("OPENBLAS_CORETYPE", None)
        else:
            os.environ["OPENBLAS_CORETYPE"] = coretype
        try:
            context = get_context("spawn")  # a fresh interpreter reads the kernel at load
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_encoder_task, *task) for task in pending.values()]
                for future in as_completed(futures):
                    benchmark, model, architecture, row, sweep = future.result()
                    tried.add(architecture)
                    if sweep is None:
                        failures[(benchmark, model)] = row
                        continue
                    print(
                        f"[mari-vs-ri] {benchmark} {model}: reproduced ({architecture})",
                        flush=True,
                    )
                    done[(benchmark, model)] = (row, sweep)
                    del pending[(benchmark, model)]
        finally:
            if saved is None:
                os.environ.pop("OPENBLAS_CORETYPE", None)
            else:
                os.environ["OPENBLAS_CORETYPE"] = saved
    if pending:
        detail = "\n".join(failures[key] for key in pending)
        raise ReproductionError(f"no BLAS kernel in {coretypes} reproduces:\n{detail}")
    return done


def _nipd(datasets: dict) -> pd.DataFrame:
    joined = _read(APD_JOINED)
    to_benchmark = {key: spec["benchmark"] for key, spec in datasets.items()}
    joined = joined[joined["dataset"].isin(to_benchmark)]
    return joined.assign(benchmark=joined["dataset"].map(to_benchmark))[
        ["benchmark", "dataset", "model", "mari", "nipd_id", "nipd_ood"]
    ]


def _check_nipd_basis(per_encoder: pd.DataFrame, nipd: pd.DataFrame, summary: pd.DataFrame):
    """The APD study must score the same runs, so its MaRI-nIPD rho is this sweep's at f = 1."""
    joined = per_encoder.merge(nipd, on=["benchmark", "model"], suffixes=("", "_apd"))
    if len(joined) != len(per_encoder):
        raise ReproductionError("the APD joined table does not cover every encoder")
    if not np.allclose(joined["mari"], joined["mari_apd"], rtol=0, atol=1e-12):
        raise ReproductionError("the APD joined table was built from different runs")
    corr = _read(APD_CORRELATION)
    corr = corr[(corr["target"] == "nipd_id") & (corr["metric"] == "mari")]
    at_tau = summary[summary["f"] == 1.0]
    for _, row in at_tau.iterrows():
        dataset = nipd.loc[nipd["benchmark"] == row["benchmark"], "dataset"].iloc[0]
        published = corr.loc[corr["scope"] == dataset, "spearman"]
        if len(published) != 1 or abs(float(published.iloc[0]) - row["rho_nipd_id"]) > 1e-9:
            raise ReproductionError(
                f"{row['benchmark']}: rho(MaRI, nIPD_ID) at f = 1 is {row['rho_nipd_id']!r}, "
                f"but the APD study reports {published.tolist()}"
            )


def run(*, benchmarks=BENCHMARKS, out_dir: Path = OUT_DIR, workers: int = 4) -> None:
    """Score every encoder on every benchmark and write the study's four tables."""
    _, datasets, by_benchmark, control = _repro_imports()
    tasks, loo_tables, ranked_by_benchmark = [], [], {}
    for benchmark in benchmarks:
        entry = by_benchmark(benchmark)
        metrics = _read(REPO / entry.metrics_rel)
        k_sweep = _read(REPO / entry.run_rel / "results" / "k_sweep_metrics.csv")
        run_k = int(metrics["k"].iloc[0])
        if metrics["k"].nunique() != 1:
            raise ReproductionError(f"{benchmark}: metrics.csv holds more than one k")
        models = metrics["model"].tolist()
        ranked = [m for m in models if m != control]
        ranked_by_benchmark[benchmark] = ranked
        loo = shared_k_loo(k_sweep, run_k=run_k, ranked=ranked)
        loo.insert(0, "benchmark", benchmark)
        loo.insert(2, "is_control", loo["removed"] == control)
        loo_tables.append(loo)
        kmax = max(int(k) for k in str(metrics["k_values"].iloc[0]).split(","))
        for _, published in metrics.iterrows():
            values = {c: published[c] for c in PUBLISHED_COLUMNS}
            tasks.append((benchmark, published["model"], values, kmax))

    done = _score_encoders(tasks, workers=workers)
    results = [(b, m, *done[(b, m)]) for b, m, _, _ in tasks]

    def _keyed(benchmark, model):
        return {"benchmark": benchmark, "model": model, "is_control": model == control}

    per_encoder = pd.DataFrame([{**_keyed(b, m), **row} for b, m, row, _ in results])
    sweep = pd.DataFrame([{**_keyed(b, m), **s} for b, m, _, rows in results for s in rows])
    nipd = _nipd(datasets)
    joined = sweep.merge(per_encoder[["benchmark", "model", "ri"]], on=["benchmark", "model"])
    joined = joined.merge(nipd[["benchmark", "model", "nipd_id", "nipd_ood"]], how="left")
    summary = pd.concat(
        [
            summarise_tau_sweep(joined[joined["benchmark"] == b], ranked=ranked)
            for b, ranked in ranked_by_benchmark.items()
        ],
        ignore_index=True,
    )
    _check_nipd_basis(per_encoder, nipd, summary)

    out_dir.mkdir(parents=True, exist_ok=True)
    per_encoder.to_csv(out_dir / "per_encoder.csv", index=False)
    sweep.to_csv(out_dir / "tau_sweep.csv", index=False)
    summary.to_csv(out_dir / "tau_sweep_summary.csv", index=False)
    pd.concat(loo_tables, ignore_index=True).to_csv(out_dir / "shared_k_loo.csv", index=False)
    print(f"[mari-vs-ri] wrote {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    run(out_dir=args.out_dir, workers=args.workers)


if __name__ == "__main__":
    main()
