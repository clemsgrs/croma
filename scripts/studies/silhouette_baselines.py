"""Batch-mixing baselines from single-cell genomics, against CRoMa and in-domain nIPD.

CRoMa is related to the silhouette, and single-cell integration benchmarking already has
two established batch-mixing scores. With the medical centre in the role of the batch and
the biological class in the role of the cell type:

* **batch-ASW** (scIB): the silhouette of centre labels computed within each biological
  class, ``1 - |s|`` per sample, averaged per class and then over classes, so 1 means
  fully mixed;
* **iLISI** (Korsunsky et al., 2019): the median over samples of the inverse Simpson's
  index of centre labels in a perplexity-weighted kNN neighbourhood, rescaled to [0, 1]
  by ``(LISI - 1) / (n_centres - 1)``.

Both come from the reference implementation, ``scib-metrics`` (``silhouette_batch`` and
``ilisi_knn``), a ``repro`` extra: this module imports it only where it scores an encoder,
so the inference half runs without it. Both are oriented so that higher means better
mixing, which is scib's own convention.

This study asks whether CRoMa carries information those scores do not: how strongly each
agrees with median CRoMa across encoders, and how well each predicts in-domain nIPD, the
downstream shortcut-susceptibility endpoint, with the head-to-head differences
``rho(CRoMa, nIPD_ID) - rho(baseline, nIPD_ID)`` on one shared encoder bootstrap. The result
is recorded whichever way it goes; nothing downstream depends on its sign.

Inputs, per cohort (Camelyon, TCGA-4x4, Tolkach-ESCA: the cohorts with nIPD):

* features from the benchmark view (``views.load_view``), exactly the rows the run
  evaluates, L2-normalised so that Euclidean distance is a monotone function of the cosine
  distance CRoMa uses; the scores on the raw features are recorded too (``*_raw``);
* median CRoMa from the run's ``metrics.csv``, resolved through ``paper_manifest``;
* ``nipd_id`` from the APD study's join, the column ``tab:apd-correlation`` correlates.

Unlike CRoMa, neither baseline drops samples from the anchor's own independence group:
their published definitions do not.

Run (on Python >= 3.11, in an environment with ``croma[repro]``, which carries
scib-metrics there)::

    python scripts/studies/silhouette_baselines.py [--workers 3]

Each (cohort, encoder) score is cached under ``cells/`` with the inputs it was computed
from (the embedding file, a digest of the evaluated rows, labels and confounders, and the
settings: :func:`cell_inputs`), so a re-run re-scores only what changed and then re-joins
and re-infers everything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "studies" / "apd"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from croma.metrics.bootstrap import (  # noqa: E402
    benjamini_hochberg,
    encoder_bootstrap_draws,
    spearman_inference,
)
from loaders import (  # noqa: E402
    CONTROL_MODEL,
    DATASETS,
    PATHOROB_DOWNSTREAM_DATASETS as COHORTS,
    read_joined,
)

STUDY_DIR = REPO / "output/studies/silhouette-baselines"
#: One JSON per (cohort, encoder): its scores and what they were computed from. The scoring
#: is the expensive half, so a re-run re-scores only cells whose inputs changed.
CELLS_DIR = STUDY_DIR / "cells"
PER_ENCODER_CSV = STUDY_DIR / "silhouette_baselines.csv"
CORRELATION_CSV = STUDY_DIR / "silhouette_baselines_correlation.csv"
PROVENANCE_JSON = STUDY_DIR / "silhouette_baselines.json"
IMPLEMENTATION = "scib-metrics"
#: Packages whose versions can move a score or an interval; recorded with every cell.
ENVIRONMENT_PACKAGES = ("scib-metrics", "jax", "jaxlib", "numpy", "scipy", "scikit-learn", "pandas")

#: iLISI's kNN graph: 90 neighbours with the sample itself among them (scib masks the self
#: edge), at perplexity 90 / 3 = 30 -- the settings scib-metrics' own Benchmarker uses.
N_NEIGHBORS = 90
PERPLEXITY = 30


def batch_asw(features: np.ndarray, labels: np.ndarray, batches: np.ndarray) -> float:
    """scib's batch-ASW: ``1 - |silhouette of batches|`` within each label, averaged."""
    from scib_metrics import silhouette_batch

    return float(silhouette_batch(features, labels, batches, rescale=True))


def exact_knn(features: np.ndarray, n_neighbors: int = N_NEIGHBORS):
    """The exact Euclidean kNN graph of ``features``, each sample among its own neighbours.

    scib-metrics accepts any graph; its Benchmarker builds an approximate one with
    pynndescent. Exact search is what that approximates, and it leaves no random state in
    the score.
    """
    from scib_metrics.nearest_neighbors import NeighborsResults
    from sklearn.neighbors import NearestNeighbors

    search = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="euclidean")
    distances, indices = search.fit(features).kneighbors(features)
    return NeighborsResults(indices=indices, distances=distances)


def ilisi(features: np.ndarray, labels: np.ndarray, batches: np.ndarray) -> float:
    """scib's iLISI of batches on the exact 90-neighbour graph, scaled to [0, 1]."""
    from scib_metrics import ilisi_knn

    del labels  # iLISI reads the batch alone
    graph = exact_knn(features)
    return float(ilisi_knn(graph, batches, perplexity=PERPLEXITY, scale=True))


#: Every per-encoder index the study computes, in table order: one function
#: ``(features, labels, batches) -> float`` each, oriented so that higher means better
#: mixing. Adding an index is adding its function here; the scoring, the CSV and the
#: inference all iterate this mapping.
INDICES = {"batch_asw": batch_asw, "ilisi": ilisi}
BASELINES = tuple(INDICES)

#: Scaled iLISI is 0 when the median sample's neighbourhood is single-centre (LISI = 1).
#: scib-metrics computes it in float32, from a 90-term sum per sample, so a floor value
#: comes back as 0 or a few epsilons either side of it, and ranked as returned, encoders on
#: the floor would be ordered by rounding. A value within the worst-case accumulation error
#: of that sum (``N_NEIGHBORS`` float32 epsilons) is the floor, and ties there.
ILISI_FLOOR_TOLERANCE = N_NEIGHBORS * float(np.finfo(np.float32).eps)


def at_ilisi_floor(values) -> np.ndarray:
    """Which scaled iLISI values are indistinguishable from the single-centre floor."""
    return np.abs(np.asarray(values, dtype=float)) <= ILISI_FLOOR_TOLERANCE


def l2_normalise(features: np.ndarray) -> np.ndarray:
    """Unit-norm rows: Euclidean distance is then monotone in the cosine distance."""
    features = np.asarray(features, dtype=np.float64)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


#: Column suffix -> feature preparation. The unsuffixed columns are the reported scores.
FEATURE_VARIANTS = {"": l2_normalise, "_raw": lambda f: np.asarray(f, dtype=np.float64)}


def score_encoder(
    features: np.ndarray, labels: np.ndarray, batches: np.ndarray
) -> dict[str, float]:
    """Every index on every feature variant, keyed ``<index><suffix>``."""
    labels, batches = np.asarray(labels), np.asarray(batches)
    scores: dict[str, float] = {}
    for suffix, prepare in FEATURE_VARIANTS.items():
        prepared = prepare(features)
        for name, index in INDICES.items():
            scores[f"{name}{suffix}"] = index(prepared, labels, batches)
    return scores


#: The pooled CRoMa column and the downstream endpoint.
CROMA = "croma"
TARGET = "nipd_id"

N_BOOT = 2000
N_PERMUTATIONS = 10_000
LEVEL = 0.95
SEED = 0

#: Score columns of the per-encoder frame: every index on every feature variant.
SCORE_COLUMNS = [f"{name}{suffix}" for suffix in FEATURE_VARIANTS for name in INDICES]


def join_cohort(
    dataset: str,
    scores: pd.DataFrame,
    run_metrics: pd.DataFrame,
    apd_join: pd.DataFrame,
) -> pd.DataFrame:
    """One cohort's per-encoder frame: the run's roster, its CRoMa, nIPD and the scores.

    The roster is the run's ``metrics.csv`` (ADR-0010: never declared), in its order;
    ``croma`` is that run's median CRoMa and ``nipd_id`` the APD join's cell for
    ``(dataset, model)``, the value ``tab:apd-correlation`` correlates. A run model without
    a downstream cell or without scores raises rather than dropping out of the panel, and
    so does a run without the natural-image control, whose exclusion is then unverifiable.
    """
    roster = run_metrics["model"].tolist()
    if CONTROL_MODEL not in roster:
        raise ValueError(f"{dataset}: the run has no {CONTROL_MODEL} control to exclude")
    cells = apd_join.loc[apd_join["dataset"] == dataset].set_index("model")
    scored = scores.set_index("model")
    for source, index in (("downstream cell in the APD join", cells.index), ("scores", scored.index)):
        missing = [m for m in roster if m not in index]
        if missing:
            raise ValueError(f"{dataset}: no {source} for run models {missing}")
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "model": roster,
            "ranked": [m != CONTROL_MODEL for m in roster],
            CROMA: run_metrics[CROMA].to_numpy(dtype=float),
            TARGET: cells.loc[roster, TARGET].to_numpy(dtype=float),
        }
    )
    for column in SCORE_COLUMNS:
        frame[column] = scored.loc[roster, column].to_numpy(dtype=float)
    return frame


def _infer_cohort(
    ranked: pd.DataFrame,
    dataset: str,
    *,
    n_boot: int,
    n_permutations: int,
    level: float,
    seed: int,
) -> list[dict]:
    """Agreement, downstream and head-to-head rows for one cohort's ranked encoders.

    One encoder draw is shared by every correlation in the cohort, so the two
    correlations a head-to-head difference subtracts are paired replicate by replicate.
    iLISI values on the floor are set to it first, so they tie rather than rank on rounding.
    """
    ranked = ranked.assign(ilisi=ranked["ilisi"].mask(at_ilisi_floor(ranked["ilisi"]), 0.0))
    n = len(ranked)
    draws = encoder_bootstrap_draws(n, n_boot=n_boot, seed=seed)

    def infer(metric: str, target: str):
        return spearman_inference(
            ranked[metric].to_numpy(dtype=float),
            ranked[target].to_numpy(dtype=float),
            level=level,
            seed=seed,
            n_permutations=n_permutations,
            draws=draws,
        )

    def row(kind: str, metric: str, target: str, ci, p_value: float = np.nan) -> dict:
        return dict(
            dataset=dataset,
            kind=kind,
            metric=metric,
            target=target,
            n=n,
            estimate=ci.point,
            ci_lo=ci.lo,
            ci_hi=ci.hi,
            p_value=p_value,
            level=level,
            n_boot=n_boot,
            n_permutations=n_permutations,
        )

    rows = [row("agreement", m, CROMA, infer(m, CROMA).ci) for m in BASELINES]
    downstream = {m: infer(m, TARGET) for m in (CROMA, *BASELINES)}
    rows += [row("downstream", m, TARGET, d.ci, d.p_value) for m, d in downstream.items()]
    rows += [
        row("head_to_head", m, TARGET, downstream[CROMA].difference_ci(downstream[m]))
        for m in BASELINES
    ]
    return rows


def associations(
    per_encoder: pd.DataFrame,
    *,
    n_boot: int = N_BOOT,
    n_permutations: int = N_PERMUTATIONS,
    level: float = LEVEL,
    seed: int = SEED,
) -> pd.DataFrame:
    """Per-cohort rank correlations across the ranked encoders, with CIs, p and q values.

    ``per_encoder`` has one row per ``(dataset, model)`` with a boolean ``ranked`` column
    and the ``croma``, ``nipd_id`` and baseline columns. The natural-image control
    (``ranked`` false) enters no correlation. Rows, per cohort in input order:

    * ``agreement``: rho(baseline, median CRoMa), with a CI only;
    * ``downstream``: rho(metric, nIPD_ID) for CRoMa and each baseline, with a CI, a
      permutation p and a Benjamini--Hochberg q over every downstream row in the frame
      (the family of this table, separate from ``tab:apd-correlation``'s);
    * ``head_to_head``: rho(CRoMa, nIPD_ID) - rho(baseline, nIPD_ID), with a paired CI.
    """
    rows: list[dict] = []
    for dataset in pd.unique(per_encoder["dataset"]):
        cohort = per_encoder[(per_encoder["dataset"] == dataset) & per_encoder["ranked"]]
        rows += _infer_cohort(
            cohort.reset_index(drop=True),
            str(dataset),
            n_boot=n_boot,
            n_permutations=n_permutations,
            level=level,
            seed=seed,
        )
    table = pd.DataFrame(rows)
    downstream = table["kind"] == "downstream"
    table["q_value"] = np.nan
    table.loc[downstream, "q_value"] = benjamini_hochberg(table.loc[downstream, "p_value"])
    return table[
        [
            "dataset",
            "kind",
            "metric",
            "target",
            "n",
            "estimate",
            "ci_lo",
            "ci_hi",
            "p_value",
            "q_value",
            "level",
            "n_boot",
            "n_permutations",
        ]
    ]


# ---------------------------------------------------------------------------
# Driver: score every (cohort, encoder) cell, join, infer, write.
# ---------------------------------------------------------------------------


def _runs() -> dict:
    """The run each cohort is read from, at the protocol the paper reports.

    Imported here rather than at module scope: ``paper_manifest`` is paper tooling, and only
    the driver needs it.
    """
    if str(REPO / "scripts" / "repro") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts" / "repro"))
    from paper_manifest import by_benchmark

    return {dataset: by_benchmark(DATASETS[dataset]["benchmark"]) for dataset in COHORTS}


def _environment() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ENVIRONMENT_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "absent"
    return versions


def _cell_path(dataset: str, model: str) -> Path:
    return CELLS_DIR / dataset / f"{model}.json"


def view_sha256(rows, labels, confounders) -> str:
    """Hex SHA-256 of everything ``score_encoder`` reads from the view besides features.

    That is, per evaluated sample in eval-manifest order: the tileset embedding row it reads
    (``view.rows``), its biological label and its confounder. The bytes hashed are the UTF-8
    encoding of the compact JSON array ``[rows, labels, confounders]`` (separators ``,`` and
    ``:``), with each row as an integer and each label and confounder as ``str`` of its
    value. The digest therefore moves with a swapped tile, a re-ordered tileset index, a
    relabelled class or a re-pointed confounder column, and not with the container or
    dtype (numpy, pandas, int32 or int64) that carries the same values.
    """
    payload = [
        [int(row) for row in np.asarray(rows).reshape(-1)],
        [str(value) for value in np.asarray(labels, dtype=object).reshape(-1)],
        [str(value) for value in np.asarray(confounders, dtype=object).reshape(-1)],
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cell_inputs(
    benchmark: str,
    *,
    embedding_bytes: int,
    embedding_mtime_ns: int,
    rows,
    labels,
    confounders,
) -> dict:
    """The cache key of one cell: what its scores were computed from.

    The embedding file (size and mtime), the evaluated view (:func:`view_sha256`) and the
    scoring settings. A cached cell is reused only when its key equals this one;
    ``n_samples`` is implied by the view digest and kept for reading.
    """
    return {
        "benchmark": benchmark,
        "embedding_bytes": int(embedding_bytes),
        "embedding_mtime_ns": int(embedding_mtime_ns),
        "n_samples": int(len(rows)),
        "view_sha256": view_sha256(rows, labels, confounders),
        "n_neighbors": N_NEIGHBORS,
        "perplexity": PERPLEXITY,
        "scores": list(SCORE_COLUMNS),
    }


def score_cell(dataset: str, benchmark: str, model: str) -> dict:
    """Score one encoder on one cohort's evaluated rows, or reuse its up-to-date cell.

    A cell is reused only when the embedding file, the evaluated view (rows, labels and
    confounders) and the scoring settings are what it was computed from (:func:`cell_inputs`).
    """
    import layout
    from views import load_view

    view = load_view(benchmark)
    stat = layout.embedding_path(view.spec.tileset, model).stat()
    labels = view.eval_manifest["label"].to_numpy()
    confounders = view.eval_manifest["confounder"].to_numpy()
    inputs = cell_inputs(
        benchmark,
        embedding_bytes=stat.st_size,
        embedding_mtime_ns=stat.st_mtime_ns,
        rows=view.rows,
        labels=labels,
        confounders=confounders,
    )
    path = _cell_path(dataset, model)
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("inputs") == inputs:
            return cached

    started = time.time()
    scores = score_encoder(view.features(model), labels, confounders)
    cell = {
        "dataset": dataset,
        "model": model,
        "inputs": inputs,
        "scores": scores,
        "environment": _environment(),
        "seconds": round(time.time() - started, 1),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cell, indent=2, sort_keys=True) + "\n")
    print(f"[silhouette-baselines] {dataset:9s} {model:16s} {cell['seconds']:7.1f}s", flush=True)
    return cell


def _score_all(runs: dict, workers: int) -> list[dict]:
    """Every cell of every cohort, the heaviest cohort's first, over ``workers`` processes.

    Spawned, not forked: jax is multithreaded and does not survive a fork.
    """
    tasks = [
        (dataset, entry.benchmark, model)
        for dataset, entry in runs.items()
        for model in pd.read_csv(REPO / entry.metrics_rel)["model"]
    ]
    if workers <= 1:
        return [score_cell(*task) for task in tasks]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(score_cell, *task) for task in tasks]
        return [future.result() for future in futures]


def _unique(cells: list[dict], key: str) -> dict:
    values = {json.dumps(cell[key], sort_keys=True) for cell in cells}
    if len(values) != 1:
        raise RuntimeError(
            f"the cells were scored under {len(values)} different {key}s; re-score them in "
            "one environment (delete output/studies/silhouette-baselines/cells/)"
        )
    return json.loads(values.pop())


def build(runs: dict, cells: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The per-encoder frame, its inference table, and the provenance both are read with."""
    apd_join = read_joined()
    frames = []
    for dataset, entry in runs.items():
        scores = pd.DataFrame(
            [{"model": c["model"], **c["scores"]} for c in cells if c["dataset"] == dataset]
        )
        run_metrics = pd.read_csv(REPO / entry.metrics_rel)
        cohort = join_cohort(dataset, scores, run_metrics, apd_join)
        cohort.insert(2, "benchmark", entry.benchmark)
        frames.append(cohort)
    per_encoder = pd.concat(frames, ignore_index=True)
    table = associations(per_encoder)
    environment = _unique(cells, "environment")
    provenance = {
        "implementation": IMPLEMENTATION,
        "implementation_version": environment[IMPLEMENTATION],
        "indices": {
            "batch_asw": "scib_metrics.silhouette_batch(rescale=True)",
            "ilisi": "scib_metrics.ilisi_knn(perplexity=30, scale=True)",
        },
        "label": "biological class",
        "batch": "medical centre (the benchmark's confounder)",
        "n_neighbors": N_NEIGHBORS,
        "perplexity": PERPLEXITY,
        "knn": "exact Euclidean (scikit-learn brute force), each sample among its neighbours",
        "features": "L2-normalised; the *_raw columns score the raw features",
        "same_group_neighbours": "included (CRoMa excludes them)",
        "ilisi_floor_tolerance": ILISI_FLOOR_TOLERANCE,
        "ilisi_floor": "values within the tolerance of 0 are set to 0 (tied) before ranking",
        "runs": {dataset: entry.metrics_rel for dataset, entry in runs.items()},
        "downstream": "output/studies/apd/apd_metrics_joined.csv",
        "inference": {
            "n_boot": N_BOOT,
            "n_permutations": N_PERMUTATIONS,
            "level": LEVEL,
            "seed": SEED,
            "bh_family": "the downstream rows of this table",
        },
        "environment": environment,
    }
    return per_encoder, table, provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Processes scoring cells in parallel (default 3).",
    )
    args = parser.parse_args(argv)

    try:
        import scib_metrics  # noqa: F401
    except ImportError:
        raise SystemExit(
            f"{IMPLEMENTATION} is not installed in {sys.executable}; it is the reference "
            "batch-ASW and iLISI this study reports. Install croma[repro], which pins it, "
            f"on Python >= 3.11 ({IMPLEMENTATION} requires it; this is Python "
            f"{platform.python_version()})."
        ) from None

    runs = _runs()
    cells = _score_all(runs, args.workers)
    per_encoder, table, provenance = build(runs, cells)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    per_encoder.to_csv(PER_ENCODER_CSV, index=False)
    table.to_csv(CORRELATION_CSV, index=False)
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2) + "\n")

    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(table[["dataset", "kind", "metric", "estimate", "ci_lo", "ci_hi", "p_value", "q_value"]])
    print(f"wrote {PER_ENCODER_CSV}\nwrote {CORRELATION_CSV}\nwrote {PROVENANCE_JSON}")


if __name__ == "__main__":
    main()
