"""CRoMa under three alternative distances: does the encoder ranking depend on cosine?

Every tile benchmark is re-scored at the headline radius (m = 5), with its own evaluation
design and its run's CRoMa search, under four geometries:

====================  ===============  =========  =============================================
variant               features         distance   what it tests
====================  ===============  =========  =============================================
cosine                as in the run    cosine     the reference: must reproduce ``croma_m5``
normalised-euclidean  L2-normalised    Euclidean  the chord identity, at m = 5 where it is
                                                  not exact
euclidean             raw              Euclidean  whether the feature norm matters
whitened              PCA-whitened     Euclidean  anisotropy: Mahalanobis distance under a
                                                  truncated covariance
====================  ===============  =========  =============================================

Whitening is fit per encoder x benchmark on the benchmark's evaluated rows (each tile once,
however many TCGA-2x2 subsets it occurs in) and uses no labels and no confounders: the
fewest leading principal components explaining at least 95% of the variance, each scaled to
unit variance. Camelyon is also run at 90% and 99% as a sensitivity check.

**Reproduction.** The cosine variant is the run's own computation, and the study refuses to
summarise unless every cosine per-sample value equals the run's stored ``croma_m5`` exactly.
A published run was computed under one of two OpenBLAS GEMM kernels, which round float32
distances differently enough to reorder near-tied neighbours, so each (benchmark, encoder)
recovers its kernel by trying the candidates in turn (each in a process of its own, since a
core type is fixed when OpenBLAS loads) and computes every other variant under the same one.

**Measures** (``summarise``) are per benchmark x variant against cosine, over the ranked
encoders (the natural-image control excluded): Spearman rho between the rankings by median
CRoMa and by LTM10, with an encoder-bootstrap CI; how many encoders' medians change sign;
the median over encoders of |delta F(0)| and of per-sample sign agreement; and, on the three
cohorts with nIPD cells, rho(median CRoMa, nIPD_ID / nIPD_OOD) under each variant, with a CI
and a permutation p. One seeded encoder draw is shared by every correlation of a benchmark.

Usage::

    python scripts/studies/distance_variants.py sweep --workers 4   # resumable
    python scripts/studies/distance_variants.py status
    python scripts/studies/distance_variants.py summarise           # sweep runs it at the end

Outputs go to ``output/studies/distance-variants/``: one ``.json`` record and one
per-sample ``.csv.gz`` (keyed by ``subset`` and ``sample_id``) per encoder x benchmark x
variant under ``cells/``, then ``summary.csv``, ``measures.csv`` and ``provenance.json``,
which records the sha256 of every input the published numbers are computed from.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from croma import CRoMa  # noqa: E402
from croma.metrics.bootstrap import encoder_bootstrap_draws, spearman_inference  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
from metrics_io import safe_model_name  # noqa: E402
from plotting.style import CONTROL_MODEL  # noqa: E402

STUDY_DIR = REPO / "output" / "studies" / "distance-variants"
DESIGN = "design.json"
SUMMARY = "summary.csv"
MEASURES = "measures.csv"
PROVENANCE = "provenance.json"

#: The tile benchmarks, heaviest first so a parallel sweep finishes evenly.
BENCHMARKS = (
    "pathorob-tcga-2x2",
    "pathorob-camelyon",
    "pathorob-tolkach-esca",
    "pathorob-tcga-4x4",
)
#: Each cohort with nIPD cells, and its key in the APD study's ``apd.csv``.
NIPD_DATASETS = {
    "pathorob-camelyon": "camelyon",
    "pathorob-tcga-4x4": "tcga_4x4",
    "pathorob-tolkach-esca": "tolkach",
}
NIPD_TARGETS = ("nipd_id", "nipd_ood")
APD_CSV = REPO / "output" / "studies" / "apd" / "apd.csv"
APD_CORRELATION_CSV = REPO / "output" / "studies" / "apd" / "apd_correlation.csv"

#: Encoder bootstrap and permutation settings; every correlation of a benchmark shares one
#: draw, so differences between its variants are paired.
SEED = 20261031
N_BOOT = 2000
N_PERMUTATIONS = 10_000
CI_LEVEL = 0.95


@dataclass(frozen=True)
class Variant:
    """One geometry: how the features are prepared, and which distance compares them."""

    name: str
    #: ``"as-run"`` (CRoMa's own normalisation), ``"unit"``, ``"raw"`` or ``"whitened"``.
    features: str
    distance: str
    #: Variance share the whitening retains; ``None`` for the unwhitened variants.
    variance: float | None = None


COSINE = Variant("cosine", "as-run", "cosine")
VARIANTS = (
    COSINE,
    Variant("normalised-euclidean", "unit", "euclidean"),
    Variant("euclidean", "raw", "euclidean"),
    Variant("whitened", "whitened", "euclidean", 0.95),
)
#: Whitening-threshold sensitivity, on Camelyon only.
SENSITIVITY_VARIANTS = {
    "pathorob-camelyon": (
        Variant("whitened-90", "whitened", "euclidean", 0.90),
        Variant("whitened-99", "whitened", "euclidean", 0.99),
    ),
}


def variants_for(benchmark: str) -> tuple[Variant, ...]:
    return VARIANTS + SENSITIVITY_VARIANTS.get(benchmark, ())


def variant(name: str) -> Variant:
    for v in VARIANTS + tuple(v for vs in SENSITIVITY_VARIANTS.values() for v in vs):
        if v.name == name:
            return v
    raise KeyError(name)


#: GEMM kernels (numpy's OpenBLAS core type) a published run may have been computed with.
CANDIDATE_KERNELS = ("Cooperlake", "SkylakeX")
SINGLE_THREAD = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
#: A unit worker's exit code when the cosine reference did not reproduce under its kernel.
NOT_REPRODUCED = 3


class IncompleteStudyError(RuntimeError):
    """The study is missing a cell, or a reference that did not reproduce."""


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Whitened:
    features: np.ndarray
    #: Retained leading components.
    n_components: int
    #: Share of the total variance they explain (at least the requested share).
    explained: float


def whiten(features: np.ndarray, variance: float) -> Whitened:
    """PCA-whiten ``features``: the fewest leading components explaining ``>= variance``.

    The covariance (``ddof=1``) is fit on the rows given, in float64, and uses nothing but
    the features. Each retained component is scaled to unit variance, so Euclidean distance
    on the result is Mahalanobis distance under the truncated covariance.
    """
    if not 0.0 < float(variance) < 1.0:
        raise ValueError(f"variance must lie in (0, 1), got {variance}")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("whitening needs a 2-D array with at least two rows")
    mean = x.mean(axis=0)
    centred = x - mean
    covariance = centred.T @ centred / (x.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues[::-1], 0.0, None)
    eigenvectors = eigenvectors[:, ::-1]
    share = np.cumsum(eigenvalues) / eigenvalues.sum()
    k = min(int(np.searchsorted(share, float(variance), side="left")) + 1, len(eigenvalues))
    basis = eigenvectors[:, :k] / np.sqrt(eigenvalues[:k])
    return Whitened(features=centred @ basis, n_components=k, explained=float(share[k - 1]))


def unit_rows(features: np.ndarray) -> np.ndarray:
    """L2-normalised rows, by the expression CRoMa's cosine path uses."""
    return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)


def prepare(features: np.ndarray, rows: np.ndarray, v: Variant) -> tuple[np.ndarray, dict]:
    """The features ``v`` scores, and what its preparation recorded.

    ``rows[i]`` names the tile behind evaluated row ``i``; whitening is fit on each tile
    once, so a tile that occurs in several TCGA-2x2 subsets is not weighted by its count.
    """
    if v.features in ("as-run", "raw"):
        return features, {}
    if v.features == "unit":
        return unit_rows(features), {}
    if v.features != "whitened":
        raise ValueError(f"unknown feature preparation {v.features!r}")
    _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
    w = whiten(features[first], float(v.variance))
    record = {
        "input_dim": int(features.shape[1]),
        "n_components": int(w.n_components),
        "explained_variance": float(w.explained),
        "whitening_rows": int(first.size),
    }
    return w.features[np.asarray(inverse).reshape(-1)], record


# --------------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class Search:
    """The run's CRoMa search, replayed as stored in its ``run_config.json``."""

    m_max: int
    start_k: int
    k_growth_factor: float
    alpha: float

    @classmethod
    def from_run_config(cls, path: Path) -> "Search":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        replay, resolved = config["replay"], config["resolved"]
        if int(resolved["croma_headline_m"]) != CROMA_HEADLINE_M:
            raise ValueError(f"{path}: headline m {resolved['croma_headline_m']}")
        return cls(
            m_max=int(replay["croma_m_max"]),
            start_k=int(replay["croma_start_k"]),
            k_growth_factor=float(replay["croma_k_growth_factor"]),
            alpha=float(replay["croma_alpha"]),
        )


@dataclass(frozen=True)
class Run:
    """One benchmark run the study re-scores, and where its stored scores live."""

    benchmark: str
    design: str
    results_dir: Path
    search: Search

    @property
    def metrics_csv(self) -> Path:
        return self.results_dir / "metrics.csv"

    @property
    def run_config(self) -> Path:
        return self.results_dir / "run_config.json"

    def per_sample_csv(self, model: str) -> Path:
        # The driver's own path rule (``Path.with_suffix``), quirks included: CONCHv1.5's
        # file is ``CONCHv1.csv``.
        base = self.results_dir / "per_sample_metrics_by_model" / safe_model_name(model)
        return base.with_suffix(".csv")

    def roster(self) -> list[str]:
        """The run's models: whatever its ``metrics.csv`` holds, never a declared list."""
        return pd.read_csv(self.metrics_csv, usecols=["model"])["model"].astype(str).tolist()


def production_run(benchmark: str) -> Run:
    """The run the paper reports for ``benchmark``, resolved through ``paper_manifest``."""
    sys.path.insert(0, str(REPO / "scripts" / "repro"))
    import benchmarks
    from paper_manifest import by_benchmark

    entry = by_benchmark(benchmark)
    results_dir = REPO / entry.run_rel / "results"
    return Run(
        benchmark=benchmark,
        design=benchmarks.get(benchmark).design,
        results_dir=results_dir,
        search=Search.from_run_config(results_dir / "run_config.json"),
    )


def score(features: np.ndarray, manifest: pd.DataFrame, *, run: Run, distance: str):
    """CRoMa at the headline radius, with the run's design and search."""
    results = CRoMa.compute(
        features,
        manifest,
        confounder_column="confounder",
        evaluation_design=run.design,
        m=list(range(1, run.search.m_max + 1)),
        alpha=run.search.alpha,
        start_k=run.search.start_k,
        k_growth_factor=run.search.k_growth_factor,
        distance=distance,
    )
    return results[CROMA_HEADLINE_M]


def per_sample_frame(result, manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-unit values keyed by ``subset`` and ``sample_id``, in occurrence order."""
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    return pd.DataFrame(
        {
            "subset": np.asarray(result.occurrence_subsets).astype(str),
            "sample_id": sample_ids[np.asarray(result.occurrence_source_indices)],
            "croma": np.asarray(result.sample_values_aligned, dtype=float),
        }
    )


KEY = ["subset", "sample_id"]


def _aligned(left: pd.DataFrame, right: pd.DataFrame, column: str) -> tuple[np.ndarray, ...]:
    """``left.croma`` and ``right[column]`` on the same units, or an error if they differ."""
    for frame, name in ((left, "left"), (right, "right")):
        if frame.duplicated(KEY).any():
            raise ValueError(f"{name} per-sample frame repeats a (subset, sample_id) key")
    joined = left[KEY + ["croma"]].merge(
        right[KEY + [column]], on=KEY, how="outer", suffixes=("", "_right"), indicator=True
    )
    if (joined["_merge"] != "both").any():
        raise ValueError("per-sample frames cover different units")
    other = column if column != "croma" else "croma_right"
    return joined["croma"].to_numpy(float), joined[other].to_numpy(float)


def reproduction(result, frame: pd.DataFrame, run: Run, model: str) -> dict:
    """How the cosine re-score compares with the run's stored values, exactly."""
    reference = pd.read_csv(
        run.per_sample_csv(model),
        usecols=KEY + ["croma_m5"],
        dtype={"subset": str, "sample_id": str},
        float_precision="round_trip",
    )
    got, want = _aligned(frame, reference, "croma_m5")
    metrics = pd.read_csv(run.metrics_csv, float_precision="round_trip").set_index("model")
    row = metrics.loc[model]
    pooled = {
        "median": [float(result.value), float(row["croma"])],
        "f0": [float(result.f0), float(row["croma_f0"])],
        "ltm10": [float(result.ltm_alpha), float(row["croma_ltm_alpha"])],
    }
    return {
        "reference": str(run.per_sample_csv(model)),
        "reference_sha256": sha256(run.per_sample_csv(model)),
        "identical": bool(np.array_equal(got, want)),
        "n_mismatched": int(np.count_nonzero(got != want)),
        "max_abs_deviation": float(np.max(np.abs(got - want))) if got.size else 0.0,
        "pooled_identical": {k: a == b for k, (a, b) in pooled.items()},
        "pooled": pooled,
    }


# --------------------------------------------------------------------------- cells


def cell_dir(root: Path, benchmark: str, model: str) -> Path:
    return Path(root) / "cells" / benchmark / safe_model_name(model)


def cell_paths(root: Path, benchmark: str, model: str, name: str) -> tuple[Path, Path]:
    base = cell_dir(root, benchmark, model)
    return base / f"{name}.json", base / f"{name}.csv.gz"


def read_cell(root: Path, benchmark: str, model: str, name: str) -> dict | None:
    record, _ = cell_paths(root, benchmark, model, name)
    if not record.exists():
        return None
    return json.loads(record.read_text(encoding="utf-8"))


def read_cell_values(root: Path, benchmark: str, model: str, name: str) -> pd.DataFrame:
    _, values = cell_paths(root, benchmark, model, name)
    return pd.read_csv(
        values, dtype={"subset": str, "sample_id": str}, float_precision="round_trip"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _json(record: dict) -> str:
    return json.dumps(record, indent=2, sort_keys=True) + "\n"


def write_cell(root: Path, record: dict, frame: pd.DataFrame) -> None:
    """The per-sample values, then the record that marks the cell finished."""
    record_path, values_path = cell_paths(
        root, record["benchmark"], record["model"], record["variant"]
    )
    # mtime=0 keeps the gzip stream a function of the values alone.
    payload = gzip.compress(frame.to_csv(index=False).encode("utf-8"), mtime=0)
    _atomic_bytes(values_path, payload)
    record = {**record, "values": values_path.name, "values_sha256": sha256(values_path)}
    _atomic_text(record_path, _json(record))


def numpy_kernel() -> str | None:
    """The core type numpy's OpenBLAS runs, which computes the embedding distances."""
    from threadpoolctl import threadpool_info

    for info in threadpool_info():
        if info.get("internal_api") == "openblas" and info.get("prefix") == "libopenblas":
            return str(info.get("architecture"))
    return None


def run_unit(
    root: Path,
    run: Run,
    model: str,
    *,
    features: np.ndarray,
    rows: np.ndarray,
    manifest: pd.DataFrame,
    kernel: str | None,
    last: bool,
    embedding_sha256: str,
) -> int:
    """Every variant of one encoder x benchmark not yet on disk; cosine first.

    Returns :data:`NOT_REPRODUCED` (writing nothing but an attempt record) when the cosine
    re-score differs from the run's stored values and ``last`` is false; with ``last`` the
    cell is written with ``reproduced: false`` and the other variants still run, so the
    deviation is on record and ``summarise`` refuses.
    """
    base = {
        "benchmark": run.benchmark,
        "model": model,
        "design": run.design,
        "m": CROMA_HEADLINE_M,
        "search": asdict(run.search),
        "kernel": kernel,
        "embedding_sha256": embedding_sha256,
        "n_features": int(features.shape[1]),
    }
    cosine = read_cell(root, run.benchmark, model, COSINE.name)
    if cosine is None:
        started = time.perf_counter()
        result = score(features, manifest, run=run, distance=COSINE.distance)
        frame = per_sample_frame(result, manifest)
        check = reproduction(result, frame, run, model)
        record = {
            **base,
            **_summary_fields(result),
            "variant": COSINE.name,
            "distance": COSINE.distance,
            "seconds": round(time.perf_counter() - started, 1),
            "reproduction": check,
            "reproduced": check["identical"],
        }
        if not check["identical"] and not last:
            attempt = cell_dir(root, run.benchmark, model) / f"cosine.{kernel}.attempt.json"
            _atomic_text(attempt, _json(record))
            return NOT_REPRODUCED
        write_cell(root, record, frame)
    elif cosine.get("kernel") != kernel:
        raise RuntimeError(
            f"{run.benchmark} / {model}: cosine was scored under {cosine.get('kernel')}, "
            f"not {kernel}; every variant of an encoder shares one kernel"
        )

    for v in variants_for(run.benchmark):
        if v.name == COSINE.name or read_cell(root, run.benchmark, model, v.name) is not None:
            continue
        started = time.perf_counter()
        prepared, preparation = prepare(features, rows, v)
        result = score(prepared, manifest, run=run, distance=v.distance)
        record = {
            **base,
            **_summary_fields(result),
            **preparation,
            "variant": v.name,
            "distance": v.distance,
            "feature_preparation": v.features,
            "whitening_variance": v.variance,
            "seconds": round(time.perf_counter() - started, 1),
        }
        write_cell(root, record, per_sample_frame(result, manifest))
    return 0


def _summary_fields(result) -> dict:
    return {
        "median": float(result.value),
        "f0": float(result.f0),
        "ltm10": float(result.ltm_alpha),
        "alpha": float(result.alpha),
        "n_units": int(len(result.sample_values_aligned)),
        "n_subsets": int(result.n_pairs),
        "k_final": int(result.k_final),
        "retries": int(result.retries),
    }


# --------------------------------------------------------------------------- measures


def sign_agreement(reference: np.ndarray, other: np.ndarray) -> float:
    """Share of units on the same side of zero; a unit is confounder-dominant iff ``<= 0``.

    The closed boundary is F(0)'s, so this agrees with it on what a sign is.
    """
    reference, other = np.asarray(reference, float), np.asarray(other, float)
    return float(np.mean((reference <= 0.0) == (other <= 0.0)))


def _ranked(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["model"] != CONTROL_MODEL]


def _correlation_fields(prefix: str, inference, *, with_p: bool) -> dict:
    fields = {
        f"{prefix}": inference.rho,
        f"{prefix}_lo": inference.ci.lo,
        f"{prefix}_hi": inference.ci.hi,
    }
    if with_p:
        fields[f"{prefix}_p"] = inference.p_value
    return fields


def measures(summary: pd.DataFrame, nipd: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per benchmark x variant, everything against cosine, over the ranked encoders.

    ``summary`` holds one row per benchmark x model x variant (``median``, ``f0``,
    ``ltm10``, ``sign_agreement``); ``nipd`` one row per benchmark x model (``nipd_id``,
    ``nipd_ood``) for the cohorts that have nIPD cells.
    """
    rows = []
    for benchmark in dict.fromkeys(summary["benchmark"]):
        cells = _ranked(summary[summary["benchmark"] == benchmark])
        cosine = cells[cells["variant"] == COSINE.name].set_index("model").sort_index()
        models = cosine.index
        draws = encoder_bootstrap_draws(len(models), n_boot=N_BOOT, seed=SEED)
        targets = None
        if nipd is not None and benchmark in set(nipd["benchmark"]):
            targets = nipd[nipd["benchmark"] == benchmark].set_index("model")
            missing = sorted(set(models) - set(targets.index))
            if missing:
                raise ValueError(f"{benchmark}: no nIPD for {missing}")
            targets = targets.loc[models]

        def correlate(x, y):
            return spearman_inference(
                np.asarray(x, float),
                np.asarray(y, float),
                level=CI_LEVEL,
                seed=SEED,
                n_permutations=N_PERMUTATIONS,
                draws=draws,
            )

        for name in dict.fromkeys(cells["variant"]):
            other = cells[cells["variant"] == name].set_index("model")
            if sorted(other.index) != sorted(models):
                raise ValueError(f"{benchmark} / {name}: encoders differ from cosine's")
            other = other.loc[models]
            row = {"benchmark": benchmark, "variant": name, "n_encoders": int(len(models))}
            if name != COSINE.name:
                row.update(
                    _correlation_fields(
                        "rank_rho_median",
                        correlate(cosine["median"], other["median"]),
                        with_p=False,
                    )
                )
                row.update(
                    _correlation_fields(
                        "rank_rho_ltm10", correlate(cosine["ltm10"], other["ltm10"]), with_p=False
                    )
                )
                row["sign_changes"] = int(
                    np.count_nonzero((cosine["median"] <= 0.0) != (other["median"] <= 0.0))
                )
                row["median_abs_delta_f0"] = float(np.median(np.abs(other["f0"] - cosine["f0"])))
                row["median_sign_agreement"] = float(np.median(other["sign_agreement"]))
            if "n_components" in other and other["n_components"].notna().all():
                row["dims_median"] = float(np.median(other["n_components"]))
                row["dims_min"] = int(other["n_components"].min())
                row["dims_max"] = int(other["n_components"].max())
            if targets is not None:
                for target in NIPD_TARGETS:
                    row.update(
                        _correlation_fields(
                            f"rho_{target}",
                            correlate(other["median"], targets[target]),
                            with_p=True,
                        )
                    )
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- orchestration


def design(runs: dict[str, Run]) -> dict:
    """What a complete study is: every (benchmark, encoder, variant) cell, and how scored."""
    return {
        "m": CROMA_HEADLINE_M,
        "seed": SEED,
        "n_boot": N_BOOT,
        "n_permutations": N_PERMUTATIONS,
        "ci_level": CI_LEVEL,
        "candidate_kernels": list(CANDIDATE_KERNELS),
        "benchmarks": {
            b: {
                "design": run.design,
                "results_dir": _rel(run.results_dir),
                "search": asdict(run.search),
                "models": run.roster(),
                "variants": [asdict(v) for v in variants_for(b)],
            }
            for b, run in runs.items()
        },
    }


def _rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def write_design(root: Path, record: dict) -> None:
    path = Path(root) / DESIGN
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != record:
            raise RuntimeError(
                f"{path} describes a different study (roster, search or variants changed); "
                "move the old study aside before starting a new one"
            )
        return
    _atomic_text(path, _json(record))


def units(record: dict) -> list[tuple[str, str]]:
    return [(b, m) for b, spec in record["benchmarks"].items() for m in spec["models"]]


def unit_finished(root: Path, record: dict, benchmark: str, model: str) -> bool:
    names = [v["name"] for v in record["benchmarks"][benchmark]["variants"]]
    return all(read_cell(root, benchmark, model, n) is not None for n in names)


def status(root: Path, record: dict) -> pd.DataFrame:
    rows = []
    for benchmark, spec in record["benchmarks"].items():
        names = [v["name"] for v in spec["variants"]]
        cells = [read_cell(root, benchmark, m, n) for m in spec["models"] for n in names]
        cosine = [read_cell(root, benchmark, m, COSINE.name) for m in spec["models"]]
        rows.append(
            {
                "benchmark": benchmark,
                "expected": len(cells),
                "finished": sum(c is not None for c in cells),
                "cosine_reproduced": sum(bool(c and c["reproduced"]) for c in cosine),
                "cosine_not_reproduced": sum(bool(c and not c["reproduced"]) for c in cosine),
            }
        )
    return pd.DataFrame(rows)


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sweep(root: Path, record: dict, *, workers: int) -> int:
    """Run every unfinished unit, ``workers`` processes at a time, each on one BLAS thread.

    A unit whose cosine re-score does not reproduce under one kernel is re-queued under the
    next; the kernel that last reproduced a benchmark is tried first for its other encoders.
    """
    root = Path(root)
    queue: list[tuple[str, str, list[str]]] = []
    for benchmark, model in units(record):
        if unit_finished(root, record, benchmark, model):
            continue
        cosine = read_cell(root, benchmark, model, COSINE.name)
        kernels = [cosine["kernel"]] if cosine else []
        queue.append((benchmark, model, kernels))
    preferred: dict[str, str] = {}
    running: dict[subprocess.Popen, tuple[str, str, list[str], str]] = {}
    failures: list[str] = []
    _log(f"sweep: {len(queue)} units to run on {workers} workers in {root}")
    while queue or running:
        while queue and len(running) < workers:
            benchmark, model, kernels = queue.pop(0)
            if not kernels:
                first = preferred.get(benchmark, CANDIDATE_KERNELS[0])
                kernels = [first] + [k for k in CANDIDATE_KERNELS if k != first]
            kernel, rest = kernels[0], kernels[1:]
            log_path = root / "logs" / benchmark / f"{safe_model_name(model)}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            argv = [sys.executable, str(Path(__file__).resolve())]
            argv += ["unit", "--root", str(root), "--benchmark", benchmark, "--model", model]
            argv += ["--kernel", kernel] + ([] if rest else ["--last"])
            env = {**os.environ, **SINGLE_THREAD, "OPENBLAS_CORETYPE": kernel}
            with open(log_path, "ab") as log:
                proc = subprocess.Popen(argv, env=env, cwd=REPO, stdout=log, stderr=log)
            running[proc] = (benchmark, model, rest, kernel)
            _log(f"start {benchmark} / {model} under {kernel}")
        time.sleep(2.0)
        for proc in [p for p in running if p.poll() is not None]:
            benchmark, model, rest, kernel = running.pop(proc)
            if proc.returncode == 0:
                cosine = read_cell(root, benchmark, model, COSINE.name) or {}
                if cosine.get("reproduced"):
                    preferred[benchmark] = kernel
                tag = "reproduced" if cosine.get("reproduced") else "NOT REPRODUCED"
                _log(f"done  {benchmark} / {model} under {kernel} (cosine {tag})")
            elif proc.returncode == NOT_REPRODUCED and rest:
                _log(f"retry {benchmark} / {model}: cosine differs under {kernel}")
                queue.insert(0, (benchmark, model, rest))
            else:
                failures.append(f"{benchmark} / {model}")
                _log(f"FAIL  {benchmark} / {model} under {kernel}: exit {proc.returncode}")
    if failures:
        _log(f"sweep finished with {len(failures)} failed units: {failures}")
        return 1
    _log("sweep finished; summarising")
    summarise(root, record)
    _log("summary written")
    return 0


def summarise(
    root: Path,
    record: dict,
    *,
    apd_csv: Path = APD_CSV,
    apd_correlation_csv: Path | None = APD_CORRELATION_CSV,
) -> pd.DataFrame:
    """``summary.csv``, ``measures.csv`` and ``provenance.json`` from a complete study.

    Refuses on a missing cell and on any cosine reference that did not reproduce. When
    ``apd_correlation_csv`` is given, cosine's nIPD correlations must equal the ones
    ``tab:apd-correlation`` reports, so both tables read the same association.
    """
    root = Path(root)
    missing = [
        f"{b} / {m} / {v['name']}"
        for b, spec in record["benchmarks"].items()
        for m in spec["models"]
        for v in spec["variants"]
        if read_cell(root, b, m, v["name"]) is None
    ]
    if missing:
        raise IncompleteStudyError(f"{len(missing)} cells missing, e.g. {missing[:5]}")
    rows, inputs = [], {}
    for benchmark, spec in record["benchmarks"].items():
        results_dir = REPO / spec["results_dir"]
        for name in ("metrics.csv", "run_config.json"):
            inputs[_rel(results_dir / name)] = sha256(results_dir / name)
        for model in spec["models"]:
            cosine = read_cell(root, benchmark, model, COSINE.name)
            if not cosine["reproduced"]:
                raise IncompleteStudyError(
                    f"{benchmark} / {model}: the cosine re-score does not reproduce the run "
                    f"({cosine['reproduction']['n_mismatched']} units differ)"
                )
            inputs[_rel(Path(cosine["reproduction"]["reference"]))] = cosine["reproduction"][
                "reference_sha256"
            ]
            inputs[f"embeddings:{benchmark}:{model}"] = cosine["embedding_sha256"]
            reference = read_cell_values(root, benchmark, model, COSINE.name)
            for v in spec["variants"]:
                cell = read_cell(root, benchmark, model, v["name"])
                values = read_cell_values(root, benchmark, model, v["name"])
                ref, other = _aligned(reference, values, "croma")
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model": model,
                        "ranked": model != CONTROL_MODEL,
                        "variant": v["name"],
                        "median": cell["median"],
                        "f0": cell["f0"],
                        "ltm10": cell["ltm10"],
                        "sign_agreement": sign_agreement(ref, other),
                        "n_units": cell["n_units"],
                        "input_dim": cell["n_features"],
                        "n_components": cell.get("n_components"),
                        "explained_variance": cell.get("explained_variance"),
                        "kernel": cell["kernel"],
                    }
                )
    summary = pd.DataFrame(rows)
    nipd = load_nipd(apd_csv)
    inputs[_rel(apd_csv)] = sha256(apd_csv)
    table = measures(summary, nipd)
    if apd_correlation_csv is not None:
        check_against_apd_correlation(table, Path(apd_correlation_csv))
        inputs[_rel(apd_correlation_csv)] = sha256(apd_correlation_csv)
    summary.to_csv(root / SUMMARY, index=False)
    table.to_csv(root / MEASURES, index=False)
    provenance = {
        "design_sha256": sha256(root / DESIGN),
        "inputs": dict(sorted(inputs.items())),
        "outputs": {SUMMARY: sha256(root / SUMMARY), MEASURES: sha256(root / MEASURES)},
    }
    _atomic_text(root / PROVENANCE, _json(provenance))
    return table


def load_nipd(path: Path = APD_CSV) -> pd.DataFrame:
    """nIPD per cohort x encoder from the APD study's cells, keyed by benchmark."""
    apd = pd.read_csv(path, float_precision="round_trip")
    by_dataset = {key: benchmark for benchmark, key in NIPD_DATASETS.items()}
    apd = apd[apd["dataset"].isin(by_dataset)].copy()
    apd["benchmark"] = apd["dataset"].map(by_dataset)
    return apd[["benchmark", "model", *NIPD_TARGETS]]


def check_against_apd_correlation(table: pd.DataFrame, path: Path) -> None:
    corr = pd.read_csv(path, float_precision="round_trip")
    for benchmark, dataset in NIPD_DATASETS.items():
        row = table[(table["benchmark"] == benchmark) & (table["variant"] == COSINE.name)]
        if row.empty:
            continue
        for target in NIPD_TARGETS:
            stored = corr[
                (corr["target"] == target) & (corr["metric"] == "croma") & (corr["scope"] == dataset)
            ]
            if len(stored) != 1:
                raise ValueError(f"{path}: no single croma/{target}/{dataset} row")
            ours, theirs = float(row[f"rho_{target}"].iloc[0]), float(stored["spearman"].iloc[0])
            if abs(ours - theirs) > 1e-12 or int(row["n_encoders"].iloc[0]) != int(stored["n"].iloc[0]):
                raise ValueError(
                    f"{benchmark}: cosine rho(CRoMa, {target}) = {ours} over "
                    f"{int(row['n_encoders'].iloc[0])} encoders, but {path} has {theirs} over "
                    f"{int(stored['n'].iloc[0])}"
                )


def _unit_main(args: argparse.Namespace) -> int:
    wanted = os.environ.get("OPENBLAS_CORETYPE", "")
    kernel = numpy_kernel()
    if wanted and kernel != wanted:
        raise RuntimeError(f"asked for {wanted}, numpy's OpenBLAS runs {kernel}")
    import layout
    from views import load_view

    run = production_run(args.benchmark)
    view = load_view(args.benchmark)
    started = time.perf_counter()
    features = view.features(args.model)
    embedding = sha256(layout.embedding_path(view.spec.tileset, args.model))
    _log(f"{args.benchmark} / {args.model}: loaded {features.shape} under {kernel}")
    code = run_unit(
        Path(args.root),
        run,
        args.model,
        features=features,
        rows=view.rows,
        manifest=view.eval_manifest,
        kernel=kernel,
        last=args.last,
        embedding_sha256=embedding,
    )
    _log(f"{args.benchmark} / {args.model}: exit {code} after {time.perf_counter() - started:.0f}s")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        name: sub.add_parser(name, help=text)
        for name, text in (
            ("sweep", "run (or resume) every unit, then summarise"),
            ("status", "count finished cells and reproduced references"),
            ("summarise", "summary, measures and provenance from a complete study"),
            ("unit", "every variant of one encoder x benchmark (a sweep worker)"),
        )
    }
    for p in commands.values():
        p.add_argument("--root", type=Path, default=STUDY_DIR)
    commands["sweep"].add_argument("--workers", type=int, default=4)
    commands["unit"].add_argument("--benchmark", required=True, choices=BENCHMARKS)
    commands["unit"].add_argument("--model", required=True)
    commands["unit"].add_argument("--kernel", required=True)
    commands["unit"].add_argument("--last", action="store_true")
    args = parser.parse_args(argv)
    args.root = args.root.absolute()

    if args.command == "unit":
        return _unit_main(args)
    record = design({b: production_run(b) for b in BENCHMARKS})
    if args.command == "status":
        table = status(args.root, record)
        print(table.to_string(index=False))
        print(f"finished {int(table['finished'].sum())} / {int(table['expected'].sum())} cells")
        return 0
    if args.command == "summarise":
        summarise(args.root, record)
        print(f"wrote {args.root / MEASURES}")
        return 0
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    args.root.mkdir(parents=True, exist_ok=True)
    write_design(args.root, record)
    return sweep(args.root, record, workers=args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
