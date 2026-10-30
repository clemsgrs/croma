"""Slide encoders versus mean-pooling of their own tile encoders, on PCaBiop.

A slide encoder is a learned aggregator over its tile encoder's embeddings. If its slide
embedding is no more robust than the plain mean of the tiles it was given, the slide-level
encoder adds nothing at the representation level that pooling would not. This study puts
each slide encoder beside that baseline, built from the *same* tiling of the *same* slides:
each slide encoder's published slide2vec config is re-run on the 1,324 PCaBiop slides
(1,000 PANDA in-domain + 324 PAR out-of-domain) with ``model.save_tile_embeddings`` on, so
one run yields both arms.

Subcommands (run in this order):

``extract``
    Re-run each pair's slide2vec config, one cohort at a time, under
    ``output/studies/slide-vs-meanpool/runs/<MODEL>/{panda,par}/``. Resumable: a cohort
    whose slides are all embedded is skipped, a partial one resumes. GPU-bound; launch it
    detached (``launch`` does that).
``status``
    Embedded / expected slides per pair and cohort.
``reduce``
    Mean-pool each slide's tile embeddings into the mean-pooled arm, compare the re-run
    slide embeddings with the published ``output/embeddings/pcabiop/<MODEL>.npy`` (per-slide
    cosine), and choose the slide arm: the published embedding when every slide reproduces
    (cosine >= ``REPRODUCTION_COSINE``), the re-run otherwise. Writes ``<MODEL>-meanpool.npy``
    (and ``<MODEL>-rerun.npy`` when the re-run is the slide arm) beside the published files,
    each with a sidecar recording its source runs, tile encoder and per-slide tile counts.
``evaluate``
    Score both arms of every reduced pair: the benchmark runner on the in-domain cohort
    (biological kNN balanced accuracy, median CRoMa, F(0), LTM_10 at the headline m) and
    the PCaBiop APD driver (nIPD_ID, nIPD_OOD), then paired differences slide - mean-pool:
    CRoMa summaries on one stratified slide bootstrap shared by every arm, nIPD with its
    spread over the paired probe replicates.

The study arms never enter the published slide panel: the benchmark runs in a study-local
output root, the APD roster for PCaBiop is the slide panel from ``model_metadata.csv``
(study arms join only when named), and the results payload reads the published runs only.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _path in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "studies" / "apd"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

#: Where the published slide2vec runs of the PCaBiop slide panel live (their config.yaml
#: is the record each re-run starts from).
FEATURES_ROOT = Path("/data/pathology/projects/clement/croma/wsi/features")

COHORTS = ("panda", "par")

#: Slide reader per cohort, as in the published runs: PANDA's TIFFs and masks are read
#: with ASAP, PAR's Leica SVS files and masks with OpenSlide. ``mask_backend`` is pinned
#: rather than left on ``auto``, which selects cuCIM for the PANDA masks and crashes.
COHORT_BACKEND = {"panda": "asap", "par": "openslide"}

#: Worker counts for the re-runs: the GPU is shared and the CPU budget is three workers.
NUM_WORKERS = 3


@dataclass(frozen=True)
class Pair:
    """One slide encoder and the tile encoder it consumes."""

    #: Published slide-panel name (``output/embeddings/pcabiop/<slide_model>.npy``).
    slide_model: str
    #: slide2vec tile encoder name and output variant the slide encoder is fed.
    tile_encoder: str
    tile_variant: str
    tile_dim: int
    #: Published slide2vec run ids behind the PANDA (in-domain) and PAR (OOD) rows.
    panda_run: str
    par_run: str
    #: Whether the PANDA run's config is in a layout the installed slide2vec can replay.
    #: Older PANDA runs predate the current config schema (and, for TITAN, used another
    #: tile size); their re-run starts from the pair's PAR config with the PANDA reader.
    panda_config_replayable: bool
    required: bool
    #: Where the pairing is stated by the slide encoder's authors.
    pairing_source: str


PAIRS: tuple[Pair, ...] = (
    Pair(
        "TITAN", "conchv15", "default", 768, "f3w2t2c1", "n02n1k2t", False, True,
        "MahmoodLab/TITAN model card: TITAN is trained on CONCH v1.5 patch features "
        "(512x512 px at 20x); CONCH v1.5 is loaded via titan.return_conch().",
    ),
    Pair(
        "MOOZY", "lunit", "default", 384, "65l8nrj8", "nchyscec", True, True,
        "AtlasAnalyticsLab/MOOZY model card: patch encoder ViT-S/8 (Lunit DINO), "
        "lunit_vit_small_patch8_dino at 224x224, 384-d.",
    ),
    Pair(
        "PRISM2", "virchow2", "cls", 1280, "t90fp3tk", "1evxzd6h", True, True,
        "paige-ai/Prism2 model card: PRISM2 requires Virchow2 tile embeddings, "
        "(N, 1280) per slide, 224x224 tiles.",
    ),
    Pair(
        "PRISM", "virchow", "cls_patch_mean", 2560, "uam66c2c", "552fd85i", False, False,
        "paige-ai/Prism model card: tile image encoder Virchow V1.",
    ),
    Pair(
        "Prov-GigaPath", "gigapath", "default", 1536, "vaqva1cq", "aetsexkb", False, False,
        "prov-gigapath/prov-gigapath model card: tile encoder embeddings plus coordinates "
        "are passed to the slide encoder.",
    ),
)

PAIRS_BY_MODEL = {pair.slide_model: pair for pair in PAIRS}


def _migrate_retired_tiling_keys(tiling: dict) -> None:
    """Map the tiling keys hs2p retired onto their replacements, in place."""
    masks = tiling.setdefault("masks", {})
    if "independent_sampling" in tiling:
        masks["strategy"] = "independent" if tiling.pop("independent_sampling") else "joint"
    params = tiling.setdefault("params", {})
    if "tissue_threshold" in params:
        coverage = masks.setdefault("min_coverage", {"background": None})
        coverage["tissue"] = params.pop("tissue_threshold")
    seg = tiling.setdefault("seg_params", {})
    # ``null`` used to mean "the default method"; it is no longer accepted. The method is
    # ignored anyway when, as here, every slide comes with a precomputed tissue mask.
    if seg.get("method") is None:
        seg["method"] = "hsv"


def extraction_config(
    source_config: dict,
    *,
    cohort: str,
    slide_csv: Path,
    run_root: Path,
) -> dict:
    """The slide2vec config that re-runs ``source_config`` on one PCaBiop cohort.

    Everything that decides the embeddings (encoder, tile size, spacing, tissue coverage)
    is the published run's; what changes is where it reads and writes, that it also saves
    the tile embeddings, that it resumes, and that it reports nowhere. The output lands in
    ``run_root/<cohort>/``.
    """
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}, got {cohort!r}")
    cfg = copy.deepcopy(source_config)
    cfg["csv"] = str(slide_csv)
    cfg["output_dir"] = str(run_root)
    cfg["resume"] = True
    cfg["resume_dirname"] = cohort
    cfg.setdefault("model", {})["save_tile_embeddings"] = True
    tiling = cfg.setdefault("tiling", {})
    _migrate_retired_tiling_keys(tiling)
    tiling["backend"] = COHORT_BACKEND[cohort]
    tiling["mask_backend"] = COHORT_BACKEND[cohort]
    speed = cfg.setdefault("speed", {})
    speed["num_dataloader_workers"] = NUM_WORKERS
    speed["num_preprocessing_workers"] = NUM_WORKERS
    cfg.setdefault("wandb", {})["enable"] = False
    return cfg


# ---------------------------------------------------------------------------
# extract: re-run each pair's slide2vec config on PCaBiop
# ---------------------------------------------------------------------------

WSI_ROOT = Path("/data/pathology/projects/clement/croma/wsi")

#: slide2vec slide lists (sample_id, image_path, mask_path) the published runs read.
SLIDE_SOURCES = {
    "panda": WSI_ROOT / "panda-slide2vec-new-format.csv",
    "par": WSI_ROOT / "par-s2v.csv",
}

#: Environment defaults for the re-runs. Remote-code encoders copy their modules into
#: HF_MODULES_CACHE, which fails on the network share, so it lives on local disk.
EXTRACTION_ENV_DEFAULTS = {
    "HF_MODULES_CACHE": "/var/tmp/hf-modules",
    "HF_TOKEN_PATH": str(Path.home() / ".cache" / "huggingface" / "token"),
}


def _cohort_rows(rows: pd.DataFrame, cohort: str) -> pd.DataFrame:
    return rows[rows["subset"].map(SUBSET_COHORT) == cohort]


def slide_list(rows: pd.DataFrame, cohort: str, source: pd.DataFrame) -> pd.DataFrame:
    """The slide2vec slide list for one cohort: its pcabiop slides, in pcabiop row order."""
    wanted = _cohort_rows(rows, cohort)["sample_id"].astype(str).tolist()
    indexed = source.assign(sample_id=source["sample_id"].astype(str)).set_index("sample_id")
    missing = [sid for sid in wanted if sid not in indexed.index]
    if missing:
        raise ValueError(f"{cohort}: {len(missing)} slide(s) absent from the slide list: {missing[:5]}")
    return indexed.loc[wanted, ["image_path", "mask_path"]].reset_index()


def cohort_progress(cohort_dir: Path, sample_ids: list[str]) -> tuple[int, int]:
    """(slides with both a tile and a slide embedding, slides expected)."""

    def _has(kind: str, sid: str) -> bool:
        directory = cohort_dir / kind
        return (directory / f"{sid}.pt").exists() or (directory / f"{sid}.npy").exists()

    done = sum(_has("tile_embeddings", s) and _has("slide_embeddings", s) for s in sample_ids)
    return int(done), len(sample_ids)


def source_config_path(pair: Pair, cohort: str) -> Path:
    """The published run config a cohort's re-run starts from."""
    if cohort == "panda" and pair.panda_config_replayable:
        return FEATURES_ROOT / pair.panda_run / "config.yaml"
    return FEATURES_ROOT / pair.par_run / "config.yaml"


def extract_pair(
    pair: Pair,
    *,
    rows: pd.DataFrame,
    study_dir: Path,
    limit: int | None = None,
) -> None:
    """Re-run one pair on both cohorts, sequentially; finished cohorts are skipped."""
    import subprocess

    import yaml

    env = dict(os.environ)
    for key, value in EXTRACTION_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    Path(env["HF_MODULES_CACHE"]).mkdir(parents=True, exist_ok=True)

    run_root = study_dir / "runs" / pair.slide_model
    run_root.mkdir(parents=True, exist_ok=True)
    for cohort in COHORTS:
        cohort_rows = _cohort_rows(rows, cohort)
        if limit is not None:
            cohort_rows = cohort_rows.head(limit)
        sample_ids = cohort_rows["sample_id"].astype(str).tolist()
        done, total = cohort_progress(run_root / cohort, sample_ids)
        if done == total:
            print(f"[extract] {pair.slide_model}/{cohort}: {done}/{total} done, skipping", flush=True)
            continue
        slide_csv = run_root / f"{cohort}.csv"
        slide_list(cohort_rows, cohort, pd.read_csv(SLIDE_SOURCES[cohort])).to_csv(
            slide_csv, index=False
        )
        source = source_config_path(pair, cohort)
        cfg = extraction_config(
            yaml.safe_load(source.read_text()),
            cohort=cohort,
            slide_csv=slide_csv,
            run_root=run_root,
        )
        cfg_path = run_root / f"{cohort}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(
            f"[extract] {pair.slide_model}/{cohort}: {done}/{total} done; running slide2vec "
            f"from {source}",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "-m", "slide2vec", str(cfg_path)], env=env, cwd=run_root, check=True
        )
        done, total = cohort_progress(run_root / cohort, sample_ids)
        if done != total:
            raise RuntimeError(
                f"{pair.slide_model}/{cohort}: slide2vec finished with {done}/{total} slides "
                f"embedded; see {run_root / cohort / 'process_list.csv'}"
            )
        print(f"[extract] {pair.slide_model}/{cohort}: {done}/{total} done", flush=True)


# ---------------------------------------------------------------------------
# reduce: two arms from one run
# ---------------------------------------------------------------------------

#: Per-slide cosine between a re-run slide embedding and the published one at or above
#: which the slide counts as reproduced. A pair keeps its published slide arm only when
#: every slide reproduces; otherwise its re-run slide embeddings are the slide arm, so the
#: two arms always come from the same tiling.
REPRODUCTION_COSINE = 0.999

TILESET = "pcabiop"
STUDY_NAME = "slide-vs-meanpool"
MEANPOOL_SUFFIX = "-meanpool"
RERUN_SUFFIX = "-rerun"

#: pcabiop ``subset`` -> the cohort (and re-run directory) its rows come from.
SUBSET_COHORT = {"ID": "panda", "OOD": "par"}


def _load_embedding(directory: Path, sample_id: str) -> np.ndarray:
    """One saved embedding: slide2vec writes ``.pt``; ``.npy`` is accepted too."""
    pt = directory / f"{sample_id}.pt"
    if pt.exists():
        import torch  # only the real re-runs need torch; the study logic does not

        return torch.load(pt, map_location="cpu").float().numpy()
    npy = directory / f"{sample_id}.npy"
    if npy.exists():
        return np.load(npy)
    raise FileNotFoundError(f"no embedding for {sample_id} under {directory}")


def _check_tile_encoder(pair: Pair, directory: Path, sample_id: str) -> None:
    """Refuse a run whose tiles were not encoded by the pair's tile encoder."""
    meta = directory / f"{sample_id}.meta.json"
    if not meta.exists():
        return
    compat = json.loads(meta.read_text()).get("compatibility", {})
    found = (compat.get("tile_encoder"), compat.get("tile_encoder_output_variant"))
    if found != (pair.tile_encoder, pair.tile_variant):
        raise ValueError(
            f"{pair.slide_model}: {meta} was encoded by {found}, "
            f"expected {(pair.tile_encoder, pair.tile_variant)}"
        )


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.einsum("ij,ij->i", a, b) / np.where(norms > 0, norms, np.nan)


def _write_new(path: Path, matrix: np.ndarray, sidecar: dict) -> None:
    """Write a study embedding and its sidecar; an identical file is left alone.

    A file that exists with other content is someone else's result (or a stale one) and is
    never replaced: move it aside first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    (path.parent / f"{path.name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")


def _conflicts(path: Path, matrix: np.ndarray) -> bool:
    if not path.exists():
        return False
    existing = np.load(path)
    return existing.shape != matrix.shape or not np.array_equal(existing, matrix)


def reduce_pair(
    pair: Pair,
    *,
    rows: pd.DataFrame,
    run_root: Path,
    embeddings_dir: Path,
    study_dir: Path,
    published_dir: Path | None = None,
) -> dict:
    """Build the pair's mean-pooled arm and choose its slide arm from one re-run.

    ``rows`` is the pcabiop row-order contract (``sample_id``, ``subset``), indexed by
    pcabiop row number; every matrix written follows it. The mean-pooled arm is the
    unweighted mean of each slide's tile embeddings. The re-run's slide embeddings are
    compared, slide by slide, with the published ``<slide_model>.npy`` (read from
    ``published_dir``, by default ``embeddings_dir``); the per-slide cosines go to
    ``study_dir/reproduction/<slide_model>.csv`` and the returned record says which slide
    arm the pair uses. Arms are written to ``embeddings_dir``.
    """
    sample_ids = rows["sample_id"].astype(str).tolist()
    cohorts = [SUBSET_COHORT[s] for s in rows["subset"]]
    unfinished = []
    for cohort in COHORTS:
        ids = [s for s, c in zip(sample_ids, cohorts) if c == cohort]
        done, total = cohort_progress(run_root / cohort, ids)
        if done != total:
            unfinished.append(f"{pair.slide_model}/{cohort}: {done}/{total} slides embedded")
    if unfinished:
        raise RuntimeError(
            "; ".join(unfinished) + f" under {run_root}. The GPU re-extraction is a "
            f"prerequisite: run `python scripts/studies/slide_vs_meanpool.py extract --pairs "
            f"{pair.slide_model}` (hours; launch it detached) before reduce."
        )
    pooled, slides, n_tiles = [], [], []
    for sid, cohort in zip(sample_ids, cohorts):
        tile_dir = run_root / cohort / "tile_embeddings"
        _check_tile_encoder(pair, tile_dir, sid)
        tiles = np.asarray(_load_embedding(tile_dir, sid), dtype=np.float64)
        if tiles.ndim != 2 or len(tiles) == 0:
            raise ValueError(f"{pair.slide_model}/{sid}: expected (n_tiles, dim), got {tiles.shape}")
        pooled.append(tiles.mean(axis=0))
        n_tiles.append(int(len(tiles)))
        slide = _load_embedding(run_root / cohort / "slide_embeddings", sid)
        slides.append(np.asarray(slide, dtype=np.float64).reshape(-1))
    meanpool = np.stack(pooled).astype(np.float32)
    rerun = np.stack(slides).astype(np.float32)

    published = np.load((published_dir or embeddings_dir) / f"{pair.slide_model}.npy")
    published = published[rows.index.to_numpy()]
    if published.shape != rerun.shape:
        raise ValueError(
            f"{pair.slide_model}: re-run slide matrix {rerun.shape} does not match the "
            f"published {published.shape}"
        )
    cosine = _rowwise_cosine(rerun, published)
    reproduced_mask = cosine >= REPRODUCTION_COSINE
    reproduced = bool(reproduced_mask.all())

    meanpool_arm = pair.slide_model + MEANPOOL_SUFFIX
    slide_arm = pair.slide_model if reproduced else pair.slide_model + RERUN_SUFFIX
    writes = [(embeddings_dir / f"{meanpool_arm}.npy", meanpool, "meanpool")]
    if not reproduced:
        writes.append((embeddings_dir / f"{slide_arm}.npy", rerun, "slide-rerun"))
    for path, matrix, _ in writes:  # check every target before writing any
        if _conflicts(path, matrix):
            raise FileExistsError(
                f"{path} exists with different content; move it aside to a dated sibling "
                "before re-reducing"
            )

    n_id = sum(c == "panda" for c in cohorts)
    source_runs = {cohort: str(run_root / cohort) for cohort in COHORTS}
    for path, matrix, arm in writes:
        sidecar = {
            "tileset": TILESET,
            "model": path.stem,
            "dim": int(matrix.shape[1]),
            "n_id": int(n_id),
            "n_ood": int(len(cohorts) - n_id),
            "study": STUDY_NAME,
            "arm": arm,
            "slide_encoder": pair.slide_model,
            "tile_encoder": pair.tile_encoder,
            "tile_encoder_output_variant": pair.tile_variant,
            "source_runs": source_runs,
            "n_tiles": n_tiles,
        }
        if arm == "meanpool":
            sidecar["pooling"] = "mean"
        _write_new(path, matrix, sidecar)

    per_slide = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "subset": rows["subset"].tolist(),
            "cosine": cosine,
            "reproduced": reproduced_mask,
            "n_tiles": n_tiles,
        }
    )
    out = study_dir / "reproduction" / f"{pair.slide_model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    per_slide.to_csv(out, index=False)

    return {
        "slide_model": pair.slide_model,
        "slide_arm": slide_arm,
        "meanpool_arm": meanpool_arm,
        "reproduced": reproduced,
        "n_slides": int(len(cosine)),
        "n_reproduced": int(reproduced_mask.sum()),
        "min_cosine": float(np.nanmin(cosine)),
        "median_cosine": float(np.nanmedian(cosine)),
        "threshold": REPRODUCTION_COSINE,
        "tile_encoder": pair.tile_encoder,
        "tile_variant": pair.tile_variant,
        "source_runs": source_runs,
    }


# ---------------------------------------------------------------------------
# evaluate: paired differences, slide - mean-pool
# ---------------------------------------------------------------------------

#: Seed and size of the stratified slide bootstrap shared by every arm.
BOOTSTRAP_SEED = 12345
BOOTSTRAP_N = 2000
BOOTSTRAP_LEVEL = 0.95
TAIL_ALPHA = 0.10

#: Paired-difference rows, in table order. The CRoMa summaries map onto the bootstrap's
#: statistics; the nIPD rows come from the probe replicates.
CROMA_STATISTICS = {"median_croma": "median", "f0": "f0", "ltm": "ltm"}
NIPD_STATISTICS = {"nipd_id": "id_test_accuracies", "nipd_ood": "ood_test_accuracies"}


def paired_differences(
    arms: list[tuple[str, str, str]],
    per_sample: pd.DataFrame,
    apd_cells: dict[str, dict],
    *,
    cramers_v,
    chance: float,
    alpha: float = TAIL_ALPHA,
    n_boot: int = BOOTSTRAP_N,
    level: float = BOOTSTRAP_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Slide-minus-mean-pool differences for every pair in ``arms``.

    ``arms`` lists ``(pair, slide_arm, meanpool_arm)``. ``per_sample`` holds each arm's
    in-domain per-sample CRoMa (``model``, ``sample_id``, ``group_id``, ``label``,
    ``confounder``, ``croma``). Median CRoMa, ``F(0)`` and ``LTM_alpha`` differences get a
    percentile CI from one stratified slide bootstrap (strata = class x provider, clusters
    = ``group_id``) drawn once and applied to every arm, so each replicate compares the two
    arms of a pair -- and every pair -- on the same resampled slides.

    ``apd_cells`` maps each arm to its APD accuracy matrices (``(n_splits, n_replicates)``).
    The probe replicates are paired across arms (same seeds, same splits), so each nIPD
    difference is reported with the standard deviation of its per-replicate differences:
    descriptive spread, not an interval.
    """
    from croma import nipd
    from croma.metrics.bootstrap import shared_draw_tail_statistics

    models = list(dict.fromkeys(m for _, slide, meanpool in arms for m in (slide, meanpool)))
    frame = per_sample[per_sample["model"].isin(models)]
    values = frame.pivot(index="sample_id", columns="model", values="croma")
    if values.isna().any().any() or set(values.columns) != set(models):
        raise ValueError("every arm needs a per-sample CRoMa value for the same slides")
    design = frame.drop_duplicates("sample_id").set_index("sample_id").loc[values.index]
    strata = (design["label"].astype(str) + "|" + design["confounder"].astype(str)).to_numpy()
    boot = shared_draw_tail_statistics(
        {m: values[m].to_numpy(dtype=float) for m in models},
        design["group_id"].astype(str).to_numpy(),
        strata=strata,
        alpha=alpha,
        n_boot=n_boot,
        level=level,
        seed=seed,
    )

    rows = []
    for pair, slide, meanpool in arms:
        for statistic, key in CROMA_STATISTICS.items():
            ci = boot.difference_ci(key, slide, meanpool)
            rows.append(
                dict(pair=pair, statistic=statistic, point=ci.point, lo=ci.lo, hi=ci.hi,
                     spread=np.nan, n=int(n_boot))
            )
        for statistic, key in NIPD_STATISTICS.items():
            a = np.asarray(apd_cells[slide][key], dtype=float)
            b = np.asarray(apd_cells[meanpool][key], dtype=float)
            if a.shape != b.shape:
                raise ValueError(f"{pair}: {key} shapes differ ({a.shape} vs {b.shape})")
            point = nipd(a, cramers_v=cramers_v, chance=chance) - nipd(
                b, cramers_v=cramers_v, chance=chance
            )
            per_replicate = np.array(
                [
                    nipd(a[:, [j]], cramers_v=cramers_v, chance=chance)
                    - nipd(b[:, [j]], cramers_v=cramers_v, chance=chance)
                    for j in range(a.shape[1])
                ]
            )
            spread = float(np.std(per_replicate, ddof=1)) if len(per_replicate) > 1 else np.nan
            rows.append(
                dict(pair=pair, statistic=statistic, point=float(point), lo=np.nan, hi=np.nan,
                     spread=spread, n=int(a.shape[1]))
            )
    return pd.DataFrame(rows)


#: The published run the representation arm replays: the slide panel's in-domain cohort.
#: Its protocol comes from the paper manifest, never from a path written here.
BENCHMARK = "panda"
BENCH_TILESET = "panda-wsi"
METRIC_COLUMNS = ("bio_knn_bacc", "croma", "croma_f0", "croma_ltm_alpha")
#: Published-arm agreement tolerance. Identical float32 inputs, rows and flags still
#: differ by up to ~5e-6 (TITAN LTM_10) from the published run: float32 distance
#: arithmetic depends on BLAS threading, and the study runs single-threaded.
PUBLISHED_ARM_ATOL = 1e-5


def _published_run():
    sys.path.insert(0, str(REPO / "scripts" / "repro"))
    from paper_manifest import by_benchmark

    return by_benchmark(BENCHMARK)


def run_representation(
    models: list[str],
    *,
    rows: pd.DataFrame,
    study_dir: Path,
    embeddings_dir: Path,
    recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The benchmark runner on the in-domain cohort, for every arm, in a study-local root.

    The arms' in-domain rows are written as a ``panda-wsi`` tileset under
    ``study_dir/bench/``, and the published run's own replay flags are passed back, so the
    only thing that differs from the published PCaBiop run is the embeddings. Returns
    ``(metrics, per_sample, benchmark_argv)``.
    """
    import benchmark
    import layout
    from run_config import read_run_config, replay_args

    entry = _published_run()
    replay = replay_args(read_run_config(REPO / entry.run_rel))
    id_rows = rows[rows["subset"] == "ID"]
    # The study tileset keeps the published tileset's row order, so the runner sees the
    # in-domain slides exactly as it did in the published run.
    tileset = pd.read_csv(layout.tileset_manifest(BENCH_TILESET))
    tileset = tileset[tileset["sample_id"].isin(set(id_rows["sample_id"]))]
    pcabiop_row = dict(zip(id_rows["sample_id"], id_rows.index))
    take = np.array([pcabiop_row[s] for s in tileset["sample_id"]])
    argv = [
        "--benchmark", BENCHMARK, "--protocol", entry.protocol, "--models", ",".join(models),
        "--progress", "off", *replay,
    ]
    if recompute:
        argv.append("--recompute-metrics")
    saved_root, saved_argv = layout.OUTPUT_ROOT, sys.argv
    layout.OUTPUT_ROOT = study_dir / "bench"
    try:
        directory = layout.embeddings_dir(BENCH_TILESET)
        directory.mkdir(parents=True, exist_ok=True)
        tileset.to_csv(layout.tileset_manifest(BENCH_TILESET), index=False)
        for model in models:
            matrix = np.load(embeddings_dir / f"{model}.npy")[take]
            np.save(directory / f"{model}.npy", matrix.astype(np.float32))
        sys.argv = ["benchmark.py", *argv]
        if benchmark.main() != 0:
            raise RuntimeError("benchmark runner failed on the study arms")
        results = layout.results_dir(entry.protocol, BENCHMARK)
    finally:
        layout.OUTPUT_ROOT, sys.argv = saved_root, saved_argv
    metrics = pd.read_csv(results / "metrics.csv")
    per_sample = pd.read_csv(results / "per_sample_metrics.csv")
    return metrics, per_sample, argv


def check_published_arms(metrics: pd.DataFrame, slide_arms: list[str]) -> None:
    """A published slide arm must score as in the published run.

    Same embeddings, same rows, same flags: a difference beyond float32 distance round-off
    (``PUBLISHED_ARM_ATOL``) means the study's representation arm is not the published
    protocol.
    """
    entry = _published_run()
    published = pd.read_csv(REPO / entry.metrics_rel).set_index("model")
    study = metrics.set_index("model")
    for arm in slide_arms:
        if arm not in published.index:
            continue
        a = study.loc[arm, list(METRIC_COLUMNS)].to_numpy(dtype=float)
        b = published.loc[arm, list(METRIC_COLUMNS)].to_numpy(dtype=float)
        if not np.allclose(a, b, rtol=0.0, atol=PUBLISHED_ARM_ATOL):
            raise ValueError(f"{arm}: study metrics {a} differ from the published run {b}")


def run_downstream(
    models: list[str], *, study_dir: Path, jobs: int, recompute: bool = False
) -> tuple[pd.DataFrame, dict]:
    """The PCaBiop APD driver for every arm, into the study directory (resumable)."""
    import apd_experiment

    out_dir = study_dir / "apd"
    apd = apd_experiment.run(
        ["pcabiop"], models, iterations=apd_experiment.CANONICAL_ITERATIONS,
        out_dir=out_dir, overwrite=recompute, jobs=jobs,
    )
    cells = {m: json.loads((out_dir / "pcabiop" / f"{m}.json").read_text()) for m in models}
    return apd[apd["dataset"] == "pcabiop"], cells


def arm_summary(
    arms: list[tuple[str, str, str]], metrics: pd.DataFrame, apd: pd.DataFrame
) -> pd.DataFrame:
    """One row per (pair, arm): the representation and downstream values the table shows."""
    metrics, apd = metrics.set_index("model"), apd.set_index("model")
    rows = []
    for pair, slide, meanpool in arms:
        for role, model in (("slide", slide), ("meanpool", meanpool)):
            rows.append(
                dict(
                    pair=pair, arm=role, model=model,
                    **{c: float(metrics.loc[model, c]) for c in METRIC_COLUMNS},
                    nipd_id=float(apd.loc[model, "nipd_id"]),
                    nipd_ood=float(apd.loc[model, "nipd_ood"]),
                )
            )
    return pd.DataFrame(rows)


def evaluate(
    pairs: list[str],
    *,
    study_dir: Path,
    embeddings_dir: Path,
    jobs: int,
    recompute: bool = False,
) -> None:
    """Score both arms of every pair and write the study's three artifacts.

    ``recompute`` bypasses the benchmark's metrics cache and the APD resume cache, as the
    one-command reproduction does.
    """
    import importlib.metadata

    import apd_experiment
    from croma.metrics.croma import CROMA_HEADLINE_M
    from loaders import DATASETS, training_correlations

    records = json.loads((study_dir / REPRODUCTION_JSON).read_text())
    missing = [p for p in pairs if p not in records]
    if missing:
        raise ValueError(f"no reduction for {missing}; run `reduce --pairs {' '.join(missing)}`")
    arms = [(p, records[p]["slide_arm"], records[p]["meanpool_arm"]) for p in pairs]
    models = [m for _, slide, meanpool in arms for m in (slide, meanpool)]
    rows = _pcabiop_rows()

    metrics, per_sample, bench_argv = run_representation(
        models, rows=rows, study_dir=study_dir, embeddings_dir=embeddings_dir,
        recompute=recompute,
    )
    check_published_arms(metrics, [slide for _, slide, _ in arms])
    apd, cells = run_downstream(models, study_dir=study_dir, jobs=jobs, recompute=recompute)

    headline = f"croma_m{int(CROMA_HEADLINE_M)}"
    per_sample = per_sample.rename(columns={headline: "croma"})[
        ["model", "sample_id", "group_id", "label", "confounder", "croma"]
    ]
    differences = paired_differences(
        arms, per_sample, cells,
        cramers_v=training_correlations("pcabiop"),
        chance=1.0 / len(DATASETS["pcabiop"]["biological_classes"]),
    )
    summary = arm_summary(arms, metrics, apd)
    summary.to_csv(study_dir / "summary.csv", index=False)
    differences.to_csv(study_dir / "differences.csv", index=False)

    extraction = study_dir / "extraction.json"
    provenance = {
        "study": STUDY_NAME,
        "pairs": {
            p: {
                **{k: v for k, v in vars(PAIRS_BY_MODEL[p]).items()},
                "reproduction": records[p],
            }
            for p in pairs
        },
        "declared_pairs": [pair.slide_model for pair in PAIRS],
        "extraction": json.loads(extraction.read_text()) if extraction.exists() else None,
        "slide2vec_version": importlib.metadata.version("slide2vec"),
        "representation": {
            "benchmark": BENCHMARK, "protocol": _published_run().protocol,
            "argv": bench_argv, "croma_headline_m": int(CROMA_HEADLINE_M),
        },
        "downstream": {
            "dataset": "pcabiop", "iterations": apd_experiment.CANONICAL_ITERATIONS,
            "probe_seed": apd_experiment.PROBE_SEED,
            "validation_fraction": apd_experiment.VAL_FRACTION,
        },
        "bootstrap": {
            "seed": BOOTSTRAP_SEED, "n_boot": BOOTSTRAP_N, "level": BOOTSTRAP_LEVEL,
            "alpha": TAIL_ALPHA, "strata": "label x confounder", "clusters": "group_id",
            "shared_draw": "one draw for every arm of every pair",
        },
    }
    (study_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"[evaluate] wrote summary.csv, differences.csv, provenance.json to {study_dir}")


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

STUDY_DIR = REPO / "output" / "studies" / STUDY_NAME
PCABIOP_DIR = REPO / "output" / "embeddings" / TILESET
REPRODUCTION_JSON = "reproduction.json"

#: The GPU is shared: a pair starts only when this much of it is free.
GPU_FREE_MIB = 20_000
GPU_POLL_SECONDS = 300


def _gpu_free_mib() -> float:
    """Free GPU memory, probed in a child process so this driver holds no CUDA context."""
    import subprocess

    probe = "import torch; print(torch.cuda.mem_get_info()[0] / 2**20)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    try:
        return float(out.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return 0.0


def _pcabiop_rows() -> pd.DataFrame:
    return pd.read_csv(PCABIOP_DIR / "manifest.csv")


def _limited(rows: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return rows
    return pd.concat([_cohort_rows(rows, c).head(limit) for c in COHORTS])


def _cmd_extract(args) -> int:
    import time

    rows = _pcabiop_rows()
    study_dir = Path(args.study_dir)
    record_path = study_dir / "extraction.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    for model in args.pairs:
        pair = PAIRS_BY_MODEL[model]
        free = _gpu_free_mib()
        if not pair.required and free < GPU_FREE_MIB:
            print(f"[extract] {model}: optional and the GPU is busy ({free:.0f} MiB free); skipped")
            record[model] = {"status": "skipped (GPU busy)", "at": time.strftime("%F %T")}
            continue
        while free < GPU_FREE_MIB:
            print(f"[extract] {model}: waiting for the GPU ({free:.0f} MiB free)", flush=True)
            time.sleep(GPU_POLL_SECONDS)
            free = _gpu_free_mib()
        started = time.strftime("%F %T")
        extract_pair(pair, rows=rows, study_dir=study_dir, limit=args.limit)
        record[model] = {"status": "done", "started": started, "finished": time.strftime("%F %T")}
        study_dir.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    study_dir.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return 0


def _cmd_status(args) -> int:
    rows = _limited(_pcabiop_rows(), args.limit)
    for pair in PAIRS:
        cells = []
        for cohort in COHORTS:
            ids = _cohort_rows(rows, cohort)["sample_id"].astype(str).tolist()
            done, total = cohort_progress(Path(args.study_dir) / "runs" / pair.slide_model / cohort, ids)
            cells.append(f"{cohort} {done}/{total}")
        kind = "required" if pair.required else "optional"
        print(f"{pair.slide_model:14s} ({kind}): " + ", ".join(cells))
    return 0


def _cmd_reduce(args) -> int:
    rows = _limited(_pcabiop_rows(), args.limit)
    study_dir = Path(args.study_dir)
    path = study_dir / REPRODUCTION_JSON
    records = json.loads(path.read_text()) if path.exists() else {}
    for model in args.pairs:
        record = reduce_pair(
            PAIRS_BY_MODEL[model],
            rows=rows,
            run_root=study_dir / "runs" / model,
            embeddings_dir=Path(args.embeddings_dir),
            study_dir=study_dir,
            published_dir=PCABIOP_DIR,
        )
        records[model] = record
        print(
            f"[reduce] {model}: {record['n_reproduced']}/{record['n_slides']} slides reproduce "
            f"(min cosine {record['min_cosine']:.4f}); slide arm {record['slide_arm']}",
            flush=True,
        )
    study_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")
    return 0


def _cmd_evaluate(args) -> int:
    evaluate(
        args.pairs,
        study_dir=Path(args.study_dir),
        embeddings_dir=Path(args.embeddings_dir),
        jobs=args.jobs,
        recompute=args.recompute,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    required_pairs = [p.slide_model for p in PAIRS if p.required]
    for name, help_ in (
        ("extract", "re-run slide2vec for each pair (GPU, hours; run detached)"),
        ("status", "embedded / expected slides per pair and cohort"),
        ("reduce", "build the mean-pooled arms and check reproduction"),
        ("evaluate", "score both arms and write the paired differences"),
    ):
        cmd = sub.add_parser(name, help=help_)
        cmd.add_argument("--study-dir", default=str(STUDY_DIR))
        if name != "evaluate":
            cmd.add_argument(
                "--limit", type=int, default=None, help="first N slides per cohort (smoke runs)"
            )
        if name != "status":
            cmd.add_argument(
                "--pairs", nargs="+", default=required_pairs, choices=list(PAIRS_BY_MODEL)
            )
        if name in ("reduce", "evaluate"):
            cmd.add_argument("--embeddings-dir", default=str(PCABIOP_DIR))
        if name == "evaluate":
            cmd.add_argument("--jobs", type=int, default=3, help="parallel APD processes")
            cmd.add_argument(
                "--recompute",
                action="store_true",
                help="bypass the benchmark metrics cache and the APD resume cache",
            )
    args = parser.parse_args(argv)
    commands = {
        "extract": _cmd_extract, "status": _cmd_status,
        "reduce": _cmd_reduce, "evaluate": _cmd_evaluate,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
