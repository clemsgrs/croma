"""Shared data loaders and configuration for the APD experiment trio.

``apd_experiment.py`` computes APD, ``apd_croma_correlation.py`` correlates it with the
faithful CRoMa/RI/MaRI metrics, and ``apd_figure.py`` plots it. They share the repo
root, the per-dataset APD configuration, and the data loaders collected here, so the
paths and join logic live in exactly one place.

PathoROB (``pathorob.*``) is imported lazily inside ``load_data`` (the only loader that
needs it) so the pure-pandas loaders (``load_joined``, ``read_joined``) and the plotting
entrypoint do not pull PathoROB/sklearn. Every function is byte-for-byte identical to the
inline code it replaces.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root autodetected from this file's location: scripts/studies/apd/loaders.py
# -> parents[3] is the croma repo root, so the reproduction works from any checkout.
REPO = Path(__file__).resolve().parents[3]
# PathoROB is a SIBLING repo (not under croma), so it cannot be autodetected. Default to
# a sibling checkout next to croma; override with PATHOROB_ROOT if it lives elsewhere.
PATHOROB = Path(os.environ.get("PATHOROB_ROOT", REPO.parent / "PathoROB"))
if str(PATHOROB) not in sys.path:
    sys.path.insert(0, str(PATHOROB))

# ---------------------------------------------------------------------------
# APD computation config (used by apd_experiment.py)
# ---------------------------------------------------------------------------
# Per-dataset constants, copied from PathoROB's apd/utils.load_data so that our
# (centre, biology) cell indexing matches the split schedule exactly. ``split_key``
# is the name get_patches_map_to_split expects (note: tcga_4x4 -> "tcga").
DATASETS = {
    "camelyon": dict(
        src="output/pathorob-camelyon",
        metadata="data/pathorob/metadata/camelyon.csv",
        split_key="camelyon",
        centers_id=["RUMC", "UMCU"],
        centers_ood=["CWZ", "RST", "LPON"],
        biological_classes=["normal", "tumor"],
        num_splits=8, num_slides_per_category=17, num_patches_per_slide=300,
    ),
    "tcga_4x4": dict(
        src="output/pathorob-tcga-4x4",
        metadata="data/pathorob/metadata/tcga_4x4.csv",
        split_key="tcga",
        centers_id=["Asterand", "Christiana Healthcare", "Roswell Park", "University of Pittsburgh"],
        centers_ood=["Cureline", "Greater Poland Cancer Center", "International Genomics Consortium", "Johns Hopkins"],
        biological_classes=["Breast_invasive_carcinoma", "Colon_adenocarcinoma", "Lung_adenocarcinoma", "Lung_squamous_cell_carcinoma"],
        num_splits=7, num_slides_per_category=12, num_patches_per_slide=30,
    ),
    "tolkach": dict(
        src="output/pathorob-tolkach-esca",
        metadata="data/pathorob/metadata/tolkach_esca.csv",
        split_key="tolkach_esca",
        centers_id=["VALSET2_WNS", "VALSET4_CHA_FULL"],
        centers_ood=["VALSET1_UKK", "VALSET3_TCGA"],
        biological_classes=["TUMOR", "MUSC_PROP", "SH_OES", "SH_MAG", "REGR_TU", "ADVENT"],
        num_splits=4, num_slides_per_category=9, num_patches_per_slide=100,
    ),
    # prostate-shift-binary: a 2-centre x 2-class design, structurally like Camelyon, but
    # with a croma-authored split schedule (split_key="prostate", see _prostate_split_map)
    # instead of Camelyon's. ID = the mirrored KI/RUMC pair (1,850 patches/cell = 50 slides
    # x 37 patches, uniform), held out OOD = NUS. Camelyon's schedule reserves a fixed 15
    # slides/cell for train+val and was tuned to its 17-slide cells, leaving only ~2 test
    # slides/cell -- a known weakness (test estimate rests on 2 slides per fold). With 50
    # slides/cell we instead use balanced load M=16, confounder step Delta=2, num_splits=9:
    # train 64*npps=2,368 (conserved across splits), test = 50-(2M+1) = 17 slides/cell
    # (idxTest=33*npps=1,221 << 1,850), reaching full confounding (favorable cell -> 0) at
    # the last split since M is even. num_patches_per_slide governs only ID pseudo-slide
    # chunking; KI/RUMC are genuinely 37/slide so pseudo-slides == real slides (slide-
    # disjoint train/test). NUS is a flat OOD test pool (no npps chunking), so its ragged
    # 24-79 patches/slide is irrelevant. See paper for the schedule-faithfulness note.
    "prostate": dict(
        src="output/prostate-shift-binary",
        metadata="data/prostate/metadata/prostate_shift.csv",
        split_key="prostate",
        centers_id=["KI", "RUMC"],
        centers_ood=["NUS"],
        biological_classes=["benign", "tumor"],
        num_splits=9, num_slides_per_category=50, num_patches_per_slide=37,
    ),
}


def _prostate_split_map(split, num_patches_per_slide):
    """croma-authored split schedule for prostate-shift-binary.

    Mirrors PathoROB's Camelyon schedule structure (2 centres x 2 classes, a
    centre<->class spurious correlation injected by skewing the per-cell training
    counts) but is re-parameterised for prostate's 50 slides/cell instead of
    Camelyon's 17. Camelyon hard-codes a balanced load of 7 slides/cell, unit
    confounder step, and reserves 2*14+1 = 15 of its 17 slides for train+val --
    leaving only ~2 test slides/cell. Re-using that here would idle 35 of our 50
    slides in test. We instead use:

        balanced load   M     = 16 slides/cell   (split 0: every cell = 16*npps)
        confounder step Delta = 2 slides/split
        num_splits            = M/Delta + 1 = 9  (split 8: favourable cell -> 0)

    so favourable cells carry (M - Delta*split)*npps training patches and
    unfavourable cells (M + Delta*split)*npps. max_train_slides = 2M = 32, hence
    idxTest = 33*npps = 1,221 and a fixed 50-33 = 17-slide (629-patch) test set
    per cell; total train = 64*npps = 2,368, conserved across splits. M is even so
    Delta=2 still reaches the fully-confounded endpoint (favourable cell = 0),
    matching Camelyon's maximal-confounder split.

    PathoROB's own get_patches_map_to_split is left untouched/verbatim; this is the
    single croma-authored deviation, documented as such in the paper.
    """
    M, delta = 16, 2
    favourable = (M - delta * split) * num_patches_per_slide
    unfavourable = (M + delta * split) * num_patches_per_slide
    tss0_pairs = [(0, 0, favourable), (0, 1, unfavourable)]
    tss1_pairs = [(1, 0, unfavourable), (1, 1, favourable)]
    return sorted(tss0_pairs + tss1_pairs), 2 * M


def load_data(model, dataset):
    """croma-cache equivalent of PathoROB's apd.utils.load_data.

    Returns the same 8-tuple PathoROB's compute() expects:
        centers_id, biological_classes, features, data_test_ood,
        num_splits, num_slides_per_category, num_patches_per_slide, feasible_splits

    where ``features[center_idx][bio_idx]`` is a list of
    (feature_vector, center_idx, bio_idx, slide_id) tuples in metadata order, and
    ``data_test_ood`` is a flat list of the same tuple shape for OOD centres.
    """
    cfg = DATASETS[dataset]
    centers_id, centers_ood = cfg["centers_id"], cfg["centers_ood"]
    bio = cfg["biological_classes"]

    md = pd.read_csv(REPO / cfg["metadata"])
    emb = np.load(REPO / cfg["src"] / "embeddings" / f"{model}.npy")
    assert len(emb) == len(md), f"{model}/{dataset}: {len(emb)} emb vs {len(md)} meta rows"

    md_id = md[md["subset"] == "ID"]
    md_ood = md[md["subset"] == "OOD"]

    features = [[[] for _ in bio] for _ in centers_id]
    for row in md_id.itertuples():
        ci, bi = centers_id.index(row.medical_center), bio.index(row.biological_class)
        features[ci][bi].append((emb[row.Index], ci, bi, row.slide_id))

    data_test_ood = [
        (emb[row.Index], centers_ood.index(row.medical_center), bio.index(row.biological_class), row.slide_id)
        for row in md_ood.itertuples()
    ]

    feasible_splits = None
    if dataset == "tolkach":
        # Reused verbatim from PathoROB (numpy/sklearn only, no torch side-effects);
        # imported lazily so the pure-pandas loaders and the figure do not pull PathoROB.
        import importlib.resources as pkg_resources

        from pathorob import resources as pathorob_resources
        with pkg_resources.files(pathorob_resources).joinpath("tolkach_splits.json").open("r") as f:
            feasible_splits = json.load(f)

    return (centers_id, bio, features, data_test_ood,
            cfg["num_splits"], cfg["num_slides_per_category"], cfg["num_patches_per_slide"],
            feasible_splits)


# ---------------------------------------------------------------------------
# APD <-> metric correlation config + loaders (used by apd_croma_correlation.py
# and apd_figure.py)
# ---------------------------------------------------------------------------
# The first three are the faithful PathoROB benchmarks that constitute the headline
# validation table (tab:apd-correlation). `prostate` is the caveated second-organ
# extension: its APD_ID corroborates the validation, but its APD_OOD rests on a single
# small out-of-domain centre (NUS: 300/class, no Gleason-3) and is reported separately
# with that caveat -- do NOT read the 4-benchmark `pooled` APD_OOD as the headline.
DATASET_KEYS = ["camelyon", "tcga_4x4", "tolkach", "prostate"]
# The headline validation table (tab:apd-correlation) pools over ONLY the three faithful
# PathoROB benchmarks (48 model-benchmark pairs). Prostate is excluded from this pool: its
# single-centre OOD arm cannot be pooled into a cross-benchmark APD_OOD statistic (see the
# `prostate` caveat above). `headline` is that 3-benchmark pool; `pooled` is all four.
HEADLINE_DATASETS = ["camelyon", "tcga_4x4", "tolkach"]
CORR_METRICS = ["croma", "ri", "mari"]

# Canonical faithful (n=20,400) metric dirs — the exact CSVs the paper's tables and
# figures read, with CRoMa already on the signed-margin scale the paper defines. The
# `dataset` column there holds the dir name, so we override it with the APD key below.
METRIC_DIR = {
    "camelyon": "output/faithful/k-star/pathorob-camelyon-faithful",
    "tcga_4x4": "output/faithful/k-star/pathorob-tcga-4x4",
    "tolkach": "output/faithful/k-star/pathorob-tolkach-esca-faithful",
    "prostate": "output/prostate-shift-binary-kirumc",
}


def load_joined(apd_csv):
    """Join APD results with the faithful CRoMa/RI/MaRI metrics, per (dataset, model)."""
    apd = pd.read_csv(apd_csv)
    frames = []
    for ds in DATASET_KEYS:
        m = pd.read_csv(REPO / METRIC_DIR[ds] / "results/metrics.csv")
        m["dataset"] = ds  # align with APD's dataset key (CSV stores the dir name)
        # Defensive: the canonical dirs are already signed-margin CRoMa; only convert
        # if a legacy ratio-scale CSV (any value > 1) is ever pointed at here.
        if m["croma"].max() > 1.0:
            m["croma"] = (m["croma"] - 1.0) / (m["croma"] + 1.0)
        frames.append(m)
    faith = pd.concat(frames, ignore_index=True)
    df = apd.merge(faith[["dataset", "model", *CORR_METRICS, "croma_ltm_alpha"]], on=["dataset", "model"], how="inner")
    missing = set(zip(apd["dataset"], apd["model"])) - set(zip(df["dataset"], df["model"]))
    if missing:
        print(f"[warn] {len(missing)} (dataset,model) APD rows had no faithful metric and were dropped: {sorted(missing)}")
    return df


def read_joined():
    """Load the APD<->metric join written by ``apd_croma_correlation.main``.

    Written by ``apd_croma_correlation.py`` with CRoMa already on the signed-margin
    scale (the paper's definition).
    """
    return pd.read_csv(REPO / "output/apd/apd_metrics_joined.csv")
