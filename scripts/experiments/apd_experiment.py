"""Average Performance Drop (APD), faithful to PathoROB, on croma embeddings.

APD (PathoROB, Bontempo et al.) measures how much a biology-only linear probe
degrades as a *spurious* correlation between medical centre and biological class
is injected into its training set. For each dataset a fixed schedule walks the
training composition from balanced (split 0, no correlation) to strongly
correlated (higher splits); the probe is trained per split and evaluated on a
held-out in-domain (ID) test set and an out-of-domain (OOD) test set of unseen
centres. APD is the mean relative accuracy change w.r.t. the balanced split:

    APD = mean_{split>0} ( acc_split / acc_split0 ) - 1      (typically < 0)

Closer to 0 = more robust. We report APD_ID and APD_OOD per (model, dataset).

Faithfulness: the split schedule, train/val/test slicing, probe and the APD
reduction are PathoROB's own code, imported verbatim from ../PathoROB. The only
thing authored here is the feature loader, which sources features from croma's
embedding cache instead of PathoROB's FeatureDataManager. croma's
``embedding_source_manifest.csv`` is row-aligned with PathoROB's metadata CSV
(verified for all three datasets), so the per-cell pseudo-slide chunking that
PathoROB performs is reproduced bit-for-bit.
"""
import os
# Pin BLAS to one thread per process: the probe grid-search parallelises poorly
# inside one process, so we fan out across (model, dataset) processes instead and
# keep each single-threaded to avoid oversubscription. Must precede numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

REPO = Path("/data/pathology/projects/clement/code/croma")
PATHOROB = Path("/data/pathology/projects/clement/code/PathoROB")
sys.path.insert(0, str(PATHOROB))

# Reused verbatim from PathoROB (numpy/sklearn only, no torch side-effects).
from pathorob.apd.utils import get_patches_map_to_split, compute_apd  # noqa: E402
from pathorob.apd.train_model import train_logistic_regression  # noqa: E402
from pathorob import resources as pathorob_resources  # noqa: E402
import importlib.resources as pkg_resources  # noqa: E402

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
        with pkg_resources.files(pathorob_resources).joinpath("tolkach_splits.json").open("r") as f:
            feasible_splits = json.load(f)

    return (centers_id, bio, features, data_test_ood,
            cfg["num_splits"], cfg["num_slides_per_category"], cfg["num_patches_per_slide"],
            feasible_splits)


def compute(model, dataset, iterations=20):
    """Faithful port of PathoROB apd.apd.compute(): per-split probe accuracies +
    APD reduction. Returns dict with apd_id, apd_ood and the raw accuracy matrices."""
    cfg = DATASETS[dataset]
    split_key = cfg["split_key"]

    random.seed(1000)
    (medical_centers, biological_classes, features, data_test_ood,
     num_splits, num_slides_per_category, num_patches_per_slide, feasible_splits) = load_data(model, dataset)

    seeds = random.sample(range(0, 10000), iterations)
    id_test_accuracies = [[] for _ in range(num_splits)]
    ood_test_accuracies = [[] for _ in range(num_splits)]

    for idx, seed in enumerate(seeds):
        random.seed(seed)

        if dataset == "tolkach":
            random_train_test_split = (
                random.randint(0, len(feasible_splits['train'][medical_centers[0]]) - 1),
                random.randint(0, len(feasible_splits['train'][medical_centers[1]]) - 1),
            )
        for i, center in enumerate(medical_centers):
            if dataset == "tolkach":
                test_cases = feasible_splits['test'][center][random_train_test_split[i]]
            for j in range(len(biological_classes)):
                dataChunkedBySlides = [features[i][j][k:k + num_patches_per_slide]
                                       for k in range(0, len(features[i][j]), num_patches_per_slide)]
                random.shuffle(dataChunkedBySlides)
                if dataset == "tolkach":
                    case_order = [dataChunkedBySlides[k][0][3] for k in range(len(dataChunkedBySlides))]
                    for test_case in list(set(test_cases) & set(case_order)):
                        dataChunkedBySlides.append(dataChunkedBySlides.pop(
                            [dataChunkedBySlides[k][0][3] for k in range(len(dataChunkedBySlides))].index(test_case)))
                features[i][j] = [item for slide in dataChunkedBySlides for item in slide]

        for split in trange(num_splits, desc=f"{model}/{dataset} iter {idx + 1}/{len(seeds)}", leave=False):
            if split_key == "prostate":
                split_map, max_train_slides = _prostate_split_map(split, num_patches_per_slide)
            else:
                split_map, max_train_slides = get_patches_map_to_split(split_key, split, num_patches_per_slide)

            data_train = [features[i][j][:num_patches] for i, j, num_patches in split_map]
            data_train = [t for cls in data_train for t in cls]

            data_validation = [features[i][j][num_patches:num_patches + int(num_patches / max_train_slides)]
                               for i, j, num_patches in split_map]
            data_validation = [t for cls in data_validation for t in cls]

            idxTest = (max_train_slides + 1) * num_patches_per_slide
            data_test_id = [features[i][j][idxTest:]
                            for i in range(len(medical_centers)) for j in range(len(biological_classes))]
            data_test_id = [t for cls in data_test_id for t in cls]

            train_x, train_y = zip(*[(p, b) for p, _, b, _ in data_train])
            val_x, val_y = zip(*[(p, b) for p, _, b, _ in data_validation])
            test_x_id, test_y_id = zip(*[(p, b) for p, _, b, _ in data_test_id])
            test_x_ood, test_y_ood = zip(*[(p, b) for p, _, b, _ in data_test_ood])

            _, _, test_scores = train_logistic_regression(
                train_x, train_y, val_x, val_y, [test_x_id, test_x_ood], [test_y_id, test_y_ood])
            id_test_accuracies[split].append(test_scores[0])
            ood_test_accuracies[split].append(test_scores[1])

    apd_id = float(np.mean(compute_apd(id_test_accuracies)))
    apd_ood = float(np.mean(compute_apd(ood_test_accuracies)))
    return dict(
        apd_id=apd_id, apd_ood=apd_ood,
        id_accuracy_means=[float(np.mean(a)) for a in id_test_accuracies],
        ood_accuracy_means=[float(np.mean(a)) for a in ood_test_accuracies],
        id_test_accuracies=id_test_accuracies, ood_test_accuracies=ood_test_accuracies,
    )


def model_list(dataset):
    return sorted(p.stem for p in (REPO / DATASETS[dataset]["src"] / "embeddings").glob("*.npy"))


def _job(args):
    """One (dataset, model) unit of work: compute APD and persist its JSON.
    Returns the summary row. Resumable: a present JSON is loaded, not recomputed."""
    dataset, model, iterations, out_dir, overwrite = args
    raw_path = Path(out_dir) / dataset / f"{model}.json"
    if raw_path.exists() and not overwrite:
        res = json.loads(raw_path.read_text())
        tag = "skip"
    else:
        res = compute(model, dataset, iterations=iterations)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(res, indent=2))
        tag = "done"
    print(f"[{tag}] {dataset}/{model}: APD_ID={res['apd_id']*100:.2f}% APD_OOD={res['apd_ood']*100:.2f}%", flush=True)
    return dict(dataset=dataset, model=model, apd_id=res["apd_id"], apd_ood=res["apd_ood"])


def run(datasets, models, iterations, out_dir, overwrite, jobs=1):
    out_dir = Path(out_dir)
    tasks = [(ds, m, iterations, str(out_dir), overwrite)
             for ds in datasets for m in (models or model_list(ds))]
    if jobs > 1:
        rows = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_job, t) for t in tasks]
            for f in as_completed(futs):
                rows.append(f.result())
    else:
        rows = [_job(t) for t in tasks]

    df = pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "apd.csv", index=False)
    print(f"\nwrote {out_dir/'apd.csv'} ({len(df)} rows)")
    return df


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--models", nargs="+", default=None, help="default: all models in each dataset's cache")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--out_dir", default="output/apd")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="parallel (model,dataset) processes")
    return p.parse_args()


if __name__ == "__main__":
    a = get_args()
    run(a.datasets, a.models, a.iterations, REPO / a.out_dir, a.overwrite, jobs=a.jobs)
