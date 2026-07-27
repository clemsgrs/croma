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
embedding cache instead of PathoROB's FeatureDataManager. croma's per-tileset
embedding matrix (``output/embeddings/<tileset>/``) is row-aligned with PathoROB's
metadata CSV (verified for all three datasets), so the per-cell pseudo-slide chunking
that PathoROB performs is reproduced bit-for-bit.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from loaders import DATASETS, REPO, _pcabiop_split_map, _prostate_split_map, load_data  # noqa: E402

import layout  # noqa: E402  (loaders put scripts/bench on sys.path on import above)

# Reused verbatim from PathoROB (numpy/sklearn only, no torch side-effects); importing
# ``loaders`` above puts the PathoROB checkout on sys.path so these resolve.
from pathorob.apd.utils import get_patches_map_to_split, compute_apd  # noqa: E402
from pathorob.apd.train_model import train_logistic_regression  # noqa: E402

#: Slide-level (num_patches_per_slide == 1) validation fraction. PathoROB reserves
#: int(num_patches / max_train_slides) validation samples per cell -- a fraction
#: 1/max_train_slides of that cell's training count -- but at slide level both operands
#: are in slides, so that ratio underflows to 0. We set the fraction explicitly instead.
#: 0.1 sits inside PathoROB's own effective range (Camelyon 1/14 = 7%, TCGA 1/8 = 12.5%,
#: Tolkach 1/6 = 17%). See loaders._pcabiop_split_map.
VAL_FRACTION = 0.1

#: nAPD (normalized Average Performance Drop) gate. nAPD divides the per-split drop by
#: the baseline's *skill* (balanced accuracy above chance) rather than by raw accuracy,
#: so it measures the fraction of *learnable* signal a confounder destroys. When the
#: probe barely beats chance at the balanced split there is almost no skill to lose, the
#: skill ratio is dominated by noise, and nAPD is not meaningful -- we report it as
#: undefined instead. The gate is on *normalized* skill = skill / (1 - chance), the
#: fraction of achievable headroom attained at baseline (= informedness for a binary task,
#: class-count invariant in [0, 1]), so one floor applies across 2-, 4- and 6-class tasks.
NAPD_NORM_SKILL_FLOOR = 0.15


def compute_napd(accuracies, chance):
    """nAPD reduction: the skill-normalized performance drop on the iteration-averaged
    accuracy curve (ratio-of-means).

    Skill = balanced accuracy above ``chance``. Like APD, nAPD averages the per-split
    ratio of confounded-split performance to the balanced split and subtracts 1, but on
    skill instead of raw accuracy: ``nAPD = mean_{split>0}(skill_split / skill_0) - 1``.

    It deviates from PathoROB's ``compute_apd`` in exactly one way: it averages the
    ``iterations`` replicates per split *before* taking the ratio, not after. This is
    deliberate. Per replicate ``nAPD = APD * acc_0 / (acc_0 - chance)``, so the
    mean-of-ratios reduction PathoROB uses for APD explodes whenever a *single* replicate's
    baseline dips near chance (``acc_0 -> chance``) -- a blow-up a gate on the *mean*
    baseline cannot catch (observed on prostate/Prost40M and pcabiop/MOOZY, where
    mean-of-ratios flips the sign). Ratio-of-means is stable, agrees with APD's ranking away
    from chance (Spearman >= 0.94 on every tile benchmark), and diverges only near chance,
    exactly where mean-of-ratios is unreliable. ``accuracies`` is (num_splits, iterations).
    """
    s = np.asarray(accuracies, dtype=float).mean(axis=1) - chance  # per-split mean skill
    return float((s[1:] / s[0]).mean() - 1)


def _napd_summary(res, chance):
    """Derive nAPD fields from a result dict's stored accuracy matrices (post-hoc, no
    re-run). ``napd_id``/``napd_ood`` are None when the baseline fails the skill gate;
    the ungated value is always kept in ``napd_id_raw``/``napd_ood_raw`` for provenance."""
    out = {"chance": chance}
    for dom, key in (("id", "id_test_accuracies"), ("ood", "ood_test_accuracies")):
        acc = np.asarray(res[key], dtype=float)
        base = float(acc[0].mean())                       # balanced-acc at split 0
        norm_skill = (base - chance) / (1 - chance)       # fraction of achievable headroom
        gated = norm_skill < NAPD_NORM_SKILL_FLOOR
        raw = compute_napd(acc, chance)  # already a scalar (ratio-of-means)
        out[f"{dom}_baseline"] = base
        out[f"{dom}_norm_skill"] = norm_skill
        out[f"{dom}_gated"] = bool(gated)
        out[f"napd_{dom}_raw"] = raw
        out[f"napd_{dom}"] = None if gated else raw
    return out


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
            elif split_key == "pcabiop":
                split_map, max_train_slides = _pcabiop_split_map(split, num_patches_per_slide)
            else:
                split_map, max_train_slides = get_patches_map_to_split(split_key, split, num_patches_per_slide)

            data_train = [features[i][j][:num_patches] for i, j, num_patches in split_map]
            data_train = [t for cls in data_train for t in cls]

            # Validation slice, per cell, placed right after that cell's training block --
            # PathoROB's structure exactly, so val inherits the schedule's marginal balance.
            # Patch level (npps > 1): int(num_patches / max_train_slides), byte-identical to
            # PathoROB. Slide level (npps == 1): that ratio underflows to 0, so use an explicit
            # fraction and push idxTest out to clear the widest train+val block. See VAL_FRACTION.
            if num_patches_per_slide == 1:
                val_take = [round(VAL_FRACTION * num_patches) for _, _, num_patches in split_map]
                idxTest = max_train_slides + round(VAL_FRACTION * max_train_slides) + 1
            else:
                val_take = [int(num_patches / max_train_slides) for _, _, num_patches in split_map]
                idxTest = (max_train_slides + 1) * num_patches_per_slide

            data_validation = [features[i][j][num_patches:num_patches + take]
                               for (i, j, num_patches), take in zip(split_map, val_take)]
            data_validation = [t for cls in data_validation for t in cls]

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
    res = dict(
        apd_id=apd_id, apd_ood=apd_ood,
        id_accuracy_means=[float(np.mean(a)) for a in id_test_accuracies],
        ood_accuracy_means=[float(np.mean(a)) for a in ood_test_accuracies],
        id_test_accuracies=id_test_accuracies, ood_test_accuracies=ood_test_accuracies,
    )
    res.update(_napd_summary(res, chance=1.0 / len(biological_classes)))
    return res


def model_list(dataset):
    return sorted(p.stem for p in layout.embeddings_dir(DATASETS[dataset]["src"]).glob("*.npy"))


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
    # nAPD is derived from the stored accuracy matrices, so cached JSONs (which predate
    # nAPD) get it here without a re-run. chance = 1 / n biological classes.
    nap = _napd_summary(res, chance=1.0 / len(DATASETS[dataset]["biological_classes"]))
    fmt = lambda v: "gated" if v is None else f"{v*100:.2f}%"
    print(f"[{tag}] {dataset}/{model}: nAPD_ID={fmt(nap['napd_id'])} nAPD_OOD={fmt(nap['napd_ood'])} "
          f"(APD_ID={res['apd_id']*100:.2f}% APD_OOD={res['apd_ood']*100:.2f}%)", flush=True)
    return dict(dataset=dataset, model=model,
                napd_id=nap["napd_id"], napd_ood=nap["napd_ood"],
                apd_id=res["apd_id"], apd_ood=res["apd_ood"],
                id_baseline=nap["id_baseline"], ood_baseline=nap["ood_baseline"],
                id_gated=nap["id_gated"], ood_gated=nap["ood_gated"])


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
    p.add_argument("--out_dir", default="output/studies/apd")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--jobs", type=int, default=1, help="parallel (model,dataset) processes")
    return p.parse_args()


if __name__ == "__main__":
    a = get_args()
    run(a.datasets, a.models, a.iterations, REPO / a.out_dir, a.overwrite, jobs=a.jobs)
