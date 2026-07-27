"""Average Performance Drop (APD), faithful to PathoROB, on croma embeddings.

APD (PathoROB, Bontempo et al.) measures how much a biology-only linear probe
degrades as a *spurious* correlation between medical centre and biological class
is injected into its training set. For each dataset a fixed schedule walks the
training composition from balanced (split 0, no correlation) to strongly
correlated (higher splits); the probe is trained per split and evaluated on a
held-out in-domain (ID) test set and an out-of-domain (OOD) test set of unseen
centres. APD is the mean relative accuracy change w.r.t. the balanced split:

    APD = mean_{split>0} ( acc_split / acc_split0 ) - 1      (typically < 0)

Closer to 0 = more robust. We report APD_ID and APD_OOD per (model, dataset),
alongside nAPD, croma's skill-normalized refinement of the same drop.

None of that is implemented here. The sweep, both reductions and PathoROB's own
schedules ship in ``croma.downstream`` (ADR-0011), where the vendored PathoROB code sits
frozen and a user of the package can run the same protocol on their own model. What this
driver owns is everything downstream of "which numbers": the model roster, the resume
cache, the per-dataset configuration in ``loaders.py``, and the CSV. Its faithfulness
claim is unchanged -- croma's per-tileset embedding matrix
(``output/embeddings/<tileset>/``) is row-aligned with PathoROB's metadata CSV, so
grouping rows into cells in metadata order reproduces PathoROB's per-cell pseudo-slide
chunking bit for bit.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from loaders import DATASETS, REPO, arrangement_for, load_data, split_schedule  # noqa: E402

import layout  # noqa: E402  (loaders put scripts/bench on sys.path on import above)

from croma import apd, napd  # noqa: E402
from croma.downstream import IN_DOMAIN, probe_sweep_over_test_sets  # noqa: E402

#: Slide-level (num_patches_per_slide == 1) validation fraction. PathoROB reserves
#: int(num_patches / max_train_slides) validation samples per cell -- a fraction
#: 1/max_train_slides of that cell's training count -- but at slide level both operands
#: are in slides, so that ratio underflows to 0. We set the fraction explicitly instead.
#: 0.1 sits inside PathoROB's own effective range (Camelyon 1/14 = 7%, TCGA 1/8 = 12.5%,
#: Tolkach 1/6 = 17%). See loaders._pcabiop_split_map.
VAL_FRACTION = 0.1

#: The key the OOD accuracy matrix comes back from the sweep under. The sweep scores the
#: unseen centres off the same training pass as the held-out in-domain rows, so both arms
#: cost one sweep rather than two.
OUT_OF_DOMAIN = "out_of_domain"


def compute(model, dataset, iterations=20):
    """Run the confounder-biased probe sweep for one (model, dataset) and reduce it.

    Returns a dict with apd_id, apd_ood and both raw accuracy matrices -- the matrices
    because every other number reported about this cell (nAPD, the baseline, whatever a
    later reporting decision needs) derives from them without re-running the sweep, which
    is the expensive part.
    """
    cfg = DATASETS[dataset]
    rows_per_slide = cfg["num_patches_per_slide"]
    cohort = load_data(model, dataset)

    accuracies = probe_sweep_over_test_sets(
        cohort.embeddings,
        cohort.confounders,
        cohort.labels,
        schedule=split_schedule(dataset, rows_per_slide),
        test_sets={OUT_OF_DOMAIN: (cohort.ood_embeddings, cohort.ood_labels)},
        rows_per_slide=rows_per_slide,
        iterations=iterations,
        # PathoROB's own rule at patch level; at slide level it underflows to zero, so the
        # study states a fraction instead -- one of its two documented slide-level
        # deviations, the other being loaders._pcabiop_split_map's schedule.
        validation_fraction=None if rows_per_slide > 1 else VAL_FRACTION,
        arrange_slides=arrangement_for(dataset, cohort.slide_ids),
    )
    id_accuracies, ood_accuracies = accuracies[IN_DOMAIN], accuracies[OUT_OF_DOMAIN]
    return dict(
        apd_id=apd(id_accuracies), apd_ood=apd(ood_accuracies),
        id_accuracy_means=list(id_accuracies.mean(axis=1)),
        ood_accuracy_means=list(ood_accuracies.mean(axis=1)),
        id_test_accuracies=id_accuracies.tolist(), ood_test_accuracies=ood_accuracies.tolist(),
    )


def _reductions(res, chance):
    """Every number the CSV carries per domain, derived from the stored accuracy matrices.

    Post-hoc, so a cached JSON gets whatever the current reductions report without a
    re-run. nAPD is reported for every cell: whether one is too close to chance to lean on
    is a reporting decision for whoever renders the table, not a property of the metric
    (ADR-0014), so nothing is suppressed here.
    """
    out = {}
    for domain, key in (("id", "id_test_accuracies"), ("ood", "ood_test_accuracies")):
        acc = np.asarray(res[key], dtype=float)
        out[f"{domain}_baseline"] = float(acc[0].mean())  # balanced-acc at split 0
        out[f"napd_{domain}"] = napd(acc, chance)
    return out


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
    # Both reductions derive from the stored accuracy matrices, so a cached JSON gets them
    # without a re-run. chance = 1 / n biological classes.
    red = _reductions(res, chance=1.0 / len(DATASETS[dataset]["biological_classes"]))
    print(f"[{tag}] {dataset}/{model}: nAPD_ID={red['napd_id']*100:.2f}% nAPD_OOD={red['napd_ood']*100:.2f}% "
          f"(APD_ID={res['apd_id']*100:.2f}% APD_OOD={res['apd_ood']*100:.2f}%)", flush=True)
    return dict(dataset=dataset, model=model,
                napd_id=red["napd_id"], napd_ood=red["napd_ood"],
                apd_id=res["apd_id"], apd_ood=res["apd_ood"],
                id_baseline=red["id_baseline"], ood_baseline=red["ood_baseline"])


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
