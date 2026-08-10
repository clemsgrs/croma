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
alongside nIPD, croma's chance-normalized integral of the degradation curve.

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
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from loaders import (  # noqa: E402
    DATASETS,
    PATHOROB_DOWNSTREAM_DATASETS,
    REPO,
    arrangement_for,
    load_data,
    pathorob_tile_panel,
    split_schedule,
    training_correlations,
)

import layout  # noqa: E402  (loaders put scripts/bench on sys.path on import above)

from croma import apd, nipd  # noqa: E402
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

#: Canonical PathoROB protocol constants. The library defaults to these same values, but
#: passing the seed explicitly makes the recorded study immune to an unrelated default
#: change and gives the study record one unambiguous fact to fingerprint.
CANONICAL_ITERATIONS = 20
PROBE_SEED = 1000


def compute(model, dataset, iterations=20):
    """Run the confounder-biased probe sweep for one (model, dataset) and reduce it.

    Returns a dict with apd_id, apd_ood and both raw accuracy matrices -- the matrices
    because every other number reported about this cell (nIPD, the baseline, whatever a
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
        seed=PROBE_SEED,
        # PathoROB's own rule at patch level; at slide level it underflows to zero, so the
        # study states a fraction instead -- one of its two documented slide-level
        # deviations, the other being loaders._pcabiop_split_map's schedule.
        validation_fraction=None if rows_per_slide > 1 else VAL_FRACTION,
        arrange_slides=arrangement_for(dataset, cohort.group_ids),
    )
    id_accuracies, ood_accuracies = accuracies[IN_DOMAIN], accuracies[OUT_OF_DOMAIN]
    return dict(
        apd_id=apd(id_accuracies), apd_ood=apd(ood_accuracies),
        id_accuracy_means=list(id_accuracies.mean(axis=1)),
        ood_accuracy_means=list(ood_accuracies.mean(axis=1)),
        id_test_accuracies=id_accuracies.tolist(), ood_test_accuracies=ood_accuracies.tolist(),
    )


def _reductions(res, *, chance, cramers_v):
    """Every number the CSV carries per domain, derived from the stored accuracy matrices.

    Post-hoc, so a cached JSON gets whatever the current reductions report without a
    re-run. nIPD is reported for every cell: whether one is too close to chance to lean on
    is a reporting decision for whoever renders the table, not a property of the metric
    (ADR-0018), so nothing is suppressed here.
    """
    out = {}
    for domain, key in (("id", "id_test_accuracies"), ("ood", "ood_test_accuracies")):
        acc = np.asarray(res[key], dtype=float)
        out[f"{domain}_baseline"] = float(acc[0].mean())  # balanced-acc at split 0
        out[f"nipd_{domain}"] = nipd(acc, cramers_v=cramers_v, chance=chance)
    return out


def model_list(dataset):
    available = {p.stem for p in layout.embeddings_dir(DATASETS[dataset]["src"]).glob("*.npy")}
    if dataset not in PATHOROB_DOWNSTREAM_DATASETS:
        return sorted(available)

    expected = pathorob_tile_panel()["model"].tolist()
    missing = sorted(set(expected) - available)
    extra = sorted(available - set(expected))
    if missing or extra:
        raise ValueError(
            f"{dataset}: embedding roster differs from the expanded PathoROB panel "
            f"(missing={missing}, extra={extra})"
        )
    return expected


def _validated_cell(res, *, dataset, model, iterations):
    """Validate one resumable raw cell before it can contribute to a summary."""
    expected_shape = (DATASETS[dataset]["num_splits"], iterations)
    for domain in ("id", "ood"):
        key = f"{domain}_test_accuracies"
        if key not in res:
            raise ValueError(f"{dataset}/{model}: cached cell is missing {key}")
        matrix = np.asarray(res[key], dtype=float)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{dataset}/{model}: {key} has shape {matrix.shape}, expected {expected_shape}"
            )
        if not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
            raise ValueError(f"{dataset}/{model}: {key} must contain finite accuracies in [0, 1]")
        means_key = f"{domain}_accuracy_means"
        means = np.asarray(res.get(means_key, []), dtype=float)
        if means.shape != (expected_shape[0],) or not np.allclose(
            means, matrix.mean(axis=1), rtol=0, atol=1e-15
        ):
            raise ValueError(
                f"{dataset}/{model}: {means_key} does not match the stored accuracy matrix"
            )
        apd_key = f"apd_{domain}"
        if apd_key not in res or not np.isclose(
            float(res[apd_key]), apd(matrix), rtol=0, atol=1e-15
        ):
            raise ValueError(
                f"{dataset}/{model}: {apd_key} does not match the stored accuracy matrix"
            )

    # Exercise the primary reduction too: this catches a non-positive balanced-skill
    # denominator and a schedule/matrix mismatch before either reaches the panel CSV.
    _reductions(
        res,
        chance=1.0 / len(DATASETS[dataset]["biological_classes"]),
        cramers_v=training_correlations(dataset),
    )
    return res


def _validate_summary_cell(row, res, *, dataset, model):
    """Prove one summary row is exactly reducible from its raw cell."""
    reductions = _reductions(
        res,
        chance=1.0 / len(DATASETS[dataset]["biological_classes"]),
        cramers_v=training_correlations(dataset),
    )
    expected = {
        "nipd_id": reductions["nipd_id"],
        "nipd_ood": reductions["nipd_ood"],
        "apd_id": res["apd_id"],
        "apd_ood": res["apd_ood"],
        "id_baseline": reductions["id_baseline"],
        "ood_baseline": reductions["ood_baseline"],
    }
    for key, value in expected.items():
        if key not in row or not np.isclose(float(row[key]), float(value), rtol=0, atol=1e-12):
            raise ValueError(
                f"{dataset}/{model}: {key} summary value does not match its raw accuracy matrix"
            )


def _atomic_write_text(path, text):
    """Replace ``path`` from a flushed same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _model_annotations(model):
    panel = pathorob_tile_panel().set_index("model")
    if model not in panel.index:
        return dict(parent_model="none", variant_role="none", ranked=True)
    row = panel.loc[model]
    return dict(
        parent_model="none" if pd.isna(row["parent_model"]) else str(row["parent_model"]),
        variant_role="none" if pd.isna(row["variant_role"]) else str(row["variant_role"]),
        ranked=bool(row["ranked"]),
    )


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
        _validated_cell(res, dataset=dataset, model=model, iterations=iterations)
        _atomic_write_text(raw_path, json.dumps(res, indent=2, allow_nan=False) + "\n")
        tag = "done"
    _validated_cell(res, dataset=dataset, model=model, iterations=iterations)
    # Both reductions derive from the stored accuracy matrices, so a cached JSON gets them
    # without a re-run. chance = 1 / n biological classes.
    red = _reductions(
        res,
        chance=1.0 / len(DATASETS[dataset]["biological_classes"]),
        cramers_v=training_correlations(dataset),
    )
    print(
        f"[{tag}] {dataset}/{model}: nIPD_ID={red['nipd_id']*100:.2f}% nIPD_OOD={red['nipd_ood']*100:.2f}% "
        f"(APD_ID={res['apd_id']*100:.2f}% APD_OOD={res['apd_ood']*100:.2f}%)",
        flush=True,
    )
    return dict(
        dataset=dataset,
        model=model,
        **_model_annotations(model),
        nipd_id=red["nipd_id"],
        nipd_ood=red["nipd_ood"],
        apd_id=res["apd_id"],
        apd_ood=res["apd_ood"],
        id_baseline=red["id_baseline"],
        ood_baseline=red["ood_baseline"],
    )


def _merged_summary(out_dir, rows):
    fresh = pd.DataFrame(rows)
    path = Path(out_dir) / "apd.csv"
    if path.exists():
        existing = pd.read_csv(path)
        replaced = set(zip(fresh["dataset"], fresh["model"]))
        keep = [
            (dataset, model) not in replaced
            for dataset, model in zip(existing["dataset"], existing["model"])
        ]
        fresh = pd.concat([existing.loc[keep], fresh], ignore_index=True)

    if fresh.duplicated(["dataset", "model"]).any():
        dup = fresh.loc[fresh.duplicated(["dataset", "model"], False), ["dataset", "model"]]
        raise ValueError(f"APD summary contains duplicate cells: {dup.to_dict('records')}")
    for column in ("parent_model", "variant_role", "ranked"):
        fresh[column] = [_model_annotations(model)[column] for model in fresh["model"]]
    return fresh.sort_values(["dataset", "model"]).reset_index(drop=True)


def _validate_complete_panel(df, *, out_dir, iterations):
    if iterations != CANONICAL_ITERATIONS:
        raise ValueError(
            f"complete expanded PathoROB panel requires {CANONICAL_ITERATIONS} iterations, "
            f"got {iterations}"
        )
    panel = pathorob_tile_panel().set_index("model")
    expected = set(panel.index)
    expected_ranking = panel["ranked"].to_dict()
    failures = []
    for dataset in PATHOROB_DOWNSTREAM_DATASETS:
        rows = df[df["dataset"] == dataset]
        observed = set(rows["model"])
        if observed != expected:
            failures.append(
                f"{dataset}: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )
            continue
        observed_ranking = rows.set_index("model")["ranked"].to_dict()
        if observed_ranking != expected_ranking:
            failures.append(f"{dataset}: ranked/control treatment differs from model metadata")
            continue
        for model in expected:
            raw_path = Path(out_dir) / dataset / f"{model}.json"
            if not raw_path.exists():
                failures.append(f"{dataset}/{model}: raw cell is missing")
                continue
            try:
                raw = _validated_cell(
                    json.loads(raw_path.read_text()),
                    dataset=dataset,
                    model=model,
                    iterations=iterations,
                )
                _validate_summary_cell(
                    rows.loc[rows["model"] == model].iloc[0],
                    raw,
                    dataset=dataset,
                    model=model,
                )
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                failures.append(str(error))
    if failures:
        raise ValueError(
            "complete expanded PathoROB panel validation failed: " + "; ".join(failures)
        )


def run(datasets, models, iterations, out_dir, overwrite, jobs=1, require_complete_panel=False):
    out_dir = Path(out_dir)
    tasks = [
        (ds, m, iterations, str(out_dir), overwrite)
        for ds in datasets
        for m in (models or model_list(ds))
    ]
    if jobs > 1:
        rows = []
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(_job, t) for t in tasks]
            for f in as_completed(futs):
                rows.append(f.result())
    else:
        rows = [_job(t) for t in tasks]

    df = _merged_summary(out_dir, rows)
    if require_complete_panel:
        _validate_complete_panel(df, out_dir=out_dir, iterations=iterations)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out_dir / "apd.csv", df.to_csv(index=False))
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
    p.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="allow a smoke/subset execution to publish an incomplete PathoROB tile summary",
    )
    return p.parse_args()


if __name__ == "__main__":
    a = get_args()
    run(
        a.datasets,
        a.models,
        a.iterations,
        REPO / a.out_dir,
        a.overwrite,
        jobs=a.jobs,
        require_complete_panel=not a.allow_incomplete_panel,
    )
