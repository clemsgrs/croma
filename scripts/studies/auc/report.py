"""Reduce the AUROC cells to nIPD under both scores, and correlate each with the metrics.

Two outputs, both post hoc on the stored cells (nothing here re-runs a sweep):

* the **reduction** (``output/studies/auc/nipd_auroc.csv``): one row per cohort x encoder x
  regime (in domain, out of domain) holding nIPD from AUROC (chance 0.5) and from balanced
  accuracy (chance 1/K), both ``croma.nipd`` on the APD study's Cramér's-V grid; the
  V = 0 and V = 1 means and the normalised endpoint g(1) under each score; the control flag;
  and CRoMa, RI and MaRI from the run ``paper_manifest`` resolves for the cohort, RI and
  MaRI at that run's operating k.
* the **correlations** (``output/studies/auc/nipd_auroc_correlation.csv``): per cohort and
  regime, over ranked encoders, Spearman rho between each metric and each nIPD, and the
  agreement rho(nIPD from AUROC, nIPD from balanced accuracy). Each carries an
  encoder-bootstrap CI and a permutation p from ``croma.metrics.bootstrap``; the tile
  cohorts' associations form one Benjamini-Hochberg family, and PCaBiop is descriptive.

Both refuse to run on an incomplete or stale set of cells: a missing number is an error,
never a skipped row. ``tab:nipd-auroc`` and its macros are built from :func:`reduce` and
:func:`correlate` directly (``scripts/repro/_nipd_auroc.py``), so they fail the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# auc_cells puts the APD study (and through its loaders, src/) on sys.path.
from auc_cells import APD_REL, AUC_REL, all_statuses, cell_path  # noqa: E402
from loaders import (  # noqa: E402
    DATASETS,
    JOIN_KEYS,
    PATHOROB_DOWNSTREAM_DATASETS,
    REPO,
    metric_csv,
    training_correlations,
)
from plotting.style import CONTROL_MODEL  # noqa: E402  (loaders put scripts/bench on sys.path)

from croma import nipd  # noqa: E402
from croma.metrics.bootstrap import (  # noqa: E402
    benjamini_hochberg,
    encoder_bootstrap_draws,
    spearman_inference,
)

#: Which AUROC-cell columns carry each regime's (AUROC, balanced accuracy) pair.
REGIMES = {"id": ("auroc_id", "balacc_id"), "ood": ("auroc_ood", "balacc_ood")}
#: AUROC's chance level, whatever the class count; balanced accuracy's is 1/K.
AUROC_CHANCE = 0.5
#: The representation metrics each nIPD is correlated with, read from the cohort's run.
METRICS = ("croma", "ri", "mari")

#: The two downstream scores, each reduced to its own nIPD.
SCORES = ("auroc", "balacc")
#: Cohorts whose associations form the Benjamini-Hochberg family: the tile cohorts, with
#: 26 ranked encoders each. PCaBiop's five slide encoders are reported descriptively.
FAMILY_DATASETS = PATHOROB_DOWNSTREAM_DATASETS
#: Encoder-bootstrap replicates and Monte-Carlo permutations per correlation, and the seed
#: both are drawn from. Fixed, so the table rebuilds byte for byte.
N_BOOT = 2000
N_PERMUTATIONS = 10_000
SEED = 0

REDUCTION_CSV = AUC_REL / "nipd_auroc.csv"
CORRELATION_CSV = AUC_REL / "nipd_auroc_correlation.csv"

_RERUN = "Run scripts/studies/auc/sweep.py, which exits 0 once every cell is current"


def _require_current_cells(root: Path) -> list[tuple[str, str]]:
    """Every expected cell, or an error naming each missing or stale one."""
    statuses = all_statuses(auc_dir=root / AUC_REL, apd_dir=root / APD_REL)
    if not statuses:
        raise FileNotFoundError(f"no APD cells under {root / APD_REL}: nothing to reduce")
    unusable = [s for s in statuses if s.state != "current"]
    if unusable:
        listed = ", ".join(f"{s.dataset}/{s.model} ({s.state})" for s in unusable)
        raise ValueError(
            f"{len(unusable)} of {len(statuses)} AUROC cells cannot be used: {listed}. {_RERUN}."
        )
    return [(s.dataset, s.model) for s in statuses]


def run_metrics(dataset: str, root: Path = REPO) -> pd.DataFrame:
    """CRoMa, RI, MaRI and k per encoder, from the run the paper reports for ``dataset``.

    One row per encoder is the run's operating point: the shared median k on the tile
    panel, each encoder's own k* on the slide panel.
    """
    path = root / metric_csv(dataset)
    frame = pd.read_csv(path)
    if frame["model"].duplicated().any():
        raise ValueError(f"{path}: more than one row per encoder, so no single operating k")
    if (frame["croma"].abs() > 1.0).any():
        raise ValueError(f"{path}: CRoMa is not on the signed-margin scale (|CRoMa| > 1)")
    return frame.set_index("model")[[*METRICS, "k"]]


def _endpoint(means: np.ndarray, chance: float) -> float:
    """g(1): the change at V = 1 as a fraction of the V = 0 skill above chance."""
    return float((means[-1] - means[0]) / (means[0] - chance))


def reduce(root: Path = REPO) -> pd.DataFrame:
    """One row per cohort x encoder x regime: nIPD and endpoints under both scores."""
    root = Path(root)
    rows = []
    metrics = {}
    for dataset, model in _require_current_cells(root):
        if dataset not in metrics:
            metrics[dataset] = run_metrics(dataset, root)
        if model not in metrics[dataset].index:
            raise ValueError(
                f"{dataset}/{model} has an AUROC cell but no row in {root / metric_csv(dataset)}"
            )
        record = json.loads(cell_path(root / AUC_REL, dataset, model).read_text("utf-8"))
        matrix = np.asarray(record["rows"], dtype=float)
        cramers_v = training_correlations(dataset)
        chance = {
            "auroc": AUROC_CHANCE,
            "balacc": 1.0 / len(DATASETS[dataset]["biological_classes"]),
        }
        for regime, pair in REGIMES.items():
            row = dict(dataset=dataset, model=model, regime=regime, control=model == CONTROL_MODEL)
            for score, column in zip(SCORES, pair):
                scores = matrix[..., record["columns"].index(column)]
                means = scores.mean(axis=1)
                row[f"{score}_v0"], row[f"{score}_v1"] = float(means[0]), float(means[-1])
                row[f"endpoint_{score}"] = _endpoint(means, chance[score])
                row[f"nipd_{score}"] = nipd(scores, cramers_v=cramers_v, chance=chance[score])
            row.update(metrics[dataset].loc[model].to_dict())
            rows.append(row)
    frame = pd.DataFrame(rows)
    # The converse of the check above: an encoder the run reports but the downstream study
    # never probed would quietly drop out of the panel the correlations range over.
    for dataset, run in metrics.items():
        unprobed = sorted(set(run.index) - set(frame.loc[frame["dataset"] == dataset, "model"]))
        if unprobed:
            raise ValueError(
                f"{dataset}: {root / metric_csv(dataset)} reports {', '.join(unprobed)}, which "
                "has no APD cell. Run that APD cell, then its AUROC cell."
            )
    order = {dataset: i for i, dataset in enumerate(JOIN_KEYS)}
    return frame.sort_values(
        ["dataset", "model", "regime"], key=lambda c: c.map(order) if c.name == "dataset" else c
    ).reset_index(drop=True)


def _pairs() -> list[tuple[str, str, str]]:
    """``(kind, x, y)`` per correlation in one cohort x regime, in table order."""
    associations = [
        ("association", metric, f"nipd_{score}") for metric in METRICS for score in SCORES
    ]
    return [*associations, ("agreement", "nipd_auroc", "nipd_balacc")]


def correlate(
    reduction: pd.DataFrame,
    *,
    n_boot: int = N_BOOT,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Spearman inference per cohort x regime x pair, over ranked encoders.

    Every correlation in a cohort shares one encoder-bootstrap draw, so their replicates
    are paired (e.g. AUROC's rho against balanced accuracy's on the same resamples). The
    BH family is every association on the tile cohorts; the agreement rows and the
    descriptive PCaBiop column sit outside it and carry no q value.
    """
    rows = []
    ranked = reduction[~reduction["control"].astype(bool)]
    for dataset in [d for d in JOIN_KEYS if d in set(ranked["dataset"])]:
        cohort = ranked[ranked["dataset"] == dataset]
        models = sorted(cohort["model"].unique())
        draws = encoder_bootstrap_draws(len(models), n_boot=n_boot, seed=seed)
        for regime in REGIMES:
            panel = cohort[cohort["regime"] == regime].set_index("model").loc[models]
            for kind, x, y in _pairs():
                inference = spearman_inference(
                    panel[x].to_numpy(),
                    panel[y].to_numpy(),
                    n_permutations=n_permutations,
                    seed=seed,
                    draws=draws,
                )
                rows.append(
                    dict(
                        dataset=dataset,
                        regime=regime,
                        kind=kind,
                        x=x,
                        y=y,
                        n=inference.n,
                        rho=inference.rho,
                        ci_lo=inference.ci.lo,
                        ci_hi=inference.ci.hi,
                        level=inference.ci.level,
                        n_boot=inference.ci.n_boot,
                        p_value=inference.p_value,
                        permutation=inference.permutation,
                        n_permutations=inference.n_permutations,
                    )
                )
    corr = pd.DataFrame(rows)
    corr["in_family"] = (corr["kind"] == "association") & corr["dataset"].isin(FAMILY_DATASETS)
    corr["q_value"] = np.nan
    corr.loc[corr["in_family"], "q_value"] = benjamini_hochberg(
        corr.loc[corr["in_family"], "p_value"].to_numpy()
    )
    return corr


def main(root: Path = REPO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write both study outputs under ``root`` and print the correlations."""
    from apd_experiment import _atomic_write_text

    root = Path(root)
    reduction = reduce(root)
    corr = correlate(reduction)
    _atomic_write_text(root / REDUCTION_CSV, reduction.to_csv(index=False))
    _atomic_write_text(root / CORRELATION_CSV, corr.to_csv(index=False))

    with pd.option_context("display.width", 160, "display.float_format", "{:+.3f}".format):
        shown = corr.assign(pair=corr["x"] + " ~ " + corr["y"])
        print(
            shown.pivot_table(
                index=["kind", "pair"], columns=["dataset", "regime"], values="rho", sort=False
            ).to_string()
        )
    print(f"wrote {root / REDUCTION_CSV} ({len(reduction)} rows)")
    print(f"wrote {root / CORRELATION_CSV} ({len(corr)} rows)")
    return reduction, corr


if __name__ == "__main__":
    main()
