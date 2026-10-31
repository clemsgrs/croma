"""Correlate CRoMa (and RI/MaRI) with the downstream drop across models.

Join the representation metrics with downstream performance degradation and report
rank correlation across models within each benchmark.

The primary target is nIPD (normalized Integrated Performance Degradation), which
normalizes by above-chance baseline skill and integrates over Cramér's V. PathoROB's
APD is retained as a continuity analysis.

Hypothesis (pre-registered): a more confounder-robust representation (higher
CRoMa) gives a downstream probe fewer exploitable shortcuts, so it suffers a
smaller drop -> nIPD closer to 0. Because nIPD is signed, we expect a positive
Spearman(CRoMa, nIPD), with ID as the cleaner shortcut-susceptibility endpoint.
OOD additionally contains transfer to an unseen acquisition distribution.

RI/MaRI are reported alongside purely as a sanity check that the drop is a sensible
yardstick — no claim that CRoMa predicts it better than RI.

Every association carries inference (``croma.metrics.bootstrap``): Spearman's rho, a 95%
encoder-bootstrap CI (one encoder draw shared by every row of a benchmark), a two-sided
permutation p (exact over all orderings for PCaBiop's five slide encoders, Monte Carlo
otherwise) and a Benjamini--Hochberg q. The BH family is every association the table
renders on the tile benchmarks; PCaBiop is a separate descriptive column outside it. The
tail statistics ``LTM_10`` and ``F(0)`` are correlated with nIPD_ID plainly and partially,
given median CRoMa (Freedman--Lane permutation p).

For rho(median CRoMa, nIPD) a second, two-level design is also written: each replicate
resamples the encoders, resamples each drawn encoder's probe runs (the columns of its
accuracy matrices) and recomputes its nIPD, and takes every encoder's median CRoMa from one
stratified slide draw shared across encoders (replicate ``b`` of
``bootstrap_uncertainty.py``'s ``bootstrap_replicates.npz``). Both designs sit side by side
in ``apd_correlation.csv``; the table generator renders one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from loaders import (
    CORR_METRICS as METRICS,
    DATASETS as DATASET_SPECS,
    JOIN_KEYS as DATASETS,
    REPO,
    STUDY_DIR,
    load_joined,
    ranked,
    training_correlations,
)

for _path in (REPO / "src", REPO / "scripts" / "repro"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from croma.downstream.nipd import nipd  # noqa: E402
from croma.metrics.bootstrap import (  # noqa: E402
    _percentile_ci,
    _row_pearson,
    _row_ranks,
    benjamini_hochberg,
    encoder_bootstrap_draws,
    partial_spearman_inference,
    spearman_inference,
)

N_BOOT = 2000
N_PERMUTATIONS = 10_000
LEVEL = 0.95
#: The uncertainty study's seed; the two-level design's probe-run stream is seeded apart.
SEED = 12345
_PROBE_RUN_STREAM = 2

TARGETS = ["nipd_id", "nipd_ood", "apd_id", "apd_ood"]
TILE_SCOPES = ["camelyon", "tcga_4x4", "tolkach"]
SLIDE_SCOPES = ["pcabiop"]
TAIL_METRICS = ["croma_ltm_alpha", "croma_f0"]
PARTIAL_ON = "croma"
#: The rows tab:apd-correlation renders: its tile cells are the Benjamini--Hochberg family.
RENDERED = (
    {("nipd_id", m) for m in [*METRICS, *TAIL_METRICS]}
    | {("nipd_id", f"{m}_given_{PARTIAL_ON}") for m in TAIL_METRICS}
    | {("nipd_ood", m) for m in METRICS}
    | {("apd_id", "croma"), ("apd_ood", "croma")}
)
#: The associations the two-level design is computed for.
TWO_LEVEL = {"nipd_id": "id_test_accuracies", "nipd_ood": "ood_test_accuracies"}


def corr_block(df, target):
    """Spearman correlation of each metric with one target, per benchmark.

    Computed on the ranked panel. The control would be doubly flattered here: its CRoMa is
    high because its biological neighbourhoods are poor, and a downstream reduction can
    flatter a model with little signal to lose. Both effects push the same way, and
    including it would inflate the association.
    """
    df = ranked(df)
    rows = []
    for metric in METRICS:
        for scope in DATASETS:
            sub = df[df["dataset"] == scope]
            sub = sub[[metric, target]].dropna()
            if len(sub) < 3:
                continue
            rho, p_s = spearmanr(sub[metric], sub[target])
            rows.append(dict(target=target, metric=metric, scope=scope, n=len(sub),
                             spearman=rho, spearman_p=p_s))
    return pd.DataFrame(rows)


def _panel(df, scope):
    """One benchmark's ranked encoders, in a fixed order every design shares."""
    return ranked(df)[ranked(df)["dataset"] == scope].sort_values("model").reset_index(drop=True)


def _inference_row(target, metric, scope, partial_on, inference, asymptotic_p, n_boot):
    return dict(
        target=target, metric=metric, scope=scope, partial_on=partial_on, n=inference.n,
        spearman=inference.rho, spearman_p=asymptotic_p, perm_p=inference.p_value,
        permutation=inference.permutation, n_permutations=inference.n_permutations,
        ci_lo=inference.ci.lo, ci_hi=inference.ci.hi, n_boot=int(n_boot), level=LEVEL,
    )


def association_table(joined, *, n_boot=N_BOOT, seed=SEED, n_permutations=N_PERMUTATIONS):
    """Every association with rho, encoder-bootstrap CI, permutation p and (in-family) BH q."""
    rows = []
    for scope in [*TILE_SCOPES, *SLIDE_SCOPES]:
        sub = _panel(joined, scope)
        if len(sub) < 3:
            continue
        kw = dict(level=LEVEL, seed=seed, n_permutations=n_permutations)
        draws = encoder_bootstrap_draws(len(sub), n_boot=n_boot, seed=seed)
        for target in TARGETS:
            metrics = [*METRICS, *TAIL_METRICS] if target == "nipd_id" else METRICS
            for metric in metrics:
                x, y = sub[metric].to_numpy(float), sub[target].to_numpy(float)
                inference = spearman_inference(x, y, draws=draws, **kw)
                rows.append(_inference_row(target, metric, scope, "", inference,
                                           float(spearmanr(x, y).pvalue), n_boot))
        if scope not in TILE_SCOPES:
            continue
        for metric in TAIL_METRICS:
            inference = partial_spearman_inference(
                sub[metric].to_numpy(float), sub["nipd_id"].to_numpy(float),
                sub[PARTIAL_ON].to_numpy(float), draws=draws, **kw,
            )
            rows.append(_inference_row("nipd_id", f"{metric}_given_{PARTIAL_ON}", scope,
                                       PARTIAL_ON, inference, inference.p_value, n_boot))
    table = pd.DataFrame(rows)
    family = table["scope"].isin(TILE_SCOPES) & pd.Series(
        [(t, m) in RENDERED for t, m in zip(table["target"], table["metric"])], index=table.index
    )
    table["in_bh_family"] = family
    table["q_bh"] = np.nan
    table.loc[family, "q_bh"] = benjamini_hochberg(table.loc[family, "perm_p"].to_numpy())
    return table


def two_level_replicates(croma_reps, accuracies, *, cramers_v, chance, draws, seed):
    """rho(median CRoMa, nIPD) per replicate of the two-level design.

    Replicate ``b`` takes encoders ``draws[b]``, each encoder's median from row ``b`` of
    ``croma_reps`` (``n_boot x n``, one shared slide draw per row) and its nIPD recomputed
    on a with-replacement draw of its probe runs (the columns of ``accuracies[j]``).
    """
    rng = np.random.default_rng([int(seed), _PROBE_RUN_STREAM])
    draws = np.asarray(draws)
    x = np.take_along_axis(np.asarray(croma_reps, dtype=float), draws, axis=1)
    y = np.empty(draws.shape, dtype=float)
    for b, picked in enumerate(draws):
        for i, j in enumerate(picked):
            acc = np.asarray(accuracies[j], dtype=float)
            runs = rng.integers(0, acc.shape[1], acc.shape[1])
            y[b, i] = nipd(acc[:, runs], cramers_v=cramers_v, chance=chance)
    return _row_pearson(_row_ranks(x), _row_ranks(y))


def _croma_replicates(scope, models, n_boot):
    from paper_manifest import by_benchmark

    path = (REPO / by_benchmark(DATASET_SPECS[scope]["benchmark"]).studies_rel
            / "bootstrap_replicates.npz")
    if not path.exists():
        raise FileNotFoundError(f"{path} is absent; run scripts/studies/bootstrap_uncertainty.py")
    with np.load(path) as saved:
        index = {m: j for j, m in enumerate(saved["models"].tolist())}
        medians = saved["median"]
    if medians.shape[0] != n_boot:
        raise ValueError(f"{path} holds {medians.shape[0]} replicates, not {n_boot}")
    return medians[:, [index[m] for m in models]]


def add_two_level(table, joined, *, n_boot=N_BOOT, seed=SEED):
    """Fill ``two_level_lo``/``two_level_hi`` for rho(median CRoMa, nIPD) on the tile scopes."""
    table = table.assign(two_level_lo=np.nan, two_level_hi=np.nan)
    for scope in TILE_SCOPES:
        sub = _panel(joined, scope)
        models = sub["model"].tolist()
        croma_reps = _croma_replicates(scope, models, n_boot)
        cramers_v = training_correlations(scope)
        chance = 1.0 / len(DATASET_SPECS[scope]["biological_classes"])
        records = [json.loads((STUDY_DIR / scope / f"{m}.json").read_text()) for m in models]
        draws = encoder_bootstrap_draws(len(sub), n_boot=n_boot, seed=seed)
        for target, key in TWO_LEVEL.items():
            accuracies = [np.asarray(r[key], dtype=float) for r in records]
            point = np.array([nipd(a, cramers_v=cramers_v, chance=chance) for a in accuracies])
            if not np.allclose(point, sub[target].to_numpy(float), rtol=0, atol=1e-12):
                raise ValueError(f"{scope}: recomputed {target} does not match apd.csv")
            reps = two_level_replicates(croma_reps, accuracies, cramers_v=cramers_v,
                                        chance=chance, draws=draws, seed=seed)
            at = (table["scope"] == scope) & (table["target"] == target) & (
                table["metric"] == "croma")
            ci = _percentile_ci(reps, float(table.loc[at, "spearman"].iloc[0]), LEVEL, n_boot)
            table.loc[at, ["two_level_lo", "two_level_hi"]] = [ci.lo, ci.hi]
    return table


def main(apd_csv, out_dir, n_boot=N_BOOT, seed=SEED):
    out_dir = Path(out_dir)
    df = load_joined(apd_csv)
    df.to_csv(out_dir / "apd_metrics_joined.csv", index=False)

    res = add_two_level(association_table(df, n_boot=n_boot, seed=seed), df,
                        n_boot=n_boot, seed=seed)
    res.to_csv(out_dir / "apd_correlation.csv", index=False)

    pd.set_option("display.width", 160, "display.float_format", lambda x: f"{x:.3f}")
    for target in TARGETS:
        ref = " (reference)" if target.startswith("apd_") else " (primary)"
        print(f"\n===== Spearman( metric , {target} ){ref} — expect POSITIVE =====")
        piv = (res[res["target"] == target]
               .pivot(index="metric", columns="scope", values="spearman")
               .reindex(columns=DATASETS))
        print(piv.to_string())
    print(f"\nwrote {out_dir/'apd_correlation.csv'} and {out_dir/'apd_metrics_joined.csv'}")


if __name__ == "__main__":
    apd_csv = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "output/studies/apd/apd.csv")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else str(REPO / "output/studies/apd")
    n_boot = int(sys.argv[3]) if len(sys.argv) > 3 else N_BOOT
    main(apd_csv, out_dir, n_boot=n_boot)
