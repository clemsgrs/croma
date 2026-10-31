"""Reduce the probe records to coverage, sensitivity and specificity, and low-data results.

The confounder-biased probe sweep (``croma.downstream``) trains a biology probe on a training
set walked from balanced (Cramér's V = 0) to fully confounded (V = 1). The probe records
(``scripts/studies/apd/probe_records.py``) keep, per cohort, encoder and tiles per training
slide, every probe's accuracy and every held-out row's decision margins. This study reduces
them, post hoc, to:

* **coverage**: per in-domain row, the replicates in which it was held out, and per cohort
  the share of rows never held out;
* **sensitivity and specificity** on the binary cohorts, per probe, in and out of domain,
  each reduced to nIPD with chance 0.5 on the APD study's V grid;
* **low data**: per cohort, tile level and encoder, the V = 0 balanced accuracy and nIPD in
  and out of domain;

and correlates the nIPDs with CRoMa across the ranked encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for _p in (REPO / "src", HERE, HERE / "apd", HERE / "auc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import apd_croma_correlation as apd_correlation  # noqa: E402
import probe_records  # noqa: E402
import report as auc_report  # noqa: E402
from loaders import DATASETS, STUDY_DIR, training_correlations  # noqa: E402
from plotting.style import CONTROL_MODEL  # noqa: E402  (loaders put scripts/bench on sys.path)

from croma import nipd  # noqa: E402
from croma.metrics.bootstrap import (  # noqa: E402
    _percentile_ci,
    benjamini_hochberg,
    encoder_bootstrap_draws,
    spearman_inference,
)

#: Where the records are read from, the APD cells the full tile level is read from, and
#: where this study writes -- all relative to the checkout.
RECORDS_REL = probe_records.RECORDS_DIR.relative_to(REPO)
APD_REL = STUDY_DIR.relative_to(REPO)
STUDY_REL = Path("output/studies/probe-reductions")

#: Chance for sensitivity and specificity, each a per-class recall.
SENS_SPEC_CHANCE = 0.5

#: The sensitivity and specificity correlations sit in ``tab:nipd-auroc`` beside the AUROC
#: study's, so they are drawn as that study draws: the same encoder-bootstrap replicates,
#: permutations and seed, hence the same encoder draw on a cohort with none undefined.
SENS_SPEC_N_BOOT = auc_report.N_BOOT
SENS_SPEC_N_PERMUTATIONS = auc_report.N_PERMUTATIONS
SENS_SPEC_SEED = auc_report.SEED
#: The low-data correlations repeat ``tab:apd-correlation``'s rho(CRoMa, nIPD) at fewer
#: tiles, so they are drawn as the APD study draws, and at full tiles reproduce its row.
LOW_DATA_N_BOOT = apd_correlation.N_BOOT
LOW_DATA_N_PERMUTATIONS = apd_correlation.N_PERMUTATIONS
LOW_DATA_SEED = apd_correlation.SEED
LEVEL = apd_correlation.LEVEL

#: The two regimes, keyed as the APD and AUROC studies key them, and the record arrays and
#: APD-cell lists that hold each.
REGIMES = {"id": "in_domain", "ood": "out_of_domain"}
APD_CELL_KEYS = {"id": "id_test_accuracies", "ood": "ood_test_accuracies"}

OUTPUTS = (
    "coverage.csv",
    "held_out_counts.csv",
    "sens_spec_probes.csv",
    "sens_spec.csv",
    "low_data.csv",
    "correlations.csv",
)


@dataclass(frozen=True)
class Coverage:
    """How often the sweep held each in-domain row out, over its replicates."""

    #: Replicates in which each in-domain row was held out, indexed by row.
    counts: np.ndarray

    @property
    def never_held_out(self) -> int:
        return int((self.counts == 0).sum())

    @property
    def share_never_held_out(self) -> float:
        return self.never_held_out / len(self.counts)


def coverage(held_out: np.ndarray, n_rows: int) -> Coverage:
    """Count, per in-domain row, the replicates whose held-out set contains it.

    ``held_out`` is a record's ``(replicates, rows held out)`` index array; ``n_rows`` the
    number of in-domain rows it indexes.
    """
    return Coverage(np.bincount(np.asarray(held_out).ravel(), minlength=n_rows))


def sensitivity_specificity(
    margins: np.ndarray, positive: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each binary probe's sensitivity and specificity on the rows it was scored on.

    ``margins`` is a record's ``(splits, replicates, rows, 1)`` binary decision margins, which
    point toward the positive class: a probe predicts positive where its margin is above zero,
    as scikit-learn's ``predict`` does. ``positive`` marks, per replicate, which of those rows
    truly are positive. Returns two ``(splits, replicates)`` matrices.
    """
    margins = np.asarray(margins, dtype=float)
    if margins.ndim != 4 or margins.shape[-1] != 1:
        raise ValueError(f"expected binary (splits, replicates, rows, 1) margins, got {margins.shape}")
    predicted = margins[..., 0] > 0
    truth = np.broadcast_to(np.asarray(positive, dtype=bool), predicted.shape)
    sensitivity = (predicted & truth).sum(axis=-1) / truth.sum(axis=-1)
    specificity = (~predicted & ~truth).sum(axis=-1) / (~truth).sum(axis=-1)
    return sensitivity, specificity


#: The record's two test sets, keyed as its arrays are (``accuracies_<name>``, ...).
TEST_SETS = ("in_domain", "out_of_domain")


def record_sensitivity_specificity(record: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every probe's sensitivity and specificity in a binary record, per test set.

    The positive class is the disease class, the second of the record's biological classes
    (``tumor`` on Camelyon, ``cancer`` on PCaBiop) -- which must also be the class a positive
    margin points to: the probe is fitted on class indices into those names, and a positive
    margin predicts ``classes[1]``. In domain, each replicate is scored on the rows it held
    out; out of domain, on every row. The two must average to the balanced accuracy the
    record stored for the same probe, which ties the decision rule here to the probe's own.
    """
    names = list(record["meta"]["biological_classes"])
    classes = [int(c) for c in record["classes"]]
    if len(names) != 2 or classes != [0, 1]:
        raise ValueError(
            f"a binary record's positive margin must point to {names[-1]!r}; the probe's "
            f"classes are {classes} over {names}"
        )
    disease = names[1]
    out = {}
    for name in TEST_SETS:
        if name == "in_domain":
            positive = np.asarray(record["id_label"])[record["held_out"]] == disease
        else:
            positive = np.asarray(record["ood_label"]) == disease
        sensitivity, specificity = sensitivity_specificity(record[f"margins_{name}"], positive)
        balanced = (sensitivity + specificity) / 2
        if not np.allclose(balanced, record[f"accuracies_{name}"], rtol=0, atol=1e-12):
            raise ValueError(
                f"{name}: sensitivity and specificity do not average to the stored balanced "
                "accuracy"
            )
        out[name] = (sensitivity, specificity)
    return out


def associate(
    panel: pd.DataFrame,
    target: str,
    *,
    n_boot: int,
    n_permutations: int,
    seed: int,
    level: float = 0.95,
) -> dict:
    """Spearman rho(CRoMa, ``target``) over a panel of encoders, with its inference.

    ``panel`` holds one row per ranked encoder (``model``, ``croma`` and ``target``). An
    encoder whose ``target`` is undefined (NaN) leaves the panel, and is named in
    ``undefined``. The encoder bootstrap is drawn from ``seed`` over the encoders that remain,
    in model order, so a panel with none undefined shares its draw with every other
    correlation computed on it from the same seed.
    """
    panel = panel.sort_values("model", kind="stable")
    defined = panel[panel[target].notna()]
    inference = spearman_inference(
        defined["croma"].to_numpy(float),
        defined[target].to_numpy(float),
        n_boot=n_boot,
        level=level,
        seed=seed,
        n_permutations=n_permutations,
    )
    return dict(
        n=inference.n,
        rho=inference.rho,
        ci_lo=inference.ci.lo,
        ci_hi=inference.ci.hi,
        level=inference.ci.level,
        n_boot=inference.ci.n_boot,
        p_value=inference.p_value,
        permutation=inference.permutation,
        n_permutations=inference.n_permutations,
        undefined=";".join(panel.loc[panel[target].isna(), "model"].astype(str)),
    )


def nipd_or_undefined(scores: np.ndarray, cramers_v, chance: float) -> float:
    """``croma.nipd`` of a ``(splits, replicates)`` score matrix, or NaN where it is undefined.

    nIPD normalises the degradation by the V = 0 skill above chance. A score whose balanced
    baseline is at or below chance -- a specificity under 0.5 from a probe that calls most
    rows positive -- has no such skill, so its nIPD is undefined rather than an error.
    """
    scores = np.asarray(scores, dtype=float)
    if scores[0].mean() <= chance:
        return float("nan")
    return float(nipd(scores, cramers_v=cramers_v, chance=chance))


# --- the run ------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    """``path`` relative to ``root``, without following the ``output``/``data`` links."""
    path = Path(os.path.abspath(path))
    try:
        return path.relative_to(os.path.abspath(root)).as_posix()
    except ValueError:
        return path.as_posix()


def full_tiles(cohort: str) -> int:
    """The rows each training slide gives at full tiles: the APD study's own sweep."""
    return int(DATASETS[cohort]["num_patches_per_slide"])


def tile_levels(cohort: str, runs: dict) -> tuple[int, ...]:
    """The full tile level, then every reduced level the records hold, largest first."""
    return tuple(sorted({full_tiles(cohort), *runs[cohort]}, reverse=True))


def low_data_cohorts(runs: dict) -> list[str]:
    """The cohorts whose records include a sweep at fewer tiles than the full one."""
    return [c for c in runs if any(t < full_tiles(c) for t in runs[c])]


def binary_cohorts(runs: dict) -> list[str]:
    """The cohorts with two biological classes, where sensitivity and specificity exist."""
    return [c for c in runs if len(DATASETS[c]["biological_classes"]) == 2]


def _record_paths(root: Path, cohort: str, tiles: int, models: list[str]) -> dict[str, Path]:
    """Every roster encoder's record at ``tiles``; the directory may hold no other encoder."""
    directory = root / RECORDS_REL / cohort / f"tiles-{tiles}"
    paths = {model: directory / f"{model}.npz" for model in models}
    missing = sorted(model for model, path in paths.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"{cohort}/tiles-{tiles}: no probe record for {missing}; run "
            "scripts/studies/apd/probe_records.py"
        )
    extra = sorted({p.stem for p in directory.glob("*.npz")} - set(models))
    if extra:
        raise ValueError(f"{cohort}/tiles-{tiles}: records for encoders outside the run: {extra}")
    return paths


def _croma(root: Path, cohort: str, inputs: dict) -> pd.Series:
    """Median CRoMa per encoder, from the run the paper reports for ``cohort``."""
    from loaders import metric_csv

    path = root / metric_csv(cohort)
    inputs[_relative(path, root)] = _sha256(path)
    return auc_report.run_metrics(cohort, root)["croma"]


def _croma_replicates(root: Path, cohort: str, models: list[str], n_boot: int, inputs: dict):
    """Each encoder's median CRoMa per bootstrap replicate (``bootstrap_uncertainty.py``)."""
    from paper_manifest import by_benchmark

    path = root / by_benchmark(DATASETS[cohort]["benchmark"]).studies_rel
    path = path / "bootstrap_replicates.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is absent; run scripts/studies/bootstrap_uncertainty.py")
    inputs[_relative(path, root)] = _sha256(path)
    with np.load(path) as saved:
        index = {m: j for j, m in enumerate(saved["models"].tolist())}
        medians = saved["median"]
    if medians.shape[0] != n_boot:
        raise ValueError(f"{path} holds {medians.shape[0]} replicates, not {n_boot}")
    return medians[:, [index[m] for m in models]]


def _apd_cell(root: Path, cohort: str, model: str, inputs: dict) -> dict[str, np.ndarray]:
    path = root / APD_REL / cohort / f"{model}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no APD cell; run scripts/studies/apd/apd_experiment.py")
    inputs[_relative(path, root)] = _sha256(path)
    cell = json.loads(path.read_text("utf-8"))
    return {regime: np.asarray(cell[key], dtype=float) for regime, key in APD_CELL_KEYS.items()}


def _with_q(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.assign(q_value=np.nan)
    family = frame["in_family"].astype(bool)
    if family.any():
        frame.loc[family, "q_value"] = benjamini_hochberg(frame.loc[family, "p_value"].to_numpy())
    return frame


def sens_spec_correlations(sens_spec: pd.DataFrame, *, n_boot: int, n_permutations: int):
    """rho(CRoMa, nIPD of sensitivity) and of specificity, per binary cohort and regime.

    Over ranked encoders whose nIPD is defined. The tile cohorts' associations form one
    Benjamini--Hochberg family; PCaBiop's slide encoders are descriptive, outside it.
    """
    rows = []
    ranked = sens_spec[~sens_spec["control"].astype(bool)]
    for cohort in dict.fromkeys(ranked["cohort"]):
        for regime in REGIMES:
            panel = ranked[(ranked["cohort"] == cohort) & (ranked["regime"] == regime)]
            for target in ("nipd_sens", "nipd_spec"):
                rows.append(
                    dict(
                        analysis="sens_spec",
                        cohort=cohort,
                        tiles=full_tiles(cohort),
                        regime=regime,
                        target=target,
                        seed=SENS_SPEC_SEED,
                        in_family=cohort in auc_report.FAMILY_DATASETS,
                        **associate(
                            panel,
                            target,
                            n_boot=n_boot,
                            n_permutations=n_permutations,
                            seed=SENS_SPEC_SEED,
                            level=LEVEL,
                        ),
                    )
                )
    return _with_q(pd.DataFrame(rows))


def low_data_correlations(
    low_data: pd.DataFrame,
    accuracies: dict,
    croma_replicates: dict,
    *,
    n_boot: int,
    n_permutations: int,
):
    """rho(CRoMa, nIPD) per low-data cohort, tile level and regime, under both CI designs.

    The encoder interval resamples the encoders; the two-level interval also resamples each
    drawn encoder's probe runs and takes its median CRoMa from one shared slide draw, as
    ``tab:apd-correlation``'s does. Every association is in one Benjamini--Hochberg family.
    """
    rows = []
    ranked = low_data[~low_data["control"].astype(bool)]
    for (cohort, tiles), level in ranked.groupby(["cohort", "tiles"], sort=False):
        models = sorted(level["model"])
        v = training_correlations(cohort)
        chance = 1.0 / len(DATASETS[cohort]["biological_classes"])
        draws = encoder_bootstrap_draws(len(models), n_boot=n_boot, seed=LOW_DATA_SEED)
        for regime in REGIMES:
            row = associate(
                level,
                f"nipd_{regime}",
                n_boot=n_boot,
                n_permutations=n_permutations,
                seed=LOW_DATA_SEED,
                level=LEVEL,
            )
            if row["undefined"]:
                raise ValueError(f"{cohort}/tiles-{tiles}: nIPD undefined for {row['undefined']}")
            replicates = apd_correlation.two_level_replicates(
                croma_replicates[cohort],
                [accuracies[(cohort, int(tiles), m)][regime] for m in models],
                cramers_v=v,
                chance=chance,
                draws=draws,
                seed=LOW_DATA_SEED,
            )
            ci = _percentile_ci(replicates, row["rho"], LEVEL, n_boot)
            rows.append(
                dict(
                    analysis="low_data",
                    cohort=cohort,
                    tiles=int(tiles),
                    regime=regime,
                    target=f"nipd_{regime}",
                    seed=LOW_DATA_SEED,
                    in_family=True,
                    two_level_lo=ci.lo,
                    two_level_hi=ci.hi,
                    **row,
                )
            )
    return _with_q(pd.DataFrame(rows))


def _sens_spec_rows(cohort, model, croma, v, record):
    """One cohort x encoder's per-probe rows and its per-regime summary rows."""
    probes, summary = [], []
    for regime, name in REGIMES.items():
        sens, spec = record_sensitivity_specificity(record)[name]
        splits, replicates = np.indices(sens.shape)
        probes.append(
            pd.DataFrame(
                dict(
                    cohort=cohort,
                    model=model,
                    regime=regime,
                    split=splits.ravel(),
                    cramers_v=np.asarray(v)[splits.ravel()],
                    replicate=replicates.ravel(),
                    sensitivity=sens.ravel(),
                    specificity=spec.ravel(),
                )
            )
        )
        summary.append(
            dict(
                cohort=cohort,
                model=model,
                control=model == CONTROL_MODEL,
                regime=regime,
                croma=croma,
                sens_v0=float(sens[0].mean()),
                spec_v0=float(spec[0].mean()),
                sens_v1=float(sens[-1].mean()),
                spec_v1=float(spec[-1].mean()),
                nipd_sens=nipd_or_undefined(sens, v, SENS_SPEC_CHANCE),
                nipd_spec=nipd_or_undefined(spec, v, SENS_SPEC_CHANCE),
            )
        )
    return probes, summary


def run(
    root: Path = REPO,
    out_dir: Path | None = None,
    *,
    runs: dict | None = None,
    n_boot: int | None = None,
    n_permutations: int | None = None,
) -> Path:
    """Reduce every cohort's records under ``root`` and write the study to ``out_dir``.

    ``runs`` maps each cohort to the tile levels its records hold (default: the records
    study's own). ``n_boot`` and ``n_permutations`` override both analyses' defaults.
    """
    root = Path(root)
    runs = dict(probe_records.RUNS if runs is None else runs)
    out_dir = Path(out_dir) if out_dir is not None else root / STUDY_REL
    binary, low_data = binary_cohorts(runs), low_data_cohorts(runs)
    for cohort in binary:
        if full_tiles(cohort) not in runs[cohort]:
            raise ValueError(f"{cohort}: sensitivity and specificity need the full-tile records")

    inputs: dict[str, str] = {}
    accuracies: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    croma_replicates: dict[str, np.ndarray] = {}
    coverage_rows, count_frames, probe_frames, sens_spec_rows, low_data_rows = [], [], [], [], []
    for cohort in runs:
        croma = _croma(root, cohort, inputs)
        models = [str(m) for m in croma.index]
        v = training_correlations(cohort)
        first = None
        for tiles in sorted(runs[cohort], reverse=True):
            for model, path in _record_paths(root, cohort, tiles, models).items():
                record = probe_records.read_record(path)
                inputs[_relative(path, root)] = _sha256(path)
                where = f"{cohort}/tiles-{tiles}/{model}"
                if not np.allclose(record["meta"]["cramers_v"], v, rtol=0, atol=1e-12):
                    raise ValueError(f"{where}: the record's V grid is not the APD study's")
                if first is None:
                    first = record
                elif not (
                    np.array_equal(record["held_out"], first["held_out"])
                    and np.array_equal(record["id_sample_id"], first["id_sample_id"])
                ):
                    raise ValueError(f"{where}: holds out other rows than the cohort's others")
                accuracies[(cohort, tiles, model)] = {
                    regime: np.asarray(record[f"accuracies_{name}"], dtype=float)
                    for regime, name in REGIMES.items()
                }
                if cohort in binary and tiles == full_tiles(cohort):
                    probes, summary = _sens_spec_rows(cohort, model, float(croma[model]), v, record)
                    probe_frames.extend(probes)
                    sens_spec_rows.extend(summary)

        n_rows = len(first["id_sample_id"])
        cov = coverage(first["held_out"], n_rows)
        coverage_rows.append(
            dict(
                cohort=cohort,
                rows=n_rows,
                replicates=int(first["held_out"].shape[0]),
                never_held_out=cov.never_held_out,
                share_never_held_out=cov.share_never_held_out,
                min_times_held_out=int(cov.counts.min()),
                median_times_held_out=float(np.median(cov.counts)),
                max_times_held_out=int(cov.counts.max()),
            )
        )
        count_frames.append(
            pd.DataFrame(
                dict(
                    cohort=cohort,
                    row=np.arange(n_rows),
                    sample_id=first["id_sample_id"],
                    group_id=first["id_group_id"],
                    label=first["id_label"],
                    confounder=first["id_confounder"],
                    times_held_out=cov.counts,
                )
            )
        )

        full = full_tiles(cohort)
        for model in models:
            # The full level is the APD study's own cell, which tab:apd-correlation and the
            # AUROC study read; a full-tile record, where there is one, must still reproduce
            # it. Every cohort's cells are checked and fingerprinted, not only the low-data
            # ones', so a re-run cell makes the sensitivity and specificity rows stale too.
            cell = _apd_cell(root, cohort, model, inputs)
            recorded = accuracies.get((cohort, full, model))
            tolerance = probe_records.REPRODUCTION_TOLERANCE
            if recorded is not None and not all(
                np.allclose(recorded[r], cell[r], rtol=0, atol=tolerance) for r in REGIMES
            ):
                raise ValueError(f"{cohort}/{model}: the full-tile record no longer reproduces")
            accuracies[(cohort, full, model)] = cell

        if cohort not in low_data:
            continue
        chance = 1.0 / len(DATASETS[cohort]["biological_classes"])
        ranked = sorted(m for m in models if m != CONTROL_MODEL)
        croma_replicates[cohort] = _croma_replicates(
            root, cohort, ranked, n_boot or LOW_DATA_N_BOOT, inputs
        )
        for tiles in tile_levels(cohort, runs):
            for model in models:
                acc = accuracies[(cohort, tiles, model)]
                low_data_rows.append(
                    dict(
                        cohort=cohort,
                        tiles=tiles,
                        full_tiles=tiles == full,
                        model=model,
                        control=model == CONTROL_MODEL,
                        croma=float(croma[model]),
                        bacc_v0_id=float(acc["id"][0].mean()),
                        bacc_v0_ood=float(acc["ood"][0].mean()),
                        nipd_id=float(nipd(acc["id"], cramers_v=v, chance=chance)),
                        nipd_ood=float(nipd(acc["ood"], cramers_v=v, chance=chance)),
                    )
                )

    sens_spec = pd.DataFrame(sens_spec_rows)
    low_data_frame = pd.DataFrame(low_data_rows)
    correlations = pd.concat(
        [
            sens_spec_correlations(
                sens_spec,
                n_boot=n_boot or SENS_SPEC_N_BOOT,
                n_permutations=n_permutations or SENS_SPEC_N_PERMUTATIONS,
            ),
            low_data_correlations(
                low_data_frame,
                accuracies,
                croma_replicates,
                n_boot=n_boot or LOW_DATA_N_BOOT,
                n_permutations=n_permutations or LOW_DATA_N_PERMUTATIONS,
            ),
        ],
        ignore_index=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "coverage.csv": pd.DataFrame(coverage_rows),
        "held_out_counts.csv": pd.concat(count_frames, ignore_index=True),
        "sens_spec_probes.csv": pd.concat(probe_frames, ignore_index=True),
        "sens_spec.csv": sens_spec,
        "low_data.csv": low_data_frame,
        "correlations.csv": correlations,
    }
    for name, frame in frames.items():
        frame.to_csv(out_dir / name, index=False)
    info = {
        "runs": {cohort: list(levels) for cohort, levels in runs.items()},
        "tile_levels": {cohort: list(tile_levels(cohort, runs)) for cohort in low_data},
        "binary_cohorts": binary,
        "low_data_cohorts": low_data,
        "sens_spec_chance": SENS_SPEC_CHANCE,
        "control": CONTROL_MODEL,
        "outputs": {name: _sha256(out_dir / name) for name in OUTPUTS},
        "inputs": dict(sorted(inputs.items())),
    }
    (out_dir / "study.json").write_text(json.dumps(info, indent=2) + "\n")
    return out_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", type=Path, default=None, help=f"default: {STUDY_REL}")
    args = p.parse_args(argv)
    out_dir = run(REPO, args.out_dir)
    for row in pd.read_csv(out_dir / "coverage.csv").itertuples():
        flag = "  (coverage below 90%)" if row.share_never_held_out > 0.1 else ""
        print(
            f"{row.cohort}: {row.never_held_out} of {row.rows} in-domain rows never held out "
            f"({row.share_never_held_out:.1%}){flag}"
        )
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
