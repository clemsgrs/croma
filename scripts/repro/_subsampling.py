"""Load the cohort-subsampling study and compute its stability measures.

The study (``scripts/studies/subsampling_stability.py``) leaves one record per replicate --
a ``scores.csv`` of every encoder's RI, MaRI and CRoMa scores and a ``replicate.json`` of
the k it chose. This module turns those into the measures behind
``tab:subsampling-stability``, ``fig:subsampling-stability`` and the macros, each computed
per cohort x fraction x metric over the *ranked* encoders. The control is computed by the
study (it belongs to the production roster, and so to the shared k) but is dropped here, at
the load, so no measure can rank it.

A study is read whole or not at all: a missing replicate, or a full-cohort run whose
reproduction check did not pass, raises :class:`IncompleteStudyError`. Nothing warns and
skips.

Measures (all on the ranked encoders; "Kendall's tau_K" throughout, so it is never read as
the MaRI temperature tau):

* **ranking agreement** -- Kendall's tau_K (tau-b) between a replicate's scores and the same
  metric's full-cohort scores; median and IQR over replicates;
* **rank-interval width** -- per encoder, the 2.5-97.5 percentile range of its rank (1 =
  best) across replicates; median over encoders;
* **drift** -- per encoder and replicate, subsample score minus full-cohort score; median
  over both, signed, in the metric's own units (a curve per metric, never compared across);
* **paired comparison** -- per replicate, tau_K(CRoMa) minus tau_K(RI), and minus
  tau_K(MaRI), both at the re-selected k; median, 2.5-97.5 percentile range, and the number
  of replicates where CRoMa's is strictly higher;
* **k and support** -- the range of the re-selected shared k, the share of encoder x
  replicate pairs whose k* differs from the encoder's full-cohort k*, and the median RI
  support beside its full-cohort value.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _path in (REPO / "scripts" / "studies", REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import subsampling_stability as study  # noqa: E402
from plotting.style import CONTROL_MODEL  # noqa: E402

#: The six scores whose stability is measured, in reporting order: column -> label.
METRICS = {
    "ri": "RI (re-selected k)",
    "ri_fixed_k": "RI (fixed k)",
    "mari": "MaRI (re-selected k)",
    "mari_fixed_k": "MaRI (fixed k)",
    "croma": "median CRoMa",
    "croma_ltm_alpha": "LTM10",
}

#: The paired comparison: CRoMa's agreement minus each of these, at the re-selected k.
PAIRED_REFERENCE = "croma"
PAIRED_AGAINST = ("ri", "mari")

#: Percentiles of the rank interval and of the paired-difference range.
INTERVAL = (2.5, 97.5)


class IncompleteStudyError(FileNotFoundError):
    """The study tree is missing a record the design calls for, or was drawn differently."""


@dataclass(frozen=True)
class Study:
    #: One row per (cohort, fraction, replicate, ranked encoder); fraction 100 is the full run.
    scores: pd.DataFrame
    #: One row per (cohort, fraction, replicate): the replicate's ``replicate.json``.
    meta: pd.DataFrame
    design: dict


def _records(root: Path, design: dict):
    for cohort in design["cohorts"]:
        yield cohort, study.FULL, 0
        for fraction in design["fractions"]:
            for replicate in range(int(design["n_replicates"])):
                yield cohort, int(fraction), replicate


def load(root: Path | None = None, *, expected: dict | None = None) -> Study:
    """Read every record of the study at ``root``; raise if any is missing.

    ``expected`` is the design the tree must have been drawn under -- by default the
    study's own, which is the only design the paper is built from.
    """
    root = Path(root) if root is not None else study.study_root()
    expected = study.design() if expected is None else expected
    design_path = root / study.DESIGN
    if not design_path.exists():
        raise IncompleteStudyError(
            f"{design_path} is missing; run scripts/studies/subsampling_stability.py sweep"
        )
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design != expected:
        raise IncompleteStudyError(
            f"{design_path} records a different design from the one expected: "
            f"{design} != {expected}"
        )

    missing: list[str] = []
    frames, metas = [], []
    for cohort, fraction, replicate in _records(root, design):
        where = study.replicate_dir(root, cohort, fraction, replicate)
        needed = [where / study.SCORES, where / study.REPLICATE_META]
        if fraction == study.FULL:
            needed.append(where / study.REPRODUCTION)
        absent = [p for p in needed if not p.exists()]
        if absent:
            missing.extend(str(p.relative_to(root)) for p in absent)
            continue
        if fraction == study.FULL:
            report = json.loads((where / study.REPRODUCTION).read_text(encoding="utf-8"))
            if report.get("identical") is not True:
                missing.append(f"{where.relative_to(root)}: reproduction did not pass")
                continue
        meta = json.loads((where / study.REPLICATE_META).read_text(encoding="utf-8"))
        if CONTROL_MODEL not in meta["roster"]:
            raise IncompleteStudyError(
                f"{where}: the control {CONTROL_MODEL} is not in the roster the shared k "
                "was selected over"
            )
        frame = pd.read_csv(where / study.SCORES)
        frame = frame[frame["model"] != CONTROL_MODEL]
        frames.append(frame.assign(cohort=cohort, fraction=fraction, replicate=replicate))
        metas.append(
            {"cohort": cohort, "fraction": fraction, "replicate": replicate, **meta}
        )
    if missing:
        raise IncompleteStudyError(
            f"{root}: {len(missing)} record(s) missing or failed, e.g. "
            + "; ".join(missing[:5])
        )
    scores = pd.concat(frames, ignore_index=True)
    rosters = scores.groupby(["cohort", "fraction", "replicate"])["model"].apply(
        lambda m: tuple(sorted(m))
    )
    for cohort, per_cohort in rosters.groupby(level="cohort"):
        if per_cohort.nunique() != 1:
            raise IncompleteStudyError(f"{cohort}: the replicates ranked different encoders")
    return Study(scores=scores, meta=pd.DataFrame(metas), design=design)


# --------------------------------------------------------------------------- measures


def _full(scores: pd.DataFrame) -> pd.DataFrame:
    return scores[scores["fraction"] == study.FULL]


def kendall_tau_k(data: Study) -> pd.DataFrame:
    """Kendall's tau_K of each replicate against its cohort's full-cohort scores, per metric."""
    rows = []
    for cohort, per_cohort in data.scores.groupby("cohort", sort=False):
        full = _full(per_cohort).set_index("model")
        for (fraction, replicate), rep in per_cohort.groupby(
            ["fraction", "replicate"], sort=True
        ):
            rep = rep.set_index("model").loc[full.index]
            for metric in METRICS:
                tau = kendalltau(rep[metric], full[metric]).statistic
                if not np.isfinite(tau):
                    raise ValueError(f"{cohort} f{fraction} r{replicate}: {metric} is constant")
                rows.append(
                    {
                        "cohort": cohort,
                        "fraction": int(fraction),
                        "replicate": int(replicate),
                        "metric": metric,
                        "tau_k": float(tau),
                    }
                )
    return pd.DataFrame(rows)


def ranking_summary(data: Study) -> pd.DataFrame:
    """Ranking agreement, rank-interval width and drift, per cohort x fraction x metric."""
    tau = kendall_tau_k(data)
    rows = []
    for (cohort, fraction), group in data.scores.groupby(["cohort", "fraction"], sort=False):
        full = _full(data.scores[data.scores["cohort"] == cohort]).set_index("model")
        for metric in METRICS:
            # Rank 1 = the highest score within the replicate; ties share their mean rank.
            ranks = group.assign(
                rank=group.groupby("replicate")[metric].rank(ascending=False, method="average")
            )
            lo, hi = INTERVAL
            width = ranks.groupby("model")["rank"].agg(
                lambda r: float(np.percentile(r, hi) - np.percentile(r, lo))
            )
            drift = group[metric].to_numpy() - full.loc[group["model"], metric].to_numpy()
            t = tau[
                (tau["cohort"] == cohort)
                & (tau["fraction"] == fraction)
                & (tau["metric"] == metric)
            ]["tau_k"]
            rows.append(
                {
                    "cohort": cohort,
                    "fraction": int(fraction),
                    "metric": metric,
                    "tau_k_median": float(np.median(t)),
                    "tau_k_q25": float(np.percentile(t, 25)),
                    "tau_k_q75": float(np.percentile(t, 75)),
                    "rank_width_median": float(np.median(width)),
                    "drift_median": float(np.median(drift)),
                    "n_replicates": int(group["replicate"].nunique()),
                    "n_encoders": int(group["model"].nunique()),
                }
            )
    return _ordered(pd.DataFrame(rows), data)


def paired_summary(data: Study) -> pd.DataFrame:
    """tau_K(CRoMa) minus tau_K(RI) and tau_K(MaRI), per replicate; per cohort x fraction."""
    tau = kendall_tau_k(data)
    tau = tau[tau["fraction"] != study.FULL]
    wide = tau.pivot_table(
        index=["cohort", "fraction", "replicate"], columns="metric", values="tau_k"
    )
    rows = []
    for (cohort, fraction), group in wide.groupby(level=["cohort", "fraction"], sort=False):
        for against in PAIRED_AGAINST:
            diff = (group[PAIRED_REFERENCE] - group[against]).to_numpy()
            lo, hi = np.percentile(diff, INTERVAL)
            rows.append(
                {
                    "cohort": cohort,
                    "fraction": int(fraction),
                    "versus": against,
                    "tau_k_diff_median": float(np.median(diff)),
                    "tau_k_diff_lo": float(lo),
                    "tau_k_diff_hi": float(hi),
                    "n_croma_higher": int((diff > 0).sum()),
                    "n_replicates": int(len(diff)),
                }
            )
    return _ordered(pd.DataFrame(rows), data)


def k_support_summary(data: Study) -> pd.DataFrame:
    """The re-selected shared k's range, k* changes and RI support, per cohort x fraction."""
    rows = []
    for (cohort, fraction), group in data.scores.groupby(["cohort", "fraction"], sort=False):
        if fraction == study.FULL:
            continue
        full = _full(data.scores[data.scores["cohort"] == cohort]).set_index("model")
        meta = data.meta[(data.meta["cohort"] == cohort) & (data.meta["fraction"] == fraction)]
        full_meta = data.meta[
            (data.meta["cohort"] == cohort) & (data.meta["fraction"] == study.FULL)
        ].iloc[0]
        changed = group["k_star"].to_numpy() != full.loc[group["model"], "k_star"].to_numpy()
        rows.append(
            {
                "cohort": cohort,
                "fraction": int(fraction),
                "shared_k_min": int(meta["shared_k"].min()),
                "shared_k_max": int(meta["shared_k"].max()),
                "full_shared_k": int(full_meta["shared_k"]),
                "k_star_changed_share": float(changed.mean()),
                "support_median": float(np.median(group["support"])),
                "full_support_median": float(np.median(full["support"])),
                "k_grid_max": int(meta["k_grid_max"].max()),
                "min_usable_neighbours": int(meta["min_usable_neighbours"].min()),
            }
        )
    return _ordered(pd.DataFrame(rows), data)


# --------------------------------------------------------------------------- macros

#: Macro spelling of each metric (LaTeX command names take letters only).
MACRO_METRIC = {
    "ri": "Ri",
    "ri_fixed_k": "RiFixedK",
    "mari": "Mari",
    "mari_fixed_k": "MariFixedK",
    "croma": "Croma",
    "croma_ltm_alpha": "Ltm",
}
MACRO_FRACTION = {25: "Quarter", 50: "Half", 75: "ThreeQuarter"}


def macros(data: Study) -> list[str]:
    r"""The study's ``\newcommand`` lines: numbers only, never a verdict.

    Per cohort (macro family = the cohort's ``paper_manifest`` prefix):
    ``\<P>SubsampleHalfKendall<Metric>`` -- median tau_K at 50%;
    ``\<P>SubsampleHalfCromaMinus{Ri,Mari}`` -- the paired difference's median at 50%, with
    ``...Range`` (its 2.5-97.5 percentile range) and ``...Higher`` (replicates where CRoMa's
    tau_K is higher); ``\<P>SubsampleQuarterSharedKRange`` -- the re-selected shared k's range
    at 25%. ``\SubsampleNReplicates`` is the replicate count those counts are out of.
    """
    from _paper_tables import bare_num, num_math
    from paper_manifest import by_benchmark

    ranking = ranking_summary(data)
    paired = paired_summary(data)
    ks = k_support_summary(data)

    def pick(frame, **where):
        mask = np.ones(len(frame), dtype=bool)
        for key, value in where.items():
            mask &= frame[key].to_numpy() == value
        if mask.sum() != 1:
            raise IncompleteStudyError(f"no single summary row for {where}")
        return frame[mask].iloc[0]

    half, quarter = 50, 25
    lines = [rf"\newcommand{{\SubsampleNReplicates}}{{{int(data.design['n_replicates'])}}}"]
    for cohort in data.design["cohorts"]:
        p = by_benchmark(cohort).prefix
        h = MACRO_FRACTION[half]
        for metric, name in MACRO_METRIC.items():
            row = pick(ranking, cohort=cohort, fraction=half, metric=metric)
            lines.append(
                rf"\newcommand{{\{p}Subsample{h}Kendall{name}}}"
                rf"{{{num_math(row['tau_k_median'])}}}"
            )
        for against in PAIRED_AGAINST:
            row = pick(paired, cohort=cohort, fraction=half, versus=against)
            stem = rf"\{p}Subsample{h}CromaMinus{MACRO_METRIC[against]}"
            lo, hi = bare_num(row["tau_k_diff_lo"]), bare_num(row["tau_k_diff_hi"])
            lines += [
                rf"\newcommand{{{stem}}}{{{num_math(row['tau_k_diff_median'])}}}",
                rf"\newcommand{{{stem}Range}}{{$[{lo}, {hi}]$}}",
                rf"\newcommand{{{stem}Higher}}{{{int(row['n_croma_higher'])}}}",
            ]
        row = pick(ks, cohort=cohort, fraction=quarter)
        lines.append(
            rf"\newcommand{{\{p}Subsample{MACRO_FRACTION[quarter]}SharedKRange}}"
            rf"{{${int(row['shared_k_min'])}$--${int(row['shared_k_max'])}$}}"
        )
    return lines


def _ordered(frame: pd.DataFrame, data: Study) -> pd.DataFrame:
    """Cohorts in the design's order, then fractions ascending: a deterministic layout."""
    order = {c: i for i, c in enumerate(data.design["cohorts"])}
    frame = frame.assign(_cohort=frame["cohort"].map(order))
    if "metric" in frame:
        frame = frame.assign(_metric=frame["metric"].map(list(METRICS).index))
    if "versus" in frame:
        frame = frame.assign(_versus=frame["versus"].map(PAIRED_AGAINST.index))
    by = ["_cohort", "fraction"] + [c for c in ("_metric", "_versus") if c in frame]
    return frame.sort_values(by).drop(columns=[c for c in frame if c.startswith("_")]).reset_index(
        drop=True
    )
