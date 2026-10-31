"""Each robustness fine-tune against its parent encoder, on the three PathoROB tile cohorts.

A robustness-targeted fine-tune and its parent share the architecture and the starting
weights and differ by one adaptation step aimed at acquisition robustness, so each pair is a
natural experiment. The pairs come from ``scripts/bench/model_metadata.csv``: every tile row
whose ``variant_role`` is ``robustness-finetune``, with its ``parent_model``. Nothing here
names a model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "src", REPO / "scripts" / "bench", REPO / "scripts" / "repro"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from croma.metrics.base import EVALUATION_DESIGN_ALL  # noqa: E402
from croma.metrics.bootstrap import shared_draw_tail_statistics  # noqa: E402
from croma.metrics.croma import CROMA_HEADLINE_M  # noqa: E402
from croma.metrics.tail import compute_tail_metrics  # noqa: E402
from croma.downstream import nipd  # noqa: E402
from plotting.style import CONTROL_MODEL  # noqa: E402

METADATA_CSV = REPO / "scripts" / "bench" / "model_metadata.csv"
FINETUNE_ROLE = "robustness-finetune"
HEADLINE_COL = f"croma_m{int(CROMA_HEADLINE_M)}"

#: The CRoMa summaries bootstrapped on a shared draw: (``TailBootstrap`` statistic, the
#: study's column stem, the run's ``metrics.csv`` column the point must reproduce).
TAIL_SUMMARIES = (
    ("median", "croma", "croma"),
    ("f0", "f0", "croma_f0"),
    ("ltm", "ltm", "croma_ltm_alpha"),
)
#: Scores reported as point differences only: (column stem, ``metrics.csv`` column). RI and
#: MaRI are read at the run's shared k.
POINT_SCORES = (("ri", "ri"), ("mari", "mari"), ("bio_bacc", "bio_knn_bacc"))
#: Tolerance for a per-sample summary to count as reproducing the run's pooled value.
_SUMMARY_ATOL = 1e-9


@dataclass(frozen=True)
class FinetunePair:
    """One fine-tune, its declared parent, and what its authors disclosed about it.

    ``method_source`` and ``data_source`` are the citation key or URL where the authors
    disclose the adaptation method and the adaptation data; ``None`` means undisclosed,
    which is what an empty metadata cell says. It is never inferred. ``benchmark_selection``
    names a benchmark the authors disclosed using during development (ADR-0020).
    """

    finetune: str
    parent: str
    method_source: str | None = None
    data_source: str | None = None
    benchmark_selection: str | None = None


def read_metadata(path: Path = METADATA_CSV) -> pd.DataFrame:
    """The committed model metadata, with every cell kept as written.

    ``keep_default_na=False`` because pandas otherwise turns the literal ``n/a`` disclosure
    marker into NaN.
    """
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def _cell(row: pd.Series, column: str) -> str | None:
    value = str(row.get(column, "") or "").strip()
    return value or None


def finetune_pairs(metadata: pd.DataFrame) -> list[FinetunePair]:
    """Every tile-panel robustness fine-tune with its declared parent, in panel order.

    Raises when a fine-tune declares no parent, or one that is not a tile-panel model: a
    pair that cannot be formed must not silently drop out of the comparison.
    """
    tile = metadata[metadata["panel"] == "tile"].sort_values("panel_order")
    tile_models = set(tile["model"].astype(str))
    pairs = []
    for _, row in tile[tile["variant_role"] == FINETUNE_ROLE].iterrows():
        finetune, parent = str(row["model"]), _cell(row, "parent_model")
        if parent is None:
            raise ValueError(f"robustness fine-tune {finetune!r} declares no parent_model")
        if parent not in tile_models:
            raise ValueError(
                f"robustness fine-tune {finetune!r} declares parent {parent!r}, "
                "which is not a tile-panel model"
            )
        pairs.append(
            FinetunePair(
                finetune=finetune,
                parent=parent,
                method_source=_cell(row, "adaptation_method_source"),
                data_source=_cell(row, "adaptation_data_source"),
                benchmark_selection=_cell(row, "benchmark_selection"),
            )
        )
    return pairs


def _require_roster(pairs: list[FinetunePair], roster: set[str], benchmark: str) -> None:
    """Fail when a pair member is not in the run: a missing parent is a broken comparison."""
    for pair in pairs:
        for role, model in (("parent", pair.parent), ("fine-tune", pair.finetune)):
            if model not in roster:
                raise ValueError(
                    f"{role} {model!r} of the {pair.parent} -> {pair.finetune} pair is absent "
                    f"from the {benchmark} run's roster"
                )


def _single(metrics: pd.DataFrame, column: str, benchmark: str) -> object:
    values = metrics[column].unique()
    if len(values) != 1:
        raise ValueError(f"{benchmark}: expected one {column} across the run, got {values}")
    return values[0]


def _aligned_samples(
    per_sample: pd.DataFrame, models: list[str], strata: tuple[str, ...], benchmark: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sample CRoMa joined on ``sample_id``: one column per model, plus the design.

    Both encoders of a pair score the same samples, so the join must be total; a sample one
    of them lacks, or a sample whose slide or stratum differs between them, fails.
    """
    sub = per_sample[per_sample["model"].isin(models)]
    for model in models:
        ids = sub.loc[sub["model"] == model, "sample_id"]
        if ids.empty or ids.duplicated().any():
            raise ValueError(f"{benchmark}: {model!r} needs one per-sample row per sample_id")
    wide = sub.pivot(index="sample_id", columns="model", values=HEADLINE_COL)[models]
    if not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError(f"{benchmark}: {models} do not all score the same samples")
    design = sub[["sample_id", "group_id", *strata]].drop_duplicates().set_index("sample_id")
    if design.index.duplicated().any():
        raise ValueError(f"{benchmark}: a sample's group or stratum differs between models")
    return wide, design.loc[wide.index]


def _stratum_labels(design: pd.DataFrame, strata: tuple[str, ...]) -> np.ndarray:
    return design[list(strata)].astype(str).agg("\x1f".join, axis=1).to_numpy()


def paired_representation(
    pairs: list[FinetunePair],
    metrics: pd.DataFrame,
    per_sample: pd.DataFrame,
    *,
    benchmark: str,
    strata: tuple[str, ...],
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> pd.DataFrame:
    """Parent, fine-tune and fine-tune-minus-parent representation scores on one run.

    ``metrics`` and ``per_sample`` are the run's ``metrics.csv`` and per-sample CRoMa rows
    (``sample_id``, ``group_id``, the ``strata`` columns, ``croma_m5``). The median CRoMa,
    ``F(0)`` and ``LTM_10`` differences get paired percentile intervals from one stratified
    cluster bootstrap: every replicate resamples ``group_id`` clusters within ``strata`` and
    applies that draw to every encoder. The per-sample summaries must reproduce the run's
    pooled values, which holds for the ``all`` evaluation design only, so any other design
    fails. RI, MaRI and biological k-NN balanced accuracy are point differences at the run's
    shared k. Ranks are by median CRoMa over the ranked panel (the natural-image control
    holds none), 1 the highest; ``rank_change`` counts places the fine-tune gains.
    """
    roster = set(metrics["model"].astype(str))
    _require_roster(pairs, roster, benchmark)
    design_name = _single(metrics, "evaluation_design", benchmark)
    if design_name != EVALUATION_DESIGN_ALL:
        raise ValueError(
            f"{benchmark}: the pooled CRoMa equals the per-sample median only under the "
            f"{EVALUATION_DESIGN_ALL!r} evaluation design, got {design_name!r}"
        )
    if int(_single(metrics, "croma_m", benchmark)) != int(CROMA_HEADLINE_M):
        raise ValueError(f"{benchmark}: run CRoMa is not at the headline m={CROMA_HEADLINE_M}")
    alpha = float(_single(metrics, "croma_alpha", benchmark))

    models = list(dict.fromkeys(m for p in pairs for m in (p.parent, p.finetune)))
    wide, design = _aligned_samples(per_sample, models, strata, benchmark)
    boot = shared_draw_tail_statistics(
        {m: wide[m].to_numpy(dtype=float) for m in models},
        design["group_id"].to_numpy(),
        strata=_stratum_labels(design, strata),
        alpha=alpha,
        n_boot=n_boot,
        level=level,
        seed=seed,
    )

    by_model = metrics.set_index("model")
    for model in models:
        for statistic, _, column in TAIL_SUMMARIES:
            point, stored = boot.ci[statistic][model].point, float(by_model.loc[model, column])
            if not np.isclose(point, stored, rtol=0.0, atol=_SUMMARY_ATOL):
                raise ValueError(
                    f"{benchmark}: {model!r} per-sample {statistic} {point!r} does not "
                    f"reproduce the run's {column} {stored!r}"
                )

    ranked = by_model.drop(index=CONTROL_MODEL, errors="ignore")["croma"].astype(float)

    def rank(model: str) -> int:
        return int(1 + (ranked > float(ranked[model])).sum())

    rows = []
    for pair in pairs:
        parent, finetune = by_model.loc[pair.parent], by_model.loc[pair.finetune]
        if int(parent["k"]) != int(finetune["k"]):
            raise ValueError(f"{benchmark}: {pair.parent} and {pair.finetune} differ in k")
        row: dict[str, object] = {
            "benchmark": benchmark,
            "finetune": pair.finetune,
            "parent": pair.parent,
            "k": int(parent["k"]),
            "n_samples": int(len(wide)),
            "n_groups": int(design["group_id"].nunique()),
            "strata": "+".join(strata),
            "n_boot": int(n_boot),
            "level": float(level),
            "seed": int(seed),
            "n_ranked": int(len(ranked)),
        }
        for statistic, stem, _ in TAIL_SUMMARIES:
            delta = boot.difference_ci(statistic, pair.finetune, pair.parent)
            row[f"parent_{stem}"] = boot.ci[statistic][pair.parent].point
            row[f"finetune_{stem}"] = boot.ci[statistic][pair.finetune].point
            row[f"delta_{stem}"] = delta.point
            row[f"delta_{stem}_lo"] = delta.lo
            row[f"delta_{stem}_hi"] = delta.hi
        for stem, column in POINT_SCORES:
            row[f"parent_{stem}"] = float(parent[column])
            row[f"finetune_{stem}"] = float(finetune[column])
            row[f"delta_{stem}"] = float(finetune[column]) - float(parent[column])
        row["parent_rank"] = rank(pair.parent)
        row["finetune_rank"] = rank(pair.finetune)
        row["rank_change"] = row["parent_rank"] - row["finetune_rank"]
        row.update(per_sample_shifts(wide[pair.parent], wide[pair.finetune], alpha=alpha))
        rows.append(row)
    return pd.DataFrame(rows)


def per_sample_shifts(parent: pd.Series, finetune: pd.Series, *, alpha: float) -> dict:
    """How the fine-tune moves each sample's CRoMa, over samples aligned on ``sample_id``.

    * ``share_rises``: the fraction of samples whose CRoMa is strictly higher.
    * ``parent_low_decile_median_shift``: the median shift over the parent's lower tail,
      the samples at or below its ``alpha`` quantile -- the same set its ``LTM`` averages.
    * ``parent_nonpositive_turn_positive``: of the parent's confounder-dominant samples
      (``CRoMa <= 0``, the ones ``F(0)`` counts), the fraction the fine-tune puts above 0.
    """
    p = parent.to_numpy(dtype=float)
    shift = finetune.to_numpy(dtype=float) - p
    low = p <= compute_tail_metrics(p, alpha=alpha).q_alpha
    nonpositive = p <= 0.0
    turned = (finetune.to_numpy(dtype=float)[nonpositive] > 0.0).mean() if nonpositive.any() else np.nan
    return {
        "share_rises": float((shift > 0.0).mean()),
        "parent_low_decile_n": int(low.sum()),
        "parent_low_decile_median_shift": float(np.median(shift[low])),
        "parent_nonpositive_n": int(nonpositive.sum()),
        "parent_nonpositive_turn_positive": float(turned),
    }


#: nIPD domains: (column suffix, the APD study cell's accuracy-matrix key).
NIPD_DOMAINS = (("id", "id_test_accuracies"), ("ood", "ood_test_accuracies"))


def paired_downstream(
    pairs: list[FinetunePair],
    cells: dict[str, dict],
    apd_rows: pd.DataFrame,
    *,
    benchmark: str,
    cramers_v,
    chance: float,
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> pd.DataFrame:
    """Parent, fine-tune and difference in nIPD, in and out of domain, on one cohort.

    ``cells`` maps each encoder to its APD study cell (the ``(n_splits, n_runs)`` accuracy
    matrices), ``apd_rows`` is the study table whose ``nipd_id``/``nipd_ood`` the cells must
    reproduce. The difference's interval resamples the probe runs: the run arrangement is a
    pure function of the probe seed, so run ``j`` is the same experiment for every encoder,
    and each replicate draws one set of run indices applied to both encoders of the pair and
    to every Cramér's-V point. Its spread is training-run noise on a fixed cohort, not
    cohort sampling.
    """
    table = apd_rows.set_index("model")
    rng = np.random.default_rng(seed)
    lo_q = (1.0 - level) / 2.0 * 100.0
    rows = []
    for pair in pairs:
        for model in (pair.parent, pair.finetune):
            if model not in cells or model not in table.index:
                raise ValueError(f"no APD cell for {model!r} on {benchmark}")
        row: dict[str, object] = {"benchmark": benchmark, "finetune": pair.finetune}
        for domain, key in NIPD_DOMAINS:
            parent = np.asarray(cells[pair.parent][key], dtype=float)
            finetune = np.asarray(cells[pair.finetune][key], dtype=float)
            if parent.shape != finetune.shape:
                raise ValueError(
                    f"{benchmark}: {pair.parent} and {pair.finetune} {key} shapes differ "
                    f"({parent.shape} vs {finetune.shape}), so their runs cannot be paired"
                )
            points = {}
            for model, acc in ((pair.parent, parent), (pair.finetune, finetune)):
                points[model] = nipd(acc, cramers_v=cramers_v, chance=chance)
                stored = float(table.loc[model, f"nipd_{domain}"])
                if not np.isclose(points[model], stored, rtol=0.0, atol=_SUMMARY_ATOL):
                    raise ValueError(
                        f"{benchmark}: {model!r} nipd_{domain} {points[model]!r} recomputed "
                        f"from its cell does not reproduce the study table's {stored!r}"
                    )
            runs = rng.integers(0, parent.shape[1], (int(n_boot), parent.shape[1]))
            deltas = np.array(
                [
                    nipd(finetune[:, r], cramers_v=cramers_v, chance=chance)
                    - nipd(parent[:, r], cramers_v=cramers_v, chance=chance)
                    for r in runs
                ]
            )
            row[f"parent_nipd_{domain}"] = points[pair.parent]
            row[f"finetune_nipd_{domain}"] = points[pair.finetune]
            row[f"delta_nipd_{domain}"] = points[pair.finetune] - points[pair.parent]
            row[f"delta_nipd_{domain}_lo"] = float(np.percentile(deltas, lo_q))
            row[f"delta_nipd_{domain}_hi"] = float(np.percentile(deltas, 100.0 - lo_q))
            row["nipd_runs"] = int(parent.shape[1])
        rows.append(row)
    return pd.DataFrame(rows)


# --- the study over the real runs ---------------------------------------------------------

#: The tile cohorts the comparison runs on, with the strata of each cohort's cluster
#: bootstrap. Camelyon and TCGA-4x4 slides each hold one class, so the draw resamples slides
#: within class x centre and keeps every cell's size; 26 of Tolkach-ESCA's 43 cases hold
#: tiles of several classes, so a case cannot nest in a class and the draw stratifies by
#: centre alone (five cases per centre and class by design).
STRATA: dict[str, tuple[str, ...]] = {
    "pathorob-camelyon": ("label", "confounder"),
    "pathorob-tcga-4x4": ("label", "confounder"),
    "pathorob-tolkach-esca": ("confounder",),
}
BENCHMARKS = tuple(STRATA)
N_BOOT = 2000
LEVEL = 0.95
SEED = 12345
OUT_CSV = REPO / "output" / "studies" / "finetune-pairs" / "finetune_pairs.csv"
#: The APD study table (``scripts/studies/apd``) the nIPD points must reproduce.
APD_CSV_REL = "output/studies/apd/apd.csv"
_PER_SAMPLE_COLUMNS = ["model", "sample_id", "group_id", "label", "confounder", HEADLINE_COL]


def input_rels(benchmark: str) -> dict[str, str]:
    """The repo-relative files a cohort's rows are computed from, by fingerprint column.

    Each row records the sha256 of these files: the cohort's ``metrics.csv`` in the run
    ``paper_manifest`` names, and the APD study table. A benchmark or APD re-run that keeps
    the pairs (an encoder joins, the median k moves) changes one of them, and
    ``scripts/repro/_finetune_pairs.load`` recomputes the fingerprints to refuse the stale
    study. The per-sample file is not fingerprinted: it is large, and the study already
    refuses per-sample points that do not reproduce ``metrics.csv``.
    """
    from paper_manifest import by_benchmark

    return {"metrics_sha256": by_benchmark(benchmark).metrics_rel, "apd_sha256": APD_CSV_REL}


def sha256(path: Path) -> str:
    """The hex sha256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _apd_dataset(benchmark: str) -> str:
    from loaders import DATASETS, PATHOROB_DOWNSTREAM_DATASETS

    keys = [k for k in PATHOROB_DOWNSTREAM_DATASETS if DATASETS[k]["benchmark"] == benchmark]
    if len(keys) != 1:
        raise ValueError(f"expected one APD dataset for {benchmark}, got {keys}")
    return keys[0]


def _apd_inputs(
    benchmark: str, models: list[str], table_path: Path
) -> tuple[dict, pd.DataFrame, dict]:
    """The APD cells, study rows (from ``table_path``) and nIPD settings for one cohort."""
    from loaders import DATASETS, STUDY_DIR, training_correlations

    dataset = _apd_dataset(benchmark)
    cells = {}
    for model in models:
        path = STUDY_DIR / dataset / f"{model}.json"
        if path.exists():
            cells[model] = json.loads(path.read_text())
    table = pd.read_csv(table_path, keep_default_na=False, na_values=[])
    settings = dict(
        cramers_v=training_correlations(dataset),
        chance=1.0 / len(DATASETS[dataset]["biological_classes"]),
    )
    return cells, table[table["dataset"] == dataset], settings


def run(*, n_boot: int = N_BOOT, seed: int = SEED, level: float = LEVEL) -> pd.DataFrame:
    """One row per fine-tune x cohort: representation scores, nIPD and per-sample shifts.

    Each cohort's run is the one ``paper_manifest`` names, so the study inherits the
    protocol the paper reports and never names one. Each row ends with the sha256 of the
    files it was computed from (:func:`input_rels`).
    """
    from paper_manifest import by_benchmark

    sys.path.insert(0, str(REPO / "scripts" / "studies" / "apd"))
    pairs = finetune_pairs(read_metadata())
    if not pairs:
        raise ValueError(f"model metadata declares no {FINETUNE_ROLE!r} rows")
    models = list(dict.fromkeys(m for p in pairs for m in (p.parent, p.finetune)))
    frames = []
    for benchmark in BENCHMARKS:
        entry = by_benchmark(benchmark)
        inputs = {column: REPO / rel for column, rel in input_rels(benchmark).items()}
        fingerprints = {column: sha256(path) for column, path in inputs.items()}
        metrics = pd.read_csv(inputs["metrics_sha256"])
        _require_roster(pairs, set(metrics["model"].astype(str)), benchmark)
        per_sample = pd.read_csv(REPO / entry.per_sample_rel, usecols=_PER_SAMPLE_COLUMNS)
        per_sample = per_sample[per_sample["model"].isin(models)]
        represent = paired_representation(
            pairs,
            metrics,
            per_sample,
            benchmark=benchmark,
            strata=STRATA[benchmark],
            n_boot=n_boot,
            seed=seed,
            level=level,
        )
        cells, apd_rows, settings = _apd_inputs(benchmark, models, inputs["apd_sha256"])
        downstream = paired_downstream(
            pairs,
            cells,
            apd_rows,
            benchmark=benchmark,
            n_boot=n_boot,
            seed=seed,
            level=level,
            **settings,
        )
        merged = represent.merge(downstream, on=["benchmark", "finetune"], validate="1:1")
        frames.append(merged.assign(**fingerprints))
    order = {p.finetune: i for i, p in enumerate(pairs)}
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["finetune", "benchmark"],
        key=lambda s: s.map(order) if s.name == "finetune" else s.map(BENCHMARKS.index),
    )
    return out.reset_index(drop=True)


def main() -> int:
    frame = run()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV.relative_to(REPO)} ({len(frame)} rows)")
    for _, r in frame.iterrows():
        print(
            f"  {r['parent']:>13s} -> {r['finetune']:<9s} {r['benchmark']:<22s} "
            f"CRoMa {r['delta_croma']:+.3f} [{r['delta_croma_lo']:+.3f}, "
            f"{r['delta_croma_hi']:+.3f}]  F0 {r['delta_f0']:+.3f}  "
            f"LTM {r['delta_ltm']:+.3f}  nIPD_ID {r['delta_nipd_id']:+.3f} "
            f"[{r['delta_nipd_id_lo']:+.3f}, {r['delta_nipd_id_hi']:+.3f}]  "
            f"rank {int(r['parent_rank'])}->{int(r['finetune_rank'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
