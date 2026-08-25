"""Evaluate the sealed pooling study against the frozen public tile panel.

This module performs analysis only.  It reads the tracked public results and the sealed
pooling-sensitivity comparisons, then publishes derived files below the study's ``analysis/``
directory.  Embeddings, per-occurrence arrays, and the #152 result files are never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

EXPECTED_ROSTER = (
    "Virchow2",
    "Virchow",
    "UNI2-h",
    "UNI",
    "CONCHv1.5",
    "CONCH",
    "H-optimus-1",
    "H-optimus-0",
    "H0-mini",
    "Prov-GigaPath",
    "Midnight-12k",
    "Prost40M",
    "Phikon",
    "Phikon-v2",
    "Hibou-L",
    "Hibou-B",
    "mSTAR",
    "GPFM",
    "MUSK",
    "GenBio-PathFM",
    "RudolfV 2",
    "RudolfV 2-B",
    "RudolfV 2-S",
    "Mascaret",
    "Phaet",
    "DINOv2-B",
)

STUDY_MODELS = ("Mascaret", "Phaet", "RudolfV 2", "RudolfV 2-B", "RudolfV 2-S")
STUDY_ALTERNATIVES = {
    "Mascaret": "cls-mean-patch",
    "Phaet": "cls-mean-patch",
    "RudolfV 2": "cls-only",
    "RudolfV 2-B": "cls-only",
    "RudolfV 2-S": "cls-only",
}
PUBLIC_COHORTS = ("camelyon", "tcga-4x4", "tolkach-esca")
COHORT_BENCHMARKS = {
    "camelyon": "pathorob-camelyon",
    "tcga-2x2": "pathorob-tcga-2x2",
    "tcga-4x4": "pathorob-tcga-4x4",
    "tolkach-esca": "pathorob-tolkach-esca",
}
FIXED_K = {"camelyon": 11, "tcga-2x2": 61, "tcga-4x4": 71, "tolkach-esca": 61}
FAMILIES = (
    ("mascaret-parent", "Mascaret", "Midnight-12k"),
    ("phaet-parent", "Phaet", "Phikon-v2"),
    ("rudolf-teacher-b", "RudolfV 2", "RudolfV 2-B"),
    ("rudolf-teacher-s", "RudolfV 2", "RudolfV 2-S"),
)
MOVEMENT_METRICS = (
    "ri",
    "mari",
    "support",
    "ss_dominated_undefined_frac",
    "oo_dominated_undefined_frac",
    "mixed_undefined_frac",
    "croma",
    "croma_ltm10",
)


@dataclass(frozen=True)
class RankingResult:
    """Ranks and explicit raw-value ties for one scenario and view."""

    rankings: pd.DataFrame
    ties: pd.DataFrame


@dataclass(frozen=True)
class RecommendationEvidence:
    """Inputs to the agreed conservative pooling decision rule."""

    supported_croma_deltas: tuple[float, ...]
    public_croma_deltas: tuple[float, ...]
    public_ltm10_deltas: tuple[float, ...]
    sign_crossings: int
    rank_worsened: bool
    left_frontier: bool
    family_reversals: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def load_roster(metadata_path: Path) -> pd.DataFrame:
    """Load and validate the exact production-ordered 26-model tile roster."""

    metadata = pd.read_csv(metadata_path, keep_default_na=False, na_values=[])
    required = {"model", "panel", "panel_order", "published_name"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"model metadata is missing columns: {missing}")
    tile = metadata.loc[metadata["panel"] == "tile"].copy()
    tile["panel_order"] = pd.to_numeric(tile["panel_order"], errors="raise").astype(int)
    tile = tile.sort_values("panel_order", kind="stable").reset_index(drop=True)
    valid = (
        tuple(tile["model"].astype(str)) == EXPECTED_ROSTER
        and tile["panel_order"].tolist() == list(range(1, 27))
        and not tile["model"].duplicated().any()
    )
    if not valid:
        raise ValueError("model metadata does not contain the exact 26-row production roster")
    published = tile["published_name"].astype(str).str.strip()
    display = published.where(published != "", tile["model"].astype(str))
    return pd.DataFrame(
        {
            "model_id": tile["model"].astype(str),
            "model": display,
            "panel_order": tile["panel_order"],
            "is_control": tile["model"].eq("DINOv2-B"),
        }
    )


def validate_public_provenance(results_root: Path) -> tuple[dict, str]:
    """Validate the tracked public cohort contract before any rank is computed."""

    path = results_root / "PROVENANCE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = payload.get("roster") == 26
    cohorts = payload.get("cohorts", {})
    for cohort in PUBLIC_COHORTS:
        observed = cohorts.get(cohort, {})
        valid = valid and observed.get("benchmark") == COHORT_BENCHMARKS[cohort]
        valid = valid and observed.get("panel") == "tile"
        valid = valid and observed.get("k") == FIXED_K[cohort]
        valid = valid and observed.get("n_models") == 26
    if not valid:
        raise ValueError("results/PROVENANCE.json does not match the frozen public cohort contract")
    declared = payload.get("files", {})
    for name in (*PUBLIC_COHORTS, "cross_benchmark"):
        relative = f"results/{name}.csv"
        artifact = results_root / f"{name}.csv"
        if declared.get(relative) != _sha256(artifact):
            raise ValueError(f"tracked public result disagrees with provenance: {relative}")
    return payload, _sha256(path)


def _validate_cohorts(cohorts: dict[str, pd.DataFrame], roster: pd.DataFrame) -> None:
    if tuple(cohorts) != PUBLIC_COHORTS:
        raise ValueError(f"public cohort order must be exactly {PUBLIC_COHORTS}")
    expected = roster["model_id"].tolist()
    for cohort, frame in cohorts.items():
        required = {"model_id", "croma", "croma_ltm10"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{cohort} is missing columns: {missing}")
        actual = frame["model_id"].astype(str).tolist()
        if actual != expected or frame["model_id"].duplicated().any():
            raise ValueError(f"{cohort} does not match the validated roster order")


def _relation(left: float, right: float, *, lower_is_better: bool = False) -> str:
    if left == right:
        return "tie"
    left_wins = left < right if lower_is_better else left > right
    return "above" if left_wins else "below"


def _transition(before: str, after: str) -> str:
    if before == after:
        return "unchanged-tie" if before == "tie" else "unchanged"
    if "tie" in (before, after):
        return "tie-transition"
    return "reversal"


def _frontier(rankings: pd.DataFrame) -> set[str]:
    points = rankings.set_index("model_id")[["mean_croma_rank", "mean_ltm10_rank"]]
    frontier: set[str] = set()
    for model, point in points.iterrows():
        dominated = False
        for other, candidate in points.iterrows():
            if other == model:
                continue
            no_worse = (
                candidate["mean_croma_rank"] <= point["mean_croma_rank"]
                and candidate["mean_ltm10_rank"] <= point["mean_ltm10_rank"]
            )
            strictly_better = (
                candidate["mean_croma_rank"] < point["mean_croma_rank"]
                or candidate["mean_ltm10_rank"] < point["mean_ltm10_rank"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.add(str(model))
    return frontier


def rank_scenario(
    cohorts: dict[str, pd.DataFrame],
    roster: pd.DataFrame,
    *,
    scenario: str,
    view: str,
) -> RankingResult:
    """Rank a public or internal scenario with roster-ordered ``method='first'`` ties."""

    _validate_cohorts(cohorts, roster)
    ordered = roster.sort_values("panel_order", kind="stable").reset_index(drop=True)
    ranked_mask = ~ordered["is_control"].astype(bool)
    ranked_ids = ordered.loc[ranked_mask, "model_id"]
    output = ordered[["model_id", "model", "panel_order", "is_control"]].copy()
    tie_rows: list[dict[str, object]] = []

    for cohort in PUBLIC_COHORTS:
        frame = cohorts[cohort].set_index("model_id").loc[ordered["model_id"]]
        for metric, label in (("croma", "croma"), ("croma_ltm10", "ltm10")):
            values = frame[metric].astype(float)
            output[f"{label}_{cohort}"] = values.to_numpy()
            ranks = values.loc[ranked_ids].rank(ascending=False, method="first").astype(int)
            output[f"{label}_rank_{cohort}"] = ranks.reindex(ordered["model_id"]).to_numpy()
            duplicated = values.loc[ranked_ids].duplicated(keep=False)
            for value in values.loc[ranked_ids][duplicated].drop_duplicates().tolist():
                tied = [model for model in ranked_ids if values.loc[model] == value]
                tie_rows.append(
                    {
                        "scenario": scenario,
                        "view": view,
                        "cohort": cohort,
                        "metric": metric,
                        "value": float(value),
                        "models": "|".join(tied),
                        "roster_order": "|".join(tied),
                        "resolved_order": "|".join(tied),
                    }
                )

    croma_rank_columns = [f"croma_rank_{cohort}" for cohort in PUBLIC_COHORTS]
    ltm_rank_columns = [f"ltm10_rank_{cohort}" for cohort in PUBLIC_COHORTS]
    output["ranked"] = ranked_mask
    output["mean_croma_rank"] = output[croma_rank_columns].mean(axis=1).round(3)
    output["mean_ltm10_rank"] = output[ltm_rank_columns].mean(axis=1).round(3)
    output["combined_mean_rank"] = (
        (output["mean_croma_rank"] + output["mean_ltm10_rank"]) / 2
    ).round(4)

    for metric in ("mean_croma_rank", "mean_ltm10_rank"):
        aggregate = output.loc[output["ranked"], ["model_id", metric]]
        duplicated = aggregate[metric].duplicated(keep=False)
        for value in aggregate.loc[duplicated, metric].drop_duplicates().tolist():
            tied = aggregate.loc[aggregate[metric] == value, "model_id"].tolist()
            tie_rows.append(
                {
                    "scenario": scenario,
                    "view": view,
                    "cohort": "cross-cohort",
                    "metric": metric,
                    "value": float(value),
                    "models": "|".join(tied),
                    "roster_order": "|".join(tied),
                    "resolved_order": "|".join(tied),
                }
            )

    ranked = output.loc[output["ranked"]].copy()
    frontier = _frontier(ranked)
    headline = ranked.sort_values(
        ["combined_mean_rank", "mean_croma_rank", "panel_order"], kind="stable"
    )["model_id"].tolist()
    headline_order = {model: index + 1 for index, model in enumerate(headline)}
    output["headline_order"] = output["model_id"].map(headline_order).astype("Int64")
    output["on_frontier"] = output["model_id"].isin(frontier) & output["ranked"]
    combined = output.loc[output["ranked"], ["model_id", "combined_mean_rank"]]
    duplicated = combined["combined_mean_rank"].duplicated(keep=False)
    for value in combined.loc[duplicated, "combined_mean_rank"].drop_duplicates().tolist():
        tied = combined.loc[combined["combined_mean_rank"] == value, "model_id"].tolist()
        resolved = sorted(tied, key=headline_order.__getitem__)
        tie_rows.append(
            {
                "scenario": scenario,
                "view": view,
                "cohort": "cross-cohort",
                "metric": "combined_mean_rank",
                "value": float(value),
                "models": "|".join(tied),
                "roster_order": "|".join(tied),
                "resolved_order": "|".join(resolved),
            }
        )
    output.insert(0, "view", view)
    output.insert(0, "scenario", scenario)
    ties = pd.DataFrame(
        tie_rows,
        columns=[
            "scenario",
            "view",
            "cohort",
            "metric",
            "value",
            "models",
            "roster_order",
            "resolved_order",
        ],
    )
    return RankingResult(rankings=output, ties=ties)


def _movement(before: float, after: float) -> str:
    if after > before:
        return "increase"
    if after < before:
        return "decrease"
    return "unchanged"


def _sign_crossing(before: float, after: float) -> str:
    if before < 0 < after:
        return "negative-to-positive"
    if before > 0 > after:
        return "positive-to-negative"
    return "none"


def classify_measurements(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Record exact movement, strict CI support, and CRoMa/LTM10 zero crossings."""

    required = {"benchmark", "model", "fixed_k", "croma_delta_ci_lo", "croma_delta_ci_hi"}
    required.update(
        f"{side}_{metric}" for side in ("canonical", "alternative") for metric in MOVEMENT_METRICS
    )
    missing = sorted(required - set(comparisons.columns))
    if missing:
        raise ValueError(f"pooling comparisons are missing columns: {missing}")
    benchmark_to_cohort = {benchmark: cohort for cohort, benchmark in COHORT_BENCHMARKS.items()}
    rows: list[dict[str, object]] = []
    for source in comparisons.to_dict("records"):
        benchmark = str(source["benchmark"])
        if benchmark not in benchmark_to_cohort:
            raise ValueError(f"unknown pooling benchmark: {benchmark}")
        cohort = benchmark_to_cohort[benchmark]
        if int(source["fixed_k"]) != FIXED_K[cohort]:
            raise ValueError(f"{benchmark} must use fixed k={FIXED_K[cohort]}")
        row: dict[str, object] = {
            "benchmark": benchmark,
            "cohort": cohort,
            "supplementary": cohort == "tcga-2x2",
            "model_id": str(source["model"]),
            "fixed_k": int(source["fixed_k"]),
        }
        for metric in MOVEMENT_METRICS:
            before = float(source[f"canonical_{metric}"])
            after = float(source[f"alternative_{metric}"])
            row[f"canonical_{metric}"] = before
            row[f"alternative_{metric}"] = after
            row[f"delta_{metric}"] = after - before
            row[f"abs_delta_{metric}"] = abs(after - before)
        croma_before = float(source["canonical_croma"])
        croma_after = float(source["alternative_croma"])
        ltm_before = float(source["canonical_croma_ltm10"])
        ltm_after = float(source["alternative_croma_ltm10"])
        lo = float(source["croma_delta_ci_lo"])
        hi = float(source["croma_delta_ci_hi"])
        supported = lo > 0 or hi < 0
        row.update(
            {
                "croma_delta_ci_lo": lo,
                "croma_delta_ci_hi": hi,
                "croma_movement": _movement(croma_before, croma_after),
                "croma_supported_shift": (
                    _movement(croma_before, croma_after) if supported else "unsupported"
                ),
                "croma_sign_crossing": _sign_crossing(croma_before, croma_after),
                "ltm10_movement": _movement(ltm_before, ltm_after),
                "ltm10_sign_crossing": _sign_crossing(ltm_before, ltm_after),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_metric_movements(classifications: pd.DataFrame) -> pd.DataFrame:
    """Normalize metric movement while keeping RI/MaRI support separate from CRoMa."""

    rows: list[dict[str, object]] = []
    cause_fields = (
        "ss_dominated_undefined_frac",
        "oo_dominated_undefined_frac",
        "mixed_undefined_frac",
    )
    for source in classifications.to_dict("records"):
        identity = {
            field: source[field]
            for field in ("benchmark", "cohort", "supplementary", "model_id", "fixed_k")
        }
        if "model" in source:
            identity["model"] = source["model"]
        for metric in ("ri", "mari", "croma", "croma_ltm10"):
            before = float(source[f"canonical_{metric}"])
            after = float(source[f"alternative_{metric}"])
            has_shared_support = metric in ("ri", "mari")
            row = {
                **identity,
                "metric": metric,
                "canonical_value": before,
                "alternative_value": after,
                "delta": after - before,
                "abs_delta": abs(after - before),
                "movement": _movement(before, after),
                "canonical_positive_support": (
                    float(source["canonical_support"]) if has_shared_support else float("nan")
                ),
                "alternative_positive_support": (
                    float(source["alternative_support"]) if has_shared_support else float("nan")
                ),
                "supported_shift": (
                    source["croma_supported_shift"] if metric == "croma" else "not-applicable"
                ),
                "sign_crossing": (
                    source["croma_sign_crossing"]
                    if metric == "croma"
                    else (
                        source["ltm10_sign_crossing"]
                        if metric == "croma_ltm10"
                        else "not-applicable"
                    )
                ),
            }
            for cause in cause_fields:
                row[f"canonical_{cause}"] = (
                    float(source[f"canonical_{cause}"]) if has_shared_support else float("nan")
                )
                row[f"alternative_{cause}"] = (
                    float(source[f"alternative_{cause}"]) if has_shared_support else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def classify_rank_changes(rankings: pd.DataFrame) -> pd.DataFrame:
    """Compare every public/internal per-cohort, mean, order, and frontier field."""

    rows: list[dict[str, object]] = []
    rank_fields = [
        *(f"croma_rank_{cohort}" for cohort in PUBLIC_COHORTS),
        *(f"ltm10_rank_{cohort}" for cohort in PUBLIC_COHORTS),
        "mean_croma_rank",
        "mean_ltm10_rank",
        "combined_mean_rank",
        "headline_order",
    ]
    for view in ("public", "internal"):
        selected = rankings[(rankings["view"] == view) & rankings["ranked"]].copy()
        canonical = selected[selected["scenario"] == "canonical"].set_index("model_id")
        alternative = selected[selected["scenario"] == "alternative"].set_index("model_id")
        if set(canonical.index) != set(alternative.index):
            raise ValueError(f"canonical and alternative {view} rank rosters differ")
        for model in canonical.sort_values("panel_order").index:
            row: dict[str, object] = {"view": view, "model_id": model}
            for field in rank_fields:
                before = float(canonical.loc[model, field])
                after = float(alternative.loc[model, field])
                row[f"canonical_{field}"] = before
                row[f"alternative_{field}"] = after
                row[f"delta_{field}"] = after - before
            before_frontier = bool(canonical.loc[model, "on_frontier"])
            after_frontier = bool(alternative.loc[model, "on_frontier"])
            row["canonical_on_frontier"] = before_frontier
            row["alternative_on_frontier"] = after_frontier
            row["frontier_change"] = (
                "entered"
                if after_frontier and not before_frontier
                else (
                    "left"
                    if before_frontier and not after_frontier
                    else "unchanged-in" if before_frontier else "unchanged-out"
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def classify_family_guards(
    measurements: pd.DataFrame,
    rankings: pd.DataFrame,
    *,
    families: tuple[tuple[str, str, str], ...] = FAMILIES,
) -> pd.DataFrame:
    """Classify cohort metric and combined-rank family relations under frozen definitions."""

    rows: list[dict[str, object]] = []
    cohort_order = list(dict.fromkeys(measurements["cohort"].astype(str)))
    values = measurements.set_index(["scenario", "cohort", "model_id"])
    rank_values = rankings.set_index(["scenario", "model_id"])
    for family, left, right in families:
        for cohort in cohort_order:
            for metric, source_metric in (("croma", "croma"), ("ltm10", "croma_ltm10")):
                before = _relation(
                    float(values.loc[("canonical", cohort, left), source_metric]),
                    float(values.loc[("canonical", cohort, right), source_metric]),
                )
                after = _relation(
                    float(values.loc[("alternative", cohort, left), source_metric]),
                    float(values.loc[("alternative", cohort, right), source_metric]),
                )
                rows.append(
                    {
                        "family": family,
                        "member_a": left,
                        "member_b": right,
                        "scope": cohort,
                        "metric": metric,
                        "canonical_relation": before,
                        "alternative_relation": after,
                        "classification": _transition(before, after),
                    }
                )
        before = _relation(
            float(rank_values.loc[("canonical", left), "combined_mean_rank"]),
            float(rank_values.loc[("canonical", right), "combined_mean_rank"]),
            lower_is_better=True,
        )
        after = _relation(
            float(rank_values.loc[("alternative", left), "combined_mean_rank"]),
            float(rank_values.loc[("alternative", right), "combined_mean_rank"]),
            lower_is_better=True,
        )
        rows.append(
            {
                "family": family,
                "member_a": left,
                "member_b": right,
                "scope": "cross-cohort",
                "metric": "combined-rank",
                "canonical_relation": before,
                "alternative_relation": after,
                "classification": _transition(before, after),
            }
        )
    return pd.DataFrame(rows)


def recommendation_from_evidence(evidence: RecommendationEvidence) -> dict[str, str]:
    """Apply the agreed conservative rule without an arbitrary effect-size threshold."""

    reasons: list[str] = []
    if not any(delta > 0 for delta in evidence.supported_croma_deltas):
        reasons.append("no-supported-croma-benefit")
    if any(delta < 0 for delta in evidence.supported_croma_deltas):
        reasons.append("supported-croma-decrease")
    if any(delta < 0 for delta in evidence.public_croma_deltas):
        reasons.append("public-croma-decrease")
    if any(delta < 0 for delta in evidence.public_ltm10_deltas):
        reasons.append("public-ltm10-decrease")
    if evidence.sign_crossings:
        reasons.append("sign-crossing")
    if evidence.rank_worsened:
        reasons.append("rank-regression")
    if evidence.left_frontier:
        reasons.append("left-frontier")
    if evidence.family_reversals:
        reasons.append("family-reversal")
    if reasons:
        return {"recommendation": "retain", "reason_codes": ";".join(reasons)}
    return {
        "recommendation": "change",
        "reason_codes": "supported-croma-benefit;no-guard-regressions",
    }


def build_impact_summary(
    *,
    classifications: pd.DataFrame,
    rank_changes: pd.DataFrame,
    family_guards: pd.DataFrame,
    alternatives: dict[str, str],
    display_names: dict[str, str],
) -> pd.DataFrame:
    """Build the requested one-row-per-encoder pooling impact summary."""

    changes = rank_changes[rank_changes["view"] == "public"].set_index("model_id")
    rows: list[dict[str, object]] = []
    for model in STUDY_MODELS:
        measured = classifications[classifications["model_id"] == model]
        public = measured[~measured["supplementary"]]
        supported = measured[measured["croma_supported_shift"] != "unsupported"]
        guards = family_guards[
            (family_guards["member_a"] == model) | (family_guards["member_b"] == model)
        ]
        reversals = int((guards["classification"] == "reversal").sum())
        sign_crossings = int(
            (measured["croma_sign_crossing"] != "none").sum()
            + (measured["ltm10_sign_crossing"] != "none").sum()
        )
        public_sign_crossings = int(
            (public["croma_sign_crossing"] != "none").sum()
            + (public["ltm10_sign_crossing"] != "none").sum()
        )
        model_changes = rank_changes[rank_changes["model_id"] == model]
        change = model_changes[model_changes["view"] == "public"].iloc[0]
        supported_public = public[public["croma_supported_shift"] != "unsupported"]
        evidence = RecommendationEvidence(
            supported_croma_deltas=tuple(supported_public["delta_croma"].astype(float)),
            public_croma_deltas=tuple(public["delta_croma"].astype(float)),
            public_ltm10_deltas=tuple(public["delta_croma_ltm10"].astype(float)),
            sign_crossings=public_sign_crossings,
            rank_worsened=bool(
                (model_changes["delta_combined_mean_rank"] > 0).any()
                or (model_changes["delta_headline_order"] > 0).any()
            ),
            left_frontier=bool((model_changes["frontier_change"] == "left").any()),
            family_reversals=reversals,
        )
        recommendation = recommendation_from_evidence(evidence)
        before_order = int(change["canonical_headline_order"])
        after_order = int(change["alternative_headline_order"])
        cohort_impact: dict[str, object] = {}
        for cohort in COHORT_BENCHMARKS:
            cohort_row = measured[measured["cohort"] == cohort].iloc[0]
            field = cohort.replace("-", "_")
            cohort_impact[f"{field}_croma_delta"] = float(cohort_row["delta_croma"])
            cohort_impact[f"{field}_croma_supported"] = (
                cohort_row["croma_supported_shift"] != "unsupported"
            )
        rows.append(
            {
                "model": display_names[model],
                "model_id": model,
                "alternative": alternatives[model],
                "supported_croma_increases": int((supported["delta_croma"] > 0).sum()),
                "supported_croma_decreases": int((supported["delta_croma"] < 0).sum()),
                "public_croma_increases": int((public["delta_croma"] > 0).sum()),
                "public_croma_decreases": int((public["delta_croma"] < 0).sum()),
                "public_ltm10_increases": int((public["delta_croma_ltm10"] > 0).sum()),
                "public_ltm10_decreases": int((public["delta_croma_ltm10"] < 0).sum()),
                **cohort_impact,
                "sign_crossings": sign_crossings,
                "rank_change": f"{before_order}->{after_order}",
                "family_reversals": reversals,
                **recommendation,
            }
        )
    return pd.DataFrame(rows)


def render_report(
    *,
    summary: pd.DataFrame,
    public_orders: dict[str, list[str]],
    public_frontiers: dict[str, list[str]],
    internal_orders: dict[str, list[str]] | None = None,
    internal_frontiers: dict[str, list[str]] | None = None,
) -> str:
    """Render a concise report led by the requested impact summary."""

    lines = [
        "# Pooling counterfactual impact",
        "",
        "## Impact summary",
        "",
        "Supported CRoMa counts include the three public cohorts and supplementary TCGA-2x2. "
        "Numerical CRoMa/LTM10 counts use the three public cohorts only.",
        "",
        "| Encoder | Alternative | CRoMa delta (Cam / 2x2 / 4x4 / Tolkach) | Supported CRoMa (+/-) | Public LTM10 (+/-) | Sign crossings | Rank | Family reversals | Recommendation |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.to_dict("records"):
        deltas = []
        for cohort in ("camelyon", "tcga_2x2", "tcga_4x4", "tolkach_esca"):
            marker = "*" if row[f"{cohort}_croma_supported"] else ""
            deltas.append(f"{row[f'{cohort}_croma_delta']:+.4f}{marker}")
        lines.append(
            f"| {row['model']} | {row['alternative']} | "
            f"{' / '.join(deltas)} | "
            f"+{row['supported_croma_increases']} / -{row['supported_croma_decreases']} | "
            f"+{row['public_ltm10_increases']} / -{row['public_ltm10_decreases']} | "
            f"{row['sign_crossings']} | {row['rank_change']} | {row['family_reversals']} | "
            f"{row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "An asterisk marks a shared-group 95% CRoMa confidence interval that strictly "
            "excludes zero. Cohort order is Camelyon, supplementary TCGA-2x2, TCGA-4x4, and "
            "Tolkach-ESCA.",
            "",
            "The decision rule adopts alternative pooling only with at least one supported public "
            "CRoMa benefit and no CRoMa, LTM10, sign, rank, frontier, or family regression.",
            "",
            "## Panel ordering",
            "",
            "Canonical order: " + ", ".join(public_orders["canonical"]),
            "",
            "Alternative order: " + ", ".join(public_orders["alternative"]),
            "",
            "Canonical frontier: " + ", ".join(public_frontiers["canonical"]),
            "",
            "Alternative frontier: " + ", ".join(public_frontiers["alternative"]),
            "",
        ]
    )
    if internal_orders is not None and internal_frontiers is not None:
        lines.extend(
            [
                "## Internal five-encoder view",
                "",
                "Internal canonical order: " + ", ".join(internal_orders["canonical"]),
                "",
                "Internal alternative order: " + ", ".join(internal_orders["alternative"]),
                "",
                "Internal canonical frontier: " + ", ".join(internal_frontiers["canonical"]),
                "",
                "Internal alternative frontier: " + ", ".join(internal_frontiers["alternative"]),
                "",
            ]
        )
    return "\n".join(lines)


def _load_public_cohorts(results_root: Path, roster: pd.DataFrame) -> dict[str, pd.DataFrame]:
    published_to_internal = dict(zip(roster["model"], roster["model_id"]))
    cohorts: dict[str, pd.DataFrame] = {}
    for cohort in PUBLIC_COHORTS:
        frame = pd.read_csv(results_root / f"{cohort}.csv")
        required = {"model", "croma", "croma_ltm10"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"tracked {cohort} results are missing columns: {missing}")
        frame["model_id"] = frame["model"].map(published_to_internal)
        if frame["model_id"].isna().any() or set(frame["model_id"]) != set(roster["model_id"]):
            raise ValueError(f"tracked {cohort} results do not match the validated roster")
        cohorts[cohort] = (
            frame.set_index("model_id")
            .loc[roster["model_id"], ["croma", "croma_ltm10"]]
            .reset_index()
        )
    return cohorts


def _validate_comparisons(
    comparisons: pd.DataFrame,
    canonical: dict[str, pd.DataFrame],
) -> None:
    expected = {
        (benchmark, model) for benchmark in COHORT_BENCHMARKS.values() for model in STUDY_MODELS
    }
    actual = set(zip(comparisons["benchmark"].astype(str), comparisons["model"].astype(str)))
    if actual != expected or len(comparisons) != len(expected):
        raise ValueError("pooling comparisons must contain the exact five-by-four panel")
    alternatives = comparisons.groupby("model")["alternative_representation"].unique().to_dict()
    canonical_representations = set(comparisons["canonical_representation"].astype(str))
    if canonical_representations != {"canonical"} or any(
        alternatives.get(model, []).tolist() != [expected_alternative]
        for model, expected_alternative in STUDY_ALTERNATIVES.items()
    ):
        raise ValueError("pooling comparisons do not contain the frozen representation contracts")
    for cohort in PUBLIC_COHORTS:
        benchmark = COHORT_BENCHMARKS[cohort]
        source = comparisons[comparisons["benchmark"] == benchmark].set_index("model")
        public = canonical[cohort].set_index("model_id")
        for model in STUDY_MODELS:
            if round(float(source.loc[model, "canonical_croma"]), 6) != round(
                float(public.loc[model, "croma"]), 6
            ) or round(float(source.loc[model, "canonical_croma_ltm10"]), 6) != round(
                float(public.loc[model, "croma_ltm10"]), 6
            ):
                raise ValueError(f"sealed canonical values disagree with tracked {cohort}: {model}")


def _validate_canonical_public_ranking(rankings: pd.DataFrame, expected_path: Path) -> None:
    """Require the canonical derivation to reproduce the tracked production aggregate."""

    expected = pd.read_csv(expected_path).drop(columns="tcga_exposed")
    actual = rankings[(rankings["scenario"] == "canonical") & (rankings["view"] == "public")].copy()
    actual = actual.sort_values(
        ["ranked", "headline_order"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    actual = actual.rename(
        columns={
            "combined_mean_rank": "mean_rank",
            "mean_croma_rank": "croma_rank",
            "mean_ltm10_rank": "ltm_rank",
            **{f"ltm10_{cohort}": f"ltm_{cohort.replace('-', '_')}" for cohort in PUBLIC_COHORTS},
            **{f"croma_{cohort}": f"croma_{cohort.replace('-', '_')}" for cohort in PUBLIC_COHORTS},
        }
    )
    actual = actual[expected.columns]
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("canonical ranks do not reproduce results/cross_benchmark.csv") from exc


def _counterfactual_cohorts(
    canonical: dict[str, pd.DataFrame], comparisons: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    alternative = {cohort: frame.copy() for cohort, frame in canonical.items()}
    for cohort in PUBLIC_COHORTS:
        source = comparisons[comparisons["benchmark"] == COHORT_BENCHMARKS[cohort]].set_index(
            "model"
        )
        target = alternative[cohort].set_index("model_id")
        for model in STUDY_MODELS:
            target.loc[model, "croma"] = round(float(source.loc[model, "alternative_croma"]), 6)
            target.loc[model, "croma_ltm10"] = round(
                float(source.loc[model, "alternative_croma_ltm10"]), 6
            )
        alternative[cohort] = target.reset_index()
    return alternative


def _scenario_measurements(
    canonical: dict[str, pd.DataFrame], alternative: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for scenario, cohorts in (("canonical", canonical), ("alternative", alternative)):
        for cohort in PUBLIC_COHORTS:
            frame = cohorts[cohort].copy()
            frame.insert(0, "cohort", cohort)
            frame.insert(0, "scenario", scenario)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _verify_source_provenance(study_root: Path) -> tuple[dict, str]:
    path = study_root / "run-provenance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("study") != "pooling-sensitivity":
        raise ValueError("source study provenance is not the sealed #152 panel")
    for relative, expected in payload.get("output_artifacts", {}).items():
        artifact = study_root / relative
        if not artifact.is_file():
            raise ValueError(f"sealed #152 artifact is missing: {relative}")
        if (
            artifact.stat().st_size != int(expected["size"])
            or _sha256(artifact) != expected["sha256"]
        ):
            raise ValueError(f"sealed #152 artifact changed: {relative}")
    return payload, _sha256(path)


def build_analysis_bundle(
    *,
    metadata_path: Path,
    results_root: Path,
    study_root: Path,
) -> dict[Path, bytes]:
    """Build every deterministic #153 analytical artifact in memory."""

    roster = load_roster(metadata_path)
    public_provenance, public_provenance_sha = validate_public_provenance(results_root)
    canonical = _load_public_cohorts(results_root, roster)
    comparisons_path = study_root / "results/comparisons.csv"
    comparisons = pd.read_csv(comparisons_path)
    _validate_comparisons(comparisons, canonical)
    classifications = classify_measurements(comparisons)
    source_provenance, source_provenance_sha = _verify_source_provenance(study_root)
    alternative = _counterfactual_cohorts(canonical, comparisons)

    ranking_results: list[RankingResult] = []
    for scenario, cohorts in (("canonical", canonical), ("alternative", alternative)):
        ranking_results.append(rank_scenario(cohorts, roster, scenario=scenario, view="public"))
        internal_roster = roster[roster["model_id"].isin(STUDY_MODELS)].copy()
        internal_cohorts = {
            cohort: frame[frame["model_id"].isin(STUDY_MODELS)].copy()
            for cohort, frame in cohorts.items()
        }
        ranking_results.append(
            rank_scenario(internal_cohorts, internal_roster, scenario=scenario, view="internal")
        )
    rankings = pd.concat([result.rankings for result in ranking_results], ignore_index=True)
    ties = pd.concat([result.ties for result in ranking_results], ignore_index=True)
    cross_benchmark_path = results_root / "cross_benchmark.csv"
    _validate_canonical_public_ranking(rankings, cross_benchmark_path)
    display_names = dict(zip(roster["model_id"], roster["model"]))
    classifications.insert(
        classifications.columns.get_loc("model_id") + 1,
        "model",
        classifications["model_id"].map(display_names),
    )
    metric_movements = build_metric_movements(classifications)
    rank_changes = classify_rank_changes(rankings)
    rank_changes.insert(2, "model", rank_changes["model_id"].map(display_names))
    family_guards = classify_family_guards(
        _scenario_measurements(canonical, alternative),
        rankings[rankings["view"] == "public"],
    )
    for field in ("member_a", "member_b"):
        family_guards[f"{field}_published"] = family_guards[field].map(display_names)
    alternatives = (
        comparisons.groupby("model", sort=False)["alternative_representation"].first().to_dict()
    )
    summary = build_impact_summary(
        classifications=classifications,
        rank_changes=rank_changes,
        family_guards=family_guards,
        alternatives=alternatives,
        display_names=display_names,
    )
    recommendations = summary[["model", "model_id", "recommendation", "reason_codes"]].copy()

    orders: dict[str, dict[str, list[str]]] = {"public": {}, "internal": {}}
    frontiers: dict[str, dict[str, list[str]]] = {"public": {}, "internal": {}}
    for view in orders:
        for scenario in ("canonical", "alternative"):
            selected = rankings[
                (rankings["scenario"] == scenario) & (rankings["view"] == view) & rankings["ranked"]
            ]
            orders[view][scenario] = selected.sort_values("headline_order")["model"].tolist()
            frontiers[view][scenario] = (
                selected.loc[selected["on_frontier"]]
                .sort_values("headline_order")["model"]
                .tolist()
            )

    summary_payload = {
        "public_orders": orders["public"],
        "public_frontiers": frontiers["public"],
        "internal_orders": orders["internal"],
        "internal_frontiers": frontiers["internal"],
        "family_reversals": family_guards.loc[
            family_guards["classification"] == "reversal"
        ].to_dict("records"),
        "tie_transitions": family_guards.loc[
            family_guards["classification"] == "tie-transition"
        ].to_dict("records"),
        "recommendations": recommendations.to_dict("records"),
    }
    files: dict[Path, bytes] = {
        Path("analysis/impact-summary.csv"): _csv_bytes(summary),
        Path("analysis/measurements.csv"): _csv_bytes(classifications),
        Path("analysis/metric-movements.csv"): _csv_bytes(metric_movements),
        Path("analysis/rankings.csv"): _csv_bytes(rankings),
        Path("analysis/ties.csv"): _csv_bytes(ties),
        Path("analysis/rank-changes.csv"): _csv_bytes(rank_changes),
        Path("analysis/family-guards.csv"): _csv_bytes(family_guards),
        Path("analysis/recommendations.csv"): _csv_bytes(recommendations),
        Path("analysis/summary.json"): (
            json.dumps(summary_payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        ),
        Path("analysis/report.md"): render_report(
            summary=summary,
            public_orders=orders["public"],
            public_frontiers=frontiers["public"],
            internal_orders=orders["internal"],
            internal_frontiers=frontiers["internal"],
        ).encode("utf-8"),
    }
    provenance = {
        "schema_version": 1,
        "study": "pooling-counterfactual",
        "decision_policy": {
            "name": "supported-benefit-with-no-guard-regressions",
            "effect_size_threshold": None,
            "public_cohorts": list(PUBLIC_COHORTS),
            "supplementary_cohort": "tcga-2x2",
        },
        "ranking_policy": {
            "input_precision": 6,
            "tie_method": "first-after-production-roster-order",
            "axis_mean_precision": 3,
            "combined_mean_precision": 4,
            "public_cohorts": list(PUBLIC_COHORTS),
            "control": "DINOv2-B-descriptive-unranked",
        },
        "inputs": {
            "model_metadata": {
                "path": "scripts/bench/model_metadata.csv",
                "sha256": _sha256(metadata_path),
                "size": metadata_path.stat().st_size,
            },
            "public_results": {
                cohort: {
                    "path": f"results/{cohort}.csv",
                    "sha256": _sha256(results_root / f"{cohort}.csv"),
                    "size": (results_root / f"{cohort}.csv").stat().st_size,
                }
                for cohort in PUBLIC_COHORTS
            },
            "pooling_comparisons": {
                "path": "results/comparisons.csv",
                "sha256": _sha256(comparisons_path),
                "size": comparisons_path.stat().st_size,
            },
            "canonical_cross_benchmark": {
                "path": "results/cross_benchmark.csv",
                "sha256": _sha256(cross_benchmark_path),
                "size": cross_benchmark_path.stat().st_size,
            },
        },
        "source_study_provenance": {
            "path": "run-provenance.json",
            "sha256": source_provenance_sha,
            "payload": source_provenance,
        },
        "public_results_provenance": {
            "path": "results/PROVENANCE.json",
            "sha256": public_provenance_sha,
            "payload": public_provenance,
        },
        "output_artifacts": {
            path.as_posix(): {"sha256": _sha256_bytes(payload), "size": len(payload)}
            for path, payload in sorted(files.items(), key=lambda item: item[0].as_posix())
        },
    }
    files[Path("analysis/run-provenance.json")] = (
        json.dumps(provenance, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return files


def publish_analysis_bundle(
    study_root: Path,
    files: dict[Path, bytes],
    *,
    check: bool = False,
    force: bool = False,
) -> str:
    """Publish only #153 paths, reusing the study's atomic confined writer."""

    if any(
        path.is_absolute()
        or len(path.parts) < 2
        or path.parts[0] != "analysis"
        or ".." in path.parts
        for path in files
    ):
        raise ValueError("counterfactual outputs must stay below analysis/")
    root = Path(study_root).resolve()
    for relative in files:
        ancestor = root
        for part in relative.parts[:-1]:
            ancestor /= part
            if ancestor.is_symlink():
                raise ValueError(f"refusing a symlinked analysis path: {ancestor}")

    from pooling_sensitivity import publish_study_bundle

    return publish_study_bundle(study_root, files, check=check, force=force)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=REPO / "scripts/bench/model_metadata.csv")
    parser.add_argument("--results-root", type=Path, default=REPO / "results")
    parser.add_argument(
        "--study-root", type=Path, default=REPO / "output/studies/pooling-sensitivity"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    files = build_analysis_bundle(
        metadata_path=args.metadata,
        results_root=args.results_root,
        study_root=args.study_root,
    )
    status = publish_analysis_bundle(
        args.study_root,
        files,
        check=bool(args.check),
        force=bool(args.force),
    )
    print(f"pooling counterfactual analysis: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
