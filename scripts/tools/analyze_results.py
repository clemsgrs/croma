"""Optional exploratory CRoMa subgroup / tier analysis (dev tooling).

This module is NOT a paper-artifact generator: nothing in ``paper/`` is built from
its output, and it must never be wired into the reproduction pipeline. It lives in
``scripts/tools/`` (the dev / exploratory cluster) precisely because it is an
optional analytical console, not part of the paper path.

What it uniquely provides:

- the model-specific CRoMa subgroup breakdown (``_build_croma_subgroup_analysis``),
- the tier1 / tier2 / tier3 subgroup-status logic
  (``_tier1_status`` / ``_tier2_status`` / ``_tier3_status``), and
- per-model action flags (``_aggregate_by_model`` + ``_model_action_flags``).

The subgroup + tier breakdown backs the paper's metric-complementarity and
lower-tail-behavior discussions; it is retained here for ad-hoc inspection only.

Analyses that used to live here but are produced by dedicated generators have been
removed to avoid duplication:

- metric correlations -> ``scripts/studies/bootstrap_uncertainty.py``,
- rank tables / rank-agreement -> ``scripts/repro/generate_results_table.py``,
- k / m-sweep sensitivity -> ``scripts/repro/figures/cross_benchmark_figure.py``.

The two ``scripts/repro/`` paths are tracked paper tooling but excluded from source
distributions; see ADR-0012.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from croma.confounders import infer_confounder_display_name
from croma.metrics.base import VALID_EVALUATION_DESIGNS
from croma.metrics.croma import CROMA_HEADLINE_M

try:
    from croma.metrics.tail import compute_tail_metrics
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from croma.metrics.tail import compute_tail_metrics


_THRESH_UNDEFINED_COVERAGE_RISK = 0.25
_THRESH_OO_DOMINATED_HIGH = 0.10
_THRESH_TAIL_GAP_LTM = 0.20
_THRESH_SUBGROUP_TAIL_PREVALENCE_RATIO = 2.0
_THRESH_TAIL_SEVERITY_MEANINGFUL_GAP = 0.10
# CRoMa is now the signed normalized margin in (-1, 1); the robustness threshold is 0.
# Absolute-level gates below are expressed in margin units (the ratio-era anchors
# 1.15 and 0.90 map through (r-1)/(r+1) to ~0.07 and ~-0.05). Delta/drop heuristics
# are left numerically unchanged and now read in margin units.
_THRESH_TIER1_MEDIAN_DELTA = 0.05
_THRESH_TIER1_NEG_DELTA = 0.05
_THRESH_TIER2_ROBUST_MEDIAN_FLOOR = 0.07
_THRESH_TIER2_LTM_CEILING = -0.05
# Drop is a within-subgroup difference (median - LTM); the margin compresses such
# differences vs the ratio era, so the 0.25 ratio threshold is lowered to 0.10.
_THRESH_TIER2_INTERNAL_DROP = 0.10
_THRESH_TIER2_NEG_FRAC_FLOOR = 0.15
_THRESH_TIER2_MIN_SAMPLES = 10
_THRESH_TIER2_MIN_NEG_COUNT = 3

_SUBGROUP_SCORE_COLUMN = f"croma_m{int(CROMA_HEADLINE_M)}"
_SUBGROUP_MIN_HEADLINE_SAMPLES = 2
_SUBGROUP_SCOPE_ORDER = ("stratum", "label", "confounder")
_SUBGROUP_SCOPE_TO_COLUMNS = {
    "stratum": ("label", "confounder"),
    "label": ("label",),
    "confounder": ("confounder",),
}
_SCOPE_SORT_ORDER = {
    "stratum": 0,
    "label": 1,
    "confounder": 2,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optional exploratory CRoMa subgroup / tier analysis and per-model action "
            "flags (dev tooling; NOT a paper-artifact generator)."
        )
    )
    parser.add_argument(
        "--metrics-csv", required=True, type=Path, help="Path to benchmark metrics CSV."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for analysis artifacts (default: <metrics parent>/analysis).",
    )
    return parser.parse_args()


def _scoped_model_labels(df: pd.DataFrame) -> pd.Series:
    labels = df["model"].astype(str)
    parts: list[pd.Series] = []
    if "evaluation_design" in df.columns:
        parts.append(df["evaluation_design"].astype(str))
    if "evaluation_unit" in df.columns:
        parts.append(df["evaluation_unit"].astype(str))
    if not parts:
        return labels
    scope = parts[0]
    for extra in parts[1:]:
        scope = scope + ";" + extra
    return labels + " [" + scope + "]"


def _aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        raise ValueError("metrics CSV must include a 'model' column")
    working = df.copy()
    working["model"] = _scoped_model_labels(working)
    numeric_cols = [
        c
        for c in working.columns
        if c != "model" and pd.api.types.is_numeric_dtype(working[c])
    ]
    if not numeric_cols:
        raise ValueError("metrics CSV has no numeric columns to analyze")
    grouped = working.groupby("model", as_index=False)[numeric_cols].mean(
        numeric_only=True
    )
    return grouped


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df


def _empty_subgroup_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset",
            "model",
            "evaluation_design",
            "evaluation_unit",
            "context_id",
            "context_label",
            "scope",
            "subgroup_name",
            "confounder_column",
            "confounder_display_name",
            "label",
            "confounder",
            "n_samples",
            "group_frac",
            "mean_croma",
            "rest_mean_croma",
            "mean_croma_delta_vs_rest",
            "median_croma",
            "rest_median_croma",
            "median_croma_delta_vs_rest",
            "croma_neg_frac",
            "croma_neg_count",
            "rest_croma_neg_frac",
            "croma_neg_frac_delta_vs_rest",
            "subgroup_q_alpha",
            "subgroup_ltm_alpha",
            "internal_tail_drop",
            "tier1_status",
            "tier2_status",
            "tail_count",
            "tail_prevalence",
            "context_tail_prevalence",
            "tail_prevalence_delta",
            "tail_prevalence_ratio",
            "tail_share",
            "tail_mean_croma",
            "rest_tail_mean_croma",
            "tail_mean_croma_delta_vs_rest",
            "tail_severity_label",
            "tier3_status",
            "n_defined_samples",
            "tail_size",
            "headline_eligible",
            "report_rank",
        ]
    )


def _empty_context_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset",
            "model",
            "evaluation_design",
            "evaluation_unit",
            "context_id",
            "context_label",
            "n_defined_samples",
            "n_labels",
            "tail_size",
            "croma_alpha",
            "pooled_mean_croma",
            "pooled_median_croma",
            "pooled_tail_mean_croma",
            "skipped",
            "skip_reason",
        ]
    )


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return parsed


def _resolve_confounder_display_name(df: pd.DataFrame) -> str:
    if "confounder_display_name" in df.columns:
        values = df["confounder_display_name"].dropna().astype(str).str.strip()
        values = values[values != ""]
        if len(values) > 0:
            return str(values.iloc[0])
    if "confounder_column" in df.columns:
        values = df["confounder_column"].dropna().astype(str).str.strip()
        values = values[values != ""]
        if len(values) > 0:
            return infer_confounder_display_name(str(values.iloc[0]))
    return "Confounder"


def _sort_tail_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        [_SUBGROUP_SCORE_COLUMN, "sample_id"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)


def _render_subgroup_name(row: pd.Series, scope: str) -> str:
    if scope == "stratum":
        return f"{row['label']} / {row['confounder']}"
    if scope == "label":
        return str(row["label"])
    return str(row["confounder"])


def _subgroup_report_sort(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.reset_index(drop=True)
    working = df.copy()
    if "tail_prevalence_ratio" in working.columns:
        working["_tail_signal"] = working["tail_prevalence_ratio"]
    else:
        working["_tail_signal"] = float("nan")
    sort_cols = [
        "_tail_signal",
        "tail_prevalence",
        "mean_croma",
        "croma_neg_frac",
        "tail_mean_croma",
        "label",
        "confounder",
    ]
    ascending = [False, False, True, False, True, True, True]
    available = [c for c in sort_cols if c in working.columns]
    asc = [ascending[sort_cols.index(c)] for c in available]
    return (
        working.sort_values(available, ascending=asc, kind="mergesort")
        .drop(columns="_tail_signal")
        .reset_index(drop=True)
    )


def _fmt_float(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(parsed):
        return "NA"
    return f"{parsed:.3f}"


def _fmt_ratio(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(parsed):
        return "NA"
    return f"{parsed:.1f}x"


def _tier1_status(
    *,
    n_samples: int,
    median_croma: float,
    rest_median_croma: float,
    median_delta: float,
    croma_neg_delta: float,
) -> str:
    if n_samples < _SUBGROUP_MIN_HEADLINE_SAMPLES:
        return "insufficient_support"
    if (
        median_delta <= -_THRESH_TIER1_MEDIAN_DELTA
        and croma_neg_delta >= _THRESH_TIER1_NEG_DELTA
    ):
        if median_croma < 0.0 and rest_median_croma >= 0.0:
            return "broad_weakness"
        if median_croma >= 0.0 and rest_median_croma >= 0.0:
            return "relative_weakness"
        if median_croma < 0.0 and rest_median_croma < 0.0:
            return "aggravated_weakness"
    return "neutral"


def _tier2_status(
    *,
    n_samples: int,
    median_croma: float,
    subgroup_ltm_alpha: float,
    internal_tail_drop: float,
    croma_neg_frac: float,
    croma_neg_count: int,
) -> str:
    if n_samples < _THRESH_TIER2_MIN_SAMPLES:
        return "insufficient_support"
    pocket_gate = (
        subgroup_ltm_alpha <= _THRESH_TIER2_LTM_CEILING
        and internal_tail_drop >= _THRESH_TIER2_INTERNAL_DROP
        and croma_neg_frac >= _THRESH_TIER2_NEG_FRAC_FLOOR
        and croma_neg_count >= _THRESH_TIER2_MIN_NEG_COUNT
    )
    if pocket_gate and median_croma < 0.0:
        return "aggravated_weakness"
    if pocket_gate and median_croma >= _THRESH_TIER2_ROBUST_MEDIAN_FLOOR:
        return "hidden_pocket"
    if (
        median_croma >= _THRESH_TIER2_ROBUST_MEDIAN_FLOOR
        and subgroup_ltm_alpha >= 0.0
        and internal_tail_drop >= _THRESH_TIER2_INTERNAL_DROP
    ):
        return "internal_spread"
    return "neutral"


def _tail_severity_label(
    *, tail_count: int, rest_tail_count: int, tail_delta: float
) -> str:
    if tail_count == 0:
        return "no_tail_samples"
    if rest_tail_count == 0 or not np.isfinite(tail_delta):
        return "no_rest_tail"
    if tail_delta <= -_THRESH_TAIL_SEVERITY_MEANINGFUL_GAP:
        return "more severe"
    if tail_delta >= _THRESH_TAIL_SEVERITY_MEANINGFUL_GAP:
        return "not more severe"
    return "similar"


def _tier3_status(
    *, n_samples: int, tail_prevalence_ratio: float, tail_severity_label: str
) -> str:
    if n_samples < _SUBGROUP_MIN_HEADLINE_SAMPLES:
        return "insufficient_support"
    enriched = (
        np.isfinite(tail_prevalence_ratio)
        and tail_prevalence_ratio >= _THRESH_SUBGROUP_TAIL_PREVALENCE_RATIO
    )
    severe = tail_severity_label == "more severe"
    if enriched and severe:
        return "tail_enriched_and_severe"
    if enriched:
        return "tail_enriched"
    if severe:
        return "tail_severe"
    return "neutral"


def _sort_tier_rows(df: pd.DataFrame, *, tier: str) -> pd.DataFrame:
    if len(df) == 0:
        return df.reset_index(drop=True)
    working = df.copy()
    working["_scope_order"] = (
        working["scope"].map(_SCOPE_SORT_ORDER).fillna(99).astype(int)
    )
    if tier == "tier1":
        status_order = {
            "broad_weakness": 0,
            "aggravated_weakness": 1,
            "relative_weakness": 2,
            "neutral": 3,
            "insufficient_support": 4,
        }
        working["_status_order"] = (
            working["tier1_status"].map(status_order).fillna(9).astype(int)
        )
        sort_cols = [
            "_status_order",
            "median_croma_delta_vs_rest",
            "croma_neg_frac_delta_vs_rest",
            "_scope_order",
            "subgroup_name",
        ]
        ascending = [True, True, False, True, True]
    elif tier == "tier2":
        status_order = {
            "hidden_pocket": 0,
            "aggravated_weakness": 1,
            "internal_spread": 2,
            "neutral": 3,
            "insufficient_support": 4,
        }
        working["_status_order"] = (
            working["tier2_status"].map(status_order).fillna(9).astype(int)
        )
        sort_cols = [
            "_status_order",
            "internal_tail_drop",
            "median_croma",
            "_scope_order",
            "subgroup_name",
        ]
        ascending = [True, False, False, True, True]
    else:
        status_order = {
            "tail_enriched_and_severe": 0,
            "tail_severe": 1,
            "tail_enriched": 2,
            "neutral": 3,
            "insufficient_support": 4,
        }
        working["_status_order"] = (
            working["tier3_status"].map(status_order).fillna(9).astype(int)
        )
        sort_cols = [
            "_status_order",
            "tail_prevalence_ratio",
            "tail_mean_croma_delta_vs_rest",
            "_scope_order",
            "subgroup_name",
        ]
        ascending = [True, False, True, True, True]
    return (
        working.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .drop(columns=["_scope_order", "_status_order"])
        .reset_index(drop=True)
    )


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _render_tier_table(scoped_rows: pd.DataFrame, *, tier: str) -> list[str]:
    sorted_rows = _sort_tier_rows(scoped_rows, tier=tier)
    if tier == "tier1":
        headers = [
            "Scope",
            "Subgroup",
            "Status",
            "Median CRoMa",
            "Rest Median",
            "Median Delta",
            "CRoMa<0 Frac",
            "Rest CRoMa<0 Frac",
            "CRoMa<0 Delta",
        ]
        rows = [
            [
                str(row["scope"]),
                str(row["subgroup_name"]),
                str(row["tier1_status"]),
                _fmt_float(row["median_croma"]),
                _fmt_float(row["rest_median_croma"]),
                _fmt_float(row["median_croma_delta_vs_rest"]),
                _fmt_float(row["croma_neg_frac"]),
                _fmt_float(row["rest_croma_neg_frac"]),
                _fmt_float(row["croma_neg_frac_delta_vs_rest"]),
            ]
            for _, row in sorted_rows.iterrows()
        ]
        return _render_markdown_table(headers, rows)
    if tier == "tier2":
        headers = [
            "Scope",
            "Subgroup",
            "Status",
            "N",
            "CRoMa<0 Frac",
            "CRoMa<0 Count",
            "Median CRoMa",
            "Subgroup LTM@alpha",
            "Drop",
        ]
        rows = [
            [
                str(row["scope"]),
                str(row["subgroup_name"]),
                str(row["tier2_status"]),
                str(int(row["n_samples"])),
                _fmt_float(row["croma_neg_frac"]),
                str(int(row["croma_neg_count"])),
                _fmt_float(row["median_croma"]),
                _fmt_float(row["subgroup_ltm_alpha"]),
                _fmt_float(row["internal_tail_drop"]),
            ]
            for _, row in sorted_rows.iterrows()
        ]
        return _render_markdown_table(headers, rows)
    headers = [
        "Scope",
        "Subgroup",
        "Status",
        "Tail Prevalence",
        "Overall Tail Prev",
        "Ratio",
        "Tail Mean CRoMa",
        "Rest Tail Mean",
        "Severity",
    ]
    rows = [
        [
            str(row["scope"]),
            str(row["subgroup_name"]),
            str(row["tier3_status"]),
            _fmt_float(row["tail_prevalence"]),
            _fmt_float(row["context_tail_prevalence"]),
            _fmt_ratio(row["tail_prevalence_ratio"]),
            _fmt_float(row["tail_mean_croma"]),
            _fmt_float(row["rest_tail_mean_croma"]),
            str(row["tail_severity_label"]),
        ]
        for _, row in sorted_rows.iterrows()
    ]
    return _render_markdown_table(headers, rows)


def _build_context_row(
    *,
    context_df: pd.DataFrame,
    dataset: str,
    model: str,
    evaluation_design: str,
    evaluation_unit: str,
    context_id: str,
    n_labels: int,
    alpha: float,
) -> dict[str, object]:
    return {
        "dataset": str(dataset),
        "model": str(model),
        "evaluation_design": str(evaluation_design),
        "evaluation_unit": str(evaluation_unit),
        "context_id": str(context_id),
        "context_label": str(context_id),
        "n_defined_samples": int(len(context_df)),
        "n_labels": int(n_labels),
        "tail_size": 0,
        "croma_alpha": float(alpha),
        "pooled_mean_croma": float("nan"),
        "pooled_median_croma": float("nan"),
        "pooled_tail_mean_croma": float("nan"),
        "skipped": False,
        "skip_reason": "",
    }


def _build_croma_subgroup_analysis(
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subgroup_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    if df_per_sample is None or len(df_per_sample) == 0:
        return _empty_subgroup_df(), _empty_context_df()

    required = {
        "dataset",
        "model",
        "evaluation_design",
        "evaluation_unit",
        "subset",
        "sample_id",
        "label",
        "confounder",
        "croma_alpha",
        _SUBGROUP_SCORE_COLUMN,
    }
    if not required.issubset(df_per_sample.columns):
        return _empty_subgroup_df(), _empty_context_df()

    working = df_per_sample.copy()
    confounder_display_name = _resolve_confounder_display_name(working)
    confounder_column = (
        str(working.get("confounder_column", pd.Series(dtype=str)).iloc[0])
        if "confounder_column" in working.columns and len(working) > 0
        else "confounder"
    )
    for col in (
        "dataset",
        "model",
        "evaluation_design",
        "evaluation_unit",
        "subset",
        "sample_id",
        "label",
        "confounder",
    ):
        working[col] = working[col].astype(str)
    working[_SUBGROUP_SCORE_COLUMN] = pd.to_numeric(
        working[_SUBGROUP_SCORE_COLUMN], errors="coerce"
    )
    working["croma_alpha"] = pd.to_numeric(working["croma_alpha"], errors="coerce")
    working["context_id"] = np.where(
        working["evaluation_design"] == "paired_2x2",
        working["subset"],
        "dataset",
    )

    group_cols = [
        "dataset",
        "model",
        "evaluation_design",
        "evaluation_unit",
        "context_id",
    ]
    for keys, group in working.groupby(group_cols, sort=True, dropna=False):
        dataset, model, evaluation_design, evaluation_unit, context_id = [
            str(v) for v in keys
        ]
        # Which design produced a row decides whether its subgroup reading is reported at
        # all, so an unrecognized value must stop the analysis rather than quietly fall
        # through to the reporting branch. Per-sample rows written before the
        # dataset_wide -> all rename have to be recomputed, not reinterpreted.
        if evaluation_design not in VALID_EVALUATION_DESIGNS:
            raise ValueError(
                f"per-sample rows for dataset {dataset!r}, model {model!r} carry "
                f"evaluation_design={evaluation_design!r}; expected one of "
                f"{list(VALID_EVALUATION_DESIGNS)}. Re-run the benchmark to recompute them."
            )
        defined = group[np.isfinite(group[_SUBGROUP_SCORE_COLUMN])].copy()
        alpha = _safe_float(
            (
                group["croma_alpha"].dropna().iloc[0]
                if group["croma_alpha"].notna().any()
                else np.nan
            ),
            0.10,
        )
        n_labels = int(defined["label"].nunique())
        context_row = _build_context_row(
            context_df=defined,
            dataset=dataset,
            model=model,
            evaluation_design=evaluation_design,
            evaluation_unit=evaluation_unit,
            context_id=context_id,
            n_labels=n_labels,
            alpha=alpha,
        )

        if len(defined) == 0:
            context_row["skipped"] = True
            context_row["skip_reason"] = (
                f"Skipped: no defined CRoMa(m={int(CROMA_HEADLINE_M)}) samples are available for this context."
            )
            context_rows.append(context_row)
            continue

        if evaluation_design == "all" and n_labels != 2:
            context_row["skipped"] = True
            context_row["skip_reason"] = (
                "Skipped: pooled all-rows subgroup interpretation is not reported because "
                "heterogeneous biological boundaries make the result clinically weak; use paired runs instead."
            )
            context_rows.append(context_row)
            continue

        defined = _sort_tail_rows(defined)
        tail_size = int(max(1, int(np.ceil(alpha * len(defined)))))
        tail_index = defined.index[:tail_size]
        defined["is_tail"] = defined.index.isin(tail_index)

        context_row["tail_size"] = int(tail_size)
        context_row["pooled_mean_croma"] = float(defined[_SUBGROUP_SCORE_COLUMN].mean())
        context_row["pooled_median_croma"] = float(
            defined[_SUBGROUP_SCORE_COLUMN].median()
        )
        context_row["pooled_tail_mean_croma"] = float(
            defined.loc[defined["is_tail"], _SUBGROUP_SCORE_COLUMN].mean()
        )
        context_rows.append(context_row)

        n_defined = int(len(defined))
        context_mean_croma = float(context_row["pooled_mean_croma"])
        context_tail_prevalence = (
            float(tail_size / n_defined) if n_defined > 0 else float("nan")
        )
        context_tail_mean_croma = float(context_row["pooled_tail_mean_croma"])
        for scope in _SUBGROUP_SCOPE_ORDER:
            cols = list(_SUBGROUP_SCOPE_TO_COLUMNS[scope])
            grouped = defined.groupby(cols, sort=True, dropna=False)
            for subgroup_key, subgroup in grouped:
                if not isinstance(subgroup_key, tuple):
                    subgroup_key = (subgroup_key,)
                label = ""
                confounder = ""
                if "label" in cols:
                    label = str(subgroup_key[cols.index("label")])
                if "confounder" in cols:
                    confounder = str(subgroup_key[cols.index("confounder")])

                n_samples = int(len(subgroup))
                tail_count = int(subgroup["is_tail"].sum())
                group_frac = float(n_samples / n_defined)
                mean_croma = float(subgroup[_SUBGROUP_SCORE_COLUMN].mean())
                median_croma = float(subgroup[_SUBGROUP_SCORE_COLUMN].median())
                croma_neg_frac = float((subgroup[_SUBGROUP_SCORE_COLUMN] < 0.0).mean())
                croma_neg_count = int((subgroup[_SUBGROUP_SCORE_COLUMN] < 0.0).sum())
                tail_prevalence = float(tail_count / n_samples)
                tail_prevalence_ratio = (
                    float(tail_prevalence / context_tail_prevalence)
                    if np.isfinite(context_tail_prevalence)
                    and context_tail_prevalence > 0
                    else float("nan")
                )
                tail_share = (
                    float(tail_count / tail_size) if tail_size > 0 else float("nan")
                )
                tail_mean_croma = (
                    float(
                        subgroup.loc[subgroup["is_tail"], _SUBGROUP_SCORE_COLUMN].mean()
                    )
                    if tail_count > 0
                    else float("nan")
                )
                rest = defined.loc[~defined.index.isin(subgroup.index)]
                rest_mean_croma = (
                    float(rest[_SUBGROUP_SCORE_COLUMN].mean())
                    if len(rest) > 0
                    else float("nan")
                )
                rest_median_croma = (
                    float(rest[_SUBGROUP_SCORE_COLUMN].median())
                    if len(rest) > 0
                    else float("nan")
                )
                rest_croma_neg_frac = (
                    float((rest[_SUBGROUP_SCORE_COLUMN] < 0.0).mean())
                    if len(rest) > 0
                    else float("nan")
                )
                rest_tail_count = int(rest["is_tail"].sum()) if len(rest) > 0 else 0
                rest_tail_mean_croma = (
                    float(rest.loc[rest["is_tail"], _SUBGROUP_SCORE_COLUMN].mean())
                    if rest_tail_count > 0
                    else float("nan")
                )
                subgroup_tail_metrics = compute_tail_metrics(
                    subgroup[_SUBGROUP_SCORE_COLUMN].to_numpy(dtype=float),
                    alpha=alpha,
                )
                median_croma_delta_vs_rest = (
                    float(median_croma - rest_median_croma)
                    if np.isfinite(rest_median_croma)
                    else float("nan")
                )
                croma_neg_frac_delta_vs_rest = (
                    float(croma_neg_frac - rest_croma_neg_frac)
                    if np.isfinite(rest_croma_neg_frac)
                    else float("nan")
                )
                tail_mean_croma_delta_vs_rest = (
                    float(tail_mean_croma - rest_tail_mean_croma)
                    if np.isfinite(tail_mean_croma) and np.isfinite(rest_tail_mean_croma)
                    else float("nan")
                )
                internal_tail_drop = (
                    float(median_croma - subgroup_tail_metrics.ltm_alpha)
                    if np.isfinite(subgroup_tail_metrics.ltm_alpha)
                    else float("nan")
                )
                subgroup_name = _render_subgroup_name(
                    pd.Series({"label": label, "confounder": confounder}),
                    scope=scope,
                )
                tier1_status = _tier1_status(
                    n_samples=n_samples,
                    median_croma=median_croma,
                    rest_median_croma=rest_median_croma,
                    median_delta=median_croma_delta_vs_rest,
                    croma_neg_delta=croma_neg_frac_delta_vs_rest,
                )
                tail_severity_label = _tail_severity_label(
                    tail_count=tail_count,
                    rest_tail_count=rest_tail_count,
                    tail_delta=tail_mean_croma_delta_vs_rest,
                )
                tier2_status = _tier2_status(
                    n_samples=n_samples,
                    median_croma=median_croma,
                    subgroup_ltm_alpha=float(subgroup_tail_metrics.ltm_alpha),
                    internal_tail_drop=internal_tail_drop,
                    croma_neg_frac=croma_neg_frac,
                    croma_neg_count=croma_neg_count,
                )
                tier3_status = _tier3_status(
                    n_samples=n_samples,
                    tail_prevalence_ratio=tail_prevalence_ratio,
                    tail_severity_label=tail_severity_label,
                )
                subgroup_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "evaluation_design": evaluation_design,
                        "evaluation_unit": evaluation_unit,
                        "context_id": context_id,
                        "context_label": context_id,
                        "scope": scope,
                        "subgroup_name": subgroup_name,
                        "confounder_column": confounder_column,
                        "confounder_display_name": confounder_display_name,
                        "label": label,
                        "confounder": confounder,
                        "n_samples": n_samples,
                        "group_frac": group_frac,
                        "mean_croma": mean_croma,
                        "rest_mean_croma": rest_mean_croma,
                        "mean_croma_delta_vs_rest": (
                            float(mean_croma - rest_mean_croma)
                            if np.isfinite(rest_mean_croma)
                            else float("nan")
                        ),
                        "median_croma": median_croma,
                        "rest_median_croma": rest_median_croma,
                        "median_croma_delta_vs_rest": median_croma_delta_vs_rest,
                        "croma_neg_frac": croma_neg_frac,
                        "croma_neg_count": croma_neg_count,
                        "rest_croma_neg_frac": rest_croma_neg_frac,
                        "croma_neg_frac_delta_vs_rest": croma_neg_frac_delta_vs_rest,
                        "subgroup_q_alpha": float(subgroup_tail_metrics.q_alpha),
                        "subgroup_ltm_alpha": float(subgroup_tail_metrics.ltm_alpha),
                        "internal_tail_drop": internal_tail_drop,
                        "tier1_status": tier1_status,
                        "tier2_status": tier2_status,
                        "tail_count": tail_count,
                        "tail_prevalence": tail_prevalence,
                        "context_tail_prevalence": context_tail_prevalence,
                        "tail_prevalence_delta": float(
                            tail_prevalence - context_tail_prevalence
                        ),
                        "tail_prevalence_ratio": tail_prevalence_ratio,
                        "tail_share": tail_share,
                        "tail_mean_croma": tail_mean_croma,
                        "rest_tail_mean_croma": rest_tail_mean_croma,
                        "tail_mean_croma_delta_vs_rest": tail_mean_croma_delta_vs_rest,
                        "tail_severity_label": tail_severity_label,
                        "tier3_status": tier3_status,
                        "n_defined_samples": n_defined,
                        "tail_size": tail_size,
                        "headline_eligible": bool(
                            n_samples >= _SUBGROUP_MIN_HEADLINE_SAMPLES
                        ),
                    }
                )

    context_out = pd.DataFrame(context_rows) if context_rows else _empty_context_df()
    if not subgroup_rows:
        return _empty_subgroup_df(), context_out

    subgroup_out = pd.DataFrame(subgroup_rows)
    ranked_frames: list[pd.DataFrame] = []
    rank_group_cols = [
        "dataset",
        "model",
        "evaluation_design",
        "evaluation_unit",
        "context_id",
        "scope",
    ]
    for _, frame in subgroup_out.groupby(rank_group_cols, sort=False, dropna=False):
        ranked = _subgroup_report_sort(frame)
        ranked["report_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_frames.append(ranked)
    subgroup_out = (
        pd.concat(ranked_frames, ignore_index=True)
        if ranked_frames
        else _empty_subgroup_df()
    )
    subgroup_out = subgroup_out.loc[:, _empty_subgroup_df().columns]
    context_out = context_out.loc[:, _empty_context_df().columns]
    return subgroup_out, context_out


def _render_croma_subgroup_markdown(
    subgroup_df: pd.DataFrame, context_df: pd.DataFrame
) -> str:
    lines: list[str] = ["# Model-Specific CRoMa Subgroup Analysis", ""]
    if len(context_df) == 0:
        lines.append(f"- No per-sample CRoMa(m={int(CROMA_HEADLINE_M)}) contexts available.")
        return "\n".join(lines) + "\n"

    grouped_models = context_df.groupby("model", sort=True, dropna=False)
    for model, model_contexts in grouped_models:
        lines.append(f"## {model}")
        lines.append("")
        for _, context in model_contexts.sort_values(
            ["dataset", "context_id"], kind="mergesort"
        ).iterrows():
            context_id = str(context["context_id"])
            dataset = str(context["dataset"])
            evaluation_design = str(context["evaluation_design"])
            lines.append(f"### {dataset} / {context_id} ({evaluation_design})")
            if bool(context["skipped"]):
                lines.append(f"- {context['skip_reason']}")
                lines.append("")
                continue

            lines.append(
                "- "
                f"Defined samples={int(context['n_defined_samples'])}, "
                f"tail_size={int(context['tail_size'])}, "
                f"pooled_mean_croma={float(context['pooled_mean_croma']):.3f}, "
                f"pooled_median_croma={float(context['pooled_median_croma']):.3f}, "
                f"tail_mean_croma={float(context['pooled_tail_mean_croma']):.3f}."
            )

            context_rows = subgroup_df[
                (subgroup_df["model"] == model)
                & (subgroup_df["dataset"] == dataset)
                & (subgroup_df["evaluation_design"] == evaluation_design)
                & (subgroup_df["context_id"] == context_id)
            ]
            lines.append("#### Broad Subgroup Weakness")
            lines.extend(_render_tier_table(context_rows, tier="tier1"))
            lines.append("")
            lines.append("#### Hidden Subgroup Pockets")
            lines.extend(_render_tier_table(context_rows, tier="tier2"))
            lines.append("")
            lines.append("#### Tail-Specific Fragility")
            lines.extend(_render_tier_table(context_rows, tier="tier3"))
            lines.append("")
        lines.append("")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Appendix: Status Definitions and Thresholds")
    lines.append("")
    lines.append(
        f"All per-sample CRoMa values used in this report are computed at **m={int(CROMA_HEADLINE_M)}** "
        f"(column `{_SUBGROUP_SCORE_COLUMN}`). "
        "\"Rest\" refers to all other defined samples outside the subgroup."
    )
    lines.append("")
    lines.append("### Broad Subgroup Weakness (Tier 1)")
    lines.append("")
    lines.append(
        "Compares a subgroup's median CRoMa to the rest. "
        "A subgroup is flagged when its median delta vs rest "
        f"<= **-{_THRESH_TIER1_MEDIAN_DELTA:.2f}** "
        f"and its CRoMa<0 fraction delta vs rest >= **{_THRESH_TIER1_NEG_DELTA:.2f}**."
    )
    lines.append("")
    lines.append("| Status | Condition |")
    lines.append("| --- | --- |")
    lines.append(
        "| `broad_weakness` | Flagged, and subgroup median < 0.0 while rest median >= 0.0 |"
    )
    lines.append(
        "| `relative_weakness` | Flagged, and both subgroup and rest median >= 0.0 |"
    )
    lines.append(
        "| `aggravated_weakness` | Flagged, and both subgroup and rest median < 0.0 |"
    )
    lines.append("| `neutral` | Not flagged |")
    lines.append("")
    lines.append("### Hidden Subgroup Pockets (Tier 2)")
    lines.append("")
    lines.append(
        "Detects subgroups where the median looks healthy but a significant "
        "fragile tail is hidden underneath. "
        f"Requires N >= **{_THRESH_TIER2_MIN_SAMPLES}**. "
        "A subgroup passes the \"pocket gate\" when all of:"
    )
    lines.append("")
    lines.append(f"- Subgroup LTM@alpha <= **{_THRESH_TIER2_LTM_CEILING:.2f}**")
    lines.append(
        f"- Internal drop (median - LTM@alpha) >= **{_THRESH_TIER2_INTERNAL_DROP:.2f}**"
    )
    lines.append(f"- CRoMa<0 fraction >= **{_THRESH_TIER2_NEG_FRAC_FLOOR:.2f}**")
    lines.append(f"- CRoMa<0 count >= **{_THRESH_TIER2_MIN_NEG_COUNT}**")
    lines.append("")
    lines.append("| Status | Condition |")
    lines.append("| --- | --- |")
    lines.append(
        f"| `hidden_pocket` | Pocket gate passed and median >= "
        f"**{_THRESH_TIER2_ROBUST_MEDIAN_FLOOR:.2f}** |"
    )
    lines.append("| `aggravated_weakness` | Pocket gate passed and median < 0.0 |")
    lines.append(
        f"| `internal_spread` | Median >= **{_THRESH_TIER2_ROBUST_MEDIAN_FLOOR:.2f}**, "
        f"LTM@alpha >= 0.0, and drop >= **{_THRESH_TIER2_INTERNAL_DROP:.2f}** |"
    )
    lines.append("| `neutral` | None of the above |")
    lines.append("")
    lines.append("### Tail-Specific Fragility (Tier 3)")
    lines.append("")
    lines.append(
        "Examines whether a subgroup is overrepresented in the global "
        "lower tail and whether its tail samples are more severely affected."
    )
    lines.append("")
    lines.append(
        f"- **Enriched**: tail prevalence ratio >= "
        f"**{_THRESH_SUBGROUP_TAIL_PREVALENCE_RATIO:.1f}x** "
        "(subgroup contributes disproportionately many tail samples)"
    )
    lines.append(
        f"- **Severe**: tail mean CRoMa delta vs rest <= "
        f"**-{_THRESH_TAIL_SEVERITY_MEANINGFUL_GAP:.2f}** "
        "(subgroup's tail samples are meaningfully worse than others')"
    )
    lines.append("")
    lines.append("| Status | Condition |")
    lines.append("| --- | --- |")
    lines.append("| `tail_enriched_and_severe` | Enriched and severe |")
    lines.append("| `tail_enriched` | Enriched only |")
    lines.append("| `tail_severe` | Severe only |")
    lines.append("| `neutral` | Neither |")
    lines.append("")
    lines.append(
        "Severity label uses the same delta: "
        f"<= **-{_THRESH_TAIL_SEVERITY_MEANINGFUL_GAP:.2f}** = \"more severe\", "
        f">= **+{_THRESH_TAIL_SEVERITY_MEANINGFUL_GAP:.2f}** = \"not more severe\", "
        "otherwise \"similar\"."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _model_action_flags(*, df_model: pd.DataFrame) -> pd.DataFrame:
    """Per-model action flags derived from the aggregated metrics table.

    Only flags computable from ``df_model`` itself are emitted (coverage risk,
    OO-dominated poor-embedding, and the lower-tail-mean gap). Rank-shift and
    k/m-sweep flags were dropped along with those duplicated sections.
    """
    rows: list[dict] = []

    # Coverage risks: RI/MaRI undefined coverage is shared in this benchmark path,
    # so emit one model-level flag using the max available undefined fraction.
    coverage_cols = [
        c
        for c in ("ri_undefined_frac", "mari_undefined_frac", "croma_undefined_frac")
        if c in df_model.columns
    ]
    if coverage_cols:
        for _, row in df_model.iterrows():
            coverage_values = [
                float(row[c]) for c in coverage_cols if np.isfinite(row[c])
            ]
            if not coverage_values:
                continue
            coverage_value = float(max(coverage_values))
            if coverage_value >= _THRESH_UNDEFINED_COVERAGE_RISK:
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": "coverage_risk",
                        "severity": "high",
                        "value": coverage_value,
                        "threshold": _THRESH_UNDEFINED_COVERAGE_RISK,
                        "detail": f"Undefined coverage is high (max undefined fraction={coverage_value:.3f}).",
                    }
                )

    # Undefined breakdown flags: retain one OO-dominated poor-embedding warning.
    oo_cols = [
        c
        for c in ("ri_oo_dominated_undefined_frac", "mari_oo_dominated_undefined_frac")
        if c in df_model.columns
    ]
    if oo_cols:
        for _, row in df_model.iterrows():
            oo_values = [float(row[c]) for c in oo_cols if np.isfinite(row[c])]
            if not oo_values:
                continue
            oo_value = float(max(oo_values))
            if oo_value >= _THRESH_OO_DOMINATED_HIGH:
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": "poor_embedding",
                        "severity": "high",
                        "value": oo_value,
                        "threshold": _THRESH_OO_DOMINATED_HIGH,
                        "detail": f"Significant undefined mass is OO-dominated (max OO fraction={oo_value:.3f}).",
                    }
                )

    # Tail gap risks for CRoMa summaries: keep only the lower-tail-mean gap.
    if {"croma", "croma_ltm_alpha"}.issubset(df_model.columns):
        df_tmp = df_model.loc[:, ["model", "croma", "croma_ltm_alpha"]].copy()
        df_tmp["tail_gap_ltm"] = df_tmp["croma"] - df_tmp["croma_ltm_alpha"]
        for _, row in df_tmp[df_tmp["tail_gap_ltm"] >= _THRESH_TAIL_GAP_LTM].iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": "tail_gap_ltm_high",
                    "severity": "medium",
                    "value": float(row["tail_gap_ltm"]),
                    "threshold": _THRESH_TAIL_GAP_LTM,
                    "detail": f"CRoMa - LTM(CRoMa) tail gap is large ({row['tail_gap_ltm']:.3f}).",
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = (
        out.sort_values(
            ["severity", "flag", "model", "value"], ascending=[True, True, True, False]
        )
        .drop_duplicates(subset=["model", "flag"], keep="first")
        .reset_index(drop=True)
    )
    return out


def main() -> int:
    args = _parse_args()
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else metrics_csv.parent / "analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(metrics_csv)
    if len(df_raw) == 0:
        raise ValueError(f"Metrics CSV is empty: {metrics_csv}")

    df_model = _aggregate_by_model(df_raw)
    action_flags_df = _model_action_flags(df_model=df_model)

    per_sample_path = metrics_csv.parent / "per_sample_metrics.csv"
    df_per_sample = _load_optional_csv(per_sample_path)
    subgroup_df, subgroup_context_df = _build_croma_subgroup_analysis(df_per_sample)

    action_flags_df.to_csv(out_dir / "model_action_flags.csv", index=False)
    if df_per_sample is not None:
        subgroup_df.to_csv(out_dir / "model_specific_croma_subgroups.csv", index=False)
        (out_dir / "model_specific_croma_subgroups.md").write_text(
            _render_croma_subgroup_markdown(subgroup_df, subgroup_context_df),
            encoding="utf-8",
        )

    print(f"[analyze_results] wrote analysis to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
