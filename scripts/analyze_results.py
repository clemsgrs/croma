import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from croma.metrics.tail import compute_tail_metrics
except ModuleNotFoundError:
    import sys

    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from croma.metrics.tail import compute_tail_metrics


_TOP_METRICS_CANONICAL = ["ri", "mari", "ccmr", "ccmr_q_alpha", "ccmr_ltm_alpha"]
_CORR_METRICS_CANONICAL = ["ri", "mari", "ccmr"]
_RANK_METRICS_CANONICAL = ["ri", "mari", "ccmr"]
_RANK_SHIFT_PAIRS = [("ri", "mari"), ("ri", "ccmr"), ("mari", "ccmr")]

_DISPLAY_NAMES = {
    "ri": "RI",
    "mari": "MaRI",
    "ccmr": "CCMR",
    "ccmr_q_alpha": "Q(CCMR)",
    "ccmr_ltm_alpha": "LTM(CCMR)",
}

_HIGHER_IS_BETTER = {
    "ri",
    "mari",
    "ccmr",
    "ccmr_q_alpha",
    "ccmr_ltm_alpha",
    "bio_knn_bacc",
}

_THRESH_RANK_SHIFT = 2.0
_THRESH_UNDEFINED_COVERAGE_RISK = 0.25
_THRESH_OO_DOMINATED_HIGH = 0.10
_THRESH_TAIL_GAP_LTM = 0.20
_THRESH_K_SWEEP_RANGE = 0.15
_THRESH_M_SWEEP_CCMR_GAIN = 0.08
_THRESH_SUBGROUP_TAIL_PREVALENCE_RATIO = 2.0
_THRESH_TAIL_SEVERITY_MEANINGFUL_GAP = 0.05
_THRESH_TIER1_MEDIAN_DELTA = 0.05
_THRESH_TIER1_LT1_DELTA = 0.05
_THRESH_TIER2_ROBUST_MEDIAN_FLOOR = 1.15
_THRESH_TIER2_LTM_CEILING = 0.90
_THRESH_TIER2_INTERNAL_DROP = 0.25
_THRESH_TIER2_LT1_FRAC_FLOOR = 0.15
_THRESH_TIER2_MIN_SAMPLES = 10
_THRESH_TIER2_MIN_LT1_COUNT = 3

_SUBGROUP_SCORE_COLUMN = "ccmr_m1"
_SUBGROUP_MIN_HEADLINE_SAMPLES = 2
_SUBGROUP_SCOPE_ORDER = ("stratum", "label", "medical_center")
_SUBGROUP_SCOPE_TO_COLUMNS = {
    "stratum": ("label", "medical_center"),
    "label": ("label",),
    "medical_center": ("medical_center",),
}
_SUBGROUP_SCOPE_TO_TITLE = {
    "stratum": "Strata",
    "label": "Biology",
    "medical_center": "Centers",
}
_SCOPE_SORT_ORDER = {
    "stratum": 0,
    "label": 1,
    "medical_center": 2,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark metrics: correlations, model ranks, and rank changes."
    )
    parser.add_argument("--metrics-csv", required=True, type=Path, help="Path to benchmark metrics CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for analysis artifacts (default: <metrics parent>/analysis).",
    )
    parser.add_argument(
        "--rank-reference",
        default="RI",
        help="Reference metric for rank deltas (case-insensitive, default: RI).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k models to highlight per metric.")
    parser.add_argument(
        "--k-sweep-csv",
        type=Path,
        default=None,
        help="Optional k-sweep CSV path (default: auto-detect next to metrics CSV).",
    )
    parser.add_argument(
        "--ccmr-m-sweep-csv",
        type=Path,
        default=None,
        help="Optional CCMR m-sweep CSV path (default: auto-detect next to metrics CSV).",
    )
    return parser.parse_args()


def _resolve_metric_name(name: str, available: list[str]) -> str:
    key = str(name).strip().lower()
    by_lower = {c.lower(): c for c in available}
    if key not in by_lower:
        raise ValueError(f"Unknown metric '{name}'. Available: {', '.join(available)}")
    return by_lower[key]


def _resolve_required_metrics(canonical: list[str], numeric_cols: list[str], label: str) -> list[str]:
    resolved: list[str] = []
    missing: list[str] = []
    for name in canonical:
        try:
            resolved.append(_resolve_metric_name(name, numeric_cols))
        except ValueError:
            missing.append(name)
    if missing:
        raise ValueError(
            f"metrics CSV missing required {label} metrics: {', '.join(missing)}. "
            f"Available numeric columns: {', '.join(numeric_cols)}"
        )
    return resolved


def _is_higher_better(metric_name: str) -> bool:
    return str(metric_name).strip().lower() in _HIGHER_IS_BETTER


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
    numeric_cols = [c for c in working.columns if c != "model" and pd.api.types.is_numeric_dtype(working[c])]
    if not numeric_cols:
        raise ValueError("metrics CSV has no numeric columns to analyze")
    grouped = working.groupby("model", as_index=False)[numeric_cols].mean(numeric_only=True)
    return grouped


def _correlation_outputs(df_model: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_df = df_model.loc[:, metrics]
    pearson = pd.DataFrame(np.nan, index=metrics, columns=metrics, dtype=float)
    spearman = pd.DataFrame(np.nan, index=metrics, columns=metrics, dtype=float)

    for m1 in metrics:
        s1 = corr_df[m1]
        unique_1 = int(s1.dropna().nunique())
        for m2 in metrics:
            s2 = corr_df[m2]
            unique_2 = int(s2.dropna().nunique())
            if m1 == m2:
                if unique_1 >= 2:
                    pearson.loc[m1, m2] = 1.0
                    spearman.loc[m1, m2] = 1.0
                continue
            if unique_1 < 2 or unique_2 < 2:
                continue
            pearson.loc[m1, m2] = float(s1.corr(s2, method="pearson"))
            spearman.loc[m1, m2] = float(s1.corr(s2, method="spearman"))

    return pearson, spearman


def _rank_table(df_model: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df_model.loc[:, ["model"]].copy()
    for metric in metrics:
        ascending = not _is_higher_better(metric)
        out[f"rank_{metric}"] = df_model[metric].rank(method="min", ascending=ascending)
    return out


def _top_models(df_model: pd.DataFrame, metrics: list[str], top_k: int) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in metrics:
        ascending = not _is_higher_better(metric)
        top = df_model.sort_values(metric, ascending=ascending).head(int(top_k))
        for pos, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append(
                {
                    "metric": str(metric),
                    "rank_position": int(pos),
                    "model": str(row["model"]),
                    "value": float(row[metric]),
                }
            )
    return pd.DataFrame(rows)


def _rank_deltas(rank_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric_a, metric_b in _RANK_SHIFT_PAIRS:
        col_a = f"rank_{metric_a}"
        col_b = f"rank_{metric_b}"
        if col_a not in rank_df.columns or col_b not in rank_df.columns:
            continue
        for _, row in rank_df.iterrows():
            rank_a = float(row[col_a])
            rank_b = float(row[col_b])
            improvement_delta = rank_a - rank_b
            if improvement_delta > 0:
                direction = "improvement"
            elif improvement_delta < 0:
                direction = "downgrade"
            else:
                direction = "no_change"
            rows.append(
                {
                    "model": str(row["model"]),
                    "metric_a": str(metric_a),
                    "metric_b": str(metric_b),
                    "pair": f"{metric_a}_vs_{metric_b}",
                    "rank_a": rank_a,
                    "rank_b": rank_b,
                    "improvement_delta": float(improvement_delta),
                    "improvement_delta_signed": f"{improvement_delta:+.0f}",
                    "direction": direction,
                    "abs_improvement_delta": float(abs(improvement_delta)),
                }
            )
    return pd.DataFrame(rows)


def _rank_agreement(rank_df: pd.DataFrame) -> pd.DataFrame:
    rank_cols = [c for c in rank_df.columns if c.startswith("rank_")]
    rows: list[dict] = []
    for i, c1 in enumerate(rank_cols):
        for c2 in rank_cols[i + 1 :]:
            metric_1 = c1[len("rank_"):]
            metric_2 = c2[len("rank_"):]
            s1 = rank_df[c1]
            s2 = rank_df[c2]
            if int(s1.dropna().nunique()) < 2 or int(s2.dropna().nunique()) < 2:
                spearman = float("nan")
                kendall = float("nan")
            else:
                spearman = float(s1.corr(s2, method="spearman"))
                kendall = float(s1.corr(s2, method="kendall"))
            rows.append(
                {
                    "metric_1": metric_1,
                    "metric_2": metric_2,
                    "spearman": spearman,
                    "kendall": kendall,
                }
            )
    return pd.DataFrame(rows)


def _strongest_corr_pairs(corr: pd.DataFrame, top_n: int = 5) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            val = float(corr.loc[c1, c2])
            if np.isfinite(val):
                rows.append((c1, c2, val))
    rows.sort(key=lambda x: abs(x[2]), reverse=True)
    return rows[: int(top_n)]


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
            "label",
            "medical_center",
            "n_samples",
            "group_frac",
            "mean_ccmr",
            "rest_mean_ccmr",
            "mean_ccmr_delta_vs_rest",
            "median_ccmr",
            "rest_median_ccmr",
            "median_ccmr_delta_vs_rest",
            "ccmr_lt1_frac",
            "ccmr_lt1_count",
            "rest_ccmr_lt1_frac",
            "ccmr_lt1_frac_delta_vs_rest",
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
            "tail_mean_ccmr",
            "rest_tail_mean_ccmr",
            "tail_mean_ccmr_delta_vs_rest",
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
            "ccmr_alpha",
            "pooled_mean_ccmr",
            "pooled_median_ccmr",
            "pooled_tail_mean_ccmr",
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


def _sort_tail_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([_SUBGROUP_SCORE_COLUMN, "sample_id"], ascending=[True, True], kind="mergesort").reset_index(drop=True)


def _render_subgroup_name(row: pd.Series, scope: str) -> str:
    if scope == "stratum":
        return f"{row['label']} / {row['medical_center']}"
    if scope == "label":
        return str(row["label"])
    return str(row["medical_center"])


def _subgroup_report_sort(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.reset_index(drop=True)
    working = df.copy()
    if "tail_prevalence_ratio" in working.columns:
        working["_tail_signal"] = working["tail_prevalence_ratio"]
    else:
        working["_tail_signal"] = float("nan")
    sort_cols = ["_tail_signal", "tail_prevalence", "mean_ccmr", "ccmr_lt1_frac", "tail_mean_ccmr", "label", "medical_center"]
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


def _fmt_pct(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(parsed):
        return "NA"
    return f"{parsed * 100.0:.1f}%"


def _fmt_ratio(value: object) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(parsed):
        return "NA"
    return f"{parsed:.1f}x"


def _render_context_heading(*, dataset: object, context_id: object, evaluation_design: object) -> str:
    return f"{dataset} / {context_id} ({evaluation_design})"


def _tier1_status(
    *,
    n_samples: int,
    median_ccmr: float,
    rest_median_ccmr: float,
    median_delta: float,
    ccmr_lt1_delta: float,
) -> str:
    if n_samples < _SUBGROUP_MIN_HEADLINE_SAMPLES:
        return "insufficient_support"
    if median_delta <= -_THRESH_TIER1_MEDIAN_DELTA and ccmr_lt1_delta >= _THRESH_TIER1_LT1_DELTA:
        if median_ccmr < 1.0 and rest_median_ccmr >= 1.0:
            return "broad_weakness"
        if median_ccmr >= 1.0 and rest_median_ccmr >= 1.0:
            return "relative_weakness"
        if median_ccmr < 1.0 and rest_median_ccmr < 1.0:
            return "aggravated_weakness"
    return "neutral"


def _tier2_status(
    *,
    n_samples: int,
    median_ccmr: float,
    subgroup_ltm_alpha: float,
    internal_tail_drop: float,
    ccmr_lt1_frac: float,
    ccmr_lt1_count: int,
) -> str:
    if n_samples < _THRESH_TIER2_MIN_SAMPLES:
        return "insufficient_support"
    pocket_gate = (
        subgroup_ltm_alpha <= _THRESH_TIER2_LTM_CEILING
        and internal_tail_drop >= _THRESH_TIER2_INTERNAL_DROP
        and ccmr_lt1_frac >= _THRESH_TIER2_LT1_FRAC_FLOOR
        and ccmr_lt1_count >= _THRESH_TIER2_MIN_LT1_COUNT
    )
    if pocket_gate and median_ccmr < 1.0:
        return "aggravated_weakness"
    if pocket_gate and median_ccmr >= _THRESH_TIER2_ROBUST_MEDIAN_FLOOR:
        return "hidden_pocket"
    if (
        median_ccmr >= _THRESH_TIER2_ROBUST_MEDIAN_FLOOR
        and subgroup_ltm_alpha >= 1.0
        and internal_tail_drop >= _THRESH_TIER2_INTERNAL_DROP
    ):
        return "internal_spread"
    return "neutral"


def _tail_severity_label(*, tail_count: int, rest_tail_count: int, tail_delta: float) -> str:
    if tail_count == 0:
        return "no_tail_samples"
    if rest_tail_count == 0 or not np.isfinite(tail_delta):
        return "no_rest_tail"
    if tail_delta <= -_THRESH_TAIL_SEVERITY_MEANINGFUL_GAP:
        return "more severe"
    if tail_delta >= _THRESH_TAIL_SEVERITY_MEANINGFUL_GAP:
        return "not more severe"
    return "similar"


def _tier3_status(*, n_samples: int, tail_prevalence_ratio: float, tail_severity_label: str) -> str:
    if n_samples < _SUBGROUP_MIN_HEADLINE_SAMPLES:
        return "insufficient_support"
    enriched = np.isfinite(tail_prevalence_ratio) and tail_prevalence_ratio >= _THRESH_SUBGROUP_TAIL_PREVALENCE_RATIO
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
    working["_scope_order"] = working["scope"].map(_SCOPE_SORT_ORDER).fillna(99).astype(int)
    if tier == "tier1":
        status_order = {
            "broad_weakness": 0,
            "aggravated_weakness": 1,
            "relative_weakness": 2,
            "neutral": 3,
            "insufficient_support": 4,
        }
        working["_status_order"] = working["tier1_status"].map(status_order).fillna(9).astype(int)
        sort_cols = ["_status_order", "median_ccmr_delta_vs_rest", "ccmr_lt1_frac_delta_vs_rest", "_scope_order", "subgroup_name"]
        ascending = [True, True, False, True, True]
    elif tier == "tier2":
        status_order = {
            "hidden_pocket": 0,
            "aggravated_weakness": 1,
            "internal_spread": 2,
            "neutral": 3,
            "insufficient_support": 4,
        }
        working["_status_order"] = working["tier2_status"].map(status_order).fillna(9).astype(int)
        sort_cols = ["_status_order", "internal_tail_drop", "median_ccmr", "_scope_order", "subgroup_name"]
        ascending = [True, False, False, True, True]
    else:
        status_order = {
            "tail_enriched_and_severe": 0,
            "tail_severe": 1,
            "tail_enriched": 2,
            "neutral": 3,
            "insufficient_support": 4,
        }
        working["_status_order"] = working["tier3_status"].map(status_order).fillna(9).astype(int)
        sort_cols = ["_status_order", "tail_prevalence_ratio", "tail_mean_ccmr_delta_vs_rest", "_scope_order", "subgroup_name"]
        ascending = [True, False, True, True, True]
    return working.sort_values(sort_cols, ascending=ascending, kind="mergesort").drop(columns=["_scope_order", "_status_order"]).reset_index(drop=True)


def _render_markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
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
            "Median CCMR",
            "Rest Median",
            "Median Delta",
            "CCMR<1 Frac",
            "Rest CCMR<1 Frac",
            "CCMR<1 Delta",
        ]
        rows = [
            [
                str(row["scope"]),
                str(row["subgroup_name"]),
                str(row["tier1_status"]),
                _fmt_float(row["median_ccmr"]),
                _fmt_float(row["rest_median_ccmr"]),
                _fmt_float(row["median_ccmr_delta_vs_rest"]),
                _fmt_float(row["ccmr_lt1_frac"]),
                _fmt_float(row["rest_ccmr_lt1_frac"]),
                _fmt_float(row["ccmr_lt1_frac_delta_vs_rest"]),
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
            "CCMR<1 Frac",
            "CCMR<1 Count",
            "Median CCMR",
            "Subgroup LTM@alpha",
            "Drop",
        ]
        rows = [
            [
                str(row["scope"]),
                str(row["subgroup_name"]),
                str(row["tier2_status"]),
                str(int(row["n_samples"])),
                _fmt_float(row["ccmr_lt1_frac"]),
                str(int(row["ccmr_lt1_count"])),
                _fmt_float(row["median_ccmr"]),
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
        "Tail Mean CCMR",
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
            _fmt_float(row["tail_mean_ccmr"]),
            _fmt_float(row["rest_tail_mean_ccmr"]),
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
        "ccmr_alpha": float(alpha),
        "pooled_mean_ccmr": float("nan"),
        "pooled_median_ccmr": float("nan"),
        "pooled_tail_mean_ccmr": float("nan"),
        "skipped": False,
        "skip_reason": "",
    }


def _build_ccmr_subgroup_analysis(df_per_sample: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        "medical_center",
        "ccmr_alpha",
        _SUBGROUP_SCORE_COLUMN,
    }
    if not required.issubset(df_per_sample.columns):
        return _empty_subgroup_df(), _empty_context_df()

    working = df_per_sample.copy()
    for col in ("dataset", "model", "evaluation_design", "evaluation_unit", "subset", "sample_id", "label", "medical_center"):
        working[col] = working[col].astype(str)
    working[_SUBGROUP_SCORE_COLUMN] = pd.to_numeric(working[_SUBGROUP_SCORE_COLUMN], errors="coerce")
    working["ccmr_alpha"] = pd.to_numeric(working["ccmr_alpha"], errors="coerce")
    working["context_id"] = np.where(
        working["evaluation_design"] == "paired_2x2",
        working["subset"],
        "dataset",
    )

    group_cols = ["dataset", "model", "evaluation_design", "evaluation_unit", "context_id"]
    for keys, group in working.groupby(group_cols, sort=True, dropna=False):
        dataset, model, evaluation_design, evaluation_unit, context_id = [str(v) for v in keys]
        defined = group[np.isfinite(group[_SUBGROUP_SCORE_COLUMN])].copy()
        alpha = _safe_float(group["ccmr_alpha"].dropna().iloc[0] if group["ccmr_alpha"].notna().any() else np.nan, 0.10)
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
            context_row["skip_reason"] = "Skipped: no defined CCMR(m=1) samples are available for this context."
            context_rows.append(context_row)
            continue

        if evaluation_design == "dataset_wide" and n_labels != 2:
            context_row["skipped"] = True
            context_row["skip_reason"] = (
                "Skipped: pooled dataset-wide subgroup interpretation is not reported because "
                "heterogeneous biological boundaries make the result clinically weak; use paired runs instead."
            )
            context_rows.append(context_row)
            continue

        defined = _sort_tail_rows(defined)
        tail_size = int(max(1, int(np.ceil(alpha * len(defined)))))
        tail_index = defined.index[:tail_size]
        defined["is_tail"] = defined.index.isin(tail_index)

        context_row["tail_size"] = int(tail_size)
        context_row["pooled_mean_ccmr"] = float(defined[_SUBGROUP_SCORE_COLUMN].mean())
        context_row["pooled_median_ccmr"] = float(defined[_SUBGROUP_SCORE_COLUMN].median())
        context_row["pooled_tail_mean_ccmr"] = float(defined.loc[defined["is_tail"], _SUBGROUP_SCORE_COLUMN].mean())
        context_rows.append(context_row)

        n_defined = int(len(defined))
        context_mean_ccmr = float(context_row["pooled_mean_ccmr"])
        context_tail_prevalence = float(tail_size / n_defined) if n_defined > 0 else float("nan")
        context_tail_mean_ccmr = float(context_row["pooled_tail_mean_ccmr"])
        for scope in _SUBGROUP_SCOPE_ORDER:
            cols = list(_SUBGROUP_SCOPE_TO_COLUMNS[scope])
            grouped = defined.groupby(cols, sort=True, dropna=False)
            for subgroup_key, subgroup in grouped:
                if not isinstance(subgroup_key, tuple):
                    subgroup_key = (subgroup_key,)
                label = ""
                medical_center = ""
                if "label" in cols:
                    label = str(subgroup_key[cols.index("label")])
                if "medical_center" in cols:
                    medical_center = str(subgroup_key[cols.index("medical_center")])

                n_samples = int(len(subgroup))
                tail_count = int(subgroup["is_tail"].sum())
                group_frac = float(n_samples / n_defined)
                mean_ccmr = float(subgroup[_SUBGROUP_SCORE_COLUMN].mean())
                median_ccmr = float(subgroup[_SUBGROUP_SCORE_COLUMN].median())
                ccmr_lt1_frac = float((subgroup[_SUBGROUP_SCORE_COLUMN] < 1.0).mean())
                ccmr_lt1_count = int((subgroup[_SUBGROUP_SCORE_COLUMN] < 1.0).sum())
                tail_prevalence = float(tail_count / n_samples)
                tail_prevalence_ratio = (
                    float(tail_prevalence / context_tail_prevalence)
                    if np.isfinite(context_tail_prevalence) and context_tail_prevalence > 0
                    else float("nan")
                )
                tail_share = float(tail_count / tail_size) if tail_size > 0 else float("nan")
                tail_mean_ccmr = (
                    float(subgroup.loc[subgroup["is_tail"], _SUBGROUP_SCORE_COLUMN].mean())
                    if tail_count > 0
                    else float("nan")
                )
                rest = defined.loc[~defined.index.isin(subgroup.index)]
                rest_mean_ccmr = float(rest[_SUBGROUP_SCORE_COLUMN].mean()) if len(rest) > 0 else float("nan")
                rest_median_ccmr = float(rest[_SUBGROUP_SCORE_COLUMN].median()) if len(rest) > 0 else float("nan")
                rest_ccmr_lt1_frac = float((rest[_SUBGROUP_SCORE_COLUMN] < 1.0).mean()) if len(rest) > 0 else float("nan")
                rest_tail_count = int(rest["is_tail"].sum()) if len(rest) > 0 else 0
                rest_tail_mean_ccmr = (
                    float(rest.loc[rest["is_tail"], _SUBGROUP_SCORE_COLUMN].mean())
                    if rest_tail_count > 0
                    else float("nan")
                )
                subgroup_tail_metrics = compute_tail_metrics(
                    subgroup[_SUBGROUP_SCORE_COLUMN].to_numpy(dtype=float),
                    alpha=alpha,
                )
                median_ccmr_delta_vs_rest = (
                    float(median_ccmr - rest_median_ccmr) if np.isfinite(rest_median_ccmr) else float("nan")
                )
                ccmr_lt1_frac_delta_vs_rest = (
                    float(ccmr_lt1_frac - rest_ccmr_lt1_frac) if np.isfinite(rest_ccmr_lt1_frac) else float("nan")
                )
                tail_mean_ccmr_delta_vs_rest = (
                    float(tail_mean_ccmr - rest_tail_mean_ccmr)
                    if np.isfinite(tail_mean_ccmr) and np.isfinite(rest_tail_mean_ccmr)
                    else float("nan")
                )
                internal_tail_drop = (
                    float(median_ccmr - subgroup_tail_metrics.ltm_alpha)
                    if np.isfinite(subgroup_tail_metrics.ltm_alpha)
                    else float("nan")
                )
                subgroup_name = _render_subgroup_name(
                    pd.Series({"label": label, "medical_center": medical_center}),
                    scope=scope,
                )
                tier1_status = _tier1_status(
                    n_samples=n_samples,
                    median_ccmr=median_ccmr,
                    rest_median_ccmr=rest_median_ccmr,
                    median_delta=median_ccmr_delta_vs_rest,
                    ccmr_lt1_delta=ccmr_lt1_frac_delta_vs_rest,
                )
                tail_severity_label = _tail_severity_label(
                    tail_count=tail_count,
                    rest_tail_count=rest_tail_count,
                    tail_delta=tail_mean_ccmr_delta_vs_rest,
                )
                tier2_status = _tier2_status(
                    n_samples=n_samples,
                    median_ccmr=median_ccmr,
                    subgroup_ltm_alpha=float(subgroup_tail_metrics.ltm_alpha),
                    internal_tail_drop=internal_tail_drop,
                    ccmr_lt1_frac=ccmr_lt1_frac,
                    ccmr_lt1_count=ccmr_lt1_count,
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
                        "label": label,
                        "medical_center": medical_center,
                        "n_samples": n_samples,
                        "group_frac": group_frac,
                        "mean_ccmr": mean_ccmr,
                        "rest_mean_ccmr": rest_mean_ccmr,
                        "mean_ccmr_delta_vs_rest": (
                            float(mean_ccmr - rest_mean_ccmr) if np.isfinite(rest_mean_ccmr) else float("nan")
                        ),
                        "median_ccmr": median_ccmr,
                        "rest_median_ccmr": rest_median_ccmr,
                        "median_ccmr_delta_vs_rest": median_ccmr_delta_vs_rest,
                        "ccmr_lt1_frac": ccmr_lt1_frac,
                        "ccmr_lt1_count": ccmr_lt1_count,
                        "rest_ccmr_lt1_frac": rest_ccmr_lt1_frac,
                        "ccmr_lt1_frac_delta_vs_rest": ccmr_lt1_frac_delta_vs_rest,
                        "subgroup_q_alpha": float(subgroup_tail_metrics.q_alpha),
                        "subgroup_ltm_alpha": float(subgroup_tail_metrics.ltm_alpha),
                        "internal_tail_drop": internal_tail_drop,
                        "tier1_status": tier1_status,
                        "tier2_status": tier2_status,
                        "tail_count": tail_count,
                        "tail_prevalence": tail_prevalence,
                        "context_tail_prevalence": context_tail_prevalence,
                        "tail_prevalence_delta": float(tail_prevalence - context_tail_prevalence),
                        "tail_prevalence_ratio": tail_prevalence_ratio,
                        "tail_share": tail_share,
                        "tail_mean_ccmr": tail_mean_ccmr,
                        "rest_tail_mean_ccmr": rest_tail_mean_ccmr,
                        "tail_mean_ccmr_delta_vs_rest": tail_mean_ccmr_delta_vs_rest,
                        "tail_severity_label": tail_severity_label,
                        "tier3_status": tier3_status,
                        "n_defined_samples": n_defined,
                        "tail_size": tail_size,
                        "headline_eligible": bool(n_samples >= _SUBGROUP_MIN_HEADLINE_SAMPLES),
                    }
                )

    context_out = pd.DataFrame(context_rows) if context_rows else _empty_context_df()
    if not subgroup_rows:
        return _empty_subgroup_df(), context_out

    subgroup_out = pd.DataFrame(subgroup_rows)
    ranked_frames: list[pd.DataFrame] = []
    rank_group_cols = ["dataset", "model", "evaluation_design", "evaluation_unit", "context_id", "scope"]
    for _, frame in subgroup_out.groupby(rank_group_cols, sort=False, dropna=False):
        ranked = _subgroup_report_sort(frame)
        ranked["report_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
        ranked_frames.append(ranked)
    subgroup_out = pd.concat(ranked_frames, ignore_index=True) if ranked_frames else _empty_subgroup_df()
    subgroup_out = subgroup_out.loc[:, _empty_subgroup_df().columns]
    context_out = context_out.loc[:, _empty_context_df().columns]
    return subgroup_out, context_out


def _render_ccmr_subgroup_markdown(subgroup_df: pd.DataFrame, context_df: pd.DataFrame) -> str:
    lines: list[str] = ["# Model-Specific CCMR Subgroup Analysis", ""]
    if len(context_df) == 0:
        lines.append("- No per-sample CCMR(m=1) contexts available.")
        return "\n".join(lines) + "\n"

    grouped_models = context_df.groupby("model", sort=True, dropna=False)
    for model, model_contexts in grouped_models:
        lines.append(f"## {model}")
        lines.append("")
        for _, context in model_contexts.sort_values(["dataset", "context_id"], kind="mergesort").iterrows():
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
                f"pooled_mean_ccmr={float(context['pooled_mean_ccmr']):.3f}, "
                f"pooled_median_ccmr={float(context['pooled_median_ccmr']):.3f}, "
                f"tail_mean_ccmr={float(context['pooled_tail_mean_ccmr']):.3f}."
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
    return "\n".join(lines).rstrip() + "\n"


def _k_sweep_sensitivity(df_k: pd.DataFrame | None) -> pd.DataFrame:
    if df_k is None:
        return pd.DataFrame()
    required = {"model", "k", "ri", "mari"}
    if not required.issubset(df_k.columns):
        return pd.DataFrame()
    working = df_k.copy()
    working["model"] = _scoped_model_labels(working)

    grouped = (
        working.groupby("model", as_index=False)
        .agg(
            k_min=("k", "min"),
            k_max=("k", "max"),
            ri_min=("ri", "min"),
            ri_max=("ri", "max"),
            mari_min=("mari", "min"),
            mari_max=("mari", "max"),
        )
        .copy()
    )
    grouped["ri_range"] = grouped["ri_max"] - grouped["ri_min"]
    grouped["mari_range"] = grouped["mari_max"] - grouped["mari_min"]
    grouped["max_range"] = grouped[["ri_range", "mari_range"]].max(axis=1)
    grouped = grouped.sort_values("max_range", ascending=False).reset_index(drop=True)
    return grouped


def _ccmr_m_sweep_sensitivity(df_m: pd.DataFrame | None) -> pd.DataFrame:
    if df_m is None:
        return pd.DataFrame()
    required = {"model", "m", "ccmr", "ccmr_q_alpha", "ccmr_ltm_alpha"}
    if not required.issubset(df_m.columns):
        return pd.DataFrame()
    working = df_m.copy()
    working["model"] = _scoped_model_labels(working)

    grouped_rows: list[dict] = []
    for model, grp in working.groupby("model"):
        grp_sorted = grp.sort_values("m", ascending=True)
        row: dict = {
            "model": str(model),
            "m_min": float(grp_sorted["m"].iloc[0]),
            "m_max": float(grp_sorted["m"].iloc[-1]),
            "ccmr_m_min": float(grp_sorted["ccmr"].iloc[0]),
            "ccmr_m_max": float(grp_sorted["ccmr"].iloc[-1]),
            "ccmr_gain": float(grp_sorted["ccmr"].iloc[-1] - grp_sorted["ccmr"].iloc[0]),
            "q_gain": float(grp_sorted["ccmr_q_alpha"].iloc[-1] - grp_sorted["ccmr_q_alpha"].iloc[0]),
            "ltm_gain": float(grp_sorted["ccmr_ltm_alpha"].iloc[-1] - grp_sorted["ccmr_ltm_alpha"].iloc[0]),
        }
        if "ccmr_retries" in grp_sorted.columns:
            row["ccmr_retries_max"] = float(grp_sorted["ccmr_retries"].max())
        if "ccmr_k_final" in grp_sorted.columns:
            row["ccmr_k_final_max"] = float(grp_sorted["ccmr_k_final"].max())
        grouped_rows.append(row)

    out = pd.DataFrame(grouped_rows)
    if len(out) == 0:
        return out
    sort_col = "ccmr_gain"
    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


def _model_action_flags(
    *,
    df_model: pd.DataFrame,
    delta_df: pd.DataFrame,
    k_sensitivity_df: pd.DataFrame,
    ccmr_m_sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    # Rank disagreements with meaningful magnitude.
    if len(delta_df) > 0:
        shifted = delta_df[delta_df["abs_improvement_delta"] >= _THRESH_RANK_SHIFT]
        for _, row in shifted.iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": f"rank_shift_{row['pair']}",
                    "severity": "high",
                    "value": float(row["improvement_delta"]),
                    "threshold": _THRESH_RANK_SHIFT,
                    "detail": (
                        f"Rank shift between {_DISPLAY_NAMES.get(row['metric_a'], row['metric_a'])} and "
                        f"{_DISPLAY_NAMES.get(row['metric_b'], row['metric_b'])} is {row['improvement_delta_signed']}."
                    ),
                }
            )

    # Coverage risks: RI/MaRI undefined coverage is shared in this benchmark path,
    # so emit one model-level flag using the max available undefined fraction.
    coverage_cols = [c for c in ("ri_undefined_frac", "mari_undefined_frac", "ccmr_undefined_frac") if c in df_model.columns]
    if coverage_cols:
        for _, row in df_model.iterrows():
            coverage_values = [float(row[c]) for c in coverage_cols if np.isfinite(row[c])]
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
    oo_cols = [c for c in ("ri_oo_dominated_undefined_frac", "mari_oo_dominated_undefined_frac") if c in df_model.columns]
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

    # Tail gap risks for CCMR summaries: keep only the lower-tail-mean gap.
    if {"ccmr", "ccmr_ltm_alpha"}.issubset(df_model.columns):
        df_tmp = df_model.loc[:, ["model", "ccmr", "ccmr_ltm_alpha"]].copy()
        df_tmp["tail_gap_ltm"] = df_tmp["ccmr"] - df_tmp["ccmr_ltm_alpha"]
        for _, row in df_tmp[df_tmp["tail_gap_ltm"] >= _THRESH_TAIL_GAP_LTM].iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": "tail_gap_ltm_high",
                    "severity": "medium",
                    "value": float(row["tail_gap_ltm"]),
                    "threshold": _THRESH_TAIL_GAP_LTM,
                    "detail": f"CCMR - LTM(CCMR) tail gap is large ({row['tail_gap_ltm']:.3f}).",
                }
            )

    # k-sweep sensitivity.
    if len(k_sensitivity_df) > 0 and "max_range" in k_sensitivity_df.columns:
        for _, row in k_sensitivity_df[k_sensitivity_df["max_range"] >= _THRESH_K_SWEEP_RANGE].iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": "k_sweep_sensitivity_high",
                    "severity": "medium",
                    "value": float(row["max_range"]),
                    "threshold": _THRESH_K_SWEEP_RANGE,
                    "detail": f"k-sweep range is high (max RI/MaRI range={row['max_range']:.3f}).",
                }
            )

    # m-sweep CCMR gain and compute cost.
    if len(ccmr_m_sensitivity_df) > 0:
        if "ccmr_gain" in ccmr_m_sensitivity_df.columns:
            for _, row in ccmr_m_sensitivity_df[ccmr_m_sensitivity_df["ccmr_gain"] >= _THRESH_M_SWEEP_CCMR_GAIN].iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": "ccmr_m_sweep_gain_high",
                        "severity": "medium",
                        "value": float(row["ccmr_gain"]),
                        "threshold": _THRESH_M_SWEEP_CCMR_GAIN,
                        "detail": f"CCMR gain across m-sweep is high ({row['ccmr_gain']:.3f}).",
                    }
                )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    out = (
        out.sort_values(["severity", "flag", "model", "value"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["model", "flag"], keep="first")
        .reset_index(drop=True)
    )
    return out


def _write_report(
    *,
    out_path: Path,
    input_csv: Path,
    df_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    top_metrics: list[str],
    corr_metrics: list[str],
    rank_metrics: list[str],
    rank_reference: str,
    top_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    pearson: pd.DataFrame,
    top_k: int,
    action_flags_df: pd.DataFrame,
    k_sensitivity_df: pd.DataFrame,
    ccmr_m_sensitivity_df: pd.DataFrame,
) -> None:
    strong_corr = _strongest_corr_pairs(pearson, top_n=8)
    coverage_cols = [c for c in ("ri_undefined_frac", "mari_undefined_frac", "ccmr_undefined_frac") if c in df_model.columns]

    lines: list[str] = []
    lines.append("# Benchmark Metrics Analysis")
    lines.append("")
    lines.append(f"- Input CSV: `{input_csv}`")
    lines.append(f"- Raw rows: {len(df_raw)}")
    lines.append(f"- Unique models analyzed: {len(df_model)}")
    lines.append(f"- Top-model metrics: {', '.join(_DISPLAY_NAMES.get(m, m) for m in top_metrics)}")
    lines.append(f"- Correlation metrics: {', '.join(_DISPLAY_NAMES.get(m, m) for m in corr_metrics)}")
    lines.append(f"- Rank-shift metrics: {', '.join(_DISPLAY_NAMES.get(m, m) for m in rank_metrics)}")
    lines.append(f"- Rank reference: `{_DISPLAY_NAMES.get(rank_reference, rank_reference)}`")
    lines.append("")
    lines.append("## Top Models By Metric")
    lines.append("")
    for metric in top_metrics:
        lines.append(f"### {_DISPLAY_NAMES.get(metric, metric)}")
        metric_rows = top_df[top_df["metric"] == metric].sort_values("rank_position").head(int(top_k))
        for _, row in metric_rows.iterrows():
            lines.append(f"- #{int(row['rank_position'])} {row['model']} ({float(row['value']):.6g})")
        lines.append("")

    lines.append("")
    lines.append("## Pearson Correlations (RI / MaRI / CCMR)")
    lines.append("")
    for m1, m2, val in strong_corr:
        lines.append(f"- `{_DISPLAY_NAMES.get(m1, m1)}` vs `{_DISPLAY_NAMES.get(m2, m2)}`: {val:.4f}")
    lines.append("")
    lines.append("## Rank Shift Analysis (Pairwise)")
    lines.append("")
    for metric_a, metric_b in _RANK_SHIFT_PAIRS:
        pair_rows = delta_df[
            (delta_df["metric_a"] == metric_a) & (delta_df["metric_b"] == metric_b)
        ].sort_values("abs_improvement_delta", ascending=False)
        lines.append(f"### {_DISPLAY_NAMES.get(metric_a, metric_a)} vs {_DISPLAY_NAMES.get(metric_b, metric_b)}")
        shown_rows = pair_rows[pair_rows["abs_improvement_delta"] >= _THRESH_RANK_SHIFT]
        if len(shown_rows) == 0:
            lines.append(f"- No rank shifts with |delta| >= {int(_THRESH_RANK_SHIFT)}.")
            lines.append("")
            continue
        for _, row in shown_rows.head(10).iterrows():
            lines.append(
                f"- {row['model']}: "
                f"{_DISPLAY_NAMES.get(metric_a, metric_a)} rank {int(row['rank_a'])} -> "
                f"{_DISPLAY_NAMES.get(metric_b, metric_b)} rank {int(row['rank_b'])} "
                f"({row['improvement_delta_signed']})"
            )
        lines.append("")

    lines.append("")
    lines.append("## Coverage Risk")
    lines.append("")
    if not coverage_cols:
        lines.append("- Undefined-fraction columns unavailable.")
    else:
        coverage_flag_models = set(
            action_flags_df.loc[action_flags_df["flag"] == "coverage_risk", "model"].astype(str)
        )
        coverage_headers = ["Model", "Undefined Frac", "Coverage Risk"]
        coverage_rows: list[list[str]] = []
        for _, row in df_model.sort_values("model").iterrows():
            coverage_values = [float(row[c]) for c in coverage_cols if np.isfinite(row[c])]
            coverage_value = float(max(coverage_values)) if coverage_values else float("nan")
            coverage_rows.append(
                [
                    str(row["model"]),
                    _fmt_float(coverage_value),
                    "yes" if str(row["model"]) in coverage_flag_models else "no",
                ]
            )
        lines.extend(_render_markdown_table(coverage_headers, coverage_rows))
    lines.append("")

    filtered_action_flags_df = action_flags_df[
        ~action_flags_df["flag"].astype(str).eq("coverage_risk")
        & ~action_flags_df["flag"].astype(str).str.startswith("rank_shift_")
    ].reset_index(drop=True)

    lines.append("")
    lines.append("## Additional Insights and Action Flags")
    lines.append("")
    if len(filtered_action_flags_df) == 0:
        lines.append("- No additional action flags triggered beyond the dedicated coverage and rank-shift sections.")
    else:
        lines.append(f"- Models with >=1 additional action flag: {filtered_action_flags_df['model'].nunique()}")
        lines.append(f"- Total unique additional flags: {len(filtered_action_flags_df)}")
        lines.append("")
        lines.append("### Triggered Flags (Top 20)")
        for _, row in filtered_action_flags_df.head(20).iterrows():
            lines.append(
                f"- {row['model']}: `{row['flag']}` (value={float(row['value']):.3f}, "
                f"threshold={float(row['threshold']):.3f})"
            )
    lines.append("")
    lines.append("## K-Sweep Sensitivity")
    lines.append("")
    if len(k_sensitivity_df) == 0:
        lines.append("- k-sweep metrics unavailable.")
    else:
        lines.append(f"- Threshold: `max(ri_range, mari_range) >= {_THRESH_K_SWEEP_RANGE:.2f}`")
        for _, row in k_sensitivity_df.head(5).iterrows():
            lines.append(
                f"- {row['model']}: ri_range={float(row['ri_range']):.3f}, "
                f"mari_range={float(row['mari_range']):.3f}, max_range={float(row['max_range']):.3f}"
            )
    lines.append("")
    lines.append("## CCMR m-Sweep Sensitivity and Cost")
    lines.append("")
    if len(ccmr_m_sensitivity_df) == 0:
        lines.append("- CCMR m-sweep metrics unavailable.")
    else:
        lines.append(f"- Sensitivity threshold: `ccmr_gain >= {_THRESH_M_SWEEP_CCMR_GAIN:.2f}`")
        for _, row in ccmr_m_sensitivity_df.head(5).iterrows():
            lines.append(
                f"- {row['model']}: ccmr_gain={float(row['ccmr_gain']):.3f}, "
                f"q_gain={float(row['q_gain']):.3f}, ltm_gain={float(row['ltm_gain']):.3f}"
            )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else metrics_csv.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(metrics_csv)
    if len(df_raw) == 0:
        raise ValueError(f"Metrics CSV is empty: {metrics_csv}")

    df_model = _aggregate_by_model(df_raw)
    numeric_cols = [c for c in df_model.columns if c != "model" and pd.api.types.is_numeric_dtype(df_model[c])]
    top_metrics = _resolve_required_metrics(_TOP_METRICS_CANONICAL, numeric_cols, "top-model")
    corr_metrics = _resolve_required_metrics(_CORR_METRICS_CANONICAL, numeric_cols, "correlation")
    rank_metrics = _resolve_required_metrics(_RANK_METRICS_CANONICAL, numeric_cols, "rank-shift")

    rank_reference = _resolve_metric_name(str(args.rank_reference), rank_metrics)

    pearson_corr, spearman_corr = _correlation_outputs(df_model, metrics=corr_metrics)
    rank_df = _rank_table(df_model, metrics=rank_metrics)
    top_df = _top_models(df_model, metrics=top_metrics, top_k=int(args.top_k))
    delta_df = _rank_deltas(rank_df)
    agreement_df = _rank_agreement(rank_df)

    k_sweep_path = Path(args.k_sweep_csv) if args.k_sweep_csv is not None else metrics_csv.parent / "k_sweep_metrics.csv"
    ccmr_m_sweep_path = (
        Path(args.ccmr_m_sweep_csv) if args.ccmr_m_sweep_csv is not None else metrics_csv.parent / "ccmr_m_sweep_metrics.csv"
    )
    per_sample_path = metrics_csv.parent / "per_sample_metrics.csv"
    df_k_sweep = _load_optional_csv(k_sweep_path)
    df_ccmr_m_sweep = _load_optional_csv(ccmr_m_sweep_path)
    df_per_sample = _load_optional_csv(per_sample_path)
    k_sensitivity_df = _k_sweep_sensitivity(df_k_sweep)
    ccmr_m_sensitivity_df = _ccmr_m_sweep_sensitivity(df_ccmr_m_sweep)
    action_flags_df = _model_action_flags(
        df_model=df_model,
        delta_df=delta_df,
        k_sensitivity_df=k_sensitivity_df,
        ccmr_m_sensitivity_df=ccmr_m_sensitivity_df,
    )
    subgroup_df, subgroup_context_df = _build_ccmr_subgroup_analysis(df_per_sample)

    pearson_corr.to_csv(out_dir / "correlation_pearson.csv")
    spearman_corr.to_csv(out_dir / "correlation_spearman.csv")
    rank_df.to_csv(out_dir / "model_ranks.csv", index=False)
    top_df.to_csv(out_dir / "top_models_by_metric.csv", index=False)
    delta_df.to_csv(out_dir / "rank_deltas.csv", index=False)
    agreement_df.to_csv(out_dir / "rank_agreement.csv", index=False)
    action_flags_df.to_csv(out_dir / "model_action_flags.csv", index=False)
    if len(k_sensitivity_df) > 0:
        k_sensitivity_df.to_csv(out_dir / "k_sweep_sensitivity.csv", index=False)
    if len(ccmr_m_sensitivity_df) > 0:
        ccmr_m_sensitivity_df.to_csv(out_dir / "ccmr_m_sweep_sensitivity.csv", index=False)
    if df_per_sample is not None:
        subgroup_df.to_csv(out_dir / "model_specific_ccmr_subgroups.csv", index=False)
        (out_dir / "model_specific_ccmr_subgroups.md").write_text(
            _render_ccmr_subgroup_markdown(subgroup_df, subgroup_context_df),
            encoding="utf-8",
        )
    _write_report(
        out_path=out_dir / "analysis_report.md",
        input_csv=metrics_csv,
        df_raw=df_raw,
        df_model=df_model,
        top_metrics=top_metrics,
        corr_metrics=corr_metrics,
        rank_metrics=rank_metrics,
        rank_reference=rank_reference,
        top_df=top_df,
        delta_df=delta_df,
        pearson=pearson_corr,
        top_k=int(args.top_k),
        action_flags_df=action_flags_df,
        k_sensitivity_df=k_sensitivity_df,
        ccmr_m_sensitivity_df=ccmr_m_sensitivity_df,
    )

    print(f"[analyze_results] wrote analysis to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
