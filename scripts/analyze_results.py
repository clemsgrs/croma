import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mari.metrics.tail import select_exact_size_tail_set


_TOP_METRICS_CANONICAL = ["ri", "mari", "ccrr", "ccrr_q_alpha", "ccrr_ltm_alpha"]
_CORR_METRICS_CANONICAL = ["ri", "mari", "ccrr"]
_RANK_METRICS_CANONICAL = ["ri", "mari", "ccrr"]
_RANK_SHIFT_PAIRS = [("ri", "mari"), ("ri", "ccrr"), ("mari", "ccrr")]

_DISPLAY_NAMES = {
    "ri": "RI",
    "mari": "MaRI",
    "ccrr": "CCRR",
    "ccrr_q_alpha": "Q(CCRR)",
    "ccrr_ltm_alpha": "LTM(CCRR)",
}

_HIGHER_IS_BETTER = {
    "ri",
    "mari",
    "ccrr",
    "ccrr_q_alpha",
    "ccrr_ltm_alpha",
    "bio_knn_bacc",
}

_THRESH_RANK_SHIFT = 2.0
_THRESH_UNDEFINED_HIGH = 0.75
_THRESH_UNDEFINED_MODERATE = 0.30
_THRESH_SS_DOMINATED_HIGH = 0.20
_THRESH_OO_DOMINATED_HIGH = 0.10
_THRESH_SS_RATIO_OF_UNDEFINED = 0.80
_THRESH_TAIL_GAP_Q = 0.15
_THRESH_TAIL_GAP_LTM = 0.20
_THRESH_K_SWEEP_RANGE = 0.15
_THRESH_M_SWEEP_CCRR_GAIN = 0.08
_THRESH_CCRR_RETRIES_HIGH = 10.0
_THRESH_CCRR_K_FINAL_HIGH = 10000.0
_THRESH_TAIL_STRATUM_ENRICHMENT = 2.0
_THRESH_TAIL_STRATUM_MIN_SUPPORT_ABS = 3
_THRESH_TAIL_STRATUM_MIN_SUPPORT_FRAC = 0.05
_THRESH_TAIL_SLIDE_ENRICHMENT = 3.0
_THRESH_TAIL_SLIDE_MIN_SUPPORT_ABS = 3
_THRESH_TAIL_SLIDE_MIN_SUPPORT_FRAC = 0.10
_GLOBAL_MODE = "global"
_PER_SAMPLE_MATCH_COLS = [
    "dataset",
    "model",
    "mode",
    "k",
    "tau",
    "ccrr_alpha",
    "ccrr_search",
    "excluded_centers",
]
_CCRR_CONTEXT_COLS = ["dataset", "mode", "k", "tau", "ccrr_alpha", "ccrr_m", "ccrr_search", "excluded_centers"]
_CCRR_REQUIRED_METRIC_COLS = {"model", "mode", "ccrr_m", "ccrr_alpha"}


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
        "--ccrr-m-sweep-csv",
        type=Path,
        default=None,
        help="Optional CCRR m-sweep CSV path (default: auto-detect next to metrics CSV).",
    )
    parser.add_argument(
        "--per-sample-csv",
        type=Path,
        default=None,
        help="Optional per-sample metrics CSV path (default: auto-detect next to metrics CSV).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
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


def _aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        raise ValueError("metrics CSV must include a 'model' column")
    numeric_cols = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("metrics CSV has no numeric columns to analyze")
    grouped = df.groupby("model", as_index=False)[numeric_cols].mean(numeric_only=True)
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


def _match_per_sample_rows(df_per_sample: pd.DataFrame, metric_row: pd.Series) -> pd.DataFrame:
    matched = df_per_sample.copy()
    for col in _PER_SAMPLE_MATCH_COLS:
        if col not in matched.columns or col not in metric_row.index:
            continue
        row_val = metric_row[col]
        if pd.isna(row_val):
            matched = matched[matched[col].isna()]
            continue
        if pd.api.types.is_numeric_dtype(matched[col]):
            matched = matched[np.isclose(matched[col].astype(float), float(row_val), equal_nan=False)]
        else:
            matched = matched[matched[col].astype(str) == str(row_val)]
    return matched.reset_index(drop=True)


def _effective_tail_strata_from_counts(tail_counts: pd.Series) -> float:
    if len(tail_counts) == 0:
        return 0.0
    probs = tail_counts.to_numpy(dtype=float)
    total = float(probs.sum())
    if total <= 0.0:
        return 0.0
    probs = probs / total
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def _tail_support_threshold(tail_size: int) -> int:
    return int(max(_THRESH_TAIL_STRATUM_MIN_SUPPORT_ABS, np.ceil(_THRESH_TAIL_STRATUM_MIN_SUPPORT_FRAC * float(tail_size))))


def _slide_support_threshold(tail_size: int) -> int:
    return int(max(_THRESH_TAIL_SLIDE_MIN_SUPPORT_ABS, np.ceil(_THRESH_TAIL_SLIDE_MIN_SUPPORT_FRAC * float(tail_size))))


def _is_global_metric_row(metric_row: pd.Series) -> bool:
    return str(metric_row.get("mode", "")).strip().lower() == _GLOBAL_MODE


def _metric_context_base(metric_row: pd.Series, *, ccrr_m: int, ccrr_col: str, alpha: float) -> dict:
    return {
        "dataset": str(metric_row.get("dataset", "")),
        "model": str(metric_row["model"]),
        "mode": str(metric_row["mode"]),
        "k": int(metric_row["k"]) if "k" in metric_row.index and pd.notna(metric_row["k"]) else np.nan,
        "tau": float(metric_row["tau"]) if "tau" in metric_row.index and pd.notna(metric_row["tau"]) else np.nan,
        "ccrr_alpha": alpha,
        "ccrr_m": ccrr_m,
        "ccrr_column": ccrr_col,
        "ccrr_search": str(metric_row.get("ccrr_search", "")),
        "excluded_centers": str(metric_row.get("excluded_centers", "")),
    }


def _resolve_ccrr_tail_context(metric_row: pd.Series, df_per_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    matched = _match_per_sample_rows(df_per_sample, metric_row)
    if len(matched) == 0:
        raise ValueError(
            "per-sample metrics CSV does not contain rows matching a global metrics row for "
            f"model={metric_row.get('model')}"
        )

    ccrr_m = int(metric_row["ccrr_m"])
    ccrr_col = f"ccrr_m{ccrr_m}"
    if ccrr_col not in matched.columns:
        raise ValueError(f"per-sample metrics CSV missing required CCRR column: {ccrr_col}")

    alpha = float(metric_row["ccrr_alpha"])
    tail_df = select_exact_size_tail_set(
        matched,
        value_column=ccrr_col,
        alpha=alpha,
        sample_id_column="sample_id",
    )
    base = _metric_context_base(metric_row, ccrr_m=ccrr_m, ccrr_col=ccrr_col, alpha=alpha)
    return matched, tail_df, base, ccrr_col


def _merge_tail_counts(
    matched: pd.DataFrame,
    tail_df: pd.DataFrame,
    *,
    group_cols: list[str],
    ccrr_col: str,
) -> tuple[pd.DataFrame, int]:
    dataset_counts = matched.groupby(group_cols, as_index=False).size().rename(columns={"size": "dataset_count"})
    tail_counts = (
        tail_df.groupby(group_cols, as_index=False)
        .agg(
            tail_count=("sample_id", "size"),
            tail_mean_ccrr=(ccrr_col, "mean"),
            tail_median_ccrr=(ccrr_col, "median"),
        )
    )
    joined = dataset_counts.merge(tail_counts, on=group_cols, how="left")
    joined["tail_count"] = joined["tail_count"].fillna(0).astype(int)
    joined["tail_mean_ccrr"] = joined["tail_mean_ccrr"].astype(float)
    joined["tail_median_ccrr"] = joined["tail_median_ccrr"].astype(float)
    tail_n = int(len(tail_df))
    joined["dataset_frac"] = joined["dataset_count"].astype(float) / float(len(matched))
    joined["tail_frac"] = joined["tail_count"].astype(float) / float(tail_n) if tail_n > 0 else 0.0
    joined["enrichment_ratio"] = joined["tail_frac"] / joined["dataset_frac"]
    return joined, tail_n


def _context_mask(df: pd.DataFrame, context_row: pd.Series, *, context_cols: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index, dtype=bool)
    for col in context_cols:
        value = context_row[col]
        if pd.isna(value):
            mask &= df[col].isna()
        else:
            mask &= df[col] == value
    return mask


def _tail_overlap_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def _ccrr_stratum_enrichment(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df_per_sample is None or len(df_per_sample) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_sample_cols = {"sample_id", "label", "medical_center"}
    if not _CCRR_REQUIRED_METRIC_COLS.issubset(df_metrics_raw.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not required_sample_cols.issubset(df_per_sample.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    enrichment_rows: list[dict] = []
    concentration_rows: list[dict] = []

    for _, metric_row in df_metrics_raw.iterrows():
        if not _is_global_metric_row(metric_row):
            continue

        matched, tail_df, base, ccrr_col = _resolve_ccrr_tail_context(metric_row, df_per_sample)
        joined, tail_n = _merge_tail_counts(
            matched,
            tail_df,
            group_cols=["label", "medical_center"],
            ccrr_col=ccrr_col,
        )
        tail_support_threshold = _tail_support_threshold(tail_n) if tail_n > 0 else _THRESH_TAIL_STRATUM_MIN_SUPPORT_ABS
        joined["flagged"] = (
            (joined["enrichment_ratio"] >= _THRESH_TAIL_STRATUM_ENRICHMENT)
            & (joined["tail_count"] >= tail_support_threshold)
        )
        for _, row in joined.iterrows():
            enrichment_rows.append(
                {
                    **base,
                    "label": str(row["label"]),
                    "medical_center": str(row["medical_center"]),
                    "dataset_count": int(row["dataset_count"]),
                    "dataset_frac": float(row["dataset_frac"]),
                    "tail_count": int(row["tail_count"]),
                    "tail_frac": float(row["tail_frac"]),
                    "enrichment_ratio": float(row["enrichment_ratio"]),
                    "tail_support_threshold": int(tail_support_threshold),
                    "tail_mean_ccrr": float(row["tail_mean_ccrr"]) if pd.notna(row["tail_mean_ccrr"]) else float("nan"),
                    "tail_median_ccrr": float(row["tail_median_ccrr"]) if pd.notna(row["tail_median_ccrr"]) else float("nan"),
                    "flagged": bool(row["flagged"]),
                }
            )

        tail_count_series = tail_df.groupby(["label", "medical_center"]).size()
        concentration_rows.append(
            {
                **base,
                "tail_sample_count": int(tail_n),
                "tail_support_threshold": int(tail_support_threshold),
                "tail_strata_count": int(len(tail_count_series)),
                "effective_tail_strata": _effective_tail_strata_from_counts(tail_count_series),
                "flagged_strata_count": int(joined["flagged"].sum()),
            }
        )

    enrichment_df = pd.DataFrame(enrichment_rows)
    concentration_df = pd.DataFrame(concentration_rows)
    if len(enrichment_df) == 0:
        return enrichment_df, concentration_df, pd.DataFrame()
    heatmap_df = enrichment_df.loc[:, ["dataset", "model", "mode", "label", "medical_center", "enrichment_ratio", "tail_count", "flagged"]].copy()
    return enrichment_df, concentration_df, heatmap_df


def _ccrr_slide_diagnostics(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_per_sample is None or len(df_per_sample) == 0:
        return pd.DataFrame(), pd.DataFrame()

    required_sample_cols = {"sample_id", "slide_id"}
    if not _CCRR_REQUIRED_METRIC_COLS.issubset(df_metrics_raw.columns):
        return pd.DataFrame(), pd.DataFrame()
    if not required_sample_cols.issubset(df_per_sample.columns):
        return pd.DataFrame(), pd.DataFrame()

    enrichment_rows: list[dict] = []
    summary_rows: list[dict] = []

    for _, metric_row in df_metrics_raw.iterrows():
        if not _is_global_metric_row(metric_row):
            continue

        matched, tail_df, base, ccrr_col = _resolve_ccrr_tail_context(metric_row, df_per_sample)
        joined, tail_n = _merge_tail_counts(
            matched,
            tail_df,
            group_cols=["slide_id"],
            ccrr_col=ccrr_col,
        )
        tail_support_threshold = _slide_support_threshold(tail_n) if tail_n > 0 else _THRESH_TAIL_SLIDE_MIN_SUPPORT_ABS
        joined["flagged"] = (
            (joined["enrichment_ratio"] >= _THRESH_TAIL_SLIDE_ENRICHMENT)
            & (joined["tail_count"] >= tail_support_threshold)
        )
        for _, row in joined.iterrows():
            enrichment_rows.append(
                {
                    **base,
                    "slide_id": str(row["slide_id"]),
                    "dataset_count": int(row["dataset_count"]),
                    "dataset_frac": float(row["dataset_frac"]),
                    "tail_count": int(row["tail_count"]),
                    "tail_frac": float(row["tail_frac"]),
                    "enrichment_ratio": float(row["enrichment_ratio"]),
                    "tail_support_threshold": int(tail_support_threshold),
                    "tail_mean_ccrr": float(row["tail_mean_ccrr"]) if pd.notna(row["tail_mean_ccrr"]) else float("nan"),
                    "tail_median_ccrr": float(row["tail_median_ccrr"]) if pd.notna(row["tail_median_ccrr"]) else float("nan"),
                    "flagged": bool(row["flagged"]),
                }
            )

        flagged_tail_count = int(joined.loc[joined["flagged"], "tail_count"].sum())
        summary_rows.append(
            {
                **base,
                "tail_sample_count": int(tail_n),
                "tail_support_threshold": int(tail_support_threshold),
                "flagged_slides_count": int(joined["flagged"].sum()),
                "flagged_tail_mass_frac": (float(flagged_tail_count) / float(tail_n)) if tail_n > 0 else 0.0,
            }
        )

    return (
        pd.DataFrame(enrichment_rows),
        pd.DataFrame(summary_rows),
    )


def _ccrr_tail_overlap(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df_per_sample is None or len(df_per_sample) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    required_sample_cols = {"sample_id", "slide_id", "label", "medical_center"}
    if not _CCRR_REQUIRED_METRIC_COLS.issubset(df_metrics_raw.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not required_sample_cols.issubset(df_per_sample.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    entries_by_context: dict[tuple[object, ...], list[dict]] = {}
    for _, metric_row in df_metrics_raw.iterrows():
        if not _is_global_metric_row(metric_row):
            continue
        matched, tail_df, base, _ = _resolve_ccrr_tail_context(metric_row, df_per_sample)
        overlap_base = {key: value for key, value in base.items() if key != "ccrr_column"}
        context_key = tuple(overlap_base[col] for col in _CCRR_CONTEXT_COLS)
        entries_by_context.setdefault(context_key, []).append(
            {
                **overlap_base,
                "universe_ids": tuple(sorted(matched["sample_id"].astype(str).tolist())),
                "tail_ids": set(tail_df["sample_id"].astype(str).tolist()),
                "tail_df": tail_df.copy(),
            }
        )

    if not entries_by_context:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    jaccard_rows: list[dict] = []
    unique_rows: list[dict] = []
    summary_rows: list[dict] = []
    always_rows: list[dict] = []
    for entries in entries_by_context.values():
        if len(entries) < 2:
            continue
        context_values = {col: entries[0][col] for col in _CCRR_CONTEXT_COLS}

        universe_ids = entries[0]["universe_ids"]
        for entry in entries[1:]:
            if entry["universe_ids"] != universe_ids:
                raise ValueError("CCRR tail overlap requires models evaluated on the same sample universe")

        model_names = sorted(str(entry["model"]) for entry in entries)
        tail_sets = {str(entry["model"]): set(entry["tail_ids"]) for entry in entries}
        tail_frames = {str(entry["model"]): entry["tail_df"] for entry in entries}

        for model_a in model_names:
            for model_b in model_names:
                set_a = tail_sets[model_a]
                set_b = tail_sets[model_b]
                union = set_a | set_b
                jaccard_rows.append(
                    {
                        **context_values,
                        "model_a": model_a,
                        "model_b": model_b,
                        "tail_size_a": int(len(set_a)),
                        "tail_size_b": int(len(set_b)),
                        "intersection_size": int(len(set_a & set_b)),
                        "union_size": int(len(union)),
                        "jaccard": _tail_overlap_jaccard(set_a, set_b),
                    }
                )

        always_fragile_ids = set.intersection(*(tail_sets[model] for model in model_names)) if model_names else set()
        for model in model_names:
            other_union = set().union(*(tail_sets[m] for m in model_names if m != model))
            unique_ids = tail_sets[model] - other_union
            model_tail = tail_frames[model]
            unique_df = model_tail[model_tail["sample_id"].astype(str).isin(unique_ids)].copy()
            for _, row in unique_df.iterrows():
                unique_rows.append(
                    {
                        **context_values,
                        "model": model,
                        "sample_id": str(row["sample_id"]),
                        "slide_id": str(row["slide_id"]),
                        "label": str(row["label"]),
                        "medical_center": str(row["medical_center"]),
                    }
                )
            off_diag = [
                _tail_overlap_jaccard(tail_sets[model], tail_sets[other])
                for other in model_names
                if other != model
            ]
            summary_rows.append(
                {
                    "model": model,
                    **context_values,
                    "tail_size": int(len(tail_sets[model])),
                    "always_fragile_count": int(len(always_fragile_ids)),
                    "unique_fragile_count": int(len(unique_ids)),
                    "mean_pairwise_jaccard": float(np.mean(off_diag)) if off_diag else float("nan"),
                    "max_pairwise_jaccard": float(np.max(off_diag)) if off_diag else float("nan"),
                    "min_pairwise_jaccard": float(np.min(off_diag)) if off_diag else float("nan"),
                }
            )

        if always_fragile_ids:
            ref_tail = tail_frames[model_names[0]]
            always_df = ref_tail[ref_tail["sample_id"].astype(str).isin(always_fragile_ids)].copy()
            for _, row in always_df.iterrows():
                always_rows.append(
                    {
                        **context_values,
                        "sample_id": str(row["sample_id"]),
                        "slide_id": str(row["slide_id"]),
                        "label": str(row["label"]),
                        "medical_center": str(row["medical_center"]),
                    }
                )

    jaccard_df = pd.DataFrame(
        jaccard_rows,
        columns=_CCRR_CONTEXT_COLS + ["model_a", "model_b", "tail_size_a", "tail_size_b", "intersection_size", "union_size", "jaccard"],
    ).sort_values(_CCRR_CONTEXT_COLS + ["model_a", "model_b"]).reset_index(drop=True)
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "dataset",
            "model",
            "mode",
            "k",
            "tau",
            "ccrr_alpha",
            "ccrr_m",
            "ccrr_search",
            "excluded_centers",
            "tail_size",
            "always_fragile_count",
            "unique_fragile_count",
            "mean_pairwise_jaccard",
            "max_pairwise_jaccard",
            "min_pairwise_jaccard",
        ],
    ).sort_values(_CCRR_CONTEXT_COLS + ["model"]).reset_index(drop=True)
    always_df = pd.DataFrame(
        always_rows,
        columns=_CCRR_CONTEXT_COLS + ["sample_id", "slide_id", "label", "medical_center"],
    ).sort_values(_CCRR_CONTEXT_COLS + ["sample_id"]).reset_index(drop=True)
    unique_df = pd.DataFrame(
        unique_rows,
        columns=_CCRR_CONTEXT_COLS + ["model", "sample_id", "slide_id", "label", "medical_center"],
    ).sort_values(_CCRR_CONTEXT_COLS + ["model", "sample_id"]).reset_index(drop=True)

    return (jaccard_df, summary_df, always_df, unique_df)


def _k_sweep_sensitivity(df_k: pd.DataFrame | None) -> pd.DataFrame:
    if df_k is None:
        return pd.DataFrame()
    required = {"model", "k", "ri", "mari"}
    if not required.issubset(df_k.columns):
        return pd.DataFrame()

    grouped = (
        df_k.groupby("model", as_index=False)
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


def _ccrr_m_sweep_sensitivity(df_m: pd.DataFrame | None) -> pd.DataFrame:
    if df_m is None:
        return pd.DataFrame()
    required = {"model", "m", "ccrr", "ccrr_q_alpha", "ccrr_ltm_alpha"}
    if not required.issubset(df_m.columns):
        return pd.DataFrame()

    grouped_rows: list[dict] = []
    for model, grp in df_m.groupby("model"):
        grp_sorted = grp.sort_values("m", ascending=True)
        row: dict = {
            "model": str(model),
            "m_min": float(grp_sorted["m"].iloc[0]),
            "m_max": float(grp_sorted["m"].iloc[-1]),
            "ccrr_m_min": float(grp_sorted["ccrr"].iloc[0]),
            "ccrr_m_max": float(grp_sorted["ccrr"].iloc[-1]),
            "ccrr_gain": float(grp_sorted["ccrr"].iloc[-1] - grp_sorted["ccrr"].iloc[0]),
            "q_gain": float(grp_sorted["ccrr_q_alpha"].iloc[-1] - grp_sorted["ccrr_q_alpha"].iloc[0]),
            "ltm_gain": float(grp_sorted["ccrr_ltm_alpha"].iloc[-1] - grp_sorted["ccrr_ltm_alpha"].iloc[0]),
        }
        if "ccrr_retries" in grp_sorted.columns:
            row["ccrr_retries_max"] = float(grp_sorted["ccrr_retries"].max())
        if "ccrr_k_final" in grp_sorted.columns:
            row["ccrr_k_final_max"] = float(grp_sorted["ccrr_k_final"].max())
        grouped_rows.append(row)

    out = pd.DataFrame(grouped_rows)
    if len(out) == 0:
        return out
    sort_col = "ccrr_gain"
    out = out.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return out


def _model_action_flags(
    *,
    df_model: pd.DataFrame,
    delta_df: pd.DataFrame,
    k_sensitivity_df: pd.DataFrame,
    ccrr_m_sensitivity_df: pd.DataFrame,
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

    # Coverage risks.
    for metric in ("ri", "mari", "ccrr"):
        col = f"{metric}_undefined_frac"
        if col not in df_model.columns:
            continue
        for _, row in df_model[df_model[col] >= _THRESH_UNDEFINED_HIGH].iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": f"coverage_risk_{metric}",
                    "severity": "high",
                    "value": float(row[col]),
                    "threshold": _THRESH_UNDEFINED_HIGH,
                    "detail": f"{_DISPLAY_NAMES.get(metric, metric)} undefined fraction is high ({row[col]:.3f}).",
                }
            )

    # Undefined breakdown flags (SS/OO).
    for metric in ("ri", "mari"):
        undef_col = f"{metric}_undefined_frac"
        ss_col = f"{metric}_ss_dominated_undefined_frac"
        oo_col = f"{metric}_oo_dominated_undefined_frac"

        if undef_col in df_model.columns and float(df_model[undef_col].max()) > 0:
            # high_undefined: > 30% undefined
            for _, row in df_model[df_model[undef_col] >= _THRESH_UNDEFINED_MODERATE].iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": f"high_undefined_{metric}",
                        "severity": "high",
                        "value": float(row[undef_col]),
                        "threshold": _THRESH_UNDEFINED_MODERATE,
                        "detail": (
                            f"{_DISPLAY_NAMES.get(metric, metric)} computed from a minority of samples "
                            f"(undefined={row[undef_col]:.3f}); score may not be representative."
                        ),
                    }
                )

        if ss_col in df_model.columns:
            # entangled_clusters: SS-dominated > 20%
            for _, row in df_model[df_model[ss_col] >= _THRESH_SS_DOMINATED_HIGH].iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": f"entangled_clusters_{metric}",
                        "severity": "medium",
                        "value": float(row[ss_col]),
                        "threshold": _THRESH_SS_DOMINATED_HIGH,
                        "detail": (
                            f"Many {_DISPLAY_NAMES.get(metric, metric)} neighborhoods are "
                            f"(class x center)-entangled (SS-dominated={row[ss_col]:.3f})."
                        ),
                    }
                )

        if oo_col in df_model.columns:
            # poor_embedding: OO-dominated > 10%
            for _, row in df_model[df_model[oo_col] >= _THRESH_OO_DOMINATED_HIGH].iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": f"poor_embedding_{metric}",
                        "severity": "high",
                        "value": float(row[oo_col]),
                        "threshold": _THRESH_OO_DOMINATED_HIGH,
                        "detail": (
                            f"Significant fraction of {_DISPLAY_NAMES.get(metric, metric)} samples "
                            f"are poorly embedded (OO-dominated={row[oo_col]:.3f})."
                        ),
                    }
                )

        # ss_dominated_undefined: SS/undefined ratio > 80%
        if ss_col in df_model.columns and undef_col in df_model.columns:
            for _, row in df_model.iterrows():
                undef_val = float(row[undef_col])
                ss_val = float(row[ss_col])
                if undef_val > 0 and (ss_val / undef_val) >= _THRESH_SS_RATIO_OF_UNDEFINED:
                    rows.append(
                        {
                            "model": str(row["model"]),
                            "flag": f"ss_dominated_undefined_{metric}",
                            "severity": "medium",
                            "value": float(ss_val / undef_val),
                            "threshold": _THRESH_SS_RATIO_OF_UNDEFINED,
                            "detail": (
                                f"Undefined {_DISPLAY_NAMES.get(metric, metric)} samples are overwhelmingly "
                                f"SS-dominated ({ss_val/undef_val:.1%}); fragility is due to cluster "
                                f"entanglement, not poor embedding."
                            ),
                        }
                    )

    # Coverage mismatch: two models with similar RI but very different undefined_frac.
    if "ri" in df_model.columns and "ri_undefined_frac" in df_model.columns and len(df_model) >= 2:
        for i, row_a in df_model.iterrows():
            for j, row_b in df_model.iterrows():
                if i >= j:
                    continue
                ri_diff = abs(float(row_a["ri"]) - float(row_b["ri"]))
                undef_diff = abs(float(row_a["ri_undefined_frac"]) - float(row_b["ri_undefined_frac"]))
                if ri_diff < 0.05 and undef_diff >= 0.15:
                    rows.append(
                        {
                            "model": f"{row_a['model']} vs {row_b['model']}",
                            "flag": "coverage_mismatch",
                            "severity": "high",
                            "value": float(undef_diff),
                            "threshold": 0.15,
                            "detail": (
                                f"Models have similar RI ({row_a['ri']:.3f} vs {row_b['ri']:.3f}) "
                                f"but different undefined fractions ({row_a['ri_undefined_frac']:.3f} vs "
                                f"{row_b['ri_undefined_frac']:.3f}); cross-model RI comparison is unreliable."
                            ),
                        }
                    )

    # Tail gap risks for CCRR summaries.
    if {"ccrr", "ccrr_q_alpha"}.issubset(df_model.columns):
        df_tmp = df_model.loc[:, ["model", "ccrr", "ccrr_q_alpha"]].copy()
        df_tmp["tail_gap_q"] = df_tmp["ccrr"] - df_tmp["ccrr_q_alpha"]
        for _, row in df_tmp[df_tmp["tail_gap_q"] >= _THRESH_TAIL_GAP_Q].iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": "tail_gap_q_high",
                    "severity": "medium",
                    "value": float(row["tail_gap_q"]),
                    "threshold": _THRESH_TAIL_GAP_Q,
                    "detail": f"CCRR - Q(CCRR) tail gap is large ({row['tail_gap_q']:.3f}).",
                }
            )
    if {"ccrr", "ccrr_ltm_alpha"}.issubset(df_model.columns):
        df_tmp = df_model.loc[:, ["model", "ccrr", "ccrr_ltm_alpha"]].copy()
        df_tmp["tail_gap_ltm"] = df_tmp["ccrr"] - df_tmp["ccrr_ltm_alpha"]
        for _, row in df_tmp[df_tmp["tail_gap_ltm"] >= _THRESH_TAIL_GAP_LTM].iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": "tail_gap_ltm_high",
                    "severity": "medium",
                    "value": float(row["tail_gap_ltm"]),
                    "threshold": _THRESH_TAIL_GAP_LTM,
                    "detail": f"CCRR - LTM(CCRR) tail gap is large ({row['tail_gap_ltm']:.3f}).",
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

    # m-sweep CCRR gain and compute cost.
    if len(ccrr_m_sensitivity_df) > 0:
        if "ccrr_gain" in ccrr_m_sensitivity_df.columns:
            for _, row in ccrr_m_sensitivity_df[ccrr_m_sensitivity_df["ccrr_gain"] >= _THRESH_M_SWEEP_CCRR_GAIN].iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": "ccrr_m_sweep_gain_high",
                        "severity": "medium",
                        "value": float(row["ccrr_gain"]),
                        "threshold": _THRESH_M_SWEEP_CCRR_GAIN,
                        "detail": f"CCRR gain across m-sweep is high ({row['ccrr_gain']:.3f}).",
                    }
                )
        if "ccrr_retries_max" in ccrr_m_sensitivity_df.columns:
            for _, row in ccrr_m_sensitivity_df[ccrr_m_sensitivity_df["ccrr_retries_max"] >= _THRESH_CCRR_RETRIES_HIGH].iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": "ccrr_search_cost_high",
                        "severity": "high",
                        "value": float(row["ccrr_retries_max"]),
                        "threshold": _THRESH_CCRR_RETRIES_HIGH,
                        "detail": f"CCRR search retries are high (max retries={row['ccrr_retries_max']:.0f}).",
                    }
                )
        if "ccrr_k_final_max" in ccrr_m_sensitivity_df.columns:
            for _, row in ccrr_m_sensitivity_df[ccrr_m_sensitivity_df["ccrr_k_final_max"] >= _THRESH_CCRR_K_FINAL_HIGH].iterrows():
                rows.append(
                    {
                        "model": str(row["model"]),
                        "flag": "ccrr_search_cost_high",
                        "severity": "high",
                        "value": float(row["ccrr_k_final_max"]),
                        "threshold": _THRESH_CCRR_K_FINAL_HIGH,
                        "detail": f"CCRR search k_final is high (max k_final={row['ccrr_k_final_max']:.0f}).",
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
    ccrr_m_sensitivity_df: pd.DataFrame,
    ccrr_stratum_enrichment_df: pd.DataFrame,
    ccrr_stratum_concentration_df: pd.DataFrame,
    ccrr_tail_overlap_jaccard_df: pd.DataFrame,
    ccrr_tail_overlap_summary_df: pd.DataFrame,
    ccrr_slide_enrichment_df: pd.DataFrame,
    ccrr_slide_summary_df: pd.DataFrame,
) -> None:
    strong_corr = _strongest_corr_pairs(pearson, top_n=8)

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
    lines.append("## Pearson Correlations (RI / MaRI / CCRR)")
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
    lines.append("## Additional Insights and Action Flags")
    lines.append("")
    if len(action_flags_df) == 0:
        lines.append("- No action flags triggered by current thresholds.")
    else:
        lines.append(f"- Models with >=1 action flag: {action_flags_df['model'].nunique()}")
        lines.append(f"- Total unique model flags: {len(action_flags_df)}")
        lines.append("")
        lines.append("### Triggered Flags (Top 20)")
        for _, row in action_flags_df.head(20).iterrows():
            lines.append(
                f"- {row['model']}: `{row['flag']}` (value={float(row['value']):.3f}, "
                f"threshold={float(row['threshold']):.3f})"
            )
    lines.append("")
    lines.append("## CCRR Tail Stratum Enrichment")
    lines.append("")
    if len(ccrr_stratum_enrichment_df) == 0:
        lines.append("- Per-sample CCRR artifact unavailable or no global-mode rows to analyze.")
    else:
        lines.append(
            f"- Flagging rule: `enrichment_ratio >= {_THRESH_TAIL_STRATUM_ENRICHMENT:.1f}` "
            f"and `tail_count >= max({_THRESH_TAIL_STRATUM_MIN_SUPPORT_ABS}, "
            f"ceil({_THRESH_TAIL_STRATUM_MIN_SUPPORT_FRAC:.2f} * tail_size))`."
        )
        for _, row in ccrr_stratum_concentration_df.head(5).iterrows():
            lines.append(
                f"- {row['model']}: effective_tail_strata={float(row['effective_tail_strata']):.3f}, "
                f"tail_sample_count={int(row['tail_sample_count'])}, "
                f"flagged_strata_count={int(row['flagged_strata_count'])}"
            )
        flagged = ccrr_stratum_enrichment_df[ccrr_stratum_enrichment_df["flagged"]]
        if len(flagged) == 0:
            lines.append("- No strata met the enrichment and minimum-support thresholds.")
        else:
            lines.append("")
            lines.append("### Flagged Strata")
            for _, row in flagged.head(10).iterrows():
                lines.append(
                    f"- {row['model']} {row['label']} / {row['medical_center']}: "
                    f"tail_count={int(row['tail_count'])}, "
                    f"enrichment_ratio={float(row['enrichment_ratio']):.3f}, "
                    f"tail_mean_ccrr={float(row['tail_mean_ccrr']):.3f}"
                )
    lines.append("")
    lines.append("## CCRR Tail Overlap")
    lines.append("")
    if len(ccrr_tail_overlap_jaccard_df) == 0:
        lines.append("- Overlap analysis unavailable (missing per-sample artifact, single model, or no shared global universe).")
    else:
        if len(ccrr_tail_overlap_summary_df) > 0:
            grouped_summary = ccrr_tail_overlap_summary_df.groupby(
                _CCRR_CONTEXT_COLS,
                dropna=False,
                as_index=False,
            )
            for _, context_row in grouped_summary.first().iterrows():
                context_mask = _context_mask(
                    ccrr_tail_overlap_summary_df,
                    context_row,
                    context_cols=_CCRR_CONTEXT_COLS,
                )
                context_summary = ccrr_tail_overlap_summary_df.loc[context_mask].copy()
                always_fragile_count = int(context_summary["always_fragile_count"].iloc[0])
                lines.append(
                    f"- Context dataset={context_row['dataset']}, m={int(context_row['ccrr_m'])}, "
                    f"alpha={float(context_row['ccrr_alpha']):.3f}: always_fragile_count={always_fragile_count}"
                )
                for _, row in context_summary.iterrows():
                    lines.append(
                        f"  - {row['model']}: unique_fragile_count={int(row['unique_fragile_count'])}, "
                        f"mean_pairwise_jaccard={float(row['mean_pairwise_jaccard']):.3f}"
                    )
        lines.append("")
        lines.append("### Strongest Pairwise Overlap")
        pair_rows = ccrr_tail_overlap_jaccard_df.loc[
            ccrr_tail_overlap_jaccard_df["model_a"] < ccrr_tail_overlap_jaccard_df["model_b"]
        ].copy()
        pair_rows = pair_rows.sort_values("jaccard", ascending=False).head(5)
        for _, row in pair_rows.iterrows():
            lines.append(
                f"- dataset={row['dataset']}, m={int(row['ccrr_m'])}, alpha={float(row['ccrr_alpha']):.3f}, "
                f"{row['model_a']} vs {row['model_b']}: jaccard={float(row['jaccard']):.3f}"
            )
    lines.append("")
    lines.append("## CCRR Slide Diagnostics")
    lines.append("")
    if len(ccrr_slide_enrichment_df) == 0:
        lines.append("- Slide-level diagnostics unavailable (missing per-sample CCRR artifact or no global-mode rows).")
    else:
        lines.append(
            f"- Diagnostic flagging rule: `enrichment_ratio >= {_THRESH_TAIL_SLIDE_ENRICHMENT:.1f}` "
            f"and `tail_count >= max({_THRESH_TAIL_SLIDE_MIN_SUPPORT_ABS}, "
            f"ceil({_THRESH_TAIL_SLIDE_MIN_SUPPORT_FRAC:.2f} * tail_size))`."
        )
        lines.append("- This output is supplementary diagnostic material, not a headline benchmark result.")
        for _, row in ccrr_slide_summary_df.head(5).iterrows():
            lines.append(
                f"- {row['model']}: flagged_slides_count={int(row['flagged_slides_count'])}, "
                f"flagged_tail_mass_frac={float(row['flagged_tail_mass_frac']):.3f}"
            )
        flagged = ccrr_slide_enrichment_df[ccrr_slide_enrichment_df["flagged"]]
        if len(flagged) == 0:
            lines.append("- No slides met the diagnostic thresholds.")
        else:
            lines.append("")
            lines.append("### Flagged Slides")
            for _, row in flagged.head(10).iterrows():
                lines.append(
                    f"- {row['model']} {row['slide_id']}: "
                    f"tail_count={int(row['tail_count'])}, "
                    f"enrichment_ratio={float(row['enrichment_ratio']):.3f}, "
                    f"tail_mean_ccrr={float(row['tail_mean_ccrr']):.3f}"
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
    lines.append("## CCRR m-Sweep Sensitivity and Cost")
    lines.append("")
    if len(ccrr_m_sensitivity_df) == 0:
        lines.append("- CCRR m-sweep metrics unavailable.")
    else:
        lines.append(f"- Sensitivity threshold: `ccrr_gain >= {_THRESH_M_SWEEP_CCRR_GAIN:.2f}`")
        if "ccrr_retries_max" in ccrr_m_sensitivity_df.columns:
            lines.append(f"- Cost threshold: `ccrr_retries_max >= {int(_THRESH_CCRR_RETRIES_HIGH)}`")
        if "ccrr_k_final_max" in ccrr_m_sensitivity_df.columns:
            lines.append(f"- Cost threshold: `ccrr_k_final_max >= {int(_THRESH_CCRR_K_FINAL_HIGH)}`")
        for _, row in ccrr_m_sensitivity_df.head(5).iterrows():
            extras: list[str] = []
            if "ccrr_retries_max" in row.index and np.isfinite(row["ccrr_retries_max"]):
                extras.append(f"retries_max={float(row['ccrr_retries_max']):.0f}")
            if "ccrr_k_final_max" in row.index and np.isfinite(row["ccrr_k_final_max"]):
                extras.append(f"k_final_max={float(row['ccrr_k_final_max']):.0f}")
            suffix = f", {', '.join(extras)}" if extras else ""
            lines.append(
                f"- {row['model']}: ccrr_gain={float(row['ccrr_gain']):.3f}, "
                f"q_gain={float(row['q_gain']):.3f}, ltm_gain={float(row['ltm_gain']):.3f}{suffix}"
            )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    mat = ax.imshow(df.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(title)
    fig.colorbar(mat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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
    ccrr_m_sweep_path = (
        Path(args.ccrr_m_sweep_csv) if args.ccrr_m_sweep_csv is not None else metrics_csv.parent / "ccrr_m_sweep_metrics.csv"
    )
    per_sample_path = Path(args.per_sample_csv) if args.per_sample_csv is not None else metrics_csv.parent / "per_sample_metrics.csv"
    df_k_sweep = _load_optional_csv(k_sweep_path)
    df_ccrr_m_sweep = _load_optional_csv(ccrr_m_sweep_path)
    df_per_sample = _load_optional_csv(per_sample_path)
    k_sensitivity_df = _k_sweep_sensitivity(df_k_sweep)
    ccrr_m_sensitivity_df = _ccrr_m_sweep_sensitivity(df_ccrr_m_sweep)
    ccrr_stratum_enrichment_df, ccrr_stratum_concentration_df, ccrr_stratum_heatmap_df = _ccrr_stratum_enrichment(
        df_raw,
        df_per_sample,
    )
    (
        ccrr_tail_overlap_jaccard_df,
        ccrr_tail_overlap_summary_df,
        ccrr_tail_overlap_always_df,
        ccrr_tail_overlap_unique_df,
    ) = _ccrr_tail_overlap(df_raw, df_per_sample)
    ccrr_slide_enrichment_df, ccrr_slide_summary_df = _ccrr_slide_diagnostics(df_raw, df_per_sample)
    action_flags_df = _model_action_flags(
        df_model=df_model,
        delta_df=delta_df,
        k_sensitivity_df=k_sensitivity_df,
        ccrr_m_sensitivity_df=ccrr_m_sensitivity_df,
    )

    pearson_corr.to_csv(out_dir / "correlation_pearson.csv")
    spearman_corr.to_csv(out_dir / "correlation_spearman.csv")
    rank_df.to_csv(out_dir / "model_ranks.csv", index=False)
    top_df.to_csv(out_dir / "top_models_by_metric.csv", index=False)
    delta_df.to_csv(out_dir / "rank_deltas.csv", index=False)
    agreement_df.to_csv(out_dir / "rank_agreement.csv", index=False)
    action_flags_df.to_csv(out_dir / "model_action_flags.csv", index=False)
    if len(k_sensitivity_df) > 0:
        k_sensitivity_df.to_csv(out_dir / "k_sweep_sensitivity.csv", index=False)
    if len(ccrr_m_sensitivity_df) > 0:
        ccrr_m_sensitivity_df.to_csv(out_dir / "ccrr_m_sweep_sensitivity.csv", index=False)
    if len(ccrr_stratum_enrichment_df) > 0:
        ccrr_stratum_enrichment_df.to_csv(out_dir / "ccrr_stratum_enrichment.csv", index=False)
        ccrr_stratum_concentration_df.to_csv(out_dir / "ccrr_stratum_concentration.csv", index=False)
        ccrr_stratum_heatmap_df.to_csv(out_dir / "ccrr_stratum_enrichment_heatmap.csv", index=False)
    if len(ccrr_tail_overlap_jaccard_df) > 0:
        ccrr_tail_overlap_jaccard_df.to_csv(out_dir / "ccrr_tail_overlap_jaccard.csv", index=False)
        ccrr_tail_overlap_summary_df.to_csv(out_dir / "ccrr_tail_overlap_summary.csv", index=False)
        ccrr_tail_overlap_always_df.to_csv(out_dir / "ccrr_tail_overlap_always_fragile_samples.csv", index=False)
        ccrr_tail_overlap_unique_df.to_csv(out_dir / "ccrr_tail_overlap_unique_fragile_samples.csv", index=False)
    if len(ccrr_slide_enrichment_df) > 0:
        ccrr_slide_enrichment_df.to_csv(out_dir / "ccrr_slide_enrichment.csv", index=False)
        ccrr_slide_summary_df.to_csv(out_dir / "ccrr_slide_summary.csv", index=False)
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
        ccrr_m_sensitivity_df=ccrr_m_sensitivity_df,
        ccrr_stratum_enrichment_df=ccrr_stratum_enrichment_df,
        ccrr_stratum_concentration_df=ccrr_stratum_concentration_df,
        ccrr_tail_overlap_jaccard_df=ccrr_tail_overlap_jaccard_df,
        ccrr_tail_overlap_summary_df=ccrr_tail_overlap_summary_df,
        ccrr_slide_enrichment_df=ccrr_slide_enrichment_df,
        ccrr_slide_summary_df=ccrr_slide_summary_df,
    )

    if not bool(args.no_plots):
        try:
            _write_heatmap(pearson_corr, out_dir / "correlation_pearson.png", "Pearson Correlation")
            _write_heatmap(spearman_corr, out_dir / "correlation_spearman.png", "Spearman Correlation")
        except ModuleNotFoundError as exc:
            print(f"[analyze_results] Plot generation skipped (missing dependency): {exc}")

    print(f"[analyze_results] wrote analysis to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
