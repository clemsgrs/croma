import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mari.metrics.tail import select_exact_size_tail_set


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
_THRESH_UNDEFINED_COVERAGE_RISK = 0.50
_THRESH_OO_DOMINATED_HIGH = 0.10
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
_THRESH_FLAGGED_STRATUM_SLIDE_ENRICHMENT = 2.0
_THRESH_FLAGGED_STRATUM_SLIDE_MIN_SUPPORT_ABS = 2
_THRESH_FLAGGED_STRATUM_SLIDE_MIN_SUPPORT_FRAC = 0.10
_THRESH_TRUSTED_CCRR = 1.0
_THRESH_SHARED_TRUSTED_MODELS_FRAC = 0.50
_PER_SAMPLE_MATCH_COLS = [
    "dataset",
    "model",
    "evaluation_design",
    "evaluation_unit",
    "k",
    "tau",
    "ccrr_alpha",
    "ccrr_search",
    "excluded_centers",
]
_CCRR_CONTEXT_COLS = [
    "dataset",
    "evaluation_design",
    "evaluation_unit",
    "k",
    "tau",
    "ccrr_alpha",
    "ccrr_m",
    "ccrr_search",
    "excluded_centers",
]
_CCRR_REQUIRED_METRIC_COLS = {"model", "ccrr_m", "ccrr_alpha"}
_SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _normalize_context_key_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).replace({"nan": "", "None": ""})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze benchmark metrics with compact per-model and cross-model finding summaries."
    )
    parser.add_argument("--metrics-csv", required=True, type=Path, help="Path to benchmark metrics CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for analysis artifacts (default: <metrics parent>/analysis).",
    )
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
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Write the detailed appendix CSVs in addition to the compact outputs.",
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


def _aggregate_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        raise ValueError("metrics CSV must include a 'model' column")
    numeric_cols = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("metrics CSV has no numeric columns to analyze")
    grouped = df.groupby("model", as_index=False)[numeric_cols].mean(numeric_only=True)
    return grouped


def _rank_table(df_model: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df_model.loc[:, ["model"]].copy()
    for metric in metrics:
        ascending = not _is_higher_better(metric)
        out[f"rank_{metric}"] = df_model[metric].rank(method="min", ascending=ascending)
    return out


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




def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return df


def _ensure_evaluation_context_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    ensured = df.copy()
    if "evaluation_design" not in ensured.columns:
        ensured["evaluation_design"] = ""
    if "evaluation_unit" not in ensured.columns:
        ensured["evaluation_unit"] = ""
    return ensured


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


def _flagged_stratum_slide_support_threshold(stratum_tail_size: int) -> int:
    return int(
        max(
            _THRESH_FLAGGED_STRATUM_SLIDE_MIN_SUPPORT_ABS,
            np.ceil(_THRESH_FLAGGED_STRATUM_SLIDE_MIN_SUPPORT_FRAC * float(stratum_tail_size)),
        )
    )


def _metric_context_base(metric_row: pd.Series, *, ccrr_m: int, ccrr_col: str, alpha: float) -> dict:
    return {
        "dataset": str(metric_row.get("dataset", "")),
        "model": str(metric_row["model"]),
        "evaluation_design": str(metric_row.get("evaluation_design", "")),
        "evaluation_unit": str(metric_row.get("evaluation_unit", "")),
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
            "per-sample metrics CSV does not contain rows matching the metrics row for "
            f"model={metric_row.get('model')}"
        )

    ccrr_m = int(metric_row["ccrr_m"])
    ccrr_col = f"ccrr_m{ccrr_m}"
    if ccrr_col not in matched.columns:
        raise ValueError(f"per-sample metrics CSV missing required CCRR column: {ccrr_col}")

    # RI/MaRI artifacts are occurrence-grained; CCRR analysis needs the unique sample universe.
    if matched["sample_id"].duplicated().any():
        dedupe_cols = ["sample_id"]
        if "source_sample_index" in matched.columns:
            dedupe_cols.append("source_sample_index")
        matched = matched.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

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


def _severity_score(severity: str | float | None) -> int:
    if severity is None or pd.isna(severity):
        return 0
    return int(_SEVERITY_ORDER.get(str(severity).strip().lower(), 0))


def _tail_overlap_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def _ccrr_group_enrichment(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
    *,
    group_cols: list[str],
    group_count_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_per_sample is None or len(df_per_sample) == 0:
        return pd.DataFrame(), pd.DataFrame()

    required_sample_cols = {"sample_id", *group_cols}
    if not _CCRR_REQUIRED_METRIC_COLS.issubset(df_metrics_raw.columns):
        return pd.DataFrame(), pd.DataFrame()
    if not required_sample_cols.issubset(df_per_sample.columns):
        return pd.DataFrame(), pd.DataFrame()

    enrichment_rows: list[dict] = []
    summary_rows: list[dict] = []

    for _, metric_row in df_metrics_raw.iterrows():

        matched, tail_df, base, ccrr_col = _resolve_ccrr_tail_context(metric_row, df_per_sample)
        joined, tail_n = _merge_tail_counts(matched, tail_df, group_cols=group_cols, ccrr_col=ccrr_col)
        tail_support_threshold = _tail_support_threshold(tail_n) if tail_n > 0 else _THRESH_TAIL_STRATUM_MIN_SUPPORT_ABS
        joined["flagged"] = (
            (joined["enrichment_ratio"] >= _THRESH_TAIL_STRATUM_ENRICHMENT)
            & (joined["tail_count"] >= tail_support_threshold)
        )
        for _, row in joined.iterrows():
            enrichment_rows.append(
                {
                    **base,
                    "dataset_count": int(row["dataset_count"]),
                    "dataset_frac": float(row["dataset_frac"]),
                    "tail_count": int(row["tail_count"]),
                    "tail_frac": float(row["tail_frac"]),
                    "enrichment_ratio": float(row["enrichment_ratio"]),
                    "tail_support_threshold": int(tail_support_threshold),
                    "tail_mean_ccrr": float(row["tail_mean_ccrr"]) if pd.notna(row["tail_mean_ccrr"]) else float("nan"),
                    "tail_median_ccrr": float(row["tail_median_ccrr"]) if pd.notna(row["tail_median_ccrr"]) else float("nan"),
                    "flagged": bool(row["flagged"]),
                    **{col: str(row[col]) for col in group_cols},
                }
            )

        tail_count_series = tail_df.groupby(group_cols).size()
        summary_rows.append(
            {
                **base,
                "tail_sample_count": int(tail_n),
                "tail_support_threshold": int(tail_support_threshold),
                group_count_col: int(len(tail_count_series)),
                f"effective_tail_{group_count_col.removeprefix('tail_').removesuffix('_count')}": _effective_tail_strata_from_counts(tail_count_series),
                f"flagged_{group_count_col.removeprefix('tail_')}": int(joined["flagged"].sum()),
            }
        )

    enrichment_df = pd.DataFrame(enrichment_rows)
    summary_df = pd.DataFrame(summary_rows)
    return enrichment_df, summary_df


def _ccrr_stratum_enrichment(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enrichment_df, summary_df = _ccrr_group_enrichment(
        df_metrics_raw,
        df_per_sample,
        group_cols=["label", "medical_center"],
        group_count_col="tail_strata_count",
    )
    if len(enrichment_df) == 0:
        return enrichment_df, summary_df, pd.DataFrame()
    heatmap_df = enrichment_df.loc[:, ["dataset", "model", "label", "medical_center", "enrichment_ratio", "tail_count", "flagged"]].copy()
    return enrichment_df, summary_df, heatmap_df


def _ccrr_label_enrichment(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _ccrr_group_enrichment(
        df_metrics_raw,
        df_per_sample,
        group_cols=["label"],
        group_count_col="tail_label_count",
    )


def _ccrr_center_enrichment(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _ccrr_group_enrichment(
        df_metrics_raw,
        df_per_sample,
        group_cols=["medical_center"],
        group_count_col="tail_center_count",
    )


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


def _ccrr_flagged_stratum_slide_concentration(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
    ccrr_stratum_enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    if df_per_sample is None or len(df_per_sample) == 0 or len(ccrr_stratum_enrichment_df) == 0:
        return pd.DataFrame()

    required_sample_cols = {"sample_id", "slide_id", "label", "medical_center"}
    if not _CCRR_REQUIRED_METRIC_COLS.issubset(df_metrics_raw.columns):
        return pd.DataFrame()
    if not required_sample_cols.issubset(df_per_sample.columns):
        return pd.DataFrame()

    flagged_strata = ccrr_stratum_enrichment_df.loc[ccrr_stratum_enrichment_df["flagged"]].copy()
    if len(flagged_strata) == 0:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, metric_row in df_metrics_raw.iterrows():
        matched, tail_df, base, _ = _resolve_ccrr_tail_context(metric_row, df_per_sample)
        overlap_base = {key: value for key, value in base.items() if key != "ccrr_column"}
        context_mask = _context_mask(
            flagged_strata,
            pd.Series(overlap_base),
            context_cols=_CCRR_CONTEXT_COLS,
        )
        context_flagged = flagged_strata.loc[context_mask].copy()
        if len(context_flagged) == 0:
            continue

        for _, flagged_row in context_flagged.iterrows():
            stratum_all = matched.loc[
                (matched["label"].astype(str) == str(flagged_row["label"]))
                & (matched["medical_center"].astype(str) == str(flagged_row["medical_center"]))
            ].copy()
            stratum_tail = tail_df.loc[
                (tail_df["label"].astype(str) == str(flagged_row["label"]))
                & (tail_df["medical_center"].astype(str) == str(flagged_row["medical_center"]))
            ].copy()
            if len(stratum_tail) == 0:
                continue
            stratum_slide_counts = (
                stratum_all.groupby("slide_id", as_index=False)
                .agg(stratum_count=("sample_id", "size"))
                .sort_values(["stratum_count", "slide_id"], ascending=[False, True], kind="mergesort")
                .reset_index(drop=True)
            )
            slide_counts = (
                stratum_tail.groupby("slide_id", as_index=False)
                .agg(tail_count=("sample_id", "size"))
                .sort_values(["tail_count", "slide_id"], ascending=[False, True], kind="mergesort")
                .reset_index(drop=True)
                .merge(stratum_slide_counts, on="slide_id", how="left")
            )
            stratum_tail_count = int(len(stratum_tail))
            stratum_count = int(len(stratum_all))
            tail_slide_count = int(len(slide_counts))
            support_threshold = _flagged_stratum_slide_support_threshold(stratum_tail_count)
            effective_tail_slides = _effective_tail_strata_from_counts(slide_counts["tail_count"])

            for _, slide_row in slide_counts.iterrows():
                slide_tail_count = int(slide_row["tail_count"])
                slide_stratum_count = int(slide_row["stratum_count"])
                tail_frac_within_stratum = float(slide_tail_count / stratum_tail_count)
                stratum_frac = float(slide_stratum_count / stratum_count)
                enrichment_ratio = (
                    tail_frac_within_stratum / stratum_frac if stratum_frac > 0 else float("nan")
                )
                rows.append(
                    {
                        **overlap_base,
                        "label": str(flagged_row["label"]),
                        "medical_center": str(flagged_row["medical_center"]),
                        "slide_id": str(slide_row["slide_id"]),
                        "slide_tail_count": slide_tail_count,
                        "stratum_tail_count": stratum_tail_count,
                        "slide_stratum_count": slide_stratum_count,
                        "stratum_count": stratum_count,
                        "tail_slide_count": tail_slide_count,
                        "effective_tail_slides_within_stratum": effective_tail_slides,
                        "tail_frac_within_stratum": tail_frac_within_stratum,
                        "stratum_frac": stratum_frac,
                        "stratum_enrichment_ratio": enrichment_ratio,
                        "tail_support_threshold": int(support_threshold),
                        "flagged": bool(
                            (enrichment_ratio >= _THRESH_FLAGGED_STRATUM_SLIDE_ENRICHMENT)
                            and (slide_tail_count >= support_threshold)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def _ccrr_tail_entries_by_context(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
    *,
    required_sample_cols: set[str],
) -> dict[tuple[object, ...], list[dict]]:
    if df_per_sample is None or len(df_per_sample) == 0:
        return {}
    if not _CCRR_REQUIRED_METRIC_COLS.issubset(df_metrics_raw.columns):
        return {}
    if not required_sample_cols.issubset(df_per_sample.columns):
        return {}

    entries_by_context: dict[tuple[object, ...], list[dict]] = {}
    for _, metric_row in df_metrics_raw.iterrows():
        matched, tail_df, base, _ = _resolve_ccrr_tail_context(metric_row, df_per_sample)
        overlap_base = {key: value for key, value in base.items() if key != "ccrr_column"}
        context_key = tuple(overlap_base[col] for col in _CCRR_CONTEXT_COLS)
        entries_by_context.setdefault(context_key, []).append(
            {
                **overlap_base,
                "matched_df": matched.copy(),
                "universe_ids": tuple(sorted(matched["sample_id"].astype(str).tolist())),
                "tail_ids": set(tail_df["sample_id"].astype(str).tolist()),
                "tail_df": tail_df.copy(),
            }
        )
    return entries_by_context


def _summarize_cross_model_group_prevalence(
    sample_prevalence_df: pd.DataFrame,
    *,
    group_cols: list[str],
) -> pd.DataFrame:
    if len(sample_prevalence_df) == 0:
        return pd.DataFrame()
    grouped = (
        sample_prevalence_df.groupby(_CCRR_CONTEXT_COLS + group_cols, as_index=False)
        .agg(
            sample_count=("sample_id", "size"),
            mean_frac_models_in_tail=("frac_models_in_tail", "mean"),
            median_frac_models_in_tail=("frac_models_in_tail", "median"),
            max_frac_models_in_tail=("frac_models_in_tail", "max"),
            n_all_models_fragile=("all_models_fragile", "sum"),
            n_half_or_more_models_fragile=("half_or_more_models_fragile", "sum"),
        )
    )
    grouped["frac_half_or_more_models_fragile"] = (
        grouped["n_half_or_more_models_fragile"].astype(float) / grouped["sample_count"].astype(float)
    )
    return grouped


def _ccrr_cross_model_prevalence(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entries_by_context = _ccrr_tail_entries_by_context(
        df_metrics_raw,
        df_per_sample,
        required_sample_cols={"sample_id", "slide_id", "label", "medical_center"},
    )
    if not entries_by_context:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    sample_rows: list[dict] = []
    for entries in entries_by_context.values():
        if len(entries) < 2:
            continue
        context_values = {col: entries[0][col] for col in _CCRR_CONTEXT_COLS}
        universe_ids = entries[0]["universe_ids"]
        for entry in entries[1:]:
            if entry["universe_ids"] != universe_ids:
                raise ValueError("CCRR cross-model prevalence requires models evaluated on the same sample universe")

        ref_df = (
            entries[0]["matched_df"]
            .loc[:, ["sample_id", "slide_id", "label", "medical_center"]]
            .copy()
        )
        ref_df["sample_id"] = ref_df["sample_id"].astype(str)
        tail_counts: dict[str, int] = {sample_id: 0 for sample_id in ref_df["sample_id"].tolist()}
        for entry in entries:
            for sample_id in entry["tail_ids"]:
                tail_counts[str(sample_id)] += 1

        n_models_total = int(len(entries))
        for _, row in ref_df.iterrows():
            sample_id = str(row["sample_id"])
            n_models_in_tail = int(tail_counts[sample_id])
            frac = float(n_models_in_tail / n_models_total)
            sample_rows.append(
                {
                    **context_values,
                    "sample_id": sample_id,
                    "slide_id": str(row["slide_id"]),
                    "label": str(row["label"]),
                    "medical_center": str(row["medical_center"]),
                    "n_models_total": n_models_total,
                    "n_models_in_tail": n_models_in_tail,
                    "frac_models_in_tail": frac,
                    "all_models_fragile": bool(n_models_in_tail == n_models_total),
                    "half_or_more_models_fragile": bool(frac >= 0.5),
                }
            )

    sample_df = pd.DataFrame(sample_rows)
    if len(sample_df) == 0:
        return sample_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    stratum_df = _summarize_cross_model_group_prevalence(
        sample_df,
        group_cols=["label", "medical_center"],
    )
    label_df = _summarize_cross_model_group_prevalence(sample_df, group_cols=["label"])
    slide_df = _summarize_cross_model_group_prevalence(sample_df, group_cols=["slide_id"])
    return sample_df, stratum_df, label_df, slide_df


def _ccrr_tail_overlap(
    df_metrics_raw: pd.DataFrame,
    df_per_sample: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entries_by_context = _ccrr_tail_entries_by_context(
        df_metrics_raw,
        df_per_sample,
        required_sample_cols={"sample_id", "slide_id", "label", "medical_center"},
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
        columns=_CCRR_CONTEXT_COLS
        + [
            "model",
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
        for _, row in df_model[df_model[col] >= _THRESH_UNDEFINED_COVERAGE_RISK].iterrows():
            rows.append(
                {
                    "model": str(row["model"]),
                    "flag": f"coverage_risk_{metric}",
                    "severity": "high",
                    "value": float(row[col]),
                    "threshold": _THRESH_UNDEFINED_COVERAGE_RISK,
                    "detail": (
                        f"{_DISPLAY_NAMES.get(metric, metric)} is undefined on at least half the dataset "
                        f"({row[col]:.3f}); pooled score may not be representative."
                    ),
                }
            )

    # Undefined breakdown flags (SS/OO).
    for metric in ("ri", "mari"):
        oo_col = f"{metric}_oo_dominated_undefined_frac"

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


def _is_reliability_flag(flag: str) -> bool:
    reliability_flags = (
        "coverage_risk_",
        "poor_embedding_",
        "k_sweep_sensitivity_high",
        "ccrr_m_sweep_gain_high",
        "ccrr_search_cost_high",
    )
    return any(str(flag).startswith(prefix) for prefix in reliability_flags)


def _reliability_rows(action_flags_df: pd.DataFrame, model: str) -> pd.DataFrame:
    if len(action_flags_df) == 0:
        return pd.DataFrame()
    rows = action_flags_df.loc[
        (action_flags_df["model"].astype(str) == str(model))
        & (action_flags_df["flag"].astype(str).map(_is_reliability_flag))
    ].copy()
    rows = rows.loc[rows["flag"].astype(str) != "ccrr_search_cost_high"].copy()
    if len(rows) == 0:
        return rows
    rows["_severity_score"] = rows["severity"].map(_severity_score)
    return rows.sort_values(["_severity_score", "flag"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _compact_reliability_label(flag: str) -> str:
    flag = str(flag)
    if flag.startswith("coverage_risk_ri"):
        return "RI coverage is poor"
    if flag.startswith("coverage_risk_mari"):
        return "MaRI coverage is poor"
    if flag.startswith("coverage_risk_ccrr"):
        return "CCRR coverage is poor"
    if flag.startswith("poor_embedding_ri"):
        return "RI undefined samples suggest poor embedding quality"
    if flag.startswith("poor_embedding_mari"):
        return "MaRI undefined samples suggest poor embedding quality"
    if flag == "k_sweep_sensitivity_high":
        return "RI/MaRI are sensitive to k"
    if flag == "ccrr_m_sweep_gain_high":
        return "CCRR changes materially across m"
    return flag.replace("_", " ")


def _rank_lookup(rank_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    if len(rank_df) == 0 or "model" not in rank_df.columns:
        return lookup
    for _, row in rank_df.iterrows():
        lookup[str(row["model"])] = {
            "rank_ri": float(row["rank_ri"]) if "rank_ri" in row.index and pd.notna(row["rank_ri"]) else float("nan"),
            "rank_mari": float(row["rank_mari"]) if "rank_mari" in row.index and pd.notna(row["rank_mari"]) else float("nan"),
            "rank_ccrr": float(row["rank_ccrr"]) if "rank_ccrr" in row.index and pd.notna(row["rank_ccrr"]) else float("nan"),
        }
    return lookup


def _characterization_base_row(
    *,
    model: str,
    rank_lookup: dict[str, dict[str, float]],
    question_scope: str,
    report_rank_within_model: int,
    source_row: pd.Series | None = None,
) -> dict:
    rank_info = rank_lookup.get(model, {})
    row_out = {
        "model": model,
        "rank_ri": rank_info.get("rank_ri", float("nan")),
        "rank_mari": rank_info.get("rank_mari", float("nan")),
        "rank_ccrr": rank_info.get("rank_ccrr", float("nan")),
        "question_scope": question_scope,
        "report_rank_within_model": int(report_rank_within_model),
    }
    if source_row is not None:
        for col in _CCRR_CONTEXT_COLS:
            if col in source_row.index:
                row_out[col] = source_row[col]
    return row_out


def _sorted_group_findings(df: pd.DataFrame, *, key_cols: list[str], model: str) -> pd.DataFrame:
    if len(df) == 0 or "model" not in df.columns:
        return pd.DataFrame()
    rows = df.loc[(df["model"].astype(str) == model) & (df["flagged"])].copy()
    if len(rows) == 0:
        return rows
    rows = rows.sort_values(
        ["enrichment_ratio", "tail_count"] + key_cols,
        ascending=[False, False] + [True] * len(key_cols),
        kind="mergesort",
    ).reset_index(drop=True)
    return rows


def _sorted_supported_stratum_severity(df: pd.DataFrame, *, model: str) -> pd.DataFrame:
    if len(df) == 0 or "model" not in df.columns:
        return pd.DataFrame()
    rows = df.loc[df["model"].astype(str) == model].copy()
    if len(rows) == 0:
        return rows
    rows = rows.loc[
        pd.to_numeric(rows["tail_count"], errors="coerce").ge(pd.to_numeric(rows["tail_support_threshold"], errors="coerce"))
        & pd.to_numeric(rows["tail_mean_ccrr"], errors="coerce").notna()
    ].copy()
    if len(rows) == 0:
        return rows
    rows = rows.sort_values(
        ["tail_mean_ccrr", "tail_count", "enrichment_ratio", "label", "medical_center"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return rows


def _sorted_flagged_stratum_slides(df: pd.DataFrame, *, model: str) -> pd.DataFrame:
    if len(df) == 0 or "model" not in df.columns:
        return pd.DataFrame()
    rows = df.loc[(df["model"].astype(str) == model) & (df["flagged"])].copy()
    if len(rows) == 0:
        return rows
    rows = rows.drop_duplicates(subset=["label", "medical_center", "slide_id"], keep="first").copy()
    rows = rows.sort_values(
        ["label", "medical_center", "stratum_enrichment_ratio", "slide_tail_count", "slide_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return rows


def _model_sort_key(characterization_df: pd.DataFrame) -> pd.DataFrame:
    if len(characterization_df) == 0:
        return pd.DataFrame(columns=["model", "mean_rank", "rank_ccrr", "rank_ri", "rank_mari"])
    rows = characterization_df.loc[:, ["model", "rank_ri", "rank_mari", "rank_ccrr"]].drop_duplicates(subset=["model"]).copy()
    rows["mean_rank"] = rows[["rank_ri", "rank_mari", "rank_ccrr"]].mean(axis=1)
    return rows.sort_values(
        ["mean_rank", "rank_ccrr", "rank_ri", "rank_mari", "model"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_single_model_tail_characterization(
    *,
    df_model: pd.DataFrame,
    rank_df: pd.DataFrame,
    action_flags_df: pd.DataFrame,
    ccrr_stratum_enrichment_df: pd.DataFrame,
    ccrr_stratum_concentration_df: pd.DataFrame,
    ccrr_label_enrichment_df: pd.DataFrame,
    ccrr_center_enrichment_df: pd.DataFrame,
    ccrr_slide_enrichment_df: pd.DataFrame,
    ccrr_flagged_stratum_slide_concentration_df: pd.DataFrame,
) -> pd.DataFrame:
    rank_lookup = _rank_lookup(rank_df)
    rows: list[dict] = []

    for model in df_model["model"].astype(str).tolist():
        reliability_rows = _reliability_rows(action_flags_df, model)
        for idx, (_, row) in enumerate(reliability_rows.iterrows(), start=1):
            rows.append(
                {
                    **_characterization_base_row(
                        model=model,
                        rank_lookup=rank_lookup,
                        question_scope="reliability",
                        report_rank_within_model=idx,
                    ),
                    "flag": str(row["flag"]),
                    "severity": str(row["severity"]) if "severity" in row.index else "",
                    "detail": str(row["detail"]) if "detail" in row.index and pd.notna(row["detail"]) else "",
                    "summary_text": _compact_reliability_label(str(row["flag"])),
                }
            )

        if len(ccrr_stratum_concentration_df) == 0 or "model" not in ccrr_stratum_concentration_df.columns:
            model_concentration = pd.DataFrame()
        else:
            model_concentration = ccrr_stratum_concentration_df.loc[
                ccrr_stratum_concentration_df["model"].astype(str) == model
            ].copy()
        if len(model_concentration) > 0:
            model_concentration = model_concentration.sort_values(
                [col for col in _CCRR_CONTEXT_COLS if col in model_concentration.columns],
                kind="mergesort",
            ).reset_index(drop=True)
            for idx, (_, row) in enumerate(model_concentration.iterrows(), start=1):
                rows.append(
                    {
                        **_characterization_base_row(
                            model=model,
                            rank_lookup=rank_lookup,
                            question_scope="tail_concentration",
                            report_rank_within_model=idx,
                            source_row=row,
                        ),
                        "tail_sample_count": int(row["tail_sample_count"]),
                        "tail_strata_count": int(row["tail_strata_count"]),
                        "flagged_strata_count": int(row["flagged_strata_count"]),
                        "effective_tail_strata": float(row["effective_tail_strata"]),
                    }
                )
        else:
            rows.append(
                {
                    **_characterization_base_row(
                        model=model,
                        rank_lookup=rank_lookup,
                        question_scope="tail_concentration",
                        report_rank_within_model=1,
                    ),
                    "tail_sample_count": float("nan"),
                    "tail_strata_count": float("nan"),
                    "flagged_strata_count": float("nan"),
                    "effective_tail_strata": float("nan"),
                    "summary_text": "Per-sample artifact unavailable",
                }
            )

        for scope_name, scope_df, key_cols in (
            ("stratum_enrichment", ccrr_stratum_enrichment_df, ["label", "medical_center"]),
            ("label_enrichment", ccrr_label_enrichment_df, ["label"]),
            ("center_enrichment", ccrr_center_enrichment_df, ["medical_center"]),
            ("slide_enrichment", ccrr_slide_enrichment_df, ["slide_id"]),
        ):
            group_rows = _sorted_group_findings(scope_df, key_cols=key_cols, model=model)
            for idx, (_, row) in enumerate(group_rows.iterrows(), start=1):
                row_out = {
                    **_characterization_base_row(
                        model=model,
                        rank_lookup=rank_lookup,
                        question_scope=scope_name,
                        report_rank_within_model=idx,
                        source_row=row,
                    ),
                    "flagged": True,
                    "tail_count": int(row["tail_count"]),
                    "tail_support_threshold": int(row["tail_support_threshold"]),
                    "tail_frac": float(row["tail_frac"]),
                    "dataset_count": int(row["dataset_count"]),
                    "dataset_frac": float(row["dataset_frac"]),
                    "enrichment_ratio": float(row["enrichment_ratio"]),
                    "tail_mean_ccrr": float(row["tail_mean_ccrr"]),
                    "tail_median_ccrr": float(row["tail_median_ccrr"]),
                }
                for key in key_cols:
                    row_out[key] = str(row[key])
                rows.append(row_out)

        severity_rows = _sorted_supported_stratum_severity(ccrr_stratum_enrichment_df, model=model)
        for idx, (_, row) in enumerate(severity_rows.iterrows(), start=1):
            rows.append(
                {
                    **_characterization_base_row(
                        model=model,
                        rank_lookup=rank_lookup,
                        question_scope="stratum_severity",
                        report_rank_within_model=idx,
                        source_row=row,
                    ),
                    "label": str(row["label"]),
                    "medical_center": str(row["medical_center"]),
                    "flagged": bool(row["flagged"]),
                    "tail_count": int(row["tail_count"]),
                    "tail_support_threshold": int(row["tail_support_threshold"]),
                    "tail_frac": float(row["tail_frac"]),
                    "dataset_count": int(row["dataset_count"]),
                    "dataset_frac": float(row["dataset_frac"]),
                    "enrichment_ratio": float(row["enrichment_ratio"]),
                    "tail_mean_ccrr": float(row["tail_mean_ccrr"]),
                    "tail_median_ccrr": float(row["tail_median_ccrr"]),
                }
            )

        stratum_slide_rows = _sorted_flagged_stratum_slides(ccrr_flagged_stratum_slide_concentration_df, model=model)
        for idx, (_, row) in enumerate(stratum_slide_rows.iterrows(), start=1):
            rows.append(
                {
                    **_characterization_base_row(
                        model=model,
                        rank_lookup=rank_lookup,
                        question_scope="stratum_slide_enrichment",
                        report_rank_within_model=idx,
                        source_row=row,
                    ),
                    "label": str(row["label"]),
                    "medical_center": str(row["medical_center"]),
                    "slide_id": str(row["slide_id"]),
                    "flagged": True,
                    "slide_tail_count": int(row["slide_tail_count"]),
                    "slide_stratum_count": int(row["slide_stratum_count"]),
                    "tail_frac_within_stratum": float(row["tail_frac_within_stratum"]),
                    "stratum_frac": float(row["stratum_frac"]),
                    "stratum_enrichment_ratio": float(row["stratum_enrichment_ratio"]),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "rank_ri",
                "rank_mari",
                "rank_ccrr",
                "question_scope",
                "report_rank_within_model",
            ]
        )

    out = pd.DataFrame(rows)
    ordered_cols = [
        "model",
        "rank_ri",
        "rank_mari",
        "rank_ccrr",
        "question_scope",
        "report_rank_within_model",
        *_CCRR_CONTEXT_COLS,
        "flag",
        "severity",
        "detail",
        "summary_text",
        "label",
        "medical_center",
        "slide_id",
        "flagged",
        "tail_count",
        "tail_support_threshold",
        "tail_frac",
        "dataset_count",
        "dataset_frac",
        "enrichment_ratio",
        "tail_mean_ccrr",
        "tail_median_ccrr",
        "tail_sample_count",
        "tail_strata_count",
        "flagged_strata_count",
        "effective_tail_strata",
        "slide_tail_count",
        "slide_stratum_count",
        "tail_frac_within_stratum",
        "stratum_frac",
        "stratum_enrichment_ratio",
    ]
    return out.loc[:, [col for col in ordered_cols if col in out.columns]]


def _context_n_models_total(sample_df: pd.DataFrame) -> pd.DataFrame:
    if len(sample_df) == 0:
        return pd.DataFrame(columns=_CCRR_CONTEXT_COLS + ["n_models_total"])
    return sample_df.groupby(_CCRR_CONTEXT_COLS, as_index=False).agg(n_models_total=("n_models_total", "max"))


def _trusted_cross_model_cohort(df_metrics_raw: pd.DataFrame) -> pd.DataFrame:
    required = set(_CCRR_CONTEXT_COLS) | {"model", "ccrr"}
    if not required.issubset(df_metrics_raw.columns):
        return pd.DataFrame(columns=_CCRR_CONTEXT_COLS + ["model", "ccrr"])
    trusted = df_metrics_raw.loc[
        pd.to_numeric(df_metrics_raw["ccrr"], errors="coerce").ge(_THRESH_TRUSTED_CCRR)
    ].copy()
    if len(trusted) == 0:
        return pd.DataFrame(columns=_CCRR_CONTEXT_COLS + ["model", "ccrr"])
    for col in _CCRR_CONTEXT_COLS:
        trusted[col] = _normalize_context_key_series(trusted[col])
    trusted["model"] = trusted["model"].astype(str)
    return trusted.loc[:, _CCRR_CONTEXT_COLS + ["model", "ccrr"]].drop_duplicates(
        subset=_CCRR_CONTEXT_COLS + ["model"],
        keep="first",
    )


def _build_cross_model_findings(
    *,
    df_metrics_raw: pd.DataFrame,
    ccrr_label_enrichment_df: pd.DataFrame,
    ccrr_stratum_enrichment_df: pd.DataFrame,
    ccrr_slide_enrichment_df: pd.DataFrame,
) -> pd.DataFrame:
    trusted_models_df = _trusted_cross_model_cohort(df_metrics_raw)
    if len(trusted_models_df) == 0:
        return pd.DataFrame(
            columns=[
                "finding_scope",
                "label",
                "medical_center",
                "slide_id",
                "n_models_trusted",
                "n_models_flagged",
                "frac_models_flagged",
                "median_enrichment_ratio",
                "median_tail_frac",
                "median_tail_count",
                "median_tail_mean_ccrr",
                "flagged_models",
                "rank_within_scope",
            ]
        )
    trusted_counts_df = trusted_models_df.groupby(_CCRR_CONTEXT_COLS, as_index=False).agg(
        n_models_trusted=("model", "nunique")
    )
    merge_cols = _CCRR_CONTEXT_COLS + ["model"]
    trusted_merge_df = trusted_models_df.copy()
    trusted_counts_merge_df = trusted_counts_df.copy()
    for col in _CCRR_CONTEXT_COLS:
        trusted_merge_df[col] = _normalize_context_key_series(trusted_merge_df[col])
        trusted_counts_merge_df[col] = _normalize_context_key_series(trusted_counts_merge_df[col])
    rows: list[dict] = []

    def add_group_scope(df: pd.DataFrame, scope: str, key_cols: list[str]) -> None:
        if len(df) == 0:
            return
        required_cols = set(_CCRR_CONTEXT_COLS) | {"model", "flagged", "enrichment_ratio", "tail_frac", "tail_count", "tail_mean_ccrr"} | set(key_cols)
        if not required_cols.issubset(df.columns):
            return
        flagged = df.loc[df["flagged"]].copy()
        if len(flagged) == 0:
            return
        for col in _CCRR_CONTEXT_COLS:
            flagged[col] = _normalize_context_key_series(flagged[col])
        flagged["model"] = flagged["model"].astype(str)
        flagged = flagged.merge(
            trusted_merge_df.loc[:, merge_cols],
            on=merge_cols,
            how="inner",
        )
        if len(flagged) == 0:
            return
        grouped = (
            flagged.groupby(_CCRR_CONTEXT_COLS + key_cols, as_index=False)
            .agg(
                n_models_flagged=("model", "nunique"),
                median_enrichment_ratio=("enrichment_ratio", "median"),
                median_tail_frac=("tail_frac", "median"),
                median_tail_count=("tail_count", "median"),
                median_tail_mean_ccrr=("tail_mean_ccrr", "median"),
                flagged_models=("model", lambda s: ",".join(sorted({str(v) for v in s}))),
            )
            .merge(trusted_counts_merge_df, on=_CCRR_CONTEXT_COLS, how="left")
        )
        grouped["frac_models_flagged"] = grouped["n_models_flagged"].astype(float) / grouped["n_models_trusted"].astype(float)
        grouped = grouped.loc[
            (grouped["n_models_trusted"] >= 2)
            & (grouped["frac_models_flagged"] >= _THRESH_SHARED_TRUSTED_MODELS_FRAC)
        ].copy()
        if len(grouped) == 0:
            return
        grouped = grouped.sort_values(
            ["frac_models_flagged", "n_models_flagged", "median_enrichment_ratio", "median_tail_count"] + key_cols,
            ascending=[False, False, False, False] + [True] * len(key_cols),
            kind="mergesort",
        ).reset_index(drop=True)
        for rank, (_, row) in enumerate(grouped.iterrows(), start=1):
            rows.append(
                {
                    "finding_scope": scope,
                    "label": str(row["label"]) if "label" in row.index else "",
                    "medical_center": str(row["medical_center"]) if "medical_center" in row.index else "",
                    "slide_id": str(row["slide_id"]) if "slide_id" in row.index else "",
                    "n_models_trusted": int(row["n_models_trusted"]),
                    "n_models_flagged": int(row["n_models_flagged"]),
                    "frac_models_flagged": float(row["frac_models_flagged"]),
                    "median_enrichment_ratio": float(row["median_enrichment_ratio"]),
                    "median_tail_frac": float(row["median_tail_frac"]),
                    "median_tail_count": float(row["median_tail_count"]),
                    "median_tail_mean_ccrr": float(row["median_tail_mean_ccrr"]),
                    "flagged_models": str(row["flagged_models"]),
                    "rank_within_scope": int(rank),
                }
            )

    add_group_scope(ccrr_stratum_enrichment_df, "stratum", ["label", "medical_center"])
    add_group_scope(ccrr_label_enrichment_df, "label", ["label"])
    add_group_scope(ccrr_slide_enrichment_df, "slide", ["slide_id"])

    return pd.DataFrame(
        rows,
        columns=[
            "finding_scope",
            "label",
            "medical_center",
            "slide_id",
            "n_models_trusted",
            "n_models_flagged",
            "frac_models_flagged",
            "median_enrichment_ratio",
            "median_tail_frac",
            "median_tail_count",
            "median_tail_mean_ccrr",
            "flagged_models",
            "rank_within_scope",
        ],
    )


def _write_single_model_tail_characterization_report(
    *,
    out_path: Path,
    input_csv: Path,
    df_raw: pd.DataFrame,
    characterization_df: pd.DataFrame,
) -> None:
    def _fmt_group(row: pd.Series, *, include_dataset: bool = True) -> str:
        pieces: list[str] = []
        if "label" in row.index and pd.notna(row.get("label")) and str(row.get("label", "")).strip():
            pieces.append(str(row["label"]))
        if "medical_center" in row.index and pd.notna(row.get("medical_center")) and str(row.get("medical_center", "")).strip():
            pieces.append(str(row["medical_center"]))
        if "slide_id" in row.index and pd.notna(row.get("slide_id")) and str(row.get("slide_id", "")).strip():
            pieces.append(str(row["slide_id"]))
        label = " / ".join(pieces)
        evidence: list[str] = []
        if "tail_count" in row.index and pd.notna(row.get("tail_count")):
            evidence.append(f"tail_count={int(row['tail_count'])}")
        if include_dataset and "dataset_count" in row.index and pd.notna(row.get("dataset_count")):
            evidence.append(f"dataset_count={int(row['dataset_count'])}")
        if "tail_frac" in row.index and pd.notna(row.get("tail_frac")):
            evidence.append(f"tail_frac={float(row['tail_frac']):.1%}")
        if include_dataset and "dataset_frac" in row.index and pd.notna(row.get("dataset_frac")):
            evidence.append(f"dataset_frac={float(row['dataset_frac']):.1%}")
        if "enrichment_ratio" in row.index and pd.notna(row.get("enrichment_ratio")):
            evidence.append(f"enrichment={float(row['enrichment_ratio']):.2f}x")
        if "tail_mean_ccrr" in row.index and pd.notna(row.get("tail_mean_ccrr")):
            evidence.append(f"tail_mean_ccrr={float(row['tail_mean_ccrr']):.3f}")
        if "tail_median_ccrr" in row.index and pd.notna(row.get("tail_median_ccrr")):
            evidence.append(f"tail_median_ccrr={float(row['tail_median_ccrr']):.3f}")
        return f"{label}: " + ", ".join(evidence)

    dataset_values = sorted(df_raw["dataset"].dropna().astype(str).unique().tolist()) if "dataset" in df_raw.columns else []
    dataset_label = ", ".join(dataset_values) if dataset_values else "unknown"
    model_order_df = _model_sort_key(characterization_df)

    lines: list[str] = []
    lines.append("# Single-Model Tail Characterization")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Input CSV: `{input_csv}`")
    lines.append(f"- Dataset: {dataset_label}")
    lines.append(f"- Models analyzed: {len(model_order_df)}")
    lines.append("")

    if len(model_order_df) == 0:
        lines.append("- No single-model tail characterization findings available.")
        lines.append("")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for _, model_meta in model_order_df.iterrows():
        model = str(model_meta["model"])
        model_rows = characterization_df.loc[characterization_df["model"].astype(str) == model].copy()
        lines.append(f"## Model {model}")
        lines.append("")
        lines.append(
            f"- Ranks: RI={int(round(float(model_meta['rank_ri'])))}; "
            f"MaRI={int(round(float(model_meta['rank_mari'])))}; "
            f"CCRR={int(round(float(model_meta['rank_ccrr'])))}"
        )
        lines.append("")

        reliability_rows = model_rows.loc[model_rows["question_scope"] == "reliability"].sort_values("report_rank_within_model")
        if len(reliability_rows) > 0:
            lines.append("### Reliability")
            lines.append("")
            for _, row in reliability_rows.iterrows():
                lines.append(f"- {row['summary_text']}")
            lines.append("")

        def add_group_section(title: str, scope: str) -> None:
            group_rows = model_rows.loc[model_rows["question_scope"] == scope].sort_values("report_rank_within_model")
            if len(group_rows) == 0:
                return
            lines.append(f"### {title}")
            lines.append("")
            for _, row in group_rows.iterrows():
                lines.append(f"- {_fmt_group(row)}")
            lines.append("")

        add_group_section("Stratum Enrichment", "stratum_enrichment")
        add_group_section("Biology Enrichment", "label_enrichment")
        add_group_section("Center Enrichment", "center_enrichment")

        concentration_rows = model_rows.loc[model_rows["question_scope"] == "tail_concentration"].sort_values("report_rank_within_model")
        if len(concentration_rows) > 0:
            lines.append("### Tail Concentration")
            lines.append("")
            for _, row in concentration_rows.iterrows():
                if pd.isna(row.get("tail_sample_count")):
                    lines.append(f"- {str(row.get('summary_text', 'Per-sample artifact unavailable'))}")
                else:
                    lines.append(
                        "- "
                        + ", ".join(
                            [
                                f"tail_sample_count={int(row['tail_sample_count'])}",
                                f"tail_strata_count={int(row['tail_strata_count'])}",
                                f"flagged_strata_count={int(row['flagged_strata_count'])}",
                                f"effective_tail_strata={float(row['effective_tail_strata']):.2f}",
                            ]
                        )
                    )
            lines.append("")

        severity_rows = model_rows.loc[model_rows["question_scope"] == "stratum_severity"].sort_values("report_rank_within_model")
        if len(severity_rows) > 0:
            lines.append("### Stratum Severity")
            lines.append("")
            for _, row in severity_rows.iterrows():
                lines.append(f"- {_fmt_group(row, include_dataset=False)}")
            lines.append("")

        slide_rows = model_rows.loc[model_rows["question_scope"] == "slide_enrichment"].sort_values("report_rank_within_model")
        within_stratum_slide_rows = model_rows.loc[
            model_rows["question_scope"] == "stratum_slide_enrichment"
        ].sort_values("report_rank_within_model")
        if len(slide_rows) > 0 or len(within_stratum_slide_rows) > 0:
            lines.append("### Slide-Level Patterns")
            lines.append("")
            if len(slide_rows) > 0:
                lines.append("Flagged slides:")
                for _, row in slide_rows.iterrows():
                    lines.append(f"- {_fmt_group(row)}")
            if len(within_stratum_slide_rows) > 0:
                lines.append("Within flagged strata:")
                for _, row in within_stratum_slide_rows.iterrows():
                    lines.append(
                        "- "
                        + f"{row['label']} / {row['medical_center']} -> {row['slide_id']}: "
                        + ", ".join(
                            [
                                f"slide_tail_count={int(row['slide_tail_count'])}",
                                f"slide_stratum_count={int(row['slide_stratum_count'])}",
                                f"tail_frac_within_stratum={float(row['tail_frac_within_stratum']):.1%}",
                                f"stratum_frac={float(row['stratum_frac']):.1%}",
                                f"enrichment={float(row['stratum_enrichment_ratio']):.2f}x",
                            ]
                        )
                    )
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_detailed_artifacts(
    *,
    out_dir: Path,
    rank_df: pd.DataFrame,
    action_flags_df: pd.DataFrame,
    k_sensitivity_df: pd.DataFrame,
    ccrr_m_sensitivity_df: pd.DataFrame,
    ccrr_label_enrichment_df: pd.DataFrame,
    ccrr_label_summary_df: pd.DataFrame,
    ccrr_center_enrichment_df: pd.DataFrame,
    ccrr_center_summary_df: pd.DataFrame,
    ccrr_stratum_enrichment_df: pd.DataFrame,
    ccrr_stratum_concentration_df: pd.DataFrame,
    ccrr_flagged_stratum_slide_concentration_df: pd.DataFrame,
    ccrr_cross_model_sample_prevalence_df: pd.DataFrame,
    ccrr_cross_model_stratum_prevalence_df: pd.DataFrame,
    ccrr_cross_model_label_prevalence_df: pd.DataFrame,
    ccrr_cross_model_slide_prevalence_df: pd.DataFrame,
    ccrr_tail_overlap_jaccard_df: pd.DataFrame,
    ccrr_tail_overlap_summary_df: pd.DataFrame,
    ccrr_tail_overlap_always_df: pd.DataFrame,
    ccrr_tail_overlap_unique_df: pd.DataFrame,
    ccrr_slide_enrichment_df: pd.DataFrame,
    ccrr_slide_summary_df: pd.DataFrame,
) -> None:
    rank_df.to_csv(out_dir / "model_ranks.csv", index=False)
    action_flags_df.to_csv(out_dir / "model_action_flags.csv", index=False)
    if len(k_sensitivity_df) > 0:
        k_sensitivity_df.to_csv(out_dir / "k_sweep_sensitivity.csv", index=False)
    if len(ccrr_m_sensitivity_df) > 0:
        ccrr_m_sensitivity_df.to_csv(out_dir / "ccrr_m_sweep_sensitivity.csv", index=False)
    if len(ccrr_label_enrichment_df) > 0:
        ccrr_label_enrichment_df.to_csv(out_dir / "ccrr_label_enrichment.csv", index=False)
        ccrr_label_summary_df.to_csv(out_dir / "ccrr_label_summary.csv", index=False)
    if len(ccrr_center_enrichment_df) > 0:
        ccrr_center_enrichment_df.to_csv(out_dir / "ccrr_center_enrichment.csv", index=False)
        ccrr_center_summary_df.to_csv(out_dir / "ccrr_center_summary.csv", index=False)
    if len(ccrr_stratum_enrichment_df) > 0:
        ccrr_stratum_enrichment_df.to_csv(out_dir / "ccrr_stratum_enrichment.csv", index=False)
        ccrr_stratum_concentration_df.to_csv(out_dir / "ccrr_stratum_concentration.csv", index=False)
    if len(ccrr_flagged_stratum_slide_concentration_df) > 0:
        ccrr_flagged_stratum_slide_concentration_df.to_csv(
            out_dir / "ccrr_flagged_stratum_slide_concentration.csv",
            index=False,
        )
    if len(ccrr_cross_model_sample_prevalence_df) > 0:
        ccrr_cross_model_sample_prevalence_df.to_csv(out_dir / "ccrr_cross_model_sample_prevalence.csv", index=False)
        ccrr_cross_model_stratum_prevalence_df.to_csv(out_dir / "ccrr_cross_model_stratum_prevalence.csv", index=False)
        ccrr_cross_model_label_prevalence_df.to_csv(out_dir / "ccrr_cross_model_label_prevalence.csv", index=False)
        ccrr_cross_model_slide_prevalence_df.to_csv(out_dir / "ccrr_cross_model_slide_prevalence.csv", index=False)
    if len(ccrr_tail_overlap_jaccard_df) > 0:
        ccrr_tail_overlap_jaccard_df.to_csv(out_dir / "ccrr_tail_overlap_jaccard.csv", index=False)
        ccrr_tail_overlap_summary_df.to_csv(out_dir / "ccrr_tail_overlap_summary.csv", index=False)
        ccrr_tail_overlap_always_df.to_csv(out_dir / "ccrr_tail_overlap_always_fragile_samples.csv", index=False)
        ccrr_tail_overlap_unique_df.to_csv(out_dir / "ccrr_tail_overlap_unique_fragile_samples.csv", index=False)
    if len(ccrr_slide_enrichment_df) > 0:
        ccrr_slide_enrichment_df.to_csv(out_dir / "ccrr_slide_enrichment.csv", index=False)
        ccrr_slide_summary_df.to_csv(out_dir / "ccrr_slide_summary.csv", index=False)


def main() -> int:
    args = _parse_args()
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else metrics_csv.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = _ensure_evaluation_context_columns(pd.read_csv(metrics_csv))
    if len(df_raw) == 0:
        raise ValueError(f"Metrics CSV is empty: {metrics_csv}")

    df_model = _aggregate_by_model(df_raw)
    numeric_cols = [c for c in df_model.columns if c != "model" and pd.api.types.is_numeric_dtype(df_model[c])]
    rank_metrics = _resolve_required_metrics(_RANK_METRICS_CANONICAL, numeric_cols, "rank-shift")
    rank_df = _rank_table(df_model, metrics=rank_metrics)
    delta_df = _rank_deltas(rank_df)

    k_sweep_path = Path(args.k_sweep_csv) if args.k_sweep_csv is not None else metrics_csv.parent / "k_sweep_metrics.csv"
    ccrr_m_sweep_path = (
        Path(args.ccrr_m_sweep_csv) if args.ccrr_m_sweep_csv is not None else metrics_csv.parent / "ccrr_m_sweep_metrics.csv"
    )
    per_sample_path = Path(args.per_sample_csv) if args.per_sample_csv is not None else metrics_csv.parent / "per_sample_metrics.csv"
    df_k_sweep = _ensure_evaluation_context_columns(_load_optional_csv(k_sweep_path))
    df_ccrr_m_sweep = _ensure_evaluation_context_columns(_load_optional_csv(ccrr_m_sweep_path))
    df_per_sample = _ensure_evaluation_context_columns(_load_optional_csv(per_sample_path))

    k_sensitivity_df = _k_sweep_sensitivity(df_k_sweep)
    ccrr_m_sensitivity_df = _ccrr_m_sweep_sensitivity(df_ccrr_m_sweep)
    ccrr_label_enrichment_df, ccrr_label_summary_df = _ccrr_label_enrichment(df_raw, df_per_sample)
    ccrr_center_enrichment_df, ccrr_center_summary_df = _ccrr_center_enrichment(df_raw, df_per_sample)
    ccrr_stratum_enrichment_df, ccrr_stratum_concentration_df, _ = _ccrr_stratum_enrichment(df_raw, df_per_sample)
    ccrr_flagged_stratum_slide_concentration_df = _ccrr_flagged_stratum_slide_concentration(
        df_raw,
        df_per_sample,
        ccrr_stratum_enrichment_df,
    )
    (
        ccrr_cross_model_sample_prevalence_df,
        ccrr_cross_model_stratum_prevalence_df,
        ccrr_cross_model_label_prevalence_df,
        ccrr_cross_model_slide_prevalence_df,
    ) = _ccrr_cross_model_prevalence(df_raw, df_per_sample)
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
    single_model_tail_characterization_df = _build_single_model_tail_characterization(
        df_model=df_model,
        rank_df=rank_df,
        action_flags_df=action_flags_df,
        ccrr_stratum_enrichment_df=ccrr_stratum_enrichment_df,
        ccrr_stratum_concentration_df=ccrr_stratum_concentration_df,
        ccrr_label_enrichment_df=ccrr_label_enrichment_df,
        ccrr_center_enrichment_df=ccrr_center_enrichment_df,
        ccrr_slide_enrichment_df=ccrr_slide_enrichment_df,
        ccrr_flagged_stratum_slide_concentration_df=ccrr_flagged_stratum_slide_concentration_df,
    )
    cross_model_findings_df = _build_cross_model_findings(
        df_metrics_raw=df_raw,
        ccrr_label_enrichment_df=ccrr_label_enrichment_df,
        ccrr_stratum_enrichment_df=ccrr_stratum_enrichment_df,
        ccrr_slide_enrichment_df=ccrr_slide_enrichment_df,
    )

    single_model_tail_characterization_df.to_csv(out_dir / "single_model_tail_characterization.csv", index=False)
    cross_model_findings_df.to_csv(out_dir / "cross_model_findings.csv", index=False)
    _write_single_model_tail_characterization_report(
        out_path=out_dir / "single_model_tail_characterization.md",
        input_csv=metrics_csv,
        df_raw=df_raw,
        characterization_df=single_model_tail_characterization_df,
    )

    if bool(args.detailed):
        _write_detailed_artifacts(
            out_dir=out_dir,
            rank_df=rank_df,
            action_flags_df=action_flags_df,
            k_sensitivity_df=k_sensitivity_df,
            ccrr_m_sensitivity_df=ccrr_m_sensitivity_df,
            ccrr_label_enrichment_df=ccrr_label_enrichment_df,
            ccrr_label_summary_df=ccrr_label_summary_df,
            ccrr_center_enrichment_df=ccrr_center_enrichment_df,
            ccrr_center_summary_df=ccrr_center_summary_df,
            ccrr_stratum_enrichment_df=ccrr_stratum_enrichment_df,
            ccrr_stratum_concentration_df=ccrr_stratum_concentration_df,
            ccrr_flagged_stratum_slide_concentration_df=ccrr_flagged_stratum_slide_concentration_df,
            ccrr_cross_model_sample_prevalence_df=ccrr_cross_model_sample_prevalence_df,
            ccrr_cross_model_stratum_prevalence_df=ccrr_cross_model_stratum_prevalence_df,
            ccrr_cross_model_label_prevalence_df=ccrr_cross_model_label_prevalence_df,
            ccrr_cross_model_slide_prevalence_df=ccrr_cross_model_slide_prevalence_df,
            ccrr_tail_overlap_jaccard_df=ccrr_tail_overlap_jaccard_df,
            ccrr_tail_overlap_summary_df=ccrr_tail_overlap_summary_df,
            ccrr_tail_overlap_always_df=ccrr_tail_overlap_always_df,
            ccrr_tail_overlap_unique_df=ccrr_tail_overlap_unique_df,
            ccrr_slide_enrichment_df=ccrr_slide_enrichment_df,
            ccrr_slide_summary_df=ccrr_slide_summary_df,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
