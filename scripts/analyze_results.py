import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
_THRESH_TAIL_GAP_Q = 0.15
_THRESH_TAIL_GAP_LTM = 0.20
_THRESH_K_SWEEP_RANGE = 0.15
_THRESH_M_SWEEP_CCRR_GAIN = 0.08
_THRESH_CCRR_RETRIES_HIGH = 10.0
_THRESH_CCRR_K_FINAL_HIGH = 10000.0


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
    df_k_sweep = _load_optional_csv(k_sweep_path)
    df_ccrr_m_sweep = _load_optional_csv(ccrr_m_sweep_path)
    k_sensitivity_df = _k_sweep_sensitivity(df_k_sweep)
    ccrr_m_sensitivity_df = _ccrr_m_sweep_sensitivity(df_ccrr_m_sweep)
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
