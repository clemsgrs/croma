"""compare_runs.py — compare RI/MaRI results between two benchmark runs.

Usage:
    python compare_runs.py \
        --run-a  <output-dir-for-run-a> \
        --run-b  <output-dir-for-run-b> \
        [--label-a <label>]       # default: inferred from dir name
        [--label-b <label>]       # default: inferred from dir name
        [--output-dir <dir>]      # default: current directory
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts" / "bench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plotting import (  # noqa: E402
    DEFAULT_DPI,
    REFERENCE_LINE_COLOR,
    TEXT_COLOR,
    _color_for_model,
    _style_axes,
)

from plotting import style as plotstyle  # noqa: E402

# Use the shared Arimo visual identity — the same style-setup every other figure
# uses — instead of a local serif rcParams override.
plotstyle.apply_style()

_JOIN_KEYS = ["dataset", "model", "confounder_column", "evaluation_design"]


def _infer_label(path: Path) -> str:
    """Derive a short human-readable label from a directory path."""
    return path.name or path.parent.name or str(path)


def _safe_filename(label: str) -> str:
    """Convert a label to a safe filename fragment."""
    return re.sub(r"[^\w\-]", "_", label).strip("_")


def _load_metrics(output_dir: Path) -> pd.DataFrame:
    csv = output_dir / "results/metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(f"No metrics.csv file found under {output_dir}")
    return pd.read_csv(csv)


def _join(df_a: pd.DataFrame, df_b: pd.DataFrame, *, label_b: str) -> pd.DataFrame:
    cols = _JOIN_KEYS + ["ri", "ri_undefined_frac", "mari", "mari_undefined_frac"]
    available_keys = [k for k in _JOIN_KEYS if k in df_a.columns and k in df_b.columns]
    a = df_a[[c for c in cols if c in df_a.columns]].copy()
    b = df_b[[c for c in cols if c in df_b.columns]].copy()
    suffix = f"_{_safe_filename(label_b)}"
    merged = a.merge(b, on=available_keys, suffixes=("", suffix))
    return merged


def _build_table(df: pd.DataFrame, *, label_a: str, label_b: str) -> pd.DataFrame:
    sfx = f"_{_safe_filename(label_b)}"
    rows = []
    for _, r in df.iterrows():
        delta_ri = float(r[f"ri{sfx}"]) - float(r["ri"])
        delta_mari = float(r[f"mari{sfx}"]) - float(r["mari"])
        rows.append(
            {
                "model": str(r["model"]),
                f"ri ({label_a})": round(float(r["ri"]), 4),
                f"ri ({label_b})": round(float(r[f"ri{sfx}"]), 4),
                "Δri": round(delta_ri, 4),
                f"mari ({label_a})": round(float(r["mari"]), 4),
                f"mari ({label_b})": round(float(r[f"mari{sfx}"]), 4),
                "Δmari": round(delta_mari, 4),
                f"undef% ({label_a})": round(float(r["ri_undefined_frac"]) * 100, 1),
                f"undef% ({label_b})": round(float(r[f"ri_undefined_frac{sfx}"]) * 100, 1),
            }
        )
    table = pd.DataFrame(rows)
    table["_sort"] = table["Δri"].abs() + table["Δmari"].abs()
    table = table.sort_values("_sort", ascending=False).drop(columns="_sort")
    return table.reset_index(drop=True)


def _print_table(table: pd.DataFrame) -> None:
    try:
        from tabulate import tabulate  # type: ignore

        print(tabulate(table, headers="keys", tablefmt="simple", showindex=False))
    except ImportError:
        print(table.to_string(index=False))


def _compute_correlations(df: pd.DataFrame, *, label_b: str) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr  # type: ignore

    sfx = f"_{_safe_filename(label_b)}"
    rows = []
    for metric in ("ri", "mari"):
        x = df[metric].to_numpy(dtype=float)
        y = df[f"{metric}{sfx}"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n >= 3:
            pr, pp = pearsonr(x[mask], y[mask])
            sr, sp = spearmanr(x[mask], y[mask])
            rows.append({
                "metric": metric,
                "pearson_r": round(float(pr), 4),
                "pearson_p": round(float(pp), 4),
                "spearman_r": round(float(sr), 4),
                "spearman_p": round(float(sp), 4),
                "n_models": n,
            })
        else:
            rows.append({
                "metric": metric,
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_r": float("nan"),
                "spearman_p": float("nan"),
                "n_models": n,
            })
    return pd.DataFrame(rows)


def _print_correlations(corr: pd.DataFrame, *, label_a: str, label_b: str) -> None:
    print(f"\nCorrelations ({label_a} vs {label_b}):")
    for _, row in corr.iterrows():
        n = int(row["n_models"])
        m = str(row["metric"])
        if n >= 3:
            print(
                f"  {m}: Pearson r={row['pearson_r']:.4f} (p={row['pearson_p']:.4f})  "
                f"Spearman r={row['spearman_r']:.4f} (p={row['spearman_p']:.4f})  n={n}"
            )
        else:
            print(f"  {m}: n/a  (need ≥3 models, got {n})")


def _scatter_plot(
    df: pd.DataFrame,
    *,
    metric: str,
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    sfx = f"_{_safe_filename(label_b)}"
    x_col = metric
    y_col = f"{metric}{sfx}"
    metric_label = metric.upper()

    vals = pd.concat([df[x_col], df[y_col]]).dropna()
    if vals.empty:
        return
    lo = float(vals.min())
    hi = float(vals.max())
    pad = (hi - lo) * 0.08 or 0.05
    lim = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    _style_axes(ax)

    ax.plot(lim, lim, color=REFERENCE_LINE_COLOR, linewidth=1.0, linestyle="--", zorder=1)

    for _, row in df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        model = str(row["model"])
        ax.scatter(
            [x], [y],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
            label=model,
        )
        ax.annotate(
            model,
            (x, y),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=7.5,
            color=TEXT_COLOR,
        )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"{metric_label} ({label_a})", fontsize=10.5, color=TEXT_COLOR)
    ax.set_ylabel(f"{metric_label} ({label_b})", fontsize=10.5, color=TEXT_COLOR)
    ax.set_title(
        f"{metric_label}: {label_a} vs {label_b}",
        fontsize=13, weight="semibold", pad=8, color=TEXT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  saved {out_path}")


def _rank_change_plot(
    df: pd.DataFrame,
    *,
    label_a: str,
    label_b: str,
    out_path: Path,
) -> None:
    """Slope / bump chart showing rank changes for RI and MaRI side by side."""
    sfx = f"_{_safe_filename(label_b)}"
    pairs = [("ri", "RI"), ("mari", "MaRI")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, max(5.0, len(df) * 0.55 + 2.0)))

    for ax, (metric, metric_label) in zip(axes, pairs):
        _style_axes(ax, grid_axis="y")

        x_col, xb_col = metric, f"{metric}{sfx}"
        valid = df[[x_col, xb_col, "model"]].dropna()
        if valid.empty:
            ax.set_visible(False)
            continue

        n = len(valid)
        ranks_a = valid[x_col].rank(ascending=False, method="min").astype(int)
        ranks_b = valid[xb_col].rank(ascending=False, method="min").astype(int)

        x_left, x_right = 0.0, 1.0
        label_pad = 0.08

        for i, row in valid.iterrows():
            model = str(row["model"])
            r_a = int(ranks_a[i])
            r_b = int(ranks_b[i])
            color = _color_for_model(model)
            delta = r_a - r_b  # positive = improved rank in run b

            lw = 1.6 + abs(delta) * 0.25
            ax.plot(
                [x_left, x_right], [r_a, r_b],
                color=color, linewidth=lw, solid_capstyle="round", zorder=2,
            )
            ax.scatter([x_left], [r_a], s=70, color=color, zorder=3, edgecolors="white", linewidths=0.8)
            ax.scatter([x_right], [r_b], s=70, color=color, zorder=3, edgecolors="white", linewidths=0.8)

            ax.text(
                x_left - label_pad, r_a, f"{r_a}. {model}",
                ha="right", va="center", fontsize=7.5, color=TEXT_COLOR,
            )
            delta_str = f"  ({'+' if delta > 0 else ''}{delta})" if delta != 0 else ""
            ax.text(
                x_right + label_pad, r_b, f"{r_b}. {model}{delta_str}",
                ha="left", va="center", fontsize=7.5, color=TEXT_COLOR,
            )

        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(n + 0.5, 0.5)
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(
            [f"{metric_label}\n({label_a})", f"{metric_label}\n({label_b})"],
            fontsize=10.0, color=TEXT_COLOR,
        )
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{metric_label} rank change", fontsize=12, weight="semibold", pad=8, color=TEXT_COLOR)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare RI/MaRI results between two benchmark runs."
    )
    parser.add_argument("--run-a", required=True, type=Path, help="Benchmark output dir for run A.")
    parser.add_argument("--run-b", required=True, type=Path, help="Benchmark output dir for run B.")
    parser.add_argument("--label-a", type=str, default=None, help="Label for run A (default: dir name).")
    parser.add_argument("--label-b", type=str, default=None, help="Label for run B (default: dir name).")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Where to write plots and CSV (default: cwd).")
    args = parser.parse_args()

    label_a = args.label_a or _infer_label(args.run_a)
    label_b = args.label_b or _infer_label(args.run_b)
    safe_a = _safe_filename(label_a)
    safe_b = _safe_filename(label_b)

    df_a = _load_metrics(args.run_a)
    df_b = _load_metrics(args.run_b)

    df = _join(df_a, df_b, label_b=label_b)
    if df.empty:
        print("No overlapping models found between the two runs.", file=sys.stderr)
        return 1

    table = _build_table(df, label_a=label_a, label_b=label_b)
    print(f"\n{'=' * 60}")
    print(f"RI / MaRI: {label_a} vs {label_b}")
    print(f"{'=' * 60}\n")
    _print_table(table)

    corr = _compute_correlations(df, label_b=label_b)
    _print_correlations(corr, label_a=label_a, label_b=label_b)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_path = out_dir / "correlations.csv"
    corr.to_csv(corr_path, index=False)
    print(f"\n  saved {corr_path}")

    print("\nPlots:")
    _scatter_plot(df, metric="ri",   label_a=label_a, label_b=label_b, out_path=out_dir / f"ri_{safe_a}_vs_{safe_b}.png")
    _scatter_plot(df, metric="mari", label_a=label_a, label_b=label_b, out_path=out_dir / f"mari_{safe_a}_vs_{safe_b}.png")
    _rank_change_plot(df, label_a=label_a, label_b=label_b, out_path=out_dir / "rank_change.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
