"""compare_pruned.py — compare RI/MaRI results with and without --prune-ss-oo.

Usage:
    python compare_pruned.py \
        --baseline <output-dir-without-prune-ss-oo> \
        --pruned   <output-dir-with-prune-ss-oo> \
        [--output-dir <dir-for-plots-and-csv>]  # default: current directory
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plotting import (  # noqa: E402
    DEFAULT_DPI,
    REFERENCE_LINE_COLOR,
    TEXT_COLOR,
    _color_for_model,
    _style_axes,
)

matplotlib.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "dejavuserif",
    }
)

_JOIN_KEYS = ["dataset", "model", "confounder_column", "evaluation_design"]


def _load_metrics(output_dir: Path) -> pd.DataFrame:
    csvs = sorted(output_dir.glob("*/results/metrics.csv"))
    if not csvs:
        raise FileNotFoundError(f"No metrics.csv files found under {output_dir}")
    return pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)


def _join(baseline: pd.DataFrame, pruned: pd.DataFrame) -> pd.DataFrame:
    cols = _JOIN_KEYS + ["ri", "ri_undefined_frac", "mari", "mari_undefined_frac"]
    available_keys = [k for k in _JOIN_KEYS if k in baseline.columns and k in pruned.columns]
    b = baseline[[c for c in cols if c in baseline.columns]].copy()
    p = pruned[[c for c in cols if c in pruned.columns]].copy()
    merged = b.merge(p, on=available_keys, suffixes=("", "_pruned"))
    return merged


def _build_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        delta_ri = float(r["ri_pruned"]) - float(r["ri"])
        delta_mari = float(r["mari_pruned"]) - float(r["mari"])
        rows.append(
            {
                "model": str(r["model"]),
                "ri": round(float(r["ri"]), 4),
                "ri_pruned": round(float(r["ri_pruned"]), 4),
                "Δri": round(delta_ri, 4),
                "mari": round(float(r["mari"]), 4),
                "mari_pruned": round(float(r["mari_pruned"]), 4),
                "Δmari": round(delta_mari, 4),
                "undef%": round(float(r["ri_undefined_frac"]) * 100, 1),
                "undef%_pruned": round(float(r["ri_undefined_frac_pruned"]) * 100, 1),
            }
        )
    table = pd.DataFrame(rows)
    table["_sort"] = (table["Δri"].abs() + table["Δmari"].abs())
    table = table.sort_values("_sort", ascending=False).drop(columns="_sort")
    return table.reset_index(drop=True)


def _print_table(table: pd.DataFrame) -> None:
    try:
        from tabulate import tabulate  # type: ignore

        print(tabulate(table, headers="keys", tablefmt="simple", showindex=False))
    except ImportError:
        print(table.to_string(index=False))


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import pearsonr  # type: ignore

    rows = []
    for metric in ("ri", "mari"):
        x = df[metric].to_numpy(dtype=float)
        y = df[f"{metric}_pruned"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n >= 3:
            r, p = pearsonr(x[mask], y[mask])
            rows.append({"metric": metric, "r": round(float(r), 4), "p_value": round(float(p), 4), "n_models": n})
        else:
            rows.append({"metric": metric, "r": float("nan"), "p_value": float("nan"), "n_models": n})
    return pd.DataFrame(rows)


def _print_correlations(corr: pd.DataFrame) -> None:
    print("\nPearson correlations (unpruned vs pruned):")
    for _, row in corr.iterrows():
        n = int(row["n_models"])
        if n >= 3:
            print(f"  r({row['metric']}, {row['metric']}_pruned) = {row['r']:.4f}  (p={row['p_value']:.4f},  n={n})")
        else:
            print(f"  r({row['metric']}, {row['metric']}_pruned) = n/a  (need ≥3 models, got {n})")


def _scatter_plot(
    df: pd.DataFrame,
    *,
    metric: str,
    out_path: Path,
) -> None:
    x_col = metric
    y_col = f"{metric}_pruned"
    label = metric.upper()

    vals = pd.concat([df[x_col], df[y_col]]).dropna()
    if vals.empty:
        return
    lo = float(vals.min())
    hi = float(vals.max())
    pad = (hi - lo) * 0.08 or 0.05
    lim = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(6.0, 5.8))
    _style_axes(ax)

    # y=x reference line
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
    ax.set_xlabel(f"{label} (unpruned)", fontsize=10.5, color=TEXT_COLOR)
    ax.set_ylabel(f"{label} (pruned)", fontsize=10.5, color=TEXT_COLOR)
    ax.set_title(f"{label}: pruned vs unpruned", fontsize=13, weight="semibold", pad=8, color=TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  saved {out_path}")


def _rank_change_plot(df: pd.DataFrame, *, out_path: Path) -> None:
    """Slope / bump chart showing rank changes for RI and MaRI side by side.

    Rank 1 = highest (best) value. Lines slope upward when a model improves
    its rank after pruning, downward when it drops.
    """
    pairs = [("ri", "RI"), ("mari", "MaRI")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, max(5.0, len(df) * 0.55 + 2.0)))

    for ax, (metric, label) in zip(axes, pairs):
        _style_axes(ax, grid_axis="y")

        x_col, xp_col = metric, f"{metric}_pruned"
        valid = df[[x_col, xp_col, "model"]].dropna()
        if valid.empty:
            ax.set_visible(False)
            continue

        n = len(valid)
        # rank ascending=False so rank 1 = best (highest value)
        ranks_base   = valid[x_col].rank(ascending=False, method="min").astype(int)
        ranks_pruned = valid[xp_col].rank(ascending=False, method="min").astype(int)

        x_left, x_right = 0.0, 1.0
        label_pad = 0.08

        for i, row in valid.iterrows():
            model = str(row["model"])
            r_base   = int(ranks_base[i])
            r_pruned = int(ranks_pruned[i])
            color = _color_for_model(model)
            delta = r_base - r_pruned  # positive = improved rank

            lw = 1.6 + abs(delta) * 0.25
            ax.plot(
                [x_left, x_right], [r_base, r_pruned],
                color=color, linewidth=lw, solid_capstyle="round", zorder=2,
            )
            ax.scatter([x_left], [r_base],   s=70, color=color, zorder=3, edgecolors="white", linewidths=0.8)
            ax.scatter([x_right], [r_pruned], s=70, color=color, zorder=3, edgecolors="white", linewidths=0.8)

            # left label
            ax.text(
                x_left - label_pad, r_base, f"{r_base}. {model}",
                ha="right", va="center", fontsize=7.5, color=TEXT_COLOR,
            )
            # right label (show Δrank if non-zero)
            delta_str = f"  ({'+' if delta > 0 else ''}{delta})" if delta != 0 else ""
            ax.text(
                x_right + label_pad, r_pruned, f"{r_pruned}. {model}{delta_str}",
                ha="left", va="center", fontsize=7.5, color=TEXT_COLOR,
            )

        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(n + 0.5, 0.5)          # rank 1 at top
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels([label, f"{label}\n(pruned)"], fontsize=10.0, color=TEXT_COLOR)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{label} rank change", fontsize=12, weight="semibold", pad=8, color=TEXT_COLOR)

    fig.tight_layout(pad=2.0)
    fig.savefig(out_path, dpi=DEFAULT_DPI)
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare RI/MaRI results with and without --prune-ss-oo."
    )
    parser.add_argument("--baseline", required=True, type=Path, help="Benchmark output dir (no pruning).")
    parser.add_argument("--pruned", required=True, type=Path, help="Benchmark output dir (with --prune-ss-oo).")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Where to write plots and CSV (default: cwd).")
    args = parser.parse_args()

    baseline_df = _load_metrics(args.baseline)
    pruned_df = _load_metrics(args.pruned)

    df = _join(baseline_df, pruned_df)
    if df.empty:
        print("No overlapping models found between the two runs.", file=sys.stderr)
        return 1

    table = _build_table(df)
    print(f"\n{'=' * 60}")
    print("RI / MaRI: pruned vs unpruned")
    print(f"{'=' * 60}\n")
    _print_table(table)

    corr = _compute_correlations(df)
    _print_correlations(corr)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_path = out_dir / "correlations.csv"
    corr.to_csv(corr_path, index=False)
    print(f"\n  saved {corr_path}")

    print("\nPlots:")
    _scatter_plot(df, metric="ri", out_path=out_dir / "ri_pruned_vs_unpruned.png")
    _scatter_plot(df, metric="mari", out_path=out_dir / "mari_pruned_vs_unpruned.png")
    _rank_change_plot(df, out_path=out_dir / "rank_change.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
