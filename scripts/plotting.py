from __future__ import annotations

from pathlib import Path

import numpy as np


# Copied from: /Volumes/temporary/clement/discern/data/eval/prostate-shift-binary/benchmark.py
MODEL_COLOR_MAP: dict[str, str] = {
    "Virchow2": "#ff7f0e",
    "Virchow": "#ffbb78",
    "UNI2-h": "#2ca02c",
    "UNI": "#98df8a",
    "CONCHv1.5": "#d62728",
    "CONCH": "#ff9896",
    "Phikon-v2": "#9467bd",
    "Phikon": "#c5b0d5",
    "H-optimus-1": "#8c564b",
    "H-optimus-0": "#c49c94",
    "H0-mini": "#d7b5b0",
    "Prov-GigaPath": "#1f77b4",
    "Midnight": "#17becf",
    "Hibou-L": "#e377c2",
    "Hibou-B": "#f7b6d2",
    "Prost40M": "#636363",
}


def _color_for_model(model: str) -> str:
    return str(MODEL_COLOR_MAP.get(str(model), "#808080"))


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _padded_unit_interval_limits(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 1.0

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    span = float(vmax - vmin)

    if span <= 1e-9:
        base = max(0.02, abs(vmin) * 0.10)
        lo = max(0.0, vmin - base)
        hi = min(1.0, vmax + base)
        if hi <= lo:
            lo, hi = max(0.0, vmin - 0.05), min(1.0, vmin + 0.05)
        return float(lo), float(hi)

    pad = max(0.02, span * 0.10)
    lo = max(0.0, vmin - pad)
    hi = min(1.0, vmax + pad)
    if hi - lo < 0.05:
        mid = 0.5 * (lo + hi)
        lo = max(0.0, mid - 0.025)
        hi = min(1.0, mid + 0.025)
    return float(lo), float(hi)


def _group_k_rows(rows: list[dict]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        model = str(row["model"])
        by_model.setdefault(model, []).append(row)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r["k"]))
    return by_model


def _set_k_axis(ax, k_ticks: list[int]) -> None:
    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.set_xlabel("k", fontsize=11)
    if len(k_ticks) <= 12:
        ax.set_xticks(k_ticks)
    else:
        tick_positions = np.linspace(float(min(k_ticks)), float(max(k_ticks)), num=8)
        tick_positions = np.unique(np.round(tick_positions).astype(int))
        ax.set_xticks(tick_positions.tolist())

    if len(k_ticks) > 1:
        span = float(max(k_ticks) - min(k_ticks))
        pad = max(0.5, 0.03 * span)
        ax.set_xlim(float(min(k_ticks)) - pad, float(max(k_ticks)) + pad)
    else:
        ax.set_xlim(float(k_ticks[0]) - 0.5, float(k_ticks[0]) + 0.5)


def _highlight_selected(ax, *, ks: np.ndarray, ys: np.ndarray, selected_k: int, color: str, marker: str) -> None:
    idx_arr = np.where(ks == int(selected_k))[0]
    if int(idx_arr.size) == 0:
        return
    idx = int(idx_arr[0])
    size = 220 if marker == "*" else 190
    ax.scatter(
        [ks[idx]],
        [ys[idx]],
        s=size,
        color=color,
        marker=marker,
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
    )


def _draw_k_curve(
    ax,
    *,
    rows: list[dict],
    value_key: str,
    ylabel: str,
    title: str,
    highlight_rules: list[tuple[str, str]],
    show_legend: bool,
    legend_outside: bool,
) -> None:
    if not rows:
        return
    by_model = _group_k_rows(rows)
    models = sorted(by_model)
    k_ticks = sorted({int(row["k"]) for row in rows})
    all_values = np.asarray([float(row[value_key]) for row in rows], dtype=float)

    _set_k_axis(ax, k_ticks)
    ax.set_ylim(*_padded_unit_interval_limits(all_values))
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, weight="bold")

    for model in models:
        model_rows = by_model[model]
        ks = np.asarray([int(r["k"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r[value_key]) for r in model_rows], dtype=float)
        color = _color_for_model(model)
        ax.plot(
            ks,
            vals,
            color=color,
            linewidth=1.8,
            alpha=0.95,
            marker=None,
            label=model,
        )
        for selected_key, marker in highlight_rules:
            selected_k = int(model_rows[0].get(selected_key, model_rows[0].get("selected_k", model_rows[0]["k"])))
            _highlight_selected(
                ax,
                ks=ks,
                ys=vals,
                selected_k=selected_k,
                color=color,
                marker=marker,
            )

    if show_legend:
        if legend_outside:
            handles, labels = ax.get_legend_handles_labels()
            ax.figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
        else:
            ax.legend(frameon=False, loc="best")


def _draw_bio_vs_center_scatter(ax, rows: list[dict], *, show_legend: bool, legend_outside: bool) -> None:
    if not rows:
        return

    xs = np.asarray([float(r["center_knn_bacc"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["bio_knn_bacc"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#6b7280", zorder=1)

    for row in sorted(rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["center_knn_bacc"])
        y = float(row["bio_knn_bacc"])
        ax.scatter(
            [x],
            [y],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
            label=model,
        )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Medical Center Accuracy", fontsize=11)
    ax.set_ylabel("Biological Accuracy", fontsize=11)
    ax.set_title("Biological vs Medical Center Accuracy", fontsize=14, weight="bold")

    if show_legend:
        if legend_outside:
            handles, labels = ax.get_legend_handles_labels()
            ax.figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
        else:
            ax.legend(frameon=False, loc="best")


def _draw_mari_vs_ri_scatter(ax, rows: list[dict], *, show_legend: bool, legend_outside: bool) -> None:
    if not rows:
        return

    xs = np.asarray([float(r["ri"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["mari"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#6b7280", zorder=1)

    for row in sorted(rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["ri"])
        y = float(row["mari"])
        ax.scatter(
            [x],
            [y],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
            label=model,
        )

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("RI", fontsize=11)
    ax.set_ylabel("MaRI", fontsize=11)
    ax.set_title("MaRI vs RI", fontsize=14, weight="bold")

    if show_legend:
        if legend_outside:
            handles, labels = ax.get_legend_handles_labels()
            ax.figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
        else:
            ax.legend(frameon=False, loc="best")


def plot_knn_bio_k_sweep(rows: list[dict], out_path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="knn_bacc",
        ylabel="Balanced accuracy",
        title="Biological Accuracy over k",
        highlight_rules=[("selected_k", "*")],
        show_legend=True,
        legend_outside=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_knn_center_k_sweep(rows: list[dict], out_path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="knn_center_bacc",
        ylabel="Balanced accuracy",
        title="Medical Center Accuracy over k",
        highlight_rules=[("selected_k_center", "X"), ("selected_k", "*")],
        show_legend=True,
        legend_outside=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ri_k_sweep(rows: list[dict], out_path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="ri",
        ylabel="RI",
        title="RI over k",
        highlight_rules=[("selected_k", "*")],
        show_legend=True,
        legend_outside=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mari_k_sweep(rows: list[dict], out_path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="mari",
        ylabel="MaRI",
        title="MaRI over k",
        highlight_rules=[("selected_k", "*")],
        show_legend=True,
        legend_outside=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.94, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bio_vs_center_scatter(rows: list[dict], out_path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_bio_vs_center_scatter(ax, rows, show_legend=True, legend_outside=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mari_vs_ri_scatter(rows: list[dict], out_path: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_mari_vs_ri_scatter(ax, rows, show_legend=True, legend_outside=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_6panel_summary(*, rows: list[dict], k_sweep_rows: list[dict], out_path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(3, 2, figsize=(15.0, 16.0))

    _draw_k_curve(
        axes[0, 0],
        rows=k_sweep_rows,
        value_key="knn_bacc",
        ylabel="Balanced accuracy",
        title="Biological Accuracy over k",
        highlight_rules=[("selected_k", "*")],
        show_legend=False,
        legend_outside=False,
    )
    _draw_k_curve(
        axes[0, 1],
        rows=k_sweep_rows,
        value_key="knn_center_bacc",
        ylabel="Balanced accuracy",
        title="Medical Center Accuracy over k",
        highlight_rules=[("selected_k_center", "X"), ("selected_k", "*")],
        show_legend=False,
        legend_outside=False,
    )
    _draw_k_curve(
        axes[1, 0],
        rows=k_sweep_rows,
        value_key="ri",
        ylabel="RI",
        title="RI over k",
        highlight_rules=[("selected_k", "*")],
        show_legend=False,
        legend_outside=False,
    )
    _draw_k_curve(
        axes[1, 1],
        rows=k_sweep_rows,
        value_key="mari",
        ylabel="MaRI",
        title="MaRI over k",
        highlight_rules=[("selected_k", "*")],
        show_legend=False,
        legend_outside=False,
    )
    _draw_bio_vs_center_scatter(axes[2, 0], rows, show_legend=False, legend_outside=False)
    _draw_mari_vs_ri_scatter(axes[2, 1], rows, show_legend=False, legend_outside=False)

    from matplotlib.lines import Line2D

    model_names = sorted({str(r["model"]) for r in k_sweep_rows} | {str(r["model"]) for r in rows})
    handles = [
        Line2D([0], [0], color=_color_for_model(model), lw=2.0, label=model)
        for model in model_names
    ]
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(4, len(handles)), frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
