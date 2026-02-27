from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


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


def plot_ccrr_m_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ccrr_rows = [
        r for r in rows
        if "m" in r and "ccrr" in r and np.isfinite(float(r["m"])) and np.isfinite(float(r["ccrr"]))
    ]
    if not ccrr_rows:
        ax.set_visible(False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    by_model: dict[str, list[dict]] = {}
    for row in ccrr_rows:
        model = str(row["model"])
        by_model.setdefault(model, []).append(row)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r["m"]))

    m_ticks = sorted({int(row["m"]) for row in ccrr_rows})
    all_values = np.asarray([float(row["ccrr"]) for row in ccrr_rows], dtype=float)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.axhline(y=1.0, linestyle="--", linewidth=1.1, color="#6b7280", zorder=1, alpha=0.8)
    ax.set_xlabel("m", fontsize=11)
    ax.set_ylabel("CCRR", fontsize=11)
    ax.set_title("CCRR over m", fontsize=14, weight="bold")

    if len(m_ticks) <= 12:
        ax.set_xticks(m_ticks)
    else:
        tick_positions = np.linspace(float(min(m_ticks)), float(max(m_ticks)), num=8)
        tick_positions = np.unique(np.round(tick_positions).astype(int))
        ax.set_xticks(tick_positions.tolist())

    if len(m_ticks) > 1:
        span = float(max(m_ticks) - min(m_ticks))
        pad = max(0.5, 0.03 * span)
        ax.set_xlim(float(min(m_ticks)) - pad, float(max(m_ticks)) + pad)
    else:
        ax.set_xlim(float(m_ticks[0]) - 0.5, float(m_ticks[0]) + 0.5)

    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    if vmax - vmin <= 1e-9:
        pad = max(0.1, abs(vmin) * 0.10)
    else:
        pad = max(0.1, (vmax - vmin) * 0.10)
    ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    for model in sorted(by_model):
        model_rows = by_model[model]
        ms = np.asarray([int(r["m"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r["ccrr"]) for r in model_rows], dtype=float)
        ax.plot(
            ms,
            vals,
            color=_color_for_model(model),
            linewidth=1.8,
            alpha=0.95,
            marker="o",
            markersize=4,
            label=model,
        )

    ax.legend(frameon=False, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bio_vs_center_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_bio_vs_center_scatter(ax, rows, show_legend=True, legend_outside=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mari_vs_ri_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_mari_vs_ri_scatter(ax, rows, show_legend=True, legend_outside=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_ccrr_vs_mari_scatter(ax, rows: list[dict], *, show_legend: bool, legend_outside: bool) -> None:
    ccrr_rows = [r for r in rows if "ccrr" in r and np.isfinite(float(r["ccrr"]))]
    if not ccrr_rows:
        ax.set_visible(False)
        return

    xs = np.asarray([float(r["mari"]) for r in ccrr_rows], dtype=float)
    ys = np.asarray([float(r["ccrr"]) for r in ccrr_rows], dtype=float)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.axhline(y=1.0, linestyle="--", linewidth=1.1, color="#6b7280", zorder=1)

    for row in sorted(ccrr_rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["mari"])
        y = float(row["ccrr"])
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

    ax.set_xlim(*_padded_unit_interval_limits(xs))
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    ax.set_ylim(max(0.0, float(ys.min()) - y_pad), float(ys.max()) + y_pad)
    ax.set_xlabel("MaRI", fontsize=11)
    ax.set_ylabel("CCRR", fontsize=11)
    ax.set_title("CCRR vs MaRI", fontsize=14, weight="bold")

    if show_legend:
        if legend_outside:
            handles, labels = ax.get_legend_handles_labels()
            ax.figure.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
        else:
            ax.legend(frameon=False, loc="best")


def _draw_ccrr_sample_distributions(ax, rows: list[dict]) -> None:

    ccrr_rows = [
        r for r in rows
        if "ccrr_samples_path" in r and "ccrr_q_alpha" in r and np.isfinite(float(r.get("ccrr", float("nan"))))
    ]
    if not ccrr_rows:
        ax.set_visible(False)
        return

    model_data = []
    for row in sorted(ccrr_rows, key=lambda r: str(r["model"])):
        path = Path(str(row["ccrr_samples_path"]))
        if not path.exists():
            continue
        values = np.load(path)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        model_data.append({
            "model": str(row["model"]),
            "values": values,
            "q_alpha": float(row["ccrr_q_alpha"]),
            "alpha": float(row["ccrr_alpha"]),
            "ccrr": float(row["ccrr"]),
        })

    if not model_data:
        ax.set_visible(False)
        return

    all_values = np.concatenate([d["values"] for d in model_data])
    x_min = max(0.0, float(np.nanpercentile(all_values, 1)) - 0.1)
    x_max = float(np.nanpercentile(all_values, 99)) + 0.1
    x_grid = np.linspace(x_min, x_max, 512)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9, zorder=0)

    # Shade the fragile region (CCRR < 1.0)
    shade_right = min(1.0, x_max)
    if shade_right > x_min:
        ax.axvspan(x_min, shade_right, color="#f5e6d3", alpha=0.55, zorder=1)
    ax.axvline(x=1.0, linestyle="--", linewidth=1.1, color="#6b7280", zorder=2, alpha=0.75)

    for d in model_data:
        values = d["values"]
        color = _color_for_model(d["model"])

        if gaussian_kde is not None:
            try:
                density = gaussian_kde(values, bw_method="scott")(x_grid)
            except Exception:
                density = None
        else:
            density = None

        if density is None:
            counts, edges = np.histogram(values, bins=40, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centers, counts, color=color, linewidth=1.5, alpha=0.85,
                    label=f"{d['model']}  CCRR={d['ccrr']:.3f}")
        else:
            ax.plot(x_grid, density, color=color, linewidth=1.6, alpha=0.9,
                    label=f"{d['model']}  CCRR={d['ccrr']:.3f}")

        q = d["q_alpha"]
        if np.isfinite(q) and x_min <= q <= x_max:
            ax.axvline(x=q, color=color, linestyle=":", linewidth=1.0, alpha=0.85, zorder=3)

    alpha_pct = int(round(model_data[0]["alpha"] * 100))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("per-sample CCRR", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(
        f"Per-sample CCRR distributions  (dotted: $Q_{{{alpha_pct}}}$,  shaded: CCRR < 1)",
        fontsize=13, weight="bold",
    )


def plot_ccrr_sample_distributions(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    _draw_ccrr_sample_distributions(ax, rows)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.96, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ccrr_vs_mari_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_ccrr_vs_mari_scatter(ax, rows, show_legend=True, legend_outside=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_6panel_summary(*, rows: list[dict], k_sweep_rows: list[dict], out_path: Path) -> None:
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
