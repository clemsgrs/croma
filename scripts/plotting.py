from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import gaussian_kde

from croma.confounders import infer_confounder_display_name

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
    "Midnight-12k": "#17becf",
    "Hibou-L": "#e377c2",
    "Hibou-B": "#f7b6d2",
    "Prost40M": "#636363",
}

_SUPPORT_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "good": ("#4f8f76", "#dcebe4"),
    "warning": ("#b88a2a", "#f3e6c8"),
    "critical": ("#c9625b", "#f2d8d5"),
}


def _color_for_model(model: str) -> str:
    return str(MODEL_COLOR_MAP.get(str(model), "#808080"))


def _clamp_fraction(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _support_status(defined_frac: float) -> str:
    frac = _clamp_fraction(defined_frac)
    if frac < 0.25:
        return "critical"
    if frac < 0.50:
        return "warning"
    return "good"


def _support_plot_rows(rows: list[dict]) -> list[dict]:
    support_rows: list[dict] = []

    for raw_row in rows:
        model = str(raw_row.get("model", "")).strip()
        if not model:
            continue

        undefined_values: list[float] = []
        for key in ("ri_undefined_frac", "mari_undefined_frac"):
            if key not in raw_row:
                continue
            try:
                undefined_frac = float(raw_row[key])
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite(undefined_frac):
                continue
            undefined_values.append(_clamp_fraction(undefined_frac))

        if not undefined_values:
            continue

        undefined_frac = float(max(undefined_values))
        defined_frac = float(1.0 - undefined_frac)
        status = _support_status(defined_frac)
        fill_color, track_color = _SUPPORT_STATUS_COLORS[status]
        support_rows.append(
            {
                "model": model,
                "undefined_frac": undefined_frac,
                "defined_frac": defined_frac,
                "status": status,
                "fill_color": fill_color,
                "track_color": track_color,
                "label": f"{int(round(defined_frac * 100.0))}%",
            }
        )

    return sorted(support_rows, key=lambda row: (row["defined_frac"], row["model"]))


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


def _confounder_display_name(rows: list[dict]) -> str:
    if not rows:
        return "Confounder"
    for row in rows:
        display = str(row.get("confounder_display_name", "")).strip()
        if display:
            return display
        column = str(row.get("confounder_column", "")).strip()
        if column:
            return infer_confounder_display_name(column)
    return "Confounder"


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


def _highlight_selected(
    ax, *, ks: np.ndarray, ys: np.ndarray, selected_k: int, color: str, marker: str
) -> None:
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
            selected_k = int(
                model_rows[0].get(
                    selected_key, model_rows[0].get("selected_k", model_rows[0]["k"])
                )
            )
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
            ax.figure.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.98, 0.5),
                frameon=False,
            )
        else:
            ax.legend(frameon=False, loc="best")


def _draw_bio_vs_confounder_scatter(
    ax, rows: list[dict], *, show_legend: bool, legend_outside: bool
) -> None:
    if not rows:
        return

    confounder_display_name = _confounder_display_name(rows)
    xs = np.asarray([float(r["confounder_knn_bacc"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["bio_knn_bacc"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.plot(
        [0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#6b7280", zorder=1
    )

    for row in sorted(rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["confounder_knn_bacc"])
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
    ax.set_xlabel(f"{confounder_display_name} Accuracy", fontsize=11)
    ax.set_ylabel("Biological Accuracy", fontsize=11)
    ax.set_title(
        f"Biological vs {confounder_display_name} Accuracy", fontsize=14, weight="bold"
    )

    if show_legend:
        if legend_outside:
            handles, labels = ax.get_legend_handles_labels()
            ax.figure.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.98, 0.5),
                frameon=False,
            )
        else:
            ax.legend(frameon=False, loc="best")


def _draw_mari_vs_ri_scatter(
    ax, rows: list[dict], *, show_legend: bool, legend_outside: bool
) -> None:
    if not rows:
        return

    xs = np.asarray([float(r["ri"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["mari"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.plot(
        [0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#6b7280", zorder=1
    )

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
            ax.figure.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.98, 0.5),
                frameon=False,
            )
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


def plot_knn_confounder_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    confounder_display_name = _confounder_display_name(rows)
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="knn_confounder_bacc",
        ylabel="Balanced accuracy",
        title=f"{confounder_display_name} Accuracy over k",
        highlight_rules=[("selected_k_confounder", "X"), ("selected_k", "*")],
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


def plot_ri_mari_support(rows: list[dict], out_path: Path) -> None:
    support_rows = _support_plot_rows(rows)
    fig_height = max(3.4, 0.85 + 0.58 * max(len(support_rows), 1))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))

    if not support_rows:
        ax.set_visible(False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    y = np.arange(len(support_rows), dtype=float)
    labels = [str(row["model"]) for row in support_rows]

    ax.set_facecolor("#fbfcfd")
    fig.patch.set_facecolor("#fbfcfd")
    ax.grid(axis="x", color="#d9dee5", linewidth=0.8, alpha=0.9, zorder=0)

    for idx, row in enumerate(support_rows):
        defined_frac = float(row["defined_frac"])
        undefined_frac = float(row["undefined_frac"])
        fill_color = str(row["fill_color"])
        track_color = str(row["track_color"])

        ax.barh(
            y[idx],
            1.0,
            color=track_color,
            edgecolor="none",
            height=0.58,
            zorder=1,
        )
        ax.barh(
            y[idx],
            defined_frac,
            color=fill_color,
            edgecolor="none",
            height=0.58,
            zorder=2,
        )

        ax.text(
            defined_frac / 2.0,
            y[idx],
            str(row["label"]),
            va="center",
            ha="center",
            fontsize=10,
            color="white",
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of evaluated samples", fontsize=11, labelpad=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Support Coverage", fontsize=13, weight="bold", pad=6)
    ax.set_ylim(float(len(support_rows) - 0.35), -0.65)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        Patch(facecolor=_SUPPORT_STATUS_COLORS["critical"][0], label="Defined <25%"),
        Patch(facecolor=_SUPPORT_STATUS_COLORS["warning"][0], label="Defined <50%"),
        Patch(facecolor=_SUPPORT_STATUS_COLORS["good"][0], label="Defined >=50%"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
        ncol=3,
        fontsize=9.5,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.985))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ccmr_m_sweep_with_ltm(rows: list[dict], out_path: Path) -> None:
    fig, (ax_ccmr, ax_ltm) = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    ccmr_rows = [
        r
        for r in rows
        if "m" in r
        and "ccmr" in r
        and np.isfinite(float(r["m"]))
        and np.isfinite(float(r["ccmr"]))
    ]
    if not ccmr_rows:
        for ax in (ax_ccmr, ax_ltm):
            ax.set_visible(False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    by_model: dict[str, list[dict]] = {}
    for row in ccmr_rows:
        model = str(row["model"])
        by_model.setdefault(model, []).append(row)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r["m"]))

    m_all = sorted({int(row["m"]) for row in ccmr_rows})
    m_min, m_max = m_all[0], m_all[-1]

    def _configure_ax(
        ax: plt.Axes, ylabel: str, title: str, values: np.ndarray
    ) -> None:
        ax.set_facecolor("#fbfcfd")
        ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
        ax.axhline(
            y=1.0, linestyle="--", linewidth=1.1, color="#6b7280", zorder=1, alpha=0.8
        )
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, weight="bold")
        finite = values[np.isfinite(values)]
        vmin = float(np.nanmin(finite)) if finite.size > 0 else 0.0
        vmax = float(np.nanmax(finite)) if finite.size > 0 else 1.0
        span = vmax - vmin
        pad = max(0.05, span * 0.10) if span > 1e-9 else max(0.1, abs(vmin) * 0.10)
        ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    ccmr_values = np.asarray([float(r["ccmr"]) for r in ccmr_rows], dtype=float)
    ltm_values = np.asarray(
        [float(r["ccmr_ltm_alpha"]) for r in ccmr_rows if "ccmr_ltm_alpha" in r],
        dtype=float,
    )
    _configure_ax(ax_ccmr, "CCMR", "CCMR over m", ccmr_values)
    if ltm_values.size > 0:
        _configure_ax(ax_ltm, "LTM", "LTM over m", ltm_values)

    for model in sorted(by_model):
        model_rows = by_model[model]
        color = _color_for_model(model)
        ms = np.asarray([int(r["m"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r["ccmr"]) for r in model_rows], dtype=float)
        ax_ccmr.plot(ms, vals, color=color, linewidth=1.8, alpha=0.95, label=model)
        if "ccmr_ltm_alpha" in model_rows[0]:
            ltms = np.asarray(
                [float(r["ccmr_ltm_alpha"]) for r in model_rows], dtype=float
            )
            ax_ltm.plot(ms, ltms, color=color, linewidth=1.8, alpha=0.95, label=model)

    # x-axis: integer ticks spanning the full sweep range
    if m_max - m_min <= 20:
        tick_positions = list(range(m_min, m_max + 1))
    else:
        step = max(1, (m_max - m_min) // 10)
        tick_positions = list(range(m_min, m_max + 1, step))
        if m_max not in tick_positions:
            tick_positions.append(m_max)
    ax_ltm.set_xticks(tick_positions)
    ax_ltm.set_xlim(m_min - 0.5, m_max + 0.5)
    ax_ltm.set_xlabel("m", fontsize=11)

    ax_ccmr.legend(frameon=False, loc="best", fontsize=9)
    ax_ltm.legend(frameon=False, loc="best", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _valid_ccmr_ltm_rows(rows: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for row in rows:
        if "ccmr" not in row or "ccmr_ltm_alpha" not in row:
            continue
        try:
            ccmr_value = float(row["ccmr"])
            ltm_value = float(row["ccmr_ltm_alpha"])
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(ccmr_value) or not np.isfinite(ltm_value):
            continue

        try:
            alpha_value = float(row.get("ccmr_alpha", float("nan")))
        except Exception:  # noqa: BLE001
            alpha_value = float("nan")

        valid.append(
            {
                "model": str(row.get("model", "")),
                "ccmr": ccmr_value,
                "ltm": ltm_value,
                "alpha": alpha_value if np.isfinite(alpha_value) else float("nan"),
            }
        )
    return valid


def _ltm_label(valid_rows: list[dict]) -> str:
    alpha_values = sorted(
        {float(r["alpha"]) for r in valid_rows if np.isfinite(float(r["alpha"]))}
    )
    if len(alpha_values) == 1:
        alpha_pct = int(round(alpha_values[0] * 100))
        return f"LTM@{alpha_pct}%"
    return "LTM(CCMR)"


def _padded_positive_limits(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    span = float(vmax - vmin)
    if span <= 1e-9:
        pad = max(0.1, abs(vmin) * 0.10, 0.05)
    else:
        pad = max(0.1, span * 0.10)
    lo = max(0.0, vmin - pad)
    hi = vmax + pad
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def plot_ccmr_ltm_comparison(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    scatter_ax, bar_ax = axes
    valid_rows = _valid_ccmr_ltm_rows(rows)

    if not valid_rows:
        scatter_ax.set_visible(False)
        bar_ax.set_visible(False)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    label_ltm = _ltm_label(valid_rows)

    # Left: CCMR vs LTM scatter.
    xs = np.asarray([float(r["ccmr"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["ltm"]) for r in valid_rows], dtype=float)
    lim_lo, lim_hi = _padded_positive_limits(np.concatenate([xs, ys]))

    scatter_ax.set_facecolor("#fbfcfd")
    scatter_ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    scatter_ax.plot(
        [lim_lo, lim_hi],
        [lim_lo, lim_hi],
        linestyle="--",
        linewidth=1.1,
        color="#6b7280",
        zorder=1,
    )
    for row in sorted(valid_rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        scatter_ax.scatter(
            [float(row["ccmr"])],
            [float(row["ltm"])],
            s=90,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )
    scatter_ax.set_xlim(lim_lo, lim_hi)
    scatter_ax.set_ylim(lim_lo, lim_hi)
    scatter_ax.set_xlabel("CCMR", fontsize=11)
    scatter_ax.set_ylabel(label_ltm, fontsize=11)
    scatter_ax.set_title(f"CCMR vs {label_ltm}", fontsize=13, weight="bold")

    # Right: sorted CCMR/LTM bars to compare rank and tail-gap by model.
    ranked_rows = sorted(
        valid_rows, key=lambda r: (float(r["ltm"]), str(r["model"])), reverse=True
    )
    model_names = [str(r["model"]) for r in ranked_rows]
    ccmr_vals = np.asarray([float(r["ccmr"]) for r in ranked_rows], dtype=float)
    ltm_vals = np.asarray([float(r["ltm"]) for r in ranked_rows], dtype=float)
    colors = [_color_for_model(model) for model in model_names]
    x = np.arange(len(ranked_rows), dtype=float)
    width = 0.38

    bar_ax.set_facecolor("#fbfcfd")
    bar_ax.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.9, zorder=0)
    bar_ax.axhline(
        y=1.0, linestyle="--", linewidth=1.1, color="#6b7280", zorder=1, alpha=0.75
    )
    bar_ax.bar(
        x - width / 2.0,
        ccmr_vals,
        width=width,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
        label="CCMR",
        zorder=3,
    )
    bar_ax.bar(
        x + width / 2.0,
        ltm_vals,
        width=width,
        color=colors,
        alpha=0.45,
        edgecolor="white",
        linewidth=0.6,
        label=label_ltm,
        zorder=3,
    )
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(model_names, rotation=35, ha="right")
    bar_ax.set_ylabel("score", fontsize=11)
    bar_ax.set_title(f"Sorted by {label_ltm}", fontsize=13, weight="bold")
    bar_ax.legend(frameon=False, loc="best")
    y_lo, y_hi = _padded_positive_limits(np.concatenate([ccmr_vals, ltm_vals]))
    bar_ax.set_ylim(y_lo, y_hi)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bio_vs_confounder_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_bio_vs_confounder_scatter(ax, rows, show_legend=True, legend_outside=False)
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


def _draw_ccmr_vs_mari_scatter(
    ax, rows: list[dict], *, show_legend: bool, legend_outside: bool
) -> None:
    ccmr_rows = [r for r in rows if "ccmr" in r and np.isfinite(float(r["ccmr"]))]
    if not ccmr_rows:
        ax.set_visible(False)
        return

    xs = np.asarray([float(r["mari"]) for r in ccmr_rows], dtype=float)
    ys = np.asarray([float(r["ccmr"]) for r in ccmr_rows], dtype=float)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9)
    ax.axhline(y=1.0, linestyle="--", linewidth=1.1, color="#6b7280", zorder=1)

    for row in sorted(ccmr_rows, key=lambda r: str(r["model"])):
        model = str(row["model"])
        x = float(row["mari"])
        y = float(row["ccmr"])
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
    ax.set_ylabel("CCMR", fontsize=11)
    ax.set_title("CCMR vs MaRI", fontsize=14, weight="bold")

    if show_legend:
        if legend_outside:
            handles, labels = ax.get_legend_handles_labels()
            ax.figure.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(0.98, 0.5),
                frameon=False,
            )
        else:
            ax.legend(frameon=False, loc="best")


def _draw_ccmr_sample_distributions(ax, rows: list[dict]) -> None:

    ccmr_rows = [
        r
        for r in rows
        if "ccmr_samples_path" in r
        and "ccmr_q_alpha" in r
        and np.isfinite(float(r.get("ccmr", float("nan"))))
    ]
    if not ccmr_rows:
        ax.set_visible(False)
        return

    model_data = []
    for row in sorted(ccmr_rows, key=lambda r: str(r["model"])):
        path = Path(str(row["ccmr_samples_path"]))
        if not path.exists():
            continue
        values = np.load(path)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        model_data.append(
            {
                "model": str(row["model"]),
                "values": values,
                "q_alpha": float(row["ccmr_q_alpha"]),
                "alpha": float(row["ccmr_alpha"]),
                "ccmr": float(row["ccmr"]),
            }
        )

    if not model_data:
        ax.set_visible(False)
        return

    all_values = np.concatenate([d["values"] for d in model_data])
    x_min = max(0.0, float(np.nanpercentile(all_values, 1)) - 0.1)
    x_max = float(np.nanpercentile(all_values, 99)) + 0.1
    x_grid = np.linspace(x_min, x_max, 512)

    ax.set_facecolor("#fbfcfd")
    ax.grid(color="#d9dee5", linewidth=0.8, alpha=0.9, zorder=0)

    # Shade the fragile region (CCMR < 1.0)
    shade_right = min(1.0, x_max)
    if shade_right > x_min:
        ax.axvspan(x_min, shade_right, color="#f5e6d3", alpha=0.55, zorder=1)
    ax.axvline(
        x=1.0, linestyle="--", linewidth=1.1, color="#6b7280", zorder=2, alpha=0.75
    )

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
            ax.step(
                centers,
                counts,
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=f"{d['model']}  CCMR={d['ccmr']:.3f}",
            )
        else:
            ax.plot(
                x_grid,
                density,
                color=color,
                linewidth=1.6,
                alpha=0.9,
                label=f"{d['model']}  CCMR={d['ccmr']:.3f}",
            )

        q = d["q_alpha"]
        if np.isfinite(q) and x_min <= q <= x_max:
            ax.axvline(
                x=q, color=color, linestyle=":", linewidth=1.0, alpha=0.85, zorder=3
            )

    alpha_pct = int(round(model_data[0]["alpha"] * 100))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("per-sample CCMR", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(
        f"Per-sample CCMR distributions  (dotted: $Q_{{{alpha_pct}}}$,  shaded: CCMR < 1)",
        fontsize=13,
        weight="bold",
    )


def plot_ccmr_sample_distributions(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    _draw_ccmr_sample_distributions(ax, rows)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            frameon=False,
            fontsize=9,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.96, 1.0))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ccmr_vs_mari_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    _draw_ccmr_vs_mari_scatter(ax, rows, show_legend=True, legend_outside=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
