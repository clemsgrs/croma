import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from croma.confounders import infer_confounder_display_name

from croma import plotstyle
from croma.plotstyle import (
    CANONICAL_MODEL_ORDER,
    COL_DOUBLE,
    COL_ONEHALF,
    COL_SINGLE,
    DEFAULT_DPI,
    FAMILY_PALETTE,
    FRAGILE_SHADE_COLOR,
    GRID_COLOR,
    METRIC_COLOR,
    MODEL_FAMILY_MAP,
    MODEL_TONE_INDEX,
    MUTED_TEXT_COLOR,
    PANEL_FACE_COLOR,
    PREC_METRIC,
    PREC_PERCENT,
    REFERENCE_LINE_COLOR,
    SPINE_COLOR,
    TEXT_COLOR,
    metric_label,
    model_sort_key,
)

# plotstyle.apply_style() runs on import; re-apply defensively in case another
# module reset rcParams after import.
plotstyle.apply_style()

LEGEND_Y = 0.02
LEGEND_MAX_COLUMNS = 4

_SUPPORT_STATUS_COLORS = plotstyle.SUPPORT_STATUS_COLORS


def _family_for_model(model: str) -> str:
    return plotstyle.family_for_model(model)


def _color_for_model(model: str) -> str:
    return plotstyle.color_for_model(model)



def _style_axes(ax, *, grid_axis: str = "both") -> None:
    plotstyle.style_axes(ax, grid_axis=grid_axis)


def _set_panel_title(ax, title: str) -> None:
    plotstyle.set_panel_title(ax, title)


def _legend_columns(n_labels: int) -> int:
    if n_labels <= 1:
        return 1
    return min(LEGEND_MAX_COLUMNS, max(2, n_labels))


def _collect_legend_entries(axes: list[plt.Axes]) -> tuple[list, list[str]]:
    handles: list = []
    labels: list[str] = []
    seen: set[str] = set()
    for ax in axes:
        if not ax.get_visible():
            continue
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            text = str(label).strip()
            if not text or text.startswith("_") or text in seen:
                continue
            seen.add(text)
            handles.append(handle)
            labels.append(text)
    return handles, labels


def _add_figure_legend(
    fig,
    axes: list[plt.Axes],
    *,
    y: float = LEGEND_Y,
    ncol: int | None = None,
    fontsize: float = 9.0,
    columnspacing: float = 1.4,
    handlelength: float = 2.2,
) -> bool:
    handles, labels = _collect_legend_entries(axes)
    if not handles:
        return False
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=int(ncol) if ncol is not None else _legend_columns(len(labels)),
        frameon=False,
        fontsize=fontsize,
        columnspacing=columnspacing,
        handlelength=handlelength,
        handletextpad=0.6,
    )
    return True


def _legend_bottom_margin(axes: list[plt.Axes]) -> float:
    _handles, labels = _collect_legend_entries(axes)
    if not labels:
        return 0.12
    ncols = _legend_columns(len(labels))
    nrows = int(math.ceil(len(labels) / float(ncols)))
    return min(0.095 + 0.042 * nrows, 0.255)


def _png_export_path(out_path: Path) -> Path:
    return out_path.parent / "png" / out_path.name


def _pdf_export_path(out_path: Path) -> Path:
    return out_path.parent / "pdf" / out_path.with_suffix(".pdf").name


def _finalize_figure(
    fig,
    *,
    out_path: Path,
    legend_axes: list[plt.Axes] | None = None,
    hspace: float = 0.30,
    wspace: float = 0.24,
    add_legend: bool = True,
    left: float = 0.10,
    right: float = 0.98,
    top: float = 0.92,
    bottom: float | None = None,
    legend_y: float = LEGEND_Y,
    legend_ncol: int | None = None,
    legend_fontsize: float = 9.0,
    legend_columnspacing: float = 1.4,
    legend_handlelength: float = 2.2,
) -> None:
    axes = [ax for ax in (legend_axes or list(fig.axes)) if ax.get_visible()]
    bottom_margin = (
        float(bottom)
        if bottom is not None
        else (_legend_bottom_margin(axes) if add_legend else 0.10)
    )
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom_margin,
        hspace=hspace,
        wspace=wspace,
    )
    if add_legend:
        _add_figure_legend(
            fig,
            axes,
            y=legend_y,
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            columnspacing=legend_columnspacing,
            handlelength=legend_handlelength,
        )
    png_path = _png_export_path(out_path)
    pdf_path = _pdf_export_path(out_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=DEFAULT_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)


def _finalize_single_panel_legend_figure(fig, *, out_path: Path, ax: plt.Axes) -> None:
    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax],
        top=0.94,
        bottom=0.245,
        legend_y=0.016,
        legend_fontsize=8.8,
        legend_columnspacing=1.25,
        legend_handlelength=2.0,
    )


def _finalize_wide_line_figure(fig, *, out_path: Path, ax: plt.Axes) -> None:
    """Finalize a wide (double-column) line plot with a compact 6-column legend
    below, leaving room for the x-axis label."""
    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax],
        top=0.92,
        bottom=0.30,
        left=0.085,
        right=0.985,
        legend_y=0.02,
        legend_ncol=6,
        legend_fontsize=plotstyle.FS_ANNOT,
        legend_columnspacing=1.1,
        legend_handlelength=1.7,
    )


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
        kstar_raw = raw_row.get("k", raw_row.get("selected_k"))
        try:
            kstar = (
                int(kstar_raw)
                if kstar_raw is not None and str(kstar_raw).strip() != ""
                else None
            )
        except (TypeError, ValueError):
            kstar = None
        support_rows.append(
            {
                "model": model,
                "kstar": kstar,
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
    _style_axes(ax)
    ax.set_xlabel("$k$")
    if len(k_ticks) <= 12:
        ax.set_xticks(k_ticks)
    else:
        ax.set_xticks(_human_friendly_integer_ticks(k_ticks))

    if len(k_ticks) > 1:
        span = float(max(k_ticks) - min(k_ticks))
        pad = max(0.5, 0.03 * span)
        ax.set_xlim(float(min(k_ticks)) - pad, float(max(k_ticks)) + pad)
    else:
        ax.set_xlim(float(k_ticks[0]) - 0.5, float(k_ticks[0]) + 0.5)


def _human_friendly_integer_ticks(k_ticks: list[int], *, max_ticks: int = 8) -> list[int]:
    if not k_ticks:
        return []
    k_min = int(min(k_ticks))
    k_max = int(max(k_ticks))
    if len(k_ticks) <= max_ticks and len(k_ticks) == (k_max - k_min + 1):
        return list(k_ticks)

    if k_max <= k_min:
        return [k_min]

    span = max(1, k_max - k_min)
    target_step = max(1.0, span / max(1, max_ticks - 1))
    magnitude = 10 ** math.floor(math.log10(target_step))
    normalized = target_step / magnitude
    if normalized <= 1.0:
        step = 1 * magnitude
    elif normalized <= 2.0:
        step = 2 * magnitude
    elif normalized <= 5.0:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    step = int(max(1, step))

    ticks = [k_min]
    tick = int(math.ceil((k_min + 1) / step) * step)
    while tick < k_max:
        ticks.append(int(tick))
        tick += step
    if ticks[-1] != k_max:
        ticks.append(k_max)
    return ticks


def _highlight_selected(
    ax, *, ks: np.ndarray, ys: np.ndarray, selected_k: int, color: str, marker: str
) -> None:
    idx_arr = np.where(ks == int(selected_k))[0]
    if int(idx_arr.size) == 0:
        return
    idx = int(idx_arr[0])
    size = 140 if marker == "*" else 110
    ax.scatter(
        [ks[idx]],
        [ys[idx]],
        s=size,
        color=color,
        marker=marker,
        edgecolors="white",
        linewidths=0.7,
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
) -> None:
    if not rows:
        return
    by_model = _group_k_rows(rows)
    models = sorted(by_model, key=lambda m: (model_sort_key(m), m))
    k_ticks = sorted({int(row["k"]) for row in rows})
    all_values = np.asarray([float(row[value_key]) for row in rows], dtype=float)

    _set_k_axis(ax, k_ticks)
    ax.set_ylim(*_padded_unit_interval_limits(all_values))
    ax.set_ylabel(ylabel)
    _set_panel_title(ax, title)

    for model in models:
        model_rows = by_model[model]
        ks = np.asarray([int(r["k"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r[value_key]) for r in model_rows], dtype=float)
        color = _color_for_model(model)
        ax.plot(
            ks,
            vals,
            color=color,
            linewidth=plotstyle.LW_SERIES,
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


def _draw_model_scatter(
    ax,
    rows: list[dict],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    diagonal: bool = False,
    hline: float | None = None,
    vline: float | None = None,
) -> None:
    """Shared one-point-per-model scatter with the project's visual identity.

    Points and the figure legend follow the canonical (family-grouped) model
    order. The data box is forced square via ``set_box_aspect``. Reference
    geometry is configurable: a ``y = x`` diagonal and/or horizontal/vertical
    threshold lines.
    """
    points = [
        (str(r["model"]), float(r[x_key]), float(r[y_key]))
        for r in rows
        if x_key in r and y_key in r
        and np.isfinite(float(r[x_key])) and np.isfinite(float(r[y_key]))
    ]
    if not points:
        return
    points.sort(key=lambda p: (model_sort_key(p[0]), p[0]))

    _style_axes(ax)
    ref_kw = dict(
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
    )
    if diagonal:
        ax.plot([0.0, 1.0], [0.0, 1.0], **ref_kw)
    if hline is not None:
        ax.axhline(y=hline, **ref_kw)
    if vline is not None:
        ax.axvline(x=vline, **ref_kw)

    for model, x, y in points:
        ax.scatter(
            [x],
            [y],
            s=52,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
            label=model,
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_box_aspect(1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _set_panel_title(ax, title)


def _draw_bio_vs_confounder_scatter(ax, rows: list[dict]) -> None:
    if not rows:
        return
    confounder_display_name = _confounder_display_name(rows)
    xs = np.asarray([float(r["confounder_knn_bacc"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["bio_knn_bacc"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)
    _draw_model_scatter(
        ax,
        rows,
        x_key="confounder_knn_bacc",
        y_key="bio_knn_bacc",
        xlabel=f"{confounder_display_name} accuracy",
        ylabel="Biological accuracy",
        title=f"Biological vs {confounder_display_name} accuracy",
        xlim=lim,
        ylim=lim,
        diagonal=True,
    )


def _draw_mari_vs_ri_scatter(ax, rows: list[dict]) -> None:
    if not rows:
        return
    xs = np.asarray([float(r["ri"]) for r in rows], dtype=float)
    ys = np.asarray([float(r["mari"]) for r in rows], dtype=float)
    combined = np.concatenate([xs, ys]) if xs.size > 0 else xs
    lim = _padded_unit_interval_limits(combined)
    _draw_model_scatter(
        ax,
        rows,
        x_key="ri",
        y_key="mari",
        xlabel="RI",
        ylabel="MaRI",
        title="MaRI vs RI",
        xlim=lim,
        ylim=lim,
        diagonal=True,
    )


def plot_knn_bio_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="knn_bacc",
        ylabel="Balanced accuracy",
        title="Biological accuracy over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)


def plot_knn_confounder_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    confounder_display_name = _confounder_display_name(rows)
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="knn_confounder_bacc",
        ylabel="Balanced accuracy",
        title=f"{confounder_display_name} accuracy over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)


def plot_ri_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="ri",
        ylabel="RI",
        title="RI over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)


def plot_mari_k_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    _draw_k_curve(
        ax,
        rows=rows,
        value_key="mari",
        ylabel="MaRI",
        title="MaRI over k",
        highlight_rules=[("selected_k", "*")],
    )
    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)


def _draw_cumulative_mean_k_curve(
    ax,
    *,
    rows: list[dict],
    value_key: str,
    ylabel: str,
    title: str,
) -> None:
    """Draw cumulative mean of ``value_key`` over k for each model.

    At x=k the y-value is ``mean(v_1, ..., v_k)`` (ascending k order), so the
    rightmost point equals the arithmetic mean across all evaluated k values —
    i.e. the value reported by ``--summarize-by-mean``.
    """
    if not rows:
        return
    by_model = _group_k_rows(rows)
    models = sorted(by_model, key=lambda m: (model_sort_key(m), m))
    k_ticks = sorted({int(row["k"]) for row in rows})

    # Compute all cumulative-mean values to set axis limits
    all_cum_vals: list[float] = []
    model_curves: list[tuple[str, np.ndarray, np.ndarray]] = []
    for model in models:
        model_rows = by_model[model]
        ks = np.asarray([int(r["k"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r[value_key]) for r in model_rows], dtype=float)
        order = np.argsort(ks)
        ks, vals = ks[order], vals[order]
        cum_means = np.cumsum(vals) / np.arange(1, len(vals) + 1, dtype=float)
        all_cum_vals.extend(cum_means.tolist())
        model_curves.append((model, ks, cum_means))

    _set_k_axis(ax, k_ticks)
    ax.set_ylim(*_padded_unit_interval_limits(np.asarray(all_cum_vals, dtype=float)))
    ax.set_ylabel(ylabel)
    _set_panel_title(ax, title)

    for model, ks, cum_means in model_curves:
        color = _color_for_model(model)
        ax.plot(ks, cum_means, color=color, linewidth=plotstyle.LW_SERIES, alpha=0.95, label=model)
        # Mark the endpoint (= reported summarize_by_mean value)
        ax.scatter(
            [ks[-1]],
            [cum_means[-1]],
            s=42,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )


def plot_ri_cumulative_mean_k_sweep(rows: list[dict], out_path: Path) -> None:
    """Single-panel cumulative mean RI over k.

    Intended for use with ``--summarize-by-mean``: each model's rightmost point
    is exactly the value reported as its final RI score.
    """
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    _draw_cumulative_mean_k_curve(
        ax,
        rows=rows,
        value_key="ri",
        ylabel="Cumulative mean RI",
        title="RI – cumulative mean over k",
    )
    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)


def plot_mari_cumulative_mean_k_sweep(rows: list[dict], out_path: Path) -> None:
    """Single-panel cumulative mean MaRI over k.

    Intended for use with ``--summarize-by-mean``: each model's rightmost point
    is exactly the value reported as its final MaRI score.
    """
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    _draw_cumulative_mean_k_curve(
        ax,
        rows=rows,
        value_key="mari",
        ylabel="Cumulative mean MaRI",
        title="MaRI – cumulative mean over k",
    )
    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)


def plot_ri_mari_support(rows: list[dict], out_path: Path) -> None:
    support_rows = _support_plot_rows(rows)
    fig_height = max(3.0, 0.85 + 0.42 * max(len(support_rows), 1))
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, fig_height))

    if not support_rows:
        ax.set_visible(False)
        png_path = _png_export_path(out_path)
        pdf_path = _pdf_export_path(out_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(png_path, dpi=DEFAULT_DPI)
        fig.savefig(pdf_path)
        plt.close(fig)
        return

    y = np.arange(len(support_rows), dtype=float)
    labels = [
        f"{row['model']} (k*={row['kstar']})"
        if row.get("kstar") is not None
        else str(row["model"])
        for row in support_rows
    ]

    ax.set_facecolor(PANEL_FACE_COLOR)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=plotstyle.LW_GRID, alpha=plotstyle.GRID_ALPHA, zorder=0)

    # Single neutral colour scheme: a light track behind a solid "defined" fill.
    track_color = "#e7ecf0"
    fill_color = "#5f7d92"

    for idx, row in enumerate(support_rows):
        defined_frac = float(row["defined_frac"])
        undefined_frac = float(row["undefined_frac"])

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
            fontsize=plotstyle.FS_ANNOT,
            color="white",
            zorder=3,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of evaluated samples", labelpad=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=plotstyle.FS_TICK)
    _set_panel_title(ax, "Support coverage")
    ax.set_ylim(float(len(support_rows) - 0.35), -0.65)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)

    png_path = _png_export_path(out_path)
    pdf_path = _pdf_export_path(out_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(png_path, dpi=DEFAULT_DPI)
    fig.savefig(pdf_path)
    plt.close(fig)


def plot_ccmr_m_sweep(rows: list[dict], out_path: Path) -> None:
    """Single-panel pooled CCMR(m) trajectory per model, with the CCMR=1 threshold."""
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    ccmr_rows = [
        r
        for r in rows
        if "m" in r
        and "ccmr" in r
        and np.isfinite(float(r["m"]))
        and np.isfinite(float(r["ccmr"]))
    ]
    if not ccmr_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, add_legend=False)
        return

    by_model: dict[str, list[dict]] = {}
    for row in ccmr_rows:
        model = str(row["model"])
        by_model.setdefault(model, []).append(row)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r["m"]))

    m_all = sorted({int(row["m"]) for row in ccmr_rows})
    m_min, m_max = m_all[0], m_all[-1]

    _style_axes(ax)
    ax.axhline(
        y=1.0,
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
        alpha=0.8,
    )
    ax.set_ylabel("CCMR")
    _set_panel_title(ax, "CCMR over m")
    ccmr_values = np.asarray([float(r["ccmr"]) for r in ccmr_rows], dtype=float)
    finite = ccmr_values[np.isfinite(ccmr_values)]
    vmin = float(np.nanmin(finite)) if finite.size > 0 else 0.0
    vmax = float(np.nanmax(finite)) if finite.size > 0 else 1.0
    span = vmax - vmin
    pad = max(0.05, span * 0.10) if span > 1e-9 else max(0.1, abs(vmin) * 0.10)
    ax.set_ylim(max(0.0, vmin - pad), vmax + pad)

    for model in sorted(by_model, key=lambda m: (model_sort_key(m), m)):
        model_rows = by_model[model]
        color = _color_for_model(model)
        ms = np.asarray([int(r["m"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r["ccmr"]) for r in model_rows], dtype=float)
        ax.plot(ms, vals, color=color, linewidth=plotstyle.LW_SERIES, alpha=0.95, label=model)

    tick_positions = _human_friendly_integer_ticks(m_all, max_ticks=6)
    ax.set_xticks(tick_positions)
    ax.set_xlim(m_min - 0.5, m_max + 0.5)
    ax.set_xlabel("$m$")

    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)


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


def plot_ccmr_ltm_scatter(rows: list[dict], out_path: Path) -> None:
    """CCMR vs LTM scatter with a horizontal CCMR=1 robustness threshold.

    The threshold line (not a y=x diagonal) makes the claim non-tautological: every
    model's fragile decile falling below it is an empirical fact, since LTM <= median
    CCMR by construction would only force points below the diagonal, not below 1.
    """
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    valid_rows = _valid_ccmr_ltm_rows(rows)

    if not valid_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    label_ltm = _ltm_label(valid_rows)
    xs = np.asarray([float(r["ccmr"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["ltm"]) for r in valid_rows], dtype=float)
    lim = _padded_positive_limits(np.concatenate([xs, ys]))

    _draw_model_scatter(
        ax,
        valid_rows,
        x_key="ccmr",
        y_key="ltm",
        xlabel="CCMR",
        ylabel=label_ltm,
        title=f"CCMR vs {label_ltm}",
        xlim=lim,
        ylim=lim,
        hline=1.0,
    )

    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_ccmr_ltm_bars(rows: list[dict], out_path: Path) -> None:
    """Per-model CCMR/LTM bars sorted by LTM, with a CCMR=1 threshold line."""
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    valid_rows = _valid_ccmr_ltm_rows(rows)

    if not valid_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    label_ltm = _ltm_label(valid_rows)
    ranked_rows = sorted(
        valid_rows, key=lambda r: (float(r["ltm"]), str(r["model"])), reverse=True
    )
    model_names = [str(r["model"]) for r in ranked_rows]
    ccmr_vals = np.asarray([float(r["ccmr"]) for r in ranked_rows], dtype=float)
    ltm_vals = np.asarray([float(r["ltm"]) for r in ranked_rows], dtype=float)
    colors = [_color_for_model(model) for model in model_names]
    x = np.arange(len(ranked_rows), dtype=float)
    width = 0.38

    _style_axes(ax, grid_axis="y")
    ax.axhline(
        y=1.0,
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
        alpha=0.75,
    )
    ax.bar(
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
    ax.bar(
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
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=24, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=plotstyle.FS_TICK, pad=4)
    ax.set_ylabel("Score")
    _set_panel_title(ax, f"Sorted by {label_ltm}")
    y_lo, y_hi = _padded_positive_limits(np.concatenate([ccmr_vals, ltm_vals]))
    ax.set_ylim(y_lo, y_hi)
    ax.legend(frameon=False, loc="upper right", fontsize=plotstyle.FS_ANNOT)

    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax],
        add_legend=False,
        left=0.10,
        right=0.985,
        top=0.90,
        bottom=0.18,
    )


def plot_bio_vs_confounder_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_bio_vs_confounder_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_mari_vs_ri_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_mari_vs_ri_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def _draw_ccmr_vs_mari_scatter(ax, rows: list[dict]) -> None:
    ccmr_rows = [
        r
        for r in rows
        if "ccmr" in r and "mari" in r and np.isfinite(float(r["ccmr"]))
    ]
    if not ccmr_rows:
        ax.set_visible(False)
        return
    xs = np.asarray([float(r["mari"]) for r in ccmr_rows], dtype=float)
    ys = np.asarray([float(r["ccmr"]) for r in ccmr_rows], dtype=float)
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    _draw_model_scatter(
        ax,
        ccmr_rows,
        x_key="mari",
        y_key="ccmr",
        xlabel="MaRI",
        ylabel="CCMR",
        title="CCMR vs MaRI",
        xlim=_padded_unit_interval_limits(xs),
        ylim=(max(0.0, float(ys.min()) - y_pad), float(ys.max()) + y_pad),
        hline=1.0,
        vline=0.5,
    )


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
    for row in ccmr_rows:
        path = Path(str(row["ccmr_samples_path"]))
        if not path.exists():
            continue
        values = np.load(path)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        alpha = float(row["ccmr_alpha"])
        alpha_pct = int(round(alpha * 100))
        model_data.append(
            {
                "model": str(row["model"]),
                "values": values,
                "q_alpha": float(row["ccmr_q_alpha"]),
                "alpha": alpha,
                "alpha_pct": alpha_pct,
                "ccmr": float(row["ccmr"]),
                "lt1_frac": float(np.mean(values < 1.0)),
            }
        )

    if not model_data:
        ax.set_visible(False)
        return

    model_data = sorted(
        model_data,
        key=lambda d: (float(d["ccmr"]), str(d["model"])),
        reverse=True,
    )
    all_values = np.concatenate([d["values"] for d in model_data])
    x_min = max(0.0, float(np.nanpercentile(all_values, 1)) - 0.1)
    x_max = float(np.nanpercentile(all_values, 99)) + 0.1
    x_grid = np.linspace(x_min, x_max, 512)

    _style_axes(ax)
    # Ridgelines read best with only the bottom axis (no box).
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.grid(False)
    ax.set_yticks([])

    # Shade the fragile region (CCMR < 1.0)
    shade_right = min(1.0, x_max)
    if shade_right > x_min:
        ax.axvspan(x_min, shade_right, color=FRAGILE_SHADE_COLOR, alpha=0.55, zorder=1)
    ax.axvline(
        x=1.0,
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=2,
        alpha=0.75,
    )

    row_centers = np.arange(len(model_data), 0, -1, dtype=float)
    row_spacing = 1.0
    amplitude = 0.72
    global_peak = 0.0
    rendered_rows: list[tuple[dict, np.ndarray, np.ndarray | None, np.ndarray, float]] = []
    for d in model_data:
        values = d["values"]
        try:
            density = gaussian_kde(values, bw_method="scott")(x_grid)
        except Exception:
            density = None

        if density is None:
            counts, edges = np.histogram(values, bins=40, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            y_values = counts
            x_values = centers
        else:
            y_values = density
            x_values = x_grid
        peak = float(np.nanmax(y_values)) if len(y_values) > 0 else 0.0
        global_peak = max(global_peak, peak)
        rendered_rows.append((d, np.asarray(x_values, dtype=float), density, np.asarray(y_values, dtype=float), peak))

    if global_peak <= 0.0:
        global_peak = 1.0

    for row_center, rendered in zip(row_centers, rendered_rows):
        d, x_values, density, y_values, peak = rendered
        color = _color_for_model(d["model"])
        scale = amplitude / max(peak, 1e-9)
        y_curve = row_center + y_values * scale
        ax.hlines(
            y=row_center,
            xmin=x_min,
            xmax=x_max,
            color=SPINE_COLOR,
            linewidth=0.7,
            alpha=0.7,
            zorder=1,
        )
        ax.fill_between(
            x_values,
            row_center,
            y_curve,
            color=color,
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )
        if density is None:
            ax.step(
                x_values,
                y_curve,
                where="mid",
                color=color,
                linewidth=1.5,
                alpha=0.90,
                zorder=3,
            )
        else:
            ax.plot(
                x_values,
                y_curve,
                color=color,
                linewidth=1.7,
                alpha=0.95,
                zorder=3,
            )

        q = float(d["q_alpha"])
        if np.isfinite(q) and x_min <= q <= x_max:
            ax.vlines(
                x=q,
                ymin=row_center - 0.18,
                ymax=row_center + amplitude,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.95,
                zorder=4,
            )

        ax.text(
            -0.02,
            row_center,
            str(d["model"]),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=plotstyle.FS_TICK,
            color=TEXT_COLOR,
        )
    alpha_pct = int(model_data[0]["alpha_pct"])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.5, float(len(model_data)) + 0.95)
    ax.set_xlabel("Per-sample CCMR")
    ax.set_ylabel("")


def plot_ccmr_sample_distributions(rows: list[dict], out_path: Path) -> None:
    valid_rows = [
        r
        for r in rows
        if "ccmr_samples_path" in r
        and "ccmr_q_alpha" in r
        and np.isfinite(float(r.get("ccmr", float("nan"))))
    ]
    fig_height = max(3.5, 0.85 + 0.45 * max(1, len(valid_rows)))
    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(COL_DOUBLE, fig_height),
        gridspec_kw={"width_ratios": [5.0, 1.6]},
    )
    alpha_values = [
        int(round(float(r["ccmr_alpha"]) * 100))
        for r in valid_rows
        if "ccmr_alpha" in r and np.isfinite(float(r["ccmr_alpha"]))
    ]
    alpha_pct = alpha_values[0] if alpha_values else 10
    _draw_ccmr_sample_distributions(ax, rows)
    info_ax.set_axis_off()
    if ax.get_visible():
        info_ax.set_ylim(ax.get_ylim())
        info_ax.set_xlim(0.0, 1.0)
        # Right edges of the numeric columns (right-aligned so digits line up).
        col_x = (0.26, 0.62, 0.99)
        header_y = float(len(valid_rows)) + 0.72
        for label, x_pos in zip(("CCMR", f"Q{alpha_pct}", "%<1"), col_x):
            info_ax.text(
                x_pos,
                header_y,
                label,
                ha="right",
                va="center",
                fontsize=plotstyle.FS_ANNOT,
                color=TEXT_COLOR,
                weight="bold",
            )
        model_data: list[dict[str, float | str]] = []
        for row in valid_rows:
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
                    "ccmr": float(row["ccmr"]),
                    "q_alpha": float(row["ccmr_q_alpha"]),
                    "lt1_frac": float(np.mean(values < 1.0)),
                }
            )
        model_data = sorted(
            model_data,
            key=lambda d: (float(d["ccmr"]), str(d["model"])),
            reverse=True,
        )
        row_centers = np.arange(len(model_data), 0, -1, dtype=float)
        for row_center, row in zip(row_centers, model_data):
            cells = (
                f"{float(row['ccmr']):.{PREC_METRIC}f}",
                f"{float(row['q_alpha']):.{PREC_METRIC}f}",
                f"{100.0 * float(row['lt1_frac']):.{PREC_PERCENT}f}%",
            )
            for value, x_pos in zip(cells, col_x):
                info_ax.text(
                    x_pos,
                    float(row_center),
                    value,
                    ha="right",
                    va="center",
                    fontsize=plotstyle.FS_ANNOT,
                    color=TEXT_COLOR,
                )
    fig.suptitle(
        "Per-sample CCMR distributions",
        fontsize=plotstyle.FS_TITLE,
        weight="bold",
        y=0.985,
        color=TEXT_COLOR,
    )
    fig.text(
        0.5,
        0.935,
        f"sorted by CCMR; dotted: $Q_{{{alpha_pct}}}$; shaded: CCMR < 1",
        ha="center",
        va="center",
        fontsize=plotstyle.FS_ANNOT,
        color=MUTED_TEXT_COLOR,
    )
    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax, info_ax],
        add_legend=False,
        left=0.165,
        right=0.94,
        top=0.885,
        bottom=0.095,
        wspace=0.01,
    )


def plot_ri_mari_sample_distributions(
    rows: list[dict], metric: str, out_path: Path
) -> None:
    """Plot per-sample RI or MaRI distributions across models (ridge-line style).

    Parameters
    ----------
    rows:
        List of per-model summary dicts as produced by benchmark.py.
    metric:
        Either ``"ri"`` or ``"mari"``.
    out_path:
        Destination path for the output PNG.
    """
    samples_key = f"{metric}_samples_path"
    metric_lbl = metric_label(metric)
    alpha_pct = 10

    valid_rows = [
        r
        for r in rows
        if samples_key in r
        and np.isfinite(float(r.get(metric, float("nan"))))
    ]
    if not valid_rows:
        return

    model_data = []
    for row in valid_rows:
        path = Path(str(row[samples_key]))
        if not path.exists():
            continue
        values = np.load(path)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        # Always compute Q10 and %<0.5 from loaded values — avoids n/a from stale cache
        q_alpha = float(np.percentile(values, alpha_pct))
        lt05_frac = float(np.mean(values < 0.5))
        model_data.append(
            {
                "model": str(row["model"]),
                "values": values,
                "metric_val": float(row[metric]),
                "q_alpha": q_alpha,
                "lt05_frac": lt05_frac,
            }
        )

    if not model_data:
        return

    model_data = sorted(
        model_data,
        key=lambda d: (float(d["metric_val"]), str(d["model"])),
        reverse=True,
    )

    fig_height = max(3.5, 0.85 + 0.45 * max(1, len(model_data)))
    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(COL_DOUBLE, fig_height),
        gridspec_kw={"width_ratios": [5.0, 1.6]},
    )

    all_values = np.concatenate([d["values"] for d in model_data])
    x_min = max(0.0, float(np.nanpercentile(all_values, 1)) - 0.05)
    x_max = min(1.0, float(np.nanpercentile(all_values, 99)) + 0.05)
    x_grid = np.linspace(x_min, x_max, 512)

    _style_axes(ax)
    # Ridgelines read best with only the bottom axis (no box).
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.grid(False)
    ax.set_yticks([])

    # Q10 is dropped (column + per-row dotted line) when it carries no
    # information, i.e. every model's value collapses to the same rounded number.
    q_rounded = {round(float(d["q_alpha"]), PREC_METRIC) for d in model_data}
    show_q = len(q_rounded) > 1

    # Shade the fragile region (RI/MaRI < 0.5 = more OS than SO neighbors)
    shade_right = min(0.5, x_max)
    if shade_right > x_min:
        ax.axvspan(x_min, shade_right, color=FRAGILE_SHADE_COLOR, alpha=0.55, zorder=1)
    ax.axvline(
        x=0.5,
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=2,
        alpha=0.75,
    )

    row_centers = np.arange(len(model_data), 0, -1, dtype=float)
    amplitude = 0.72

    rendered_rows = []
    for d in model_data:
        values = d["values"]
        try:
            density = gaussian_kde(values, bw_method="scott")(x_grid)
        except Exception:
            density = None

        if density is None:
            counts, edges = np.histogram(values, bins=40, density=True)
            centers_hist = 0.5 * (edges[:-1] + edges[1:])
            y_values = counts
            x_values = centers_hist
        else:
            y_values = density
            x_values = x_grid
        peak = float(np.nanmax(y_values)) if len(y_values) > 0 else 0.0
        rendered_rows.append(
            (d, np.asarray(x_values, dtype=float), density, np.asarray(y_values, dtype=float), peak)
        )

    for row_center, rendered in zip(row_centers, rendered_rows):
        d, x_values, density, y_values, peak = rendered
        color = _color_for_model(d["model"])
        scale = amplitude / max(peak, 1e-9)
        y_curve = row_center + y_values * scale
        ax.hlines(
            y=row_center,
            xmin=x_min,
            xmax=x_max,
            color=SPINE_COLOR,
            linewidth=0.7,
            alpha=0.7,
            zorder=1,
        )
        ax.fill_between(
            x_values,
            row_center,
            y_curve,
            color=color,
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )
        if density is None:
            ax.step(x_values, y_curve, where="mid", color=color, linewidth=1.5, alpha=0.90, zorder=3)
        else:
            ax.plot(x_values, y_curve, color=color, linewidth=1.7, alpha=0.95, zorder=3)

        q = float(d["q_alpha"])
        if show_q and np.isfinite(q) and x_min <= q <= x_max:
            ax.vlines(
                x=q,
                ymin=row_center - 0.18,
                ymax=row_center + amplitude,
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.95,
                zorder=4,
            )

        ax.text(
            -0.02,
            row_center,
            str(d["model"]),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=plotstyle.FS_TICK,
            color=TEXT_COLOR,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.5, float(len(model_data)) + 0.95)
    ax.set_xlabel(f"Per-sample {metric_lbl}")
    ax.set_ylabel("")

    info_ax.set_axis_off()
    info_ax.set_ylim(ax.get_ylim())
    info_ax.set_xlim(0.0, 1.0)
    # Right edges of the numeric columns (right-aligned so digits line up).
    # The Q10 column is included only when it is informative.
    if show_q:
        headers = (metric_lbl, f"Q{alpha_pct}", "%<0.5")
        col_x = (0.26, 0.62, 0.99)
    else:
        headers = (metric_lbl, "%<0.5")
        col_x = (0.46, 0.99)
    header_y = float(len(model_data)) + 0.72
    for label, x_pos in zip(headers, col_x):
        info_ax.text(
            x_pos,
            header_y,
            label,
            ha="right",
            va="center",
            fontsize=plotstyle.FS_ANNOT,
            color=TEXT_COLOR,
            weight="bold",
        )
    for row_center, d in zip(row_centers, model_data):
        if show_q:
            cells = (
                f"{float(d['metric_val']):.{PREC_METRIC}f}",
                f"{float(d['q_alpha']):.{PREC_METRIC}f}",
                f"{100.0 * float(d['lt05_frac']):.{PREC_PERCENT}f}%",
            )
        else:
            cells = (
                f"{float(d['metric_val']):.{PREC_METRIC}f}",
                f"{100.0 * float(d['lt05_frac']):.{PREC_PERCENT}f}%",
            )
        for value, x_pos in zip(cells, col_x):
            info_ax.text(
                x_pos,
                float(row_center),
                value,
                ha="right",
                va="center",
                fontsize=plotstyle.FS_ANNOT,
                color=TEXT_COLOR,
            )

    fig.suptitle(
        f"Per-sample {metric_lbl} distributions",
        fontsize=plotstyle.FS_TITLE,
        weight="bold",
        y=0.985,
        color=TEXT_COLOR,
    )
    caption = f"sorted by {metric_lbl}; "
    if show_q:
        caption += f"dotted: $Q_{{{alpha_pct}}}$; "
    caption += f"shaded: {metric_lbl} < 0.5"
    fig.text(
        0.5,
        0.935,
        caption,
        ha="center",
        va="center",
        fontsize=plotstyle.FS_ANNOT,
        color=MUTED_TEXT_COLOR,
    )
    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax, info_ax],
        add_legend=False,
        left=0.165,
        right=0.94,
        top=0.885,
        bottom=0.095,
        wspace=0.01,
    )


def plot_ccmr_vs_mari_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_ccmr_vs_mari_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def _draw_q_alpha_vs_ccmr_scatter(ax, rows: list[dict]) -> None:
    valid_rows = [
        r
        for r in rows
        if "ccmr" in r
        and "ccmr_q_alpha" in r
        and np.isfinite(float(r["ccmr"]))
        and np.isfinite(float(r["ccmr_q_alpha"]))
    ]
    if not valid_rows:
        ax.set_visible(False)
        return

    alpha_pct_values = [
        int(round(float(r["ccmr_alpha"]) * 100))
        for r in valid_rows
        if "ccmr_alpha" in r and np.isfinite(float(r["ccmr_alpha"]))
    ]
    alpha_pct = alpha_pct_values[0] if alpha_pct_values else 10

    xs = np.asarray([float(r["ccmr"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["ccmr_q_alpha"]) for r in valid_rows], dtype=float)
    x_pad = max(0.1, (xs.max() - xs.min()) * 0.10) if xs.size > 0 else 0.5
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    _draw_model_scatter(
        ax,
        valid_rows,
        x_key="ccmr",
        y_key="ccmr_q_alpha",
        xlabel="CCMR",
        ylabel=f"Q{alpha_pct}",
        title=f"Q{alpha_pct} vs CCMR",
        xlim=(max(0.0, float(xs.min()) - x_pad), float(xs.max()) + x_pad),
        ylim=(max(0.0, float(ys.min()) - y_pad), float(ys.max()) + y_pad),
        hline=1.0,
        vline=1.0,
    )


def plot_q_alpha_vs_ccmr_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_q_alpha_vs_ccmr_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)