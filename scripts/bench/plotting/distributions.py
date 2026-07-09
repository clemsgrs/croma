"""Distribution archetype: per-sample ridge-line distributions across models.

Covers the per-sample CRoMa distributions (with an adjacent numeric info panel).
The CRoMa variant shares the ``_draw_croma_sample_distributions`` drawer with its
``plot_*`` wrapper.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from croma import plotstyle
from croma.plotstyle import (
    COL_DOUBLE,
    FRAGILE_SHADE_COLOR,
    MUTED_TEXT_COLOR,
    PREC_METRIC,
    PREC_PERCENT,
    REFERENCE_LINE_COLOR,
    SPINE_COLOR,
    TEXT_COLOR,
)

from .base import _color_for_model, _style_axes
from .export import _finalize_figure


def _draw_croma_sample_distributions(ax, rows: list[dict]) -> None:
    croma_rows = [
        r
        for r in rows
        if "croma_samples_path" in r
        and "croma_q_alpha" in r
        and np.isfinite(float(r.get("croma", float("nan"))))
    ]
    if not croma_rows:
        ax.set_visible(False)
        return

    model_data = []
    for row in croma_rows:
        path = Path(str(row["croma_samples_path"]))
        if not path.exists():
            continue
        values = np.load(path)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        alpha = float(row["croma_alpha"])
        alpha_pct = int(round(alpha * 100))
        model_data.append(
            {
                "model": str(row["model"]),
                "values": values,
                "q_alpha": float(row["croma_q_alpha"]),
                "alpha": alpha,
                "alpha_pct": alpha_pct,
                "croma": float(row["croma"]),
                "neg_frac": float(np.mean(values < 0.0)),
            }
        )

    if not model_data:
        ax.set_visible(False)
        return

    model_data = sorted(
        model_data,
        key=lambda d: (float(d["croma"]), str(d["model"])),
        reverse=True,
    )
    all_values = np.concatenate([d["values"] for d in model_data])
    x_min = float(np.nanpercentile(all_values, 1)) - 0.1
    x_max = float(np.nanpercentile(all_values, 99)) + 0.1
    x_grid = np.linspace(x_min, x_max, 512)

    _style_axes(ax)
    # Ridgelines read best with only the bottom axis (no box).
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.grid(False)
    ax.set_yticks([])

    # Shade the fragile region (CRoMa < 0)
    shade_right = min(0.0, x_max)
    if shade_right > x_min:
        ax.axvspan(x_min, shade_right, color=FRAGILE_SHADE_COLOR, alpha=0.55, zorder=1)
    ax.axvline(
        x=0.0,
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
    ax.set_xlabel("Per-sample CRoMa")
    ax.set_ylabel("")


def plot_croma_sample_distributions(rows: list[dict], out_path: Path) -> None:
    valid_rows = [
        r
        for r in rows
        if "croma_samples_path" in r
        and "croma_q_alpha" in r
        and np.isfinite(float(r.get("croma", float("nan"))))
    ]
    fig_height = max(3.5, 0.85 + 0.45 * max(1, len(valid_rows)))
    fig, (ax, info_ax) = plt.subplots(
        1,
        2,
        figsize=(COL_DOUBLE, fig_height),
        gridspec_kw={"width_ratios": [5.0, 1.6]},
    )
    alpha_values = [
        int(round(float(r["croma_alpha"]) * 100))
        for r in valid_rows
        if "croma_alpha" in r and np.isfinite(float(r["croma_alpha"]))
    ]
    alpha_pct = alpha_values[0] if alpha_values else 10
    _draw_croma_sample_distributions(ax, rows)
    info_ax.set_axis_off()
    if ax.get_visible():
        info_ax.set_ylim(ax.get_ylim())
        info_ax.set_xlim(0.0, 1.0)
        # Right edges of the numeric columns (right-aligned so digits line up).
        col_x = (0.26, 0.62, 0.99)
        header_y = float(len(valid_rows)) + 0.72
        for label, x_pos in zip(("CRoMa", f"Q{alpha_pct}", "%<0"), col_x):
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
            path = Path(str(row["croma_samples_path"]))
            if not path.exists():
                continue
            values = np.load(path)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                continue
            model_data.append(
                {
                    "model": str(row["model"]),
                    "croma": float(row["croma"]),
                    "q_alpha": float(row["croma_q_alpha"]),
                    "neg_frac": float(np.mean(values < 0.0)),
                }
            )
        model_data = sorted(
            model_data,
            key=lambda d: (float(d["croma"]), str(d["model"])),
            reverse=True,
        )
        row_centers = np.arange(len(model_data), 0, -1, dtype=float)
        for row_center, row in zip(row_centers, model_data):
            cells = (
                f"{float(row['croma']):.{PREC_METRIC}f}",
                f"{float(row['q_alpha']):.{PREC_METRIC}f}",
                f"{100.0 * float(row['neg_frac']):.{PREC_PERCENT}f}%",
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
        "Per-sample CRoMa distributions",
        fontsize=plotstyle.FS_TITLE,
        weight="bold",
        y=0.985,
        color=TEXT_COLOR,
    )
    fig.text(
        0.5,
        0.935,
        f"sorted by CRoMa; dotted: $Q_{{{alpha_pct}}}$; shaded: CRoMa < 0",
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
