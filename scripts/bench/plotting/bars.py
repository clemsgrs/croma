"""Bar archetype: support-coverage bars and per-model CRoMa/LTM bars.

Holds the support-coverage data-prep helpers (only consumed here) alongside the
two bar entry points. The LTM row/label helpers are shared with the scatter
archetype and live in ``base``.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from . import style as plotstyle
from .style import (
    COL_DOUBLE,
    DEFAULT_DPI,
    GRID_COLOR,
    PANEL_FACE_COLOR,
    REFERENCE_LINE_COLOR,
)

from .base import (
    _color_for_model,
    _ltm_label,
    _padded_signed_limits,
    _set_panel_title,
    _style_axes,
    _valid_croma_ltm_rows,
)
from .export import _finalize_figure, _pdf_export_path, _png_export_path

_SUPPORT_STATUS_COLORS = plotstyle.SUPPORT_STATUS_COLORS


def _clamp_fraction(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _support_status(support: float) -> str:
    frac = _clamp_fraction(support)
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

        try:
            support = float(raw_row["support"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(support):
            continue
        support = _clamp_fraction(support)
        status = _support_status(support)
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
                "support": support,
                "status": status,
                "fill_color": fill_color,
                "track_color": track_color,
                "label": f"{int(round(support * 100.0))}%",
            }
        )

    return sorted(support_rows, key=lambda row: (row["support"], row["model"]))


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

    # Single neutral colour scheme: a light track behind a solid support fill.
    track_color = "#e7ecf0"
    fill_color = "#5f7d92"

    for idx, row in enumerate(support_rows):
        support = float(row["support"])

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
            support,
            color=fill_color,
            edgecolor="none",
            height=0.58,
            zorder=2,
        )

        ax.text(
            support / 2.0,
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


def plot_croma_ltm_bars(rows: list[dict], out_path: Path) -> None:
    """Per-model CRoMa/LTM bars sorted by LTM, with a CRoMa=0 threshold line."""
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    valid_rows = _valid_croma_ltm_rows(rows)

    if not valid_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    label_ltm = _ltm_label(valid_rows)
    ranked_rows = sorted(
        valid_rows, key=lambda r: (float(r["ltm"]), str(r["model"])), reverse=True
    )
    model_names = [str(r["model"]) for r in ranked_rows]
    croma_vals = np.asarray([float(r["croma"]) for r in ranked_rows], dtype=float)
    ltm_vals = np.asarray([float(r["ltm"]) for r in ranked_rows], dtype=float)
    colors = [_color_for_model(model) for model in model_names]
    x = np.arange(len(ranked_rows), dtype=float)
    width = 0.38

    _style_axes(ax, grid_axis="y")
    ax.axhline(
        y=0.0,
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
        alpha=0.75,
    )
    ax.bar(
        x - width / 2.0,
        croma_vals,
        width=width,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
        label="CRoMa",
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
    y_lo, y_hi = _padded_signed_limits(np.concatenate([croma_vals, ltm_vals]))
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
