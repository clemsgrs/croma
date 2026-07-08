"""k-sweep archetype: per-model metric curves over the neighbourhood size ``k``.

The four thin wrappers (biological accuracy, confounder accuracy, RI, MaRI) each
route through the shared ``_draw_k_curve`` drawer; they are kept separate for
discoverability rather than collapsed into one parametrised entry point.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from croma import plotstyle
from croma.plotstyle import COL_DOUBLE, model_sort_key

from .base import (
    _color_for_model,
    _confounder_display_name,
    _group_k_rows,
    _highlight_selected,
    _padded_unit_interval_limits,
    _set_k_axis,
    _set_panel_title,
)
from .export import _finalize_wide_line_figure


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
