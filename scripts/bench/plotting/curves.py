"""Curve archetype: per-model line trajectories over a sweep parameter.

Covers the pooled CRoMa(m) trajectory drawn per model.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from . import style as plotstyle
from .style import COL_DOUBLE, REFERENCE_LINE_COLOR, model_sort_key

from .base import (
    _color_for_model,
    _human_friendly_integer_ticks,
    _set_panel_title,
    _style_axes,
)
from .export import _finalize_figure, _finalize_wide_line_figure


def plot_croma_m_sweep(rows: list[dict], out_path: Path) -> None:
    """Single-panel pooled CRoMa(m) trajectory per model, with the CRoMa=0 threshold."""
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, 4.6))
    croma_rows = [
        r
        for r in rows
        if "m" in r
        and "croma" in r
        and np.isfinite(float(r["m"]))
        and np.isfinite(float(r["croma"]))
    ]
    if not croma_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, add_legend=False)
        return

    by_model: dict[str, list[dict]] = {}
    for row in croma_rows:
        model = str(row["model"])
        by_model.setdefault(model, []).append(row)
    for model in by_model:
        by_model[model] = sorted(by_model[model], key=lambda r: int(r["m"]))

    m_all = sorted({int(row["m"]) for row in croma_rows})
    m_min, m_max = m_all[0], m_all[-1]

    _style_axes(ax)
    ax.axhline(
        y=0.0,
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=1,
        alpha=0.8,
    )
    ax.set_ylabel("CRoMa")
    _set_panel_title(ax, "CRoMa over m")
    croma_values = np.asarray([float(r["croma"]) for r in croma_rows], dtype=float)
    finite = croma_values[np.isfinite(croma_values)]
    vmin = float(np.nanmin(finite)) if finite.size > 0 else 0.0
    vmax = float(np.nanmax(finite)) if finite.size > 0 else 1.0
    span = vmax - vmin
    pad = max(0.05, span * 0.10) if span > 1e-9 else max(0.1, abs(vmin) * 0.10)
    ax.set_ylim(vmin - pad, vmax + pad)

    for model in sorted(by_model, key=lambda m: (model_sort_key(m), m)):
        model_rows = by_model[model]
        color = _color_for_model(model)
        ms = np.asarray([int(r["m"]) for r in model_rows], dtype=int)
        vals = np.asarray([float(r["croma"]) for r in model_rows], dtype=float)
        ax.plot(ms, vals, color=color, linewidth=plotstyle.LW_SERIES, alpha=0.95, label=model)

    tick_positions = _human_friendly_integer_ticks(m_all, max_ticks=6)
    ax.set_xticks(tick_positions)
    ax.set_xlim(m_min - 0.5, m_max + 0.5)
    ax.set_xlabel("$m$")

    _finalize_wide_line_figure(fig, out_path=out_path, ax=ax)
