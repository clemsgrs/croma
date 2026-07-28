"""Distribution archetype: per-sample ridge-line distributions across models.

Covers the per-sample CRoMa distributions. ``_load_croma_sample_rows`` reads and ranks
the per-model sample arrays, ``_draw_croma_sample_distributions`` renders them onto a
supplied axes, and ``plot_croma_sample_distributions`` is the figure-level wrapper.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

from . import style as plotstyle
from .style import (
    COL_DOUBLE,
    FRAGILE_SHADE_COLOR,
    MUTED_TEXT_COLOR,
    REFERENCE_LINE_COLOR,
    SPINE_COLOR,
    TEXT_COLOR,
)

from .base import _color_for_model, _style_axes
from .export import _finalize_figure

# Per-sample CRoMa is a signed, normalised margin bounded by construction to (-1, 1)
# -- see croma.metrics.croma._compute_sample_croma. Framing every benchmark on the
# full domain keeps the zero-margin boundary at the centre (so the shaded fragile
# region reads as exactly half the panel) and makes the figure comparable across
# benchmarks, which data-driven limits would not be.
CROMA_DOMAIN = (-1.0, 1.0)
CROMA_TICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)

# One ridgeline per model, so the figure height scales with the roster: 3.5in for the
# 4 slide encoders, 10.3in for the 21 tile encoders. Reserve the title and axis-label
# bands in inches rather than in figure fractions -- a fraction that leaves room for
# the x-label on the tall figure silently clips it off the short one.
_TITLE_BAND_IN = 0.60  # suptitle + subtitle, above the axes
_XLABEL_BAND_IN = 0.52  # tick labels + x-label, below the axes
_TITLE_BASELINE_IN = 0.16  # suptitle baseline, below the figure top
_SUBTITLE_BASELINE_IN = 0.36  # subtitle baseline, below the figure top


def _load_croma_sample_rows(
    rows: list[dict], models: list[str] | None = None
) -> list[dict]:
    """Load each model's per-sample CRoMa array, ranked by pooled CRoMa (best first).

    ``models`` optionally restricts the roster to a curated subset, selected by model name
    and order-independent -- the returned rows are always ranked by pooled CRoMa so a subset
    stays a faithful zoom-in of the full ridgeline, best on top. A name in ``models`` with no
    finite CRoMa row is silently skipped, same as any other missing row.
    """
    wanted = set(models) if models is not None else None
    model_data: list[dict] = []
    for row in rows:
        if "croma_samples_path" not in row:
            continue
        if wanted is not None and str(row["model"]) not in wanted:
            continue
        if not np.isfinite(float(row.get("croma", float("nan")))):
            continue
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
                "values": values,
                "croma": float(row["croma"]),
            }
        )
    return sorted(
        model_data,
        key=lambda d: (float(d["croma"]), str(d["model"])),
        reverse=True,
    )


def _draw_croma_sample_distributions(ax, model_data: list[dict]) -> None:
    if not model_data:
        ax.set_visible(False)
        return

    x_min, x_max = CROMA_DOMAIN
    x_grid = np.linspace(x_min, x_max, 512)

    _style_axes(ax)
    # Ridgelines read best with only the bottom axis (no box).
    for spine_name in ("top", "right", "left"):
        ax.spines[spine_name].set_visible(False)
    ax.grid(False)
    ax.set_yticks([])

    # Shade the fragile region (CRoMa < 0), exactly the left half of the domain.
    ax.axvspan(x_min, 0.0, color=FRAGILE_SHADE_COLOR, alpha=0.45, zorder=1)
    ax.axvline(
        x=0.0,
        linestyle="--",
        linewidth=plotstyle.LW_REFERENCE,
        color=REFERENCE_LINE_COLOR,
        zorder=2,
        alpha=0.75,
    )

    row_centers = np.arange(len(model_data), 0, -1, dtype=float)
    amplitude = 0.72
    rendered_rows: list[tuple[dict, np.ndarray, np.ndarray | None, np.ndarray, float]] = []
    for d in model_data:
        values = d["values"]
        try:
            density = gaussian_kde(values, bw_method="scott")(x_grid)
        except Exception:
            density = None

        if density is None:
            counts, edges = np.histogram(values, bins=40, range=(x_min, x_max), density=True)
            x_values = 0.5 * (edges[:-1] + edges[1:])
            y_values = counts
        else:
            y_values = density
            x_values = x_grid
        peak = float(np.nanmax(y_values)) if len(y_values) > 0 else 0.0
        rendered_rows.append((d, np.asarray(x_values, dtype=float), density, np.asarray(y_values, dtype=float), peak))

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
    ax.set_xticks(list(CROMA_TICKS))
    ax.set_ylim(0.5, float(len(model_data)) + 0.95)
    ax.set_xlabel("Per-sample CRoMa")
    ax.set_ylabel("")


def _ridgeline_figure_height(n_models: int) -> float:
    return max(3.5, 0.85 + 0.45 * max(1, n_models))


def _ridgeline_bands(fig_height: float) -> dict[str, float]:
    """Figure-fraction positions for a fixed physical title and x-label band."""
    return {
        "top": 1.0 - _TITLE_BAND_IN / fig_height,
        "bottom": _XLABEL_BAND_IN / fig_height,
        "title_y": 1.0 - _TITLE_BASELINE_IN / fig_height,
        "subtitle_y": 1.0 - _SUBTITLE_BASELINE_IN / fig_height,
    }


def plot_croma_sample_distributions(
    rows: list[dict], out_path: Path, models: list[str] | None = None
) -> None:
    """Render the per-sample CRoMa ridgeline for ``rows`` (optionally a subset).

    Passing ``models`` restricts the figure to a curated subset; the default draws the full
    roster. The clean ridgeline style is fixed either way.
    """
    model_data = _load_croma_sample_rows(rows, models=models)
    fig_height = _ridgeline_figure_height(len(model_data))
    bands = _ridgeline_bands(fig_height)
    fig, ax = plt.subplots(figsize=(COL_DOUBLE, fig_height))
    _draw_croma_sample_distributions(ax, model_data)
    fig.suptitle(
        "Per-sample CRoMa distributions",
        fontsize=plotstyle.FS_TITLE,
        weight="bold",
        y=bands["title_y"],
        color=TEXT_COLOR,
    )
    fig.text(
        0.5,
        bands["subtitle_y"],
        "sorted by CRoMa; shaded: CRoMa < 0 (confounder-dominant)",
        ha="center",
        va="center",
        fontsize=plotstyle.FS_ANNOT,
        color=MUTED_TEXT_COLOR,
    )
    _finalize_figure(
        fig,
        out_path=out_path,
        legend_axes=[ax],
        add_legend=False,
        left=0.135,
        right=0.985,
        top=bands["top"],
        bottom=bands["bottom"],
    )
