"""Shared setup and drawing primitives for the plotting archetypes.

This module holds the common visual-identity wrappers (all routing through
``croma.plotstyle``), the geometry/limit helpers shared across archetypes, and
the shared ``_draw_*`` primitives (notably the one-point-per-model scatter drawer
reused by every scatter wrapper). Archetype-specific ``plot_*`` entry points and
their dedicated ``_draw_*`` helpers live in the per-archetype submodules.
"""

import math

import numpy as np
import matplotlib
from scipy.stats import linregress

matplotlib.use("Agg", force=True)

from croma.confounders import infer_confounder_display_name

from croma import plotstyle
from croma.plotstyle import (
    REFERENCE_LINE_COLOR,
    model_sort_key,
)

# plotstyle.apply_style() runs on import; re-apply defensively in case another
# module reset rcParams after import.
plotstyle.apply_style()


def _family_for_model(model: str) -> str:
    return plotstyle.family_for_model(model)


def _color_for_model(model: str) -> str:
    return plotstyle.color_for_model(model)


def _style_axes(ax, *, grid_axis: str = "both") -> None:
    plotstyle.style_axes(ax, grid_axis=grid_axis)


def _set_panel_title(ax, title: str) -> None:
    plotstyle.set_panel_title(ax, title)


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
    marker_size: float = 52.0,
    exposed: set[str] | frozenset[str] | None = None,
) -> None:
    """Shared one-point-per-model scatter with the project's visual identity.

    Points and the figure legend follow the canonical (family-grouped) model
    order. The data box is forced square via ``set_box_aspect``. Reference
    geometry is configurable: a ``y = x`` diagonal and/or horizontal/vertical
    threshold lines. ``marker_size`` overrides the point area (the Pareto panels
    enlarge it so the family colour reads); ``exposed``, when given, suffixes those
    models' legend labels with a dagger (used when exposure is flagged in the
    legend rather than on the point).
    """
    exposed = set(exposed or ())
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
        label = rf"{model} $\dagger$" if model in exposed else model
        ax.scatter(
            [x],
            [y],
            s=marker_size,
            color=_color_for_model(model),
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
            label=label,
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_box_aspect(1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _set_panel_title(ax, title)


def _draw_trend_line(ax, xs, ys) -> float | None:
    """Faint dotted least-squares fit across the observed x-range: a guide to the eye.

    Returns the fitted slope (``None`` when there is nothing to fit), so a caller that
    also annotates a rank correlation can assert the two agree in sign -- a dotted line
    sloping against the rho printed beside it would be a panel that contradicts itself.

    Least squares, not the rank-robust Theil--Sen that a Spearman rho would otherwise
    suggest: where the relation saturates (CRoMa vs APD on TCGA and Tolkach), the median
    pairwise slope is set by the steep low-CRoMa cluster and the line leaves the panel
    over half its span, tracking no part of the cloud. Minimising vertical residuals is
    what makes a line *look* like the trend. The sign guard above is what keeps that
    convenience honest.

    Drawn beneath the points and only across the data range -- never extrapolated to the
    panel edge, which would assert a fit outside the observed models.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    # Two points make a line, not a trend; a constant x has no slope to estimate.
    if x.size < 3 or float(np.ptp(x)) <= 0.0:
        return None

    fit = linregress(x, y)
    ends = np.array([x.min(), x.max()])
    ax.plot(
        ends,
        fit.intercept + fit.slope * ends,
        linestyle=":",
        linewidth=plotstyle.LW_REFERENCE,
        color=plotstyle.TREND_LINE_COLOR,
        alpha=plotstyle.TREND_ALPHA,
        zorder=2,
    )
    return float(fit.slope)


def _padded_signed_limits(values: np.ndarray) -> tuple[float, float]:
    """Padded limits for a signed quantity (e.g. the CRoMa margin in ``(-1, 1)``).

    The lower bound is not clamped to 0, so fragile models with negative margins
    are not clipped.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    span = float(vmax - vmin)
    if span <= 1e-9:
        pad = max(0.1, abs(vmin) * 0.10, 0.05)
    else:
        pad = max(0.1, span * 0.10)
    lo = vmin - pad
    hi = vmax + pad
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _pareto_frontier_max_max(points: list[tuple[str, float, float]]) -> list[str]:
    """Names of the non-dominated points under a *max-max* preference.

    Each point is ``(name, x, y)``; larger ``x`` **and** larger ``y`` are both better (for the
    Pareto figure, ``x`` is the median margin and ``y`` is ``LTM`` -- a milder, less-negative
    tail is the larger value). A point ``p`` is dominated iff some other point ``q`` satisfies
    ``q.x >= p.x`` and ``q.y >= p.y`` with at least one inequality strict; points equal on both
    coordinates keep each other. The returned names are ordered by ``x`` ascending -- the order
    the frontier staircase is drawn, upper-left (mildest tail) to lower-right (highest median).

    Ties on ``x`` are resolved before the sweep by ordering ``y`` descending, so an equal-``x``
    point with a smaller ``y`` is correctly dropped as dominated.
    """
    ordered = sorted(points, key=lambda p: (-p[1], -p[2]))
    frontier: list[tuple[str, float, float]] = []
    best_y = -math.inf
    for name, x, y in ordered:
        if y > best_y:
            frontier.append((name, x, y))
            best_y = y
    frontier.sort(key=lambda p: (p[1], -p[2]))
    return [name for name, _x, _y in frontier]


def _valid_croma_ltm_rows(rows: list[dict]) -> list[dict]:
    valid: list[dict] = []
    for row in rows:
        if "croma" not in row or "croma_ltm_alpha" not in row:
            continue
        try:
            croma_value = float(row["croma"])
            ltm_value = float(row["croma_ltm_alpha"])
        except Exception:  # noqa: BLE001
            continue
        if not np.isfinite(croma_value) or not np.isfinite(ltm_value):
            continue

        try:
            alpha_value = float(row.get("croma_alpha", float("nan")))
        except Exception:  # noqa: BLE001
            alpha_value = float("nan")

        valid.append(
            {
                "model": str(row.get("model", "")),
                "croma": croma_value,
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
    return "LTM(CRoMa)"
