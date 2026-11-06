"""Scatter archetype: one-point-per-model scatters.

Every wrapper builds on the shared ``_draw_model_scatter`` primitive in ``base``
via a thin per-comparison ``_draw_*`` helper (biological vs confounder, MaRI vs
RI, CRoMa vs MaRI, Q-alpha vs CRoMa) or, for the CRoMa/LTM scatter, by calling
the primitive directly. The named wrappers stay separate for discoverability.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from . import style as plotstyle
from .style import COL_ONEHALF, DEFAULT_DPI

#: The Pareto panels enlarge the default scatter marker so each family colour reads clearly, with
#: the hollow frontier ring scaled to sit outside the bigger fill. Benchmark-exposed encoders are
#: NOT marked on the point -- an on-marker dot or cross competed with the fill colour and hurt
#: readability -- but by a dagger after their name in its point label; the caption spells the
#: dagger out.
_PARETO_MARKER_S = 90.0
_PARETO_RING_S = 210.0

#: Every point on a Pareto panel is labelled directly and the legend is gone: at 25 encoders a
#: family palette collapses into half a dozen near-identical blues, and matching a dot to a
#: 25-swatch legend is a lookup nobody completes. Bold tags name the frontier (the panel's
#: claim); these smaller muted labels name everyone else, so colour is a family cue rather than
#: the sole identity channel.
_POINT_LABEL_FS = 6.0

#: Shared geometry of both Pareto panels. One definition, because ``_pareto_pt_scale`` converts
#: typographic points to data units from exactly these numbers -- a margin edited in one place
#: but not the other silently mis-sizes every reserved label box.
_PARETO_FIGSIZE = (COL_ONEHALF, 5.2)
_PARETO_MARGINS = dict(top=0.95, bottom=0.11, left=0.135, right=0.835)

#: The Pareto panels encode biological accuracy on the point fill, read off a colour bar in the
#: strip the right margin leaves: panel-high, beside the plot, clear of its frontier tags.
_PARETO_COLORBAR_RECT = (
    _PARETO_MARGINS["right"] + 0.025,
    _PARETO_MARGINS["bottom"],
    0.022,
    _PARETO_MARGINS["top"] - _PARETO_MARGINS["bottom"],
)


def _sequential_fill(values: dict[str, float], *, low_is_best: bool = False):
    """Each model's fill on the shared sequential ramp, plus the mappable its colour bar reads.

    The best value takes the ramp's dark end: the highest one, or the lowest when
    ``low_is_best`` (a rank). A rank's colour bar is flipped so that its best end, too, is dark
    and on top.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ramp = plotstyle.sequential_cmap()
    lo, hi = min(values.values()), max(values.values())
    norm = Normalize(vmin=lo, vmax=hi)
    position = (lambda v: 1.0 - float(norm(v))) if low_is_best else (lambda v: float(norm(v)))
    fills = {model: ramp(position(v)) for model, v in values.items()}
    mappable = ScalarMappable(norm=norm, cmap=ramp.reversed() if low_is_best else ramp)
    return fills, mappable


def _pareto_colorbar(fig, mappable, *, label: str, low_is_best: bool = False) -> None:
    bar = plotstyle.add_colorbar(fig, mappable, rect=_PARETO_COLORBAR_RECT, label=label)
    if low_is_best:
        bar.ax.invert_yaxis()


def _daggered(name: str, exposed: set[str] | frozenset[str]) -> str:
    """The display label for a point: the model's short published name, daggered when
    exposed. ``exposed`` holds the names as the rows carry them."""
    label = plotstyle.published_model_name(name, short=True)
    return rf"{label} $\dagger$" if name in exposed else label


def _halo(linewidth: float) -> list:
    """A white outline for in-plot text, so a label stays legible where a leader, the frontier
    staircase or a gridline passes behind it. Used for the bold frontier tags."""
    return [pe.withStroke(linewidth=linewidth, foreground="white")]


from .base import (
    _confounder_display_name,
    _draw_model_scatter,
    _ltm_label,
    _padded_signed_limits,
    _padded_unit_interval_limits,
    _pareto_frontier_max_max,
    _valid_croma_ltm_rows,
)
from .export import _finalize_figure, _finalize_single_panel_legend_figure


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


def _draw_croma_vs_mari_scatter(ax, rows: list[dict]) -> None:
    croma_rows = [
        r for r in rows if "croma" in r and "mari" in r and np.isfinite(float(r["croma"]))
    ]
    if not croma_rows:
        ax.set_visible(False)
        return
    xs = np.asarray([float(r["mari"]) for r in croma_rows], dtype=float)
    ys = np.asarray([float(r["croma"]) for r in croma_rows], dtype=float)
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    _draw_model_scatter(
        ax,
        croma_rows,
        x_key="mari",
        y_key="croma",
        xlabel="MaRI",
        ylabel="CRoMa",
        title="CRoMa vs MaRI",
        xlim=_padded_unit_interval_limits(xs),
        ylim=(float(ys.min()) - y_pad, float(ys.max()) + y_pad),
        hline=0.0,
        vline=0.5,
    )


def _draw_q_alpha_vs_croma_scatter(ax, rows: list[dict]) -> None:
    valid_rows = [
        r
        for r in rows
        if "croma" in r
        and "croma_q_alpha" in r
        and np.isfinite(float(r["croma"]))
        and np.isfinite(float(r["croma_q_alpha"]))
    ]
    if not valid_rows:
        ax.set_visible(False)
        return

    alpha_pct_values = [
        int(round(float(r["croma_alpha"]) * 100))
        for r in valid_rows
        if "croma_alpha" in r and np.isfinite(float(r["croma_alpha"]))
    ]
    alpha_pct = alpha_pct_values[0] if alpha_pct_values else 10

    xs = np.asarray([float(r["croma"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["croma_q_alpha"]) for r in valid_rows], dtype=float)
    x_pad = max(0.1, (xs.max() - xs.min()) * 0.10) if xs.size > 0 else 0.5
    y_pad = max(0.1, (ys.max() - ys.min()) * 0.10) if ys.size > 0 else 0.5
    _draw_model_scatter(
        ax,
        valid_rows,
        x_key="croma",
        y_key="croma_q_alpha",
        xlabel="CRoMa",
        ylabel=f"Q{alpha_pct}",
        title=f"Q{alpha_pct} vs CRoMa",
        xlim=(float(xs.min()) - x_pad, float(xs.max()) + x_pad),
        ylim=(float(ys.min()) - y_pad, float(ys.max()) + y_pad),
        hline=0.0,
        vline=0.0,
    )


def plot_bio_vs_confounder_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_bio_vs_confounder_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_mari_vs_ri_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_mari_vs_ri_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_croma_ltm_scatter(rows: list[dict], out_path: Path) -> None:
    """CRoMa vs LTM scatter with a horizontal CRoMa=0 robustness threshold.

    The threshold line (not a y=x diagonal) makes the claim non-tautological: every
    model's fragile decile falling below it is an empirical fact, since LTM <= median
    CRoMa by construction would only force points below the diagonal, not below 0.
    """
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    valid_rows = _valid_croma_ltm_rows(rows)

    if not valid_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    label_ltm = _ltm_label(valid_rows)
    xs = np.asarray([float(r["croma"]) for r in valid_rows], dtype=float)
    ys = np.asarray([float(r["ltm"]) for r in valid_rows], dtype=float)
    lim = _padded_signed_limits(np.concatenate([xs, ys]))

    _draw_model_scatter(
        ax,
        valid_rows,
        x_key="croma",
        y_key="ltm",
        xlabel="CRoMa",
        ylabel=label_ltm,
        title=f"CRoMa vs {label_ltm}",
        xlim=lim,
        ylim=lim,
        hline=0.0,
    )

    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_croma_vs_mari_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_croma_vs_mari_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def plot_q_alpha_vs_croma_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 5.6))
    _draw_q_alpha_vs_croma_scatter(ax, rows)
    _finalize_single_panel_legend_figure(fig, out_path=out_path, ax=ax)


def _pareto_boundary(
    frontier_xy: list[tuple[float, float]],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> list[tuple[float, float]]:
    """The separating staircase from upper-left (mildest tail) to lower-right (highest median).

    ``frontier_xy`` is the non-dominated set, ``x`` ascending. The boundary runs in at the top
    frontier point's ``y`` level, steps down-and-right through each frontier point, and drops to
    the axis floor beneath the highest-median point: everything below-left of it is dominated on
    both axes. Extending to the panel edges is honest here -- unlike a fitted trend, the Pareto
    boundary is *defined* for every ``y``, so the edges carry no claim beyond the data.
    """
    xlo, _xhi = xlim
    _ylo_pad, _yhi = ylim
    fx = [x for x, _ in frontier_xy]
    fy = [y for _, y in frontier_xy]
    boundary: list[tuple[float, float]] = [(xlo, fy[0])]
    for i, (x, y) in enumerate(frontier_xy):
        boundary.append((x, y))
        if i + 1 < len(frontier_xy):
            boundary.append((x, fy[i + 1]))  # step down at this x to the next (lower) tail
    boundary.append((fx[-1], ylim[0]))  # drop to the floor beneath the highest-median model
    return boundary


def _pareto_pt_scale(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, float]:
    """Data units per typographic point for the fixed Pareto figure cell (the shared
    ``_PARETO_FIGSIZE``/``_PARETO_MARGINS`` both panels pass to ``_finalize_figure``), so tag
    room can be reserved and labels laid out before the axes exist."""
    ax_w_pt = (_PARETO_MARGINS["right"] - _PARETO_MARGINS["left"]) * _PARETO_FIGSIZE[0] * 72.0
    ax_h_pt = (_PARETO_MARGINS["top"] - _PARETO_MARGINS["bottom"]) * _PARETO_FIGSIZE[1] * 72.0
    return (xlim[1] - xlim[0]) / ax_w_pt, (ylim[1] - ylim[0]) / ax_h_pt


#: Frontier-tag geometry, in typographic points: a tag's preferred spot is this far beside its
#: ring (on its routed side) and this far above it -- the spot every tag took before the crowded
#: corner of a 27-encoder roster forced a search.
_TAG_DX_PT = 9.0
_TAG_DY_PT = 8.0
_RING_R_PT = (_PARETO_RING_S / np.pi) ** 0.5
_MARKER_R_PT = (_PARETO_MARKER_S / np.pi) ** 0.5
#: Clearance kept between a label and anything it must not touch, in points.
_LABEL_PAD_PT = 1.0


def _text_extent_pt(text: str, fontsize: float, *, bold: bool = False) -> tuple[float, float]:
    """``text``'s drawn width and height in points, measured in the font it will be drawn in.

    Labels are laid out before the axes exist, so their boxes cannot be read off the figure. A
    per-character estimate drifts by several points on a long bold name -- exactly the slack a
    crowded corner does not have -- so the text is measured by the renderers that draw it.

    The width is the wider of the two exports': the PDF sets unhinted glyphs (and a mathtext
    run, such as a dagger, in a box rounded up to whole points), and the PNG sets glyphs hinted
    at ``DEFAULT_DPI``. Both run up to a couple of points wider than the 72-dpi hinted measure,
    which let two labels a pad apart touch in print. The height is the 72-dpi hinted line box
    (one pixel is one point), at least that of ``"lp"`` as matplotlib sizes a line: a little
    taller than either export draws it, a margin the crowded corners rely on.
    """
    from matplotlib.backends.backend_agg import RendererAgg
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import text_to_path

    prop = FontProperties(size=fontsize, weight="bold" if bold else "normal")
    ismath = "$" in text
    # The PDF backend's own measure: mathtext parsed at the drawn size, plain text unhinted.
    if ismath:
        pdf = text_to_path.mathtext_parser.parse(text, 72, prop).width
    else:
        pdf, _, _ = text_to_path.get_text_width_height_descent(text, prop, ismath=False)
    png, _, _ = RendererAgg(1, 1, DEFAULT_DPI).get_text_width_height_descent(
        text, prop, ismath=ismath
    )
    renderer = RendererAgg(1, 1, 72)
    _, height, _ = renderer.get_text_width_height_descent(text, prop, ismath=ismath)
    _, line, _ = renderer.get_text_width_height_descent("lp", prop, ismath=False)
    return float(max(pdf, png * 72.0 / DEFAULT_DPI)), float(max(height, line))


def _pad(box, by: float = _LABEL_PAD_PT):
    return (box[0] - by, box[1] + by, box[2] - by, box[3] + by)


def _box_overlap(a, b) -> float:
    """Area shared by two ``(x0, x1, y0, y1)`` boxes."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[2], b[2])
    )


def _disk_reach(box, disk) -> float:
    """How far a ``(cx, cy, r)`` disk reaches into ``box``: 0 when clear."""
    cx, cy, r = disk
    nx = min(max(cx, box[0]), box[1])
    ny = min(max(cy, box[2]), box[3])
    return max(0.0, r - float(np.hypot(nx - cx, ny - cy)))


def _segment_reach(p, q, disk) -> float:
    """How far a ``(cx, cy, r)`` disk reaches across the segment ``p``--``q``: 0 when clear."""
    cx, cy, r = disk
    dx, dy = q[0] - p[0], q[1] - p[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else min(1.0, max(0.0, ((cx - p[0]) * dx + (cy - p[1]) * dy) / length2))
    return max(0.0, r - float(np.hypot(p[0] + t * dx - cx, p[1] + t * dy - cy)))


def _line_inside(box, segments) -> float:
    """Total length of the axis-aligned ``segments`` that runs through ``box``."""
    total = 0.0
    for (x0, y0), (x1, y1) in segments:
        if y0 == y1 and box[2] < y0 < box[3]:
            total += max(0.0, min(max(x0, x1), box[1]) - max(min(x0, x1), box[0]))
        elif x0 == x1 and box[0] < x0 < box[1]:
            total += max(0.0, min(max(y0, y1), box[3]) - max(min(y0, y1), box[2]))
    return total


def _segment_inside(p, q, box) -> float:
    """Length of the segment ``p``--``q`` that runs through ``box`` (Liang-Barsky clipping)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    edges = ((-dx, p[0] - box[0]), (dx, box[1] - p[0]), (-dy, p[1] - box[2]), (dy, box[3] - p[1]))
    for step, room in edges:
        if step == 0:
            if room < 0:
                return 0.0
            continue
        t = room / step
        if step < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 >= t1:
            return 0.0
    return (t1 - t0) * float(np.hypot(dx, dy))


def _outside(box, width: float, height: float) -> float:
    """How far ``box`` pokes out of the ``width`` x ``height`` panel, summed over its edges."""
    return (
        max(0.0, -box[0]) + max(0.0, box[1] - width) + max(0.0, -box[2]) + max(0.0, box[3] - height)
    )


class _PanelPoints:
    """Data <-> typographic-point conversion for one Pareto panel's final limits.

    Laying labels out in points keeps the geometry isotropic -- a disk is round, a gap of 2 pt is
    2 pt both ways -- whatever the aspect of the data ranges.
    """

    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        self.xlo, self.ylo = xlim[0], ylim[0]
        self.x_per_pt, self.y_per_pt = _pareto_pt_scale(xlim, ylim)
        self.width = (xlim[1] - xlim[0]) / self.x_per_pt
        self.height = (ylim[1] - ylim[0]) / self.y_per_pt

    def pt(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.xlo) / self.x_per_pt, (y - self.ylo) / self.y_per_pt

    def data(self, x: float, y: float) -> tuple[float, float]:
        return self.xlo + x * self.x_per_pt, self.ylo + y * self.y_per_pt

    def box_pt(self, box):
        x0, y0 = self.pt(box[0], box[2])
        x1, y1 = self.pt(box[1], box[3])
        return (x0, x1, y0, y1)

    def box_data(self, box):
        x0, y0 = self.data(box[0], box[2])
        x1, y1 = self.data(box[1], box[3])
        return (x0, x1, y0, y1)

    def disks(self, coords, frontier_xy) -> list[tuple[float, float, float]]:
        """Every point as a disk in points: a frontier member as its ring, the rest as dots."""
        ringed = set(frontier_xy)
        return [
            (*self.pt(x, y), _RING_R_PT if (x, y) in ringed else _MARKER_R_PT) for x, y in coords
        ]

    def segments(self, polyline) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        pts = [self.pt(x, y) for x, y in polyline]
        return list(zip(pts[:-1], pts[1:]))


def _layout_frontier_tags(
    frontier: list[str],
    frontier_xy: list[tuple[float, float]],
    all_xy: list[tuple[float, float]],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], list[dict]]:
    """Place the bold frontier tags and expand the limits so each lands inside the panel.

    Shared by both Pareto panels, which differ only in how they build the base limits. Each tag
    prefers the spot up-and-right of its ring, in the empty non-dominated wedge (the Pareto
    property guarantees no other model point sits there), joined by a thin leader; a frontier
    point within 8% of the widest ``x`` in the cloud is at the right edge, where the up-right wedge
    is off-panel, so it prefers up-and-LEFT into the open interior instead (extending the axis
    rightward there would open a wide empty band). Right/top room is reserved per tag for that
    preferred spot.

    The preferred spot is only a preference. On a crowded roster the frontier rings themselves
    cluster -- three within a ring's width of each other in a corner -- and a tag there lands on a
    neighbour's ring or a dominated point, hiding what it should name. So each tag, highest ring
    first, scores every spot on a grid around its ring and takes the best: a tag never covers a
    point or ring (its own included), another tag or the panel edge, unless no spot avoids it;
    then, less weightily, its leader should not cut through other points and it should not sit
    on the frontier staircase; and, least, it should stay near its preferred spot. A clear
    preferred spot therefore always wins. Returns the expanded ``(xlim, ylim)`` and the per-tag
    draw specs for ``_draw_frontier_tags`` (called after the scatter, so tags sit on top); each
    spec also carries its ``box``, which the point labels then keep clear of.
    """
    xlo, xhi = xlim
    ylo, yhi = ylim
    if not frontier:
        return (xlo, xhi), (ylo, yhi), []
    x_per_pt, y_per_pt = _pareto_pt_scale(xlim, ylim)
    extents = [_text_extent_pt(name, plotstyle.FS_ANNOT, bold=True) for name in frontier]

    xs = [x for x, _ in all_xy]
    xspan = float(max(xs) - min(xs))
    xmax = float(max(xs))
    tag_left = [(xmax - fx) < 0.08 * xspan for fx, _fy in frontier_xy]
    for (width, _h), left, (fx, _fy) in zip(extents, tag_left, frontier_xy):
        room = (width + _TAG_DX_PT + 3.0) * x_per_pt
        if left:
            xlo = min(xlo, fx - room)
        else:
            xhi = max(xhi, fx + room)
    top = max(y for _, y in frontier_xy)
    yhi = max(yhi, top + (_TAG_DY_PT + 1.35 * plotstyle.FS_ANNOT) * y_per_pt)

    xlim, ylim = (xlo, xhi), (ylo, yhi)
    panel = _PanelPoints(xlim, ylim)
    disks = panel.disks(all_xy, frontier_xy)
    staircase = panel.segments(_pareto_boundary(frontier_xy, xlim=xlim, ylim=ylim))

    placed: list[tuple] = []  # (box, leader start, ring) of every tag laid so far, in points
    tags: list[dict | None] = [None] * len(frontier)
    for i in sorted(range(len(frontier)), key=lambda i: frontier_xy[i][1], reverse=True):
        rx, ry = panel.pt(*frontier_xy[i])
        width, height = extents[i]
        own = (rx, ry, _RING_R_PT)
        others = [d for d in disks if np.hypot(d[0] - rx, d[1] - ry) > 1e-9]
        beside = _TAG_DX_PT + 2.0  # the text starts 2 pt past the leader's end, as drawn
        pref_x0 = rx - beside - width if tag_left[i] else rx + beside
        pref_y = ry + _TAG_DY_PT
        x0s = np.arange(rx - beside - width - 18.0, rx + beside + 18.0 + 1e-9, 3.0)
        x0s = np.union1d(x0s, [rx - beside - width, rx + beside])
        best = None
        for y in ry + np.arange(-60.0, 60.0 + 1e-9, 2.0):
            for x0 in x0s:
                box = (float(x0), float(x0) + width, float(y) - height / 2, float(y) + height / 2)
                pad = _pad(box)
                hard = (
                    sum(_disk_reach(pad, d) * 2.0 * d[2] for d in (own, *others))
                    + sum(_box_overlap(pad, b) for b, _s, _r in placed)
                    + 50.0 * _outside(box, panel.width, panel.height) * height
                )
                start = (min(max(rx, pad[0]), pad[1]), min(max(ry, pad[2]), pad[3]))
                soft = (
                    3.0 * sum(_segment_reach(start, (rx, ry), d) for d in others)
                    + 3.0 * sum(_disk_reach(pad, (s[0], s[1], 0.5)) for _b, s, _r in placed)
                    + 0.3 * _line_inside(box, staircase)
                )
                for b, _s, _r in placed:  # do not cut through a tag already laid
                    blocker = (*_box_centre(b), min(_box_half(b)))
                    soft += 3.0 * _segment_reach(start, (rx, ry), blocker)
                score = 1000.0 * hard + soft + 0.5 * float(np.hypot(x0 - pref_x0, y - pref_y))
                if best is None or score < best[0]:
                    best = (score, box, start)
        _score, box, start = best
        placed.append((box, start, (rx, ry)))
        right_edge = (box[0] + box[1]) / 2 < rx  # anchor the edge that faces the ring
        anchor = (box[1], (box[2] + box[3]) / 2) if right_edge else (box[0], (box[2] + box[3]) / 2)
        tags[i] = {
            "name": frontier[i],
            "ring": frontier_xy[i],
            "anchor": panel.data(*anchor),
            "leader_from": panel.data(*start),
            "box": panel.box_data(box),
            "ha": "right" if right_edge else "left",
            "text_dx": 0,
        }
    return xlim, ylim, [t for t in tags if t is not None]


def _box_centre(box) -> tuple[float, float]:
    return (box[0] + box[1]) / 2, (box[2] + box[3]) / 2


def _box_half(box) -> tuple[float, float]:
    return (box[1] - box[0]) / 2, (box[3] - box[2]) / 2


def _draw_frontier_tags(ax, tags: list[dict]) -> None:
    """Draw the bold, white-haloed frontier labels from ``_layout_frontier_tags`` and their thin
    leaders. The halo keeps each tag legible where a leader or the frontier staircase passes
    behind it."""
    for t in tags:
        fx, fy = t["ring"]
        ax.annotate(
            "",
            xy=(fx, fy),
            xytext=t.get("leader_from", t["anchor"]),
            arrowprops=dict(
                arrowstyle="-", color=plotstyle.SPINE_COLOR, linewidth=0.6, shrinkA=0.0, shrinkB=3.0
            ),
            zorder=4,
        )
        ax.annotate(
            t["name"],
            xy=t["anchor"],
            xytext=(t["text_dx"], 0),
            textcoords="offset points",
            ha=t["ha"],
            va="center",
            fontsize=plotstyle.FS_ANNOT,
            color=plotstyle.TEXT_COLOR,
            weight="bold",
            zorder=6,
            path_effects=_halo(2.6),
        )


def _layout_point_labels(
    labels: dict[str, str],
    coords: dict[str, tuple[float, float]],
    *,
    frontier: list[str],
    frontier_tags: list[dict],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    lines: list[list[tuple[float, float]]] = (),
) -> list[dict]:
    """Place a small label beside every non-frontier point, dodging what is already inked.

    Greedy, top-down: each label tries sixteen bearings at seven radii and keeps the candidate
    that scores best. Covering a point (a frontier ring at its ring's size), a bold
    frontier tag, a label already placed or the panel edge is what a placement must avoid, so
    those dominate the score; a label run through by one of ``lines`` (the frontier staircase,
    the zero references -- each a polyline in data units) or by a leader, or pushed out to a far
    ring, where it needs a leader of its own (one that should not cut through other points or
    labels), is merely discouraged, distance more than a line. Among equally clear spots the
    eight compass points beside the point come first, in a fixed order. The greedy order is
    y-descending so the dodging cascades downward, the direction with room on these
    bottom-heavy panels.
    """
    panel = _PanelPoints(xlim, ylim)
    frontier_xy = [coords[name] for name in frontier]
    disks = panel.disks(list(coords.values()), frontier_xy)
    tag_boxes = [panel.box_pt(t["box"]) for t in frontier_tags if "box" in t]
    segments = [s for line in lines for s in panel.segments(line)]
    # Leaders already drawn -- the frontier tags' and, as they are placed, the far labels' --
    # which a label should not sit on and a new leader should not cut through text to join.
    leaders = [
        (panel.pt(*t["leader_from"]), panel.pt(*t["ring"]))
        for t in frontier_tags
        if "leader_from" in t
    ]

    near = _MARKER_R_PT + 2.0
    compass = [
        (near, 0.0, "left"),
        (-near, 0.0, "right"),
        (0.0, near + 2.5, "center"),
        (0.0, -(near + 2.5), "center"),
        (near - 1.0, near - 1.0, "left"),
        (-(near - 1.0), near - 1.0, "right"),
        (near - 1.0, -(near - 1.0), "left"),
        (-(near - 1.0), -(near - 1.0), "right"),
    ]
    # The eight bearings between the compass points, for a point wedged into a knot of others
    # where none of the eight main ones is clear.
    between = [
        (near * np.cos(a), near * np.sin(a), "left" if np.cos(a) > 0 else "right")
        for a in np.deg2rad([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
    ]
    candidates = [
        (dx * scale, dy * scale, ha, scale)
        for scale in (1.0, 1.45, 1.9, 2.4, 2.8, 3.4, 4.0)
        for dx, dy, ha in compass + between
    ]
    specs: list[dict] = []
    placed: list[tuple[float, float, float, float]] = []
    on_frontier = set(frontier)
    order = sorted(
        (name for name in coords if name not in on_frontier),
        key=lambda name: coords[name][1],
        reverse=True,
    )
    for name in order:
        text = labels[name]
        width, height = _text_extent_pt(text, _POINT_LABEL_FS)
        px, py = panel.pt(*coords[name])
        best = None
        for dx_pt, dy_pt, ha, scale in candidates:
            x = px + dx_pt
            x0 = x if ha == "left" else x - width if ha == "right" else x - width / 2
            box = (x0, x0 + width, py + dy_pt - height / 2, py + dy_pt + height / 2)
            pad = _pad(box, 0.75)
            hard = (
                sum(_disk_reach(pad, d) * 2.0 * d[2] for d in disks)
                + 2.0 * sum(_box_overlap(pad, t) for t in tag_boxes)
                + 3.0 * sum(_box_overlap(pad, p) for p in placed)
                + 50.0 * _outside(box, panel.width, panel.height) * height
            )
            # A far label needs a leader, and a fan of leaders reads worse than a label the halo
            # lifts off a line, so distance outweighs a line crossing; a leader is a line too.
            soft = (
                0.25 * _line_inside(box, segments)
                + 3.0 * sum(_segment_inside(a, b, pad) for a, b in leaders)
                + 12.0 * (scale - 1.0)
            )
            leader = ((x, py + dy_pt), (px, py)) if scale > 1.0 else None
            if leader is not None:
                soft += 3.0 * sum(
                    _segment_reach(*leader, d) for d in disks if (d[0], d[1]) != (px, py)
                ) + 3.0 * sum(_segment_inside(*leader, b) for b in (*placed, *tag_boxes))
            score = 100.0 * hard + soft
            if best is None or score < best[0]:
                best = (score, dx_pt, dy_pt, ha, box, leader)
        _score, dx_pt, dy_pt, ha, box, leader = best
        far = leader is not None
        placed.append(box)
        if far:
            leaders.append(leader)
        specs.append(
            {
                "text": text,
                "point": coords[name],
                "offset": (dx_pt, dy_pt),
                "ha": ha,
                "leader": far,
            }
        )
    return specs


def _draw_point_labels(ax, specs: list[dict]) -> None:
    """Draw the muted non-frontier labels from ``_layout_point_labels``. Haloed like the bold
    tags, so a label crossing the staircase or a gridline stays legible; a label the dodging
    pushed to the far ring gets a thin leader back to its point, because at that distance the
    pairing stops being self-evident."""
    for spec in specs:
        if spec["leader"]:
            ax.annotate(
                "",
                xy=spec["point"],
                xytext=spec["offset"],
                textcoords="offset points",
                arrowprops=dict(
                    arrowstyle="-",
                    color=plotstyle.SPINE_COLOR,
                    linewidth=0.5,
                    shrinkA=1.0,
                    shrinkB=2.5,
                ),
                zorder=4,
            )
        ax.annotate(
            spec["text"],
            xy=spec["point"],
            xytext=spec["offset"],
            textcoords="offset points",
            ha=spec["ha"],
            va="center",
            fontsize=_POINT_LABEL_FS,
            color=plotstyle.MUTED_TEXT_COLOR,
            zorder=5.5,
            path_effects=_halo(1.8),
        )


def plot_croma_pareto(
    rows: list[dict],
    out_path: Path,
    *,
    exposed: set[str] | frozenset[str] | None = None,
    ltm_alpha_pct: int = 10,
) -> None:
    """Median \\code{CRoMa} vs the worst-decile mean ``LTM``, with the Pareto frontier drawn.

    The two irreducible robustness axes -- central tendency (``x``: median margin, higher is
    more biology-dominant) and tail severity (``y``: ``LTM``, higher is a milder tail) -- on
    one panel. Larger on both is better, so the non-dominated set is the upper-right frontier;
    every other encoder is beaten on *both* axes at once. This is the decision procedure the
    tail argument implies: rank by the median, but a model with a co-leading median and a
    heavier tail is Pareto-dominated, not tied.

    ``rows`` carry margin-scale ``croma`` (median) and ``croma_ltm_alpha`` (``LTM``), already
    normalised by the caller (see ``_distributions``). ``exposed`` names the encoders whose
    pretraining overlaps a cohort in this benchmark; each is flagged with a dagger after its name
    in the legend (an on-marker glyph competed with the fill colour), exactly as the rank-aggregate
    overview marks its exposed set.

    Every ringed frontier encoder is labelled in bold on every panel -- the frontier is the
    answer to "so which model is best?", so it is named rather than left to the caption. The
    dashed \\code{CRoMa}\\,=\\,0 and ``LTM``\\,=\\,0 references, the shaded dominated region and
    the frontier staircase are always drawn.
    """
    fig, ax = plt.subplots(figsize=_PARETO_FIGSIZE)
    valid_rows = _valid_croma_ltm_rows(rows)
    if not valid_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    exposed = set(exposed or ())
    coords = {r["model"]: (float(r["croma"]), float(r["ltm"])) for r in valid_rows}
    bio = {str(r["model"]): float(r["bio"]) for r in rows if np.isfinite(r.get("bio", np.nan))}
    unscored = sorted(set(coords) - set(bio))
    if unscored:
        raise ValueError(f"no biological k-NN balanced accuracy ('bio') for {unscored}")
    fills, mappable = _sequential_fill({m: bio[m] for m in coords})
    xs = np.asarray([x for x, _ in coords.values()], dtype=float)
    ys = np.asarray([y for _, y in coords.values()], dtype=float)

    # The non-dominated set, x-ascending / y-descending. Computed before the limits so the room
    # the frontier tags need on the right and top can be folded into them.
    frontier = _pareto_frontier_max_max([(m, x, y) for m, (x, y) in coords.items()])
    frontier_xy = [coords[m] for m in frontier]

    # Keep both zero references in view: the CRoMa=0 (biology vs confounder) vertical and the
    # LTM=0 horizontal -- the unreached ideal of a tail that never crosses into
    # confounder-dominance, so it reads as a ceiling every model sits below. Both bounds reach
    # toward 0 with ``max``/``min``; the y-upper must use ``max`` so the LTM=0 line stays on the
    # panel when every tail is negative (a ``min`` here would clamp it off the top).
    xpad = max(0.03, (float(xs.max()) - float(xs.min())) * 0.08)
    ypad = max(0.03, (float(ys.max()) - float(ys.min())) * 0.08)
    xlo = min(float(xs.min()), 0.0) - xpad
    ylo = float(ys.min()) - ypad
    xhi = max(float(xs.max()), 0.0) + xpad
    yhi = max(float(ys.max()), 0.0) + ypad

    # Reserve room for the bold frontier tags and lay them out (drawn after the scatter, below),
    # so each up-right/up-left tag lands inside the panel rather than clipping the spine.
    frontier_labels = [_daggered(name, exposed) for name in frontier]
    xlim, ylim, frontier_tags = _layout_frontier_tags(
        frontier_labels, frontier_xy, list(coords.values()), xlim=(xlo, xhi), ylim=(ylo, yhi)
    )

    _draw_model_scatter(
        ax,
        valid_rows,
        x_key="croma",
        y_key="ltm",
        xlabel="Median CRoMa  (typical case)",
        ylabel=rf"Tail severity  $\mathrm{{LTM}}_{{{ltm_alpha_pct}}}$",
        title="Median robustness vs tail severity",
        xlim=xlim,
        ylim=ylim,
        hline=0.0,
        vline=0.0,
        marker_size=_PARETO_MARKER_S,
        colors=fills,
    )
    # The scatter primitive forces a square data box (its own identity); here that compresses
    # the axes into the bottom legend. Let the panel fill its figure cell instead (cf.
    # apd_figure), which keeps the x-label clear of the roster legend.
    ax.set_box_aspect(None)
    # Biological accuracy on the fill: is the robust corner bought with class separation?
    _pareto_colorbar(fig, mappable, label="Biological k-NN balanced accuracy")

    boundary = _pareto_boundary(frontier_xy, xlim=xlim, ylim=ylim)

    # Faint shade of the dominated region (below-left of the boundary): everything here is
    # beaten on both axes. Drawn beneath the points and the grid.
    shade = boundary + [(xlim[0], ylim[0])]
    ax.fill(
        [p[0] for p in shade],
        [p[1] for p in shade],
        color=plotstyle.SPINE_COLOR,
        alpha=0.05,
        linewidth=0,
        zorder=0.5,
    )
    # The frontier itself: a solid neutral staircase, heavier than the dashed zero references.
    ax.plot(
        [p[0] for p in boundary],
        [p[1] for p in boundary],
        color=plotstyle.SPINE_COLOR,
        linewidth=1.1,
        zorder=2.5,
        solid_capstyle="butt",
    )
    # Ring the non-dominated encoders so "these are the only undominated choices" reads at a
    # glance.
    ax.scatter(
        [x for x, _ in frontier_xy],
        [y for _, y in frontier_xy],
        s=_PARETO_RING_S,
        facecolors="none",
        edgecolors=plotstyle.SPINE_COLOR,
        linewidths=1.1,
        zorder=5,
    )
    # Label every ringed frontier encoder in bold -- the frontier answers "so which model is
    # best?", so it is named on every panel. Placement (up-right/up-left routing, y-de-collision,
    # white halo) is handled by _layout_frontier_tags/_draw_frontier_tags, shared with the
    # rank-aggregate panel.
    _draw_frontier_tags(ax, frontier_tags)

    # Every dominated encoder gets a small muted label beside its point (exposure daggered
    # there, as on the frontier tags), and the legend is gone -- see _POINT_LABEL_FS.
    _draw_point_labels(
        ax,
        _layout_point_labels(
            {name: _daggered(name, exposed) for name in coords},
            coords,
            frontier=frontier,
            frontier_tags=frontier_tags,
            xlim=xlim,
            ylim=ylim,
            # The staircase and both zero references run through the cloud; a label is
            # nudged off them where a clear spot exists (the halo covers the rest).
            lines=[boundary, [(xlim[0], 0.0), (xlim[1], 0.0)], [(0.0, ylim[0]), (0.0, ylim[1])]],
        ),
    )

    _finalize_figure(fig, out_path=out_path, legend_axes=[ax], add_legend=False, **_PARETO_MARGINS)


def plot_rank_pareto(
    rows: list[dict],
    out_path: Path,
    *,
    n_benchmarks: int = 3,
) -> None:
    r"""Mean median-\code{CRoMa} rank vs mean ``LTM_10`` rank across the tile benchmarks.

    The single-figure overview of the per-benchmark Pareto panels. Each pathology encoder is
    ranked within every tile benchmark by median \code{CRoMa} (rank 1 = highest median) and by
    tail severity ``LTM_10`` (rank 1 = mildest tail); the axes are the mean of those ranks across
    the ``n_benchmarks`` benchmarks (1 = best on average). Ranking is scale-free, so -- unlike
    averaging the raw margins -- a benchmark with wider margins cannot dominate the aggregate,
    and an in-distribution boost on a TCGA-containing benchmark counts as one rank rather than a
    large margin.

    ``rows`` carry ``median_rank``, ``tail_rank`` and an ``exposed`` flag (see ``_rank_pareto``).
    Fewer is better on both axes, so the panel is drawn in negated *goodness* coordinates: the
    best corner is the upper-right, matching the upper-right-is-better reading of the
    per-benchmark Pareto panels, while the tick labels read the positive ranks. The non-dominated set is the
    upper-right frontier, ringed exactly as in ``plot_croma_pareto`` and, as there, bold-labelled
    in place (via the shared ``_layout_frontier_tags``/``_draw_frontier_tags``); the shaded region
    below-left of the staircase is dominated on both axes. TCGA-exposed encoders carry a dagger
    after their name in the legend, because two of the three benchmarks contain a TCGA cohort, so an
    exposed encoder's mean rank may still reflect pretraining overlap.
    """
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    fig, ax = plt.subplots(figsize=_PARETO_FIGSIZE)
    points = [
        (
            str(r["model"]),
            float(r["median_rank"]),
            float(r["tail_rank"]),
            bool(r.get("exposed", False)),
        )
        for r in rows
        if "median_rank" in r
        and "tail_rank" in r
        and np.isfinite(float(r["median_rank"]))
        and np.isfinite(float(r["tail_rank"]))
    ]
    if not points:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    # Goodness coordinates: negate the ranks so larger is better and the frontier is upper-right,
    # exactly as in plot_croma_pareto. A FuncFormatter restores the positive rank on the ticks,
    # and a MultipleLocator keeps them at whole ranks (5, 10, 15, ...) rather than half-ranks.
    coords = {name: (-mr, -tr) for name, mr, tr, _ in points}
    exposed = {name for name, _, _, is_exposed in points if is_exposed}
    # Raw balanced accuracy is not commensurable across benchmarks (TCGA-4x4 tops out lower
    # than Camelyon), so the overview's fill is the mean biological-accuracy rank instead.
    bio_rank = {
        str(r["model"]): float(r["bio_rank"])
        for r in rows
        if np.isfinite(float(r.get("bio_rank", np.nan)))
    }
    unscored = sorted(set(coords) - set(bio_rank))
    if unscored:
        raise ValueError(f"no mean biological-accuracy rank ('bio_rank') for {unscored}")
    fills, mappable = _sequential_fill({m: bio_rank[m] for m in coords}, low_is_best=True)
    scatter_rows = [{"model": name, "gx": gx, "gy": gy} for name, (gx, gy) in coords.items()]

    gxs = np.asarray([gx for gx, _ in coords.values()], dtype=float)
    gys = np.asarray([gy for _, gy in coords.values()], dtype=float)
    xpad = max(0.4, (float(gxs.max()) - float(gxs.min())) * 0.08)
    ypad = max(0.4, (float(gys.max()) - float(gys.min())) * 0.08)
    xlim = (float(gxs.min()) - xpad, float(gxs.max()) + xpad)
    ylim = (float(gys.min()) - ypad, float(gys.max()) + ypad)

    # The best-rank frontier rings sit in the extreme upper-right corner; give them a floor of room
    # to clear the top and right spines (the ring is ~16 pt across) rather than being cropped.
    x_per_pt, y_per_pt = _pareto_pt_scale(xlim, ylim)
    xlim = (xlim[0], max(xlim[1], float(gxs.max()) + 11.0 * x_per_pt))
    ylim = (ylim[0], max(ylim[1], float(gys.max()) + 11.0 * y_per_pt))

    # Bold-label the ringed frontier here too, exactly as the per-benchmark panels do: compute the
    # non-dominated set, reserve the room its tags need, and lay them out (drawn after the scatter).
    frontier = _pareto_frontier_max_max([(name, gx, gy) for name, (gx, gy) in coords.items()])
    frontier_xy = [coords[name] for name in frontier]
    frontier_labels = [_daggered(name, exposed) for name in frontier]
    xlim, ylim, frontier_tags = _layout_frontier_tags(
        frontier_labels, frontier_xy, list(coords.values()), xlim=xlim, ylim=ylim
    )

    _draw_model_scatter(
        ax,
        scatter_rows,
        x_key="gx",
        y_key="gy",
        xlabel="Mean rank by median CRoMa  (1 = best)",
        ylabel=r"Mean rank by tail severity $\mathrm{LTM}_{10}$  (1 = best)",
        title=f"Aggregate robustness across {n_benchmarks} tile benchmarks",
        xlim=xlim,
        ylim=ylim,
        marker_size=_PARETO_MARKER_S,
        colors=fills,
    )
    ax.set_box_aspect(None)
    _pareto_colorbar(
        fig,
        mappable,
        label="Mean rank by biological k-NN accuracy  (1 = best)",
        low_is_best=True,
    )
    # Ranks start at 1: the 0 tick that the room above the best point can reach is left bare
    # rather than printed as "-0".
    show_rank = FuncFormatter(lambda value, _pos: f"{-value:.0f}" if -value >= 0.5 else "")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(5))
        axis.set_major_formatter(show_rank)

    boundary = _pareto_boundary(frontier_xy, xlim=xlim, ylim=ylim)

    shade = boundary + [(xlim[0], ylim[0])]
    ax.fill(
        [p[0] for p in shade],
        [p[1] for p in shade],
        color=plotstyle.SPINE_COLOR,
        alpha=0.05,
        linewidth=0,
        zorder=0.5,
    )
    ax.plot(
        [p[0] for p in boundary],
        [p[1] for p in boundary],
        color=plotstyle.SPINE_COLOR,
        linewidth=1.1,
        zorder=2.5,
        solid_capstyle="butt",
    )
    ax.scatter(
        [x for x, _ in frontier_xy],
        [y for _, y in frontier_xy],
        s=_PARETO_RING_S,
        facecolors="none",
        edgecolors=plotstyle.SPINE_COLOR,
        linewidths=1.1,
        zorder=5,
    )
    # Bold frontier labels, drawn by the same helper the per-benchmark panels use.
    _draw_frontier_tags(ax, frontier_tags)

    # Every dominated encoder gets a small muted label beside its point. TCGA-exposed
    # encoders (two of the three benchmarks contain a TCGA cohort, so an exposed encoder's
    # mean rank may reflect pretraining overlap) are daggered in their labels, never by a
    # glyph on the point: an on-marker dot or cross competed with the fill colour.
    _draw_point_labels(
        ax,
        _layout_point_labels(
            {name: _daggered(name, exposed) for name in coords},
            coords,
            frontier=frontier,
            frontier_tags=frontier_tags,
            xlim=xlim,
            ylim=ylim,
            lines=[boundary],
        ),
    )

    _finalize_figure(fig, out_path=out_path, legend_axes=[ax], add_legend=False, **_PARETO_MARGINS)
