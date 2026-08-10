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
from .style import COL_ONEHALF


#: The Pareto panels enlarge the default scatter marker so each family colour reads clearly, with
#: the hollow frontier ring scaled to sit outside the bigger fill. Benchmark-exposed encoders are
#: NOT marked on the point -- an on-marker dot or cross competed with the fill colour and hurt
#: readability -- but by a dagger after their name in the legend (see ``_draw_model_scatter``'s
#: ``exposed`` argument); the caption spells the dagger out.
_PARETO_MARKER_S = 90.0
_PARETO_RING_S = 210.0


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
        r
        for r in rows
        if "croma" in r and "mari" in r and np.isfinite(float(r["croma"]))
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


def _pareto_pt_scale(
    xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[float, float]:
    """Data units per typographic point for the fixed Pareto figure cell (the margins both panels
    pass to ``_finalize_figure``), so tag room can be reserved before the axes exist. A bold 7 pt
    glyph is ~0.6 pt wide per character."""
    ax_w_pt = (0.975 - 0.135) * COL_ONEHALF * 72.0
    ax_h_pt = (0.95 - 0.30) * 6.3 * 72.0
    return (xlim[1] - xlim[0]) / ax_w_pt, (ylim[1] - ylim[0]) / ax_h_pt


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
    goes up-and-right of its ring into the empty non-dominated wedge (the Pareto property
    guarantees no other model point sits there), joined by a thin leader; a frontier point within
    8% of the widest ``x`` in the cloud is at the right edge, where the up-right wedge is off-panel,
    so it routes up-and-LEFT into the open interior instead (extending the axis rightward there
    would open a wide empty band). Right/top room is reserved per tag so it lands inside the spine.

    Placement is top-down: each tag starts one ``dy`` above its ring, is pushed to at least
    ``stack_gap`` below the tag above it (so y-clustered tags never overprint), and is then dropped
    further down off any *non-frontier* marker its label box would cover. That last step matters for
    an up-left interior tag, whose wedge -- unlike the up-right one -- is not Pareto-empty (on the
    rank panel GenBio-PathFM's tag would otherwise sit on the Virchow2 dot). The nudge is downward
    only, so a frontier label never hides a competitor's point. ``all_xy`` is every model point (the
    frontier included; it is filtered out here). Returns the expanded ``(xlim, ylim)`` and the
    per-tag draw specs for ``_draw_frontier_tags`` (called after the scatter, so tags sit on top).
    """
    xlo, xhi = xlim
    ylo, yhi = ylim
    x_per_pt, y_per_pt = _pareto_pt_scale(xlim, ylim)
    dx = 9.0 * x_per_pt   # tag anchor sits this far right (or left) of the ring centre
    dy = 8.0 * y_per_pt   # ...and this far above it
    char_w = 0.60 * plotstyle.FS_ANNOT * x_per_pt
    line_h = 1.35 * plotstyle.FS_ANNOT * y_per_pt
    stack_gap = 1.55 * plotstyle.FS_ANNOT * y_per_pt  # min vertical spacing between two tags
    marker_r = (float(_PARETO_MARKER_S) / np.pi) ** 0.5  # scatter marker radius, in points
    clear_x = marker_r * x_per_pt          # half-width of a marker, in data units
    clear_y = (marker_r + 0.35 * plotstyle.FS_ANNOT) * y_per_pt  # marker half-height + breathing room

    xs = [x for x, _ in all_xy]
    xspan = float(max(xs) - min(xs))
    xmax = float(max(xs))
    tag_left = {
        name: (xmax - fx) < 0.08 * xspan for name, (fx, _fy) in zip(frontier, frontier_xy)
    }
    for name, (fx, _fy) in zip(frontier, frontier_xy):
        room = len(name) * char_w + dx + 3.0 * x_per_pt
        if tag_left[name]:
            xlo = min(xlo, fx - room)
        else:
            xhi = max(xhi, fx + room)
    if frontier_xy:
        yhi = max(yhi, max(y for _, y in frontier_xy) + dy + line_h)

    others = [p for p in all_xy if p not in set(frontier_xy)]  # markers a tag must not cover

    order = sorted(range(len(frontier)), key=lambda i: frontier_xy[i][1], reverse=True)
    label_y: dict[int, float] = {}
    prev = None
    for i in order:
        name = frontier[i]
        fx = frontier_xy[i][0]
        ly = frontier_xy[i][1] + dy
        if prev is not None:
            ly = min(ly, prev - stack_gap)  # stay clear of the tag above
        # Drop below any non-frontier marker whose centre falls inside this tag's label box. Loop
        # because dropping past one marker can bring a lower one into range; each pass lowers ly
        # past at least one point, so it terminates.
        x0 = (fx - dx) - len(name) * char_w if tag_left[name] else fx + dx
        x1 = fx - dx if tag_left[name] else (fx + dx) + len(name) * char_w
        moved = True
        while moved:
            moved = False
            for px, py in others:
                in_x = x0 - clear_x <= px <= x1 + clear_x
                # A marker at or above the label whose bottom reaches into the label box: drop the
                # label so its top clears the marker bottom. ``py >= ly`` keeps this downward-only,
                # so ly strictly decreases each push and the loop terminates.
                if in_x and py >= ly and py - clear_y < ly + line_h / 2:
                    ly = py - clear_y - line_h / 2
                    moved = True
        label_y[i] = ly
        prev = ly

    tags = []
    for i, (name, (fx, fy)) in enumerate(zip(frontier, frontier_xy)):
        left = tag_left[name]
        anchor_x = fx - dx if left else fx + dx
        tags.append(
            {
                "name": name,
                "ring": (fx, fy),
                "anchor": (anchor_x, label_y[i]),
                "ha": "right" if left else "left",
                "text_dx": -2 if left else 2,
            }
        )
    return (xlo, xhi), (ylo, yhi), tags


def _draw_frontier_tags(ax, tags: list[dict]) -> None:
    """Draw the bold, white-haloed frontier labels from ``_layout_frontier_tags`` and their thin
    leaders. The halo keeps each tag legible where a leader or the frontier staircase passes
    behind it."""
    for t in tags:
        fx, fy = t["ring"]
        ax.annotate(
            "", xy=(fx, fy), xytext=t["anchor"],
            arrowprops=dict(arrowstyle="-", color=plotstyle.SPINE_COLOR,
                            linewidth=0.6, shrinkA=1.5, shrinkB=3.0),
            zorder=4,
        )
        ax.annotate(
            t["name"], xy=t["anchor"], xytext=(t["text_dx"], 0), textcoords="offset points",
            ha=t["ha"], va="center", fontsize=plotstyle.FS_ANNOT,
            color=plotstyle.TEXT_COLOR, weight="bold", zorder=6,
            path_effects=_halo(2.6),
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
    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 6.3))
    valid_rows = _valid_croma_ltm_rows(rows)
    if not valid_rows:
        ax.set_visible(False)
        _finalize_figure(fig, out_path=out_path, legend_axes=[ax])
        return

    exposed = set(exposed or ())
    coords = {r["model"]: (float(r["croma"]), float(r["ltm"])) for r in valid_rows}
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
    xlim, ylim, frontier_tags = _layout_frontier_tags(
        frontier, frontier_xy, list(coords.values()), xlim=(xlo, xhi), ylim=(ylo, yhi)
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
        exposed=exposed,
    )
    # The scatter primitive forces a square data box (its own identity); here that compresses
    # the axes into the bottom legend. Let the panel fill its figure cell instead (cf.
    # apd_figure), which keeps the x-label clear of the roster legend.
    ax.set_box_aspect(None)

    boundary = _pareto_boundary(frontier_xy, xlim=xlim, ylim=ylim)

    # Faint shade of the dominated region (below-left of the boundary): everything here is
    # beaten on both axes. Drawn beneath the points and the grid.
    shade = boundary + [(xlim[0], ylim[0])]
    ax.fill(
        [p[0] for p in shade], [p[1] for p in shade],
        color=plotstyle.SPINE_COLOR, alpha=0.05, linewidth=0, zorder=0.5,
    )
    # The frontier itself: a solid neutral staircase, heavier than the dashed zero references.
    ax.plot(
        [p[0] for p in boundary], [p[1] for p in boundary],
        color=plotstyle.SPINE_COLOR, linewidth=1.1, zorder=2.5, solid_capstyle="butt",
    )
    # Ring the non-dominated encoders so "these are the only undominated choices" reads at a
    # glance.
    ax.scatter(
        [x for x, _ in frontier_xy], [y for _, y in frontier_xy],
        s=_PARETO_RING_S, facecolors="none", edgecolors=plotstyle.SPINE_COLOR,
        linewidths=1.1, zorder=5,
    )
    # Label every ringed frontier encoder in bold -- the frontier answers "so which model is
    # best?", so it is named on every panel. Placement (up-right/up-left routing, y-de-collision,
    # white halo) is handled by _layout_frontier_tags/_draw_frontier_tags, shared with the
    # rank-aggregate panel.
    _draw_frontier_tags(ax, frontier_tags)

    # Encoders exposed to a cohort of this benchmark are flagged by a dagger after their name in
    # the legend (passed to _draw_model_scatter above as ``exposed``), never by a glyph on the
    # point: an on-marker dot or cross competed with the fill colour and hurt readability. The
    # caption spells the dagger out.

    # The full 25-model ranked roster makes a multi-row legend; the shared finaliser caps
    # its bottom margin below that, so the x-label lands on the top legend row. Reserve the
    # room explicitly instead.
    _finalize_figure(
        fig, out_path=out_path, legend_axes=[ax],
        top=0.95, bottom=0.30, left=0.135, right=0.975,
        legend_y=0.02, legend_ncol=4, legend_fontsize=8.4,
        legend_columnspacing=1.2, legend_handlelength=1.9,
    )


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

    fig, ax = plt.subplots(figsize=(COL_ONEHALF, 6.3))
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
    xlim, ylim, frontier_tags = _layout_frontier_tags(
        frontier, frontier_xy, list(coords.values()), xlim=xlim, ylim=ylim
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
        exposed=exposed,
    )
    ax.set_box_aspect(None)
    show_rank = FuncFormatter(lambda value, _pos: f"{-value:.0f}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(5))
        axis.set_major_formatter(show_rank)

    boundary = _pareto_boundary(frontier_xy, xlim=xlim, ylim=ylim)

    shade = boundary + [(xlim[0], ylim[0])]
    ax.fill(
        [p[0] for p in shade], [p[1] for p in shade],
        color=plotstyle.SPINE_COLOR, alpha=0.05, linewidth=0, zorder=0.5,
    )
    ax.plot(
        [p[0] for p in boundary], [p[1] for p in boundary],
        color=plotstyle.SPINE_COLOR, linewidth=1.1, zorder=2.5, solid_capstyle="butt",
    )
    ax.scatter(
        [x for x, _ in frontier_xy], [y for _, y in frontier_xy],
        s=_PARETO_RING_S, facecolors="none", edgecolors=plotstyle.SPINE_COLOR,
        linewidths=1.1, zorder=5,
    )
    # Bold frontier labels, drawn by the same helper the per-benchmark panels use.
    _draw_frontier_tags(ax, frontier_tags)

    # TCGA-exposed encoders (two of the three benchmarks contain a TCGA cohort, so an exposed
    # encoder's mean rank may reflect pretraining overlap) are flagged by a dagger after their name
    # in the legend (passed to _draw_model_scatter above as ``exposed``), never by a glyph on the
    # point: an on-marker dot or cross competed with the fill colour. The caption spells it out.

    _finalize_figure(
        fig, out_path=out_path, legend_axes=[ax],
        top=0.95, bottom=0.30, left=0.135, right=0.975,
        legend_y=0.02, legend_ncol=4, legend_fontsize=8.4,
        legend_columnspacing=1.2, legend_handlelength=1.9,
    )
