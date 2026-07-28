"""Figure finalisation and save helpers.

Shared legend collection/placement and the ``_finalize_*`` routines that write
each figure to both ``png/`` and ``pdf/`` subdirectories. All rendering stays
consistent with ``plotting.style``.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from . import style as plotstyle
from .style import DEFAULT_DPI

LEGEND_Y = 0.02
LEGEND_MAX_COLUMNS = 4


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
