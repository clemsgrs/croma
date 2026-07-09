"""Benchmark plotting library, decomposed into per-archetype submodules.

The public API is intentionally identical to the previous single-file
``plotting`` module: every ``plot_*`` entry point (and the shared helpers that
call sites and tests import) is re-exported here, so ``from plotting import
plot_ri_k_sweep`` and ``import plotting`` keep working unchanged.

Archetype submodules:

* :mod:`plotting.base` — visual-identity wrappers, geometry/limit helpers, and
  the shared ``_draw_model_scatter`` primitive.
* :mod:`plotting.export` — legend collection and the ``_finalize_*`` save helpers.
* :mod:`plotting.ksweep` — k-sweep curves (bio/confounder accuracy, RI, MaRI).
* :mod:`plotting.curves` — cumulative-mean curves and the CRoMa(m) trajectory.
* :mod:`plotting.bars` — support-coverage bars and CRoMa/LTM bars.
* :mod:`plotting.scatter` — one-point-per-model scatters.
* :mod:`plotting.distributions` — per-sample ridge-line distributions.

All figures continue to route through :mod:`croma.plotstyle`.
"""

from croma import plotstyle
from croma.confounders import infer_confounder_display_name
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

from .base import (
    _color_for_model,
    _confounder_display_name,
    _draw_model_scatter,
    _family_for_model,
    _group_k_rows,
    _highlight_selected,
    _human_friendly_integer_ticks,
    _ltm_label,
    _padded_signed_limits,
    _padded_unit_interval_limits,
    _set_k_axis,
    _set_panel_title,
    _style_axes,
    _valid_croma_ltm_rows,
)
from .export import (
    LEGEND_MAX_COLUMNS,
    LEGEND_Y,
    _add_figure_legend,
    _collect_legend_entries,
    _finalize_figure,
    _finalize_single_panel_legend_figure,
    _finalize_wide_line_figure,
    _legend_bottom_margin,
    _legend_columns,
    _pdf_export_path,
    _png_export_path,
)
from .ksweep import (
    _draw_k_curve,
    plot_knn_bio_k_sweep,
    plot_knn_confounder_k_sweep,
    plot_mari_k_sweep,
    plot_ri_k_sweep,
)
from .curves import (
    plot_croma_m_sweep,
)
from .bars import (
    _SUPPORT_STATUS_COLORS,
    _clamp_fraction,
    _support_plot_rows,
    _support_status,
    plot_croma_ltm_bars,
    plot_ri_mari_support,
)
from .scatter import (
    _draw_bio_vs_confounder_scatter,
    _draw_croma_vs_mari_scatter,
    _draw_mari_vs_ri_scatter,
    _draw_q_alpha_vs_croma_scatter,
    plot_bio_vs_confounder_scatter,
    plot_croma_ltm_scatter,
    plot_croma_vs_mari_scatter,
    plot_mari_vs_ri_scatter,
    plot_q_alpha_vs_croma_scatter,
)
from .distributions import (
    _draw_croma_sample_distributions,
    plot_croma_sample_distributions,
)

__all__ = [
    # k-sweeps
    "plot_knn_bio_k_sweep",
    "plot_knn_confounder_k_sweep",
    "plot_ri_k_sweep",
    "plot_mari_k_sweep",
    # CRoMa(m)
    "plot_croma_m_sweep",
    # bars
    "plot_ri_mari_support",
    "plot_croma_ltm_bars",
    # scatters
    "plot_bio_vs_confounder_scatter",
    "plot_mari_vs_ri_scatter",
    "plot_croma_ltm_scatter",
    "plot_croma_vs_mari_scatter",
    "plot_q_alpha_vs_croma_scatter",
    # distributions
    "plot_croma_sample_distributions",
]
