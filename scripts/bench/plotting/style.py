"""Single source of truth for figure style across croma.

This module defines the shared visual identity used by every plot in the
project -- the benchmark plot archetypes alongside it, and the standalone
figure and experiment scripts under ``scripts/``. It lives with the plotting
library rather than in the installed ``croma`` package because nothing in the
metrics library uses it, and its bundled typefaces would otherwise put ~2 MB of
fonts into every ``pip install croma`` (ADR-0017). The goal is a clean,
consistent, journal-quality ("Nature-style") look:

- a bundled sans-serif typeface (Arimo, Arial-metric, Apache-2.0) registered
  from ``plotting/fonts/`` so figures render identically on any machine;
- canonical column widths (single / one-and-a-half / double) so text appears
  at the same physical size in every figure;
- a cohesive, print-safe colour palette that keeps the model-family hue +
  within-family tone structure;
- fixed metric symbols/casing, number precision, and a canonical model order.

Import side effect: calling :func:`apply_style` (done automatically on import)
registers the fonts and applies the global ``rcParams``. Set the font in one
place here to swap the typeface project-wide.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
from matplotlib import font_manager

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"
_MODEL_METADATA = Path(__file__).resolve().parent.parent / "model_metadata.csv"

# ---------------------------------------------------------------------------
# Canonical figure widths (inches). Mirrors journal column widths.
# ---------------------------------------------------------------------------
COL_SINGLE = 3.46  # ~88 mm, single column
COL_ONEHALF = 4.72  # ~120 mm, one-and-a-half column
COL_DOUBLE = 7.20  # ~183 mm, double column / full text width

DEFAULT_DPI = 300

# ---------------------------------------------------------------------------
# Typography (points). One uniform set of sizes used everywhere.
# ---------------------------------------------------------------------------
FONT_FAMILY = "Arimo"
FS_BASE = 8.0  # default text
FS_TICK = 7.0  # tick labels
FS_LABEL = 8.0  # axis labels
FS_TITLE = 9.0  # panel / figure titles (bold)
FS_ANNOT = 7.0  # in-plot annotations, side tables
FS_PANEL_TAG = 9.0  # bold a/b/c panel tags

# ---------------------------------------------------------------------------
# Line / chrome weights (points).
# ---------------------------------------------------------------------------
LW_SPINE = 0.6
LW_GRID = 0.5
LW_SERIES = 1.4
LW_REFERENCE = 0.9
GRID_ALPHA = 0.15

# ---------------------------------------------------------------------------
# Neutral colours.
# ---------------------------------------------------------------------------
TEXT_COLOR = "#1a1a1a"
MUTED_TEXT_COLOR = "#5b6b7b"
GRID_COLOR = "#9aa3ad"
SPINE_COLOR = "#3a3a3a"
REFERENCE_LINE_COLOR = "#6b7280"
PANEL_FACE_COLOR = "#ffffff"
FRAGILE_SHADE_COLOR = "#f3e3d2"

# Guide-to-the-eye trend fits. Dotted and faint so they never compete with the dashed
# reference geometry (which marks exact, meaningful values) or with the data points.
TREND_LINE_COLOR = "#5b6b7b"
TREND_ALPHA = 0.55

# ---------------------------------------------------------------------------
# Model family and order come from the machine-readable provenance table; this module owns only
# the visual palette assigned to each family. That keeps checkpoint relationships and plot order
# auditable in the same source that generates the model tables (ADR-0005).
def _load_model_identity(path: Path = _MODEL_METADATA) -> tuple[dict[str, str], dict[str, int], list[str]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("plot_order", "")).strip():
                rows.append(row)
    rows.sort(key=lambda row: int(row["plot_order"]))

    families: dict[str, str] = {}
    tones: dict[str, int] = {}
    order: list[str] = []
    for row in rows:
        model = str(row["model"])
        family = str(row["family"])
        tone = int(row["family_order"])
        if model in families:
            if (families[model], tones[model]) != (family, tone):
                raise ValueError(f"inconsistent plot identity for duplicate metadata row {model!r}")
            continue
        families[model] = family
        tones[model] = tone
        order.append(model)
    return families, tones, order


MODEL_FAMILY_MAP, MODEL_TONE_INDEX, CANONICAL_MODEL_ORDER = _load_model_identity()


# Family hue + within-family tone palette.
# Families share a hue; tones (light -> dark) distinguish members.
# Palette is retuned for cohesion and best-effort colourblind separation
# (colour-only; no marker/dash backup).
# Per-family tones ordered light -> dark. Saturated for a lively, readable look.
FAMILY_PALETTE: dict[str, list[str]] = {
    "paige": ["#fb9a3c", "#e2660c"],  # orange
    "uni": ["#4cc183", "#1f9d55"],  # green
    "conch": ["#ee4d5a", "#c2222f"],  # red (cooler, away from orange)
    "phikon": ["#9b80e6", "#5b3fc4"],  # violet
    "hoptimus": ["#67b0d0", "#2f86b8", "#1f5f88"],  # steel blue (3 tones)
    "gigapath": ["#e0a01a"],  # gold / amber
    "midnight": ["#16b6c4"],  # cyan / teal
    "hibou": ["#e07ad0", "#b1318f"],  # magenta
    "prost": ["#8a8682"],  # neutral grey
    # Ported from slide2vec — provisional single-tone hues (validate via dataviz).
    "mstar": ["#6d5bd0"],  # indigo
    "gpfm": ["#d94a8c"],  # rose
    "musk": ["#2aa198"],  # teal
    "genbio": ["#9a7d0a"],  # olive / amber
    "rudolfv2": ["#8fc4df", "#4b98c2", "#246b98"],  # blue, teacher -> students
    "waiv": ["#ef8a62", "#b94b73"],  # coral -> berry
    "dinov2": ["#5f6b7a"],  # slate grey — natural-image control
    "other": ["#aab2bb", "#7e8790", "#565d65"],
}

# The natural-image control: pretrained on LVD-142M, never on a whole-slide image. It is a
# floor, not a competitor -- it is excluded from rankings, from per-column bolding, and from
# every cross-model correlation. Its positive CRoMa is an artifact of weak structure of
# either kind (it has the lowest biological k-NN accuracy of the panel), which is precisely
# what makes it useful: it calibrates what a positive margin means on a poor representation.
CONTROL_MODEL = "DINOv2-B"

# Regime hues for the pretraining-scale figure. This is the one figure whose colour encodes
# a model *attribute* rather than its family, because every point is directly labelled.
REGIME_PALETTE: dict[str, str] = {
    "vision-only": "#2f86b8",  # steel blue
    "VLFM": "#c2222f",  # red
}

# Fixed colour per metric (for single-metric accents and cross-metric panels).
METRIC_COLOR: dict[str, str] = {
    "ri": "#5b6b7b",
    "mari": "#2f6f8f",
    "croma": "#9a5b9a",
    "ltm": "#b3651a",
}

# Canonical metric symbols / casing (matches paper/main.tex).
METRIC_LABEL: dict[str, str] = {
    "ri": "RI",
    "mari": "MaRI",
    "croma": "CRoMa",
    "ltm": "LTM",
}

# Semantic neighbour-type colours (SS / SO / OS / OO) for the neighbourhood-composition
# and rank diagnostics. These are role-based encodings (not model-family hues), tuned to
# sit within the shared palette's hue space. In the rank diagnostic, OS doubles as the
# impostor / lower-tail accent and SO as the neutral baseline ("rest") series.
NEIGHBOR_TYPE_COLOR: dict[str, str] = {
    "SS": "#8a8682",  # same biology, same confounder   — neutral grey
    "SO": "#2f86b8",  # same biology, other confounder  — steel blue
    "OS": "#d12f2f",  # other biology, same confounder  — red (impostor)
    "OO": "#e0a01a",  # other biology, other confounder — gold / amber
}

# Two-way pretraining-regime colours (vision--language vs vision-only) for the
# scale/robustness scatter. Reuses existing family hues rather than new ink: the
# disclosed VLFMs *are* the CONCH family, so they take its red; vision-only
# encoders take the steel blue already used for the SO neighbour role. The result
# is a strong, print- and colourblind-safe red/blue split.
REGIME_COLOR: dict[str, str] = {
    "VLFM": "#c2222f",  # vision--language (CONCH-family red)
    "vision-only": "#2f86b8",  # vision-only (steel blue)
}

# Status colours (defined-support tiers): (fill, track).
SUPPORT_STATUS_COLORS: dict[str, tuple[str, str]] = {
    "good": ("#3f8f6e", "#dcebe4"),
    "warning": ("#c79a3a", "#f3e6c8"),
    "critical": ("#c45c55", "#f2d8d5"),
}

# Number formatting precision.
PREC_METRIC = 3  # RI / MaRI / CRoMa values
PREC_PERCENT = 1  # percentage annotations


def _register_fonts() -> None:
    """Register the bundled Arimo family with matplotlib's font manager."""
    if not _FONTS_DIR.is_dir():
        return
    for ttf in sorted(_FONTS_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(ttf))
        except Exception:  # noqa: BLE001 - font registration is best-effort
            continue


def apply_style() -> None:
    """Register fonts and apply the global matplotlib rcParams."""
    _register_fonts()
    matplotlib.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": PANEL_FACE_COLOR,
            "savefig.facecolor": "white",
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "standard",
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": FS_BASE,
            "axes.titlesize": FS_TITLE,
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_ANNOT,
            "axes.edgecolor": SPINE_COLOR,
            "axes.linewidth": LW_SPINE,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "xtick.major.width": LW_SPINE,
            "ytick.major.width": LW_SPINE,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.color": GRID_COLOR,
            "grid.linewidth": LW_GRID,
            "grid.alpha": GRID_ALPHA,
            "legend.frameon": False,
            "lines.linewidth": LW_SERIES,
            "lines.solid_capstyle": "round",
        }
    )


def style_axes(ax, *, grid_axis: str = "both") -> None:
    """Apply the shared chrome (light grid + thin full box) to an axes."""
    ax.set_facecolor(PANEL_FACE_COLOR)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right", "left", "bottom"):
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color(SPINE_COLOR)
        ax.spines[spine_name].set_linewidth(LW_SPINE)
    ax.tick_params(
        axis="both",
        colors=TEXT_COLOR,
        labelsize=FS_TICK,
        width=LW_SPINE,
        direction="out",
    )
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    grid_kw = dict(color=GRID_COLOR, linewidth=LW_GRID, alpha=GRID_ALPHA)
    if grid_axis == "y":
        ax.grid(axis="y", **grid_kw)
    elif grid_axis == "x":
        ax.grid(axis="x", **grid_kw)
    else:
        ax.grid(**grid_kw)


def set_panel_title(ax, title: str) -> None:
    """Apply the shared panel-title style."""
    ax.set_title(title, fontsize=FS_TITLE, weight="bold", pad=6)


def title_with_subtitle(ax, title: str, subtitle: str) -> None:
    """Bold panel title with a muted descriptive subtitle stacked beneath it.

    The title is padded up and the subtitle is offset a few points above the
    axes so the two never overlap regardless of figure size.
    """
    ax.set_title(title, fontsize=FS_TITLE, weight="bold", color=TEXT_COLOR, pad=16)
    ax.annotate(
        subtitle,
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FS_ANNOT,
        color=MUTED_TEXT_COLOR,
    )


def family_for_model(model: str) -> str:
    return str(MODEL_FAMILY_MAP.get(str(model), "other"))


def color_for_model(model: str) -> str:
    """Return the canonical colour for a model (family hue + tone)."""
    family = family_for_model(model)
    palette = FAMILY_PALETTE.get(family, FAMILY_PALETTE["other"])
    tone_idx = int(MODEL_TONE_INDEX.get(str(model), 0)) % len(palette)
    return str(palette[tone_idx])


def model_sort_key(model: str) -> int:
    """Canonical position of a model; unknown models sort to the end."""
    try:
        return CANONICAL_MODEL_ORDER.index(str(model))
    except ValueError:
        return len(CANONICAL_MODEL_ORDER)


def metric_label(metric: str) -> str:
    return METRIC_LABEL.get(str(metric).lower(), str(metric).upper())


# Apply on import so any module that imports plotstyle inherits the identity.
apply_style()
