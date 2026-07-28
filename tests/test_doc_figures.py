"""The rules that turn a rendered figure into a themed pair of documentation assets.

Every published figure is committed twice, light and dark, and the dark variant is derived
from the light one by rewriting colours rather than by re-rendering. That derivation is a
handful of thresholds, and a threshold nobody tests is a threshold that drifts: the wash
rule here exists because a warm pale shade counted as a semantic colour, kept its
lightness, and covered half of one figure in a pale block on a dark page.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "tools"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_doc_figures as bdf  # noqa: E402


def _lightness(hex_colour: str) -> float:
    import colorsys

    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[1]


def _hue(hex_colour: str) -> float:
    import colorsys

    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)[0]


# --------------------------------------------------------------------------------------
# Ink
# --------------------------------------------------------------------------------------


def test_near_black_ink_becomes_light():
    assert _lightness(bdf._to_dark("#1a1a1a")) > 0.8


def test_near_white_panel_fill_becomes_dark_but_not_black():
    """An unclamped inversion reads as a hole punched in the page, not a raised panel."""
    dark = bdf._to_dark("#fafafa")
    # The clamp is applied before the round-trip to 8-bit channels, so allow half a step.
    assert bdf.DARK_INK_MIN - 1 / 255 <= _lightness(dark) < 0.2


def test_a_semantic_colour_keeps_its_hue():
    """SO green stays green: the neighbour-type colours are the figure's vocabulary."""
    assert abs(_hue(bdf._to_dark("#2ca02c")) - _hue("#2ca02c")) < 0.01


def test_a_dark_semantic_colour_is_lifted_for_contrast():
    assert _lightness(bdf._to_dark("#1f77b4")) >= bdf.DARK_SERIES_MIN_LIGHTNESS - 1e-9


def test_a_pale_tint_inverts_like_a_background_despite_its_saturation():
    """The ridgeline's fragile-region shade. Pale enough to be a wash, tinted enough to
    have escaped the neutral rule -- which is exactly why it needed its own."""
    dark = bdf._to_dark("#f3e3d2")
    assert _lightness(dark) < 0.25
    assert abs(_hue(dark) - _hue("#f3e3d2")) < 0.01


# --------------------------------------------------------------------------------------
# Background
# --------------------------------------------------------------------------------------

_SVG_HEAD = '<svg width="100.0" height="50.0" viewBox="0 0 100.0 50.0">'


def test_a_full_canvas_rect_is_stripped():
    svg = f'{_SVG_HEAD}<rect width="100.0" height="50.0" fill="#ffffff"/><g/></svg>'
    assert "<rect" not in bdf._strip_background(svg)


def test_a_full_canvas_path_is_stripped():
    """matplotlib traces the page background as a path, not a rect. Missing this left
    every plotted figure in a white box."""
    svg = f'{_SVG_HEAD}<path d="M0 0H100.0V50.0H0Z" fill="#ffffff"/><g/></svg>'
    assert "<path" not in bdf._strip_background(svg)


def test_a_white_shape_that_is_not_the_canvas_survives():
    """The panel fill behind the axes is also white, and it has to stay: it inverts to a
    dark panel and reads as a raised surface in both themes."""
    svg = f'{_SVG_HEAD}<path d="M10 10H90V40H10Z" fill="#ffffff"/></svg>'
    assert '<path d="M10 10H90V40H10Z" fill="#ffffff"/>' in bdf._strip_background(svg)


def test_the_background_is_stripped_before_it_is_recoloured():
    """Order matters: the strip matches on white, and the dark pass has already turned
    the background grey by the time a later strip would look for it."""
    svg = f'{_SVG_HEAD}<path d="M0 0H100.0V50.0H0Z" fill="#ffffff"/><g/></svg>'
    assert "<path" not in bdf._recolour(svg, "dark")


# --------------------------------------------------------------------------------------
# Size
# --------------------------------------------------------------------------------------


def test_path_coordinates_are_rounded():
    svg = '<path d="M69.984123 484.242813L70.846309 12.5"/>'
    assert bdf._shrink_paths(svg) == '<path d="M69.98 484.24L70.85 12.5"/>'


def test_a_glyph_placement_transform_is_left_alone():
    """Glyphs are drawn in a 1000-unit em space and scaled by 0.001. Rounding that matrix
    to two decimals collapses every letter in the figure to a point."""
    svg = '<g transform="matrix(.001,0,0,.001,0,0)"><path d="M506.5 75.25"/></g>'
    assert "matrix(.001,0,0,.001,0,0)" in bdf._shrink_paths(svg)


def test_rounding_conserves_the_path_command_letters():
    svg = '<path d="M1.23456 2.34567C3.45678 4.5 5.5 6.5 7.5 8.5Z"/>'
    out = bdf._shrink_paths(svg)
    assert "".join(c for c in out if c.isalpha() and c not in "path d") == "MCZ"


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def test_every_registered_figure_is_committed_as_the_pair_it_promises():
    """A figure in the registry that was never built is a broken image on the site, and
    `sphinx -W` only catches it if some page happens to reference it."""
    dest = ROOT / "docs" / "_static" / "figures"
    for name, figure in bdf.FIGURES.items():
        if isinstance(figure, bdf.RasterFigure):
            assert (dest / f"{name}.jpg").exists(), name
            continue
        for theme in ("light", "dark"):
            assert (dest / f"{name}-{theme}.svg").exists(), f"{name}-{theme}"
