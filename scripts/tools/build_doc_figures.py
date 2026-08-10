"""Render the documentation site's figures as themed SVG (plus one raster montage).

The site publishes figures the same way it publishes numbers: as committed artifacts.
Nothing is fetched or compiled at docs-build time, because the docs build runs on a
clean checkout that cannot see ``paper/`` or ``output/`` (both git-ignored).

Two figure families, two treatments (the results Pareto panels are no longer built here:
``docs/_static/pareto.js`` draws them in the browser from the committed ``results/`` CSVs,
so the panel and the tables beside it come from one export):

**Vector figures** (the standalone TikZ schematics) convert to SVG and get a light and a
dark variant. The dark variant is produced by inverting the lightness of the *neutral*
inks only -- the near-black strokes and text, and the near-white panel fills. Saturated
colours carry meaning (SO green, OS red, SS purple, OO orange, anchor blue) and are
preserved, with a mild lift so they stay legible against a dark background. This is a
transformation of the rendered figure rather than a recompile, which is what lets it run
without a LaTeX toolchain.

**Raster figures** (the dataset montage, which is photographic) get one downscaled
variant. Photographs do not theme, and a 4 MB PNG has no place on a docs page.

Run from the repository root::

    python scripts/tools/build_doc_figures.py            # everything
    python scripts/tools/build_doc_figures.py ri_mari    # one figure
"""

from __future__ import annotations

import argparse
import colorsys
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES = ROOT / "paper" / "figures"
DEST = ROOT / "docs" / "_static" / "figures"

# A colour is treated as neutral ink -- and therefore inverted for the dark variant --
# when it carries essentially no hue. Everything above this saturation is a semantic
# series colour and keeps its identity in both themes.
NEUTRAL_SATURATION_MAX = 0.12

# Inverted neutrals are clamped away from pure black and pure white: an unclamped
# inversion turns the near-white panel fill (#fafafa) into near-black, which reads as a
# hole punched in the page rather than as a raised panel.
DARK_INK_MIN = 0.13
DARK_INK_MAX = 0.93

# Saturated colours are lifted toward this lightness on dark backgrounds. Without it the
# darker series colours (the anchor blue in particular) lose contrast against the page.
DARK_SERIES_MIN_LIGHTNESS = 0.58

# A colour this pale is a background wash, not ink, whatever its hue -- no legible figure
# draws a line or a letter at this lightness on a white page. Washes invert like neutrals
# even when they are tinted, because their job is to sit *behind* things. Without this rule
# the ridgeline's warm fragile-region shade (#f3e3d2, saturation 0.58) counted as a
# semantic colour, kept its lightness, and covered the left half of the dark figure in a
# pale block.
LIGHT_WASH_MIN_LIGHTNESS = 0.80

HEX = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class VectorFigure:
    """A figure published as a light/dark SVG pair."""

    name: str
    source: Path

    def build(self) -> list[Path]:
        import pymupdf

        with pymupdf.open(self.source) as doc:
            svg = doc[0].get_svg_image()

        written = []
        for theme in ("light", "dark"):
            out = DEST / f"{self.name}-{theme}.svg"
            out.write_text(_recolour(svg, theme), encoding="utf-8")
            written.append(out)
        return written


@dataclass(frozen=True)
class RasterFigure:
    """A photographic figure published as a single downscaled raster."""

    name: str
    source: Path
    max_width: int = 1600
    quality: int = 82

    def build(self) -> list[Path]:
        from PIL import Image

        with Image.open(self.source) as img:
            img = img.convert("RGB")
            if img.width > self.max_width:
                height = round(img.height * self.max_width / img.width)
                img = img.resize((self.max_width, height), Image.LANCZOS)
            out = DEST / f"{self.name}.jpg"
            img.save(out, "JPEG", quality=self.quality, optimize=True, progressive=True)
        return [out]


FIGURES: dict[str, VectorFigure | RasterFigure] = {
    # Explanatory schematics (standalone TikZ; sources tracked under docs/figures/src/).
    "concept_metrics": VectorFigure("concept_metrics", PAPER_FIGURES / "concept_metrics.pdf"),
    "ri_mari": VectorFigure("ri_mari", PAPER_FIGURES / "ri_mari.pdf"),
    "croma_geometry": VectorFigure("croma_geometry", PAPER_FIGURES / "croma_geometry.pdf"),
    # Dataset orientation.
    "dataset_cardinality": VectorFigure(
        "dataset_cardinality", PAPER_FIGURES / "dataset_cardinality.pdf"
    ),
    "dataset_montage": RasterFigure(
        "dataset_montage", PAPER_FIGURES / "png" / "dataset_montage.png"
    ),
}


def _recolour(svg: str, theme: str) -> str:
    """Rewrite every hex colour in ``svg`` for the requested theme."""
    # Strip before recolouring, never after: the background is identified by being white,
    # and the dark pass has already turned it into a dark grey by the time a later strip
    # would look for it. Recolouring first left every plotted figure sitting in a #212121
    # box, a shade off furo's dark page and visible as a seam.
    body = _strip_background(svg)
    if theme == "dark":
        body = HEX.sub(lambda m: _to_dark(m.group(0)), body)
    return _fluid_root(_shrink_paths(body))


def _fluid_root(svg: str) -> str:
    """Drop the root ``width``/``height``, leaving the ``viewBox`` to carry the size.

    Last in the pipeline on purpose: :func:`_strip_canvas_path` reads the canvas size
    from those attributes. Without them the standalone view (the click-through target)
    scales to the browser window instead of pinning to the render size in points, and
    the inline ``<img>`` takes its size from the page's CSS either way.
    """
    end = svg.index(">") + 1
    root = re.sub(r'\s+(?:width|height)="[\d.]+"', "", svg[:end])
    return root + svg[end:]


#: Decimals kept in path coordinates. The converter emits six, which on a figure a few
#: hundred points wide resolves a ten-thousandth of a point -- three orders of magnitude
#: below a rendered pixel, and most of the file. On the densest plotted figures this
#: halved the page asset.
COORD_DECIMALS = 2

_PATH_D = re.compile(r'(\sd=")([^"]*)(")')
_NUMBER = re.compile(r"-?\d+\.\d+")


def _shrink_paths(svg: str) -> str:
    """Round path coordinates to :data:`COORD_DECIMALS`.

    Confined to ``d`` attributes on purpose. Glyphs are placed by a
    ``transform="matrix(.001,...)"``, and rounding *that* to two decimals would collapse
    every letter in the figure to a point.
    """

    def round_number(match: re.Match) -> str:
        value = round(float(match.group(0)), COORD_DECIMALS)
        text = f"{value:.{COORD_DECIMALS}f}".rstrip("0").rstrip(".")
        return text or "0"

    return _PATH_D.sub(
        lambda m: m.group(1) + _NUMBER.sub(round_number, m.group(2)) + m.group(3), svg
    )


def _strip_background(svg: str) -> str:
    """Drop the opaque page background, so the docs page shows through at the margins.

    Two renderers, two shapes. TikZ emits a ``<rect>``; matplotlib emits a ``<path>``
    tracing the full canvas, which is why the first form alone left every results panel
    sitting in a white box on a dark page.

    The panel fill *behind the axes* is a different white and is deliberately kept: it
    inverts to a dark panel and reads as a raised surface in both themes.
    """
    svg = re.sub(r'<rect[^>]*?fill="#ffffff"[^>]*?/>\s*', "", svg, count=1)
    return _strip_canvas_path(svg)


def _strip_canvas_path(svg: str) -> str:
    """Drop a white ``<path>`` whose outline is exactly the canvas rectangle."""
    size = re.search(r'width="([\d.]+)"\s+height="([\d.]+)"', svg)
    if size is None:
        return svg
    width, height = size.group(1), size.group(2)
    # matplotlib's PDF is y-flipped, so the canvas path is "M0 0 H<w> V<h> H0 Z" -- with
    # the numbers as the converter chose to print them, hence the tolerant separators.
    canvas = rf"M0\s*0\s*H{re.escape(width)}\s*V{re.escape(height)}\s*H0\s*Z"
    return re.sub(
        rf'<path[^>]*?d="{canvas}"[^>]*?fill="#ffffff"\s*/>\s*',
        "",
        svg,
        count=1,
    )


def _to_dark(hex_colour: str) -> str:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)

    if saturation <= NEUTRAL_SATURATION_MAX or lightness >= LIGHT_WASH_MIN_LIGHTNESS:
        # Neutral ink or a tinted background wash: invert lightness, keeping the hue and
        # staying off both extremes.
        lightness = min(DARK_INK_MAX, max(DARK_INK_MIN, 1.0 - lightness))
    else:
        # Semantic colour: same hue, lifted just enough to hold contrast on dark.
        lightness = max(lightness, DARK_SERIES_MIN_LIGHTNESS)

    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in (r, g, b)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "figures",
        nargs="*",
        metavar="FIGURE",
        help=f"figures to build (default: all). One of: {', '.join(sorted(FIGURES))}",
    )
    args = parser.parse_args(argv)

    unknown = sorted(set(args.figures) - set(FIGURES))
    if unknown:
        parser.error(f"unknown figure(s): {', '.join(unknown)}")

    DEST.mkdir(parents=True, exist_ok=True)
    selected = args.figures or sorted(FIGURES)

    missing = [name for name in selected if not FIGURES[name].source.exists()]
    if missing:
        # paper/ is git-ignored, so this script only runs where the rendered figures
        # already exist. Say so plainly rather than emitting a half-built figure set.
        for name in missing:
            print(f"missing source for {name}: {FIGURES[name].source}", file=sys.stderr)
        return 1

    for name in selected:
        for path in FIGURES[name].build():
            print(f"{path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
