"""Render the documentation site's figures as themed SVG (plus one raster montage).

The site publishes figures the same way it publishes numbers: as committed artifacts.
Nothing is fetched or compiled at docs-build time, because the docs build runs on a
clean checkout that cannot see ``paper/`` or ``output/`` (both git-ignored).

Two figure families, two treatments:

**Vector figures** (the standalone TikZ schematics and the matplotlib analysis plots)
convert to SVG and get a light and a dark variant. The dark variant is produced by
inverting the lightness of the *neutral* inks only -- the near-black strokes and text,
and the near-white panel fills. Saturated colours carry meaning (SO green, OS red, SS
purple, OO orange, anchor blue) and are preserved, with a mild lift so they stay legible
against a dark background. This is a transformation of the rendered figure rather than a
recompile, which is what lets it run without a LaTeX toolchain.

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
    if theme == "light":
        # The light variant is the figure as drawn, minus the opaque page background:
        # the near-white panel fills stay, but nothing paints the full canvas, so the
        # page shows through at the margins.
        return _strip_background(svg)
    return _strip_background(HEX.sub(lambda m: _to_dark(m.group(0)), svg))


def _strip_background(svg: str) -> str:
    """Drop a full-canvas white rectangle if the renderer emitted one."""
    return re.sub(
        r'<rect[^>]*?fill="#ffffff"[^>]*?/>\s*',
        "",
        svg,
        count=1,
    )


def _to_dark(hex_colour: str) -> str:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)

    if saturation <= NEUTRAL_SATURATION_MAX:
        # Neutral ink: invert lightness, then keep it off both extremes.
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
