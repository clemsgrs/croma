"""The per-sample CRoMa ridgelines fit the page whatever the encoder count.

A ridgeline gains a row per encoder, so its PDF grows taller as the panel grows: on the
27-encoder panel, drawn at a fixed width, the Camelyon and both supplementary ridgelines were
taller than the text block. Every distribution float caps the graphic's height as well as
its width, keeping its aspect ratio, so whichever bound binds first sizes it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "scripts" / "repro"
if str(REPRO) not in sys.path:
    sys.path.insert(0, str(REPRO))

import generate_distribution_floats as gdf  # noqa: E402
from _distributions import CONTROL_MODEL, SUPP_BENCHMARKS  # noqa: E402

_GRAPHIC = re.compile(r"\\includegraphics\[(?P<options>[^]]*)\]")


def _tile_roster() -> SimpleNamespace:
    control = SimpleNamespace(name=CONTROL_MODEL, is_control=True)
    encoder = SimpleNamespace(name="Encoder", is_control=False)
    return SimpleNamespace(models=[encoder, control], by_name={CONTROL_MODEL: control})


def _slide_roster() -> SimpleNamespace:
    encoder = SimpleNamespace(name="Encoder", is_control=False)
    return SimpleNamespace(models=[encoder], by_name={"Encoder": encoder})


def test_every_ridgeline_is_bounded_in_height_and_keeps_its_aspect_ratio():
    floats = [
        gdf.build_figure(_tile_roster()),
        *(gdf.build_supp_figure(entry, _tile_roster()) for entry in SUPP_BENCHMARKS),
        gdf.build_panda_figure(_slide_roster()),
    ]
    for tex in floats:
        (options,) = [m["options"].split(",") for m in _GRAPHIC.finditer(tex)]
        assert r"width=0.78\linewidth" in options
        assert r"height=0.85\textheight" in options
        assert "keepaspectratio" in options
